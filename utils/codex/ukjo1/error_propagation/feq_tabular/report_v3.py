"""Final, reproducible FEQ known-MDP report; artifacts are written under logs.

Run analyze_v2 --final first. This additionally audits the separately
registered adverse behavior-model tests and exports mechanism diagnostics.
"""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_v2 import sha, write_csv
from .engine_v2 import METHODS


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["case"], row["method"]), []).append(row)
    result = []
    for (case, method), group in sorted(groups.items()):
        row = dict(case=case, method=method, seeds=len(group), steps=group[0]["steps"])
        for metric, first in group[0].items():
            if metric in ("case", "method", "mode", "seed", "steps"):
                continue
            if isinstance(first, (int, float)):
                x = np.array([r[metric] for r in group], dtype=float)
                assert np.isfinite(x).all(), (case, method, metric)
                row[metric + "_mean"] = float(x.mean())
                row[metric + "_sd"] = float(x.std(ddof=1))
        result.append(row)
    return result


def audit_supplement(root):
    plan = json.loads((root / "flow_misspec_registration_v3.json").read_text())
    assert all(sha(p) == h for p, h in plan["source_sha256"].items())
    assert json.loads((root / "flow_misspec_completed_v3.json").read_text())["status"] == "complete"
    rows = []
    for case in plan["conditions"]:
        for seed in case["seeds"]:
            for method in case["methods"]:
                label = METHODS[method]
                path = root / "raw" / case["name"] / f"seed{seed}" / f"{label}.npz"
                obj = json.loads(path.with_suffix(".json").read_text())
                assert obj["identity"] == dict(case=case, seed=seed, method=method, label=label)
                assert obj["source_sha256"] == plan["source_sha256"]
                assert sha(path) == obj["data_sha256"]
                assert obj["metrics"]["identity_error"] < 1e-10
                rows.append(obj["metrics"])
    assert len(rows) == plan["trajectories"] == 192
    # Model misspecification is a target-only intervention. IQL must be
    # unaffected even though its own Q/V learning noise is still present.
    comparisons = 0
    for mode in ("clean", "iid", "field"):
        for seed in range(100, 108):
            with np.load(root / "raw" / f"misspec_{mode}_eps0.05" / f"seed{seed}" / "iql.npz") as ref:
                for epsilon in ("0.5", "0.95", "0.995"):
                    with np.load(root / "raw" / f"misspec_{mode}_eps{epsilon}" / f"seed{seed}" / "iql.npz") as other:
                        for key in ("q_history", "v_history", "target", "policy_return", "goal_success"):
                            np.testing.assert_array_equal(ref[key], other[key])
                    comparisons += 1
    return rows, comparisons


def mechanism_rows(root):
    components = []
    early = []
    starts = []
    state_actions = []
    for case in ("main_iid", "main_field", "main_clean_long"):
        for seed in range(100, 132):
            for method in ("iql", "feq_n9", "feq_population"):
                with np.load(root / "raw" / case / f"seed{seed}" / f"{method}.npz") as z:
                    tail = z["step"] >= 900000
                    for idx, name in enumerate(z["component_names"]):
                        mean = z["decomposition"][tail, idx, 0].mean()
                        rms = z["decomposition"][tail, idx, 1].mean()
                        components.append(dict(case=case, method=method, seed=seed,
                                               component=str(name), mean=float(mean), mean_window_rms=float(rms)))
                    # First100K: export every scheduled observation, not a
                    # retrospectively chosen most favorable stopping point.
                    for idx in np.flatnonzero((z["step"] > 0) & (z["step"] <= 100000)):
                        early.append(dict(case=case, method=method, seed=seed,
                                          step=int(z["step"][idx]), q_rmse=float(z["metrics"][idx, 0]),
                                          goal_success=float(z["goal_success"][idx]),
                                          policy_return=float(z["policy_return"][idx])))
                    starts.append(dict(case=case, method=method, seed=seed,
                                       true_start_value=float(z["v_tau_true"][0]),
                                       optimal_start_value=float(z["v_optimal"][0]),
                                       true_start_progress_q=float(z["q_tau_true"][0, 0]),
                                       final_progress_q=float(z["q"][0, 0]),
                                       final_safe_q=float(z["q"][0, 1]),
                                       final_wait_q=float(z["q"][0, 2:].max()),
                                       final_greedy_action=int(z["q"][0].argmax())))
                    for state in range(len(z["q"])):
                        q = z["q"][state]
                        truth = z["q_tau_true"][state]
                        state_actions.append(dict(case=case, method=method, seed=seed, state=state,
                            progress_q=float(q[0]), rival_q=float(q[1:].max()),
                            progress_gap=float(q[0]-q[1:].max()),
                            true_progress_gap=float(truth[0]-truth[1:].max()),
                            greedy_action=int(q.argmax())))
    aggregate = []
    for case in ("main_iid", "main_field", "main_clean_long"):
        for method in ("iql", "feq_n9", "feq_population"):
            for component in dict.fromkeys(r["component"] for r in components):
                selected = [r for r in components if (r["case"], r["method"], r["component"]) == (case, method, component)]
                row = dict(case=case, method=method, component=component, seeds=len(selected))
                for key in ("mean", "mean_window_rms"):
                    x = np.array([r[key] for r in selected])
                    row[key + "_mean"] = float(x.mean())
                    row[key + "_sd"] = float(x.std(ddof=1))
                aggregate.append(row)
    return components, aggregate, early, starts, state_actions


def figures(root, summary, early):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(iql="#d55e00", feq_n9="#0072b2", feq_population="#009e73")
    labels = dict(iql="IQL (Q + V)", feq_n9="FEQ-N9", feq_population="Population FEQ")
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for method in colors:
        selected = [r for r in early if r["case"] == "main_clean_long" and r["method"] == method]
        times = sorted({r["step"] for r in selected})
        for ax, metric in zip(axes, ("q_rmse", "goal_success")):
            x = np.array([[r[metric] for r in selected if r["step"] == t] for t in times])
            mean, sd = x.mean(1), x.std(1, ddof=1)
            ax.plot(np.array(times) / 1000, mean, color=colors[method], label=labels[method])
            low, high = mean - sd, mean + sd
            if metric == "goal_success":
                low, high = np.maximum(0, low), np.minimum(1, high)
            ax.fill_between(np.array(times) / 1000, low, high, color=colors[method], alpha=.15)
            ax.set_xlabel("Replay updates (thousands)")
            ax.grid(alpha=.15)
    axes[0].set_ylabel(r"Q RMSE to exact $Q_\tau$")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Exact goal probability")
    fig.suptitle("Noise-free 25-state chain: early learning, 32 independent seeds")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"clean_early_32seeds.{ext}", dpi=200)
    plt.close(fig)

    methods = ["iql", "iql_full_v1", "iql_full_v9", "iql_full_v81", "iql_exact_v_repair", "feq_n9"]
    names = ["IQL", "Full V\n1 step", "Full V\n9 steps", "Full V\n81 steps", "Exact V\nrepair", "FEQ-N9"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("iid", "field")):
        chosen = [next(r for r in summary if r["case"] == f"ablation_{mode}_V_fit" and r["method"] == m) for m in methods]
        x = np.arange(len(methods))
        means = [100 * r["tail_success_mean"] for r in chosen]
        sd = [100 * r["tail_success_sd"] for r in chosen]
        ax.bar(x, means, color=["#d55e00"] * 4 + ["#009e73", "#0072b2"])
        ax.errorbar(x, means, yerr=sd, fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(-2, 115)
        ax.set_ylabel("Exact goal probability (%)")
        ax.set_title("IID" if mode == "iid" else "Correlated readout field")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("V interventions:8 paired seeds,500K replay updates; mean +/- sample SD")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"v_intervention_8seeds.{ext}", dpi=200)
    plt.close(fig)


def report(root, summary, components, audit):
    def row(case, method):
        return next(r for r in summary if r["case"] == case and r["method"] == method)

    def table(cases, methods):
        lines = ["condition | method | seeds | updates | Q_RMSE mean +/- SD | goal % mean +/- SD | discounted return"]
        for case in cases:
            for method in methods:
                r = row(case, method)
                lines.append(f"{case} | {method} | {r['seeds']} | {r['steps']} | "
                             f"{r['tail_q_rmse_mean']:.6f} +/- {r['tail_q_rmse_sd']:.6f} | "
                             f"{100*r['tail_success_mean']:.3f} +/- {100*r['tail_success_sd']:.3f} | "
                             f"{r['tail_return_mean']:.6f}")
        return "\n".join(lines)

    component_text = []
    for r in components:
        if r["case"] in ("main_iid", "main_field") and r["component"] in ("V_fit", "sampling", "Q_readout"):
            if (r["method"] == "iql" and r["component"] == "V_fit") or (r["method"] == "feq_n9" and r["component"] != "V_fit"):
                component_text.append(f"{r['case']} | {r['method']} | {r['component']} | "
                                      f"mean={r['mean_mean']:+.8f} | mean window RMS={r['mean_window_rms_mean']:.8f}")
    analytic = json.loads((root / "analysis/analytic_witness.json").read_text())
    assert analytic["status"] == "PASS"
    text = f"""FEQ Known-MDP / Tabular 메커니즘 검증 — 최종 연구 보고서
2026-11-24 KST

결론
FEQ-N9이 IQL보다 정확한 Q와 높은 목표 도달률을 갖는 known-MDP 설정을 찾았고,
탐색 시드와 겹치지 않는32개 시드에서 확인했다. 이는 조건부 메커니즘 증거다.
명시적인 근사오차 대리모형에서 추가 V 회귀를 제거하는 이득을 보여준다.
모든 순수 tabular 문제나 신경망 FEQ 전체의 우월성, 실제 OGBench 실패 원인의
확정을 주장하지 않는다. 잡음 없는 장기 학습에서는 IQL이 FEQ-N9보다 정확하다.

실행 및 데이터 감사
독립 확인:928회; 별도 behavior 모델 오차 대조:192회; 합계{audit['confirmed_trajectories']}회.
탐색396회(162+144+90)는 분리 보관했다. 탐색 시드0~5, 확인 시드100~131.
주 검증은32시드, 나머지 기전/스트레스 대조군은8시드(100~107)다.
확인 조건과 결정 기준을 학습 전 registration JSON으로 고정했다.
등록 소스 해시와 모든 확인 NPZ 체크섬, 오차분해 항등식을 검사했다.
동일 시드 비교는 replay/행동/근사오차 난수열을 공유한다.
주 결과는900K,905K,...,1M의21개 관측점 평균을 각 시드에서 구한 뒤
시드 평균과 sample SD(ddof=1)로 요약한다. 독립 표본 수는32이며21x32가 아니다.
진단 성분은 위 관측 시점에 끝나는 직전5K 업데이트 창의 통계다.
Q오차와 성공률의 paired bootstrap95% CI가 두 잡음 모드 모두 FEQ 이득 쪽으로
0을 제외하는 사전 지정 기준을 통과했다. CI 원본은 paired_benefits_final_v2.csv.
기존 GPU 학습은 수정/종료하지 않고 새 실험은 CPU에서 수행했다.

1. Known MDP와 진짜 값
주 환경은13개 비종료 상태 s0,...,s12와 terminal 상태를 갖는 지연보상 chain이다.
연속 행동[-1,1]을4개 같은 길이 구간으로 나누고 각 구간 안에서 R,P,Q는 일정하다.
구간0은 다음 상태로 전진, s12에서는 reward1로 종료한다. 다른 행동은 reward0으로
대기한다. 단 s0의 구간1은 안전한 작은 보상을 받고 종료하는 safe exit다.
safe reward는 true population-expectile 경로 Q(s0,progress)의0.8배로 사전 정의했다.
모든 전이와 보상은 결정적이며 행동의 support가 빠지는 상태는 없다.
gamma=.999, tau=.9, true behavior는 모든 상태에서4구간 균등분포다.
최적 출발 값은 gamma^12={.999**12:.9f}. 행렬 R,P,continuation과 모든 참값은
각 raw NPZ에 저장했다. true Q*와 V*는 최적 Bellman 고정점이고,
Q_tau^mu,V_tau^mu는 true behavior의 population-expectile Bellman 고정점이다.
후자는 일반적으로 Q*,V*와 다르므로 주 RMSE 참값으로 Q_tau^mu를 사용한다.
Q* 대비 최종 RMSE도 별도 열 q_rmse_optimal에 전부 보존했다.
모든 참값의 수치 Bellman residual <1e-10을 확인했다.

데이터는 각 상태에서1024개의 offline 행동을 균등 behavior로 수집한 count다.
공통 replay는 상태를 균등 추출한 뒤 그 상태의 경험적 행동분포 mu_data를 사용한다.
이는 정상 상태의 한 긴 rollout 데이터셋이 아니라 reset 가능한 known-MDP 수집이다.
FEQ 행동모델은 mu_data의 MLE다. Gaussian -> Gaussian CDF -> 학습한 inverse CDF
로 가는 조건부 단조1D transport flow가 정확히 이 분포를 생성한다.
engine은 그 연속 flow의 구간 인덱스 분포를 직접 샘플링한다. Q가 구간 안에서
일정하므로 수학적으로 동일하다. 신경망 flow-matching 학습이라고 부르지 않는다.
유한 데이터로 인한 true behavior와 mu_data의 차이는 dataset 성분으로 분리한다.
주 조건에서 flow_fit=0인 것은 경험분포를 정확히 표현해서이며 true behavior를
유한 데이터에서 완벽히 알아냈다는 의미가 아니다.

2. 업데이트를 어떻게 맞췄는가
IQL: delta=Q_target(s,a)+eps_Q - (V(s)+eps_V),
     V(s) <- V(s)+2*alpha_V*|tau-1[delta<0]|*delta,
     y=r+gamma*continuation*(V(s')+eps_V(s')).
FEQ-N9: a'_1,...,a'_9 ~ fitted behavior(s'),
     e9 = 정확한9표본 expectile{{Q_target(s',a'_k)+eps_Q(s',a'_k)}},
     y=r+gamma*continuation*e9. 별도로 학습하는 V는 없다.
둘 다 Q(s,a) <- Q(s,a)+alpha_Q*(y-Q(s,a)), alpha_Q=alpha_V=.1이다.
target EMA=.005이며 기존 FEQ 연구 구현과 같이 그 step의 old-online Q를 쓴다.
초기 Q,V는0. FEQ의 후보 수9와 실행용 best-of-9를 구별한다.
population FEQ는9표본 대신 모델분포 전체에 대한 정확한 expectile을 사용한다.
여기서는 단일 Q를 비교해 V 경로를 분리했다. 신경망 twin-Q min/mean 차이나
MDN 단일 actor 학습 효과를 함께 비교하는 실험은 아니다.
동일 replay 업데이트 수를 사용하며 동일 계산량/Q조회 예산 비교는 아니다.
후보 중복은 같은 상태-행동의 같은 Q/readout 값을 재사용한다.
경험적 expectile의 root 오차는 수치 정밀도이지만 population expectile에 대한
유한표본 bias/variance까지 없어진다는 뜻은 아니다.

주 noise 조건: sigma_Q=sigma_V=.001의 독립적인 centered Gaussian readout 오차.
IID는 시간/상태/행동 독립(같은 step 같은 query는 공유)이고,
field는 고정 정규화 cosine feature8개, AR(1)rho=.99로 공간/시간 상관을 준다.
V 오차 field와 Q 오차 field는 독립이다. 이는 NN의 근사오차 대리모형이며
tabular 표현이 자체적으로 이런 오차를 만들어 냈다는 주장이 아니다.
stored table에 잡음을 누적해서 더하는 방식도 아니다.
주 조건의 TD target clipping은 off; 두 방법 모두 동일한 유효[0,1] bound로
clipping하는 별도 대조에서도 결과의 방향이 유지된다.

배포 비교는 두 방법 모두 동일한 정확한 경험 behavior에서9개를 뽑고 stored Q로
가장 높은 후보를 고르는 정책이다. wrong-flow stress에서도 이 배포 분포는
고정해 target 경로의 모델 오차만 분리했다. 실제 배포 flow까지 망가뜨린 경우의
정책 성능이라고 해석하면 안 된다. return은 (I-gamma P_pi)^(-1)r_pi,
성공률은1000step 안에 reward1 terminal에 도달할 확률을 전이행렬로 정확히 계산했다.
safe exit는 성공에 포함하지 않는다. 에피소드 평가 잡음이 없는 수치다.

3. 주 결과:32개 독립 학습 시드,1M replay updates
{table(['main_iid','main_field','main_clean_long'], ['iql','feq_n9','feq_population'])}

IID에서 FEQ-N9 Q RMSE는 IQL보다{row('main_iid','iql')['tail_q_rmse_mean']/row('main_iid','feq_n9')['tail_q_rmse_mean']:.2f}배 작고,
field에서는{row('main_field','iql')['tail_q_rmse_mean']/row('main_field','feq_n9')['tail_q_rmse_mean']:.2f}배 작다.
population FEQ도 IQL보다 좋으므로 FEQ-N9의 표본 bias만으로 전체 격차를 설명하지 못한다.
그러나 FEQ-N9이 population FEQ보다 좋은 부분에는 유한표본의 하향 bias가 Q-readout의
상향 bias를 상쇄하는 효과가 있다. N 증가가 무조건 좋다는 결론은 내리지 않는다.
IQL은 초기에는 Q오차가 감소하다가 후반에는 값이 부풀며 wait 행동을 택한다.
대기 행동은 같은 상태로 돌아오기 때문에 Q오차가 작아 보여도 목표 성공률을 크게
떨어뜨린다. 모든 값이 NaN/Inf로 발산해서 발생하는 실패는 아니다.
출발 s0에서는 IQL도 전진을 선택하지만 그 이후 상태에서 대기를 더 높게 평가한다.
예를 들어 IID seed100의 마지막 상태에서 IQL의 전진 Q=1.000000이고
최대 대기 Q=1.001115이다. 참 대기 Q는0.998667이다. FEQ-N9의 최대 대기 Q는
0.998998이어서 전진을 택한다. 이 예시는 정책 병목의 설명용이며 주 통계는32시드다.
state_action_diagnostics_v3.csv에는 모든 상태/시드의 Q gap과 선택을 보존했다.

4. V 경로를 직접 바꾼 대조:8시드,500K
{table(['ablation_iid_V_fit','ablation_field_V_fit'], ['iql','iql_full_v1','iql_full_v9','iql_full_v81','iql_exact_v_repair','feq_n9'])}
Full V는 방문 상태에서 모든4행동의 population V gradient를1/9/81번 적용한다.
같은 관측 Q/V field를 사용하며 추가 계산을 소모한다. 단순 seed 교체가 아니다.
Exact V repair는 backup을 그 시점 관측 targetQ의 data-distribution expectile로
대체한 인과 대조이며, 표준 IQL과 동일한 알고리즘이라고 부르지 않는다.
원래 IQL의 성능이 V 최적화 증가/정확한 V 대체로 회복되는 점은 추가 V 경로가
문제에 기여했다는 근거다. 이 repair는 모델=data인 population FEQ와 Q궤적이
bitwise 동일함을 수치 검증했다.

5. V에만 추가 noise를 줘서 만든 결과인가:8시드,200K
{table(['control_iid_zero','control_iid_Q_only','control_iid_V_only','control_iid_balanced','control_iid_large_noise'], ['iql','feq_n9','feq_population'])}
Q-only에서도 IQL과 FEQ는 동일한 Q 오차장을 보며 V-readout noise는0이다.
따라서 유리한 결과 전체를 추가 V-noise 하나만으로 설명할 수는 없다.
V-only는 perfect Q라고 부르지 않는다. Q 자체는 계속 업데이트되므로 V오차가
Q로 전달된다. perfect-Q 대조는 별도의 openloop 조건이다.
sigma=.002에서는 FEQ도 목표 도달에 실패한다. 더 큰 잡음에 대한 강건성을 주장하지 않는다.

두 방법을 모두 clipping한 대조(8시드,500K):
{table(['ablation_iid_clip','ablation_field_clip'], ['iql','feq_n9'])}

6. 잔차의 크기와 편향을 구별해야 한다
정확한 step별 target error = prior_Q + dataset + V_fit + flow_fit + sampling
                           + Q_readout + clipping.
IQL V_fit = gamma*c*(V_observed(s')-expectile_mu_data(Q_target(s',.))).
FEQ의 flow/sampling/Q_readout을 각각 차분으로 정의해 같은 합으로 분해했다.
각 성분의 평균/RMS뿐 아니라 교차2차모멘트도 저장했고 총 MSE 항등식을 검증했다.
RMS를 단순히 더하면 안 된다. component_target_stats_v3.csv에는32시드별 성분,
component_target_summary_v3.csv에는 시드 집계를 저장했다.

성분의 평균과 평균 window RMS(bootstrap gamma와 terminal mask 적용 후):
{chr(10).join(component_text)}

FEQ의 유한표본 RMS도 IQL의 V_fit RMS에 비해 무시할 만큼 작지는 않다.
따라서 'V잔차가 항상 더 크므로 FEQ가 낫다'는 설명은 성립하지 않는다.
이 환경에서는 추가 V 회귀/관측 오차가 만드는 양의 편향과 폐루프 전파가 핵심이다.
FEQ의 표본 편향은 Q-readout의 상향 편향을 일부 상쇄한다.
openloop에서는 stored Q를 true Q로 고정했는데도 IQL V-fit 잔차가 남았다.
이때 정책 개선은 주장할 수 없다(Q가 모든 방법에서 고정되어 있기 때문).

7. 피드백 해석과 한계
IQL: Q_target -> V 추정 -> Q -> Q_target.
FEQ: Q_target -> 표본 expectile -> Q -> Q_target.
FEQ도 Q-bootstrap 폐루프를 갖는다. 'Q오차가 Q에만 머물러 전파되지 않는다'고
쓰면 부정확하다. 제거되는 것은 별도 V 최적화/표현 오차의 상태와 경로다.
행동모델은 offline count만 보므로 Q에서 행동모델로 가는 gradient가 없다.
IQL에도 Q와 독립인 BC를 붙일 수 있어 이 독립성만으로 상대 우월성이 증명되지는 않는다.
잘못된 behavior 모델4조건을 바꾼72개의 동일 IQL 비교에서 Q/V/정책 궤적이
bitwise 동일함을 확인했다. 이는 해당 target-only 개입의 경로 분리를 검사한다.
동일한 additive backup-value error delta라면 어느 방법이든 gamma*P*delta로
Q target에 들어간다. V라는 이름이 붙어서 동등한 오차가 더 치명적인 것은 아니다.
수축 연산자에서의 일반적인 상계 gamma/(1-gamma)*||delta||_infinity도 양쪽에
해당한다. 이 실험은 서로 다른 잔차 발생 메커니즘을 비교한다.
noiseless local mean-update Jacobian의 spectral radius는 truth에서
IQL0.998919, population FEQ0.998742로 둘 다1보다 작다.
잘못 학습된 IQL Q 근처에서는 두 방식 모두1에 가까운 느린 모드를 갖는다.
이는 국소 평균 선형화이며 stochastic 안정성 증명이나 발산 부재 증명은 아니다.

8. analytic 보조 예제
정책 성능 예제가 아닌1상태1행동 reward0 self-loop의 expected-gradient surrogate.
IID readout 오차 아래 V loss의 최적점은
  V=Q_target+e_tau(N(0,sigma_Q^2+sigma_V^2)).
따라서 Q=gamma*V, Q_target=Q에서
  Q_IQL=gamma/(1-gamma)*e_tau(N(0,sigma_Q^2+sigma_V^2)), Q_FEQ=0.
tau=.9, sigma_Q=sigma_V=.001이면 V 잔차는{analytic['rows'][-1]['value_residual']:.9f}이고,
gamma=.999에서는 IQL 고정점이{analytic['rows'][-1]['IQL_exact_fixed_point']:.6f}이다.
5M expected-gradient 업데이트로 고정점을 확인했다. nonlinear stochastic 알고리즘의
평균 궤적이라고 해석하지 않는다. tau=.5 또는 noise0이면 이 편향 메커니즘은 없다.
이 부록은 추가 V회귀가 zero-mean 관측오차를 편향으로 바꾸는 경로를 설명하지만,
모든 NN 근사오차나 실제 task의 수치를 설명하는 이론은 아니다.

9. 불리한 조건도 포함한다
무잡음25상태 chain에서는 FEQ가 초기 Q추정에 더 빨리 도달하지만 충분히 학습하면
IQL Q RMSE0.000574 < FEQ-N9 0.002026이다. 성공률은 둘 다 약100%다.
초기100K의 모든5K 관측점은 clean_early_all_checkpoints_v3.csv와 그림으로 저장했다.
단일 좋은 checkpoint만 고른 결론이 아니다. population FEQ의 오차는 더 작다.
seed100의 정확한 finite-N9 Bellman 고정점을 multinomial count 전수 합으로
계산했다. Q_tau와 그 유한표본 고정점의 RMSE 약0.00203이며 학습한 FEQ와의
차이는 약0.000228이다. 유한 N의 고정점 bias가 장기 오차의 주요 부분이다.

adverse flow stress에서는 mu_model=(1-epsilon)*mu_data+epsilon*[0,1/3,1/3,1/3].
bootstrap 모델의 전진 행동 질량을 잃게 만든다. 모든 중간 epsilon도 보존했다.
clean epsilon=.995에서 FEQ Q RMSE 약0.792, 성공률0%인 반면 IQL 성공률은100%다.
즉 behavior 모델 품질 조건 없이 FEQ를 권할 수 없다. epsilon=.995는 매우 큰
모델 오류이며 현실적인 학습 오류의 크기라고 주장하지 않는다.
원래 flow_eps.01/.05/.2는 균등 true behavior 쪽 shrinkage 대조라서 adverse
misspecification으로 부르지 않고 이 별도 stress로 보완했다.
8시드,200K의 stress 전 결과는 supplement_mean_sd_v3.csv에 보존했다.

10. 논문에서 권하는 표현
권장: 'A controlled known-MDP experiment shows that removing the separately
estimated value function can reduce biased bootstrap errors, provided the
behavior approximation and finite-sample expectile errors remain controlled.'
권장: 'The V-fitting interventions support a contribution from the extra
value-regression path; FEQ still propagates critic errors through its backup.'
피할 표현: 'Tabular IQL intrinsically diverges', 'FEQ computes the true expectile
exactly with9 samples', 'FEQ removes all error propagation', 'Behavior fitting
independence alone proves superiority', 'These injected noises explain all OGBench failures'.
주 표에는32시드 main + V intervention + zero-noise reversal을 함께 제시하는 것이
타당하다. IID와 correlated noise가 같은 방향임을 보고하되 독립적인 두 실제
신경망 벤치마크로 셈하지 않는다. 이 결과만으로 FEQ의 novelty가 확정되지는 않는다.
원 IQL의 value regression 설명: https://arxiv.org/abs/2110.06169

11. 파일과 재현
코드: utils/codex/ukjo1/error_propagation/feq_tabular/
로그/원본/그림/이 보고서: {root}
사용자가 알려 준 ICLR tabular 코드의 잡음 대조 취지를 참고했으며 그 코드는 수정하지 않았다.
confirmation_registration_v2.json, flow_misspec_registration_v3.json: 고정 조건/해시
raw/: 시드별 모든 Q,V,참값,행동모델,정책/잔차 궤적
analysis/mean_sd_final_v2.csv: 본 실험 모든 조건
analysis/paired_benefits_final_v2.csv: paired95% CI
analysis/supplement_mean_sd_v3.csv: adverse behavior 모델 대조
analysis/audit_complete_v3.json: 전체 감사 결과
figures/learning_curves_32seeds.pdf: 무잡음 장기 반전까지 포함한 전체 곡선
figures/clean_early_32seeds.pdf: 무잡음 초기 학습
figures/v_intervention_8seeds.pdf: V 개입 대조

repository root에서 아래를 실행한다. 이미 완료된 raw는 identity/체크섬이 같으면 재사용한다.
재현을 다른 경로에서 할 때 원 등록 JSON에는 절대경로 해시가 있으므로 새 output root를
사용한다. 결과 파일만 임의 복사해 registration을 덮어쓰지 않는다.

PY=/home/ukjo1/anaconda3/bin/python
OUT=logs/codex/ukjo1/error_propagation/diagnosis/feq_known_mdp_20261124_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.test_core --output "$OUT/validation/core_v2.json"
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.confirm_v2 --root "$OUT" --workers 4
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.flow_misspec_v3 --root "$OUT" --workers 2
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.analyze_v2 --root "$OUT" --final
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.linear_feedback --root "$OUT"
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.analytic_witness --root "$OUT"
$PY -m utils.codex.ukjo1.error_propagation.feq_tabular.report_v3 --root "$OUT"

의존성: Python3.10, NumPy1.23.5, Numba, SciPy, Matplotlib. 기존 GPU/JAX 환경은 변경하지 않았다.
"""
    # The report is Korean prose in a plain-text artifact; code stays in utils.
    (root / "analysis/FINAL_REPORT_KO_v3.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit_main = json.loads((root / "analysis/audit_final_v2.json").read_text())
    assert audit_main["status"] == "PASS" and audit_main["full_confirmation_complete"]
    supplement, comparisons = audit_supplement(root)
    main_rows = []
    plan = json.loads((root / "confirmation_registration_v2.json").read_text())
    for case in plan["conditions"]:
        for seed in case["seeds"]:
            for meta in (root / "raw" / case["name"] / f"seed{seed}").glob("*.json"):
                main_rows.append(json.loads(meta.read_text())["metrics"])
    assert len(main_rows) == 928
    all_summary = summarize(main_rows + supplement)
    out = root / "analysis"
    write_csv(out / "supplement_per_seed_v3.csv", supplement)
    write_csv(out / "supplement_mean_sd_v3.csv", summarize(supplement))
    write_csv(out / "all_conditions_mean_sd_v3.csv", all_summary)
    components, aggregate, early, starts, state_actions = mechanism_rows(root)
    write_csv(out / "component_target_stats_v3.csv", components)
    write_csv(out / "component_target_summary_v3.csv", aggregate)
    write_csv(out / "clean_early_all_checkpoints_v3.csv", early)
    write_csv(out / "start_action_diagnostics_v3.csv", starts)
    write_csv(out / "state_action_diagnostics_v3.csv", state_actions)
    audit = dict(status="PASS", confirmed_trajectories=len(main_rows) + len(supplement),
                 main_trajectories=len(main_rows), adverse_flow_trajectories=len(supplement),
                 IQL_model_intervention_bitwise_identical_pairs=comparisons,
                 main_audit_sha256=sha(out / "audit_final_v2.json"),
                 report_source_sha256=sha(__file__))
    figures(root, all_summary, early)
    report(root, all_summary, aggregate, audit)
    (out / "audit_complete_v3.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
