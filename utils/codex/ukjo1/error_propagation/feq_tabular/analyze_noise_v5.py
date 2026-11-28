"""Audit FEQ/IQL robustness sweeps and plot success against error strength."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze_v2 import boot, write_csv, sha

METRICS = ("tail_success", "tail_model_success", "tail_return", "tail_q_rmse",
    "tail_q_rmse_optimal", "flow_TV_data", "flow_TV_true", "flow_reference_target_rms",
    "V_reference_target_rms", "model_progress_probability", "oracle_model_goal_coverage",
    "target_V_fit_rms", "target_V_fit_mean", "target_flow_fit_rms", "target_flow_fit_mean",
    "target_sampling_rms", "target_Q_readout_rms", "target_total_rms", "sigma_v")


def point_key(p):
    return p["family"], p["panel_mode"], p["label"], p.get("common_q_sigma", -1.), p["x"]


def load_rows(root, plan, final):
    rows, lookup = [], {}
    for case_id, case in plan["cases"].items():
        for seed in plan["seeds"]:
            meta = root / "raw" / case_id / f"seed{seed}.json"
            if not meta.exists():
                continue
            obj = json.loads(meta.read_text())
            assert obj["identity"] == dict(case_id=case_id, case=case, seed=seed)
            assert obj["source_sha256"] == plan["source_sha256"]
            if final:
                assert sha(meta.with_suffix(".npz")) == obj["data_sha256"]
            assert obj["metrics"]["identity_error"] < 1e-10
            row = obj["metrics"]
            rows.append(row)
            lookup[case_id, seed] = row
    return rows, lookup


def summarize(plan, lookup):
    summary, per_point = [], {}
    for p in plan["points"]:
        values = [lookup[p["case_id"], seed] for seed in plan["seeds"] if (p["case_id"], seed) in lookup]
        if not values:
            continue
        per_point[point_key(p)] = {v["seed"]: v for v in values}
        record = dict(p, seeds=len(values), expected_seeds=len(plan["seeds"]))
        for metric in METRICS:
            x = np.array([r[metric] for r in values])
            record[metric + "_mean"] = float(x.mean())
            record[metric + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else 0.
            if metric in ("tail_success", "tail_model_success"):
                _, low, high = boot(x)
                record[metric + "_ci_low"] = low
                record[metric + "_ci_high"] = high
        if p["family"] == "private":
            x = np.array([r["V_reference_target_rms"] if r["method"] == 0 else r["flow_reference_target_rms"] for r in values])
        else:
            x = np.full(len(values), p["x"])
        record["plot_x_mean"] = float(x.mean())
        record["plot_x_sd"] = float(x.std(ddof=1)) if len(x) > 1 else 0.
        summary.append(record)
    return summary, per_point


def compare_points(plan, per_point):
    pairs = []
    for p in plan["points"]:
        if p["family"] == "private" and p["label"] == "FEQ-N9":
            references = ["IQL V-only", "IQL V-only clipped"]
        elif p["family"] == "shared_Q" and p["label"].startswith("FEQ"):
            references = ["IQL Q-only", "IQL Q+V"]
        else:
            continue
        feq = per_point.get(point_key(p), {})
        for label in references:
            ref_key = list(point_key(p))
            ref_key[2] = label
            iql = per_point.get(tuple(ref_key), {})
            seeds = sorted(feq.keys() & iql.keys())
            if not seeds:
                continue
            record = dict(family=p["family"], panel_mode=p["panel_mode"], x=p["x"],
                          FEQ=p["label"], IQL=label, seeds=len(seeds))
            for metric, sign in (("tail_success", 1), ("tail_q_rmse", -1)):
                values = [sign*(feq[s][metric]-iql[s][metric]) for s in seeds]
                mean, low, high = boot(values)
                record[metric + "_FEQ_benefit"] = mean
                record[metric + "_ci_low"] = low
                record[metric + "_ci_high"] = high
            if p["family"] == "private":
                for seed in seeds:
                    np.testing.assert_allclose(feq[seed]["flow_reference_target_rms"],
                                               iql[seed]["V_reference_target_rms"], atol=1e-13)
            pairs.append(record)
    return pairs


def collapse_rows(summary):
    groups = defaultdict(list)
    for r in summary:
        groups[r["family"], r["panel_mode"], r["label"], r.get("common_q_sigma", -1.)].append(r)
    result = []
    for key, values in groups.items():
        values.sort(key=lambda r: r["x"])
        x = np.array([r["plot_x_mean"] for r in values])
        y = np.array([r["tail_success_mean"] for r in values])
        row = dict(family=key[0], mode=key[1], method=key[2], common_q_sigma=key[3],
                   min_seeds=min(r["seeds"] for r in values), points=len(values),
                   baseline_success=float(y[0]),
                   recoveries_over5pp=int(np.sum(np.diff(y) > .05)),
                   normalized_success_auc=float(np.trapz(y, x)/(x[-1]-x[0])) if len(x)>1 and x[-1]>x[0] else None)
        for threshold in (.8, .5):
            below = np.flatnonzero(y < threshold)
            tag = str(int(threshold*100))
            idx = int(below[0]) if len(below) else None
            row["first_below"+tag+"_grid_strength"] = values[idx]["x"] if idx is not None else None
            row["first_below"+tag+"_reference_x"] = float(x[idx]) if idx is not None else None
            row["previous"+tag+"_reference_x"] = float(x[idx-1]) if idx is not None and idx>0 else None
        result.append(row)
    return result


def plot(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    colors = {"IQL V-only":"#d55e00", "IQL V-only clipped":"#cc79a7", "FEQ-N9":"#0072b2", "Population FEQ":"#009e73",
        "IQL Q-only":"#d55e00", "IQL Q+V":"#a23a00", "FEQ data MLE":"#0072b2",
        "FEQ progress -50%":"#009e73", "FEQ progress -95%":"#777777",
        "FEQ random logits0.5":"#56b4e9", "FEQ random logits1.0":"#6e43a3"}

    def draw(ax, selected, metric, factor=1., scale_x=1.):
        selected = sorted(selected, key=lambda r:r["x"])
        label = selected[0]["label"]
        x = np.array([r["plot_x_mean"]*scale_x for r in selected])
        mean = np.array([r[metric+"_mean"]*factor for r in selected])
        color = colors[label]
        ax.plot(x, mean, marker="o", markersize=3, label=label, color=color,
                linestyle="--" if "clipped" in label or label=="IQL Q+V" else "-")
        if metric+"_ci_low" in selected[0]:
            low = np.array([r[metric+"_ci_low"]*factor for r in selected])
            high = np.array([r[metric+"_ci_high"]*factor for r in selected])
        else:
            sd = np.array([r[metric+"_sd"]*factor for r in selected])
            low, high = np.maximum(1e-10, mean-sd), mean+sd
        ax.fill_between(x, low, high, color=color, alpha=.12)
        ax.grid(alpha=.15)

    for family, filename in (("private", "matched_private_errors"), ("shared_Q", "shared_q_noise")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), sharex="col")
        for col, mode in enumerate(("iid", "field")):
            selected = [r for r in summary if r["family"]==family and r["panel_mode"]==mode]
            for label in dict.fromkeys(r["label"] for r in selected):
                s = [r for r in selected if r["label"]==label]
                draw(axes[0,col],s,"tail_success",factor=100,scale_x=1000)
                draw(axes[1,col],s,"tail_q_rmse",scale_x=1000)
            axes[0,col].set_title("IID value/Q readout errors" if mode=="iid" else "Correlated value/Q readout fields")
            axes[0,col].set_ylim(-2,102)
            axes[0,col].axhline(50,ls=":",color="black",lw=.8)
            axes[1,col].set_yscale("log")
            axes[1,col].set_xlabel(r"Matched reference target RMS $\delta$ ($10^{-3}$)" if family=="private" else r"Shared Q readout $\sigma_Q$ ($10^{-3}$)")
        axes[0,0].set_ylabel("Exact goal probability (%)")
        axes[1,0].set_ylabel(r"Q RMSE to true $Q_\tau$")
        axes[0,0].legend(fontsize=8,loc="best")
        title = "Private errors: IQL V-only vs fixed imperfect behavior flow; Q readout noise=0" if family=="private" else "Common Q noise, including misspecified FEQ behavior flows"
        fig.suptitle(title+"\n16 independent seeds; success bands=95% bootstrap CI; RMSE bands=sample SD",fontsize=11)
        fig.tight_layout()
        for extension in ("png","pdf"):
            fig.savefig(folder/f"{filename}.{extension}",dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11.5,7),sharex="col")
    for col, kind in enumerate(("deplete","logit")):
        for sigma,color in ((0.,"#0072b2"),(.001,"#d55e00")):
            selected=sorted([r for r in summary if r["family"]=="flow" and r["error_kind"]==kind and r["common_q_sigma"]==sigma],key=lambda r:r["x"])
            x=np.array([r["x"] for r in selected])
            for ax,metric in zip(axes[:,col],("tail_success","tail_model_success")):
                ax.plot(x,[100*r[metric+"_mean"] for r in selected],"o-",color=color,markersize=3,label=f"Q noise={sigma:g}")
                ax.fill_between(x,[100*r[metric+"_ci_low"] for r in selected],[100*r[metric+"_ci_high"] for r in selected],color=color,alpha=.15)
                ax.set_ylim(-2,102);ax.grid(alpha=.15)
            if sigma==0.:
                axes[1,col].plot(x,[100*r["oracle_model_goal_coverage_mean"] for r in selected],"k:",label="Oracle-Q proposal coverage")
        axes[0,col].set_title("Progress-mode underestimation" if kind=="deplete" else "Fixed random-logit model error")
        axes[1,col].set_xlabel("Fraction of progress probability removed" if kind=="deplete" else "Gaussian logit perturbation scale")
        axes[1,col].legend(fontsize=8)
    axes[0,0].set_ylabel("Goal probability with common proposals (%)")
    axes[1,0].set_ylabel("Goal probability with imperfect flow (%)")
    fig.suptitle("FEQ-N9 behavior-model errors: value learning and proposal coverage\n16 seeds;95% bootstrap CI",fontsize=11)
    fig.tight_layout()
    for extension in ("png","pdf"):
        fig.savefig(folder/f"behavior_model_errors.{extension}",dpi=200)
    plt.close(fig)


def report(root, status, summary, collapse, pairs):
    def line(r):
        return (f"{r['panel_mode']} | {r['label']} | level={r['x']:g} | "
                f"reference_x={r['plot_x_mean']:.7f} | n={r['seeds']} | "
                f"success={100*r['tail_success_mean']:.2f} +/- {100*r['tail_success_sd']:.2f}% | "
                f"model_proposals={100*r['tail_model_success_mean']:.2f}% | "
                f"Q RMSE={r['tail_q_rmse_mean']:.6f} | TV(data)={r['flow_TV_data_mean']:.4f}")
    lines=["FEQ / IQL 잡음 및 행동모델 근사오차 강도 실험", "2026-11-25 KST", "",
        f"실행 상태: {status['completed_trajectories']}/{status['expected_trajectories']}; complete={status['complete']}",
        "이전 V residual 제거 대조와 별도로 새 시드200~215의16개 paired seed를 사용했다.",
        "13상태 지연보상 chain,4행동 구간, gamma=.999,tau=.9,N=9,200K 업데이트.",
        "주 지표는180K~200K의5개 관측점 평균을 먼저 시드별 계산하고16시드 평균/SD로 요약한다.",
        "성공률은1000step 목표 도달확률의 정확한 전이행렬 계산이며 에피소드 추정잡음은 없다.",
        "시드별 target/Q 오차, component mean/RMS/교차모멘트를 저장했다.", "",
        "행동모델: 데이터 MLE 자체에도 true behavior에 대한 유한 데이터 오차가 있다.",
        "depletion은 전진확률의 epsilon비율을 제거하고 나머지 행동 비율을 유지하며 재분배한다.",
        "random-logit은 log(mu_data)에 시드/상태/행동별 고정 Gaussian 오차를 더한 softmax다.",
        "모든 model은 critic 학습 전에 고정되고 학습한 Q나 그 gradient를 사용하지 않는다.",
        "양의 확률로 support를 유지하며 같은 Gaussian->inverseCDF transport로 생성 가능하다.",
        "신경망 flow-matching 모델을 실제로 재학습한 실험이 아니라 그 분포 근사오차의 대리모형이다.", "",
        "1. 공통 Q 조회잡음 sweep", 
        "같은 sigma_Q/난수 Q field를 IQL과 FEQ에 준다. IQL Q-only는 sigma_V=0이다.",
        "IQL Q+V는 추가로 sigma_V=sigma_Q이므로 총 오차 예산이 같은 방법이라고 해석하지 않는다.",
        "FEQ는 추가 V가 없지만 행동모델/finite-N 오차가 존재한다."]
    lines += [line(r) for r in summary if r["family"]=="shared_Q"]
    lines += ["", "2. V-only와 Flow-only의 reference target RMS를 맞춘 대조",
        "Q 조회잡음=0. FEQ의 오차 있는 model을 Q와 독립적으로 먼저 고정한다.",
        "고정 참조 Qref는 경험적 behavior의 정확한 expectile Bellman 고정점이다.",
        "delta^2=gamma^2*E_replay[c^2*(e_tau(Qref,mu_model)-e_tau(Qref,mu_data))^2].",
        "IQL의 V noise sigma_V를 gamma*sigma_V*sqrt(E[c^2])=delta가 되도록 고른다.",
        "각 paired seed에서 source error RMS가 일치함을 확인했다. 그래프 x는 이 delta의 시드평균이다.",
        "이 보정은 참조점에서 추가 오차의 크기만 맞춘다. 학습 도중 총 target error를 맞추지 않는다.",
        "특히 V 오차는 centered IID/AR field이고, model bias는 고정된 하향 편향이다.",
        "부호/시간상관/상태별 위치를 일치시킨 실험이 아니므로 V라는 이유만으로 모든 동등 오차보다",
        "더 위험하다는 증명으로 쓰면 안 된다. finite-N 오차도 reference source budget에 포함하지 않는다.",
        "FEQ 곡선은 양쪽 private 패널에서 동일한16시드 결과를 공유한다(독립32시드가 아님).",
        "Population FEQ는 모델분포 전체를 사용하는 보조 대조이며9표본 추정기가 아니다.",
        "source budget은 clipping 전 기준이다. 학습전이 수를 맞췄고 Q 조회/계산량까지 같은 비교는 아니다.",
        "IQL target clipping[0,1] 대조도 함께 제시한다."]
    lines += [line(r) for r in summary if r["family"]=="private"]
    lines += ["", "3. Flow model 오차의 고유 강도 sweep",
        "depletion epsilon과 random-logit scale은 V의 value-unit sigma와 다른 단위다.",
        "TV(data),TV(true),참조 target RMS, 실제 학습 중 flow target error를 각각 CSV에 기록했다.",
        "주 비교는 모든 Q에 동일 mu_data best9 후보를 사용해 Q 품질을 분리한다.",
        "추가로 잘못된 model에서 best9 후보를 뽑은 정책도 평가한다. oracle-Q의 모델 후보 성능은",
        "제안 행동 coverage가 목표 도달을 제한하는 정도를 확인하는 대조다."]
    lines += [line(r) for r in summary if r["family"]=="flow"]
    lines += ["", "붕괴 기준: 시드평균 성공률이 처음50% 미만(80%도 별도)인 측정 grid point.",
        "관측 grid 사이의 정확한 임계값을 보간하지 않았다. recovery가 있으면 단조 붕괴로 부르지 않는다.",
        "baseline부터 낮은 model은 zero-noise 성능도 함께 보고해야 한다.",
        "noise_collapse.csv: 최초 crossing과 직전 grid,회복 횟수,AUC.",
        "paired_benefits.csv: 동일 seed,동일 공통 sigma 또는 matched delta에서의 FEQ-IQL 차이와 bootstrap95% CI.",
        "selection이나 실패 조건 삭제 없이 전체 조건을 보존한다.", "",
        "코드: utils/codex/ukjo1/error_propagation/feq_tabular/noise_sweep_v5.py",
        "분석: utils/codex/ukjo1/error_propagation/feq_tabular/analyze_noise_v5.py",
        f"로그/그림/등록: {root}",
        "재현: repository root에서 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1을 설정하고",
        "python -m utils.codex.ukjo1.error_propagation.feq_tabular.noise_sweep_v5 --root <output> --workers 4",
        "python -m utils.codex.ukjo1.error_propagation.feq_tabular.analyze_noise_v5 --root <output> --final",
        "기존 결과가 있으면 source/identity/data checksum이 같은 경우에만 재사용한다."]
    (root/"analysis"/"RESULTS_KO.txt").write_text("\n".join(lines)+"\n")


def state_profile_analysis(root, parent_summary, final):
    folder=root/"state_profile_v6"
    if not (folder/"registration.json").exists():
        return None
    plan=json.loads((folder/"registration.json").read_text())
    rows=[]
    for meta in sorted((folder/"raw").glob("seed*/*.json")):
        obj=json.loads(meta.read_text())
        assert obj["source_sha256"]==plan["source_sha256"]
        if final:
            assert sha(meta.with_suffix(".npz"))==obj["data_sha256"]
        rows.append(obj["metrics"])
    if final:
        assert len(rows)==640 and (folder/"completed.json").exists()
        assert all(sha(p)==h for p,h in plan["source_sha256"].items())
    grouped=defaultdict(list)
    for r in rows:
        grouped[r["mode"],r["clip"],r["epsilon"]].append(r)
    summary=[]
    for (mode,clip,epsilon),values in sorted(grouped.items()):
        rec=dict(mode=mode,clip=clip,epsilon=epsilon,seeds=len(values))
        for metric in ("tail_success","tail_return","tail_q_rmse","reference_target_rms",
                       "reference_error_state0_energy_fraction","target_V_fit_mean","target_V_fit_rms"):
            x=np.array([r[metric] for r in values]);rec[metric+"_mean"]=float(x.mean())
            rec[metric+"_sd"]=float(x.std(ddof=1)) if len(x)>1 else 0.
            if metric=="tail_success":
                _,low,high=boot(x);rec[metric+"_ci_low"]=low;rec[metric+"_ci_high"]=high
        summary.append(rec)
    out=root/"analysis"
    write_csv(out/"state_profile_per_seed.csv",rows)
    write_csv(out/"state_profile_mean_sd.csv",summary)
    comparison=[]
    for r in summary:
        parent=next(p for p in parent_summary if p["family"]=="private" and p["label"]=="FEQ-N9" and p["panel_mode"]==r["mode"] and p["x"]==r["epsilon"])
        uniform=next(p for p in parent_summary if p["family"]=="private" and p["label"]==("IQL V-only clipped" if r["clip"] else "IQL V-only") and p["panel_mode"]==r["mode"] and p["x"]==r["epsilon"])
        comparison.append(dict(mode=r["mode"],clip=r["clip"],epsilon=r["epsilon"],seeds=r["seeds"],
            reference_target_rms=r["reference_target_rms_mean"],
            IQL_uniform_success=uniform["tail_success_mean"],IQL_statewise_success=r["tail_success_mean"],
            FEQ_success=parent["tail_success_mean"],
            IQL_statewise_q_rmse=r["tail_q_rmse_mean"],FEQ_q_rmse=parent["tail_q_rmse_mean"],
            state0_source_energy=r["reference_error_state0_energy_fraction_mean"]))
    write_csv(out/"state_profile_comparison.csv",comparison)
    lines=["상태별 source MSE를 맞춘 추가 대조", "",
        f"완료 {len(rows)}/640. Q 조회잡음=0. 각 상태의 sigma_V(s)를 그 상태의 flow-induced reference expectile error 절댓값으로 설정했다.",
        "이전 global-RMS 대조와 달리 오차가 집중되는 상태까지 맞춘다.",
        "V 오차의 부호/시간 변화와 flow bias의 부호/고정성까지 맞춘 것은 아니다.",
        "따라서 전체 RMS만 맞춘 결과와 이 결과가 다르면 앞선 격차의 일부는 오차 위치에서 비롯된 것이다.",
        "mode | clip | epsilon | matched RMS | IQL uniform V | IQL statewise V | FEQ | state0 error-energy fraction"]
    for r in comparison:
        lines.append(f"{r['mode']} | {r['clip']} | {r['epsilon']:g} | {r['reference_target_rms']:.7f} | "
                     f"{100*r['IQL_uniform_success']:.2f}% | {100*r['IQL_statewise_success']:.2f}% | "
                     f"{100*r['FEQ_success']:.2f}% | {100*r['state0_source_energy']:.2f}%")
    (out/"STATE_PROFILE_KO.txt").write_text("\n".join(lines)+"\n")
    if final:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11.7,7.1),sharex='col')
        for col,mode in enumerate(('iid','field')):
            for label,color,clip in [('IQL statewise V','#6e43a3',False),('IQL statewise V clipped','#777777',True)]:
                selected=sorted([r for r in summary if r['mode']==mode and r['clip']==clip],key=lambda r:r['epsilon'])
                x=np.array([r['reference_target_rms_mean'] for r in selected])*1000
                for ax,metric,factor in ((axes[0,col],'tail_success',100),(axes[1,col],'tail_q_rmse',1)):
                    mean=np.array([r[metric+'_mean'] for r in selected])*factor
                    ax.plot(x,mean,'o-',color=color,label=label,markersize=3)
                    if metric=='tail_success':
                        ax.fill_between(x,[100*r[metric+'_ci_low'] for r in selected],[100*r[metric+'_ci_high'] for r in selected],color=color,alpha=.12)
            for label,color in [('IQL V-only','#d55e00'),('FEQ-N9','#0072b2')]:
                selected=sorted([r for r in parent_summary if r['family']=='private' and r['panel_mode']==mode and r['label']==label],key=lambda r:r['x'])
                x=np.array([r['plot_x_mean'] for r in selected])*1000
                display='IQL uniform V' if label=='IQL V-only' else label
                for ax,metric,factor in ((axes[0,col],'tail_success',100),(axes[1,col],'tail_q_rmse',1)):
                    ax.plot(x,[factor*r[metric+'_mean'] for r in selected],'o-',color=color,label=display,markersize=3)
            axes[0,col].set_title('IID V errors' if mode=='iid' else 'Correlated V errors')
            axes[0,col].set_ylim(-2,102);axes[1,col].set_yscale('log')
            axes[1,col].set_xlabel(r'Reference target RMS $\delta$ ($10^{-3}$)')
            axes[0,col].grid(alpha=.15);axes[1,col].grid(alpha=.15)
        axes[0,0].set_ylabel('Exact goal probability (%)');axes[1,0].set_ylabel(r'Q RMSE to true $Q_\tau$')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Controlling the location of private errors: statewise source MSE matching\n16 paired seeds; V noise and static flow bias still differ in sign/time structure',fontsize=11)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(root/'figures'/f'state_profile_control.{ext}',dpi=200)
        plt.close(fig)
    return dict(completed=len(rows),expected=640,complete=len(rows)==640)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--final",action="store_true")
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/"registration.json").read_text())
    rows,lookup=load_rows(root,plan,args.final)
    summary,per_point=summarize(plan,lookup)
    pairs=compare_points(plan,per_point)
    collapse=collapse_rows(summary)
    status=dict(completed_trajectories=len(rows),expected_trajectories=plan["trajectories"],
                complete=len(rows)==plan["trajectories"],independent_training_seeds=len(plan["seeds"]),
                matching="Source target RMS matched per seed at fixed reference Q; not equal total learned error distributions.")
    if args.final:
        assert status["complete"] and (root/"completed.json").exists()
        assert all(sha(p)==h for p,h in plan["source_sha256"].items())
        assert all(r["seeds"]==16 for r in summary)
    out=root/"analysis";out.mkdir(exist_ok=True)
    write_csv(out/"per_seed.csv",rows)
    write_csv(out/"mean_sd.csv",summary)
    write_csv(out/"paired_benefits.csv",pairs)
    write_csv(out/"noise_collapse.csv",collapse)
    (out/"audit.json").write_text(json.dumps(status,indent=2)+"\n")
    report(root,status,summary,collapse,pairs)
    if args.final:
        plot(root,summary)
    state_status=state_profile_analysis(root,summary,args.final)
    status['state_profile_supplement']=state_status
    (out/"audit.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status),flush=True)


if __name__=="__main__":
    main()
