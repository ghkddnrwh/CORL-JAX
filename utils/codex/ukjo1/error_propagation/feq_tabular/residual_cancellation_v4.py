"""Diagnostic oracle cancelling V residual relative to current learned Q.

External V noise is zero throughout. Q readout noise is still used by V
training. Q parameters are never replaced by truth. Unlike method6 in v2,
the reference expectile uses the stored, not noisy, targetQ and all actions.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .analyze_v2 import boot, write_csv
from .behavior_flow import fit, offline_counts
from .confirm_v2 import atomic, sha
from .engine_residual_v4 import run, COMPONENTS
from .mdp import chain, solve, evaluate_policy
from .report_v3 import summarize


def worker(payload):
    root_text, mode, seed, hashes = payload
    root = Path(root_text)
    mdp = chain(12, .25, .999, .9, .8)
    qt, vt, _ = solve(mdp)
    qs, vs, _ = solve(mdp, optimal=True)
    data = fit(offline_counts(mdp.behavior, seed, 1024))
    sigma_q = 0. if mode == "clean" else .001
    case = f"residual_cancel_{mode}"
    folder = root / "raw" / case / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, scale in [(0, 1.), (0, .5), (0, .25), (0, 0.), (1, 1.)]:
        label = f"iql_k{scale:g}" if method == 0 else "feq_n9"
        path = folder / f"{label}.npz"
        identity = dict(case=case, mode=mode, seed=seed, method=method, residual_scale=scale)
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            obj = json.loads(meta_path.read_text())
            assert obj["identity"] == identity and obj["source_sha256"] == hashes
            assert sha(path) == obj["data_sha256"]
            rows.append(obj["metrics"])
            continue
        out = run(method, mdp.reward, mdp.successor, mdp.continuation, data, data,
                  mdp.gamma, mdp.tau, 200000, .1, .1, .005, seed, qt, vt,
                  sigma_q=sigma_q, sigma_v=0., rho=.99 if mode == "field" else 0.,
                  spatial=mode == "field", log_every=5000,
                  reference_behavior=mdp.behavior, residual_scale=scale)
        steps, logs, q, target, value, qhistory, vhistory, decomp, cross, err = out
        assert err < 1e-10 and np.isfinite(q).all()
        np.testing.assert_allclose(cross[1:].sum((1, 2)), decomp[1:, -1, 1]**2, atol=1e-10)
        success, returns = [], []
        for snapshot in qhistory:
            ret, suc = evaluate_policy(mdp, snapshot, data)
            returns.append(ret)
            success.append(suc)
        tail = steps >= 180000
        row = dict(case=case, mode=mode, method=label, seed=seed, steps=200000,
                   residual_scale=scale, sigma_q=sigma_q, sigma_v=0.,
                   tail_q_rmse=float(logs[tail, 0].mean()),
                   tail_return=float(np.array(returns)[tail].mean()),
                   tail_success=float(np.array(success)[tail].mean()),
                   target_V_fit_mean=float(decomp[tail, 2, 0].mean()),
                   target_V_fit_rms=float(decomp[tail, 2, 1].mean()),
                   identity_error=float(err))
        # Verify unchanged baselines against the already-completed experiment.
        baseline = None
        if method == 1 or scale == 1.:
            old_mode = "field" if mode == "field" else "iid"
            old_case = f"control_{old_mode}_{'zero' if mode == 'clean' else 'Q_only'}"
            baseline = root / "raw" / old_case / f"seed{seed}" / ("feq_n9.npz" if method else "iql.npz")
            with np.load(baseline) as z:
                np.testing.assert_array_equal(qhistory, z["q_history"])
                np.testing.assert_array_equal(vhistory, z["v_history"])
        if method == 0 and scale == 0.:
            # Removing this residual must equal a noise-free population backup,
            # even though the separate V table still sees noisy Q in its loss.
            baseline = root / "raw/control_iid_zero" / f"seed{seed}" / "feq_population.npz"
            with np.load(baseline) as z:
                np.testing.assert_array_equal(qhistory, z["q_history"])
            np.testing.assert_array_equal(decomp[:, 2], np.zeros_like(decomp[:, 2]))
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, step=steps, metrics=logs, q=q, target=target, value=value,
                            q_history=qhistory, v_history=vhistory, decomposition=decomp,
                            component_cross=cross, component_names=np.array(COMPONENTS),
                            policy_return=np.array(returns), goal_success=np.array(success),
                            q_tau_true=qt, v_tau_true=vt, q_optimal=qs, v_optimal=vs,
                            data_behavior=data, reward=mdp.reward, successor=mdp.successor,
                            continuation=mdp.continuation)
        temp.replace(path)
        atomic(meta_path, dict(identity=identity, source_sha256=hashes,
                              data_sha256=sha(path), baseline_checked=str(baseline), metrics=row))
        rows.append(row)
    print(json.dumps(dict(case=case, seed=seed, completed=len(rows))), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    names = ("mdp.py", "behavior_flow.py", "engine_v2.py", "engine_residual_v4.py",
             "residual_cancellation_v4.py", "confirm_v2.py", "analyze_v2.py", "report_v3.py")
    hashes = {str((Path(__file__).parent / n).resolve()): sha(Path(__file__).parent / n) for n in names}
    plan = dict(date="2026-11-24", seeds=list(range(100, 108)), modes=["clean", "iid", "field"],
                scales=[1., .5, .25, 0.], steps=200000, source_sha256=hashes,
                sigma_v=0., sigma_q_noisy_modes=.001, trajectories=120,
                intervention="backup=e_tau(data,current_stored_targetQ)+k*(V-e_tau(data,current_stored_targetQ)); V loss unchanged",
                scope="Diagnostic oracle only; uses all actions of current learned targetQ, never trueQ. Not FEQ-N9 or an implementable new method.")
    reg = root / "residual_cancellation_registration_v4.json"
    if reg.exists():
        assert json.loads(reg.read_text()) == plan
    else:
        atomic(reg, plan)
        target = root / "residual_cancellation_source_v4"
        target.mkdir(exist_ok=True)
        for p in hashes:
            shutil.copyfile(p, target / Path(p).name)
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, (str(root), mode, seed, hashes)) for mode in plan["modes"] for seed in plan["seeds"]]
        for f in as_completed(futures):
            rows.extend(f.result())
            atomic(root / "residual_cancellation_progress_v4.json", dict(completed=len(rows), total=120))
    assert len(rows) == 120
    assert all(sha(p) == h for p, h in hashes.items())
    out = root / "analysis"
    write_csv(out / "residual_cancellation_per_seed_v4.csv", rows)
    summary = summarize(rows)
    write_csv(out / "residual_cancellation_mean_sd_v4.csv", summary)
    paired = []
    for mode in plan["modes"]:
        base = {r["seed"]: r for r in rows if r["mode"] == mode and r["method"] == "iql_k1"}
        corrected = {r["seed"]: r for r in rows if r["mode"] == mode and r["method"] == "iql_k0"}
        for metric, sign in [("tail_q_rmse", 1), ("tail_success", -1)]:
            values = [sign*(base[s][metric]-corrected[s][metric]) for s in base]
            mean, lo, hi = boot(values)
            paired.append(dict(mode=mode, metric=metric, benefit=mean, ci_low=lo, ci_high=hi, seeds=8))
    write_csv(out / "residual_cancellation_paired_v4.csv", paired)
    lines = ["V expectile 잔차의 oracle 제거 대조", "",
             "외부 V noise=0. V 업데이트/데이터/Q조회 noise/EMA/LR는 그대로 유지한다.",
             "E=e_tau(mu_data,current stored targetQ), rV=V-E, backup=E+k*rV.",
             "k=1은 기존 IQL, k=0은 residual만 제거한 진단 대조다. trueQ로 교체하지 않는다.",
             "이것은9개 sampled-action expectile이 아니고 모든4행동을 사용한 oracle다.",
             "이전 exact-V repair는 noisy targetQ의 root였지만 이번 oracle는 stored clean targetQ의 root다.",
             "진단 개입이며 실제 적용할 새 알고리즘을 제안하는 것이 아니다.",
             "8개 paired 학습 시드100~107,200K,180K~200K 시드별 평균 후 mean +/- sample SD.", "",
             "mode | method | Q RMSE | success % | applied V residual RMS | applied V residual mean"]
    for r in summary:
        lines.append(f"{r['case']} | {r['method']} | "
                     f"{r['tail_q_rmse_mean']:.6f} +/- {r['tail_q_rmse_sd']:.6f} | "
                     f"{100*r['tail_success_mean']:.3f} +/- {100*r['tail_success_sd']:.3f} | "
                     f"{r['target_V_fit_rms_mean']:.8f} | {r['target_V_fit_mean_mean']:+.8f}")
    lines += ["", "V 잡음0과 V residual0은 다르다. Q 관측잡음과 변하는 target은 V residual을 만들 수 있다.",
                  "이 개입은 추가 잔차 경로의 인과 효과를 검사하며, 동등한 Q오차보다 V오차가 항상 더 위험하다는 증명은 아니다.",
                  "k=0의 Q궤적은 무잡음 population FEQ와 동일하고, k=1/FEQ-N9은 이전 baseline과 bitwise 동일함을 확인했다."]
    (out / "RESIDUAL_CANCELLATION_KO_v4.txt").write_text("\n".join(lines)+"\n")
    atomic(root / "residual_cancellation_completed_v4.json", dict(status="complete", trajectories=120,
           seconds=time.monotonic()-start, source_sha256=hashes,
           checks=["k1 IQL and FEQ9 baselines bitwise identical", "k0 Q trajectory equals noise-free population backup",
                   "target-component MSE and per-step decomposition identities"]))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
