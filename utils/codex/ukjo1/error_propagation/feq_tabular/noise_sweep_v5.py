"""FEQ/IQL error-strength sweeps with imperfect behavior flows, CPU only.

Three registered views reuse unique training runs: private-error matching,
common-Q-noise sweeps, and native behavior-model-error sweeps. All scientific
artifacts and source snapshots are written to the supplied logs directory.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .behavior_flow import fit, offline_counts, transport, region
from .confirm_v2 import atomic, sha
from .engine_v2 import run, COMPONENTS, METRICS
from .mdp import chain, solve, evaluate_policy
from .noise_models_v5 import behavior_model, reference_error, matched_v_sigma, successor_weights

SEEDS = list(range(200, 216))
EPSILONS = [0., .1, .25, .5, .75, .9, .95, .98, .995, .999]
SIGMAS = [0., .00025, .0005, .00075, .001, .0015, .002, .003]


def build_plan():
    cases, points = {}, []

    def add(family, label, x, panel_mode, method=0, mode="iid", sq=0., sv=0.,
            match=None, kind="none", strength=0., clip=False, **extra):
        if strength == 0:
            kind, strength = "none", 0.
        if match == 0:
            match = None
        if sq == 0 and sv == 0 and match is None:
            mode = "iid"  # Identical no-noise kernels need only one training run.
        cfg = dict(method=method, mode=mode, sigma_q=sq, sigma_v=sv,
                   v_match_epsilon=match, flow_kind=kind, flow_strength=strength, clip=clip)
        key = "c" + hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:14]
        if key in cases:
            assert cases[key] == cfg
        cases[key] = cfg
        points.append(dict(family=family, label=label, x=x, panel_mode=panel_mode,
                           case_id=key, **extra))

    for mode in ("iid", "field"):
        for epsilon in EPSILONS:
            add("private", "IQL V-only", epsilon, mode, mode=mode, match=epsilon)
            add("private", "IQL V-only clipped", epsilon, mode, mode=mode, match=epsilon, clip=True)
            add("private", "FEQ-N9", epsilon, mode, method=1, kind="deplete", strength=epsilon)
            add("private", "Population FEQ", epsilon, mode, method=2, kind="deplete", strength=epsilon)
        for sigma in SIGMAS:
            add("shared_Q", "IQL Q-only", sigma, mode, mode=mode, sq=sigma)
            add("shared_Q", "IQL Q+V", sigma, mode, mode=mode, sq=sigma, sv=sigma)
            for label, kind, strength in [("FEQ data MLE", "none", 0.),
                    ("FEQ progress -50%", "deplete", .5),
                    ("FEQ progress -95%", "deplete", .95),
                    ("FEQ random logits0.5", "logit", .5),
                    ("FEQ random logits1.0", "logit", 1.)]:
                add("shared_Q", label, sigma, mode, method=1, mode=mode, sq=sigma,
                    kind=kind, strength=strength)
    for sigma in (0., .001):
        for kind, values in [("deplete", EPSILONS), ("logit", [0., .25, .5, 1., 2., 4.])]:
            for strength in values:
                add("flow", f"FEQ {kind}", strength, "iid", method=1, sq=sigma,
                    kind=kind, strength=strength, common_q_sigma=sigma, error_kind=kind)
    return cases, points


def validate():
    checks = []
    mdp = chain(12, .25, .999, .9, .8)
    for seed in (200, 215):
        data = fit(offline_counts(mdp.behavior, seed, 1024))
        q, _, _ = solve(mdp, data)
        previous = -1.
        for epsilon in EPSILONS:
            model = behavior_model(data, "deplete", epsilon, seed)
            tv = np.abs(model-data).sum(-1)/2
            np.testing.assert_allclose(tv, epsilon*data[:, 0], atol=1e-13)
            sigma, delta = matched_v_sigma(mdp, q, data, epsilon, seed)
            np.testing.assert_allclose(delta, mdp.gamma*sigma*np.sqrt(successor_weights(mdp, data).sum()), atol=1e-14)
            assert delta >= previous
            previous = delta
        rng = np.random.default_rng(2017)
        z = rng.normal(size=200000)
        for kind, strength in [("deplete", .9), ("logit", 2.)]:
            model = behavior_model(data, kind, strength, seed)
            for s in (0, 6, 12):
                bins = region(transport(model[s], z))
                np.testing.assert_allclose(np.bincount(bins, minlength=4)/len(z), model[s], atol=.006)
    checks += ["Positive normalized flow models; exact depletion TV",
               "Reference one-step target RMS equals calibrated V-noise RMS",
               "Continuous Gaussian transport realizes corrupted behavior probabilities",
               "Error construction and random-logit tapes do not use learned Q"]
    return checks


def worker(payload):
    root_text, seed, cases, hashes = payload
    root = Path(root_text)
    mdp = chain(12, .25, .999, .9, .8)
    qt, vt, _ = solve(mdp)
    qo, vo, _ = solve(mdp, optimal=True)
    counts = offline_counts(mdp.behavior, seed, 1024)
    data = fit(counts)
    qref, vref, _ = solve(mdp, data)
    rows = []
    for case_id, cfg in cases.items():
        folder = root / "raw" / case_id
        folder.mkdir(exist_ok=True, parents=True)
        path = folder / f"seed{seed}.npz"
        meta = path.with_suffix(".json")
        identity = dict(case_id=case_id, case=cfg, seed=seed)
        if meta.exists():
            saved = json.loads(meta.read_text())
            assert saved["identity"] == identity and saved["source_sha256"] == hashes
            assert sha(path) == saved["data_sha256"]
            rows.append(saved["metrics"])
            continue
        model = behavior_model(data, cfg["flow_kind"], cfg["flow_strength"], seed)
        flow_delta, flow_shift, weights = reference_error(mdp, qref, data, model)
        sv = cfg["sigma_v"]
        matched_delta = 0.
        if cfg["v_match_epsilon"] is not None:
            sv, matched_delta = matched_v_sigma(mdp, qref, data, cfg["v_match_epsilon"], seed)
        out = run(cfg["method"], mdp.reward, mdp.successor, mdp.continuation,
                  data, model, mdp.gamma, mdp.tau, 200000, .1, .1, .005, seed, qt, vt,
                  sigma_q=cfg["sigma_q"], sigma_v=sv, rho=.99 if cfg["mode"] == "field" else 0.,
                  spatial=cfg["mode"] == "field", log_every=5000, clip=cfg["clip"],
                  reference_behavior=mdp.behavior, num_candidates=9)
        steps, logs, q, target, v, qhist, vhist, decomp, cross, err = out
        assert err < 1e-10 and np.isfinite(q).all()
        np.testing.assert_allclose(cross[1:].sum((1, 2)), decomp[1:, -1, 1]**2, atol=1e-10)
        shared = np.array([evaluate_policy(mdp, snapshot, data) for snapshot in qhist])
        own = shared.copy() if np.array_equal(data, model) else np.array([evaluate_policy(mdp, snapshot, model) for snapshot in qhist])
        oracle_return, coverage = evaluate_policy(mdp, qo, model)
        tail = steps >= 180000
        row = dict(case_id=case_id, seed=seed, steps=200000, method=cfg["method"],
                   mode=cfg["mode"], sigma_q=cfg["sigma_q"], sigma_v=sv,
                   flow_kind=cfg["flow_kind"], flow_strength=cfg["flow_strength"], clip=cfg["clip"],
                   flow_reference_target_rms=flow_delta, matched_reference_target_rms=matched_delta,
                   V_reference_target_rms=float(mdp.gamma*sv*np.sqrt(weights.sum())),
                   flow_TV_data=float(np.abs(model-data).sum(-1).mean()/2),
                   flow_TV_true=float(np.abs(model-mdp.behavior).sum(-1).mean()/2),
                   model_progress_probability=float(model[:, 0].mean()),
                   oracle_model_goal_coverage=coverage, oracle_model_return=oracle_return,
                   tail_q_rmse=float(logs[tail, 0].mean()),
                   tail_q_rmse_optimal=float(np.sqrt(((qhist[tail]-qo)**2).mean((1, 2))).mean()),
                   tail_success=float(shared[tail, 1].mean()), tail_return=float(shared[tail, 0].mean()),
                   tail_model_success=float(own[tail, 1].mean()), tail_model_return=float(own[tail, 0].mean()),
                   final_success=float(shared[-1, 1]), identity_error=float(err))
        for i, name in enumerate(COMPONENTS):
            row[f"target_{name}_mean"] = float(decomp[tail, i, 0].mean())
            row[f"target_{name}_rms"] = float(decomp[tail, i, 1].mean())
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, step=steps, metrics=logs, metric_names=np.array(METRICS),
            q=q, target=target, value=v, q_history=qhist, v_history=vhist,
            decomposition=decomp, component_cross=cross, component_names=np.array(COMPONENTS),
            shared_policy=shared, model_policy=own, q_tau_true=qt, v_tau_true=vt,
            q_optimal=qo, v_optimal=vo, q_ref_data=qref, v_ref_data=vref,
            offline_counts=counts, data_behavior=data, model_behavior=model,
            reference_flow_shift=flow_shift, reference_successor_weights=weights,
            reward=mdp.reward, successor=mdp.successor, continuation=mdp.continuation)
        temporary.replace(path)
        atomic(meta, dict(identity=identity, source_sha256=hashes, data_sha256=sha(path), metrics=row))
        rows.append(row)
        atomic(root / "progress" / f"seed{seed}.json", dict(completed=len(rows), total=len(cases)))
        print(json.dumps(dict(seed=seed, completed=len(rows), total=len(cases), case=case_id)), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cases, points = build_plan()
    if args.dry_run:
        print(json.dumps(dict(unique_cases=len(cases), panel_points=len(points), seeds=SEEDS,
                              trajectories=len(cases)*len(SEEDS), updates=len(cases)*len(SEEDS)*200000)), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "progress").mkdir(exist_ok=True)
    lease = (root / "training.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__).parent / n for n in ("noise_models_v5.py", "noise_sweep_v5.py",
               "engine_v2.py", "mdp.py", "behavior_flow.py", "confirm_v2.py")]
    hashes = {str(p.resolve()): sha(p) for p in sources}
    checks = validate()
    atomic(root / "validation.json", dict(status="PASS", checks=checks, source_sha256=hashes))
    plan = dict(date_KST="2026-11-25", seeds=SEEDS, steps=200000, cases=cases, points=points,
                trajectories=len(cases)*len(SEEDS), updates=len(cases)*len(SEEDS)*200000,
                source_sha256=hashes, state_count=13, actions=4, gamma=.999, tau=.9,
                dataset_per_state=1024, target_ema=.005, alpha_q=.1, alpha_v=.1,
                common_Q_sigmas=SIGMAS, behavior_depletion_strengths=EPSILONS,
                deployment_candidates=9, bootstrap_candidates=9,
                primary_metric="seed-level average of exact goal probabilities at180K,185K,190K,195K,200K",
                collapse="first tested strength with mean success<50%; report80% too and flag recoveries",
                error_matching="Flow is preselected independently of Q. IQL V sigma is chosen so its additive source error and flow-induced error have identical replay-weighted target RMS at the fixed data-expectile reference Q.",
                limits=["Equal reference RMS does not match sign, temporal persistence or per-state error profile.",
                        "Matched flow bias is fixed per seed; V fields are IID or rho.99. This is not identical noise distributions.",
                        "Actual training target errors, dataset error and finite-N sampling error are retained separately.",
                        "Q-table error can emerge even when Q readout sigma=0.",
                        "Primary policies use common data proposals; own-model proposals are also evaluated with oracle coverage.",
                        "Single-Q tabular value-update mechanism; no neural flow fitting or MDN actor validation.",
                        "No outcome-dependent model/seed selection; publish failures and clipping controls."])
    path = root / "registration.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic(path, plan)
        (root / "source_snapshot").mkdir(exist_ok=True)
        for p in sources:
            shutil.copyfile(p, root / "source_snapshot" / p.name)
    begun = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, (str(root), seed, cases, hashes)) for seed in SEEDS]
        for f in as_completed(futures):
            done += len(f.result())
            atomic(root / "progress.json", dict(completed=done, expected=plan["trajectories"], seconds=time.monotonic()-begun))
    assert done == plan["trajectories"]
    assert all(sha(p) == h for p, h in hashes.items())
    atomic(root / "completed.json", dict(status="complete", trajectories=done, seconds=time.monotonic()-begun))
    print("NOISE_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
