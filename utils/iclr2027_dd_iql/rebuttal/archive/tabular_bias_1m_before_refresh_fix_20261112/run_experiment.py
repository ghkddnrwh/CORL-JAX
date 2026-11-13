"""Run or resume the six-method 1M sweep without depending on historical logs."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

for variable in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

from common import case_path, cases, CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR, METHODS

CONTEXT = {}


def generate_seed(noise, slot, size, mode):
    import numpy as np
    from compare_tabular_methods import NNLikeErrorField
    from scipy.signal import lfilter

    seed = CONFIG["seeds"][slot]
    width = max(4, size)
    if mode.endswith("_iid"):
        rng = np.random.default_rng(100003 + seed)
        for start in range(0, CONFIG["updates"], 20000):
            end = min(start + 20000, CONFIG["updates"])
            noise[slot, start:end] = rng.standard_normal((end - start, 2, size, width))[
                ..., :size
            ]
        return seed
    field = NNLikeErrorField(
        n=2,
        height=1,
        width=size + 1,
        sigma=1.0,
        rho=0.99,
        feature_dim=64,
        shared_fraction=0.0,
        lengthscale=0.35,
        action_scale=1.0,
        seed=100003 + seed,
        spatial_correlation=True,
        **({"num_actions": width} if width != 4 else {})
    )
    initial = field.full()[:, 0, :size]
    carry = np.zeros((1, 2, size, width))
    for start in range(0, CONFIG["updates"], 20000):
        end = min(start + 20000, CONFIG["updates"])
        length = end - start
        innovations = np.empty((length, 2, size, width))
        offset = 1 if start == 0 else 0
        if offset:
            innovations[0] = initial
        latent = field.rng.standard_normal((length - offset, 3, 64))
        projected = np.einsum(
            "tnd,sad->tnsa", latent[:, 1:], field.phi[0, :size], optimize=True
        )
        innovations[offset:] = np.sqrt(1 - 0.99**2) * projected
        values, carry = lfilter([1.0], [1.0, -0.99], innovations, axis=0, zi=carry)
        noise[slot, start:end] = values[..., :size]
    return seed


def run_one(case):
    import numpy as np
    from engine import run_compiled

    start = time.monotonic()
    size, mode, method, delay, tau = case
    h = run_compiled(
        method, CONTEXT["noise"], CONTEXT["actions"], delay or 1, tau or 0.9
    )
    q = h["q_final"].mean(1)
    b = q.max(-1).mean(-1)
    np.testing.assert_allclose(b, h["v_bias"][:, -1], atol=1e-10, rtol=0)
    assert np.isfinite(h["q_final"]).all()
    path = case_path(CONTEXT["results_dir"], case)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **h)
    temporary.replace(path)
    meta = dict(
        num_states=size,
        num_actions=size,
        noise_mode=mode,
        method=method,
        delay=delay,
        expectile_tau=tau,
        updates=1000000,
        seed_ids=CONFIG["seeds"],
        gamma=0.999,
        alpha_q=0.1,
        alpha_v=0.1,
        q_target_tau=0.005,
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        execution="Strict-float compiled tabular reference; fastmath=False",
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return dict(
        size=size,
        mode=mode,
        method=method,
        D=delay,
        tau=tau,
        mean=float(b.mean()),
        sd=float(b.std(ddof=1)),
        seconds=time.monotonic() - start,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--sizes", nargs="+", type=int, choices=CONFIG["sizes"], default=CONFIG["sizes"]
    )
    parser.add_argument(
        "--noise-modes",
        nargs="+",
        choices=CONFIG["noise_modes"],
        default=CONFIG["noise_modes"],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--noise-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert args.workers > 0 and args.noise_workers > 0
    assert (
        CONFIG["gamma"],
        CONFIG["alpha_q"],
        CONFIG["alpha_v"],
        CONFIG["q_target_tau"],
    ) == (0.999, 0.1, 0.1, 0.005)
    assert CONFIG["updates"] == 1000000 and len(CONFIG["seeds"]) == 20
    selected = list(cases(args.sizes, args.noise_modes, args.methods))
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    conditions=len(selected),
                    updates=CONFIG["updates"],
                    seeds=20,
                    algorithms=args.methods,
                    results_dir=str(args.results_dir.resolve()),
                ),
                indent=2,
            )
        )
        return
    import sys

    import numpy as np

    sys.path.insert(0, str(CODE_DIR / "reference"))
    from engine import validate_compiled
    from stress_iid_min_ddq_bias import make_noise

    for name, digest in CONFIG["reference_hashes"].items():
        assert (
            hashlib.sha256((CODE_DIR / "reference" / name).read_bytes()).hexdigest()
            == digest
        )
    pending = []
    for case in selected:
        path = case_path(args.results_dir, case)
        if path.exists():
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["seed_ids"] == CONFIG["seeds"] and meta["updates"] == 1000000
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
        else:
            pending.append(case)
    if not pending:
        print(
            json.dumps(
                dict(
                    conditions=len(selected),
                    missing=0,
                    status="All requested files verified",
                )
            )
        )
        return
    (args.results_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (args.results_dir / "experiment_config.json").write_text(
        json.dumps(CONFIG, indent=2) + "\n"
    )
    checks = validate_compiled()
    assert len(checks) == 72 and all(c["passed"] for c in checks)
    (args.results_dir / "provenance/engine_validation.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    source_path = CODE_DIR / "reference/compare_tabular_methods.py"
    for size in args.sizes:
        for mode in args.noise_modes:
            jobs = [case for case in pending if case[:2] == (size, mode)]
            if not jobs:
                continue
            print(
                json.dumps(
                    dict(event="group_start", size=size, mode=mode, pending=len(jobs))
                ),
                flush=True,
            )
            noise = np.empty((20, 1000000, 2, size, size))
            with ThreadPoolExecutor(max_workers=args.noise_workers) as pool:
                futures = [
                    pool.submit(generate_seed, noise, i, size, mode) for i in range(20)
                ]
                for future in as_completed(futures):
                    print(
                        json.dumps(
                            dict(
                                event="noise_seed_ready",
                                size=size,
                                mode=mode,
                                seed=future.result(),
                            )
                        ),
                        flush=True,
                    )
            reference = make_noise(
                source_path.read_text(),
                source_path,
                [200, 219],
                2011,
                1.0,
                mode,
                0.99,
                64,
                0.35,
                1.0,
                size,
                size,
            )
            np.testing.assert_allclose(
                noise[[0, 19], :2011], reference, atol=1e-12, rtol=0
            )
            actions = np.stack(
                [
                    np.random.default_rng(2000003 + seed).integers(
                        max(4, size), size=1000000
                    )
                    % size
                    for seed in CONFIG["seeds"]
                ]
            )
            CONTEXT.update(noise=noise, actions=actions, results_dir=args.results_dir)
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                for future in as_completed(
                    [pool.submit(run_one, case) for case in jobs]
                ):
                    result = future.result()
                    print(json.dumps(result), flush=True)
                    with (args.results_dir / "provenance/completed_jobs.jsonl").open(
                        "a"
                    ) as f:
                        f.write(json.dumps(result) + "\n")
            CONTEXT.clear()
            del noise, actions
    assert all(case_path(args.results_dir, case).is_file() for case in selected)
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
