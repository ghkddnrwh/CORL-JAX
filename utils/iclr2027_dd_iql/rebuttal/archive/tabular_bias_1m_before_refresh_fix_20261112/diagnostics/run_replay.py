"""Replay only S=A=4 at 1M updates, retaining the existing seeds and updates."""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from capture import (
    diagnostic_schedule,
    pack,
    run_seed_diagnostics,
    SOURCE,
    validate_capture,
)
from common import case_path, cases, CONFIG, DEFAULT_RESULTS_DIR
from engine import METHOD_IDS
from measure import attach_outputs
from run_experiment import generate_seed

CONTEXT = {}
METHODS = ("max_min", "df_min", "sd_iql", "dd_iql")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(case):
    start = time.monotonic()
    _, mode, method, delay, tau = case
    steps = diagnostic_schedule()
    results = [
        run_seed_diagnostics(
            METHOD_IDS[method],
            CONTEXT["noise"][slot],
            CONTEXT["actions"][slot],
            delay or 1,
            tau or 0.9,
            steps,
        )
        for slot in range(20)
    ]
    h = pack(results, steps)
    original_path = case_path(DEFAULT_RESULTS_DIR, case)
    metadata = json.loads(original_path.with_suffix(".json").read_text())
    assert digest(original_path) == metadata["data_sha256"]
    matched = []
    with np.load(original_path) as original:
        for key in original.files:
            np.testing.assert_array_equal(
                h[key], original[key], err_msg=f"{case}: {key}"
            )
            matched.append(key)
    attach_outputs(h, method, delay or 1, tau or 0.9)
    assert np.all(np.isfinite(h["diagnostic_q"]))
    path = case_path(CONTEXT["results_dir"], case)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **h)
    temporary.replace(path)
    record = dict(
        method=method,
        noise_mode=mode,
        delay=delay,
        expectile_tau=tau,
        num_states=4,
        num_actions=4,
        updates=1000000,
        seed_ids=CONFIG["seeds"],
        diagnostic_samples=len(steps),
        observation_timing="before update t and before snapshot refresh",
        original_path=str(original_path),
        original_sha256=metadata["data_sha256"],
        original_keys_bitwise_equal=matched,
        original_bitwise_equal=True,
        actual_gate_and_backup_verified=True,
        operator_decomposition_verified=True,
        data_sha256=digest(path),
        seconds=time.monotonic() - start,
    )
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return {
        k: record[k]
        for k in (
            "method",
            "noise_mode",
            "delay",
            "expectile_tau",
            "original_bitwise_equal",
            "seconds",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR.parent / "tabular_bias_diagnostics_s4_1m",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--noise-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    provenance = args.results_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    print("VALIDATING_CAPTURE", flush=True)
    checks = validate_capture()
    (provenance / "capture_validation.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    (provenance / "instrumented_training_loop.py").write_text(SOURCE)
    print(
        json.dumps(dict(capture_checks=len(checks), bitwise_training_match=True)),
        flush=True,
    )
    if args.validate_only:
        return
    selected = list(cases([4], methods=METHODS))
    assert len(selected) == 50
    config = dict(
        CONFIG,
        sizes=[4],
        methods=list(METHODS),
        conditions=50,
        comparison="Only S=A=4 at 1M; exact replay with pre-update oracle diagnostics.",
        diagnostic_schedule=(
            "one pre-update sample per 1000-step block; fixed permutation of "
            "ages 0..999; additionally 1, 100k, 1M"
        ),
        diagnostic_reference="Q^circ=V^circ=F^circ=0",
    )
    (args.results_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    for mode in CONFIG["noise_modes"]:
        pending = []
        for case in selected:
            if case[1] != mode:
                continue
            path = case_path(args.results_dir, case)
            if path.exists() and path.with_suffix(".json").exists():
                meta = json.loads(path.with_suffix(".json").read_text())
                assert (
                    meta["original_bitwise_equal"]
                    and digest(path) == meta["data_sha256"]
                )
            else:
                pending.append(case)
        if not pending:
            continue
        print(
            json.dumps(dict(event="generating_noise", mode=mode, pending=len(pending))),
            flush=True,
        )
        noise = np.empty((20, 1000000, 2, 4, 4))
        with ThreadPoolExecutor(max_workers=args.noise_workers) as pool:
            futures = [
                pool.submit(generate_seed, noise, slot, 4, mode) for slot in range(20)
            ]
            for future in as_completed(futures):
                print(
                    json.dumps(
                        dict(event="noise_ready", mode=mode, seed=future.result())
                    ),
                    flush=True,
                )
        actions = np.stack(
            [
                np.random.default_rng(2000003 + seed).integers(4, size=1000000)
                for seed in CONFIG["seeds"]
            ]
        )
        CONTEXT.update(noise=noise, actions=actions, results_dir=args.results_dir)
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            for future in as_completed([pool.submit(run_one, case) for case in pending]):
                record = future.result()
                print(json.dumps(record), flush=True)
                with (provenance / "completed_jobs.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
        CONTEXT.clear()
        del noise, actions
    assert all(case_path(args.results_dir, c).is_file() for c in selected)
    print("REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
