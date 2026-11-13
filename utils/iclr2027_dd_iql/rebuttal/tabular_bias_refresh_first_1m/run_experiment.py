"""Run only DF-min, SD and DD with current-first filters, 4 states and 1M updates."""

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
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

import numpy as np
from common import case_path, cases, CODE_DIR, CONFIG, DEFAULT_RESULTS_DIR, METHOD_IDS
from engine import diagnostic_schedule, pack, run_seed
from measure import attach_outputs
from noise import generate_seed

CONTEXT = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(case):
    start = time.monotonic()
    _, mode, method, delay, tau = case
    steps = diagnostic_schedule()
    results = [
        run_seed(
            METHOD_IDS[method],
            CONTEXT["noise"][mode][slot],
            CONTEXT["actions"][slot],
            delay,
            tau,
            steps,
        )
        for slot in range(20)
    ]
    h = pack(results, steps, CONFIG["updates"], delay)
    attach_outputs(h, method, delay, tau)
    np.testing.assert_array_equal(h["refresh_count"], CONFIG["updates"] // delay)
    np.testing.assert_array_equal(h["diagnostic_noise_lag"], (steps - 1) % delay)
    assert len(np.unique(h["diagnostic_noise_lag"])) == delay
    assert np.isfinite(h["q_final"]).all()
    path = case_path(CONTEXT["results_dir"], case)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **h)
    temporary.replace(path)
    q_bias = h["q_final"].mean(1).max(-1).mean(-1)
    record = dict(
        method=method,
        noise_mode=mode,
        delay=delay,
        expectile_tau=tau,
        num_states=4,
        num_actions=4,
        updates=CONFIG["updates"],
        seed_ids=CONFIG["seeds"],
        snapshot_timing=CONFIG["snapshot_timing"],
        observation_timing="after block-start refresh, before training update",
        refresh_count_per_seed=CONFIG["updates"] // delay,
        zero_noise_lag_updates_per_seed=CONFIG["updates"] // delay,
        diagnostic_samples=len(steps),
        actual_gate_and_backup_verified=True,
        noise_lags_verified=list(range(delay)),
        data_sha256=digest(path),
        seconds=time.monotonic() - start,
        final_max_q_bias_mean=float(q_bias.mean()),
        final_max_q_bias_sd=float(q_bias.std(ddof=1)),
    )
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return {
        k: record[k]
        for k in (
            "method",
            "noise_mode",
            "delay",
            "expectile_tau",
            "final_max_q_bias_mean",
            "final_max_q_bias_sd",
            "seconds",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--noise-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = list(cases())
    assert len(selected) == 48 and args.workers > 0 and args.noise_workers > 0
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    conditions=48,
                    trajectories=960,
                    config=CONFIG,
                    results_dir=str(args.results_dir),
                )
            )
        )
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.results_dir / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise ValueError(f"Existing results have different settings: {config_path}")
    config_path.write_text(json.dumps(CONFIG, indent=2) + "\n")
    provenance = args.results_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    source_hashes = {}
    for source in CODE_DIR.iterdir():
        if source.is_file() and source.suffix in (".py", ".json", ".txt", ".md"):
            target = provenance / "code_snapshot" / source.name
            target.parent.mkdir(exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f"Code changed since this run: {source}")
            shutil.copy2(source, target)
            source_hashes[source.name] = digest(source)
    (provenance / "code_manifest.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n"
    )
    # Compile once in the parent; forked workers reuse the validated implementation.
    tiny_noise = np.zeros((2, 2, 4, 4))
    run_seed(0, tiny_noise, np.zeros(2, dtype=np.int64), 1, 0.9, np.array([1, 2]))
    all_pending = []
    noise_by_mode = {}
    for mode in CONFIG["noise_modes"]:
        pending = []
        for case in selected:
            if case[1] != mode:
                continue
            path = case_path(args.results_dir, case)
            if path.exists():
                meta = json.loads(path.with_suffix(".json").read_text())
                assert meta["snapshot_timing"] == CONFIG["snapshot_timing"]
                assert digest(path) == meta["data_sha256"]
            else:
                pending.append(case)
        if not pending:
            continue
        print(
            json.dumps(
                dict(event="generating_noise", mode=mode, conditions=len(pending))
            ),
            flush=True,
        )
        noise = np.empty((20, CONFIG["updates"], 2, 4, 4))
        with ThreadPoolExecutor(max_workers=args.noise_workers) as pool:
            futures = [
                pool.submit(generate_seed, noise, slot, mode) for slot in range(20)
            ]
            for future in as_completed(futures):
                print(
                    json.dumps(
                        dict(event="noise_ready", mode=mode, seed=future.result())
                    ),
                    flush=True,
                )
        noise_by_mode[mode] = noise
        all_pending.extend(pending)
    actions = np.stack(
        [
            np.random.default_rng(2000003 + seed).integers(4, size=CONFIG["updates"])
            for seed in CONFIG["seeds"]
        ]
    )
    CONTEXT.update(noise=noise_by_mode, actions=actions, results_dir=args.results_dir)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        futures = [pool.submit(run_one, case) for case in all_pending]
        for future in as_completed(futures):
            record = future.result()
            print(json.dumps(record), flush=True)
            with (provenance / "completed_jobs.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
    CONTEXT.clear()
    assert all(case_path(args.results_dir, case).exists() for case in selected)
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
