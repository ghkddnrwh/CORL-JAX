"""Paired DF D=1 / DD D=100 controls for the four-state weight hypothesis.

The interventions replace clean target-Q inputs to filter construction AND V
regression by a within-state action mean or a whole-head table mean, at every update.
Injected readout fields, actions, Q backups, target EMA and V update rates stay
the same. This removes learned action-specific Q deviations from the Q-to-V
path throughout training; it does not remove extreme injected readouts.
Original code and raw logs are read-only. Baseline arrays must match saved runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = ROOT / "utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m"
sys.path.insert(0, str(ORIGINAL))
from engine import diagnostic_schedule
from noise import generate_seed

SOURCE = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m"
OUTPUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_df_dd_weight_hypothesis"
CASES = (("df_min", 0, 1), ("dd_iql", 2, 100))
NOISES = ("nn_like_field_iid", "nn_like_field")
KEYS = (
    "diagnostic_q",
    "diagnostic_target_q",
    "diagnostic_value",
    "diagnostic_delayed_observed_q",
    "diagnostic_delayed_clean_q",
    "diagnostic_delayed_value",
    "diagnostic_weights",
    "q_final",
    "q_target_final",
    "value_final",
    "update_moments",
)
MOMENTS = (
    "weight",
    "receiver_error",
    "weight_times_receiver_error",
    "weight_times_clean_minus_v",
    "v_before",
    "clean_receiver_q",
)


@njit(cache=True, fastmath=False)
def simulate(method, noise, actions, delay, tau, steps, flatten):
    updates, heads, states, acount = noise.shape
    vheads = 1 if method == 0 else heads
    q = np.zeros((heads, states, acount))
    target = q.copy()
    value = np.zeros((vheads, states))
    frozen_clean = q.copy()
    frozen_y = q.copy()
    frozen_v = value.copy()
    weights = np.zeros((vheads, states, acount))
    qin = q.copy()
    old_v, next_v = np.zeros(vheads), np.zeros(vheads)
    observed = np.zeros(heads)
    clean_obs = np.zeros(heads)
    npoints = len(steps)
    dq = np.empty((npoints, heads, states, acount))
    dt = np.empty_like(dq)
    dy = np.empty_like(dq)
    dc = np.empty_like(dq)
    dv = np.empty((npoints, vheads, states))
    dfv = np.empty_like(dv)
    dw = np.empty((npoints, vheads, states, acount))
    # Every row averages 1000 actual sampled updates, not diagnostic snapshots.
    moments = np.zeros((updates // 1000, 6))
    state, index = 0, 0
    for t in range(updates):
        for i in range(heads):
            for s in range(states):
                if flatten == 2:
                    mu = 0.0
                    for ss in range(states):
                        for aa in range(acount):
                            mu += target[i, ss, aa] / (states * acount)
                    for a in range(acount):
                        qin[i, s, a] = mu
                elif flatten == 1:
                    mu = 0.0
                    for a in range(acount):
                        mu += target[i, s, a] / acount
                    for a in range(acount):
                        qin[i, s, a] = mu
                else:
                    qin[i, s, :] = target[i, s, :]
        if t % delay == 0:
            frozen_clean[:] = qin
            for i in range(heads):
                for s in range(states):
                    for a in range(acount):
                        frozen_y[i, s, a] = qin[i, s, a] + noise[t, i, s, a]
            frozen_v[:] = value
            for i in range(vheads):
                for s in range(states):
                    for a in range(acount):
                        if method == 0:
                            adv = min(frozen_y[0, s, a], frozen_y[1, s, a])
                            adv -= frozen_v[0, s]
                        else:
                            peer = 1 - i
                            adv = frozen_y[peer, s, a] - frozen_v[peer, s]
                        weights[i, s, a] = tau if adv >= 0 else 1 - tau
        action = actions[t]
        successor = action % states
        sample = index < npoints and t + 1 == steps[index]
        if sample:
            dq[index], dt[index], dv[index] = q, target, value
            dy[index], dc[index] = frozen_y, frozen_clean
            dfv[index], dw[index] = frozen_v, weights
        for i in range(vheads):
            old_v[i], next_v[i] = value[i, state], value[i, successor]
        for i in range(heads):
            clean_obs[i] = qin[i, state, action]
            observed[i] = clean_obs[i] + noise[t, i, state, action]
        if method == 0:
            obs = min(observed[0], observed[1])
            clean = min(clean_obs[0], clean_obs[1])
            weight = weights[0, state, action]
            backup = 0.999 * next_v[0]
            for i in range(heads):
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
            rate = 0.1 * weight
            value[0, state] = (1.0 - rate) * old_v[0] + rate * obs
            row = t // 1000
            moments[row, 0] += weight / 1000
            moments[row, 1] += (obs - clean) / 1000
            moments[row, 2] += weight * (obs - clean) / 1000
            moments[row, 3] += weight * (clean - old_v[0]) / 1000
            moments[row, 4] += old_v[0] / 1000
            moments[row, 5] += clean / 1000
        else:
            for i in range(heads):
                weight = weights[i, state, action]
                backup = 0.999 * next_v[i]
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                rate = 0.1 * weight
                value[i, state] = (1.0 - rate) * old_v[i] + rate * observed[i]
                row, count = t // 1000, 1000 * heads
                moments[row, 0] += weight / count
                moments[row, 1] += (observed[i] - clean_obs[i]) / count
                moments[row, 2] += weight * (observed[i] - clean_obs[i]) / count
                moments[row, 3] += weight * (clean_obs[i] - old_v[i]) / count
                moments[row, 4] += old_v[i] / count
                moments[row, 5] += clean_obs[i] / count
        for i in range(heads):
            for s in range(states):
                for a in range(acount):
                    target[i, s, a] = 0.995 * target[i, s, a] + 0.005 * q[i, s, a]
        if sample:
            index += 1
        state = successor
    return dq, dt, dv, dy, dc, dfv, dw, q, target, value, moments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job(mode, seed, out):
    seed = int(seed)
    noise = np.empty((1, 1_000_000, 2, 4, 4))
    # generate_seed maps slot to seed=200+slot, so use a strided slot view.
    # A 20-slot broadcast avoids allocating 20 full noise trajectories.
    shared = np.lib.stride_tricks.as_strided(
        noise, shape=(20,) + noise.shape[1:], strides=(0,) + noise.strides[1:]
    )
    generate_seed(shared, seed - 200, mode)
    field = noise[0]
    actions = np.random.default_rng(2_000_003 + seed).integers(4, size=1_000_000)
    steps = diagnostic_schedule()
    verified = []
    for method, method_id, delay in CASES:
        source = SOURCE / "raw/s4_a4" / mode / f"d{delay}/tau0.99/{method}.npz"
        expected_hash = json.loads(source.with_suffix(".json").read_text())[
            "data_sha256"
        ]
        assert digest(source) == expected_hash
        for flatten, variant in (
            (0, "baseline"),
            (1, "flat_clean_inputs"),
            (2, "flat_whole_table"),
        ):
            result = simulate(method_id, field, actions, delay, 0.99, steps, flatten)
            h = dict(zip(KEYS, result))
            h["diagnostic_step"] = steps
            h["moment_end_step"] = np.arange(1000, 1_000_001, 1000)
            if not flatten:
                with np.load(source, allow_pickle=False) as ref:
                    np.testing.assert_array_equal(
                        field[steps - 1], ref["diagnostic_noise"][seed - 200]
                    )
                    for key in KEYS:
                        if key in ("update_moments", "diagnostic_delayed_clean_q"):
                            continue
                        np.testing.assert_array_equal(h[key], ref[key][seed - 200])
                verified.append(str(source.relative_to(ROOT)))
            path = out / "raw" / mode / method / variant / f"seed{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **h)
    return dict(noise=mode, seed=seed, baseline_bitwise_verified=verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 220)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Compile once before fork, so worker compilation cannot race.
    tiny = np.zeros((1000, 2, 4, 4))
    simulate(0, tiny, np.zeros(1000, dtype=np.int64), 1, 0.99, np.array([1, 1000]), 0)
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(job, mode, seed, args.output)
            for mode in NOISES
            for seed in args.seeds
        ]
        for future in as_completed(futures):
            record = future.result()
            checks.append(record)
            print(json.dumps(record), flush=True)
    manifest = dict(
        updates=1_000_000,
        tau=0.99,
        seeds=args.seeds,
        cases=CASES,
        noise_modes=NOISES,
        variants=["baseline", "flat_clean_inputs", "flat_whole_table"],
        intervention=(
            "Within-state action mean or whole-table mean clean Q in filter "
            "and V receiver; every update"
        ),
        moments=MOMENTS,
        baseline_bitwise_checks=checks,
        code_sha256=digest(Path(__file__)),
        dependency_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ORIGINAL / "engine.py",
                ORIGINAL / "noise.py",
                ORIGINAL / "common.py",
                ORIGINAL / "experiment_config.json",
            ]
        },
    )
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
