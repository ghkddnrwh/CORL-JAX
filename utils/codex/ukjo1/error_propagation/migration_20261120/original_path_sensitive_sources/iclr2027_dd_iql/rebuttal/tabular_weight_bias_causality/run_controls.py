"""Paired oracle controls for bias versus mean-weight causality (S=A=4).

The IID DF correction integrates the known min-of-two-N(0,1) receiver noise:
  c(g) = E[w(Z-g) Z] / E[w(Z-g)], g=V-clean_Q.
Subtracting c(g) from V's receiver leaves its actual gate untouched and cancels
the conditional expected readout-noise term. This is a diagnostic oracle, not
a proposed algorithm. It includes the min-noise mean AND selection covariance.
The correction is not applied to temporally correlated NN-like noise.

Other interventions replace clean Q with oracle zero in filter/V readouts, or
add +0.6 to DD's Q Bellman target at every update. No old files are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_weight_bias_causality"
NOISES = ("nn_like_field_iid", "nn_like_field")
METHODS = (("df_min", 0, 1), ("dd_iql", 2, 100))
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
    "raw_receiver_noise",
    "weight_times_raw_noise",
    "weight_times_clean_minus_v",
    "weight_times_correction",
    "v_before",
    "clean_receiver_q",
)


@njit(cache=True)
def iid_df_moments(gap, tau):
    """Exact conditional moments when the two DF clean heads are identical."""
    survival = 0.5 * math.erfc(gap / math.sqrt(2))
    phi = math.exp(-gap * gap / 2) / math.sqrt(2 * math.pi)
    high_first = 2 * phi * survival - 0.5 / math.sqrt(math.pi) * math.erfc(gap)
    mean_w = (1 - tau) + (2 * tau - 1) * survival**2
    weighted_z = -(1 - tau) / math.sqrt(math.pi) + (2 * tau - 1) * high_first
    return mean_w, weighted_z


@njit(cache=True, fastmath=False)
def simulate(
    method,
    noise,
    actions,
    delay,
    tau,
    steps,
    oracle_clean_q,
    correction_fraction,
    bellman_offset,
):
    updates, heads, states, acount = noise.shape
    vheads = 1 if method == 0 else heads
    q = np.zeros((heads, states, acount))
    target = q.copy()
    value = np.zeros((vheads, states))
    qin = q.copy()
    frozen_clean, frozen_y = q.copy(), q.copy()
    frozen_v = value.copy()
    weights = np.zeros((vheads, states, acount))
    old_v, next_v = np.zeros(vheads), np.zeros(vheads)
    observed, clean_obs = np.zeros(heads), np.zeros(heads)
    count = len(steps)
    dq = np.empty((count, heads, states, acount))
    dt, dy, dc = np.empty_like(dq), np.empty_like(dq), np.empty_like(dq)
    dv = np.empty((count, vheads, states))
    dfv = np.empty_like(dv)
    dw = np.empty((count, vheads, states, acount))
    moments = np.zeros((updates // 1000, 7))
    state, index = 0, 0
    for t in range(updates):
        if oracle_clean_q:
            qin[:] = 0.0
        else:
            qin[:] = target
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
        sample = index < count and t + 1 == steps[index]
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
            correction = 0.0
            if correction_fraction != 0.0:
                # method=DF, D=1 and IID are validated by the caller.
                # Identical Q heads follow from identical zero initialization
                # and the exact same Q backup at every DF update.
                w_expected, wz_expected = iid_df_moments(old_v[0] - clean, tau)
                correction = correction_fraction * wz_expected / w_expected
            backup = 0.999 * next_v[0] + bellman_offset
            for i in range(heads):
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
            rate = 0.1 * weight
            value[0, state] = (1.0 - rate) * old_v[0] + rate * (obs - correction)
            row = t // 1000
            moments[row, 0] += weight / 1000
            moments[row, 1] += (obs - clean) / 1000
            moments[row, 2] += weight * (obs - clean) / 1000
            moments[row, 3] += weight * (clean - old_v[0]) / 1000
            moments[row, 4] += weight * correction / 1000
            moments[row, 5] += old_v[0] / 1000
            moments[row, 6] += clean / 1000
        else:
            for i in range(heads):
                weight = weights[i, state, action]
                backup = 0.999 * next_v[i] + bellman_offset
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                rate = 0.1 * weight
                value[i, state] = (1.0 - rate) * old_v[i] + rate * observed[i]
                row, denom = t // 1000, 1000 * heads
                moments[row, 0] += weight / denom
                moments[row, 1] += (observed[i] - clean_obs[i]) / denom
                moments[row, 2] += weight * (observed[i] - clean_obs[i]) / denom
                moments[row, 3] += weight * (clean_obs[i] - old_v[i]) / denom
                moments[row, 5] += old_v[i] / denom
                moments[row, 6] += clean_obs[i] / denom
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


def controls(mode, method):
    result = [("baseline", False, 0.0, 0.0), ("oracle_clean_q", True, 0.0, 0.0)]
    if method == "df_min" and mode == "nn_like_field_iid":
        result += [
            ("noise_bias_half", False, 0.5, 0.0),
            ("noise_bias_full", False, 1.0, 0.0),
        ]
    if method == "dd_iql":
        result += [("bellman_plus_0.6", False, 0.0, 0.6)]
    return result


def run_job(mode, seed, out):
    noise = np.empty((1, 1_000_000, 2, 4, 4))
    buffer = np.lib.stride_tricks.as_strided(
        noise, shape=(20,) + noise.shape[1:], strides=(0,) + noise.strides[1:]
    )
    generate_seed(buffer, seed - 200, mode)
    field = noise[0]
    actions = np.random.default_rng(2_000_003 + seed).integers(4, size=1_000_000)
    steps = diagnostic_schedule()
    checks = []
    for method, method_id, delay in METHODS:
        source = SOURCE / "raw/s4_a4" / mode / f"d{delay}/tau0.99/{method}.npz"
        meta = json.loads(source.with_suffix(".json").read_text())
        assert digest(source) == meta["data_sha256"]
        for name, oracle_q, fraction, forcing in controls(mode, method):
            if fraction:
                assert method == "df_min" and delay == 1 and mode.endswith("_iid")
            values = simulate(
                method_id,
                field,
                actions,
                delay,
                0.99,
                steps,
                oracle_q,
                fraction,
                forcing,
            )
            h = dict(zip(KEYS, values))
            h["diagnostic_step"] = steps
            h["moment_end_step"] = np.arange(1000, 1_000_001, 1000)
            if name == "baseline":
                with np.load(source, allow_pickle=False) as ref:
                    np.testing.assert_array_equal(
                        field[steps - 1], ref["diagnostic_noise"][seed - 200]
                    )
                    for key in KEYS:
                        if key not in ("update_moments", "diagnostic_delayed_clean_q"):
                            np.testing.assert_array_equal(h[key], ref[key][seed - 200])
                checks.append(
                    dict(
                        source=str(source.relative_to(ROOT)),
                        sha256=meta["data_sha256"],
                        seed=seed,
                    )
                )
            destination = out / "raw" / mode / method / name / f"seed{seed}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **h)
    return dict(mode=mode, seed=seed, baseline_bitwise_checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(200, 220)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Compile before workers fork. Keep a copy of the actual executed sources.
    tiny = np.zeros((1000, 2, 4, 4))
    simulate(
        0,
        tiny,
        np.zeros(1000, dtype=np.int64),
        1,
        0.99,
        np.array([1, 1000]),
        False,
        0.0,
        0.0,
    )
    provenance = args.output / "provenance"
    provenance.mkdir(exist_ok=True)
    dependencies = [
        Path(__file__),
        ORIGINAL / "engine.py",
        ORIGINAL / "noise.py",
        ORIGINAL / "common.py",
        ORIGINAL / "experiment_config.json",
    ]
    hashes = {}
    for source in dependencies:
        (provenance / source.name).write_bytes(source.read_bytes())
        hashes[str(source.relative_to(ROOT))] = digest(source)
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [
            pool.submit(run_job, mode, seed, args.output)
            for mode in NOISES
            for seed in args.seeds
        ]
        for task in as_completed(tasks):
            record = task.result()
            checks.append(record)
            print(
                json.dumps(
                    dict(
                        mode=record["mode"],
                        seed=record["seed"],
                        baseline_checks=len(record["baseline_bitwise_checks"]),
                    )
                ),
                flush=True,
            )
    manifest = dict(
        seeds=args.seeds,
        tau=0.99,
        updates=1_000_000,
        moments=MOMENTS,
        noise_modes=NOISES,
        methods=METHODS,
        controls={
            mode: {method: controls(mode, method) for method, _, _ in METHODS}
            for mode in NOISES
        },
        baseline_checks=checks,
        source_sha256=hashes,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
