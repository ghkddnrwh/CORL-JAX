"""Standalone four-state bias experiment: DF-IQL, SD-IQL and DD-IQL.

Run ``python tabular_bias_s4.py --help`` for full and shortened runs.
All experiment, diagnostic and plotting code is contained in this file.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep worker processes from creating additional BLAS/OpenMP thread pools.
for variable in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "1"

import numba
import numpy as np
import scipy
from numba import njit
from scipy.signal import lfilter

METHODS = ("df_iql", "sd_iql", "dd_iql")
LABELS = {"df_iql": "DF-IQL", "sd_iql": "SD-IQL", "dd_iql": "DD-IQL"}
COLORS = {"df_iql": "#cc79a7", "sd_iql": "#d55e00", "dd_iql": "#009e73"}
NOISES = ("iid", "nn_like_field")
NOISE_LABELS = {"iid": "IID", "nn_like_field": "NN-like field"}
PROXIES = (
    "oracle",
    "accumulated",
    "readout",
    "td_update",
    "td_observed",
    "temporal_observed",
    "temporal_clean",
    "loo_observed",
    "loo_clean",
)


def diagnostic_schedule(updates):
    """Sample all 1000 delay phases and exactly 101 tail times at 1M steps."""
    if updates == 1000000:
        offsets = np.random.default_rng(20261112).permutation(1000)
        return np.unique(
            np.r_[1000 * np.arange(1000) + offsets + 1, 1, 100000, updates]
        ).astype(np.int64)
    return np.unique(
        np.r_[np.linspace(1, updates, min(updates, 1000), dtype=np.int64), updates]
    )


def generate_noise(seed, mode, updates):
    """Unit-variance Gaussian errors; NN-like fields use AR(1), rho=0.99."""
    size = actions = 4
    noise = np.empty((updates, 2, size, actions))
    rng = np.random.default_rng(100003 + seed)
    if mode == "iid":
        for start in range(0, updates, 20000):
            end = min(start + 20000, updates)
            noise[start:end] = rng.standard_normal((end - start, 2, size, actions))[
                ..., :size
            ]
        return noise
    if mode != "nn_like_field":
        raise ValueError(mode)
    # An extra feature-coordinate anchor is unused by the four-state MDP.
    coords = np.empty((1, size + 1, actions, 2 + actions))
    for c in range(size + 1):
        for a in range(actions):
            coords[0, c, a, :2] = (0.0, c / size)
            coords[0, c, a, 2:] = np.eye(actions)[a]
    projection = rng.normal(0.0, 1.0 / 0.35, size=(2 + actions, 63))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(63,))
    phi = np.concatenate(
        (np.full((1, size + 1, actions, 1), 0.25), np.cos(coords @ projection + phase)),
        axis=-1,
    )
    phi = phi / np.linalg.norm(phi, axis=-1, keepdims=True)
    shared = rng.standard_normal(64)
    private = rng.standard_normal((2, 64))
    latent = np.sqrt(1.0) * private + np.sqrt(0.0) * shared[None, :]
    initial = np.einsum("hwad,nd->nhwa", phi, latent)[:, 0, :size]
    carry = np.zeros((1, 2, size, actions))
    for start in range(0, updates, 20000):
        end = min(start + 20000, updates)
        length = end - start
        innovations = np.empty((length, 2, size, actions))
        offset = 1 if start == 0 else 0
        if offset:
            innovations[0] = initial
        draws = rng.standard_normal((length - offset, 3, 64))
        projected = np.einsum(
            "tnd,sad->tnsa", draws[:, 1:], phi[0, :size], optimize=True
        )
        innovations[offset:] = np.sqrt(1 - 0.99**2) * projected
        values, carry = lfilter([1.0], [1.0, -0.99], innovations, axis=0, zi=carry)
        noise[start:end] = values[..., :size]
    return noise


@njit(cache=True, fastmath=False)
def run_seed(method, noise, actions, delay, tau, diagnostic_steps, log_every=1000):
    updates, n, states, action_count = noise.shape
    assert 0 <= method <= 2 and n == 2 and states == 4 and action_count == 4
    vn = 1 if method == 0 else n
    q = np.zeros((n, states, action_count))
    target = q.copy()
    value = np.zeros((vn, states))
    frozen_y = np.zeros_like(q)
    frozen_noise = np.zeros_like(q)
    frozen_v = value.copy()
    weights = np.zeros((vn, states, action_count))
    count = 1 + (updates + log_every - 1) // log_every
    bias = np.zeros(count)
    mean_q = np.zeros(count)
    rmse = np.zeros(count)
    value_curve = np.zeros((count, states))
    q100 = q.copy()
    v100 = value.copy()
    target100 = target.copy()
    samples = len(diagnostic_steps)
    dq = np.empty((samples, n, states, action_count))
    dt = np.empty_like(dq)
    de = np.empty_like(dq)
    df = np.empty_like(dq)
    dfn = np.empty_like(dq)
    dv = np.empty((samples, vn, states))
    dfv = np.empty_like(dv)
    dw = np.empty((samples, vn, states, action_count))
    used_w = np.empty((samples, vn))
    used_backup = np.empty((samples, n))
    ds = np.empty(samples, dtype=np.int64)
    da = np.empty(samples, dtype=np.int64)
    old_v = np.zeros(vn)
    next_v = np.zeros(vn)
    observed = np.zeros(n)
    state = 0
    logged = 0
    sample_index = 0
    refresh_count = 0
    for t in range(updates):
        # The first update consumes the very same readout noise as the new filter.
        if t % delay == 0:
            refresh_count += 1
            for i in range(n):
                for s in range(states):
                    for a in range(action_count):
                        frozen_y[i, s, a] = target[i, s, a] + noise[t, i, s, a]
                        frozen_noise[i, s, a] = noise[t, i, s, a]
            frozen_v[:] = value
            for i in range(vn):
                for s in range(states):
                    for a in range(action_count):
                        if method == 0:
                            advantage = (
                                min(frozen_y[0, s, a], frozen_y[1, s, a])
                                - frozen_v[0, s]
                            )
                        else:
                            source = 1 - i if method == 2 else i
                            advantage = frozen_y[source, s, a] - frozen_v[source, s]
                        weights[i, s, a] = tau if advantage >= 0.0 else 1.0 - tau
        action = actions[t]
        successor = action % states
        sample = sample_index < samples and t + 1 == diagnostic_steps[sample_index]
        if sample:
            dq[sample_index] = q
            dt[sample_index] = target
            de[sample_index] = noise[t]
            df[sample_index] = frozen_y
            dfn[sample_index] = frozen_noise
            dv[sample_index] = value
            dfv[sample_index] = frozen_v
            dw[sample_index] = weights
            ds[sample_index] = state
            da[sample_index] = action
        for i in range(vn):
            old_v[i] = value[i, state]
            next_v[i] = value[i, successor]
        for i in range(n):
            observed[i] = target[i, state, action] + noise[t, i, state, action]
        if method == 0:
            observation = min(observed[0], observed[1])
            weight = weights[0, state, action]
            backup = 0.999 * next_v[0]
            for i in range(n):
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
            rate = 0.1 * weight
            value[0, state] = (1.0 - rate) * old_v[0] + rate * observation
            if sample:
                used_w[sample_index, 0] = weight
                used_backup[sample_index, :] = backup
        else:
            for i in range(n):
                weight = weights[i, state, action]
                backup = 0.999 * next_v[i]
                q[i, state, action] = 0.9 * q[i, state, action] + 0.1 * backup
                rate = 0.1 * weight
                value[i, state] = (1.0 - rate) * old_v[i] + rate * observed[i]
                if sample:
                    used_w[sample_index, i] = weight
                    used_backup[sample_index, i] = backup
        for i in range(n):
            for s in range(states):
                for a in range(action_count):
                    target[i, s, a] = 0.995 * target[i, s, a] + 0.005 * q[i, s, a]
        if sample:
            sample_index += 1
        state = successor
        if t + 1 == 100000:
            q100 = q.copy()
            v100 = value.copy()
            target100 = target.copy()
        if (t + 1) % log_every == 0 or t + 1 == updates:
            logged += 1
            total = 0.0
            square_total = 0.0
            max_total = 0.0
            for s in range(states):
                best = -np.inf
                for a in range(action_count):
                    estimate = (q[0, s, a] + q[1, s, a]) / 2.0
                    best = max(best, estimate)
                    total += estimate
                    square_total += estimate * estimate
                max_total += best
                value_curve[logged, s] = (
                    value[0, s] if vn == 1 else (value[0, s] + value[1, s]) / 2.0
                )
            bias[logged] = max_total / states
            mean_q[logged] = total / (states * action_count)
            rmse[logged] = np.sqrt(square_total / (states * action_count))
    return (
        q,
        bias,
        mean_q,
        rmse,
        value_curve,
        value,
        target,
        q100,
        v100,
        target100,
        dq,
        dt,
        de,
        df,
        dfn,
        dv,
        dfv,
        dw,
        used_w,
        used_backup,
        ds,
        da,
        refresh_count,
    )


OUTPUT_KEYS = (
    "q_final",
    "max_q_bias_curve",
    "q_mean_bias_curve",
    "q_rmse_curve",
    "value_table",
    "value_final",
    "q_target_final",
    "checkpoint_q_100k",
    "checkpoint_value_100k",
    "checkpoint_q_target_100k",
    "diagnostic_q",
    "diagnostic_target_q",
    "diagnostic_noise",
    "diagnostic_delayed_observed_q",
    "diagnostic_delayed_noise",
    "diagnostic_value",
    "diagnostic_delayed_value",
    "diagnostic_weights",
    "diagnostic_used_weight",
    "diagnostic_used_backup",
    "diagnostic_state",
    "diagnostic_action",
    "refresh_count",
)


def pack(results, steps, updates, delay, log_every=1000):
    h = {key: np.stack([r[k] for r in results]) for k, key in enumerate(OUTPUT_KEYS)}
    h["step"] = np.unique(np.r_[np.arange(0, updates + 1, log_every), updates])
    h["diagnostic_step"] = steps
    h["diagnostic_snapshot_age"] = (steps - 1) % delay
    h["diagnostic_noise_lag"] = h["diagnostic_snapshot_age"].copy()
    h["value_bias"] = h["value_table"].mean(-1)
    h["value_rmse"] = np.sqrt((h["value_table"] ** 2).mean(-1))
    h["greedy_policy_final"] = h["q_final"].mean(1).argmax(-1)
    return h


def derive(h, method, delay, tau):
    """Recover the actual pre-update gates, receiver errors and filter outputs."""
    q = h["diagnostic_q"]
    target = h["diagnostic_target_q"]
    epsilon = h["diagnostic_noise"]
    value = h["diagnostic_value"]
    seeds, samples, _, states, actions = q.shape
    row = np.arange(seeds)[:, None]
    col = np.arange(samples)[None, :]
    visited_s = h["diagnostic_state"]
    visited_a = h["diagnostic_action"]
    successor = visited_a % states
    y = target + epsilon
    frozen_y = h["diagnostic_delayed_observed_q"]
    frozen_clean = frozen_y - h["diagnostic_delayed_noise"]
    frozen_v = h["diagnostic_delayed_value"]
    age_zero = (h["diagnostic_step"] - 1) % delay == 0
    frozen_clean[:, age_zero] = target[:, age_zero]

    if method == "df_iql":
        receiver = y.min(2, keepdims=True)
        receiver_clean = target.min(2, keepdims=True)
        snapshot = frozen_y.min(2, keepdims=True)
        snapshot_clean = frozen_clean.min(2, keepdims=True)
        source = snapshot
        advantage = source - frozen_v[..., None]
        online = q.mean(2, keepdims=True)
        loo_observed = y[:, :, :1] - y[:, :, 1:]
        loo_clean = target[:, :, :1] - target[:, :, 1:]
    else:
        receiver, receiver_clean = y, target
        snapshot, snapshot_clean = frozen_y, frozen_clean
        peer = [1, 0] if method == "dd_iql" else [0, 1]
        source = frozen_y[:, :, peer]
        advantage = source - frozen_v[:, :, peer, :, None]
        online = q
        loo_observed = y - y[:, :, ::-1]
        loo_clean = target - target[:, :, ::-1]
    weights = np.where(advantage >= 0, tau, 1.0 - tau)
    np.testing.assert_array_equal(weights, h["diagnostic_weights"])
    used = weights.transpose(0, 1, 3, 4, 2)[row, col, visited_s, visited_a]
    np.testing.assert_array_equal(used, h["diagnostic_used_weight"])
    backup = 0.999 * value.transpose(0, 1, 3, 2)[row, col, successor]
    if method == "df_iql":
        backup = np.repeat(backup, 2, -1)
    np.testing.assert_array_equal(backup, h["diagnostic_used_backup"])
    normalized = weights / weights.sum(-1, keepdims=True)
    operator = (normalized * receiver).sum(-1)
    accumulated = (normalized * receiver_clean).sum(-1)
    readout_error = receiver - receiver_clean
    readout = (normalized * readout_error).sum(-1)
    np.testing.assert_allclose(operator, accumulated + readout, rtol=0, atol=2e-12)
    next_value = value[..., np.arange(actions) % states]
    bellman_target = 0.999 * next_value[..., None, :]
    proxies = dict(
        oracle=receiver,
        accumulated=receiver_clean,
        readout=readout_error,
        td_update=online - bellman_target,
        td_observed=receiver - bellman_target,
        temporal_observed=receiver - snapshot,
        temporal_clean=receiver_clean - snapshot_clean,
        loo_observed=loo_observed,
        loo_clean=loo_clean,
    )
    return dict(
        operator=operator,
        operator_accumulated=accumulated,
        operator_readout=readout,
        weights=weights,
        proxies=proxies,
        q=q,
        q_ensemble=q.mean(2),
        value=value,
    )


def coupling(weights, error, cross_section=False):
    """Temporal statistics at fixed receiver/state/action, then average units."""
    if cross_section:
        # Convert fixed-time, within-state/action observations into the same
        # reducer layout: seed x observations x fixed units x 1 x 1.
        seeds, samples, receivers, states, actions = weights.shape
        weights = weights.transpose(0, 3, 4, 1, 2).reshape(
            seeds, states * actions, samples * receivers, 1, 1
        )
        error = error.transpose(0, 3, 4, 1, 2).reshape(
            seeds, states * actions, samples * receivers, 1, 1
        )
    w = weights - weights.mean(1, keepdims=True)
    e = error - error.mean(1, keepdims=True)
    covariance = (w * e).mean(1)
    scale = np.sqrt((w * w).mean(1) * (e * e).mean(1))
    valid = scale > 1e-20
    correlation = np.divide(
        covariance, scale, out=np.full_like(scale, np.nan), where=valid
    )
    valid_count = valid.sum((1, 2, 3))

    def average(x):
        return np.divide(
            np.nansum(x, axis=(1, 2, 3)),
            valid_count,
            out=np.full(len(weights), np.nan),
            where=valid_count > 0,
        )

    return dict(
        abs_corr=average(np.abs(correlation)),
        signed_corr=average(correlation),
        covariance=covariance.mean((1, 2, 3)),
        valid_fraction=valid.mean((1, 2, 3)),
    )


def measure_run(h, method, delay, tau, seed, mode, updates):
    """Measure final Q bias and pre-update filtering errors in the final 10%."""
    derived = derive(h, method, delay, tau)
    for field in ("operator", "operator_accumulated", "operator_readout"):
        h["diagnostic_" + field] = derived[field]
    tail = h["diagnostic_step"] > updates - max(1, updates // 10)
    row = dict(
        noise=NOISE_LABELS[mode],
        method=method,
        D=delay,
        tau=tau,
        seed=seed,
        updates=updates,
        diagnostic_samples=int(tail.sum()),
    )
    row["max_q_bias"] = float(h["q_final"][0].mean(0).max(-1).mean())
    for key in ("operator", "q", "q_ensemble", "value"):
        x = derived[key][:, tail]
        row[key + "_bias"] = float(x.mean())
        row[key + "_rmse"] = float(np.sqrt((x * x).mean()))
        row[key + "_over"] = float(np.maximum(x, 0).mean())
        row[key + "_under"] = float(np.maximum(-x, 0).mean())
    for component in ("accumulated", "readout"):
        row["operator_" + component] = float(
            derived["operator_" + component][:, tail].mean()
        )
    for proxy in PROXIES:
        for cross_section in (False, True):
            stats = coupling(
                derived["weights"][:, tail],
                derived["proxies"][proxy][:, tail],
                cross_section,
            )
            suffix = "_cross_section" if cross_section else ""
            for key, value in stats.items():
                row[proxy + suffix + "_" + key] = float(value[0])
    return row


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_job(job):
    """Reuse one seed's noise stream across all method/delay/expectile cases."""
    config, mode, seed, output = job
    directory = Path(output) / "raw" / mode / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    cases = [
        (m, d, tau) for m in METHODS for d in config["delays"] for tau in config["taus"]
    ]
    noise = actions = None
    rows = []
    for method, delay, tau in cases:
        path = directory / f"{method}_d{delay}_tau{tau:g}.npz"
        metadata_path = path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            stored = json.loads(metadata_path.read_text())
            if digest(path) != stored["data_sha256"]:
                raise ValueError(f"Raw data checksum mismatch: {path.name}")
            rows.append(stored["metrics"])
            continue
        if noise is None:
            noise = generate_noise(seed, mode, config["updates"])
            actions = np.random.default_rng(2000003 + seed).integers(
                4, size=config["updates"]
            )
        steps = diagnostic_schedule(config["updates"])
        result = run_seed(METHODS.index(method), noise, actions, delay, tau, steps)
        h = pack([result], steps, config["updates"], delay)
        if not np.isfinite(h["q_final"]).all():
            raise ValueError("Non-finite learned Q")
        assert h["refresh_count"][0] == (config["updates"] + delay - 1) // delay
        metrics = measure_run(h, method, delay, tau, seed, mode, config["updates"])
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **h)
        temporary.replace(path)
        metadata_path.write_text(
            json.dumps(dict(data_sha256=digest(path), metrics=metrics), indent=2) + "\n"
        )
        rows.append(metrics)
        print(
            json.dumps(
                dict(
                    noise=NOISE_LABELS[mode],
                    seed=seed,
                    method=LABELS[method],
                    D=delay,
                    tau=tau,
                    max_q_bias=metrics["max_q_bias"],
                )
            ),
            flush=True,
        )
    return rows


def summarize(rows):
    metadata = ("noise", "method", "D", "tau", "seed", "updates", "diagnostic_samples")
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in metadata[:4])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        group.sort(key=lambda row: row["seed"])
        row = dict(zip(metadata[:4], key))
        row.update(
            seeds=len(group),
            updates=group[0]["updates"],
            diagnostic_samples=group[0]["diagnostic_samples"],
        )
        for metric in group[0]:
            if metric in metadata:
                continue
            x = np.asarray([r[metric] for r in group], float)
            x = x[np.isfinite(x)]
            row[metric + "_mean"] = float(x.mean()) if len(x) else np.nan
            row[metric + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        result.append(row)
    return result


def read_summary(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["method"] not in METHODS or row["noise"] not in NOISE_LABELS.values():
            raise ValueError("Summary must contain only the three four-state methods")
        for key in row:
            if key not in ("noise", "method"):
                row[key] = float(row[key])
    return rows


PLOT_METRICS = (
    ("max_q_bias", "Q-bias", "max_q_bias"),
    ("operator_bias", "Bias against noise-free\nfiltering target", "filter_bias"),
    ("readout_abs_corr", "Coupling", "coupling_readout_temporal_abs"),
)


def render_figures(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    filenames = []
    settings = sorted({(r["noise"], r["tau"]) for r in rows})
    for metric, ylabel, name in PLOT_METRICS:
        finite = [r for r in rows if np.isfinite(r[metric + "_mean"])]
        bounds = [
            (
                r[metric + "_mean"] - np.nan_to_num(r[metric + "_sd"], nan=0),
                r[metric + "_mean"] + np.nan_to_num(r[metric + "_sd"], nan=0),
            )
            for r in finite
        ]
        low = min([0] + [b[0] for b in bounds])
        high = max([0] + [b[1] for b in bounds])
        margin = 0.07 * max(high - low, 1)
        for noise, tau in settings:
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            points = [r for r in rows if r["noise"] == noise and r["tau"] == tau]
            for method in METHODS:
                selected = sorted(
                    [r for r in points if r["method"] == method], key=lambda r: r["D"]
                )
                selected = [r for r in selected if np.isfinite(r[metric + "_mean"])]
                if not selected:
                    ax.plot([], [], color=COLORS[method], label=LABELS[method])
                    continue
                ax.errorbar(
                    [r["D"] for r in selected],
                    [r[metric + "_mean"] for r in selected],
                    yerr=[np.nan_to_num(r[metric + "_sd"], nan=0) for r in selected],
                    color=COLORS[method],
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    capsize=3,
                    label=LABELS[method],
                )
            ax.axhline(0, color="black", linewidth=0.65, zorder=0)
            ax.set_xscale("log")
            delays = sorted({int(r["D"]) for r in points})
            ax.set_xticks(delays)
            ax.set_xticklabels(delays)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("Delay $D$", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(
                (0, 1) if metric == "readout_abs_corr" else (low - margin, high + margin)
            )
            ax.tick_params(labelsize=11)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=9, framealpha=0.9)
            fig.tight_layout(pad=0.7)
            tag = "iid" if noise == "IID" else "nn_like_field"
            filename = f"s4_a4_{name}_{tag}_tau{str(float(tau)).replace('.', 'p')}.png"
            fig.savefig(directory / filename, dpi=220)
            plt.close(fig)
            filenames.append(filename)
    return filenames


def latex_value(mean, sd, signed, digits):
    if not np.isfinite(mean):
        return "N/A"
    value = format(mean, f"{'+' if signed else ''}.{digits}f")
    if not np.isfinite(sd):
        return f"${value}$"
    return f"${value}{{\\pm}}{sd:.{digits}f}$"


def render_tables(rows, output):
    directory = output / "tables"
    directory.mkdir(exist_ok=True)
    delays = sorted({int(r["D"]) for r in rows})
    columns = len(delays) + 1
    settings = sorted({(r["noise"], r["tau"]) for r in rows})
    descriptions = {
        "max_q_bias": "Final clean ensemble-mean max-Q bias, relative to zero.",
        "operator_bias": "Signed filtering-output bias relative to zero.",
        "readout_abs_corr": "Mean absolute temporal readout-error correlation.",
    }
    for metric, _, name in PLOT_METRICS:
        coupling_metric = metric == "readout_abs_corr"
        lines = [
            r"% Requires \usepackage{booktabs}",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{"
            + descriptions[metric]
            + r" Mean $\pm$ sample SD across seeds; $S=A=4$, $N=2$, $\gamma=0.999$.}",
            r"\begin{tabular}{l" + "c" * len(delays) + "}",
            r"\toprule",
            "Method & " + " & ".join(f"$D={d}$" for d in delays) + r" \\",
        ]
        for noise, tau in settings:
            lines += [
                r"\midrule",
                rf"\multicolumn{{{columns}}}{{l}}{{{noise}, $\tau={tau:g}$}} \\",
            ]
            for method in METHODS:
                values = []
                for delay in delays:
                    row = next(
                        r
                        for r in rows
                        if r["noise"] == noise
                        and r["tau"] == tau
                        and r["D"] == delay
                        and r["method"] == method
                    )
                    values.append(
                        latex_value(
                            row[metric + "_mean"],
                            row[metric + "_sd"],
                            not coupling_metric,
                            3 if coupling_metric else 2,
                        )
                    )
                lines.append(LABELS[method] + " & " + " & ".join(values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        (directory / f"s4_a4_{name}.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results_s4"))
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 220)))
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.99])
    parser.add_argument("--noise", choices=NOISES, nargs="+", default=list(NOISES))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Render a saved summary without running simulations",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1 or any(d < 1 for d in args.delays):
        parser.error("Steps, workers and delays must be positive")
    if any(not 0 < tau < 1 for tau in args.taus):
        parser.error("Expectile levels must lie strictly between zero and one")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    for key in ("seeds", "delays", "taus", "noise"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f"Duplicate entries in --{key}")
    config = dict(
        states=4,
        actions=4,
        ensemble_size=2,
        methods=list(METHODS),
        updates=args.steps,
        seeds=sorted(args.seeds),
        delays=sorted(args.delays),
        taus=sorted(args.taus),
        noise=list(args.noise),
        gamma=0.999,
        alpha_q=0.1,
        alpha_v=0.1,
        target_q_ema=0.005,
        sigma=1.0,
        field_rho=0.99,
        features=64,
        field_lengthscale=0.35,
        snapshot_timing="refresh_before_first_update_then_reuse_D_minus_1",
        source_sha256=digest(__file__),
    )
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    config=config,
                    conditions=3 * len(args.noise) * len(args.delays) * len(args.taus),
                    trajectories=3
                    * len(args.noise)
                    * len(args.delays)
                    * len(args.taus)
                    * len(args.seeds),
                ),
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.summary_csv:
        rows = read_summary(args.summary_csv)
        if (args.output / "config.json").exists():
            parser.error("Use a separate output directory when rendering a summary")
        metadata = dict(
            mode="render_saved_summary", input_sha256=digest(args.summary_csv)
        )
    else:
        path = args.output / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            parser.error("Output directory has different settings or source code")
        path.write_text(json.dumps(config, indent=2) + "\n")
        tiny = np.zeros((2, 2, 4, 4))
        run_seed(0, tiny, np.zeros(2, dtype=np.int64), 1, 0.9, np.array([1, 2]))
        jobs = [
            (config, mode, seed, str(args.output))
            for mode in args.noise
            for seed in sorted(args.seeds)
        ]
        if args.workers == 1:
            groups = [run_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                groups = list(pool.map(run_job, jobs))
        per_seed = [row for group in groups for row in group]
        write_csv(args.output / "per_seed_metrics.csv", per_seed)
        rows = summarize(per_seed)
        metadata = dict(
            mode="simulation", conditions=len(rows), trajectories=len(per_seed)
        )
    write_csv(args.output / "summary.csv", rows)
    figures = [] if args.skip_plots else render_figures(rows, args.output)
    render_tables(rows, args.output)
    metadata.update(
        seconds=time.monotonic() - started,
        figures=figures,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
    )
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(status="complete", **metadata)), flush=True)


if __name__ == "__main__":
    main()
