"""Read-only audit of mean expectile weights and Q/V geometry in local runs.

No network training or counterfactual optimizer updates. Frozen-output substitutions
are algebraic diagnostics only. Raw arrays are retained with their measured phase.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long"
OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/neural_weight_mechanism_audit"
ENV_NAMES = {
    "antmaze-large-navigate-singletask-v0": "AM-L",
    "cube-single-play-singletask-v0": "Cube-S",
}
METHODS = ["IQL rho=.005", "IQL rho=.0005", "DD-IQL"]
COLORS = dict(zip(METHODS, ["#2674b2", "#ca7327", "#31926d"]))
WINDOWS = {
    "early_5k_100k": (5000, 100000),
    "late_900k_1M": (900000, 1000000),
    "at_1M": (1000000, 1000000),
    "at_2M": (2000000, 2000000),
}
IDS = ["env", "gamma", "method", "rho", "seed", "primary"]
FILES = {}
CHECKS = {
    "weight_identity_max_error": 0.0,
    "probe_weight_max_error": 0.0,
    "probe_adv_max_relative_error": 0.0,
    "duplicate_records_removed": 0,
    "probe_record_count": 0,
    "raw_inputs_unchanged": False,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_file(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in FILES:
        FILES[key] = {"path": key, "sha256": digest(path), "size": path.stat().st_size}


def read_npz(path):
    record_file(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def method_name(c):
    if "ensemble_size" in c:
        return "DD-IQL"
    return f"IQL rho={c['tau']:.6f}".rstrip("0").replace("rho=0.", "rho=.")


def finite_mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size and np.isfinite(x).all() else np.nan


def dedup(frame, step):
    CHECKS["duplicate_records_removed"] += int(frame.duplicated(step).sum())
    return (
        frame.drop_duplicates(step, keep="last").sort_values(step).reset_index(drop=True)
    )


def load_probes(run_dir):
    segments = []
    for p in (run_dir / "diagnostic_probes").glob("*/metadata.npz"):
        meta = read_npz(p)
        desc = json.loads(str(meta["metadata_json"]))
        segments.append((int(desc["created_ns"]), p.parent, meta, desc))
    frames, previous_desc = [], None
    meta = desc = None
    for _, directory, meta, desc in sorted(segments):
        if previous_desc:
            assert previous_desc["probe_fingerprint"] == desc["probe_fingerprint"]
        # A resumed segment supersedes the stale continuation at/after its start.
        for i, frame in enumerate(frames):
            frames[i] = {
                k: v[frame["state_step"] < desc["start_state_step"]]
                for k, v in frame.items()
            }
        chunks = [read_npz(p) for p in sorted(directory.glob("probe_trace_*.npz"))]
        if not chunks:
            continue
        keys = [
            "state_step",
            "update_step",
            "source_q1",
            "source_q2",
            "source_q",
            "source_v",
            "source_adv",
            "weight",
            "gate",
        ]
        frame = {k: np.concatenate([z[k] for z in chunks]) for k in keys}
        assert np.all(np.diff(frame["state_step"]) > 0)
        frames.append(frame)
        previous_desc = desc
    if not frames:
        return None
    data = {k: np.concatenate([z[k] for z in frames]) for k in frames[0]}
    assert np.all(np.diff(data["state_step"]) > 0)
    assert np.array_equal(data["update_step"], data["state_step"] + 1)
    for k in ["source_q1", "source_q2", "source_q", "source_v", "source_adv", "weight"]:
        assert np.isfinite(data[k]).all(), (run_dir, k)
    q, v, a = [data[k].astype(float) for k in ["source_q", "source_v", "source_adv"]]
    assert np.array_equal(
        data["source_q"], np.minimum(data["source_q1"], data["source_q2"])
    )
    assert np.array_equal(data["gate"], (a >= 0).astype(np.int8))
    err = np.max(np.abs(data["weight"] - (0.01 + 0.98 * (a >= 0))))
    CHECKS["probe_weight_max_error"] = max(CHECKS["probe_weight_max_error"], float(err))
    np.testing.assert_allclose(
        data["weight"], 0.01 + 0.98 * (a >= 0), rtol=2e-6, atol=1e-8
    )
    relerr = np.max(np.abs(a - (q - v)) / np.maximum(1.0, np.abs(q) + np.abs(v)))
    assert relerr < 2e-7
    CHECKS["probe_adv_max_relative_error"] = max(
        CHECKS["probe_adv_max_relative_error"], float(relerr)
    )
    CHECKS["probe_record_count"] += len(q)
    return data, meta, desc


def probe_metrics(probes, identity):
    d, meta, desc = probes
    q, v, a, q1, q2 = [
        d[k].astype(float)
        for k in ["source_q", "source_v", "source_adv", "source_q1", "source_q2"]
    ]
    high = a >= 0

    def weight(g):
        return 0.01 + 0.98 * g.mean(axis=1)

    meanq = (q1 + q2) / 2
    w = weight(high)
    f = pd.DataFrame(
        {
            **identity,
            "state_step": d["state_step"],
            "step": d["update_step"],
            "weight": w,
            "high_fraction": high.mean(1),
            "q_mean": q.mean(1),
            "v_mean": v.mean(1),
            "mean_head_weight": weight(meanq >= v),
            "q1_weight": weight(q1 >= v),
            "q2_weight": weight(q2 >= v),
            "fraction_q_positive": (q > 0).mean(1),
            "fraction_q_over_1000": (q > 1000).mean(1),
            "q_v_corr": [np.corrcoef(x, y)[0, 1] for x, y in zip(q, v)],
            "a_mean": a.mean(1),
            "a_std": a.std(1),
            "gap_over_a_std": -a.mean(1) / np.maximum(a.std(1), 1e-30),
            "tie_fraction": (a == 0).mean(1),
            "mc_gap_mean": (q - meta["batch_mc_returns"].reshape(1, -1)).mean(1),
        }
    )
    f["min_to_mean_weight_increase"] = f.mean_head_weight - f.weight
    rows = []
    for window, (lo, hi) in WINDOWS.items():
        mask = (f.step >= lo) & (f.step <= hi)
        if not mask.any():
            continue
        selected = high[mask]
        occupancy = selected.mean(0)
        asel = a[mask]
        ids = {**identity, "window": window, "n_snapshots": int(mask.sum())}
        row = {
            **ids,
            **{
                k: finite_mean(f.loc[mask, k])
                for k in [
                    "weight",
                    "high_fraction",
                    "mean_head_weight",
                    "q1_weight",
                    "q2_weight",
                    "min_to_mean_weight_increase",
                    "fraction_q_positive",
                    "fraction_q_over_1000",
                    "q_v_corr",
                    "gap_over_a_std",
                    "tie_fraction",
                ]
            },
            "ever_high_fraction": float((occupancy > 0).mean()),
            "always_high_fraction": float((occupancy == 1).mean()),
        }
        visits = selected.sum(0)
        row["top10pct_samples_share_high_events"] = (
            float(
                np.sort(visits)[-int(np.ceil(len(visits) * 0.1)) :].sum() / visits.sum()
            )
            if visits.sum()
            else np.nan
        )
        # Changes are between sampled checkpoints, not observed optimizer steps.
        ii = np.where(mask)[0]
        ii = ii[(ii > 0) & (d["update_step"][np.maximum(ii - 1, 0)] >= lo)]
        oldh, newh = high[ii - 1], high[ii]
        dq, dv = q[ii] - q[ii - 1], v[ii] - v[ii - 1]
        hl = oldh & ~newh
        row["high_to_low_count"] = int(hl.sum())
        row["high_to_low_q_increases_fraction"] = finite_mean((dq[hl] > 0).astype(float))
        row["high_to_low_v_faster_fraction"] = finite_mean(
            (dv[hl] > dq[hl]).astype(float)
        )
        row["high_to_low_v_only_crosses_fraction"] = finite_mean(
            (q[ii - 1][hl] < v[ii][hl]).astype(float)
        )
        row["flip_fraction"] = finite_mean((oldh != newh).astype(float))
        # Direct positive/negative weighted-residual balance on the fixed probe.
        pos = np.maximum(asel, 0).mean(1)
        neg = np.maximum(-asel, 0).mean(1)
        row["foc_imbalance_normalized"] = finite_mean(
            (0.99 * pos - 0.01 * neg) / np.maximum(0.99 * pos + 0.01 * neg, 1e-30)
        )
        rows.append(row)
    return f, rows


def load_runs():
    all_frames, runs, probe_frames, probe_rows, probe_raw, phases, eval_frames = (
        [],
        [],
        [],
        [],
        {},
        [],
        [],
    )
    for cfg_path in sorted(INPUT.rglob("config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg["env"] not in ENV_NAMES or cfg["iql_tau"] != 0.99:
            continue
        if cfg["tau"] not in [0.005, 0.0005, 0.00005, 0.000005]:
            continue
        record_file(cfg_path)
        method = method_name(cfg)
        identity = dict(
            env=ENV_NAMES[cfg["env"]],
            gamma=cfg["discount"],
            method=method,
            rho=cfg["tau"],
            seed=cfg["seed"],
            primary=method in METHODS,
        )
        assert cfg["normalize_reward"] is False
        assert (
            cfg["diagnostic_batch_size"] == 8192
            and cfg["diagnostic_seed_offset"] == 10000
        )
        if method == "DD-IQL":
            assert cfg["ensemble_size"] == 20 and cfg["delayed_update_period"] == 100
            assert cfg["delayed_expectile_self_index"] is False
        directory = cfg_path.parent
        raw = read_npz(directory / "bias_diagnostics.npz")
        f = pd.DataFrame({k.removeprefix("diag/"): v for k, v in raw.items()}).rename(
            columns={"timestep": "step"}
        )
        f = dedup(f, "step")
        assert np.isfinite(f[["filter_weight_mean", "q_mean", "v_mean"]]).all().all()
        assert f.step.iloc[-1] >= 1_000_000
        assert f.filter_weight_mean.between(0.01 - 1e-6, 0.99 + 1e-6).all()
        f["high_fraction"] = (f.filter_weight_mean - 0.01) / 0.98
        f["filter_ess_fraction"] = f.filter_weight_mean**2 / (
            0.99**2 * f.high_fraction + 0.01**2 * (1 - f.high_fraction)
        )
        f["receiver_gap"] = f.v_mean - f.q_mean
        f["source_gap"] = (
            (f.delayed_v_mean - f.delayed_q_mean)
            if method == "DD-IQL"
            else f.receiver_gap
        )
        f["mc_selection_shift"] = f.filter_receiver_mc_weighted_error - f.q_minus_mc_mean
        f["td_selection_shift"] = (
            f.filter_receiver_td_weighted_error - f.td_residual_mean
        )
        if method != "DD-IQL":
            # Exact binary-gate moment identities for the single IQL filter.
            # DD reports averages of per-head covariances, which cannot be
            # combined with pooled moments this way without per-head arrays.
            mu = f.adv_mean if "adv_mean" in f else -f.receiver_gap
            positive = f.filter_source_adv_cov / 0.98 + f.high_fraction * mu
            negative = positive - mu
            assert (positive >= -1e-6).all() and (negative >= -1e-6).all()
            f["positive_adv_conditional_mean"] = positive / f.high_fraction
            f["negative_adv_conditional_abs_mean"] = negative / (1 - f.high_fraction)
            f["positive_negative_adv_magnitude_ratio"] = (
                f.positive_adv_conditional_mean / f.negative_adv_conditional_abs_mean
            )
            f["foc_imbalance_normalized"] = (0.99 * positive - 0.01 * negative) / (
                0.99 * positive + 0.01 * negative
            )
        if "source_adv_nonnegative_fraction" in f:
            err = np.max(
                np.abs(
                    f.filter_weight_mean
                    - (0.01 + 0.98 * f.source_adv_nonnegative_fraction)
                )
            )
            assert err < 1e-6
            CHECKS["weight_identity_max_error"] = max(
                CHECKS["weight_identity_max_error"], float(err)
            )
            assert (f.q_finite_fraction == 1).all() and (f.v_finite_fraction == 1).all()
            f["gap_over_adv_std"] = -f.adv_mean / f.adv_std
            f["q_tail_over_std"] = (f.q_p99 - f.q_p50) / f.q_std
            f["q_median_over_mean"] = np.where(
                (f.q_p50 > 1000) & (f.q_mean > 1000), f.q_p50 / f.q_mean, np.nan
            )
            f["q_p99_over_median"] = np.where(f.q_p50 > 1000, f.q_p99 / f.q_p50, np.nan)
            f["q_p99_over_abs_p01"] = np.where(
                (f.q_p01 > 1000), f.q_p99 / f.q_p01, np.nan
            )
        for k, v in identity.items():
            f[k] = v
        all_frames.append(f)
        probes = load_probes(directory)
        if probes:
            pf, pr = probe_metrics(probes, identity)
            probe_frames.append(pf)
            probe_rows.extend(pr)
            probe_raw[
                (identity["env"], identity["gamma"], method, identity["seed"])
            ] = probes
        availability = dict(
            **identity,
            source_dir=str(directory.relative_to(ROOT)),
            common_rows=len(f),
            first_step=int(f.step.iloc[0]),
            last_step=int(f.step.iloc[-1]),
            diagnostic_freq=cfg["diagnostic_freq"],
            has_quantiles="q_p50" in f,
            has_probe=probes is not None,
            max_timesteps=cfg["max_timesteps"],
            probe_fingerprint=probes[2]["probe_fingerprint"] if probes else "",
            saved_keys=len(raw),
        )
        runs.append(availability)
        if method == "DD-IQL":
            dr = read_npz(directory / "delayed_diagnostics.npz")
            df = dedup(
                pd.DataFrame({k.removeprefix("diag/"): v for k, v in dr.items()}).rename(
                    columns={"timestep": "step"}
                ),
                "step",
            )
            assert np.array_equal(df.delay_age, df.step % 100)
            shared = f[["step", "filter_weight_mean"]].merge(
                df[["step", "filter_weight_mean"]], on="step"
            )
            np.testing.assert_allclose(
                shared.filter_weight_mean_x, shared.filter_weight_mean_y, atol=0, rtol=0
            )
            for window, (lo, hi) in WINDOWS.items():
                sub = df[df.step.between(lo, hi)]
                if sub.empty:
                    continue
                for phase in ["all", 0, 25, 50, 75]:
                    values = sub if phase == "all" else sub[sub.delay_age == phase]
                    phases.append(
                        {
                            **identity,
                            "window": window,
                            "phase": str(phase),
                            "weight": finite_mean(values.filter_weight_mean),
                            "n": len(values),
                        }
                    )
        er = read_npz(directory / "eval_logs.npz")
        ef = pd.DataFrame({k: v for k, v in er.items() if v.ndim == 1}).rename(
            columns={"timestep": "step"}
        )
        for k, v in identity.items():
            ef[k] = v
        eval_frames.append(ef)
    data = pd.concat(all_frames, ignore_index=True)
    inventory = pd.DataFrame(runs)
    assert inventory.primary.sum() == 48 and len(inventory) == 64
    for _, g in inventory[inventory.primary].groupby(["env", "gamma", "method"]):
        assert set(g.seed) == {0, 1, 2, 3}
    # Confirm paired AM-L fixed examples really match across rho settings.
    count = 0
    for (env, gamma, method, seed), probes in probe_raw.items():
        if env == "AM-L" and method == METHODS[0]:
            other = probe_raw[(env, gamma, METHODS[1], seed)]
            assert probes[2]["probe_fingerprint"] == other[2]["probe_fingerprint"]
            count += 1
    CHECKS["paired_AM_L_probe_identities_verified"] = count
    return (
        data,
        inventory,
        pd.concat(probe_frames, ignore_index=True),
        pd.DataFrame(probe_rows),
        probe_raw,
        pd.DataFrame(phases),
        pd.concat(eval_frames, ignore_index=True),
    )


def summarize(data):
    metrics = [
        "filter_weight_mean",
        "high_fraction",
        "filter_ess_fraction",
        "q_mean",
        "v_mean",
        "source_gap",
        "receiver_gap",
        "q_minus_mc_mean",
        "v_minus_mc_mean",
        "q_minus_mc_rmse",
        "mc_selection_shift",
        "td_selection_shift",
        "filter_source_adv_corr",
        "filter_receiver_adv_corr",
        "filter_receiver_td_corr",
        "source_receiver_error_corr",
        "q_p01",
        "q_p50",
        "q_p99",
        "q_max",
        "q_std",
        "v_p50",
        "adv_mean",
        "adv_p50",
        "adv_p95",
        "adv_p99",
        "adv_std",
        "gap_over_adv_std",
        "q_tail_over_std",
        "q_median_over_mean",
        "q_p99_over_median",
        "twin_q_abs_gap",
        "twin_q_corr",
        "positive_adv_conditional_mean",
        "negative_adv_conditional_abs_mean",
        "positive_negative_adv_magnitude_ratio",
        "foc_imbalance_normalized",
    ]
    rows = []
    for ident, g in data.groupby(IDS):
        ids = dict(zip(IDS, ident))
        for window, (lo, hi) in WINDOWS.items():
            part = g[g.step.between(lo, hi)]
            if part.empty:
                continue
            rows.append(
                {
                    **ids,
                    "window": window,
                    "n_checkpoints": len(part),
                    **{k: finite_mean(part[k]) for k in metrics},
                }
            )
        # Exact Table-2-like selection of 20 logged checkpoints, separately labeled.
        part = g[g.step.between(5000, 100000)].sort_values("step")
        indexes = np.floor((np.arange(20) + 0.5) * len(part) / 20).astype(int)
        part = part.iloc[indexes]
        rows.append(
            {
                **ids,
                "window": "early_20_checkpoints",
                "n_checkpoints": len(part),
                **{k: finite_mean(part[k]) for k in metrics},
            }
        )
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby([k for k in IDS if k != "seed"] + ["window"])[
        metrics
    ].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return per_seed, summary.reset_index()


def paired_comparisons(per_seed):
    rows = []
    for (env, gamma, window), g in per_seed[per_seed.primary].groupby(
        ["env", "gamma", "window"]
    ):
        pivot = g.set_index(["method", "seed"])
        for left, right in [
            (METHODS[0], METHODS[1]),
            (METHODS[0], METHODS[2]),
            (METHODS[1], METHODS[2]),
        ]:
            for seed in range(4):
                if (left, seed) not in pivot.index or (right, seed) not in pivot.index:
                    continue
                a, b = pivot.loc[(left, seed)], pivot.loc[(right, seed)]
                rows.append(
                    dict(
                        env=env,
                        gamma=gamma,
                        window=window,
                        seed=seed,
                        left=left,
                        right=right,
                        right_minus_left_weight=b.filter_weight_mean
                        - a.filter_weight_mean,
                        left_over_right_abs_v=abs(a.v_mean) / max(abs(b.v_mean), 1e-30),
                    )
                )
    return pd.DataFrame(rows)


def onset_summary(data):
    rows = []
    for ident, g in data[data.primary & (data.method != "DD-IQL")].groupby(IDS):
        g = g[g.step <= 1e6].sort_values("step")
        row = dict(zip(IDS, ident))
        for key in ["q_mean", "q_p01", "q_p50", "q_p99"]:
            values = g[key].to_numpy()
            hits = np.where(
                (values[:-2] > 1000) & (values[1:-1] > 1000) & (values[2:] > 1000)
            )[0]
            row[key + "_over1000_onset"] = (
                float(g.step.iloc[hits[0]]) if len(hits) else np.nan
            )
        before = g[g.step < row["q_mean_over1000_onset"]]
        if len(before):
            row["before_Q1000_weight"] = finite_mean(before.filter_weight_mean)
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory(ax, frame, metric, method, maximum=1e6, minimum=2500, label=None):
    s = frame[(frame.method == method) & frame.step.between(minimum, maximum)]
    if s.empty or metric not in s or s[metric].isna().all():
        return
    p = s.pivot(index="step", columns="seed", values=metric).dropna()
    if p.empty:
        return
    assert len(p.columns) == 4
    col = COLORS.get(method, "#777777")
    for seed in p:
        ax.plot(p.index / 1e6, p[seed], color=col, lw=0.5, alpha=0.2)
    ax.plot(
        p.index / 1e6,
        p.mean(1),
        color=col,
        lw=1.8,
        label=label or method,
        ls="--" if method == "DD-IQL" else "-",
    )


def savefig(fig, pdf, name):
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plots(data, probe, raw, phases):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )
    pairs = [(e, g) for e in ["AM-L", "Cube-S"] for g in [0.99, 0.999]]
    with PdfPages(OUT / "weight_mechanism_report.pdf") as pdf:
        for metric, title, symlog, maximum in [
            ("filter_weight_mean", "Mean expectile filtering weight", False, 1e6),
            ("v_mean", "Mean V on the fixed diagnostic batch", True, 1e6),
            ("source_gap", "Mean source V minus source Q", True, 1e6),
            (
                "filter_weight_mean",
                "Early filtering weights (before large positive drift)",
                False,
                100000,
            ),
            (
                "filter_source_adv_corr",
                "Correlation: filter weight and its source advantage",
                False,
                1e6,
            ),
            (
                "mc_selection_shift",
                "Weighted MC residual minus unweighted MC residual (proxy)",
                True,
                1e6,
            ),
        ]:
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
            for ax, (env, gamma) in zip(axes.flat, pairs):
                sub = data[(data.env == env) & (data.gamma == gamma) & data.primary]
                for method in METHODS:
                    trajectory(ax, sub, metric, method, maximum)
                ax.set(
                    title=f"{env} | $\\gamma={gamma}$",
                    xlabel="Updates (millions)",
                    ylabel=metric,
                )
                ax.grid(alpha=0.2)
                ax.set_xlim(0, maximum / 1e6)
                if metric == "filter_weight_mean":
                    ax.set_ylim(0, 0.65)
                if symlog:
                    ax.set_yscale("symlog", linthresh=1 if metric != "v_mean" else 100)
            axes.flat[0].legend(fontsize=8, frameon=False)
            fig.suptitle(title + r" | $\tau=.99$, four seeds", fontsize=13)
            savefig(fig, pdf, f"{metric}_{int(maximum)}")
        # Full-batch quantiles: averaged per-seed quantiles, not pooled quantiles.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
        for ax, (env, gamma) in zip(axes.flat, pairs):
            sub = data[
                (data.env == env)
                & (data.gamma == gamma)
                & (data.method == METHODS[0])
                & (data.step <= 1e6)
            ]
            avg = sub.groupby("step")[["q_p01", "q_p50", "q_p99", "v_mean"]].mean()
            x = avg.index.to_numpy() / 1e6
            ax.fill_between(
                x,
                avg.q_p01.to_numpy(),
                avg.q_p99.to_numpy(),
                alpha=0.22,
                color=COLORS[METHODS[0]],
                label="Q 1st-99th percentile",
            )
            ax.plot(x, avg.q_p50, label="Q median", color=COLORS[METHODS[0]])
            ax.plot(x, avg.v_mean, label="Mean V", color="#ac3a4e", ls="--")
            ax.set(
                title=f"{env} | $\\gamma={gamma}$",
                xlabel="Updates (millions)",
                ylabel="Value",
                yscale="symlog",
            )
            ax.set_yscale("symlog", linthresh=100)
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            "IQL rho=.005: Q quantiles and V over training\n"
            "8192 dataset examples; mean of four seed-specific quantiles",
            fontsize=12,
        )
        savefig(fig, pdf, "full_batch_q_quantiles")
        # Frozen min-vs-mean substitution at the identical 128 examples / V.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
        for ax, (env, gamma) in zip(axes.flat, pairs):
            sub = probe[
                (probe.env == env)
                & (probe.gamma == gamma)
                & (probe.method == METHODS[0])
            ]
            for metric, col, label in [
                ("weight", "#2674b2", "Actual min-Q gate"),
                ("mean_head_weight", "#9b56a0", "Replace min by mean; V fixed"),
            ]:
                p = sub[sub.step.between(5000, 1e6)].pivot(
                    index="step", columns="seed", values=metric
                )
                ax.plot(p.index / 1e6, p.mean(1), color=col, label=label)
            dd = data[(data.env == env) & (data.gamma == gamma)]
            trajectory(
                ax, dd, "filter_weight_mean", "DD-IQL", label="DD-IQL (8192 examples)"
            )
            ax.set(
                title=f"{env} | $\\gamma={gamma}$",
                xlabel="Updates (millions)",
                ylabel="Mean weight",
                ylim=(0, 0.65),
            )
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            "Instantaneous min-Q effect | frozen outputs, no retraining\n"
            "IQL uses a 128-example pre-update subset; DD is post-update",
            fontsize=12,
        )
        savefig(fig, pdf, "frozen_min_vs_mean")
        # Plot every saved example for seed 0, with fixed exemplar selection.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
        for ax, (env, gamma) in zip(axes.flat, pairs):
            d, _, _ = raw[(env, gamma, METHODS[0], 0)]
            mask = (d["update_step"] >= 5000) & (d["update_step"] <= 1e6)
            a = d["source_adv"][mask].astype(float)
            z = a / np.maximum(a.std(1, keepdims=True), 1e-30)
            im = ax.imshow(
                z.T,
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
                vmin=-3,
                vmax=3,
                extent=[
                    d["update_step"][mask][0] / 1e6,
                    d["update_step"][mask][-1] / 1e6,
                    0,
                    128,
                ],
            )
            ax.set(
                title=f"{env} | $\\gamma={gamma}$ | seed 0",
                xlabel="Updates (millions)",
                ylabel="Fixed example ID",
            )
        fig.colorbar(im, ax=list(axes.flat), label="(Q - V) / SD(Q - V)", shrink=0.8)
        fig.suptitle(
            "Fixed-example advantages: sign controls weight\n"
            "Each row is one recorded state-action pair, "
            "not multiple actions at one state",
            fontsize=12,
        )
        savefig(fig, pdf, "fixed_example_advantages")
        # Marginal Q and paired A standardized, late checkpoint, all seeds.
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
        for col, (env, gamma) in enumerate(pairs):
            for seed in range(4):
                d, _, _ = raw[(env, gamma, METHODS[0], seed)]
                idx = np.where(d["update_step"] == 1_000_000)[0][0]
                q = d["source_q"][idx].astype(float)
                v = d["source_v"][idx].astype(float)
                a = q - v
                for row, val, label, c in [
                    (0, q / np.median(q), "Q / median(Q)", "#2674b2"),
                    (1, a / a.std(), "(Q - V) / SD(Q - V)", "#ac3a4e"),
                ]:
                    axes[row, col].plot(
                        np.sort(val), np.arange(1, 129) / 128, color=c, alpha=0.6, lw=1
                    )
                    axes[row, col].set(xlabel=label, ylabel="Empirical CDF")
                    axes[row, col].grid(alpha=0.2)
            axes[0, col].set_title(f"{env} | $\\gamma={gamma}$")
            axes[1, col].axvline(0, color="k", lw=0.7, ls="--")
        fig.suptitle(
            "IQL rho=.005 at 1M | four separate seed CDFs on 128 fixed examples",
            fontsize=13,
        )
        savefig(fig, pdf, "late_q_and_advantage_cdf")
        # Existing smaller-rho runs are supplementary only; no new training.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), layout="constrained")
        extra = ["IQL rho=.00005", "IQL rho=.000005"]
        for col, gamma in enumerate([0.99, 0.999]):
            sub = data[(data.env == "Cube-S") & (data.gamma == gamma)]
            for metric, ax in [
                ("filter_weight_mean", axes[0, col]),
                ("v_mean", axes[1, col]),
            ]:
                for i, method in enumerate(METHODS + extra):
                    p = sub[(sub.method == method) & (sub.step <= 1e6)].pivot(
                        index="step", columns="seed", values=metric
                    )
                    ax.plot(
                        p.index / 1e6,
                        p.mean(1),
                        label=method,
                        color=["#2674b2", "#ca7327", "#31926d", "#9b56a0", "#6c6c6c"][i],
                    )
                ax.set(
                    title=f"Cube-S | $\\gamma={gamma}$",
                    xlabel="Updates (millions)",
                    ylabel=metric,
                )
                ax.grid(alpha=0.2)
            axes[1, col].set_yscale("symlog", linthresh=100)
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(
            "Supplement: existing additional smaller-rho logs | four-seed means",
            fontsize=13,
        )
        savefig(fig, pdf, "supplement_smaller_rho")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    data, inventory, probe, probe_summary, raw, phase, evals = load_runs()
    per_seed, summary = summarize(data)
    paired = paired_comparisons(per_seed)
    onsets = onset_summary(data)
    outputs = {
        "inventory": inventory,
        "per_seed_summary": per_seed,
        "condition_summary": summary,
        "paired_comparisons": paired,
        "probe_per_seed_summary": probe_summary,
        "dd_delay_phase_summary": phase,
        "onset_summary": onsets,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    data.to_csv(OUT / "diagnostic_trajectories.csv.gz", index=False)
    probe.to_csv(OUT / "probe_trajectories.csv.gz", index=False)
    evals.to_csv(OUT / "evaluation_trajectories.csv.gz", index=False)
    # Preserve measurements' own times: full batch post-update t versus small
    # probe pre-update t-1, with upcoming update t retained as plotting coordinate.
    plots(data, probe, raw, phase)
    for info in FILES.values():
        assert digest(ROOT / info["path"]) == info["sha256"], info["path"]
    CHECKS["raw_inputs_unchanged"] = True
    CHECKS["input_file_count"] = len(FILES)
    CHECKS["primary_run_count"] = 48
    CHECKS["supplemental_run_count"] = 16
    CHECKS["all_primary_core_metrics_finite_through_1M"] = True
    record_file(Path(__file__))
    (OUT / "source_manifest.json").write_text(
        json.dumps(list(FILES.values()), indent=2) + "\n"
    )
    (OUT / "verification.json").write_text(json.dumps(CHECKS, indent=2) + "\n")
    for w in ["early_5k_100k", "late_900k_1M", "at_1M"]:
        print("\n", w)
        print(
            summary[(summary.primary) & (summary.window == w)][
                [
                    "env",
                    "gamma",
                    "method",
                    "filter_weight_mean_mean",
                    "filter_weight_mean_std",
                    "high_fraction_mean",
                    "q_mean_mean",
                    "v_mean_mean",
                ]
            ].to_string(index=False)
        )
    print(
        "\nPROBE",
        probe_summary[probe_summary.primary & (probe_summary.window == "late_900k_1M")][
            [
                "env",
                "gamma",
                "method",
                "seed",
                "weight",
                "mean_head_weight",
                "q_v_corr",
                "ever_high_fraction",
                "high_to_low_q_increases_fraction",
            ]
        ].to_string(index=False),
    )
    print("Saved", OUT)


if __name__ == "__main__":
    main()
