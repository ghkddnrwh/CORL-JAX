"""Summarize paired controls, exact gate geometry and IID mean-field mechanism."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/tabular_df_dd_weight_hypothesis"
PREVIOUS = (
    ROOT / "utils/iclr2027_dd_iql/rebuttal/tabular_s4_weight_distribution/analyze.py"
)
spec = importlib.util.spec_from_file_location("previous_audit", PREVIOUS)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
CASES = {"df_min": 1, "dd_iql": 100}
LABELS = {"df_min": "DF-IQL, D=1", "dd_iql": "DD-IQL, D=100"}
COLORS = {"df_min": "#cc79a7", "dd_iql": "#0072b2"}
VARIANTS = ("baseline", "flat_clean_inputs", "flat_whole_table")
STYLES = dict(zip(VARIANTS, ("-", "--", ":")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_arrays(h, method):
    clean, y, v = [
        h[k]
        for k in (
            "diagnostic_delayed_clean_q",
            "diagnostic_delayed_observed_q",
            "diagnostic_delayed_value",
        )
    ]
    if method == "df_min":
        return clean.min(1, keepdims=True), y.min(1, keepdims=True), v
    return clean[:, ::-1], y[:, ::-1], v[:, ::-1]


def point_metrics(h, method):
    c, y, v = source_arrays(h, method)
    q = h["diagnostic_target_q"]
    w = h["diagnostic_weights"]
    np.testing.assert_array_equal(w, np.where(y >= v[..., None], 0.99, 1 - 0.99))
    mu = c.mean(-1)
    flat = np.where(mu[..., None] + (y - c) >= v[..., None], 0.99, 0.01)
    # Counterfactuals operate on recorded frozen sources, with V fixed unless
    # explicitly replaced by the clean state mean. No retraining in these columns.
    out = dict(
        mean_weight=w.mean((1, 2, 3)),
        high_fraction=(w == 0.99).mean((1, 2, 3)),
        learned_target_q=q.mean((1, 2, 3)),
        source_clean_q=c.mean((1, 2, 3)),
        learned_v=h["diagnostic_value"].mean((1, 2)),
        source_gap=(v - mu).mean((1, 2)),
        source_clean_action_sd=c.std(-1).mean((1, 2)),
        source_clean_table_sd=c.std((2, 3)).mean(1),
        source_clean_upper_gap=(c.max(-1) - mu).mean((1, 2)),
        readout_action_sd=(y - c).std(-1).mean((1, 2)),
        snapshot_flat_clean_weight=flat.mean((1, 2, 3)),
        snapshot_zero_gap_weight=np.where(y >= mu[..., None], 0.99, 0.01).mean(
            (1, 2, 3)
        ),
        snapshot_flat_zero_gap_weight=np.where(y - c >= 0, 0.99, 0.01).mean((1, 2, 3)),
    )
    # An equal shift of each source state Q and its V cannot change any sign.
    np.testing.assert_array_equal(
        y >= v[..., None], y - mu[..., None] >= (v - mu)[..., None]
    )
    return out


@njit(cache=True)
def homogeneous_df(updates):
    """Approximate expected dynamics with identical clean Q at all 16 entries.

    This is a homogeneous mean-field approximation to DF, NOT an exact expectation
    of the stochastic process: E[f(g)] is replaced by f(E[g]). Fresh IID min-noise
    is analytically integrated. Uniform sampling gives Q/V rates .1/16 and .1/4.
    """
    q, target, v = 0.0, 0.0, 0.0
    curve = np.empty((updates // 1000 + 1, 5))
    curve[0] = np.array([q, target, v, v - target, 0.255])
    for t in range(updates):
        gap = v - target
        survival = 0.5 * math.erfc(gap / math.sqrt(2))
        phi = math.exp(-gap * gap / 2) / math.sqrt(2 * math.pi)
        positive = (
            2 * phi * survival
            - gap * survival * survival
            - 0.5 / math.sqrt(math.pi) * math.erfc(gap)
        )
        score = 0.01 * (-1 / math.sqrt(math.pi) - gap) + 0.98 * positive
        old_v = v
        v += 0.1 / 4 * score
        q += 0.1 / 16 * (0.999 * old_v - q)
        target = 0.995 * target + 0.005 * q
        if (t + 1) % 1000 == 0:
            gap = v - target
            weight = 0.01 + 0.98 * (0.5 * math.erfc(gap / math.sqrt(2))) ** 2
            curve[(t + 1) // 1000] = np.array([q, target, v, gap, weight])
    return curve


def save(fig, out, name, booklet):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / (name + "." + ext), dpi=200, bbox_inches="tight")
    booklet.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def line(ax, x, ys, label, color, linestyle="-", band=True):
    ys = np.asarray(ys)
    mean, sd = ys.mean(0), ys.std(0, ddof=1)
    ax.plot(x, mean, color=color, linestyle=linestyle, label=label, lw=1.6)
    if band:
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10)


def binned_line(ax, x, ys, label, color, linestyle="-", band=True):
    # Average each seed within 20k-update bins, then show across-seed sample SD.
    masks = [(x > left) & (x <= left + 0.02) for left in np.arange(0, 1, 0.02)]
    centers = np.array([x[mask].mean() for mask in masks])
    values = np.stack([np.asarray(ys)[:, mask].mean(1) for mask in masks], axis=1)
    line(ax, centers, values, label, color, linestyle, band)


def plot_case(noise, traces, raw, steps, out, booklet):
    slug = "iid" if noise.endswith("_iid") else "nn_like"
    keys = ("learned_target_q", "mean_weight", "source_clean_action_sd", "source_gap")
    ylabel = (
        "Mean clean target Q",
        "Mean filtering weight",
        "Within-state clean-Q action SD",
        "Source V minus mean clean Q",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for method in CASES:
        for ax, key in zip(axes.flat, keys):
            draw = binned_line if key == "mean_weight" else line
            draw(
                ax,
                steps / 1e6,
                traces[(method, "baseline")][key],
                LABELS[method],
                COLORS[method],
            )
    for ax, lab in zip(axes.flat, ylabel):
        ax.set(xlabel="Updates (millions)", ylabel=lab, xlim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    save(fig, out, slug + "_df_d1_vs_dd_d100", booklet)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, CASES):
        h = raw[method]
        c, y, v = h["source_clean"], h["source_readout"], h["source_v"]
        late = h["diagnostic_step"] > 900_000
        mu = c[late].mean(-1)
        audit.ecdf(ax, c[late] - mu[..., None], "Clean Q - state mean", "0.35", "--")
        audit.ecdf(ax, y[late] - mu[..., None], "Noisy source Q - state mean", "#0072b2")
        audit.ecdf(ax, v[late] - mu, "V - state mean clean Q", "#d55e00")
        ax.set(
            xlabel="Value relative to clean state mean",
            ylabel="Cumulative fraction",
            ylim=(0, 1),
        )
        ax.text(0.04, 0.95, LABELS[method], transform=ax.transAxes, va="top")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, out, slug + "_centered_q_and_v", booklet)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, method in enumerate(CASES):
        for variant in VARIANTS:
            label = {
                "baseline": "Original",
                "flat_clean_inputs": "Action-mean Q inputs",
                "flat_whole_table": "Whole-table-mean Q inputs",
            }[variant]
            for col, key in enumerate(("learned_target_q", "mean_weight")):
                draw = binned_line if col == 1 else line
                color = (
                    COLORS[method]
                    if variant == "baseline"
                    else {"flat_clean_inputs": "#e69f00", "flat_whole_table": "0.2"}[
                        variant
                    ]
                )
                draw(
                    axes[row, col],
                    steps / 1e6,
                    traces[(method, variant)][key],
                    label,
                    color,
                    STYLES[variant],
                    band=variant == "baseline",
                )
        axes[row, 0].set_ylabel(LABELS[method] + "\nMean clean target Q")
        axes[row, 1].set_ylabel(LABELS[method] + "\nMean filtering weight (20k bins)")
    for ax in axes.flat:
        ax.set(xlabel="Updates (millions)", xlim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    save(fig, out, slug + "_paired_flattening_controls", booklet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    (out / "figures").mkdir(exist_ok=True)
    audit.style()
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert len(manifest["baseline_bitwise_checks"]) == 40
    rows, frames, file_hashes, examples = [], [], {}, []
    traces_by_noise = {}
    max_balance_error = 0.0
    with PdfPages(out / "hypothesis_report.pdf") as booklet:
        for noise in audit.NOISES:
            traces, originals = {}, {}
            for method in CASES:
                for variant in VARIANTS:
                    points = []
                    for seed in range(200, 220):
                        path = out / "raw" / noise / method / variant / f"seed{seed}.npz"
                        file_hashes[str(path.relative_to(out))] = digest(path)
                        with np.load(path, allow_pickle=False) as data:
                            h = {key: data[key] for key in data.files}
                        assert all(np.isfinite(a).all() for a in h.values())
                        point = point_metrics(h, method)
                        points.append(point)
                        steps = h["diagnostic_step"]
                        late = steps > 900_000
                        assert late.sum() == 101
                        if variant != "baseline":
                            c = h["diagnostic_delayed_clean_q"]
                            np.testing.assert_array_equal(
                                c, np.broadcast_to(c[..., :1], c.shape)
                            )
                            if variant == "flat_whole_table":
                                np.testing.assert_array_equal(
                                    c, np.broadcast_to(c[:, :, :1, :1], c.shape)
                                )
                        mom = h["update_moments"][-100:].mean(0)
                        # Exact bookkeeping of ALL committed V updates since zero init.
                        accumulated = 0.1 * 1000 * h["update_moments"][:, 2:4].sum()
                        actual = h["value_final"].sum() / h["value_final"].shape[0]
                        error = abs(accumulated - actual)
                        max_balance_error = max(max_balance_error, error)
                        np.testing.assert_allclose(
                            accumulated, actual, rtol=0, atol=1e-6
                        )
                        rows.append(
                            dict(
                                noise=noise,
                                method=method,
                                delay=CASES[method],
                                variant=variant,
                                seed=seed,
                                **{k: a[late].mean() for k, a in point.items()},
                                final_max_q_bias=h["q_final"].mean(0).max(-1).mean(),
                                used_weight=mom[0],
                                receiver_error=mom[1],
                                weighted_error=mom[2],
                                weight_error_cov=mom[2] - mom[0] * mom[1],
                                weighted_clean_residual=mom[3],
                                mean_v_update=0.1 * (mom[2] + mom[3]),
                            )
                        )
                        if variant == "baseline":
                            originals.setdefault(method, []).append(h)
                    trace = {
                        key: np.stack([p[key] for p in points]) for key in points[0]
                    }
                    traces[(method, variant)] = trace
                    frame = pd.DataFrame(dict(step=steps))
                    for key, a in trace.items():
                        frame[key + "_mean"] = a.mean(0)
                        frame[key + "_sd"] = a.std(0, ddof=1)
                    frame["noise"], frame["method"], frame["variant"] = (
                        noise,
                        method,
                        variant,
                    )
                    frames.append(frame)
            # CDF expects a time axis first; concatenate head/state samples over seeds.
            combined = {}
            for method in CASES:
                hs = originals[method]
                sources = [source_arrays(h, method) for h in hs]
                combined[method] = {
                    key: np.concatenate([source[j] for source in sources], axis=1)
                    for j, key in enumerate(
                        ("source_clean", "source_readout", "source_v")
                    )
                }
                combined[method]["diagnostic_step"] = steps
                # Numeric source snapshots at fixed seed/state/head, pre-update 1M.
                h = hs[0]
                c, y, v = source_arrays(h, method)
                for a in range(4):
                    examples.append(
                        dict(
                            noise=noise,
                            method=method,
                            delay=CASES[method],
                            seed=200,
                            step=int(steps[-1]),
                            gate_step=int(steps[-1] - (steps[-1] - 1) % CASES[method]),
                            receiver=0,
                            state=0,
                            action=a,
                            clean_q=c[-1, 0, 0, a],
                            readout=y[-1, 0, 0, a],
                            source_v=v[-1, 0, 0],
                            weight=h["diagnostic_weights"][-1, 0, 0, a],
                        )
                    )
            plot_case(noise, traces, combined, steps, out, booklet)
            traces_by_noise[noise] = traces

        homogeneous = homogeneous_df(1_000_000)
        pd.DataFrame(
            dict(
                step=np.arange(0, 1_000_001, 1000),
                q=homogeneous[:, 0],
                target_q=homogeneous[:, 1],
                v=homogeneous[:, 2],
                gap=homogeneous[:, 3],
                mean_weight=homogeneous[:, 4],
            )
        ).to_csv(out / "iid_homogeneous_approximation.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        traces = traces_by_noise["nn_like_field_iid"]
        for ax, key, col, label in (
            (axes[0], "learned_target_q", 1, "Mean clean target Q"),
            (axes[1], "mean_weight", 4, "Mean filtering weight"),
        ):
            draw = binned_line if key == "mean_weight" else line
            draw(
                ax,
                steps / 1e6,
                traces[("df_min", "baseline")][key],
                "DF-IQL, D=1 (20 seeds)",
                COLORS["df_min"],
            )
            ax.plot(
                np.arange(0, 1_000_001, 1000) / 1e6,
                homogeneous[:, col],
                "k--",
                lw=1.5,
                label="Homogeneous approximation: no learned outliers",
            )
            ax.set(xlabel="Updates (millions)", ylabel=label, xlim=(0, 1))
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
        save(fig, out, "iid_homogeneous_mechanism", booklet)

    per_seed = pd.DataFrame(rows)
    assert len(per_seed) == 240
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    identity = ["noise", "method", "delay", "variant"]
    summary = per_seed.groupby(identity, sort=False).agg(
        {key: ["mean", "std"] for key in per_seed if key not in identity + ["seed"]}
    )
    summary.columns = [
        key + ("_sd" if stat == "std" else "_mean") for key, stat in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    differences = []
    for (noise, method), part in per_seed.groupby(["noise", "method"]):
        baseline = part[part.variant == "baseline"].set_index("seed")
        for variant in VARIANTS[1:]:
            alternate = part[part.variant == variant].set_index("seed")
            for key in ("mean_weight", "learned_target_q", "source_gap"):
                d = alternate[key] - baseline[key]
                differences.append(
                    dict(
                        noise=noise,
                        method=method,
                        variant=variant,
                        metric=key,
                        paired_delta_mean=d.mean(),
                        paired_delta_sd=d.std(ddof=1),
                        seeds=len(d),
                    )
                )
    pd.DataFrame(differences).to_csv(out / "paired_differences.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "time_summary.csv.gz", index=False)
    pd.DataFrame(examples).to_csv(out / "example_snapshots.csv", index=False)
    # Verify the min-normal positive-part identity against numerical integration.
    quad_errors = []

    def score(gap):
        survival = ndtr(-gap)
        positive = (
            2 * np.exp(-(gap**2) / 2) / np.sqrt(2 * np.pi) * survival
            - gap * survival**2
            - ndtr(-np.sqrt(2) * gap) / np.sqrt(np.pi)
        )
        return 0.01 * (-1 / np.sqrt(np.pi) - gap) + 0.98 * positive

    for gap in (-2.0, 0.0, 0.64, 1.0, 3.0):
        numerical = quad(
            lambda z: (0.99 if z >= gap else 0.01)
            * (z - gap)
            * 2
            * np.exp(-z * z / 2)
            / np.sqrt(2 * np.pi)
            * ndtr(-z),
            -12,
            12,
            points=[gap],
            epsabs=1e-11,
        )[0]
        quad_errors.append(abs(score(gap) - numerical))
    assert max(quad_errors) < 1e-10
    stationary_gap = brentq(score, 0, 3)
    verification = dict(
        trajectories=len(per_seed),
        seeds=20,
        baseline_bitwise_trajectories=80,
        original_cases=4,
        variants=VARIANTS,
        final_window_snapshots=101,
        all_gate_reconstructions_exact=True,
        flat_source_inputs_verified=True,
        all_arrays_finite=True,
        max_full_run_V_accounting_error=max_balance_error,
        min_gaussian_quadrature_max_error=max(quad_errors),
        homogeneous_approximation_final=dict(
            zip(("q", "target_q", "v", "gap", "mean_weight"), homogeneous[-1])
        ),
        homogeneous_stationary_gap=stationary_gap,
        original_run_code_sha256=manifest["code_sha256"],
        current_runner_sha256=digest(Path(__file__).with_name("run_control.py")),
        runner_post_execution_change="Black formatting and import order only",
        analysis_code_sha256=digest(Path(__file__)),
        prior_analysis_sha256=digest(PREVIOUS),
        raw_sha256=file_hashes,
    )
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        summary[
            [*identity, "mean_weight_mean", "learned_target_q_mean", "source_gap_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
