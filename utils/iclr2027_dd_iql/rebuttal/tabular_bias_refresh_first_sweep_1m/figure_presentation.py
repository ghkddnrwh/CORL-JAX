"""Presentation of cached bias diagnostics; no learning updates or new estimates."""

import csv
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import COLORS, CONFIG, IQL, LABELS
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

BASELINES = ("max_min", "double_q")
PLOT_METHODS = BASELINES + IQL
INDIVIDUAL_METHODS = IQL
NOISES = ("IID", "NN-like field")


def read_rows(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


@lru_cache(maxsize=None)
def clipped_rows(root, size, checkpoint=False):
    rows = [
        r
        for r in read_rows(Path(root) / "tables/min_q_baseline_reference.csv")
        if r["states"] == size
    ]
    assert len(rows) == 2
    if checkpoint:
        for row in rows:
            with np.load(row["source_npz"]) as stored:
                q = stored["checkpoint_q_100k"]
            bias = q.mean(1).max(-1).mean(-1)
            row["max_q_bias_mean"] = float(bias.mean())
            row["max_q_bias_sd"] = float(bias.std(ddof=1))
    return rows


def draw_methods(ax, rows, metric, noise, tau, methods, root, size, checkpoint=False):
    handles, plotted = [], []
    for method in methods:
        label, color = LABELS[method], COLORS[method]
        if method in BASELINES:
            if metric != "max_q_bias":
                handles.append(
                    Line2D([], [], color=color, linestyle="--", label=f"{label} (N/A)")
                )
                plotted.append(dict(method=method, label=label, applicable=False))
                continue
            points = (
                clipped_rows(str(root), size, checkpoint)
                if method == "max_min"
                else rows
            )
            row = next(
                r for r in points if r["method"] == method and r["noise"] == noise
            )
            mean, sd = row[metric + "_mean"], row[metric + "_sd"]
            handle = ax.axhline(
                mean, color=color, linestyle="--", linewidth=1.8, label=label
            )
            ax.axhspan(mean - sd, mean + sd, color=color, alpha=0.10)
            handles.append(handle)
            plotted.append(
                dict(
                    method=method, label=label, applicable=True, mean=mean, sd=sd, D=None
                )
            )
            continue
        points = sorted(
            [
                r
                for r in rows
                if r["method"] == method and r["noise"] == noise and r["tau"] == tau
            ],
            key=lambda r: r["D"],
        )
        assert len(points) == 4
        means = [r[metric + "_mean"] for r in points]
        sds = [r[metric + "_sd"] for r in points]
        assert np.isfinite(means).all() and np.isfinite(sds).all()
        handle = ax.errorbar(
            [r["D"] for r in points],
            means,
            yerr=sds,
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=label,
        )
        handles.append(handle)
        plotted.append(
            dict(
                method=method,
                label=label,
                applicable=True,
                mean=means,
                sd=sds,
                D=[r["D"] for r in points],
            )
        )
    ax.axhline(0, color="black", linewidth=0.65, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(CONFIG["delays"])
    ax.set_xticklabels(CONFIG["delays"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Delay $D$")
    ax.grid(alpha=0.15)
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)
    return plotted


def panel_plot(
    rows, metric, ylabel, filename, directory, scope, size, root, checkpoint=False
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, tau in enumerate(CONFIG["tau_values"]):
        for j, noise in enumerate(NOISES):
            ax = axes[i, j]
            draw_methods(
                ax, rows, metric, noise, tau, PLOT_METHODS, root, size, checkpoint
            )
            ax.set(ylabel=ylabel, title=f"{noise}; tau={tau}")
    fig.suptitle(
        f"{size} states / {size} actions; current-first filters; {scope}\n"
        "20 seeds; mean ± sample SD; exact reference = 0"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(Path(directory) / filename, dpi=180)
    plt.close(fig)


def coupling_scatter(summaries, directory, size, cross_section=False):
    suffix = "_cross_section" if cross_section else ""
    proxy_names = ("oracle", "td_update", "temporal_observed", "loo_observed")
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for col, (noise, tau) in enumerate(
        (n, t) for n in NOISES for t in CONFIG["tau_values"]
    ):
        points = [r for r in summaries if r["noise"] == noise and r["tau"] == tau]
        for row, proxy in enumerate(proxy_names):
            ax = axes[row, col]
            for point in points:
                x = point[proxy + suffix + "_abs_corr_mean"]
                if np.isfinite(x):
                    ax.scatter(
                        x,
                        point["operator_rmse_mean"],
                        color=COLORS[point["method"]],
                        marker={1: "o", 10: "s", 100: "^", 1000: "D"}[point["D"]],
                        s=35,
                    )
            ax.set(
                xlabel=f"Mean |corr(weight, {proxy})|",
                ylabel="Filtering RMSE",
                yscale="log",
                title=f"{noise}; tau={tau}",
            )
            ax.grid(alpha=0.15)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.label.set_size(9)
    handles = [
        Line2D([], [], color=COLORS[m], marker="o", linestyle="", label=LABELS[m])
        for m in IQL
    ]
    handles += [
        Line2D([], [], color=COLORS[m], linestyle="--", label=f"{LABELS[m]} (N/A)")
        for m in BASELINES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10)
    mode = "Within-checkpoint" if cross_section else "Temporal"
    fig.suptitle(
        f"{mode} coupling vs oracle filtering RMSE; S=A={size}; last 100k\n"
        "D: circle 1, square 10, triangle 100, diamond 1000. "
        "Undefined correlations omitted.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    name = (
        "coupling_cross_section_vs_filtering_accuracy.png"
        if cross_section
        else "coupling_vs_filtering_accuracy.png"
    )
    fig.savefig(Path(directory) / name, dpi=160)
    plt.close(fig)


def individual_plots(final, diagnostic, root):
    directory = Path(root) / "figures_s4_individual"
    directory.mkdir(exist_ok=True)
    specs = (
        ("max_q_bias", "Q-bias", final, "max_q_bias"),
        (
            "filter_bias",
            "Bias against noise-free\nfiltering target",
            diagnostic,
            "operator_bias",
        ),
        (
            "coupling_readout_temporal_abs",
            "Coupling",
            diagnostic,
            "readout_abs_corr",
        ),
    )
    records = []
    for name, ylabel, rows, metric in specs:
        bounds = [
            (
                r[metric + "_mean"] - r[metric + "_sd"],
                r[metric + "_mean"] + r[metric + "_sd"],
            )
            for r in rows
            if r["method"] in INDIVIDUAL_METHODS
        ]
        low, high = min(0, min(x[0] for x in bounds)), max(x[1] for x in bounds)
        margin = 0.07 * (high - low)
        limits = (
            (0, 1) if metric == "readout_abs_corr" else (low - margin, high + margin)
        )
        for noise in NOISES:
            noise_tag = "iid" if noise == "IID" else "nn_like_field"
            for tau in CONFIG["tau_values"]:
                fig, ax = plt.subplots(figsize=(6.2, 4.4))
                plotted = draw_methods(
                    ax, rows, metric, noise, tau, INDIVIDUAL_METHODS, root, 4
                )
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_xlabel("Delay $D$", fontsize=12)
                ax.set_ylim(*limits)
                ax.tick_params(labelsize=11)
                assert not ax.get_title() and fig._suptitle is None
                assert len(fig.axes) == 1
                labels = [x.get_text() for x in ax.get_legend().get_texts()]
                assert labels == [LABELS[m] for m in INDIVIDUAL_METHODS]
                assert all(p["method"] in INDIVIDUAL_METHODS for p in plotted)
                fig.tight_layout(pad=0.7)
                tau_tag = str(tau).replace(".", "p")
                filename = f"s4_a4_{name}_{noise_tag}_tau{tau_tag}_1m.png"
                fig.savefig(directory / filename, dpi=220)
                records.append(
                    dict(
                        file=str((directory / filename).relative_to(root)),
                        metric=metric,
                        noise=noise,
                        tau=tau,
                        axes=1,
                        ylabel=ax.get_ylabel(),
                        title="",
                        figure_title=None,
                        legend=labels,
                        series=plotted,
                    )
                )
                plt.close(fig)
    assert len(records) == 12
    return records
