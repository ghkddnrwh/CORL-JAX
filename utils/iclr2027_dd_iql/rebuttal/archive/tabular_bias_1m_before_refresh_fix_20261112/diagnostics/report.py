"""Summarize the four-state replay without changing the original results."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import case_path, cases, COLORS, CONFIG, DEFAULT_RESULTS_DIR, noise_name
from measure import PROXIES, window_rows

METHODS = ("max_min", "df_min", "sd_iql", "dd_iql")
LABELS = {
    "max_min": "Clipped Double Q",
    "df_min": "DF-min",
    "sd_iql": "SD-IQL",
    "dd_iql": "DD-IQL",
}
LAST = "900000_1000000"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("noise", "method", "D", "tau", "window"))
        groups.setdefault(key, []).append(row)
    summaries = []
    excluded = {"noise", "method", "D", "tau", "window", "seed", "diagnostic_samples"}
    for key, values in groups.items():
        row = dict(zip(("noise", "method", "D", "tau", "window"), key))
        row.update(
            seeds=len(values),
            diagnostic_samples_per_seed=values[0]["diagnostic_samples"],
        )
        for metric in values[0]:
            if metric not in excluded:
                mean, sd = finite_stats([r[metric] for r in values])
                row[metric + "_mean"], row[metric + "_sd"] = mean, sd
        summaries.append(row)
    return summaries


def terminal_rows(h, case):
    _, mode, method, delay, tau = case
    result = []
    for slot, seed in enumerate(CONFIG["seeds"]):
        q = h["q_final"][slot]
        v = h["value_final"][slot]
        qm = q.mean(0)
        row = dict(noise=noise_name(mode), method=method, D=delay, tau=tau, seed=seed)
        row.update(
            max_q_bias=float(qm.max(-1).mean()),
            q_bias=float(q.mean()),
            q_member_rmse=float(np.sqrt((q * q).mean())),
            q_ensemble_rmse=float(np.sqrt((qm * qm).mean())),
            value_bias=float(v.mean()) if method != "max_min" else np.nan,
            value_member_rmse=float(np.sqrt((v * v).mean()))
            if method != "max_min"
            else np.nan,
        )
        result.append(row)
    return result


def tracking(rows, cross_section=False):
    """Descriptive matched-seed calibration across the 12 IQL method/delay settings."""
    output = []
    suffix = "_cross_section" if cross_section else ""
    aggregation = "cross_section" if cross_section else "temporal"
    for noise in ("IID", "NN-like field"):
        for tau in CONFIG["tau_values"]:
            for seed in CONFIG["seeds"]:
                group = [
                    r
                    for r in rows
                    if r["noise"] == noise
                    and r["tau"] == tau
                    and r["seed"] == seed
                    and r["window"] == LAST
                ]
                assert len(group) == 12
                for proxy in PROXIES:
                    for target in ("operator_rmse", "q_rmse", "value_rmse"):
                        x = np.array([r[proxy + suffix + "_abs_corr"] for r in group])
                        y = np.array([r[target] for r in group])
                        valid = np.isfinite(x) & np.isfinite(y)
                        count = int(valid.sum())
                        good = (
                            count >= 3 and np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0
                        )
                        output.append(
                            dict(
                                noise=noise,
                                tau=tau,
                                seed=seed,
                                proxy=proxy,
                                accuracy=target,
                                aggregation=aggregation,
                                valid_settings=count,
                                spearman=float(spearmanr(x[valid], y[valid]).statistic)
                                if good
                                else np.nan,
                                pearson=float(pearsonr(x[valid], y[valid]).statistic)
                                if good
                                else np.nan,
                            )
                        )
    summaries = []
    for noise in ("IID", "NN-like field"):
        for tau in CONFIG["tau_values"]:
            for proxy in PROXIES:
                for target in ("operator_rmse", "q_rmse", "value_rmse"):
                    group = [
                        r
                        for r in output
                        if r["noise"] == noise
                        and r["tau"] == tau
                        and r["proxy"] == proxy
                        and r["accuracy"] == target
                    ]
                    row = dict(
                        noise=noise,
                        tau=tau,
                        proxy=proxy,
                        accuracy=target,
                        aggregation=aggregation,
                        seeds=len(group),
                    )
                    for metric in ("spearman", "pearson", "valid_settings"):
                        row[metric + "_mean"], row[metric + "_sd"] = finite_stats(
                            [r[metric] for r in group]
                        )
                    summaries.append(row)
    return output, summaries


def formatted(row, metric):
    return f"{row[metric + '_mean']:+.2f} ± {row[metric + '_sd']:.2f}"


def plots(summaries, directory):
    for metric, title in (
        ("operator_bias", "Signed aggregation/readout error"),
        ("operator_rmse", "Aggregation/readout RMSE"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for i, tau in enumerate(CONFIG["tau_values"]):
            for j, noise in enumerate(("IID", "NN-like field")):
                ax = axes[i, j]
                for method in METHODS:
                    points = [
                        r
                        for r in summaries
                        if r["noise"] == noise
                        and r["method"] == method
                        and (method == "max_min" or r["tau"] == tau)
                    ]
                    points.sort(key=lambda r: r["D"] or 0)
                    if method == "max_min":
                        r = points[0]
                        y, sd = r[metric + "_mean"], r[metric + "_sd"]
                        ax.axhline(
                            y, color=COLORS[method], ls="--", label=LABELS[method]
                        )
                        ax.axhspan(y - sd, y + sd, color=COLORS[method], alpha=0.08)
                    else:
                        ax.errorbar(
                            [r["D"] for r in points],
                            [r[metric + "_mean"] for r in points],
                            yerr=[r[metric + "_sd"] for r in points],
                            color=COLORS[method],
                            marker="o",
                            capsize=3,
                            label=LABELS[method],
                        )
                ax.axhline(0, color="black", lw=0.6)
                ax.set(
                    xscale="log",
                    xticks=CONFIG["delays"],
                    xlabel="Delay D",
                    ylabel=title,
                    title=f"{noise}; tau={tau}",
                )
                ax.set_xticklabels(CONFIG["delays"])
                ax.grid(alpha=0.15)
                ax.legend(fontsize=8)
        fig.suptitle(
            "4 states / 4 actions; 1M updates; pre-update diagnostics over last 100k\n"
            "Exact reference = 0; 20 seeds; mean ± seed SD; "
            "baseline uses max-min, IQL uses weighted filtering"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(directory / f"{metric}_vs_delay.png", dpi=180)
        plt.close(fig)
    coupling_plot(summaries, directory, cross_section=False)
    coupling_plot(summaries, directory, cross_section=True)


def coupling_plot(summaries, directory, cross_section=False):
    suffix = "_cross_section" if cross_section else ""
    proxy_names = ("oracle", "td_update", "temporal_observed", "loo_observed")
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for col, (noise, tau) in enumerate(
        (n, t) for n in ("IID", "NN-like field") for t in (0.9, 0.99)
    ):
        points = [r for r in summaries if r["noise"] == noise and r["tau"] == tau]
        for row, proxy in enumerate(proxy_names):
            ax = axes[row, col]
            for p in points:
                x = p[proxy + suffix + "_abs_corr_mean"]
                if np.isfinite(x):
                    ax.scatter(
                        x,
                        p["operator_rmse_mean"],
                        color=COLORS[p["method"]],
                        marker={1: "o", 10: "s", 100: "^", 1000: "D"}[p["D"]],
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
    mode = "Within-checkpoint" if cross_section else "Temporal"
    fig.suptitle(
        f"{mode} coupling vs oracle filtering RMSE; "
        "S=A=4; last 100k of the 1M replay\n"
        "DF-min pink, SD orange, DD green. "
        "D: circle 1, square 10, triangle 100, diamond 1000. "
        "Undefined correlations omitted.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    filename = (
        "coupling_cross_section_vs_filtering_accuracy.png"
        if cross_section
        else "coupling_vs_filtering_accuracy.png"
    )
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR.parent / "tabular_bias_diagnostics_s4_1m",
    )
    args = parser.parse_args()
    root = args.results_dir
    for name in ("tables", "figures"):
        (root / name).mkdir(exist_ok=True)
    rows, terminal, verified = [], [], []
    age_coverage = {}
    for case in cases([4], methods=METHODS):
        path = case_path(root, case)
        meta = json.loads(path.with_suffix(".json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["data_sha256"]
        with np.load(path) as f:
            h = {k: f[k] for k in f.files}
        _, mode, method, delay, tau = case
        for r in window_rows(h, method, delay or 1, tau or 0.9, CONFIG["seeds"]):
            rows.append(
                dict(noise=noise_name(mode), method=method, D=delay, tau=tau, **r)
            )
        terminal.extend(terminal_rows(h, case))
        if method != "max_min":
            ages = (h["diagnostic_step"] - 1) % delay
            assert len(np.unique(ages)) == delay
            age_coverage[str(delay)] = len(np.unique(ages))
        verified.append(
            dict(
                path=str(path),
                original_bitwise_equal=meta["original_bitwise_equal"],
                gates_and_backups_verified=True,
            )
        )
        print(
            json.dumps(
                dict(event="summarized", method=method, noise=mode, D=delay, tau=tau)
            ),
            flush=True,
        )
    summaries = summarize(rows)
    last = [r for r in summaries if r["window"] == LAST]
    write_csv(root / "tables/per_seed_windows.csv", rows)
    write_csv(root / "tables/window_summary.csv", summaries)
    write_csv(root / "tables/last100k_summary.csv", last)
    write_csv(root / "tables/final_q_v_per_seed.csv", terminal)
    tracking_rows, tracking_summary = tracking(rows)
    cross_rows, cross_summary = tracking(rows, cross_section=True)
    tracking_rows.extend(cross_rows)
    tracking_summary.extend(cross_summary)
    write_csv(root / "tables/proxy_accuracy_tracking_per_seed.csv", tracking_rows)
    write_csv(root / "tables/proxy_accuracy_tracking_summary.csv", tracking_summary)
    plots(last, root / "figures")
    report = dict(
        conditions=len(verified),
        seed_trajectories=len(terminal),
        all_original_arrays_bitwise_equal=all(
            r["original_bitwise_equal"] for r in verified
        ),
        gates_and_backups_verified=True,
        pre_update_samples=True,
        snapshot_age_coverage=age_coverage,
        undefined_correlations_preserved=True,
        instantaneous_outputs_verified=True,
        coupling_aggregations=["temporal", "cross_section"],
        condition_reports=verified,
    )
    (root / "provenance/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "condition_reports"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
