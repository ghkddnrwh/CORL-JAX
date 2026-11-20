"""Archived IQL/DD-IQL versus new slow-target IQL, tau=.99, AM-L/Cube-S.

Read local sources only. Main figures/tables stop at 1M; slow-control final-three
evaluation summaries separately use 1.8M, 1.9M and 2M. No training or W&B calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import SymmetricalLogLocator

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "soft_target_control_tau099")
)
sys.path.insert(
    0, str(ROOT / "utils/iclr2027_dd_iql/dd_iql_diagnostic_tables_mixed_logs")
)
import orl_diag_common as paper
import plot_control as previous

OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/soft_target_control_tau099_am_cube"
HISTORY = ROOT / "logs/iclr2027_dd_iql/archive/history_20261115"
ENVS = {
    "antmaze-large-navigate-singletask-v0": ("antmaze_large", "AntMaze Large"),
    "cube-single-play-singletask-v0": ("cube_single", "Cube Single"),
}
METHODS = ("IQL", "IQL-slow", "DD-IQL")
GAMMAS = (0.99, 0.999)
LABELS = {
    "IQL": r"IQL ($\rho=0.005$)",
    "IQL-slow": r"IQL ($\rho=0.0005$)",
    "DD-IQL": "DD-IQL",
}
COLORS = previous.COLORS
STYLES = {0.99: "-", 0.999: (0, (5, 2.5))}
WEIGHT_WINDOW = 50_000
WEIGHT_MARKERS = {0.99: "o", 0.999: "s"}
METRICS = previous.METRICS
TABLE_METRICS = ("filter_td", "filter_adv", "mean_weight")
CHECKPOINTS = np.arange(100_000, 1_000_001, 100_000)
IDS = ["env", "gamma", "method", "seed"]
FILES = {}
CHECKS = {}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    if path is not None:
        FILES[str(path.relative_to(ROOT))] = sha256(path)


def identity(run):
    return dict(env=run.env, gamma=run.gamma, method=run.method, seed=run.seed)


def wanted(c):
    return (
        c.get("env") in ENVS and c.get("iql_tau") == 0.99 and c.get("discount") in GAMMAS
    )


def discover():
    runs = []
    for group in ("am_l_iql", "cs_iql"):
        for cfg in sorted((previous.DEFAULT_IQL / group).glob("*/config.json")):
            c = previous.read_config(cfg)
            if wanted(c):
                path = cfg.parent / "history.jsonl"
                diag, ev = previous.read_history(path)
                runs.append(previous.Run("IQL", c, cfg, path, diag, ev, None))
    for method, root in (
        ("DD-IQL", previous.DEFAULT_DD),
        ("IQL-slow", previous.DEFAULT_CONTROL),
    ):
        for cfg in sorted(root.rglob("config.yaml")):
            c = previous.read_config(cfg)
            if not wanted(c):
                continue
            if method == "DD-IQL":
                if c.get("ensemble_size") != 20 or c.get("delayed_update_period") != 100:
                    continue
            elif c.get("tau") != 0.0005 or "ensemble_size" in c:
                continue
            path = cfg.parent / "bias_diagnostics.npz"
            ep = cfg.parent / "eval_logs.npz"
            diag = previous.read_npz(path, tuple(METRICS.values()))
            ev = previous.read_npz(ep, ("eval/success_rate",))
            assert not diag.timestep.duplicated().any(), path
            assert not ev.timestep.duplicated().any(), ep
            runs.append(previous.Run(method, c, cfg, path, diag, ev, ep))
    keys = [tuple(identity(r).values()) for r in runs]
    assert len(keys) == len(set(keys)) == 48
    inventory, missing = [], []
    for r in runs:
        for path in (r.config_path, r.data_path, r.eval_path):
            record(path)
        expected_rho = 0.0005 if r.method == "IQL-slow" else 0.005
        assert r.config["tau"] == expected_rho
        assert r.config["n_episodes"] == 50
        expected_end = 2_000_000 if r.method == "IQL-slow" else 1_000_000
        assert r.evaluations.timestep.max() == expected_end
        assert r.diagnostics.timestep.max() >= expected_end
        assert not r.diagnostics.timestep.duplicated().any()
        assert not r.evaluations.timestep.duplicated().any()
        main = r.diagnostics[r.diagnostics.timestep <= 1_000_000]
        assert np.isfinite(
            main[[METRICS["value"], METRICS["mean_weight"]]].to_numpy()
        ).all()
        for metric in TABLE_METRICS:
            bad = main[~np.isfinite(main[METRICS[metric]])]
            missing.extend(
                dict(**identity(r), metric=metric, timestep=int(t)) for t in bad.timestep
            )
        assert np.isfinite(r.evaluations["eval/success_rate"]).all()
        assert r.evaluations["eval/success_rate"].between(0, 1).all()
        inventory.append(
            dict(
                **identity(r),
                rho=expected_rho,
                expectile=0.99,
                source="new slow control" if r.method == "IQL-slow" else "paper archive",
                diagnostic_end=int(r.diagnostics.timestep.max()),
                evaluation_end=int(r.evaluations.timestep.max()),
                diagnostic_freq=r.config["diagnostic_freq"],
                n_episodes=r.config["n_episodes"],
                max_timesteps=r.config["max_timesteps"],
                config_path=str(r.config_path.relative_to(ROOT)),
                diagnostic_path=str(r.data_path.relative_to(ROOT)),
                evaluation_path=str((r.eval_path or r.data_path).relative_to(ROOT)),
            )
        )
    for env in ENVS:
        for gamma in GAMMAS:
            for method in METHODS:
                assert sorted(r.seed for r in select(runs, env, gamma, method)) == [
                    0,
                    1,
                    2,
                    3,
                ]
    pd.DataFrame(inventory).to_csv(OUT / "data/run_inventory.csv", index=False)
    pd.DataFrame(missing, columns=IDS + ["metric", "timestep"]).to_csv(
        OUT / "data/nonfinite_diagnostics.csv", index=False
    )
    CHECKS["nonfinite_table_diagnostic_records"] = missing

    # Bind the archived baselines to the exact files used in earlier comparisons.
    am = json.loads(
        (
            HISTORY / "antmaze_large_v_curves/source_manifest.json"
        ).read_text()
    )
    cube = json.loads(
        (HISTORY / "soft_target_control_tau099/source_manifest.json").read_text()
    )
    expected_hashes = {}
    for r in am["runs"]:
        if r["expectile"] == 0.99:
            expected_hashes[r["data_path"]] = r["data_sha256"]
            expected_hashes[r["config_path"]] = r["config_sha256"]
    for r in cube:
        if r["env"] == "cube-single-play-singletask-v0" and r["method"] != "IQL-slow":
            expected_hashes.update(r["files"])
    for path, expected in expected_hashes.items():
        assert FILES[path] == expected, path
    CHECKS["archived_file_hashes_matched_to_previous_comparisons"] = len(expected_hashes)

    fields = [
        "discount",
        "iql_tau",
        "tau",
        "max_timesteps",
        "batch_size",
        "hidden_dim",
        "n_hidden",
        "vf_lr",
        "qf_lr",
        "actor_lr",
        "beta",
        "normalize",
        "normalize_reward",
        "diagnostic_freq",
        "diagnostic_batch_size",
        "diagnostic_seed_offset",
        "n_episodes",
        "eval_freq",
    ]
    differences = []
    for slow in (r for r in runs if r.method == "IQL-slow"):
        old = next(
            r
            for r in runs
            if r.method == "IQL"
            and (r.env, r.gamma, r.seed) == (slow.env, slow.gamma, slow.seed)
        )
        differences.append(
            dict(
                **identity(slow),
                differences={
                    k: [old.config.get(k), slow.config.get(k)]
                    for k in fields
                    if old.config.get(k) != slow.config.get(k)
                },
            )
        )
    (OUT / "config_comparison.json").write_text(json.dumps(differences, indent=2) + "\n")
    return runs


def select(runs, env, gamma, method):
    return sorted(
        [r for r in runs if (r.env, r.gamma, r.method) == (env, gamma, method)],
        key=lambda r: r.seed,
    )


def paper_spec(r):
    return paper.RunSpec(
        algo=r.method,
        env=r.env,
        discount=r.gamma,
        expectile=0.99,
        seed=r.seed,
        data_format="wandb_jsonl" if r.method == "IQL" else "local_npz",
        diag_path=r.data_path,
        eval_path=r.eval_path,
        config_path=r.config_path,
        delay_period=100 if r.method == "DD-IQL" else None,
    )


def table_diagnostics(runs):
    rows, points, errors = [], [], []
    shared_grids = {}
    for env in ENVS:
        for gamma in GAMMAS:
            group = [r for r in runs if (r.env, r.gamma) == (env, gamma)]
            for metric in TABLE_METRICS:
                grid = np.arange(5000, 1_000_001, 2500)
                values = np.stack(
                    [
                        r.diagnostics.set_index("timestep")[METRICS[metric]].reindex(
                            grid
                        )
                        for r in group
                    ]
                )
                shared_grids[env, gamma, metric] = grid[np.isfinite(values).all(axis=0)]
    for r in runs:
        for window, end, maximum in (
            ("paper_5k_100k", 100_000, 20),
            ("full_5k_1m", 1_000_000, None),
        ):
            for metric in TABLE_METRICS:
                frame, available = previous.finite_values(
                    r, METRICS[metric], 5000, end, maximum
                )
                if window == "full_5k_1m":
                    grid = shared_grids[r.env, r.gamma, metric]
                    frame = frame[frame.timestep.isin(grid)]
                    np.testing.assert_array_equal(frame.timestep, grid)
                values = frame[METRICS[metric]].to_numpy()
                if metric == "filter_td":
                    values = np.abs(values)
                value = values.mean()
                if window == "paper_5k_100k":
                    assert len(values) == 20
                    ref = paper.seed_metric_mean(
                        paper_spec(r),
                        METRICS[metric],
                        5000,
                        100000,
                        absolute=(metric == "filter_td"),
                        max_points=20,
                    )
                    np.testing.assert_allclose(value, ref, rtol=0, atol=1e-12)
                    if r.method != "IQL-slow":
                        errors.append(abs(value - ref))
                rows.append(
                    dict(
                        **identity(r),
                        window=window,
                        metric=metric,
                        start=5000,
                        end=end,
                        available_points=available,
                        selected_points=len(values),
                        value=value,
                    )
                )
                points.extend(
                    dict(
                        **identity(r),
                        window=window,
                        metric=metric,
                        timestep=int(t),
                        value=float(v),
                    )
                    for t, v in zip(frame.timestep, values)
                )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["env", "gamma", "method", "window", "metric"], sort=False)
        .value.agg(
            mean="mean",
            sd_population=lambda x: x.std(ddof=0),
            sd_sample=lambda x: x.std(ddof=1),
            n="size",
        )
        .reset_index()
    )
    assert summary.n.eq(4).all()
    CHECKS["baseline_table2_seed_metrics_checked_against_original_code"] = len(errors)
    CHECKS["baseline_table2_max_error"] = float(max(errors))
    old = pd.read_csv(
        HISTORY / "soft_target_control_tau099/data/diagnostic_summary.csv"
    )
    old = old[
        (old.env == "cube-single-play-singletask-v0")
        & old.method.isin(["IQL", "DD-IQL"])
        & (old.window == "paper_5k_100k")
        & old.metric.isin(TABLE_METRICS)
    ]
    old = old.merge(
        summary,
        on=["env", "gamma", "method", "window", "metric"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    assert len(old) == 12
    for name in ("mean", "sd_population"):
        np.testing.assert_allclose(
            old[name + "_old"], old[name + "_new"], rtol=0, atol=1e-12
        )
    CHECKS["cube_archived_table_rows_reproduced"] = 12
    per_seed.to_csv(OUT / "data/table2_seed_means.csv", index=False)
    summary.to_csv(OUT / "data/table2_summary.csv", index=False)
    pd.DataFrame(points).to_csv(OUT / "data/table2_selected_points.csv.gz", index=False)
    return summary


def trajectories(runs):
    diag_frames, eval_frames, controls, curves, checkpoints = [], [], [], [], []
    for r in runs:
        diag = r.diagnostics[r.diagnostics.timestep <= 1_000_000].copy()
        ev = r.evaluations[r.evaluations.timestep <= 1_000_000].copy()
        for k, v in identity(r).items():
            diag[k] = ev[k] = v
        ev["success_percent"] = 100 * ev["eval/success_rate"]
        diag_frames.append(diag)
        eval_frames.append(ev)
        if r.method == "IQL-slow":
            control = r.evaluations[r.evaluations.timestep <= 2_000_000].copy()
            for k, v in identity(r).items():
                control[k] = v
            control["success_percent"] = 100 * control["eval/success_rate"]
            controls.append(control)
        for metric, frame, key in (
            ("value", diag, "diag/v_mean"),
            ("success", ev, "success_percent"),
            ("mean_weight", diag, METRICS["mean_weight"]),
        ):
            f = frame.set_index("timestep")[key].reindex(CHECKPOINTS)
            assert np.isfinite(f).all()
            checkpoints.extend(
                dict(**identity(r), metric=metric, timestep=int(t), value=float(v))
                for t, v in f.items()
            )
    pd.concat(diag_frames).to_csv(
        OUT / "data/diagnostics_through_1m.csv.gz", index=False
    )
    pd.concat(eval_frames).to_csv(OUT / "data/evaluations_through_1m.csv", index=False)
    pd.concat(controls).to_csv(
        OUT / "data/slow_control_evaluations_through_2m.csv", index=False
    )
    for env in ENVS:
        for gamma in GAMMAS:
            for method in METHODS:
                group = select(runs, env, gamma, method)
                for metric in ("value", "mean_weight", "success"):
                    grid = (
                        CHECKPOINTS
                        if metric == "success"
                        else np.arange(2500, 1_000_001, 2500)
                    )
                    if metric == "success":
                        values = (
                            np.stack(
                                [
                                    r.evaluations.set_index("timestep")[
                                        "eval/success_rate"
                                    ].reindex(grid)
                                    for r in group
                                ]
                            )
                            * 100
                        )
                    else:
                        values = np.stack(
                            [
                                r.diagnostics.set_index("timestep")[
                                    METRICS[metric]
                                ].reindex(grid)
                                for r in group
                            ]
                        )
                    assert np.isfinite(values).all(), (env, gamma, method, metric)
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    curves.extend(
                        dict(
                            env=env,
                            gamma=gamma,
                            method=method,
                            metric=metric,
                            timestep=int(t),
                            mean=float(m),
                            sd_sample=float(s),
                            n=4,
                        )
                        for t, m, s in zip(grid, mean, sd)
                    )
    curves = pd.DataFrame(curves)
    assert curves.timestep.max() == 1_000_000
    curves.to_csv(OUT / "data/curve_summary.csv", index=False)
    cp = pd.DataFrame(checkpoints)
    cp.to_csv(OUT / "data/checkpoint_100k_per_seed.csv", index=False)
    cps = (
        cp.groupby(["env", "gamma", "method", "metric", "timestep"], sort=False)
        .value.agg(mean="mean", sd_sample=lambda x: x.std(ddof=1), n="size")
        .reset_index()
    )
    checkpoint_rows = len(ENVS) * len(GAMMAS) * len(METHODS) * len(CHECKPOINTS) * 3
    assert len(cps) == checkpoint_rows and cps.n.eq(4).all()
    cps.to_csv(OUT / "data/checkpoint_100k_summary.csv", index=False)
    match = cps.merge(
        curves,
        on=["env", "gamma", "method", "metric", "timestep"],
        suffixes=("_table", "_curve"),
        validate="one_to_one",
    )
    assert len(match) == checkpoint_rows
    np.testing.assert_allclose(
        match.mean_table, match.mean_curve, rtol=1e-14, atol=1e-12
    )
    np.testing.assert_allclose(
        match.sd_sample_table, match.sd_sample_curve, rtol=1e-14, atol=1e-12
    )
    CHECKS["checkpoint_rows_matching_figures"] = len(match)
    return curves, cps


def last_three(runs):
    samples, per_seed = [], []
    for r in runs:
        if r.method != "IQL-slow":
            continue
        for end in (1_000_000, 2_000_000):
            f = r.evaluations[r.evaluations.timestep <= end].tail(3)
            np.testing.assert_array_equal(
                f.timestep, [end - 200_000, end - 100_000, end]
            )
            x = f["eval/success_rate"].to_numpy() * 100
            per_seed.append(
                dict(
                    **identity(r),
                    horizon=end,
                    first_step=int(f.timestep.iloc[0]),
                    last_step=end,
                    checkpoints=3,
                    success_percent=x.mean(),
                    temporal_sd_sample=x.std(ddof=1),
                )
            )
            samples.extend(
                dict(
                    **identity(r), horizon=end, timestep=int(t), success_percent=float(v)
                )
                for t, v in zip(f.timestep, x)
            )
    seeds = pd.DataFrame(per_seed)
    summary = (
        seeds.groupby(["env", "gamma", "horizon"], sort=False)
        .success_percent.agg(
            mean="mean",
            sd_sample=lambda x: x.std(ddof=1),
            sd_population=lambda x: x.std(ddof=0),
            n="size",
        )
        .reset_index()
    )
    assert summary.n.eq(4).all()
    pd.DataFrame(samples).to_csv(
        OUT / "data/slow_last3_individual_evaluations.csv", index=False
    )
    seeds.to_csv(OUT / "data/slow_last3_per_seed.csv", index=False)
    summary.to_csv(OUT / "data/slow_last3_summary.csv", index=False)
    return summary


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def weight_display_summary(runs):
    """Average common-grid records within each seed, then compute seed SD.

    Disjoint 50k windows include (start, end] and are plotted at their end.
    This affects only weight figures; the raw curves and Table 2 stay unchanged.
    """
    grid = np.arange(2500, 1_000_001, 2500)
    window_size = WEIGHT_WINDOW // 2500
    endpoints = np.arange(WEIGHT_WINDOW, 1_000_001, WEIGHT_WINDOW)
    rows = []
    for run in runs:
        values = run.diagnostics.set_index("timestep")[
            METRICS["mean_weight"]
        ].reindex(grid).to_numpy()
        assert np.isfinite(values).all()
        means = values.reshape(len(endpoints), window_size).mean(axis=1)
        np.testing.assert_allclose(means.mean(), values.mean(), rtol=1e-14, atol=1e-15)
        rows.extend(
            dict(
                **identity(run), timestep=int(end), window_start=int(end-WEIGHT_WINDOW),
                samples=window_size, value=float(value),
            )
            for end, value in zip(endpoints, means)
        )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["env", "gamma", "method", "timestep"], sort=False)
        .value.agg(mean="mean", sd_sample=lambda x: x.std(ddof=1), n="size")
        .reset_index()
    )
    assert len(summary) == 240 and summary.n.eq(4).all()
    summary["metric"] = "mean_weight"
    per_seed.to_csv(OUT / "data/weight_50k_per_seed.csv", index=False)
    summary.to_csv(OUT / "data/weight_50k_summary.csv", index=False)
    CHECKS["weight_display"] = dict(
        window_steps=WEIGHT_WINDOW, points_per_curve=20, samples_per_seed_window=20,
        window="(start, end], plotted at end", uncertainty="sample SD of four seed window means",
        raw_curves_and_tables_use_original_aggregation=True,
    )
    return summary


def draw_panel(ax, curves, env, metric, runs, value_seed_lines=True):
    for method in METHODS:
        for gamma in GAMMAS:
            f = curves[
                (curves.env == env)
                & (curves.gamma == gamma)
                & (curves.method == method)
                & (curves.metric == metric)
            ]
            x, y, sd = (
                f.timestep.to_numpy() / 1e6,
                f["mean"].to_numpy(),
                f.sd_sample.to_numpy(),
            )
            lower, upper = y - sd, y + sd
            if metric == "success":
                lower, upper = np.maximum(0, lower), np.minimum(100, upper)
            if metric == "value" and value_seed_lines:
                for run in select(runs, env, gamma, method):
                    seed_values = run.diagnostics.set_index("timestep")[
                        METRICS[metric]
                    ].reindex(f.timestep)
                    ax.plot(
                        x,
                        seed_values,
                        color=COLORS[method],
                        ls=STYLES[gamma],
                        lw=0.7,
                        alpha=0.22,
                    )
            else:
                ax.fill_between(
                    x, lower, upper, color=COLORS[method],
                    alpha=0.07 if metric == "mean_weight" else 0.10, linewidth=0
                )
            weight_style = (
                dict(
                    marker=WEIGHT_MARKERS[gamma], ms=4.5, mfc="white", mew=1.2,
                    markevery=(GAMMAS.index(gamma), 2),
                )
                if metric == "mean_weight"
                else dict(marker="o" if metric == "success" else None, ms=3.2)
            )
            ax.plot(
                x,
                y,
                color=COLORS[method],
                ls=STYLES[gamma],
                lw=1.8 if metric == "mean_weight" else 2.2,
                **weight_style,
            )
    ax.set(xlabel="Training steps (millions)", xlim=(0, 1))
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    if metric == "value":
        ax.set_ylabel("V")
        ax.set_yscale("symlog", linthresh=1000, linscale=3, base=10)
        locator = SymmetricalLogLocator(linthresh=1000, base=10)
        locator.set_params(numticks=9)
        ax.yaxis.set_major_locator(locator)
        if value_seed_lines:
            observed = np.concatenate(
                [
                    r.diagnostics.loc[
                        r.diagnostics.timestep <= 1_000_000, METRICS["value"]
                    ].to_numpy()
                    for r in runs
                    if r.env == env
                ]
            )
            # Set bounds in data space: a linear autoscale margin before symlog
            # can otherwise create a huge empty negative half-axis.
            ax.set_ylim(
                min(-1000, 1.2 * observed.min()), max(1000, 1.2 * observed.max())
            )
    elif metric == "mean_weight":
        ax.set(ylabel="Mean weight", ylim=(0, 0.65))
    else:
        ax.set(ylabel="Success rate (%)", ylim=(-1, 101))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def legends(fig, metric=None):
    methods = [
        Line2D([0], [0], color=COLORS[m], lw=2.5, label=LABELS[m]) for m in METHODS
    ]
    gammas = [
        Line2D(
            [0], [0], color="0.25", ls=STYLES[g], lw=2.0, label=rf"$\gamma={g}$",
            marker=WEIGHT_MARKERS[g] if metric == "mean_weight" else None,
            mfc="white", ms=4.5,
        )
        for g in GAMMAS
    ]
    fig.legend(
        handles=methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.legend(
        handles=gammas,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.5,
    )


def compact_layout(fig):
    """Pack axes beneath both legend rows without clipping labels or legends."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bottom = min(
        legend.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
        for legend in fig.legends
    )
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, legend_bottom - 0.008))


def figures(curves, runs, weight_curves=None):
    style()
    if weight_curves is None:
        weight_curves = weight_display_summary(runs)
    display_curves = pd.concat(
        [curves[curves.metric != "mean_weight"], weight_curves],
        ignore_index=True,
    )
    stems = {
        "value": "figure1_value",
        "mean_weight": "mean_weight",
        "success": "learning_curve",
    }
    with PdfPages(OUT / "figures_preview.pdf") as book:
        for env, (slug, _) in ENVS.items():
            for metric, stem in stems.items():
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                draw_panel(ax, display_curves, env, metric, runs)
                legends(fig, metric=metric)
                compact_layout(fig)
                assert len(fig.axes) == 1 and not ax.get_title()
                for ext in ("pdf", "png"):
                    fig.savefig(
                        OUT / "figures" / f"{stem}_{slug}.{ext}", dpi=250,
                        bbox_inches="tight", pad_inches=0.04,
                    )
                book.savefig(fig, bbox_inches="tight", pad_inches=0.04)
                plt.close(fig)
    for env, (slug, _) in ENVS.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        draw_panel(ax, curves, env, "value", runs, value_seed_lines=False)
        legends(fig)
        compact_layout(fig)
        assert len(fig.axes) == 1 and not ax.get_title()
        for ext in ("pdf", "png"):
            fig.savefig(
                OUT / "figures" / f"figure1_value_mean_sd_{slug}.{ext}", dpi=250,
                bbox_inches="tight", pad_inches=0.04,
            )
        plt.close(fig)
    CHECKS["single_panel_figures_without_titles"] = 8
    CHECKS["figure_layout"] = "tight_layout, tight bounding box, 0.04-inch padding"
    CHECKS["figure_ylabels"] = dict(value="V", mean_weight="Mean weight", success="Success rate (%)")


def pm(mean, sd, digits=3, scientific=False):
    if scientific and max(abs(mean), abs(sd)) >= 1e4:
        exponent = int(np.floor(np.log10(max(abs(mean), abs(sd)))))
        scale = 10.0**exponent
        return rf"$({mean / scale:.2f}\pm{sd / scale:.2f})\times10^{{{exponent}}}$"
    return rf"${mean:.{digits}f}\pm{sd:.{digits}f}$"


def write_table(name, caption, label, columns, headers, body):
    text = [
        r"% Requires \usepackage{booktabs,graphicx}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        r"\resizebox{\linewidth}{!}{%",
        "\\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table}",
    ]
    (OUT / "tables" / name).write_text("\n".join(text) + "\n")


def latex_tables(diag, cp, last):
    for window in ("paper_5k_100k", "full_5k_1m"):
        rows = []
        for env in ENVS:
            for gamma in GAMMAS:
                for method in METHODS:
                    f = diag[
                        (diag.env == env)
                        & (diag.gamma == gamma)
                        & (diag.method == method)
                        & (diag.window == window)
                    ].set_index("metric")
                    cells = [
                        pm(f.loc[m, "mean"], f.loc[m, "sd_population"])
                        for m in TABLE_METRICS
                    ]
                    rows.append(
                        " & ".join([ENVS[env][1], str(gamma), LABELS[method], *cells])
                        + r" \\"
                    )
                rows.append(r"\midrule")
        caption = (
            r"Filtering diagnostics at expectile $\tau=0.99$. "
            + (
                "Original Table 2 aggregation: 5k--100k, 20 checkpoints per seed. "
                if window.startswith("paper")
                else "5k--1M on common finite 2,500-step checkpoints: 398 points for AntMaze Large at $\\gamma=0.99$ correlations, 399 otherwise. "
            )
            + r"Mean $\pm$ population SD of four seed-level time averages. Filter--TD takes the absolute value of each logged correlation before averaging; Filter--Adv is signed. IQL and DD-IQL use the original archives; only IQL with $\rho=0.0005$ uses new runs. DD-IQL uses $N=20$, $D=100$, and $\rho=0.005$."
        )
        write_table(
            f"table2_{window}.tex",
            caption,
            "tab:amcube_" + window,
            "llcccc",
            [
                "Task",
                r"$\gamma$",
                "Method",
                r"Filter--TD ($|r|$)",
                r"Filter--Adv ($r$)",
                "Mean weight",
            ],
            rows[:-1],
        )
    for metric, stem, quantity in (
        ("success", "learning_100k", "Success rate (percent)"),
        ("value", "value_100k", "Mean V on the fixed diagnostic batch"),
        ("mean_weight", "mean_weight_100k", "Mean filtering weight on the fixed diagnostic batch"),
    ):
        combined = []
        for env, (slug, name) in ENVS.items():
            rows = []
            for gamma in GAMMAS:
                for step in CHECKPOINTS:
                    f = cp[
                        (cp.env == env)
                        & (cp.gamma == gamma)
                        & (cp.metric == metric)
                        & (cp.timestep == step)
                    ].set_index("method")
                    cells = [
                        pm(
                            f.loc[m, "mean"],
                            f.loc[m, "sd_sample"],
                            digits=3 if metric == "mean_weight" else 2,
                            scientific=(metric == "value"),
                        )
                        for m in METHODS
                    ]
                    rows.append(
                        " & ".join([str(gamma), f"{step // 1000}k", *cells]) + r" \\"
                    )
                rows.append(r"\midrule")
            filename = f"{stem}_{slug}.tex"
            caption = (
                f"{name}: {quantity} at each 100k checkpoint, through 1M, "
                + r"with expectile $\tau=0.99$. Mean $\pm$ sample SD across four seeds at the exact recorded checkpoint. No temporal averaging. DD-IQL: $N=20$, $D=100$, $\rho=0.005$."
            )
            if metric == "mean_weight":
                caption += " The weight trajectory plots separately use 50k-window averages."
            write_table(
                filename,
                caption,
                f"tab:amcube_{stem}_{slug}",
                "llccc",
                [r"$\gamma$", "Step", *[LABELS[m] for m in METHODS]],
                rows[:-1],
            )
            combined.append((OUT / "tables" / filename).read_text())
        (OUT / "tables" / f"{stem}.tex").write_text("\n".join(combined))
    rows = []
    for env in ENVS:
        for gamma in GAMMAS:
            f = last[(last.env == env) & (last.gamma == gamma)].set_index("horizon")
            rows.append(
                " & ".join(
                    [
                        ENVS[env][1],
                        str(gamma),
                        *[
                            pm(f.loc[end, "mean"], f.loc[end, "sd_sample"], digits=2)
                            for end in (1_000_000, 2_000_000)
                        ],
                    ]
                )
                + r" \\"
            )
    caption = (
        r"Extended training of IQL with $\rho=0.0005$ and expectile $\tau=0.99$. "
        r"Success rate (percent), averaged within each seed over the last three evaluation checkpoints ending at the indicated budget: 0.8M/0.9M/1M and 1.8M/1.9M/2M. Reported uncertainty is the sample SD across the four seed-level averages; each checkpoint uses 50 evaluation episodes."
    )
    write_table(
        "slow_last3_1m_2m.tex",
        caption,
        "tab:amcube_slow_extended",
        "llcc",
        ["Task", r"$\gamma$", "Ending at 1M", "Ending at 2M"],
        rows,
    )


def slow_success_table(first_step, last_step):
    """Tabulate recorded slow-IQL success at exact checkpoints in an interval."""
    evaluations = pd.read_csv(
        OUT / "data/slow_control_evaluations_through_2m.csv",
        float_precision="round_trip",
    )
    selected = evaluations[evaluations.timestep.between(first_step, last_step)].copy()
    assert selected.method.eq("IQL-slow").all()
    assert np.isfinite(selected.success_percent).all()
    assert not selected.duplicated(IDS + ["timestep"]).any()
    grid = np.arange(((first_step + 99_999) // 100_000) * 100_000, last_step + 1, 100_000)
    assert len(grid) > 0
    for env in ENVS:
        for gamma in GAMMAS:
            for seed in range(4):
                group = selected[
                    (selected.env == env) & (selected.gamma == gamma) & (selected.seed == seed)
                ].sort_values("timestep")
                np.testing.assert_array_equal(group.timestep, grid)
    selected["timestep"] = selected.timestep.astype(int)
    summary = (
        selected.groupby(["timestep", "env", "gamma"], sort=False)
        .success_percent.agg(mean="mean", sd_sample=lambda x: x.std(ddof=1), n="size")
        .reset_index()
    )
    assert summary.n.eq(4).all()
    stem = f"slow_success_{first_step // 1000}k_{last_step // 1000}k"
    selected.to_csv(OUT / "data" / f"{stem}_per_seed.csv", index=False)
    summary.to_csv(OUT / "data" / f"{stem}_summary.csv", index=False)
    columns = [(env, gamma) for env in ENVS for gamma in GAMMAS]
    rows = []
    for step in grid:
        frame = summary[summary.timestep == step].set_index(["env", "gamma"])
        cells = [
            pm(frame.loc[key, "mean"], frame.loc[key, "sd_sample"], digits=2)
            for key in columns
        ]
        rows.append(" & ".join([f"{step // 1000}k", *cells]) + r" \\")
    headers = ["Step"] + [
        rf"\shortstack{{{ENVS[env][1]}\\$\gamma={gamma}$}}" for env, gamma in columns
    ]
    caption = (
        r"Success rate (percent) of IQL with $\rho=0.0005$ and expectile $\tau=0.99$, "
        + f"from {first_step // 1000}k to {last_step // 1000}k steps. "
        + r"Mean $\pm$ sample SD across four seeds at each recorded 100k checkpoint; "
        r"50 evaluation episodes per seed and checkpoint. No temporal averaging."
    )
    write_table(f"{stem}.tex", caption, f"tab:amcube_{stem}", "lcccc", headers, rows)
    CHECKS["slow_success_checkpoint_table"] = dict(
        first_step=first_step, last_step=last_step, interval_steps=100_000,
        rows=len(grid), condition_cells=len(summary), seed_evaluations=len(selected),
        aggregation="exact checkpoint; mean and sample SD across four seeds",
    )


def write_preview():
    preamble = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{graphicx,booktabs,amsmath,float}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
    ]
    for name, title, caption in (
        (
            "figure1_value",
            "Value trajectories",
            r"Fixed diagnostic-batch mean V; symlog axis with linear threshold 1,000. Thin lines are individual seeds, thick lines are means. This is \texttt{diag/v\_mean}, whereas the original Figure 1 script used training-minibatch \texttt{v\_mean}.",
        ),
        (
            "mean_weight",
            "Filtering weight trajectories",
            r"Mean expectile filtering weight on the fixed diagnostic batch, averaged within each seed over disjoint 50k-step windows and plotted at window ends. Bands show the sample SD across the four seed-level window averages. Circles indicate $\gamma=0.99$; squares indicate $\gamma=0.999$. Markers alternate between windows to remain visible when curves overlap.",
        ),
        (
            "learning_curve",
            "Learning curves",
            r"Evaluation success rate; 50 episodes at each recorded checkpoint. Bands are truncated to the feasible 0--100 percent range.",
        ),
    ):
        for slug, env_name in ENVS.values():
            preamble.extend(
                [
                    rf"\noindent\includegraphics[width=\linewidth]{{figures/{name}_{slug}.pdf}}\par",
                    f"{env_name}: {title}. "
                    + caption
                    + r" Expectile $\tau=0.99$; each plot combines $\gamma=0.99$ (solid) and $\gamma=0.999$ (dashed). Means use four seeds; bands in the weight and learning curves show sample SD. DD-IQL uses $N=20$, $D=100$, $\rho=0.005$. All curves stop at 1M. IQL and DD-IQL are from the original archives; only the slow-target control uses new logs.",
                    r"\clearpage",
                ]
            )
    for name in (
        "table2_paper_5k_100k",
        "table2_full_5k_1m",
        "learning_100k_antmaze_large",
        "learning_100k_cube_single",
        "value_100k_antmaze_large",
        "value_100k_cube_single",
        "mean_weight_100k_antmaze_large",
        "mean_weight_100k_cube_single",
        "slow_last3_1m_2m",
        "slow_success_1100k_2000k",
    ):
        preamble.extend([rf"\input{{tables/{name}.tex}}", r"\clearpage"])
    preamble.append(r"\end{document}")
    (OUT / "comparison_preview.tex").write_text("\n".join(preamble) + "\n")


def notes(last):
    rows = last[last.horizon == 2_000_000]
    result_lines = [
        f"  {ENVS[r.env][1]}, gamma={r.gamma}: {r['mean']:.2f} +/- {r.sd_sample:.2f}%"
        for _, r in rows.iterrows()
    ]
    text = (
        """tau=0.99: AntMaze Large / Cube Single, old baselines + new slow-target IQL

comparison_preview.pdf: 환경별 단일 그림 6쪽과 LaTeX 표 10쪽을 모은 미리보기.
figures_preview.pdf: 환경별 단일 그림 6쪽만 모은 미리보기.
figures/: figure1_value, mean_weight, learning_curve 뒤에 _antmaze_large 또는
_cube_single을 붙인 PDF/PNG. 환경마다 별도 single plot이며 제목은 넣지 않음.
figure1_value_mean_sd_{환경}는 평균+/-SD 음영을 사용하는 추가 V 그림.
V 그림의 y축은 V, weight 그림의 y축은 Mean weight.
모든 개별 그림은 tight_layout 및 bbox_inches=tight, pad_inches=0.04로 저장.
각 그림의 한 축에 세 알고리즘 x 두 gamma를 표시.
이전 두 환경 합본 그림은 ../../archive/history_20261115/am_cube_combined_figures/에 보존.
실선 gamma=.99, 점선 gamma=.999. 색은 알고리즘. DD-IQL N20/D100/rho=.005.

데이터 선택
IQL rho=.005: logs/archive/wandb_logs_2026_09_25_orl_smooth_144/{am_l_iql,cs_iql}
DD-IQL: logs/archive/iclr2027/iql_based/ogbench/diagnostics_dd_iql
IQL rho=.0005만 logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long 사용.
새 long 경로의 rho=.005 IQL 및 DD-IQL은 이번 결과에 사용하지 않음.
각 환경/gamma/방법에 seed0--3, 총 48개 run. 진단/평가 모두 1M 존재.
run_inventory.csv와 source_manifest.json에 실제 경로와 SHA256을 기록.

지표와 표준편차
Value: diag/v_mean. 세 방법 모두 고정 진단 batch(8192)에서 측정한 V 평균.
원 Figure1 코드의 학습 미니배치 v_mean과 구분됨. 원 Figure1 스타일만 따름.
V 축: symlog(linthresh=1000, linscale=3). 0 근처는 선형, 바깥은 로그.
V 곡선: 진단 2500-step 공통 격자의 실제 기록. 학습곡선은 100k마다 평가 성공률(%).
Mean weight 그림만 50k 비중첩 구간 (start, end]의 seed별 평균을 구한 뒤,
네 seed의 평균과 sample SD(ddof=1)를 표시한다. 구간당 원 진단 20점, 곡선당 20점이며
x 좌표는 구간 끝(50k,...,1M)이다. gamma=.99는 실선/원, .999는 점선/사각형.
겹친 곡선의 마커가 가려지지 않도록 두 gamma의 마커는 번갈아 표시한다.
data/weight_50k_per_seed.csv와 weight_50k_summary.csv에 그림용 집계를 저장한다.
원자료 curve_summary.csv와 Table2/100k 표에는 이 구간 평균을 적용하지 않는다.
V/learning 곡선은 보간/평활화/외삽 없이 seed4개 평균을 사용한다.
V 주 그림은 얇은 개별 seed 궤적과 굵은 평균선을 표시한다. Learning 음영은 sample SD.
추가 V SD 그림의 큰 음수 음영은 mean-SD가 음수이기 때문이며 관측 V의 음수 폭발을
뜻하지 않는다. 100k 표에는 이 SD를 포함하고, 주 V 그림은 실제 seed 궤적을 보여준다.
성공률 음영만 0--100%로 잘랐으며 CSV와 표의 평균/SD는 자르지 않음.

tables/table2_paper_5k_100k.tex: 논문과 같은 5k--100k, seed당 최대20 midpoint
진단시점 선택, seed별 시간평균 후 네 seed의 mean +/- population SD(ddof=0).
필터-TD: 기록된 scalar corr마다 절대값을 취한 후 시간평균.
필터-Adv: signed corr. Mean weight: diag/filter_weight_mean.
IQL의 원 진단 주기는250, DD/slow는2500이므로 원 Table2와 같이 각 native 격자에서
20개를 선택한다. 세 방법의 선택 시점이 완전히 같지는 않다.
tables/table2_full_5k_1m.tex: 별도 결과. 5k--1M의 공통2500-step 격자에서 계산.
AM-L gamma=.99의 slow seed1은45k에서 두 correlation이 NaN이다. 원 Table2 방식은
유한값 중20개를 선택한다. 1M 공통 비교표는 해당 조건의 두 correlation에서45k를
세 방법/모든 seed에 대해 제외하여398점, 나머지는399점을 사용한다.
NaN을 0으로 채우지 않으며 nonfinite_diagnostics.csv에 원래 위치를 기록했다.
원 Table2를 1M 평균으로 바꿔치기하지 않는다.

tables/learning_100k.tex, value_100k.tex, mean_weight_100k.tex: 환경별 표 두 개씩 포함.
100k,200k,...,1M 정확한 시점의 mean +/- sample SD(ddof=1). 세 방법을 나란히 표시.
Mean weight 표도 해당 시점의 원 진단값을 사용하며, 그림의 50k 구간 평균과 구분됨.
Mean weight는 소수점 세 자리, CSV에는 반올림 전 값을 저장한다.
경로/패키지 의존성 없이 table 환경을 그대로 복사 가능; booktabs,graphicx 필요.
data/에는 각 seed의 원 추출값, 모든 집계값 및 Table2 선택시점도 보존.

2M 연장 결과
tables/slow_success_1100k_2000k.tex: IQL rho=.0005만 1.1M,1.2M,...,2M의
시점별 성공률을 표시. 두 환경 x 두 gamma를 한 표에 나란히 배치한다.
각 시점의 네 seed 평균 +/- sample SD이며 시간 평균을 적용하지 않는다.
data/slow_success_1100k_2000k_per_seed.csv 및 _summary.csv에 해당 값을 보존.
tables/slow_last3_1m_2m.tex: 각 seed에서 마지막3개 평가의 성공률을 먼저 평균,
이 네 seed 평균의 평균과 sample SD를 보고. 12개 평가를 독립 seed로 취급하지 않음.
1M 비교: 800k,900k,1M. 2M 비교: 1.8M,1.9M,2M. 각 평가50episodes.
이 2M 연장 결과의 표 및 CSV만 1M 이후를 사용하며, 주 비교 그림은 1M까지 유지.
마지막3개 2M 평가 결과:
"""
        + "\n".join(result_lines)
        + """

해석 범위
이 로그에서는 2M 마지막 세 평가에서도 성능이 0 또는 0에 가까웠다.
유한한 학습/평가 결과이므로 모든 더 긴 budget에서도 실패한다고 단정하지 않는다.
slow run은 max_timesteps=2M, 기존 IQL은1M이며 actor cosine LR schedule도 이 길이에
의존한다. 따라서 1M 시점의 정책 비교는 rho만 바뀐 완전한 단일변수 실험은 아니다.
주요 설정 차이는 config_comparison.json에 저장. Q/V 학습률 및 구조는 별도 확인 가능.
TD correlation과 V 크기만으로 oracle overestimation bias를 측정했다고 해석하지 않는다.

재생성 (프로젝트 루트)
python utils/iclr2027_dd_iql/rebuttal/soft_target_control_tau099_am_cube/build_comparison.py
출력 폴더에서 pdflatex -interaction=nonstopmode -halt-on-error comparison_preview.tex
원본 학습 코드/로그를 변경하지 않는다. W&B 및 네트워크에 접근하지 않는다.
"""
    )
    (OUT / "README.txt").write_text(text, encoding="utf-8")


def main():
    for child in ("data", "figures", "tables"):
        (OUT / child).mkdir(parents=True, exist_ok=True)
    runs = discover()
    print("Loaded 32 archived baselines and 16 new slow controls.", flush=True)
    diag = table_diagnostics(runs)
    curves, cp = trajectories(runs)
    last = last_three(runs)
    figures(curves, runs)
    latex_tables(diag, cp, last)
    slow_success_table(1_100_000, 2_000_000)
    write_preview()
    notes(last)
    for path, expected in FILES.items():
        assert sha256(ROOT / path) == expected
    record(Path(__file__))
    record(Path(previous.__file__))
    record(Path(paper.__file__))
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            dict(files=FILES, runs=48, old_baselines=32, new_controls=16), indent=2
        )
        + "\n"
    )
    CHECKS.update(
        runs=48,
        all_main_outputs_end_at_1m=True,
        all_conditions_have_four_seeds=True,
        input_files_unchanged=True,
        input_file_count=len(FILES),
        figure_pdf_count=8,
        script_sha256=sha256(Path(__file__)),
    )
    (OUT / "verification.json").write_text(json.dumps(CHECKS, indent=2) + "\n")
    print(last.to_string(index=False))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
