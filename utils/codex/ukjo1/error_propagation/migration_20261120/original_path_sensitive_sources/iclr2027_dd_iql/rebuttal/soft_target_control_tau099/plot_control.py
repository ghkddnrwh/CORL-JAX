"""Offline Fig. 1 / Table 2 comparison for the tau=0.99 slow-EMA control.

Reads existing archives and local NPZ files; never launches training or W&B.
Run from the repository root. See the generated README for estimands/coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import SymmetricalLogLocator

try:
    from orjson import loads as json_loads
except ImportError:
    json_loads = json.loads


ENVS = {
    "antsoccer-arena-navigate-singletask-v0": ("antsoccer", "AntSoccer Arena", 0.995),
    "cube-single-play-singletask-v0": ("cube_single", "Cube Single", 0.99),
}
METHODS = ("IQL", "DD-IQL", "IQL-slow")
LABELS = {
    "IQL": r"IQL ($\rho=0.005$)",
    "DD-IQL": r"DD-IQL ($\rho=0.005$)",
    "IQL-slow": r"IQL ($\rho=0.0005$)",
}
COLORS = {"IQL": "#1f77b4", "DD-IQL": "#ff7f0e", "IQL-slow": "#2ca02c"}
METRICS = {
    "value": "diag/v_mean",
    "filter_td": "diag/filter_receiver_td_corr",
    "filter_adv": "diag/filter_source_adv_corr",
    "mean_weight": "diag/filter_weight_mean",
}
Y_LABELS = {
    "value": r"$V$",
    "filter_td": r"$|\mathrm{Corr}(\omega,\delta_{\mathrm{TD}})|$",
    "filter_adv": r"$\mathrm{Corr}(\omega,A_{\mathrm{source}})$",
    "mean_weight": r"Mean filtering weight $\mathbb{E}[\omega]$",
}
REPO = Path(__file__).resolve().parents[4]
DEFAULT_CONTROL = REPO / "logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long"
DEFAULT_IQL = REPO / "logs/archive/wandb_logs_2026_09_25_orl_smooth_144"
DEFAULT_DD = REPO / "logs/archive/iclr2027/iql_based/ogbench/diagnostics_dd_iql"
DEFAULT_OUT = REPO / "logs/iclr2027_dd_iql/rebuttal/soft_target_control_tau099"
CAP = 1_000_000


@dataclass
class Run:
    method: str
    config: dict
    config_path: Path
    data_path: Path
    diagnostics: pd.DataFrame
    evaluations: pd.DataFrame
    eval_path: Path | None

    @property
    def env(self):
        return self.config["env"]

    @property
    def gamma(self):
        return float(self.config["discount"])

    @property
    def seed(self):
        return int(self.config["seed"])


def read_config(path):
    raw = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else yaml.safe_load(path.read_text())
    )
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v
        for k, v in raw.items()
    }


def relevant(config):
    return config.get("env") in ENVS and np.isclose(config.get("iql_tau", -1), 0.99)


def normalized_frame(rows):
    if not rows:
        return pd.DataFrame(columns=["timestep"])
    # Last recorded value per metric/step, including an explicit NaN or Inf.
    # Sparse W&B log calls can contribute different metrics at the same step.
    by_step = {}
    for row in rows:
        by_step.setdefault(row["timestep"], {}).update(row)
    return pd.DataFrame([by_step[step] for step in sorted(by_step)])


def read_history(path):
    diagnostics, evaluations = [], []
    with path.open("rb") as stream:
        for line in stream:
            row = json_loads(line)
            for container in ("history", "data", "metrics"):
                if isinstance(row.get(container), dict):
                    row = {**row, **row[container]}
                    break
            step = next(
                (
                    row[k]
                    for k in ("training/timestep", "timestep", "train/timestep")
                    if isinstance(row.get(k), (int, float)) and np.isfinite(row[k])
                ),
                None,
            )
            keys = set(METRICS.values()) & row.keys()
            if step is None:
                if keys:
                    raise ValueError(
                        f"Diagnostic row has no explicit training timestep: {path}"
                    )
                continue
            if keys and step <= CAP:
                diagnostics.append(
                    {"timestep": int(step), **{k: float(row[k]) for k in keys}}
                )
            if "eval/success_rate" in row:
                evaluations.append(
                    {
                        "timestep": int(step),
                        "eval/success_rate": float(row["eval/success_rate"]),
                    }
                )
    return normalized_frame(diagnostics), normalized_frame(evaluations)


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        return pd.DataFrame(
            {k: np.asarray(data[k], dtype=float) for k in ("timestep", *keys)}
        )


def discover(args):
    runs = []
    for group in ("as_iql", "cs_iql"):
        for cfg in sorted((args.iql_root / group).glob("*/config.json")):
            config = read_config(cfg)
            if not relevant(config):
                continue
            path = cfg.parent / "history.jsonl"
            diag, evaluations = read_history(path)
            runs.append(Run("IQL", config, cfg, path, diag, evaluations, None))
    for method, root in (("DD-IQL", args.dd_root), ("IQL-slow", args.control_root)):
        for cfg in sorted(root.rglob("config.yaml")):
            config = read_config(cfg)
            if not relevant(config):
                continue
            if method == "DD-IQL" and (
                config.get("ensemble_size") != 20
                or config.get("delayed_update_period") != 100
            ):
                continue
            path, eval_path = (
                cfg.parent / "bias_diagnostics.npz",
                cfg.parent / "eval_logs.npz",
            )
            diag = read_npz(path, tuple(METRICS.values()))
            if method == "IQL-slow":
                assert diag.timestep.max() == 2_000_000, path
            diag = diag[diag.timestep <= CAP].reset_index(drop=True)
            evaluations = read_npz(eval_path, ("eval/success_rate",))
            runs.append(Run(method, config, cfg, path, diag, evaluations, eval_path))
    identities = [(r.method, r.env, r.gamma, r.seed) for r in runs]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Duplicate run identities; choose a single explicit source root."
        )
    for env, (_, _, gamma) in ENVS.items():
        for discount in (gamma, 0.999):
            for method in METHODS:
                group = [
                    r
                    for r in runs
                    if (r.env, r.gamma, r.method) == (env, discount, method)
                ]
                assert sorted(r.seed for r in group) == [0, 1, 2, 3], (
                    env,
                    discount,
                    method,
                )
                expected_rho = 0.0005 if method == "IQL-slow" else 0.005
                assert all(np.isclose(r.config["tau"], expected_rho) for r in group)
    assert len(runs) == 48
    return runs


def condition_runs(runs, env, gamma, method=None):
    return sorted(
        [
            r
            for r in runs
            if r.env == env
            and r.gamma == gamma
            and (method is None or r.method == method)
        ],
        key=lambda r: (r.method, r.seed),
    )


def finite_values(run, metric, start, end, max_points=None):
    frame = run.diagnostics
    mask = frame.timestep.between(start, end) & np.isfinite(frame[metric])
    frame = frame.loc[mask, ["timestep", metric]].copy()
    available = len(frame)
    if max_points is not None and len(frame) > max_points:
        indices = np.floor(
            (np.arange(max_points) + 0.5) * len(frame) / max_points
        ).astype(int)
        frame = frame.iloc[indices]
    return frame, available


def summaries(runs):
    per_seed, selected = [], []
    for env, (_, _, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            group = condition_runs(runs, env, gamma)
            common_end = min(CAP, min(int(r.diagnostics.timestep.max()) for r in group))
            # Paper window: original logging cadence, 20 midpoint samples.
            # The longer comparison uses the same 2500-step grid for every method/seed.
            for window, end, max_points in (
                ("paper_5k_100k", 100_000, 20),
                ("matched_5k_common_end", common_end, None),
            ):
                for run in group:
                    for short, metric in METRICS.items():
                        frame, available = finite_values(
                            run, metric, 5000, end, max_points
                        )
                        if window.startswith("matched"):
                            frame = frame[frame.timestep.mod(2500).eq(0)]
                            expected = np.arange(5000, end + 1, 2500)
                            assert np.array_equal(frame.timestep.to_numpy(), expected), (
                                run.data_path,
                                metric,
                            )
                        values = frame[metric].to_numpy()
                        if short == "filter_td":
                            values = np.abs(values)
                        per_seed.append(
                            {
                                "env": env,
                                "gamma": gamma,
                                "method": run.method,
                                "seed": run.seed,
                                "window": window,
                                "metric": short,
                                "start": 5000,
                                "end": end,
                                "available_points": available,
                                "selected_points": len(values),
                                "value": float(values.mean()),
                            }
                        )
                        if window.startswith("paper"):
                            selected.extend(
                                {
                                    "env": env,
                                    "gamma": gamma,
                                    "method": run.method,
                                    "seed": run.seed,
                                    "metric": short,
                                    "timestep": int(t),
                                    "value": float(v),
                                }
                                for t, v in zip(frame.timestep, values)
                            )
    per_seed = pd.DataFrame(per_seed)
    summary = (
        per_seed.groupby(
            ["env", "gamma", "method", "window", "metric", "start", "end"], sort=False
        )
        .value.agg(mean="mean", sd_population=lambda x: np.std(x, ddof=0), n="size")
        .reset_index()
    )
    assert summary.n.eq(4).all()
    return per_seed, summary, pd.DataFrame(selected)


def style():
    # Same font, line width, band opacity and symlog transform as plot_v_mean_curve.py.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, path):
    fig.tight_layout(pad=0.15, h_pad=0.15, w_pad=0.15)
    for extension in ("pdf", "png"):
        fig.savefig(
            path.with_suffix("." + extension),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def trajectory_plots(runs, out):
    curve_rows = []
    for env, (slug, _, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            setting = "original" if gamma == original_gamma else "high_gamma"
            for short, metric in METRICS.items():
                fig, ax = plt.subplots(figsize=(10, 6))
                for method in METHODS:
                    group = condition_runs(runs, env, gamma, method)
                    start = max(int(r.diagnostics.timestep.min()) for r in group)
                    end = min(CAP, min(int(r.diagnostics.timestep.max()) for r in group))
                    grid = np.arange(int(np.ceil(start / 2500)) * 2500, end + 1, 2500)
                    # Exact recorded samples, never interpolation or forward filling.
                    values = np.stack(
                        [
                            r.diagnostics.set_index("timestep")[metric]
                            .reindex(grid)
                            .to_numpy()
                            for r in group
                        ]
                    )
                    assert np.isfinite(values).all(), (env, gamma, method, metric)
                    if short == "filter_td":
                        values = np.abs(values)
                    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                    ax.plot(
                        grid,
                        mean,
                        color=COLORS[method],
                        linewidth=2.5,
                        label=LABELS[method],
                    )
                    ax.fill_between(
                        grid,
                        mean - sd,
                        mean + sd,
                        color=COLORS[method],
                        alpha=0.15,
                        linewidth=0,
                    )
                    for t, m, s in zip(grid, mean, sd):
                        curve_rows.append(
                            {
                                "env": env,
                                "gamma": gamma,
                                "method": method,
                                "metric": short,
                                "timestep": t,
                                "mean": m,
                                "sd_sample": s,
                                "n": len(group),
                            }
                        )
                ax.set_xlabel("Step")
                ax.set_ylabel(Y_LABELS[short])
                ax.set_xlim(0, CAP)
                if short == "value":
                    ax.set_yscale("symlog", linthresh=1e3, linscale=3.0, base=10)
                    locator = SymmetricalLogLocator(linthresh=1e3, base=10)
                    locator.set_params(numticks=11)
                    ax.yaxis.set_major_locator(locator)
                elif short == "mean_weight":
                    ax.set_ylim(0, 0.65)
                ax.legend(frameon=False, loc="best")
                ax.grid(True, alpha=0.3)
                save_figure(fig, out / "figures" / f"{slug}_{setting}_{short}")
    return pd.DataFrame(curve_rows)


def diagnostic_bars(summary, out):
    paper = summary[summary.window == "paper_5k_100k"]
    metrics = ("filter_td", "filter_adv", "mean_weight")
    for env, (slug, _, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, metric in zip(axes, metrics):
                group = paper[
                    (paper.env == env)
                    & (paper.gamma == gamma)
                    & (paper.metric == metric)
                ]
                group = group.set_index("method").loc[list(METHODS)]
                ax.bar(
                    np.arange(3),
                    group["mean"],
                    yerr=group.sd_population,
                    color=[COLORS[m] for m in METHODS],
                    alpha=0.9,
                    capsize=4,
                    width=0.65,
                )
                ax.set_xticks(
                    np.arange(3),
                    ["IQL", "DD-IQL", "IQL\n" + r"$\rho=0.0005$"],
                    fontsize=15,
                )
                ax.set_ylabel(Y_LABELS[metric], fontsize=18)
                ax.grid(True, axis="y", alpha=0.3)
                ax.set_axisbelow(True)
            setting = "original" if gamma == original_gamma else "high_gamma"
            save_figure(fig, out / "figures" / f"{slug}_{setting}_table2_diagnostics")


def performance(runs, out):
    rows = []
    for run in runs:
        for step in (1_000_000, 2_000_000):
            data = run.evaluations[run.evaluations.timestep.eq(step)]
            if data.empty:
                continue
            rows.append(
                {
                    "env": run.env,
                    "gamma": run.gamma,
                    "method": run.method,
                    "seed": run.seed,
                    "timestep": step,
                    "success_percent": float(data.iloc[-1]["eval/success_rate"]) * 100,
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["env", "gamma", "method", "timestep"], sort=False)
        .success_percent.agg(
            mean="mean", sd_sample=lambda x: np.std(x, ddof=1), n="size"
        )
        .reset_index()
    )
    # Control-only 2M plot; full per-evaluation values are preserved in a separate CSV.
    all_control = []
    for env, (slug, _, original_gamma) in ENVS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for gamma, color in ((original_gamma, "#1f77b4"), (0.999, "#d62728")):
            group = condition_runs(runs, env, gamma, "IQL-slow")
            grid = sorted(
                set.intersection(*(set(r.evaluations.timestep) for r in group))
            )
            values = (
                np.stack(
                    [
                        r.evaluations.set_index("timestep")["eval/success_rate"]
                        .loc[grid]
                        .to_numpy()
                        for r in group
                    ]
                )
                * 100
            )
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(grid, mean, color=color, linewidth=2.5, label=rf"$\gamma={gamma}$")
            ax.fill_between(
                grid, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0
            )
            for run, seed_values in zip(group, values):
                all_control.extend(
                    {
                        "env": env,
                        "gamma": gamma,
                        "seed": run.seed,
                        "timestep": int(t),
                        "success_percent": float(v),
                    }
                    for t, v in zip(grid, seed_values)
                )
        ax.axvline(CAP, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Success rate (%)")
        ax.set_xlim(0, 2_000_000)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        save_figure(fig, out / "figures" / f"{slug}_slow_control_success_2m")
    pd.DataFrame(all_control).to_csv(
        out / "data/control_success_trajectory_2m.csv", index=False
    )
    return frame, summary


def write_tex(summary, out):
    for window in ("paper_5k_100k", "matched_5k_common_end"):
        endpoints = summary[summary.window == window].groupby("env").end.first()
        coverage = ", ".join(
            f"{ENVS[env][1]} {end / 1000:g}k" for env, end in endpoints.items()
        )
        rows = [
            r"% Requires \usepackage{booktabs,graphicx}",
            r"\begin{table}[t]",
            r"\centering",
            '\\caption{Expectile $\\tau=0.99$: filtering diagnostics for the slow '
            'target-update control. '
            + (
                r"Paper-compatible 5k--100k window, 20 checkpoints per seed. "
                if window.startswith("paper")
                else "Common 2500-step checkpoints from 5k to the shared endpoint: "
                + coverage
                + ". "
            )
            + r"Mean $\pm$ population SD of four seed-level means. "
            r"The Filter--TD column takes the absolute "
            r"value of each logged scalar correlation before time averaging.}",
            rf"\label{{tab:tau099_slow_control_{window}}}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            'Task & $\\gamma$ & Method & Filter--TD ($|r|$) & Filter--Adv ($r$) & Mean '
            'weight \\\\',
            r"\midrule",
        ]
        for env, (_, name, original_gamma) in ENVS.items():
            for gamma in (original_gamma, 0.999):
                for method in METHODS:
                    group = summary[
                        (summary.env == env)
                        & (summary.gamma == gamma)
                        & (summary.method == method)
                        & (summary.window == window)
                    ].set_index("metric")
                    cells = [
                        f"${group.loc[k, 'mean']:.3f}\\pm"
                        f"{group.loc[k, 'sd_population']:.3f}$"
                        for k in ("filter_td", "filter_adv", "mean_weight")
                    ]
                    rows.append(
                        f"{name} & {gamma} & {LABELS[method]} & "
                        + " & ".join(cells)
                        + r" \\"
                    )
                rows.append(r"\midrule")
        rows[-1] = r"\bottomrule"
        rows.extend([r"\end{tabular}%", "}", r"\end{table}", ""])
        (out / "tables" / f"table2_{window}.tex").write_text("\n".join(rows))


def write_figure_tex(out):
    for metric in ("value", "mean_weight"):
        rows = [
            r"% Paths are relative to this output directory.",
            r"% Requires \usepackage{graphicx}",
            r"\begin{figure*}[t]",
            r"\centering",
        ]
        for slug, name, original_gamma in ENVS.values():
            for setting, gamma in (("original", original_gamma), ("high_gamma", 0.999)):
                rows.extend(
                    [
                        r"\begin{minipage}{0.49\textwidth}\centering",
                        rf"\includegraphics[width=\linewidth]{{figures/{slug}_{setting}_{metric}.pdf}}",
                        rf"{{\small {name}, $\gamma={gamma}$}}",
                        r"\end{minipage}%",
                    ]
                )
            rows.append(r"\par\smallskip")
        estimand = (
            'Mean $V$ on the fixed diagnostic batch, with the symlog axis used in '
            'Fig.~1. '
            if metric == "value"
            else r"Mean expectile filtering weight on the fixed diagnostic batch. "
        )
        rows.extend(
            [
                r"\caption{Expectile $\tau=0.99$. "
                + estimand
                + r"Lines and bands show mean $\pm$ sample SD across four seeds. "
                'IQL uses $\\rho=0.005$ or $0.0005$; DD-IQL uses $\\rho=0.005$, $N=20$, '
                '$D=100$. '
                'All diagnostics stop at 1M. AntSoccer DD-IQL ends at 800k, the last '
                'common '
                r"recorded step across its four seeds; no values are extrapolated.}",
                rf"\label{{fig:tau099_slow_control_{metric}}}",
                r"\end{figure*}",
                "",
            ]
        )
        (out / "tables" / f"figure_{metric}_tau099.tex").write_text("\n".join(rows))
    preview = [
        r"\documentclass{article}",
        r"\usepackage[margin=15mm]{geometry}",
        r"\usepackage{booktabs,graphicx}",
        r"\begin{document}",
    ]
    for name in (
        "table2_paper_5k_100k",
        "table2_matched_5k_common_end",
        "figure_value_tau099",
        "figure_mean_weight_tau099",
    ):
        preview.extend([rf"\input{{tables/{name}.tex}}", r"\clearpage"])
    preview.append(r"\end{document}")
    (out / "comparison_preview.tex").write_text("\n".join(preview) + "\n")


def write_readme(summary, curves, perf, out):
    paper = summary[
        summary.window.eq("paper_5k_100k") & summary.metric.eq("mean_weight")
    ]
    value_end = (
        curves[curves.metric.eq("value")]
        .sort_values("timestep")
        .groupby(["env", "gamma", "method"], sort=False)
        .tail(1)
    )
    value_end.to_csv(out / "data/value_last_common_checkpoint.csv", index=False)
    text = [
        "# IQL slow-target control at expectile τ=0.99",
        "",
        '기존 코드·원본 로그는 변경하지 않았습니다. 모든 '
        '결과는 로컬 archive/NPZ를 읽어 생성하며 W&B 접속이 필요 '
        '없습니다.',
        "",
        '- Expectile **τ=0.99** (`iql_tau`); target EMA **ρ** '
        '(`tau`); discount **γ** '
        '(`discount`).',
        '- 세 방법: IQL ρ=0.005, DD-IQL ρ=0.005 (N=20, D=100), IQL '
        'ρ=0.0005. 각각 seed 0–3.',
        '- 제어 실험은 AntSoccer Arena와 Cube Single에서 Original γ / '
        'γ=0.999, 총 16 run이 2M까지 '
        '완료되었습니다.',
        '- 모든 진단은 **1M 이하**만 사용합니다. AntSoccer DD-IQL의 '
        '원본 진단 마지막 step은 seed별 '
        '800k–900k여서 그림에서는 네 seed가 모두 있는 '
        '**800k**까지만 표시합니다.',
        '- 원 Figure 1의 Scene에 해당하는 새 제어 로그는 없으므로, '
        '새 제어 로그가 있는 두 환경을 그립니다.',
        "",
        "## 그림",
        "",
        "[표와 주요 그림 4쪽 미리보기](comparison_preview.pdf) "
        "(`comparison_preview.tex`으로 컴파일).",
        "",
        '모든 그림은 title 없이 PNG/PDF로 저장했습니다. 다음 '
        '파일명은 `figures/` 기준입니다.',
        "",
        '| 환경 | γ | V 궤적 (Fig. 1 대응) | Mean weight 궤적 | Table 2 '
        '대응 세 지표 |',
        "|---|---:|---|---|---|",
    ]
    for slug, name, original_gamma in ENVS.values():
        for setting, gamma in (("original", original_gamma), ("high_gamma", 0.999)):
            links = [
                f"[{label}](figures/{slug}_{setting}_{metric}.pdf)"
                for label, metric in (
                    ("V", "value"),
                    ("Weight", "mean_weight"),
                    ("TD / Adv / Weight", "table2_diagnostics"),
                )
            ]
            text.append(f"| {name} | {gamma} | " + " | ".join(links) + " |")
    text.extend(
        [
            "",
            '추가로 `*_filter_td` / `*_filter_adv` 파일에 두 '
            'correlation의 1M 이하 궤적도 있습니다.',
            '`*_slow_control_success_2m`은 작은 ρ 제어 실험만의 평가 '
            '성공률 궤적입니다.',
            "",
            "## 측정·집계",
            "",
            'V는 세 방법 모두 **`diag/v_mean`** 입니다. 각 seed의 고정 '
            '진단 batch 8192개에서 측정한 평균이며, '
            'DD-IQL은 V ensemble도 평균합니다. 원 Figure 1 코드의 학습 '
            'minibatch `v_mean`과는 batch가 '
            '다릅니다. 스타일은 '
            '`utils/iclr2027_dd_iql/plot_v_mean_curve.py`의 Times New Roman '
            '/ STIX, 선 2.5, 음영 0.15, symlog(linthresh=1000, '
            'linscale=3)을 따릅니다.',
            "",
            '궤적은 실제 저장된 2500-step 공통 checkpoint를 '
            '사용합니다. 보간, 평활화, extrapolation, seed '
            '교체가 없습니다. 선은 네 seed의 평균, 음영은 **표본 '
            'SD(ddof=1)** 입니다. 평균−SD가 음수가 되는 것은 일부 '
            'seed 간 scale 차이가 크기 때문이며 음의 V 관측을 '
            '뜻하지 않습니다.',
            "",
            "Table 2 대응 지표:",
            "",
            '1. Filter–TD: `abs(diag/filter_receiver_td_corr)`를 '
            'checkpoint별로 구한 뒤 시간 '
            '평균합니다. DD에서는 로그 자체가 head별 signed '
            'correlation의 평균이므로, 이 값은 mean absolute '
            'per-head correlation과 다릅니다. TD는 oracle error가 아닌 '
            'Bellman residual proxy입니다.',
            '2. Filter–Adv: `diag/filter_source_adv_corr`의 signed '
            '시간 평균입니다.',
            '3. Mean weight: `diag/filter_weight_mean`의 시간 평균입니다. '
            'τ=0.99에서 ω∈{0.01, '
            '0.99}입니다. Mean weight는 sign 빈도를 반영하며 이 '
            '숫자만으로 ESS/성능의 인과관계를 입증하지는 '
            '않습니다.',
            "",
            '`tables/table2_paper_5k_100k.tex`: 원 Table 2 코드와 같은 '
            '**5k–100k '
            'inclusive**, seed마다 최대 20개의 index-bin midpoint '
            'checkpoint를 선택합니다. 시간 평균 '
            '후 네 seed의 평균±**모집단 SD(ddof=0)** 를 보고합니다. '
            'IQL Cube의 원 logging cadence가 '
            '250이고 다른 run은 2500인 점도 원 집계와 동일하게 '
            '유지했습니다. 선택 step은 '
            '`data/paper_window_selected_checkpoints.csv`에 있습니다.',
            "",
            '`tables/table2_matched_5k_common_end.tex`: 모든 '
            '방법·seed가 공통으로 가진 2500-step '
            'grid를 사용한 긴 구간 집계입니다. AntSoccer는 5k–800k, '
            'Cube는 5k–1M입니다. 원 Table 2의 '
            'early-window 수치와 구분해서 사용해야 합니다.',
            "",
            'Figure용 LaTeX는 `tables/figure_value_tau099.tex`, '
            '`tables/figure_mean_weight_tau099.tex`입니다. 그래픽 경로는 '
            '이 출력 디렉터리에서 컴파일하는 기준입니다.',
            "",
            "## 확인된 결과",
            "",
            "초기 5k–100k mean weight (평균±모집단 SD):",
            "",
            "| 환경 | γ | IQL ρ=.005 | DD-IQL | IQL ρ=.0005 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for env, (_, name, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            group = paper[(paper.env == env) & (paper.gamma == gamma)].set_index(
                "method"
            )
            cells = [
                f"{group.loc[m, 'mean']:.3f}±{group.loc[m, 'sd_population']:.3f}"
                for m in METHODS
            ]
            text.append(f"| {name} | {gamma} | " + " | ".join(cells) + " |")
    text.extend(
        [
            "",
            '각 방법의 마지막 공통 V checkpoint (평균; DD AntSoccer만 '
            '800k, 나머지 1M):',
            "",
            "| 환경 | γ | IQL ρ=.005 | DD-IQL | IQL ρ=.0005 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for env, (_, name, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            group = value_end[
                (value_end.env == env) & (value_end.gamma == gamma)
            ].set_index("method")
            cells = [f"{group.loc[m, 'mean']:.4g}" for m in METHODS]
            text.append(f"| {name} | {gamma} | " + " | ".join(cells) + " |")
    text.extend(
        [
            "",
            '작은 ρ는 값 증가를 크게 늦추지만 1M 내 drift를 '
            '제거하지 못했습니다. 따라서 이 실험에서 ‘drift가 '
            '사라졌는데도 '
            '실패한다’고 기술하면 안 됩니다. 낮은 mean weight도 '
            '지속되므로, drift와 weight concentration 중 '
            '하나만을 원인으로 확정할 수 없습니다. TD correlation '
            '감소도 true value bias 감소의 증명으로 해석하지 '
            '않습니다.',
            "",
            "## 2M 학습 시간 확인",
            "",
            '아래는 작은 ρ 제어의 **정확히 1M/2M checkpoint** 평가 '
            '성공률(50 episodes/seed), 네 seed '
            '평균±표본 SD이며 best checkpoint나 last-window 평균이 '
            '아닙니다.',
            "",
            "| 환경 | γ | 1M success (%) | 2M success (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for env, (_, name, original_gamma) in ENVS.items():
        for gamma in (original_gamma, 0.999):
            group = perf[
                (perf.env == env) & (perf.gamma == gamma) & (perf.method == "IQL-slow")
            ].set_index("timestep")
            cells = [
                f"{group.loc[t, 'mean']:.1f}±{group.loc[t, 'sd_sample']:.1f}"
                for t in (CAP, 2 * CAP)
            ]
            text.append(f"| {name} | {gamma} | " + " | ".join(cells) + " |")
    text.extend(
        [
            "",
            '2M에서도 성공률은 0–2.5%입니다. 이 관측은 해당 '
            '조건에서 ‘1M의 부족한 예산만으로 실패가 '
            '설명된다’는 설명을 약화하지만, '
            '더 긴 모든 예산에서 회복이 불가능하다고 '
            '증명하지는 않습니다.',
            "",
            'Config 대조에서 IQL 제어의 '
            'network/optimizer/dataset/diagnostic batch 설정은 '
            'baseline과 일치합니다. 달라진 항목은 '
            '`control_config_comparison.json`에 있습니다. '
            'max_timesteps=2M은 actor cosine LR decay의 horizon도 '
            '바꿉니다(진단용 IQL 코드 '
            '`actor_lr_schedule`). 따라서 이 2M run은 1M에서 끝나는 '
            'baseline을 그대로 연장한 paired '
            'continuation은 아닙니다. Q/V 업데이트에는 actor가 '
            '입력되지 않습니다. 성능의 인과 비교에는 이 schedule '
            '차이를 함께 밝혀야 합니다.',
            "",
            "## 재현 및 출처",
            "",
            "```bash",
            'python '
            'utils/iclr2027_dd_iql/rebuttal/soft_target_control_tau099/plot_control.py',
            "```",
            "",
            '- IQL baseline: '
            '`logs/archive/wandb_logs_2026_09_25_orl_smooth_144/{a'
            's_iql,cs_iql}/.../history.jsonl`',
            '- DD-IQL: '
            '`logs/archive/iclr2027/iql_based/ogbench/diagnostics_'
            'dd_iql/.../bias_diagnostics.npz`',
            '- Slow IQL: '
            '`logs/iclr2027/basic/iql_jax/ogbench/soft_target_abla'
            'tion/long/.../bias_diagnostics.npz`',
            '- 사용한 config/history/NPZ의 경로 및 SHA256: '
            '`source_manifest.json`.',
            '- 원본에서 추출한 1M 이하 진단 값, per-seed 집계, 그림 '
            '선/음영 수치, 1M/2M 성공률: `data/`.',
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(text))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(runs, out):
    records, raw_rows, differences = [], [], []
    compare_fields = (
        "discount",
        "iql_tau",
        "hidden_dim",
        "n_hidden",
        "beta",
        "vf_lr",
        "qf_lr",
        "actor_lr",
        "batch_size",
        "normalize",
        "normalize_reward",
        "diagnostic_batch_size",
        "diagnostic_seed_offset",
        "seed",
        "env",
    )
    for run in runs:
        record = {
            "method": run.method,
            "env": run.env,
            "gamma": run.gamma,
            "seed": run.seed,
            "rho": run.config["tau"],
            "expectile": run.config["iql_tau"],
            "diag_start": int(run.diagnostics.timestep.min()),
            "diag_end_used": int(run.diagnostics.timestep.max()),
            "diag_points_used": len(run.diagnostics),
            "files": {},
        }
        for path in (run.config_path, run.data_path, run.eval_path):
            if path is not None:
                record["files"][str(path.relative_to(REPO))] = sha256(path)
        records.append(record)
        for row in run.diagnostics.to_dict("records"):
            raw_rows.append(
                {
                    "env": run.env,
                    "gamma": run.gamma,
                    "method": run.method,
                    "seed": run.seed,
                    **row,
                }
            )
        if run.method == "IQL-slow":
            original = next(
                r
                for r in runs
                if (r.env, r.gamma, r.seed, r.method)
                == (run.env, run.gamma, run.seed, "IQL")
            )
            for key in compare_fields:
                assert run.config.get(key) == original.config.get(key), (
                    run.config_path,
                    key,
                )
            changed = {
                k: {"IQL": original.config.get(k), "IQL-slow": run.config.get(k)}
                for k in (
                    "tau",
                    "max_timesteps",
                    "diagnostic_freq",
                    "eval_freq",
                    "log_wandb",
                )
                if original.config.get(k) != run.config.get(k)
            }
            differences.append(
                {
                    "env": run.env,
                    "gamma": run.gamma,
                    "seed": run.seed,
                    "differences": changed,
                }
            )
    pd.DataFrame(raw_rows).to_csv(
        out / "data/diagnostic_samples_through_1m.csv", index=False
    )
    (out / "source_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "control_config_comparison.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--iql-root", type=Path, default=DEFAULT_IQL)
    parser.add_argument("--dd-root", type=Path, default=DEFAULT_DD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for folder in (
        args.out,
        args.out / "data",
        args.out / "figures",
        args.out / "tables",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    runs = discover(args)
    per_seed, summary, selected = summaries(runs)
    per_seed.to_csv(args.out / "data/diagnostic_seed_means.csv", index=False)
    summary.to_csv(args.out / "data/diagnostic_summary.csv", index=False)
    selected.to_csv(args.out / "data/paper_window_selected_checkpoints.csv", index=False)
    provenance(runs, args.out)
    style()
    curves = trajectory_plots(runs, args.out)
    curves.to_csv(args.out / "data/trajectory_summary.csv", index=False)
    diagnostic_bars(summary, args.out)
    seed_performance, performance_summary = performance(runs, args.out)
    seed_performance.to_csv(
        args.out / "data/performance_seed_endpoints.csv", index=False
    )
    performance_summary.to_csv(args.out / "data/performance_endpoints.csv", index=False)
    write_tex(summary, args.out)
    write_figure_tex(args.out)
    write_readme(summary, curves, performance_summary, args.out)
    print(summary[summary.window == "paper_5k_100k"].to_string(index=False))
    print(performance_summary.to_string(index=False))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
