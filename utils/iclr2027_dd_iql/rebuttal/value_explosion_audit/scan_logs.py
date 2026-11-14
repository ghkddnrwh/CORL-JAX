"""Audit value-scale growth in the paper's nine-environment IQL/DD-IQL logs.

Read-only with respect to training artifacts. Run from the repository root.
An operational flag describes large observed scale growth, not mathematical
divergence or oracle value bias. See the generated README for the criterion.
"""

from __future__ import annotations

import argparse
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

try:
    from orjson import loads
except ImportError:
    loads = json.loads


ROOT = Path(__file__).resolve().parents[4]
ENVS = {
    "antmaze-large-navigate-singletask-v0": ("AM-L", 0.99),
    "humanoidmaze-medium-navigate-singletask-v0": ("HM-M", 0.995),
    "humanoidmaze-large-navigate-singletask-v0": ("HM-L", 0.995),
    "antsoccer-arena-navigate-singletask-v0": ("AS-A", 0.995),
    "cube-single-play-singletask-v0": ("Cube-S", 0.99),
    "cube-double-play-singletask-v0": ("Cube-D", 0.99),
    "puzzle-3x3-play-singletask-v0": ("Puzzle-3", 0.99),
    "puzzle-4x4-play-singletask-v0": ("Puzzle-4", 0.99),
    "scene-play-singletask-v0": ("Scene", 0.99),
}
BASE_END = 1_000_000
COLORS = {"IQL": "#1f77b4", "DD-IQL": "#ff7f0e"}


def config(path):
    payload = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else yaml.safe_load(path.read_text())
    )
    return {
        k: v.get("value") if isinstance(v, dict) and "value" in v else v
        for k, v in payload.items()
    }


def history_values(path):
    rows = {}
    with path.open("rb") as stream:
        for line in stream:
            row = loads(line)
            for container in ("history", "data", "metrics"):
                if isinstance(row.get(container), dict):
                    row = {**row, **row[container]}
                    break
            # Wide W&B exports also contain null placeholders on evaluation rows.
            if row.get("diag/v_mean") is None:
                continue
            step = next(
                (
                    row[k]
                    for k in ("training/timestep", "timestep", "global_step")
                    if isinstance(row.get(k), (int, float)) and np.isfinite(row[k])
                ),
                None,
            )
            if step is None:
                raise ValueError(f"V diagnostic has no explicit training step: {path}")
            rows[int(step)] = float(row["diag/v_mean"])
    steps = np.array(sorted(rows), dtype=int)
    return steps, np.array([rows[t] for t in steps], dtype=float)


def source_runs(args):
    runs = []
    sources = (
        ("IQL", args.iql_root, "config.json"),
        ("DD-IQL", args.dd_root, "config.yaml"),
        ("IQL-slow", args.slow_root, "config.yaml"),
    )
    for method, root, filename in sources:
        for path in sorted(root.rglob(filename)):
            cfg = config(path)
            assert cfg["env"] in ENVS, path
            assert cfg["iql_tau"] in (0.9, 0.99), path
            expected_rho = 0.0005 if method == "IQL-slow" else 0.005
            assert np.isclose(cfg["tau"], expected_rho), path
            if method == "DD-IQL":
                assert cfg["ensemble_size"] == 20 and cfg["delayed_update_period"] == 100
            data_path = path.parent / (
                "history.jsonl" if method == "IQL" else "bias_diagnostics.npz"
            )
            if not data_path.exists():
                steps, values = np.array([], dtype=int), np.array([], dtype=float)
            elif method == "IQL":
                steps, values = history_values(data_path)
            else:
                with np.load(data_path, allow_pickle=False) as archive:
                    steps = np.asarray(archive["timestep"], dtype=int)
                    values = np.asarray(archive["diag/v_mean"], dtype=float)
            status_file = path.parent / "training_status.yaml"
            status = (
                yaml.safe_load(status_file.read_text()) if status_file.exists() else {}
            )
            runs.append(
                {
                    "method": method,
                    "env": cfg["env"],
                    "gamma": cfg["discount"],
                    "expectile": cfg["iql_tau"],
                    "rho": cfg["tau"],
                    "seed": cfg["seed"],
                    "config_path": str(path.relative_to(ROOT)),
                    "data_path": str(data_path.relative_to(ROOT)),
                    "planned_steps": cfg["max_timesteps"],
                    "status": status.get("status", "exported"),
                    "steps": steps,
                    "values": values,
                }
            )
            if len(runs) % 24 == 0:
                print(f"Read {len(runs)} runs", flush=True)
    for method, expected in (("IQL", 144), ("DD-IQL", 144), ("IQL-slow", 16)):
        assert sum(r["method"] == method for r in runs) == expected, method
    identities = [
        (r["method"], r["env"], r["gamma"], r["expectile"], r["seed"]) for r in runs
    ]
    assert len(set(identities)) == len(identities)
    return runs


def large_growth(steps, values, threshold):
    """First sampled sequence above threshold spanning >=10k, >=3 observations."""
    first, count = None, 0
    for step, value in zip(steps, values):
        if np.isfinite(value) and abs(value) > threshold:
            if first is None:
                first, count = int(step), 0
            count += 1
            if count >= 3 and step - first >= 10_000:
                return first, int(step)
        else:
            first, count = None, 0
    return None, None


def run_metrics(run, cap):
    mask = run["steps"] <= cap
    steps, values = run["steps"][mask], run["values"][mask]
    row = {k: v for k, v in run.items() if k not in ("steps", "values")}
    row["cap"] = cap
    row.update(
        {
            "n_points": len(steps),
            "end_step": int(steps[-1]) if len(steps) else 0,
            "complete": bool(len(steps) and steps[-1] >= cap),
            "nonfinite_observations": int((~np.isfinite(values)).sum()),
        }
    )
    finite = np.isfinite(values)
    early = values[(steps >= 5000) & (steps <= 100_000) & finite]
    scale = max(1.0, float(np.median(np.abs(early)))) if len(early) else np.nan
    row["early_median_abs_v"] = scale
    row["v_last"] = float(values[-1]) if len(values) else np.nan
    row["v_min"] = float(values[finite].min()) if finite.any() else np.nan
    row["v_max"] = float(values[finite].max()) if finite.any() else np.nan
    row["max_abs_v"] = float(np.abs(values[finite]).max()) if finite.any() else np.nan
    for step in (100_000, 500_000, 1_000_000, 2_000_000):
        selected = values[steps == step] if step <= cap else np.array([])
        row[f"v_at_{step}"] = float(selected[-1]) if len(selected) else np.nan
    for factor in (10, 100, 1000):
        threshold = max(1000.0, factor * scale) if np.isfinite(scale) else np.nan
        first, confirmed = large_growth(steps, values, threshold)
        row[f"growth_{factor}x"] = first is not None
        row[f"first_crossing_{factor}x"] = first
        row[f"confirmed_step_{factor}x"] = confirmed
        row[f"threshold_{factor}x"] = threshold
    row["assessment"] = (
        "large_growth_observed"
        if row["growth_100x"]
        else "nonfinite_observed"
        if row["nonfinite_observations"]
        else "criterion_not_met_full_budget"
        if row["complete"]
        else "criterion_not_met_partial_log"
    )
    return row


def aggregate(frame):
    records = []
    for identity, group in frame.groupby(
        ["method", "env", "gamma", "expectile", "cap"], sort=False
    ):
        method, env, gamma, expectile, cap = identity
        endpoint = group[f"v_at_{cap}"]
        record = {
            "method": method,
            "env": env,
            "short_env": ENVS[env][0],
            "gamma": gamma,
            "expectile": expectile,
            "cap": cap,
            "n_seeds": len(group),
            "n_complete": int(group.complete.sum()),
            "end_min": int(group.end_step.min()),
            "end_max": int(group.end_step.max()),
            "n_nonfinite": int(group.nonfinite_observations.gt(0).sum()),
            "v_endpoint_n": int(endpoint.notna().sum()),
            "v_endpoint_mean": float(endpoint.mean()),
            "v_last_mean": float(group.v_last.mean()),
            "v_last_min": float(group.v_last.min()),
            "v_last_max": float(group.v_last.max()),
            "max_abs_v": float(group.max_abs_v.max()),
        }
        for factor in (10, 100, 1000):
            record[f"n_growth_{factor}x"] = int(group[f"growth_{factor}x"].sum())
        record["sensitivity_counts_differ"] = (
            len({record[f"n_growth_{f}x"] for f in (10, 100, 1000)}) > 1
        )
        records.append(record)
    return pd.DataFrame(records)


def plots(runs, out):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "pdf.fonttype": 42,
        }
    )
    booklet = PdfPages(out / "value_trajectories_overview.pdf")
    for expectile in (0.9, 0.99):
        for setting in ("original", "high_gamma"):
            fig, axes = plt.subplots(3, 3, figsize=(13, 11))
            for ax, (env, (short, original_gamma)) in zip(axes.flat, ENVS.items()):
                gamma = original_gamma if setting == "original" else 0.999
                panel_max = 0.0
                for method in ("IQL", "DD-IQL"):
                    group = [
                        r
                        for r in runs
                        if (r["method"], r["env"], r["gamma"], r["expectile"])
                        == (method, env, gamma, expectile)
                    ]
                    assert sorted(r["seed"] for r in group) == [0, 1, 2, 3]
                    if any(not len(r["steps"]) for r in group):
                        continue
                    grid = sorted(
                        set.intersection(
                            *(set(r["steps"][r["steps"] <= BASE_END]) for r in group)
                        )
                    )
                    grid = np.array([t for t in grid if t % 2500 == 0], dtype=int)
                    values = np.stack(
                        [
                            pd.Series(r["values"], index=r["steps"])
                            .reindex(grid)
                            .to_numpy()
                            for r in group
                        ]
                    )
                    values[~np.isfinite(values)] = np.nan
                    if np.isfinite(values).any():
                        panel_max = max(panel_max, float(np.nanmax(np.abs(values))))
                    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                    for seed_values in values:
                        ax.plot(
                            grid / 1e6,
                            seed_values,
                            color=COLORS[method],
                            alpha=0.15,
                            linewidth=0.7,
                        )
                    ax.plot(
                        grid / 1e6,
                        mean,
                        color=COLORS[method],
                        label=method,
                        linewidth=1.8,
                    )
                    ax.fill_between(
                        grid / 1e6,
                        mean - sd,
                        mean + sd,
                        color=COLORS[method],
                        alpha=0.1,
                        linewidth=0,
                    )
                ax.set_title(rf"{short}, $\gamma={gamma}$")
                ax.set_xlim(0, 1)
                if panel_max > 1000:
                    ax.set_yscale("symlog", linthresh=1000, linscale=2, base=10)
                    ax.locator_params(axis="y", numticks=7)
                else:
                    ax.locator_params(axis="y", nbins=5)
                ax.grid(alpha=0.25)
                ax.set_xlabel(r"Step ($\times10^6$)")
                ax.set_ylabel(r"$V$")
            setting_label = (
                "Original discount" if setting == "original" else "High discount"
            )
            fig.suptitle(fr"Expectile $\tau={expectile}$; {setting_label}", fontsize=14)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                ncol=2, frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            for extension in ("pdf", "png"):
                fig.savefig(
                    out / "figures" / f"tau{expectile}_{setting}.{extension}",
                    dpi=220,
                    bbox_inches="tight",
                )
            booklet.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    booklet.close()


def write_readme(summary, out):
    text = [
        "# Value explosion audit: nine paper environments",
        "",
        "IQL/DD-IQL 각 144 run과 작은 rho IQL 16 run을 확인했습니다.",
        "원본 코드와 로그는 변경하지 않았습니다.",
        "",
        "## 판정 기준",
        "",
        "관측된 V의 큰 규모 증가를 판정합니다.",
        "수학적 발산이나 oracle value bias를 증명하는 지표는 아닙니다.",
        "",
        "- 고정 진단 batch의 `diag/v_mean` 사용 (batch=8192).",
        "- 주 비교: 1M 이내, rho=.005, tau=.9/.99, Original/high gamma.",
        "- seed별 초기 5k-100k에서 S=max(1, median(|V|)) 계산.",
        "- |V| > max(1000, 100S)가 연속 관측 3점 이상이며",
        "  첫 관측~마지막 관측 간격이 10k step 이상이면 flag.",
        "- 10S / 1000S 기준의 민감도 결과도 CSV에 기록했습니다.",
        "- 로그가 1M보다 짧으면 관측 구간 밖의 안정성을 주장하지 않습니다.",
        "- run별 최소/최대/마지막 V, 1M V, 최초 crossing과 출처는 CSV 참조.",
        "- W&B export의 null placeholder는 진단 누락으로 처리합니다.",
        "- 평균 V 기준이므로 일부 상태나 head의 상쇄된 오차까지",
        "  배제하는 진단은 아닙니다. 값의 정확도는 별도 문제입니다.",
        "",
        "## Baseline: 큰 증가가 관측된 seed 수 / 전체 4 seed",
        "",
        "각 셀은 IQL / DD-IQL 순서입니다.",
        "`*`는 어느 한 방법에라도 1M 미만 로그가 있는 조건입니다.",
        "",
        "| Environment | Original γ | τ=.9, Original | τ=.9, γ=.999 | "
        "τ=.99, Original | τ=.99, γ=.999 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for env, (short, gamma_original) in ENVS.items():
        cells = []
        for expectile, gamma in (
            (0.9, gamma_original),
            (0.9, 0.999),
            (0.99, gamma_original),
            (0.99, 0.999),
        ):
            group = summary[
                (summary.env == env)
                & (summary.expectile == expectile)
                & (summary.gamma == gamma)
                & summary.method.isin(["IQL", "DD-IQL"])
                & summary.cap.eq(BASE_END)
            ].set_index("method")
            counts = [
                str(int(group.loc[m, "n_growth_100x"])) + "/4" for m in ("IQL", "DD-IQL")
            ]
            cells.append(
                " / ".join(counts) + ("*" if group.n_complete.lt(4).any() else "")
            )
        text.append(f"| {short} | {gamma_original} | " + " | ".join(cells) + " |")
    text.extend(
        [
            "",
            "## 해석과 로그 범위",
            "",
            "Puzzle-3와 Scene은 Original gamma, tau=.9에서 각각 한 seed가",
            "약 2.2-2.3천까지 증가했습니다. 10배 기준에는 해당하지만",
            "주요 100배 기준에는 미달하는 경계 사례로 구분합니다.",
            "",
            "1M 미만으로 기록이 끝난 조건:",
            "",
        ]
    )
    partial = summary[summary.cap.eq(BASE_END) & summary.n_complete.lt(4)]
    for _, row in partial.iterrows():
        text.append(
            f"- {row.method}, {row.short_env}, tau={row.expectile}, "
            f"gamma={row.gamma}: {row.n_complete}/4 seed가 1M까지 기록됨. "
            f"최종 step 범위 {row.end_min:,}-{row.end_max:,}."
        )
    text.extend(
        [
            "",
            "## Files",
            "",
            "- `condition_summary.csv`: 조건별 seed 수, 완료 수, 판정 수, 최종 V.",
            "- `run_summary.csv`: seed별 수치와 판정, 원본 파일.",
            "- `v_trajectories.csv.gz`: baseline 1M / 작은 rho 2M 이내 V.",
            "- `figures/`: tau/gamma별 9개 환경 V 궤적 (PNG/PDF).",
            "  실선은 4 seed 평균, 옅은 선은 각 seed, 음영은 표본 SD.",
            "  모든 seed에 기록된 공통 step만 그리고 외삽하지 않습니다.",
            "- AM-L 등 작은 rho의 이전 실험에는 V 진단이 없습니다.",
            "  작은 rho의 V는 long 아래 AS-A/Cube-S 16 run에서 확인합니다.",
            "",
            "## 그림",
            "",
            "- [네 설정의 V 궤적 모음](value_trajectories_overview.pdf)",
            "- [tau=.9, Original gamma](figures/tau0.9_original.pdf)",
            "- [tau=.9, gamma=.999](figures/tau0.9_high_gamma.pdf)",
            "- [tau=.99, Original gamma](figures/tau0.99_original.pdf)",
            "- [tau=.99, gamma=.999](figures/tau0.99_high_gamma.pdf)",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iql-root",
        type=Path,
        default=ROOT / "logs/archive/wandb_logs_2026_09_25_orl_smooth_144",
    )
    parser.add_argument(
        "--dd-root",
        type=Path,
        default=ROOT / "logs/archive/iclr2027/iql_based/ogbench/diagnostics_dd_iql",
    )
    parser.add_argument(
        "--slow-root",
        type=Path,
        default=ROOT / "logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs/iclr2027_dd_iql/rebuttal/value_explosion_audit",
    )
    args = parser.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    runs = source_runs(args)
    rows = [run_metrics(r, BASE_END) for r in runs]
    rows.extend(run_metrics(r, 2_000_000) for r in runs if r["method"] == "IQL-slow")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "run_summary.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(args.out / "condition_summary.csv", index=False)
    trajectory_rows = []
    for run in runs:
        cap = 2_000_000 if run["method"] == "IQL-slow" else BASE_END
        trajectory_rows.extend(
            {
                "method": run["method"],
                "env": run["env"],
                "gamma": run["gamma"],
                "expectile": run["expectile"],
                "seed": run["seed"],
                "step": int(t),
                "v": float(v),
            }
            for t, v in zip(run["steps"], run["values"])
            if t <= cap
        )
    pd.DataFrame(trajectory_rows).to_csv(args.out / "v_trajectories.csv.gz", index=False)
    write_readme(summary, args.out)
    print(summary.to_string(index=False), flush=True)
    plots(runs, args.out)
    metadata = {
        "runs": len(runs),
        "baseline_cap": BASE_END,
        "criterion_factor": 100,
        "criterion_floor": 1000,
        "minimum_span": 10000,
        "minimum_observations": 3,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
