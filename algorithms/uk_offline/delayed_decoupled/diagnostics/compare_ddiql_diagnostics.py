"""Compare three persistent DDIQL/LDDIQL diagnostic runs without W&B.

Example:
  python compare_ddiql_diagnostics.py \
      --ddiql path/to/ddiql/run \
      --ddiql_fixed path/to/fixed/run \
      --lddiql path/to/lddiql/run \
      --output comparison

Requires numpy and matplotlib. Each path may be a run directory or metrics.jsonl.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOTS = [
    ("eval", "eval/reward_mean", "Evaluation return"),
    ("eval", "eval/d4rl_normalized_score_mean", "Normalized return"),
    ("diagnostic", "q/corr_mean", "Raw Q prediction correlation"),
    ("diagnostic", "td_residual/corr_mean", "Bellman residual correlation"),
    ("diagnostic", "v/corr_mean", "V prediction correlation"),
    ("diagnostic", "value_residual/corr_mean", "Value residual correlation"),
    ("diagnostic", "active_filter/corr_mean", "Active filter correlation"),
    ("diagnostic", "loo_filter/corr_mean", "Counterfactual LOO filter correlation"),
    ("diagnostic", "active_filter/pair_sign_agreement_mean", "Filter sign agreement"),
    ("diagnostic", "active_filter/all_sign_agree_fraction", "Unanimous filter sign fraction"),
    ("diagnostic", "filter/loo_small_margin_fraction", "LOO near-zero margin fraction"),
    ("diagnostic", "q/ensemble_std_mean", "Q ensemble disagreement (std)"),
    ("diagnostic", "filter/active_vs_loo_sign_disagree_fraction", "Active vs LOO sign disagreement"),
    ("diagnostic", "critic/td_residual_rmse", "Bellman residual RMSE"),
    ("train", "v_grad_norm", "Value gradient norm"),
]


def read_run(path):
    path = Path(path)
    log_path = path / "metrics.jsonl" if path.is_dir() else path
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    records = {"train": [], "diagnostic": [], "eval": []}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") in records:
                metrics = event["metrics"]
                if event["event"] == "diagnostic":
                    metrics = metrics["scalars"]
                records[event["event"]].append((int(event["timestep"]), metrics))
    manifest_path = log_path.parent / "diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return records, manifest


def series(records, event_type, key):
    items = [(step, metrics.get(key)) for step, metrics in records[event_type]]
    items = [(step, float(value)) for step, value in items if value is not None]
    items.sort(key=lambda item: item[0])
    return (np.asarray([x[0] for x in items]), np.asarray([x[1] for x in items]))


def latest(records, event_type, key):
    _, values = series(records, event_type, key)
    return float(values[-1]) if values.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddiql", required=True, help="Rotating DDIQL run folder")
    parser.add_argument("--ddiql_fixed", required=True, help="DDIQL filter_index_update_period=-1 run")
    parser.add_argument("--lddiql", required=True, help="LDDIQL run folder")
    parser.add_argument("--output", required=True, help="New or existing output folder")
    args = parser.parse_args()

    runs = {
        "DDIQL rotating": read_run(args.ddiql),
        "DDIQL fixed": read_run(args.ddiql_fixed),
        "LDDIQL": read_run(args.lddiql),
    }
    probe_hashes = {
        name: manifest.get("probe", {}).get("probe_index_sha256")
        for name, (_, manifest) in runs.items()
    }
    if None in probe_hashes.values() or len(set(probe_hashes.values())) != 1:
        raise ValueError(
            "The runs have different or unknown fixed diagnostic probes. "
            "Match dataset, seed and diagnostic_batch_size before comparing. "
            f"Found hashes: {probe_hashes}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 3, figsize=(17, 18))
    for ax, (event, metric, title) in zip(axes.flat, PLOTS):
        plotted = False
        for name, (records, _) in runs.items():
            steps, values = series(records, event, metric)
            if steps.size:
                ax.plot(steps, values, label=name, linewidth=1.7)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, "No recorded data", ha="center", va="center", transform=ax.transAxes)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_path = output / "ddiql_comparison.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)

    summary_rows = []
    for name, (records, manifest) in runs.items():
        row = {
            "run": name,
            "ensemble_size": manifest.get("config", {}).get("ensemble_size"),
            "filter_index_update_period": manifest.get("config", {}).get("filter_index_update_period"),
        }
        for event, metric, _ in PLOTS:
            row[metric] = latest(records, event, metric)
        _, eval_values = series(records, "eval", "eval/reward_mean")
        row["eval/best_reward_mean"] = float(np.max(eval_values)) if eval_values.size else float("nan")
        summary_rows.append(row)
    summary_path = output / "ddiql_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved chart: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
