"""Plot the matched 300k-to-1M frozen-model description, not a causal test."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = args.root / "analysis"
    profiles = ("layernorm", "bounded_v2")
    data = {p: json.loads((output / f"cube_{p}_advantage_drift_300k_1m.json").read_text())
            for p in profiles}
    seeds = {d["query_seed"] for d in data.values()}
    assert len(seeds) == 1
    arrays = {p: np.load(output / f"cube_{p}_advantage_drift_300k_1m.npz") for p in profiles}
    assert np.array_equal(arrays[profiles[0]]["query_indices"], arrays[profiles[1]]["query_indices"])
    assert np.array_equal(arrays[profiles[0]]["bank_indices"], arrays[profiles[1]]["bank_indices"])
    counts = sorted(map(int, data[profiles[0]]["probe_counts"]))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    names = {"layernorm": "LayerNorm IQL", "bounded_v2": "Bounded + V2 + fresh"}
    colors = {"layernorm": "#b94b48", "bounded_v2": "#26798b"}
    for p in profiles:
        measurements = data[p]["probe_counts"]
        axes[0].plot(counts, [measurements[str(k)]["within_state_Q_order_reversal_fraction"]
                             for k in counts], marker="o", label=names[p], color=colors[p])
        axes[1].plot(counts, [measurements[str(k)]["uncapped_normalized_action_TV_mean"]
                             for k in counts], marker="o", color=colors[p])
    keys = ["advantage_state_shift_centered_rms", "within_state_action_drift_rms"]
    for i, p in enumerate(profiles):
        values = [data[p]["probe_counts"]["32"][key] for key in keys]
        axes[2].bar(np.arange(2)+(i-.5)*.35, values, width=.35, color=colors[p], label=names[p])
    for ax in axes[:2]:
        ax.set_xticks(counts)
        ax.set_xlabel("Borrowed action probes per state")
        ax.set_ylim(0, 1)
    axes[0].set_title("Learned-Q pair order changes")
    axes[0].set_ylabel("Fraction")
    axes[1].set_title("Change in uncapped action weights")
    axes[1].set_ylabel("Mean state-normalized TV")
    axes[2].set_xticks([0, 1], ["State shift", "Action drift"])
    axes[2].set_title("Advantage drift components (32 probes)")
    axes[2].set_ylabel("RMS, after global shift removal")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Cube / training seed 0 / same 512 validation states / 300k to 1M")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .045), frameon=False)
    fig.text(.5, .01, "Borrowed actions are not oracle behavior samples. Descriptive model drift; no causal attribution.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(output / "cube_advantage_drift_comparison.png", dpi=180)
    fig.savefig(output / "cube_advantage_drift_comparison.pdf")
    plt.close(fig)
    rows = [dict(profile=p, **data[p]["probe_counts"][str(k)])
            for p in profiles for k in counts]
    # Nested weight details remain in the source JSON; the CSV contains scalars.
    fields = [k for k, v in rows[0].items() if not isinstance(v, dict)]
    with (output / "cube_advantage_drift_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for array in arrays.values():
        array.close()


if __name__ == "__main__":
    main()
