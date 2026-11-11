"""Compare residual IID-noise bias without changing the training implementation.

The frozen-Q experiment isolates max/min selection bias. The GridWorld
experiment calls train() from compare_tabular_methods.py with N=2 throughout.
Both wall-penalty settings and every seed are reported; negative bias and RMSE
are retained so suppression of overestimation is not confused with accuracy.

For four tied true action values and independent standard Gaussian errors,
E[max_a min(eps_1a, eps_2a)] is approximately +0.27427. This is residual
overestimation despite a reduction from the ordinary max estimator.

DD-Q here is the legacy delayed-argmax baseline (method='dd_q'). It does
not include the learned expectile V of the new method='dd_iql'.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "clipped_double_q": "Min Q (N=2)",
    "double_q": "Double Q (N=2)",
    "dd_q": "DD-Q (N=2, D=1)",
}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1))


def gaussian_maxmin_expectation(num_actions=4):
    """Integrate the CDF of max_a min(eps_1a, eps_2a), eps ~ N(0,1)."""
    x = np.linspace(0.0, 10.0, 100_001)
    phi = np.array([0.5 * (1.0 + math.erf(z / math.sqrt(2))) for z in x])
    positive_cdf = (1.0 - (1.0 - phi) ** 2) ** num_actions
    negative_cdf = (1.0 - phi**2) ** num_actions
    return float(np.trapz(1.0 - positive_cdf - negative_cdf, x))


def frozen_q_experiment(args, output):
    rows = []
    gaps = [0.0, 0.5, 2.0]
    draws = np.arange(args.samples)
    for seed in range(args.num_seeds):
        rng = np.random.default_rng(1_000_000 + seed)
        online_noise = rng.standard_normal((args.samples, 2, 4))
        delayed_noise = rng.standard_normal((args.samples, 2, 4))
        for gap in gaps:
            true_q = np.array([0.0, -gap, -gap, -gap])
            online = true_q + online_noise
            delayed = true_q + delayed_noise
            own_actions = online.argmax(axis=2)
            peer_actions = delayed.argmax(axis=2)
            estimates = {
                "q_learning": online[:, 0].max(axis=1),
                "clipped_double_q": online.min(axis=1).max(axis=1),
                "double_q": 0.5 * (
                    online[draws, 1, own_actions[:, 0]]
                    + online[draws, 0, own_actions[:, 1]]
                ),
                "dd_q": 0.5 * (
                    online[draws, 0, peer_actions[:, 1]]
                    + online[draws, 1, peer_actions[:, 0]]
                ),
            }
            for method, values in estimates.items():
                rows.append(dict(seed=seed, action_gap=gap, method=method,
                                 bootstrap_bias=float(values.mean())))
    write_csv(output / "frozen_q_per_seed.csv", rows)
    summary = []
    for gap in gaps:
        for method in ["q_learning", *METHODS]:
            values = [r["bootstrap_bias"] for r in rows
                      if r["action_gap"] == gap and r["method"] == method]
            mean, sd = mean_sd(values)
            summary.append(dict(action_gap=gap, method=method, bias_mean=mean,
                                monte_carlo_se=sd / math.sqrt(args.num_seeds)))
    write_csv(output / "frozen_q_summary.csv", summary)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in ["q_learning", *METHODS]:
        selected = [r for r in summary if r["method"] == method]
        ax.errorbar(gaps, [r["bias_mean"] for r in selected],
                    yerr=[1.96 * r["monte_carlo_se"] for r in selected],
                    marker="o", capsize=3,
                    label=METHODS.get(method, "Q-learning"))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="True action gap / noise SD",
           ylabel="Expected bootstrap estimate - max true Q",
           title="Frozen true Q: 4 real actions, independent N(0,1) errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "frozen_q_bias.png", dpi=200)
    plt.close(fig)
    return summary


def grid_experiment(args, module, output):
    seed_rows, curve_rows, summaries = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, wall in zip(axes, [-2.0, 0.0]):
        for method, label in METHODS.items():
            all_histories = []
            for seed in range(args.num_seeds):
                history = module.train(
                    method, grid_height=2, grid_width=args.grid_width,
                    num_updates=args.updates, alpha=0.1, gamma=0.99,
                    epsilon=0.1, noise_mode="nn_like_field_iid", noise_sigma=1.0,
                    field_rho=0.999, field_features=64, field_shared_fraction=0.0,
                    field_lengthscale=0.35, field_action_scale=0.2,
                    field_innovation="gaussian", field_student_df=5.0,
                    wall_reward=wall, step_reward=0.0, goal_reward=1.0,
                    max_dim_multiplier=1, behavior_use_all_k=True,
                    double_num_q=2, dd_num_q=2, dd_delay_steps=1,
                    log_every=500, seed=seed,
                )
                all_histories.append(history)
                tail = history["step"] > 0.75 * args.updates
                row = dict(wall_reward=wall, method=method, seed=seed)
                for metric in ["v_bias", "v_rmse", "q_rmse", "initial_v_error"]:
                    row["tail_" + metric] = float(history[metric][tail].mean())
                row["tail_absolute_v_bias"] = float(np.abs(history["v_bias"][tail]).mean())
                seed_rows.append(row)
                for index, step in enumerate(history["step"]):
                    curve_rows.append(dict(wall_reward=wall, method=method,
                                           seed=seed, step=int(step),
                                           v_bias=float(history["v_bias"][index])))
            selected = [r for r in seed_rows
                        if r["wall_reward"] == wall and r["method"] == method]
            summary = dict(wall_reward=wall, method=method, num_seeds=args.num_seeds)
            for metric in ["tail_v_bias", "tail_absolute_v_bias", "tail_v_rmse",
                           "tail_q_rmse", "tail_initial_v_error"]:
                mean, sd = mean_sd([r[metric] for r in selected])
                summary[metric + "_mean"] = mean
                summary[metric + "_sd"] = sd
            summary["positive_bias_seed_count"] = sum(r["tail_v_bias"] > 0 for r in selected)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            curves = np.stack([h["v_bias"] for h in all_histories])
            mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            steps = all_histories[0]["step"]
            line, = ax.plot(steps, mean, label=label)
            ax.fill_between(steps, mean - sd, mean + sd,
                            color=line.get_color(), alpha=0.18)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Environment updates", title=f"Wall reward = {wall:g}")
    axes[0].set_ylabel("Global V bias: mean(V - V*)")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Original GridWorld trainer: 2 x {args.grid_width}, IID noise, "
                 f"N=2, {args.num_seeds} seeds (mean +/- 1 SD)")
    fig.tight_layout()
    fig.savefig(output / "grid_bias.png", dpi=200)
    fig.savefig(output / "grid_bias.pdf")
    plt.close(fig)
    write_csv(output / "grid_per_seed.csv", seed_rows)
    write_csv(output / "grid_curves.csv", curve_rows)
    write_csv(output / "grid_summary.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Frozen-Q Monte Carlo samples per seed and action gap")
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--grid-width", type=int, default=10)
    args = parser.parse_args()
    if args.num_seeds < 2 or args.samples < 1 or args.updates < 2000 or args.grid_width < 2:
        parser.error("Need >=2 seeds, >=1 sample, >=2000 updates and grid width >=2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_path = Path(__file__).with_name("compare_tabular_methods.py")
    source = source_path.read_text()
    (args.output_dir / "compare_tabular_methods_snapshot.py").write_text(source)
    module = types.ModuleType("iid_bias_training_snapshot")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), vars(module))
    metadata = {
        "source_path": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "num_seeds": args.num_seeds, "seed_ids": list(range(args.num_seeds)),
        "frozen_q_samples_per_seed": args.samples,
        "frozen_q_action_values": "[0, -gap, -gap, -gap], gap in [0, 0.5, 2]",
        "frozen_q_analytic_min_bias_at_zero_gap": gaussian_maxmin_expectation(),
        "grid_shape": [2, args.grid_width], "updates": args.updates,
        "gamma": 0.99, "alpha": 0.1, "epsilon": 0.1,
        "noise_mode": "nn_like_field_iid", "sigma": 1.0,
        "effective_rho": 0.0, "field_shared_fraction": 0.0,
        "wall_rewards": [-2.0, 0.0], "step_reward": 0.0, "goal_reward": 1.0,
        "num_estimators": 2, "dd_delay_steps": 1,
        "tail_definition": "Per-seed mean at logged steps > 0.75 * updates",
        "limitations": [
            "Frozen-Q experiment isolates bootstrap bias, not learned critic bias.",
            "GridWorld is smaller than the previous 2x150 experiments.",
            "DD-Q and Double Q can underestimate; signed bias and RMSE are reported.",
            "This max_a min_i target is not the TD3 actor-selected-action target.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    frozen_summary = frozen_q_experiment(args, args.output_dir)
    print("FROZEN_Q_ZERO_GAP", json.dumps([r for r in frozen_summary
                                            if r["action_gap"] == 0]), flush=True)
    grid_experiment(args, module, args.output_dir)
    print("Results:", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
