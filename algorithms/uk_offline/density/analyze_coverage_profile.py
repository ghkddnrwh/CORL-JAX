# analyze_coverage_profile.py
#
# Phase 0 analysis for DCS-IQL.
#
# Validates the core premise on a real dataset BEFORE any RL training:
#   "state density rho(s) and local behavioral diversity b(s) are distinct
#    signals" (diagnosis D1 of DC-IQL). If they were near-perfectly rank
#    correlated, gating on J(s) = rho x b would add nothing over density.
#
# Reads a coverage profile cache produced by dcs_iql_jax.py /
# coverage_profile.py (no JAX needed; numpy/scipy/matplotlib only) and reports:
#   - raw kNN statistics summary
#   - Pearson + Spearman correlations between density and each diversity signal
#   - fraction of "dense-but-homogeneous" states (where a density-only signal
#     such as DC-IQL's tau(s) fires without justification)
#   - gate coverage under the chosen percentiles
# and saves histogram/hexbin figures plus a CSV summary.
#
# Usage (run from the directory containing coverage_profile.py):
#   python analyze_coverage_profile.py --cache coverage_profiles/scene-play-singletask-v0/coverage_profile_k8.npz
#   python analyze_coverage_profile.py --profile_root coverage_profiles --env scene-play-singletask-v0
#
# The cache is created automatically on the first dcs_iql_jax.py run, or you
# can build one manually with coverage_profile.compute_coverage_profile.

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import coverage_profile as covp


def resolve_cache_path(args: argparse.Namespace) -> Path:
    if args.cache is not None:
        return Path(args.cache)
    if args.profile_root is None or args.env is None:
        raise SystemExit("Provide --cache PATH, or both --profile_root and --env.")
    env_name = args.env.replace("/", "_").replace(":", "_")
    filename = f"coverage_profile_k{args.k}.npz"
    root = Path(args.profile_root) / env_name
    if args.seed is not None:
        return root / f"seed_{args.seed}" / filename
    return root / filename


def describe(name: str, values: np.ndarray) -> dict:
    q = np.percentile(values, [5, 50, 95])
    row = {
        "metric": name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }
    print(
        f"  {name:<18s} mean={row['mean']:.4f} std={row['std']:.4f} "
        f"p05={row['p05']:.4f} p50={row['p50']:.4f} p95={row['p95']:.4f}"
    )
    return row


def correlate(name: str, x: np.ndarray, y: np.ndarray, max_points: int, rng) -> dict:
    if x.shape[0] > max_points:
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    pearson = float(stats.pearsonr(xs, ys)[0])
    spearman = float(stats.spearmanr(xs, ys)[0])
    print(f"  {name:<34s} pearson={pearson:+.3f} spearman={spearman:+.3f}")
    return {"pair": name, "pearson": pearson, "spearman": spearman}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 coverage profile analysis for DCS-IQL")
    parser.add_argument("--cache", type=str, default=None, help="direct path to coverage_profile_k*.npz")
    parser.add_argument("--profile_root", type=str, default=None, help="dcs_profile_path used during training")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="only if dcs_cache_by_seed was used")
    parser.add_argument("--k", type=int, default=8, help="dcs_k used when the cache was built")
    parser.add_argument("--percentile_low", type=float, default=5.0)
    parser.add_argument("--percentile_high", type=float, default=95.0)
    parser.add_argument("--gate_low", type=float, default=60.0)
    parser.add_argument("--gate_high", type=float, default=95.0)
    parser.add_argument("--diversity_mode", type=str, default="product",
                        choices=["action", "displacement", "product"])
    parser.add_argument("--out_dir", type=str, default="coverage_analysis")
    parser.add_argument("--max_points", type=int, default=200_000,
                        help="subsample size for correlations and scatter plots")
    args = parser.parse_args()

    cache_path = resolve_cache_path(args)
    profile = covp.load_coverage_profile_cache(cache_path, expected_metadata=None)
    if profile is None:
        raise SystemExit(f"Could not load coverage profile cache: {cache_path}")

    meta = profile.get("__metadata__", {})
    n = profile["knn_radius"].shape[0]
    k = profile["neighbor_indices"].shape[1]
    rng = np.random.default_rng(0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = meta.get("env", cache_path.stem)

    print("=======================================================")
    print(f"Coverage profile: {cache_path}")
    print(f"  env={meta.get('env', '?')}  N={n}  k={k}  normalize={meta.get('normalize', '?')}")
    print("=======================================================")

    print("\n[1] Raw kNN statistics")
    raw_rows = [
        describe("knn_radius", profile["knn_radius"]),
        describe("action_spread", profile["action_spread"]),
        describe("disp_dispersion", profile["disp_dispersion"]),
    ]

    density = covp.percentile_confidence(
        profile["knn_radius"], args.percentile_low, args.percentile_high, invert=True
    )
    act_div = covp.percentile_confidence(
        profile["action_spread"], args.percentile_low, args.percentile_high
    )
    disp_div = covp.percentile_confidence(
        profile["disp_dispersion"], args.percentile_low, args.percentile_high
    )
    junction = covp.build_junction_score(density, act_div, disp_div, args.diversity_mode)
    gate = covp.gate_from_junction(junction, args.gate_low, args.gate_high)

    print(f"\n[2] Scaled confidences (percentiles [{args.percentile_low}, {args.percentile_high}])")
    conf_rows = [
        describe("density_conf", density),
        describe("action_div_conf", act_div),
        describe("disp_div_conf", disp_div),
        describe(f"junction({args.diversity_mode})", junction),
        describe("gate", gate),
    ]

    print("\n[3] Density vs diversity correlations (D1 check)")
    corr_rows = [
        correlate("density_conf vs action_div_conf", density, act_div, args.max_points, rng),
        correlate("density_conf vs disp_div_conf", density, disp_div, args.max_points, rng),
        correlate("action_div_conf vs disp_div_conf", act_div, disp_div, args.max_points, rng),
    ]

    dense = density > 0.7
    dense_homog = float(np.mean(dense & (act_div < 0.3)))
    dense_diverse = float(np.mean(dense & (act_div > 0.7)))
    gate_active = float(np.mean(gate > 0.0))
    gate_full = float(np.mean(gate >= 1.0))
    gated = gate > 0.0
    gated_density = float(np.mean(density[gated])) if gated.any() else float("nan")
    gated_act_div = float(np.mean(act_div[gated])) if gated.any() else float("nan")

    print("\n[4] Key fractions")
    print(f"  dense-but-homogeneous (density>0.7 & act_div<0.3): {dense_homog:.3f}")
    print("    -> states where a density-only signal (DC-IQL tau(s)) fires without justification")
    print(f"  dense-and-diverse     (density>0.7 & act_div>0.7): {dense_diverse:.3f}")
    print("    -> candidate junctions DCS-IQL pools over")
    print(f"  gate active fraction (gate>0): {gate_active:.3f}   fully-on (gate=1): {gate_full:.3f}")
    print(f"  among gated states: mean density={gated_density:.3f}, mean act_div={gated_act_div:.3f}")

    spearman_da = corr_rows[0]["spearman"]
    print("\n[5] Heuristic reading")
    if abs(spearman_da) > 0.85:
        print("  density and action diversity are largely redundant on this dataset;")
        print("  junction gating may add little over density alone — inspect before training.")
    elif abs(spearman_da) > 0.5:
        print("  density and action diversity are partially related but not interchangeable;")
        print("  J(s) = density x diversity carries information beyond either alone.")
    else:
        print("  density and action diversity are clearly distinct signals here —")
        print("  this is the regime the DCS-IQL design targets (D1 confirmed).")

    # ----- figures ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, vals) in zip(
        axes.ravel(),
        [
            ("density_conf", density),
            ("action_div_conf", act_div),
            ("disp_div_conf", disp_div),
            (f"junction ({args.diversity_mode})", junction),
        ],
    ):
        ax.hist(vals, bins=60, color="#4878a8", alpha=0.9)
        ax.set_title(name)
        ax.set_xlim(0, 1)
    fig.suptitle(f"Coverage profile confidences — {tag} (N={n}, k={k})")
    fig.tight_layout()
    hist_path = out_dir / "coverage_hists.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    if n > args.max_points:
        idx = rng.choice(n, size=args.max_points, replace=False)
    else:
        idx = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ydata, yname, corr in zip(
        axes,
        [act_div[idx], disp_div[idx]],
        ["action_div_conf", "disp_div_conf"],
        [corr_rows[0], corr_rows[1]],
    ):
        hb = ax.hexbin(density[idx], ydata, gridsize=50, bins="log", cmap="viridis")
        ax.set_xlabel("density_conf")
        ax.set_ylabel(yname)
        ax.set_title(f"spearman={corr['spearman']:+.3f}, pearson={corr['pearson']:+.3f}")
        fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.suptitle(f"Density vs diversity — {tag}")
    fig.tight_layout()
    scatter_path = out_dir / "density_vs_diversity.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # ----- CSV summary ------------------------------------------------------
    csv_path = out_dir / "coverage_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "value"])
        writer.writerow(["meta", "cache_path", str(cache_path)])
        writer.writerow(["meta", "env", meta.get("env", "")])
        writer.writerow(["meta", "n", n])
        writer.writerow(["meta", "k", k])
        writer.writerow(["meta", "diversity_mode", args.diversity_mode])
        writer.writerow(["meta", "gate_percentiles", f"{args.gate_low}/{args.gate_high}"])
        for row in raw_rows + conf_rows:
            for key in ("mean", "std", "p05", "p50", "p95"):
                writer.writerow(["stats", f"{row['metric']}.{key}", row[key]])
        for row in corr_rows:
            writer.writerow(["correlation", f"{row['pair']}.pearson", row["pearson"]])
            writer.writerow(["correlation", f"{row['pair']}.spearman", row["spearman"]])
        writer.writerow(["fractions", "dense_but_homogeneous", dense_homog])
        writer.writerow(["fractions", "dense_and_diverse", dense_diverse])
        writer.writerow(["fractions", "gate_active", gate_active])
        writer.writerow(["fractions", "gate_fully_on", gate_full])
        writer.writerow(["fractions", "gated_mean_density", gated_density])
        writer.writerow(["fractions", "gated_mean_act_div", gated_act_div])

    print("\nSaved:")
    print(f"  {hist_path}")
    print(f"  {scatter_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
