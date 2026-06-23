#!/usr/bin/env python3
"""
Visualize DC-IQL density confidence caches for one environment and a seed range.

This version is designed for the workflow:

    density_model_path + env + seed range + save root

It automatically saves outputs under:

    {save_path}/{env_name}/

Expected density cache layout:

    {density_model_path}/{env}/seed_{seed}/density_confidence.npz

For example:

    ./density_models/halfcheetah-medium-expert-v2/seed_0/density_confidence.npz
    ./density_models/halfcheetah-medium-expert-v2/seed_1/density_confidence.npz
    ./density_models/halfcheetah-medium-expert-v2/seed_2/density_confidence.npz

Main default outputs:
    - density_summary_by_seed.csv
    - density_summary_seed_average.csv
    - mean_density_cache.npz
    - mean_density_hist.png
    - mean_density_ecdf.png
    - std_density_across_seeds_hist.png
    - density_summary_by_seed.png
    - density_boxplot_by_seed.png
    - density_ecdf_overlay_by_seed.png
    - 2D PCA state distribution colored by seed-averaged density confidence
    - 3D PCA state distribution colored by seed-averaged density confidence

Per-seed individual histogram/ECDF files are not saved by default.
Use --save_per_seed_plots only when you explicitly want them.

Important:
    2D/3D state-distribution plots require the original offline dataset observations.
    This script reloads the dataset from --env using D4RL or OGBench, then applies
    the same observation normalization as the training script before PCA projection.

Examples:

    python visualize_density_by_env_seed_v2.py \
        --density_model_path ./density_models \
        --env halfcheetah-medium-expert-v2 \
        --seed_start 0 \
        --seed_end 4 \
        --save_path ./density_visualizations

This saves to:

    ./density_visualizations/halfcheetah-medium-expert-v2/

With tau(s) visualization:

    python visualize_density_by_env_seed_v2.py \
        --density_model_path ./density_models \
        --env halfcheetah-medium-expert-v2 \
        --seed_start 0 \
        --seed_end 4 \
        --save_path ./density_visualizations \
        --dc_tau_min 0.7 \
        --dc_tau_max 0.99

Skip dataset loading and PCA plots:

    python visualize_density_by_env_seed_v2.py \
        --density_model_path ./density_models \
        --env halfcheetah-medium-expert-v2 \
        --seed_start 0 \
        --seed_end 4 \
        --save_path ./density_visualizations \
        --skip_state_plots
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DENSITY_FILE_NAME = "density_confidence.npz"


def safe_path_name(name: str) -> str:
    """Match the simple safe-name rule used by the density cache code."""
    return str(name).replace("/", "_").replace(":", "_")


def safe_file_stem(name: str) -> str:
    """Create a safe filename stem."""
    name = str(name)
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name.strip("_") or "unknown"


def parse_metadata(raw: Any) -> Dict[str, Any]:
    """Parse JSON metadata saved inside density_confidence.npz."""
    if raw is None:
        return {}

    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    except Exception as exc:
        return {"metadata_parse_error": str(exc)}

    return {"metadata_parse_error": f"Unsupported metadata type: {type(raw)}"}


def candidate_cache_paths(
    density_model_path: Path,
    env: str,
    seed: int,
    density_file_name: str = DENSITY_FILE_NAME,
) -> List[Path]:
    """Return possible cache paths for one env/seed."""
    safe_env = safe_path_name(env)

    candidates = [
        density_model_path / safe_env / f"seed_{seed}" / density_file_name,
        density_model_path / env / f"seed_{seed}" / density_file_name,
        density_model_path / safe_env / str(seed) / density_file_name,
        density_model_path / env / str(seed) / density_file_name,
    ]

    unique_candidates = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique_candidates.append(path)
            seen.add(key)

    return unique_candidates


def resolve_cache_path(
    density_model_path: Path,
    env: str,
    seed: int,
    density_file_name: str = DENSITY_FILE_NAME,
) -> Optional[Path]:
    """Find cache file path for one env/seed."""
    for path in candidate_cache_paths(density_model_path, env, seed, density_file_name):
        if path.exists():
            return path
    return None


def load_density_cache(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load density confidence and metadata."""
    payload = np.load(path, allow_pickle=False)

    if "density_confidence" not in payload:
        raise KeyError(f"{path} does not contain key 'density_confidence'.")

    density_confidence = np.asarray(payload["density_confidence"], dtype=np.float32).reshape(-1)

    metadata = {}
    if "metadata" in payload:
        metadata = parse_metadata(payload["metadata"])

    return density_confidence, metadata


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    """Compute robust summary statistics."""
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return {
            "n": int(x.size),
            "finite_n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(x.size),
        "finite_n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def metadata_to_row(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten selected metadata fields into a CSV row."""
    return {
        "metadata_env": metadata.get("env", ""),
        "metadata_seed": metadata.get("seed", ""),
        "normalize": metadata.get("normalize", ""),
        "observations_shape": metadata.get("observations_shape", ""),
        "dc_density_k": metadata.get("dc_density_k", ""),
        "dc_density_subsample": metadata.get("dc_density_subsample", ""),
        "dc_density_chunk_size": metadata.get("dc_density_chunk_size", ""),
        "dc_density_percentile_low": metadata.get("dc_density_percentile_low", ""),
        "dc_density_percentile_high": metadata.get("dc_density_percentile_high", ""),
    }


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def is_ogbench_env(env_name: str) -> bool:
    return "singletask" in env_name or "oraclerep" in env_name


def load_env_dataset_observations(env_name: str, normalize: bool = True) -> np.ndarray:
    """
    Reload offline dataset observations for state-distribution PCA plots.

    D4RL:
        gym.make(env_name)
        d4rl.qlearning_dataset(env)

    OGBench:
        ogbench.make_env_and_datasets(env_name)
    """
    if is_ogbench_env(env_name):
        try:
            import ogbench
        except ImportError as exc:
            raise ImportError(
                "OGBench env requested, but `ogbench` is not installed. "
                "Use --skip_state_plots or install ogbench."
            ) from exc

        _, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        observations = np.asarray(train_dataset["observations"], dtype=np.float32)
    else:
        try:
            import gym
            import d4rl
        except ImportError as exc:
            raise ImportError(
                "D4RL env requested, but `gym` or `d4rl` is not installed. "
                "Use --skip_state_plots or install the required packages."
            ) from exc

        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        observations = np.asarray(dataset["observations"], dtype=np.float32)

    if normalize:
        mean, std = compute_mean_std(observations, eps=1e-3)
        observations = normalize_states(observations, mean, std).astype(np.float32)

    return observations


def sample_indices(n: int, max_points: int, rng_seed: int) -> np.ndarray:
    """Return deterministic sampled indices."""
    if max_points <= 0 or n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(rng_seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def pca_projection_numpy(x: np.ndarray, n_components: int) -> np.ndarray:
    """
    Compute PCA projection using only NumPy.

    This avoids adding a scikit-learn dependency.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")

    if x.shape[0] < 2:
        raise ValueError("PCA requires at least two samples.")

    n_components = min(n_components, x.shape[1])
    centered = x - np.mean(x, axis=0, keepdims=True)

    # SVD of sample matrix. Vt rows are principal directions.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projection = centered @ components.T

    return projection.astype(np.float32)


def save_histogram(
    values: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int,
    value_range: Optional[Tuple[float, float]] = None,
) -> Path:
    finite = values[np.isfinite(values)]

    plt.figure(figsize=(8, 5))
    plt.hist(finite, bins=bins, range=value_range)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_ecdf(
    values: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    xlim: Optional[Tuple[float, float]] = None,
) -> Path:
    finite = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(finite) + 1) / max(len(finite), 1)

    plt.figure(figsize=(8, 5))
    plt.plot(finite, y)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative fraction")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_seed_summary_plot(summary_df: pd.DataFrame, env: str, save_path: Path) -> Path:
    """Plot seed-wise mean/median/P05/P95 of density confidence."""
    fig_path = save_path / "density_summary_by_seed.png"
    plot_df = summary_df.sort_values("seed")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["seed"], plot_df["mean"], marker="o", label="Mean")
    plt.plot(plot_df["seed"], plot_df["median"], marker="o", label="Median")
    plt.plot(plot_df["seed"], plot_df["p05"], marker="o", label="P05")
    plt.plot(plot_df["seed"], plot_df["p95"], marker="o", label="P95")
    plt.xlabel("Seed")
    plt.ylabel("Density confidence c(s)")
    plt.title(f"Density confidence summary by seed\n{env}")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return fig_path


def save_seed_boxplot(
    densities_by_seed: Dict[int, np.ndarray],
    env: str,
    save_path: Path,
    max_points_per_seed: int,
    rng_seed: int,
) -> Optional[Path]:
    """Plot distribution comparison across seeds."""
    rng = np.random.default_rng(rng_seed)

    values = []
    labels = []

    for seed in sorted(densities_by_seed.keys()):
        density = densities_by_seed[seed]
        finite = density[np.isfinite(density)]

        if finite.size == 0:
            continue

        if finite.size > max_points_per_seed:
            idx = rng.choice(finite.size, size=max_points_per_seed, replace=False)
            finite = finite[idx]

        values.append(finite)
        labels.append(str(seed))

    if not values:
        return None

    fig_path = save_path / "density_boxplot_by_seed.png"

    plt.figure(figsize=(max(8, len(values) * 0.8), 5))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.xlabel("Seed")
    plt.ylabel("Density confidence c(s)")
    plt.title(f"Density confidence distribution by seed\n{env}")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return fig_path


def save_ecdf_overlay(
    densities_by_seed: Dict[int, np.ndarray],
    env: str,
    save_path: Path,
    max_points_per_seed: int,
    rng_seed: int,
) -> Optional[Path]:
    """Overlay seed-wise density ECDF curves."""
    if not densities_by_seed:
        return None

    rng = np.random.default_rng(rng_seed)
    fig_path = save_path / "density_ecdf_overlay_by_seed.png"

    plt.figure(figsize=(8, 5))
    plotted = False

    for seed in sorted(densities_by_seed.keys()):
        density = densities_by_seed[seed]
        finite = density[np.isfinite(density)]

        if finite.size == 0:
            continue

        if finite.size > max_points_per_seed:
            idx = rng.choice(finite.size, size=max_points_per_seed, replace=False)
            finite = finite[idx]

        finite = np.sort(finite)
        y = np.arange(1, len(finite) + 1) / max(len(finite), 1)

        plt.plot(finite, y, label=f"seed {seed}")
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.xlabel("Density confidence c(s)")
    plt.ylabel("Cumulative fraction")
    plt.title(f"Density confidence ECDF overlay\n{env}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return fig_path


def save_state_distribution_2d(
    projection_2d: np.ndarray,
    density_values: np.ndarray,
    env: str,
    save_path: Path,
    name: str,
) -> Path:
    """Save 2D PCA scatter colored by density confidence."""
    fig_path = save_path / f"{name}_state_distribution_2d_pca.png"

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projection_2d[:, 0],
        projection_2d[:, 1],
        c=density_values,
        s=3,
        alpha=0.6,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"State distribution in 2D PCA\n{env}")
    plt.colorbar(scatter, label="Density confidence c(s)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    return fig_path


def save_state_distribution_3d(
    projection_3d: np.ndarray,
    density_values: np.ndarray,
    env: str,
    save_path: Path,
    name: str,
) -> Path:
    """Save 3D PCA scatter colored by density confidence."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig_path = save_path / f"{name}_state_distribution_3d_pca.png"

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        projection_3d[:, 0],
        projection_3d[:, 1],
        projection_3d[:, 2],
        c=density_values,
        s=3,
        alpha=0.6,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"State distribution in 3D PCA\n{env}")
    fig.colorbar(scatter, ax=ax, label="Density confidence c(s)", shrink=0.7)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    return fig_path


def save_projection_csv(
    projection: np.ndarray,
    density_values: np.ndarray,
    indices: np.ndarray,
    save_path: Path,
    name: str,
) -> Path:
    """Save sampled projection coordinates and density values."""
    data = {
        "index": indices,
        "density_confidence": density_values,
    }

    for dim in range(projection.shape[1]):
        data[f"pc{dim + 1}"] = projection[:, dim]

    df = pd.DataFrame(data)
    csv_path = save_path / f"{name}_state_distribution_pca_points.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def load_requested_density_caches(
    density_model_path: Path,
    env: str,
    seed_start: int,
    seed_end: int,
    density_file_name: str,
    missing: str,
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Load density caches for the requested seed range."""
    summary_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    densities_by_seed: Dict[int, np.ndarray] = {}

    expected_n: Optional[int] = None

    for seed in range(seed_start, seed_end + 1):
        cache_path = resolve_cache_path(
            density_model_path=density_model_path,
            env=env,
            seed=seed,
            density_file_name=density_file_name,
        )

        if cache_path is None:
            tried = candidate_cache_paths(
                density_model_path=density_model_path,
                env=env,
                seed=seed,
                density_file_name=density_file_name,
            )
            missing_rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "tried_paths": " | ".join(str(path) for path in tried),
                }
            )
            message = f"[missing] env={env}, seed={seed}"
            if missing == "error":
                raise FileNotFoundError(message + "\nTried:\n" + "\n".join(str(path) for path in tried))
            print(message)
            continue

        print(f"[load] seed={seed}: {cache_path}")
        density, metadata = load_density_cache(cache_path)

        if expected_n is None:
            expected_n = density.shape[0]
        elif density.shape[0] != expected_n:
            raise ValueError(
                f"Density length mismatch for seed {seed}: "
                f"expected {expected_n}, got {density.shape[0]}"
            )

        densities_by_seed[seed] = density

        stats = summarize_array(density)
        row = {
            "env": env,
            "seed": seed,
            "cache_path": str(cache_path),
            **stats,
            **metadata_to_row(metadata),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    missing_df = pd.DataFrame(missing_rows)

    return densities_by_seed, summary_df, missing_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize DC-IQL density caches for one env and seed range."
    )

    parser.add_argument(
        "--density_model_path",
        type=str,
        required=True,
        help="Root path where density caches are saved.",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Target environment name, e.g. halfcheetah-medium-expert-v2.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="First seed to visualize, inclusive.",
    )
    parser.add_argument(
        "--seed_end",
        type=int,
        required=True,
        help="Last seed to visualize, inclusive.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Root save directory. The script automatically appends /{env_name}.",
    )
    parser.add_argument(
        "--density_file_name",
        type=str,
        default=DENSITY_FILE_NAME,
        help="Density cache filename. Default: density_confidence.npz.",
    )
    parser.add_argument(
        "--missing",
        type=str,
        choices=["skip", "error"],
        default="skip",
        help="What to do when a seed cache is missing.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "--dc_tau_min",
        type=float,
        default=None,
        help="Optional tau min. If provided with --dc_tau_max, tau histograms are saved.",
    )
    parser.add_argument(
        "--dc_tau_max",
        type=float,
        default=None,
        help="Optional tau max. If provided with --dc_tau_min, tau histograms are saved.",
    )
    parser.add_argument(
        "--max_points_per_seed",
        type=int,
        default=100_000,
        help="Max sampled points per seed for seed-wise aggregate plots.",
    )
    parser.add_argument(
        "--max_state_plot_points",
        type=int,
        default=100_000,
        help="Max sampled states for 2D/3D PCA state-distribution plots.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="RNG seed for visualization downsampling.",
    )
    parser.add_argument(
        "--skip_state_plots",
        action="store_true",
        help="Skip dataset loading and 2D/3D PCA state-distribution plots.",
    )
    parser.add_argument(
        "--no_normalize_observations",
        action="store_true",
        help="Do not normalize dataset observations before PCA. Default matches training normalization.",
    )
    parser.add_argument(
        "--save_per_seed_plots",
        action="store_true",
        help=(
            "Also save individual histogram/ECDF files for each seed. "
            "Default is False to keep the output directory compact."
        ),
    )
    parser.add_argument(
        "--per_seed_state_plots",
        action="store_true",
        help="Also save 2D PCA state plots for each seed, not only seed-averaged density.",
    )

    args = parser.parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("--seed_end must be greater than or equal to --seed_start.")

    density_model_path = Path(args.density_model_path).expanduser().resolve()
    save_root = Path(args.save_path).expanduser().resolve()
    save_path = save_root / safe_path_name(args.env)
    save_path.mkdir(parents=True, exist_ok=True)

    if not density_model_path.exists():
        raise FileNotFoundError(f"density_model_path does not exist: {density_model_path}")

    save_tau = args.dc_tau_min is not None and args.dc_tau_max is not None
    if (args.dc_tau_min is None) ^ (args.dc_tau_max is None):
        print("Only one of --dc_tau_min/--dc_tau_max was provided. Skipping tau histograms.")
        save_tau = False

    generated_files: List[str] = []

    densities_by_seed, summary_df, missing_df = load_requested_density_caches(
        density_model_path=density_model_path,
        env=args.env,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        density_file_name=args.density_file_name,
        missing=args.missing,
    )

    if not densities_by_seed:
        missing_path = save_path / "missing_density_caches.csv"
        missing_df.to_csv(missing_path, index=False)
        raise RuntimeError(
            f"No density cache was loaded. Missing cache report saved to: {missing_path}"
        )

    loaded_seeds = sorted(densities_by_seed.keys())

    summary_df = summary_df.sort_values("seed")
    summary_csv_path = save_path / "density_summary_by_seed.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    generated_files.append(str(summary_csv_path))

    if not missing_df.empty:
        missing_csv_path = save_path / "missing_density_caches.csv"
        missing_df.to_csv(missing_csv_path, index=False)
        generated_files.append(str(missing_csv_path))

    # Optional seed-wise individual plots.
    # These are disabled by default because they create many files.
    if args.save_per_seed_plots:
        per_seed_dir = save_path / "per_seed"
        per_seed_dir.mkdir(parents=True, exist_ok=True)

        for seed in loaded_seeds:
            density = densities_by_seed[seed]

            hist_path = save_histogram(
                values=density,
                output_path=per_seed_dir / f"seed_{seed}_density_hist.png",
                title=f"Density confidence histogram\n{args.env} / seed {seed}",
                xlabel="Density confidence c(s)",
                bins=args.bins,
                value_range=(0.0, 1.0),
            )
            ecdf_path = save_ecdf(
                values=density,
                output_path=per_seed_dir / f"seed_{seed}_density_ecdf.png",
                title=f"Density confidence ECDF\n{args.env} / seed {seed}",
                xlabel="Density confidence c(s)",
                xlim=(0.0, 1.0),
            )
            generated_files.extend([str(hist_path), str(ecdf_path)])

            if save_tau:
                tau_values = args.dc_tau_min + (args.dc_tau_max - args.dc_tau_min) * density
                tau_path = save_histogram(
                    values=tau_values,
                    output_path=per_seed_dir / f"seed_{seed}_tau_hist.png",
                    title=f"State-wise tau(s) histogram\n{args.env} / seed {seed}",
                    xlabel="State-wise expectile tau(s)",
                    bins=args.bins,
                    value_range=(args.dc_tau_min, args.dc_tau_max),
                )
                generated_files.append(str(tau_path))

    # Seed-averaged density.
    density_stack = np.stack([densities_by_seed[seed] for seed in loaded_seeds], axis=0)
    mean_density = np.mean(density_stack, axis=0).astype(np.float32)
    std_density = np.std(density_stack, axis=0).astype(np.float32)

    mean_stats = summarize_array(mean_density)
    std_stats = summarize_array(std_density)

    mean_summary_df = pd.DataFrame(
        [
            {
                "env": args.env,
                "seed_start": args.seed_start,
                "seed_end": args.seed_end,
                "loaded_seeds": ",".join(str(seed) for seed in loaded_seeds),
                "stat_source": "mean_density_across_seeds",
                **mean_stats,
            },
            {
                "env": args.env,
                "seed_start": args.seed_start,
                "seed_end": args.seed_end,
                "loaded_seeds": ",".join(str(seed) for seed in loaded_seeds),
                "stat_source": "std_density_across_seeds",
                **std_stats,
            },
        ]
    )
    mean_summary_path = save_path / "density_summary_seed_average.csv"
    mean_summary_df.to_csv(mean_summary_path, index=False)
    generated_files.append(str(mean_summary_path))

    mean_cache_path = save_path / "mean_density_cache.npz"
    np.savez_compressed(
        mean_cache_path,
        mean_density=mean_density,
        std_density=std_density,
        loaded_seeds=np.asarray(loaded_seeds, dtype=np.int32),
        env=args.env,
    )
    generated_files.append(str(mean_cache_path))

    mean_hist_path = save_histogram(
        values=mean_density,
        output_path=save_path / "mean_density_hist.png",
        title=f"Seed-averaged density confidence histogram\n{args.env}",
        xlabel="Seed-averaged density confidence c(s)",
        bins=args.bins,
        value_range=(0.0, 1.0),
    )
    mean_ecdf_path = save_ecdf(
        values=mean_density,
        output_path=save_path / "mean_density_ecdf.png",
        title=f"Seed-averaged density confidence ECDF\n{args.env}",
        xlabel="Seed-averaged density confidence c(s)",
        xlim=(0.0, 1.0),
    )
    std_hist_path = save_histogram(
        values=std_density,
        output_path=save_path / "std_density_across_seeds_hist.png",
        title=f"Density confidence standard deviation across seeds\n{args.env}",
        xlabel="Std. across seeds",
        bins=args.bins,
    )
    generated_files.extend([str(mean_hist_path), str(mean_ecdf_path), str(std_hist_path)])

    summary_plot_path = save_seed_summary_plot(
        summary_df=summary_df,
        env=args.env,
        save_path=save_path,
    )
    generated_files.append(str(summary_plot_path))

    if len(densities_by_seed) >= 2:
        boxplot_path = save_seed_boxplot(
            densities_by_seed=densities_by_seed,
            env=args.env,
            save_path=save_path,
            max_points_per_seed=args.max_points_per_seed,
            rng_seed=args.rng_seed,
        )
        if boxplot_path is not None:
            generated_files.append(str(boxplot_path))

        overlay_path = save_ecdf_overlay(
            densities_by_seed=densities_by_seed,
            env=args.env,
            save_path=save_path,
            max_points_per_seed=args.max_points_per_seed,
            rng_seed=args.rng_seed,
        )
        if overlay_path is not None:
            generated_files.append(str(overlay_path))

    if save_tau:
        mean_tau_values = args.dc_tau_min + (args.dc_tau_max - args.dc_tau_min) * mean_density
        mean_tau_path = save_histogram(
            values=mean_tau_values,
            output_path=save_path / "mean_tau_hist.png",
            title=f"Seed-averaged tau(s) histogram\n{args.env}",
            xlabel="Seed-averaged state-wise expectile tau(s)",
            bins=args.bins,
            value_range=(args.dc_tau_min, args.dc_tau_max),
        )
        generated_files.append(str(mean_tau_path))

    # 2D/3D state-distribution plots using observations and seed-averaged density.
    if not args.skip_state_plots:
        print("[dataset] Loading observations for state-distribution plots...")
        observations = load_env_dataset_observations(
            args.env,
            normalize=not args.no_normalize_observations,
        )

        if observations.shape[0] != mean_density.shape[0]:
            raise ValueError(
                "Dataset observation count does not match density confidence length. "
                f"observations={observations.shape[0]}, density={mean_density.shape[0]}. "
                "Check that the density cache was produced for the same env/dataset/preprocessing."
            )

        plot_indices = sample_indices(
            n=observations.shape[0],
            max_points=args.max_state_plot_points,
            rng_seed=args.rng_seed,
        )
        sampled_obs = observations[plot_indices]
        sampled_mean_density = mean_density[plot_indices]
        sampled_std_density = std_density[plot_indices]

        n_pca_components = min(3, sampled_obs.shape[1])
        projection = pca_projection_numpy(sampled_obs, n_components=n_pca_components)

        projection_csv_path = save_projection_csv(
            projection=projection,
            density_values=sampled_mean_density,
            indices=plot_indices,
            save_path=save_path,
            name="mean_density",
        )
        generated_files.append(str(projection_csv_path))

        if projection.shape[1] >= 2:
            plot_2d_path = save_state_distribution_2d(
                projection_2d=projection[:, :2],
                density_values=sampled_mean_density,
                env=args.env,
                save_path=save_path,
                name="mean_density",
            )
            generated_files.append(str(plot_2d_path))

            std_2d_path = save_state_distribution_2d(
                projection_2d=projection[:, :2],
                density_values=sampled_std_density,
                env=args.env,
                save_path=save_path,
                name="std_density_across_seeds",
            )
            generated_files.append(str(std_2d_path))

        if projection.shape[1] >= 3:
            plot_3d_path = save_state_distribution_3d(
                projection_3d=projection[:, :3],
                density_values=sampled_mean_density,
                env=args.env,
                save_path=save_path,
                name="mean_density",
            )
            generated_files.append(str(plot_3d_path))

        if args.per_seed_state_plots and projection.shape[1] >= 2:
            for seed in loaded_seeds:
                sampled_seed_density = densities_by_seed[seed][plot_indices]
                seed_2d_path = save_state_distribution_2d(
                    projection_2d=projection[:, :2],
                    density_values=sampled_seed_density,
                    env=args.env,
                    save_path=save_path,
                    name=f"seed_{seed}",
                )
                generated_files.append(str(seed_2d_path))

    generated_txt_path = save_path / "generated_files.txt"
    generated_txt_path.write_text("\n".join(generated_files) + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"Save directory: {save_path}")
    print(f"Loaded seeds: {loaded_seeds}")
    print(f"Saved generated file list: {generated_txt_path}")

    print("\nSeed-wise density summary:")
    print(
        summary_df[
            [
                "seed",
                "n",
                "mean",
                "std",
                "min",
                "p05",
                "median",
                "p95",
                "max",
            ]
        ].to_string(index=False)
    )

    print("\nSeed-averaged density summary:")
    print(mean_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
