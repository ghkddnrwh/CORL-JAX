import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


# ============================================================
# Publication-style plotting defaults
# ============================================================
# Reference style:
# - Times New Roman
# - STIX math font
# - Paper-style font sizes
# - No unicode-minus issue
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",

    "font.size": 20,
    "axes.labelsize": 20,

    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "legend.fontsize": 16,

    "axes.unicode_minus": False,
})


# ============================================================
# Basic settings
# ============================================================

ENTITY = "ukjo19"

METRIC = "v_mean"


# ============================================================
# Hyperparameters
# ============================================================

# TARGET_ENV = "antsoccer"
TARGET_ENV = "scene"

# Number of points requested from W&B history
HISTORY_SAMPLES = 300

# Number of interpolation points shared across runs
COMMON_GRID_SIZE = 1200


# ============================================================
# Y-axis settings
# ============================================================

USE_SYMLOG = True

# Linear region around zero when symlog is used
Y_LINTHRESH = 1e3

# Visual width of the linear region
Y_LINSCALE = 3.0


# ============================================================
# Save directory
# ============================================================

BASE_SAVE_DIR = os.path.join(
    "plots",
    "iclr2027",
    "wandb_v_mean_plots",
)


# ============================================================
# Run IDs
# ============================================================

ENV_RUN_GROUPS = {
    "scene": {
        "IQL-gamma-0.99": {
            "algorithm_name": "IQL",
            "gamma": "0.99",
            "project": "ORL-SMOOTH",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "390e8fc2-35d0-4edd-b653-c4cc238e3d32",
                "6c9ae2f1-f5da-419b-90c5-1ac35c562523",
                "863b6cd9-2aa4-4a57-a861-2682c876dfd1",
                "4a84c785-236d-4850-9a67-158fbe676dcd",
            ],
        },

        "DD-IQL-gamma-0.99": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.99",
            "project": "ORL-BIAS",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "01b18dfd-f2ce-4a79-aaa6-54b2805b9c26",
                "8036caeb-8e93-466e-9477-b265e5fc4189",
                "f3ece97d-4440-4b9e-9cdc-0a6dc4097f17",
                "aa68ed5c-1603-41d3-9629-badb655fde0a",
            ],
        },

        "IQL-gamma-0.999": {
            "algorithm_name": "IQL",
            "gamma": "0.999",
            "project": "ORL-SMOOTH",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "8f8cb938-448d-4790-8ad8-b25001d38e55",
                "1432e302-e1bc-459b-a460-a09e378862c3",
                "bef798e1-a4d3-43bb-93b3-d8a6338fc031",
                "5e671ea6-4b49-49e4-9ec6-9854139887ea",
            ],
        },

        "DD-IQL-gamma-0.999": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.999",
            "project": "ORL-BIAS",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "d70575e9-40c0-4100-9a04-eb318ffc18c4",
                "7481881b-28fd-434e-9cce-2734130c15b7",
                "471e9bd1-002a-4f6b-bdf5-4573de298105",
                "a8f7c1d9-4de1-4a07-81f4-a09455c93c14",
            ],
        },
    },

    "antsoccer": {
        "IQL-gamma-0.995": {
            "algorithm_name": "IQL",
            "gamma": "0.995",
            "project": "ORL-SMOOTH",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "3f232fe1-c544-4ec3-ba28-c110e8a4d772",
                "1ce0dc63-1b29-4c56-83fd-95f068d34ac7",
                "9395278f-3ecd-4396-9e38-1c3580e72056",
                "5cac61c7-f2f7-48c6-855f-5be2b9d86062",
            ],
        },

        "DD-IQL-gamma-0.995": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.995",
            "project": "ORL-BIAS",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "31a3c37c-c579-40d8-a654-d4fb4a25872a",
                "93351066-5480-4464-849f-b2b37d6108b0",
                "b9420265-a754-47be-9553-68f3aff8395d",
                "d2147ecd-db56-4c18-8b42-c15ed1a0b6f3",
            ],
        },

        "IQL-gamma-0.999": {
            "algorithm_name": "IQL",
            "gamma": "0.999",
            "project": "ORL-SMOOTH",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "f2cef72e-59a8-487d-bd7d-4ce99e9c191f",
                "d6903e07-065e-4ea8-bbe8-78936fa85a4a",
                "82cb95fc-4f7b-440b-a147-4b1419ef4e29",
                "793d2833-4be1-4692-a4ec-a0094df8ab9f",
            ],
        },

        "DD-IQL-gamma-0.999": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.999",
            "project": "ORL-BIAS",
            "run_ids": [
                # seeds 0, 1, 2, 3
                "654b05e2-c394-4336-8259-7e01e0666362",
                "d5904c54-b63e-412f-8686-f5704e12e8f4",
                "082582d1-9507-484e-9ea7-13e39a80a877",
                "d87a70ec-9fdd-4f85-8cc9-77ad1cc7aa18",
            ],
        },
    },
}


# ============================================================
# Validate environment
# ============================================================

if TARGET_ENV not in ENV_RUN_GROUPS:
    raise ValueError(
        f"Unknown TARGET_ENV: {TARGET_ENV}. "
        f"Choose one of {list(ENV_RUN_GROUPS.keys())}."
    )

RUN_GROUPS = ENV_RUN_GROUPS[TARGET_ENV]


# ============================================================
# Output directories
# ============================================================

SAVE_DIR = os.path.join(
    BASE_SAVE_DIR,
    TARGET_ENV,
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


CACHE_DIR = os.path.join(
    SAVE_DIR,
    "cache",
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True,
)


# ============================================================
# W&B API
# ============================================================

api = wandb.Api()


# ============================================================
# Load metric from one W&B run
# ============================================================

def load_metric_from_run(
    run,
    project,
    metric,
):
    """
    Load one metric from one W&B run.

    If a cached CSV exists, use it instead of downloading
    the history again.
    """

    cache_path = os.path.join(
        CACHE_DIR,
        (
            f"{TARGET_ENV}_"
            f"{project}_"
            f"{run.id}_"
            f"{metric}_"
            f"samples{HISTORY_SAMPLES}.csv"
        ),
    )

    # --------------------------------------------------------
    # Load cache
    # --------------------------------------------------------
    if os.path.exists(cache_path):
        print(f"    load cache: {cache_path}")
        return pd.read_csv(cache_path)

    # --------------------------------------------------------
    # Download from W&B
    # --------------------------------------------------------
    df = run.history(
        samples=HISTORY_SAMPLES,
        keys=[metric],
        x_axis="_step",
        pandas=True,
    )

    # --------------------------------------------------------
    # Check whether the requested metric exists
    # --------------------------------------------------------
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Keep only step and metric
    # --------------------------------------------------------
    df = df[
        [
            "_step",
            metric,
        ]
    ].dropna()

    df = df.rename(
        columns={
            "_step": "step",
            metric: "value",
        }
    )

    # --------------------------------------------------------
    # Convert to numeric values
    # --------------------------------------------------------
    df["step"] = pd.to_numeric(
        df["step"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "step",
            "value",
        ]
    )

    # --------------------------------------------------------
    # Store run metadata
    # --------------------------------------------------------
    df["run_id"] = run.id
    df["run_name"] = run.name
    df["project"] = project

    # --------------------------------------------------------
    # Handle possible duplicated steps
    # --------------------------------------------------------
    df = (
        df
        .groupby(
            [
                "run_id",
                "run_name",
                "project",
                "step",
            ],
            as_index=False,
        )["value"]
        .mean()
        .sort_values("step")
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------
    if not df.empty:
        df.to_csv(
            cache_path,
            index=False,
        )

    return df


# ============================================================
# Interpolate runs onto a common step grid
# ============================================================

def interpolate_group_runs(
    raw,
    group_name,
    project,
):
    """
    Interpolate all runs in one algorithm group onto
    a shared x-axis.

    The shared range is restricted to the overlapping
    step range across all valid runs.
    """

    run_dfs = []

    # --------------------------------------------------------
    # Collect each run
    # --------------------------------------------------------
    for run_id, run_df in raw.groupby("run_id"):
        run_df = run_df.sort_values("step")

        steps = run_df["step"].to_numpy()
        values = run_df["value"].to_numpy()

        # np.interp requires at least two points
        if len(steps) < 2:
            continue

        run_dfs.append(
            (
                run_id,
                steps,
                values,
            )
        )

    if not run_dfs:
        return None

    # --------------------------------------------------------
    # Compute common overlapping range
    # --------------------------------------------------------
    start_step = max(
        steps[0]
        for _, steps, _ in run_dfs
    )

    end_step = min(
        steps[-1]
        for _, steps, _ in run_dfs
    )

    if start_step >= end_step:
        print(
            f"[WARN] no overlapping step range "
            f"for group: {group_name}"
        )
        return None

    # --------------------------------------------------------
    # Common interpolation grid
    # --------------------------------------------------------
    common_steps = np.linspace(
        start_step,
        end_step,
        COMMON_GRID_SIZE,
    )

    interpolated = []

    # --------------------------------------------------------
    # Interpolate each run
    # --------------------------------------------------------
    for run_id, steps, values in run_dfs:
        interp_values = np.interp(
            common_steps,
            steps,
            values,
        )

        tmp = pd.DataFrame({
            "group": group_name,
            "project": project,
            "run_id": run_id,
            "step": common_steps,
            "value": interp_values,
        })

        interpolated.append(tmp)

    interpolated = pd.concat(
        interpolated,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Mean / std across runs
    # --------------------------------------------------------
    stats = (
        interpolated
        .groupby(
            [
                "group",
                "project",
                "step",
            ]
        )["value"]
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
        .reset_index()
        .sort_values("step")
    )

    # If only one run contributes,
    # pandas std is NaN.
    stats["std"] = stats["std"].fillna(0.0)

    return stats


# ============================================================
# Compute statistics for one group
# ============================================================

def compute_group_stats(
    group_name,
    info,
):
    """
    Download/load all runs belonging to one group and
    calculate the interpolated mean/std curve.
    """

    project = info["project"]
    run_ids = info["run_ids"]

    dfs = []

    print()
    print(f"Loading env: {TARGET_ENV}")
    print(f"Loading group: {group_name}")
    print(f"Project: {project}")

    # --------------------------------------------------------
    # Load individual runs
    # --------------------------------------------------------
    for run_id in run_ids:
        run_id = run_id.strip()

        run_path = (
            f"{ENTITY}/"
            f"{project}/"
            f"{run_id}"
        )

        run = api.run(run_path)

        print(
            f"  run: {run.id} | {run.name}"
        )

        df = load_metric_from_run(
            run=run,
            project=project,
            metric=METRIC,
        )

        if df.empty:
            print(
                f"  [WARN] no {METRIC} found in run: "
                f"{run.id} | {run.name}"
            )
            continue

        df["group"] = group_name

        dfs.append(df)

    # --------------------------------------------------------
    # No valid runs
    # --------------------------------------------------------
    if not dfs:
        print(
            f"[WARN] no valid runs for group: "
            f"{group_name}"
        )
        return None

    raw = pd.concat(
        dfs,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Interpolate and aggregate
    # --------------------------------------------------------
    stats = interpolate_group_runs(
        raw=raw,
        group_name=group_name,
        project=project,
    )

    return stats


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Collect all group statistics
    # ========================================================

    all_stats_list = []

    for group_name, info in RUN_GROUPS.items():

        stats = compute_group_stats(
            group_name=group_name,
            info=info,
        )

        if stats is not None:
            all_stats_list.append(stats)

    if not all_stats_list:
        raise RuntimeError(
            f"No valid data found for metric: {METRIC}"
        )

    all_stats = pd.concat(
        all_stats_list,
        ignore_index=True,
    )


    # ========================================================
    # Output paths
    # ========================================================

    scale_name = (
        "symlog"
        if USE_SYMLOG
        else "linear"
    )

    csv_path = os.path.join(
        SAVE_DIR,
        (
            f"{TARGET_ENV}_"
            f"{METRIC}_"
            f"mean_std_"
            f"{scale_name}.csv"
        ),
    )

    fig_path = os.path.join(
        SAVE_DIR,
        (
            f"{TARGET_ENV}_"
            f"{METRIC}_"
            f"mean_std_"
            f"{scale_name}.png"
        ),
    )


    # ========================================================
    # Save statistics CSV
    # ========================================================

    all_stats.to_csv(
        csv_path,
        index=False,
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for group_name, info in RUN_GROUPS.items():

        stats = all_stats[
            all_stats["group"] == group_name
        ].sort_values("step")

        if stats.empty:
            continue

        # ----------------------------------------------------
        # Extract curve
        # ----------------------------------------------------
        x = stats["step"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = stats["std"].to_numpy()

        lower = mean - std
        upper = mean + std

        # ----------------------------------------------------
        # Legend label
        #
        # Example:
        # IQL (γ = 0.99)
        # DD-IQL (γ = 0.999)
        # ----------------------------------------------------
        label = (
            rf"{info['algorithm_name']} "
            rf"($\gamma$ = {info['gamma']})"
        )

        # ----------------------------------------------------
        # Mean curve
        # ----------------------------------------------------
        line = ax.plot(
            x,
            mean,
            label=label,
            linewidth=2.5,
        )[0]

        # Use exactly the same color for the std band
        color = line.get_color()

        # ----------------------------------------------------
        # Mean ± std region
        # ----------------------------------------------------
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.15,
            linewidth=0,
        )


    # ========================================================
    # Axis labels
    # ========================================================

    ax.set_xlabel("Step")

    # Math-style V using STIX
    ax.set_ylabel(r"$V$")


    # ========================================================
    # Y scale
    # ========================================================

    if USE_SYMLOG:
        ax.set_yscale(
            "symlog",
            linthresh=Y_LINTHRESH,
            linscale=Y_LINSCALE,
            base=10,
        )


    # ========================================================
    # Legend
    # ========================================================

    # Paper-style legend without bounding box
    ax.legend(
        frameon=False,
    )


    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        True,
        alpha=0.3,
    )


    # ========================================================
    # Layout
    # ========================================================

    # Reduce unnecessary outer whitespace
    fig.tight_layout(
        pad=0.15,
        h_pad=0.15,
        w_pad=0.15,
    )


    # ========================================================
    # Save figure
    # ========================================================

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


    # ========================================================
    # Print saved paths
    # ========================================================

    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


    # ========================================================
    # Final timestep summary
    # ========================================================

    print()
    print("===== Final timestep summary =====")

    for group_name, info in RUN_GROUPS.items():

        stats = all_stats[
            all_stats["group"] == group_name
        ].sort_values("step")

        if stats.empty:
            continue

        final_row = stats.iloc[-1]

        algorithm_name = info["algorithm_name"]
        gamma = info["gamma"]

        final_step = final_row["step"]
        final_mean = final_row["mean"]
        final_std = final_row["std"]
        final_count = int(final_row["count"])

        print(
            f"{TARGET_ENV} | "
            f"{algorithm_name}(gamma = {gamma}) | "
            f"step = {final_step:.0f} | "
            f"{METRIC} mean = {final_mean:.6g} | "
            f"std = {final_std:.6g} | "
            f"n = {final_count}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()