import os
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt

ENTITY = "ukjo19"

METRIC = "v_mean"

# =========================
# Hyperparameters
# =========================

# TARGET_ENV = "antsoccer"
TARGET_ENV = "scene"

HISTORY_SAMPLES = 300
COMMON_GRID_SIZE = 1200

USE_SYMLOG = True
Y_LINTHRESH = 1e3
Y_LINSCALE = 3.0

BASE_SAVE_DIR = os.path.join("plots", "robut_learning", "wandb_v_mean_plots")

# =========================
# Run IDs
# =========================

ENV_RUN_GROUPS = {
    "scene": {
        "IQL-gamma-0.99": {
            "algorithm_name": "IQL",
            "gamma": "0.99",
            "project": "ORL-SMOOTH",
            "run_ids": [
                "9c661b38-8147-468c-8881-e7591ba416cd",
                "c97b018c-e54a-454d-aa35-b8cbb1b03819",
                "8c3384a1-6f82-4eb8-ba67-80f43e21bd58",
                "44078d21-f672-41e7-8bc9-7fc5d7d4409a",
            ],
        },


        "DD-IQL-gamma-0.99": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.99",
            "project": "ORL-BIAS",
            "run_ids": [
                "15383621-b7b2-4d29-bf9a-81aa2d91e49e",
                "1ab10d9f-9219-494f-9151-3e3f4c7dac26",
                "d9cf8520-b818-451e-9971-b906c8b538fb",
            ],
        },

        "IQL-gamma-0.999": {
            "algorithm_name": "IQL",
            "gamma": "0.999",
            "project": "ORL-SMOOTH",
            "run_ids": [
                "b305334b-2cfb-43a8-a81d-0271453bf471",
                "dec74135-b1a6-4c21-a994-9ad28f21ed86",
                "252461ca-8f7a-4d39-9467-3e621888681c",
                "e8e697e0-1793-4cf8-a243-4968f6b7714c",
            ],
        },
        "DD-IQL-gamma-0.999": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.999",
            "project": "ORL-BIAS",
            "run_ids": [
                "70ddd5f9-edf4-49c7-ba1b-865f400655b0",
                "abbf6c39-8d79-4e33-94f2-66504652db0c",
                "18fdc52b-d6e5-478a-b676-f4f4a80005d2",
            ],
        },

    },

    "antsoccer": {


        "IQL-gamma-0.995": {
            "algorithm_name": "IQL",
            "gamma": "0.995",
            "project": "ORL-SMOOTH",
            "run_ids": [
                "203ea79c-5c0b-40b9-8265-c9d18d407a4f",
                "72a698b8-8cee-4da4-a9c3-96e8fe6fe9d9",
                "a6158fab-d446-4f4a-9fe8-6cf9c6083f39",
                "954d0c21-cdbd-463e-9c3c-18cc84e3fe13",
            ],
        },
        "DD-IQL-gamma-0.995": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.995",
            "project": "ORL-BIAS",
            "run_ids": [
                "596f8ddf-62a8-4fd2-ab6c-03c2954e09f2",
                "eb17c33f-a8fb-4cb5-be49-1a579445034b",
                "a40c9332-f544-4afb-88ac-14175cb2d036",
                "9ac43a3a-9e5e-4def-8b64-805a036cf9b9",
            ],
        },

        "IQL-gamma-0.999": {
            "algorithm_name": "IQL",
            "gamma": "0.999",
            "project": "ORL-SMOOTH",
            "run_ids": [
                "42ed45ad-9d90-4a0e-ab64-2b266ed13cda",
                "7e2d3a01-fea2-457c-be2b-4c2c5e28f7bf",
                "7ec2e88a-4a88-49f0-897f-6e744c464bc5",
                "fa857df6-b861-480b-8456-944110517bf3",
            ],
        },
        "DD-IQL-gamma-0.999": {
            "algorithm_name": "DD-IQL",
            "gamma": "0.999",
            "project": "ORL-BIAS",
            "run_ids": [
                "a21cb767-f2bc-48e6-b915-3e9d6e59e74e",
                "2f158076-4057-497a-9fd7-11a689a451ee",
                "7f513092-e26c-499e-a9a4-d0799de2ae3b",
                "9e06886b-2678-43d4-a288-e2e8d8b83143",
            ],
        },
    },
}

if TARGET_ENV not in ENV_RUN_GROUPS:
    raise ValueError(
        f"Unknown TARGET_ENV: {TARGET_ENV}. "
        f"Choose one of {list(ENV_RUN_GROUPS.keys())}."
    )

RUN_GROUPS = ENV_RUN_GROUPS[TARGET_ENV]

SAVE_DIR = os.path.join(BASE_SAVE_DIR, TARGET_ENV)
os.makedirs(SAVE_DIR, exist_ok=True)

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

api = wandb.Api()


def load_metric_from_run(run, project, metric):
    cache_path = os.path.join(
        CACHE_DIR,
        f"{TARGET_ENV}_{project}_{run.id}_{metric}_samples{HISTORY_SAMPLES}.csv"
    )

    if os.path.exists(cache_path):
        print(f"    load cache: {cache_path}")
        return pd.read_csv(cache_path)

    df = run.history(
        samples=HISTORY_SAMPLES,
        keys=[metric],
        x_axis="_step",
        pandas=True,
    )

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    df = df[["_step", metric]].dropna()

    df = df.rename(columns={
        "_step": "step",
        metric: "value",
    })

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["step", "value"])

    df["run_id"] = run.id
    df["run_name"] = run.name
    df["project"] = project

    df = (
        df
        .groupby(["run_id", "run_name", "project", "step"], as_index=False)["value"]
        .mean()
        .sort_values("step")
    )

    if not df.empty:
        df.to_csv(cache_path, index=False)

    return df


def interpolate_group_runs(raw, group_name, project):
    run_dfs = []

    for run_id, run_df in raw.groupby("run_id"):
        run_df = run_df.sort_values("step")

        steps = run_df["step"].to_numpy()
        values = run_df["value"].to_numpy()

        if len(steps) < 2:
            continue

        run_dfs.append((run_id, steps, values))

    if not run_dfs:
        return None

    start_step = max(steps[0] for _, steps, _ in run_dfs)
    end_step = min(steps[-1] for _, steps, _ in run_dfs)

    if start_step >= end_step:
        print(f"[WARN] no overlapping step range for group: {group_name}")
        return None

    common_steps = np.linspace(start_step, end_step, COMMON_GRID_SIZE)

    interpolated = []

    for run_id, steps, values in run_dfs:
        interp_values = np.interp(common_steps, steps, values)

        tmp = pd.DataFrame({
            "group": group_name,
            "project": project,
            "run_id": run_id,
            "step": common_steps,
            "value": interp_values,
        })

        interpolated.append(tmp)

    interpolated = pd.concat(interpolated, ignore_index=True)

    stats = (
        interpolated
        .groupby(["group", "project", "step"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("step")
    )

    stats["std"] = stats["std"].fillna(0.0)

    return stats


def compute_group_stats(group_name, info):
    project = info["project"]
    run_ids = info["run_ids"]

    dfs = []

    print(f"\nLoading env: {TARGET_ENV}")
    print(f"Loading group: {group_name}")
    print(f"Project: {project}")

    for run_id in run_ids:
        run_id = run_id.strip()

        run_path = f"{ENTITY}/{project}/{run_id}"
        run = api.run(run_path)

        print(f"  run: {run.id} | {run.name}")

        df = load_metric_from_run(run, project, METRIC)

        if df.empty:
            print(f"  [WARN] no {METRIC} found in run: {run.id} | {run.name}")
            continue

        df["group"] = group_name
        dfs.append(df)

    if not dfs:
        print(f"[WARN] no valid runs for group: {group_name}")
        return None

    raw = pd.concat(dfs, ignore_index=True)

    stats = interpolate_group_runs(
        raw=raw,
        group_name=group_name,
        project=project,
    )

    return stats


all_stats = []

for group_name, info in RUN_GROUPS.items():
    stats = compute_group_stats(group_name, info)

    if stats is not None:
        all_stats.append(stats)

if not all_stats:
    raise RuntimeError(f"No valid data found for metric: {METRIC}")

all_stats = pd.concat(all_stats, ignore_index=True)

scale_name = "symlog" if USE_SYMLOG else "linear"

csv_path = os.path.join(
    SAVE_DIR,
    f"{TARGET_ENV}_{METRIC}_mean_std_{scale_name}.csv"
)

fig_path = os.path.join(
    SAVE_DIR,
    f"{TARGET_ENV}_{METRIC}_mean_std_{scale_name}.png"
)

all_stats.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize=(10, 6))

for group_name, info in RUN_GROUPS.items():
    stats = all_stats[all_stats["group"] == group_name]

    if stats.empty:
        continue

    x = stats["step"].to_numpy()
    mean = stats["mean"].to_numpy()
    std = stats["std"].to_numpy()

    lower = mean - std
    upper = mean + std

    label = rf"{info['algorithm_name']}($\gamma$ = {info['gamma']})"

    line = ax.plot(
        x,
        mean,
        label=label,
        linewidth=2.0,
    )[0]

    color = line.get_color()

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=0.12,
        color=color,
        linewidth=0,
    )

ax.set_xlabel("step")
ax.set_ylabel("V")

if USE_SYMLOG:
    ax.set_yscale(
        "symlog",
        linthresh=Y_LINTHRESH,
        linscale=Y_LINSCALE,
        base=10,
    )

ax.legend(frameon=True)

ax.grid(True, alpha=0.28, which="major")
ax.grid(True, alpha=0.12, which="minor")

fig.tight_layout()
fig.savefig(fig_path, dpi=300)
plt.close(fig)

print(f"\nSaved CSV: {csv_path}")
print(f"Saved figure: {fig_path}")

print("\n===== Final timestep summary =====")

for group_name, info in RUN_GROUPS.items():
    stats = all_stats[all_stats["group"] == group_name].sort_values("step")

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