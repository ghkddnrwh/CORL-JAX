"""Count goal labels and check transition alignment without changing training.

Goal-frame frequency is not a certificate of behavior coverage or task
solvability. The official regular loader removes invalid last-state actions;
time-limit transitions still have a valid separately stored next observation.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from utils.propagation_research.reward_bounds import REWARD_COMPONENTS
from utils.propagation_research.summarize import ENVIRONMENTS


def describe(data):
    reward = np.asarray(data["rewards"]).reshape(-1)
    mask = np.asarray(data["masks"]).reshape(-1)
    terminal = np.asarray(data["terminals"]).reshape(-1)
    goal = reward == 0
    assert len(reward) == len(mask) == len(data["actions"]) == len(data["next_observations"])
    # At trajectory boundaries the next stored current state belongs to a new
    # episode. All other adjacent rows should agree exactly with next_obs.
    indices = np.flatnonzero(terminal[:-1] == 0)
    mismatch = 0
    for start in range(0, len(indices), 50000):
        ix = indices[start:start+50000]
        mismatch += int(np.count_nonzero(np.any(
            data["next_observations"][ix] != data["observations"][ix+1], axis=-1)))
    values, counts = np.unique(reward, return_counts=True)
    return {"rows": len(reward), "state_dim": data["observations"].shape[-1],
            "action_dim": data["actions"].shape[-1], "zero_reward_goal_frames": int(goal.sum()),
            "zero_reward_goal_fraction": float(goal.mean()), "zero_bootstrap_mask_frames": int((mask == 0).sum()),
            "goal_and_zero_mask_disagreements": int(np.count_nonzero(goal != (mask == 0))),
            "original_trajectory_end_flags": int(np.count_nonzero(terminal)),
            "adjacent_non_end_transition_pairs_checked": len(indices),
            "adjacent_non_end_transition_mismatches": mismatch,
            "compact_valids_field_present": "valids" in data,
            "reward_counts": {str(float(v)): int(c) for v, c in zip(values, counts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench
    import ogbench.utils

    results = []
    for env_name in ENVIRONMENTS:
        env, train, validation = ogbench.make_env_and_datasets(env_name)
        env.close()
        row = {"environment": env_name, "reward_components": REWARD_COMPONENTS[env_name],
               "train": describe(train), "validation": describe(validation)}
        assert min(train["rewards"]) >= -REWARD_COMPONENTS[env_name] and max(train["rewards"]) <= 0
        results.append(row)
        print(env_name, "goal frames", row["train"]["zero_reward_goal_frames"],
              "alignment mismatches", row["train"]["adjacent_non_end_transition_mismatches"], flush=True)
        del train, validation
    result = {"time": time.time(), "environments": results,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ogbench_loader_sha256": hashlib.sha256(Path(ogbench.utils.__file__).read_bytes()).hexdigest(),
        "caveats": ["Goal frames count stored relabeled states, not independent successful trajectories.",
            "Few or absent goal frames can limit direct goal anchors but do not prove task impossibility.",
            "A positive number of goal examples does not guarantee sufficient conditional action coverage or an accurate learned policy.",
            "This descriptive data audit does not isolate a cause of the remaining policy failures."]}
    output = args.root/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output/"dataset_goal_profile.json").write_text(json.dumps(result, indent=2))
    rows = [{"environment": e["environment"], "split": split,
             **{k: v for k, v in e[split].items() if not isinstance(v, dict)}}
            for e in results for split in ("train", "validation")]
    with (output/"dataset_goal_profile.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"dataset_profile_source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
