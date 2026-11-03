"""Conservative reward bounds for the requested default OGBench tasks.

OGBench relabel_utils.py uses success-1 for navigation and
sum(component_successes)-component_count for manipulation. These are reward
bounds, not oracle values or conditional-residual labels. Puzzle-4x4 has 16
components even though this dataset only exhibits reward as low as -15.
"""

REWARD_COMPONENTS = {
    "antmaze-large-navigate-singletask-v0": 1,
    "humanoidmaze-medium-navigate-singletask-v0": 1,
    "humanoidmaze-large-navigate-singletask-v0": 1,
    "antsoccer-arena-navigate-singletask-v0": 1,
    "cube-single-play-singletask-v0": 1,
    "cube-double-play-singletask-v0": 2,
    "puzzle-3x3-play-singletask-v0": 9,
    "puzzle-4x4-play-singletask-v0": 16,
    "scene-play-singletask-v0": 5,
}


def return_bounds(environment, discount, reward_min, reward_max, source="empirical"):
    if source == "ogbench":
        lower, upper = -REWARD_COMPONENTS[environment], 0.
        if not lower <= reward_min <= reward_max <= upper:
            raise ValueError("Observed rewards contradict the declared OGBench reward bounds")
    elif source == "empirical":
        lower, upper = reward_min, reward_max
    else:
        raise ValueError(f"Unknown reward bound source: {source}")
    return min(lower, 0.)/(1-discount), max(upper, 0.)/(1-discount)


def effective_backup_rule(configuration, dataset):
    if not configuration["identity"]["profile"].get("bound_backup"):
        return "none"
    cfg = configuration["identity"]["config"]
    if cfg["env"] not in REWARD_COMPONENTS:
        return "empirical_reward_bounds"
    known = return_bounds(cfg["env"], cfg["discount"], dataset["reward_min"],
                          dataset["reward_max"], source="ogbench")
    # The source label differs across legacy runs, but identical numeric bounds
    # enter the exact same update. Keep the unequal Puzzle-4x4 interval separate.
    actual = (dataset["return_lower"], dataset["return_upper"])
    return "known_reward_bounds" if actual == known else "empirical_reward_bounds"
