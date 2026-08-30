import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Grid World
# ============================================================

# 0: left, 1: right, 2: up, 3: down
ACTIONS = [
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
]


def env_step(
    state,
    action,
    grid_height,
    grid_width,
    wall_reward=-1.0,
    step_reward=0.0,
    goal_reward=1.0,
):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < grid_height and 0 <= nc < grid_width):
        return state, wall_reward, False

    next_state = (nr, nc)
    goal = (0, grid_width - 1)

    if next_state == goal:
        return next_state, goal_reward, True

    return next_state, step_reward, False


# ============================================================
# 2. Exact Optimal Q* and V*
# ============================================================

def get_optimal_q(
    grid_height,
    grid_width,
    gamma,
    wall_reward=-1.0,
    step_reward=0.0,
    goal_reward=1.0,
):
    # With non-positive step/wall rewards and a positive terminal reward,
    # the shortest path to the goal is optimal.
    if step_reward > 0 or wall_reward > 0 or goal_reward <= 0:
        raise ValueError(
            "Analytic Q* assumes STEP_REWARD <= 0, WALL_REWARD <= 0, "
            "and GOAL_REWARD > 0."
        )

    goal = (0, grid_width - 1)
    V_star = np.zeros((grid_height, grid_width))

    for r in range(grid_height):
        for c in range(grid_width):
            if (r, c) == goal:
                continue

            distance = r + (grid_width - 1 - c)

            # distance-1 ordinary moves, then one terminal goal move.
            if distance == 1:
                V_star[r, c] = goal_reward
            elif gamma == 1.0:
                V_star[r, c] = step_reward * (distance - 1) + goal_reward
            else:
                step_return = step_reward * (1.0 - gamma ** (distance - 1)) / (1.0 - gamma)
                V_star[r, c] = step_return + (gamma ** (distance - 1)) * goal_reward

    Q_star = np.zeros((grid_height, grid_width, 4))

    for r in range(grid_height):
        for c in range(grid_width):
            state = (r, c)

            if state == goal:
                continue

            for action in range(4):
                next_state, reward, done = env_step(
                    state,
                    action,
                    grid_height,
                    grid_width,
                    wall_reward=wall_reward,
                    step_reward=step_reward,
                    goal_reward=goal_reward,
                )

                if done:
                    Q_star[r, c, action] = reward
                else:
                    Q_star[r, c, action] = reward + gamma * V_star[next_state]

    return Q_star, V_star


# ============================================================
# 3. epsilon-greedy
# ============================================================

def choose_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(4)
    return np.argmax(q_values)


# ============================================================
# 4. Training
# ============================================================

def train(
    method,
    grid_height=5,
    grid_width=5,
    num_updates=100_000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    behavior_sigma=0.1,
    bootstrap_sigma=0.1,
    wall_reward=-1.0,
    step_reward=0.0,
    goal_reward=1.0,
    max_dim_multiplier=1,
    dd_num_q=2,
    dd_delay_steps=1,
    log_every=500,
    seed=0,
):
    if max_dim_multiplier < 1:
        raise ValueError("max_dim_multiplier must be >= 1")

    if method == "dd_q":
        if dd_num_q < 2:
            raise ValueError("DD-Q requires dd_num_q >= 2")
        if dd_delay_steps < 1:
            raise ValueError("DD-Q requires dd_delay_steps >= 1")

    rng = np.random.default_rng(seed)

    start = (grid_height - 1, 0)
    goal = (0, grid_width - 1)

    Q_star, V_star = get_optimal_q(
        grid_height,
        grid_width,
        gamma,
        wall_reward=wall_reward,
        step_reward=step_reward,
        goal_reward=goal_reward,
    )

    q_shape = (grid_height, grid_width, 4)

    # --------------------------------------------------------
    # Q tables
    # --------------------------------------------------------
    if method == "q_learning":
        Q = np.zeros(q_shape)

    elif method in ("double_q", "clipped_double_q"):
        Q1 = np.zeros(q_shape)
        Q2 = np.zeros(q_shape)

    elif method == "dd_q":
        # online Q ensemble: [N, H, W, 4]
        Qs = np.zeros((dd_num_q, *q_shape))

        # delayed snapshot used only for selection
        Q_delayed = Qs.copy()

        # For a D-step block, member i uses member (i + shift) % N.
        # shift cycles 1, 2, ..., N-1, 1, ...
        peer_shift = 1

    else:
        raise ValueError("Unknown method")

    history = {
        "step": [],
        "V": [],
        "q_rmse": [],
        "v_rmse": [],
        "v_bias": [],
        "initial_v_error": [],
        "initial_v_squared_error": [],
    }

    mask = np.ones((grid_height, grid_width), dtype=bool)
    mask[goal] = False

    state = start

    # ========================================================
    # Main learning loop
    # ========================================================
    for t in range(1, num_updates + 1):

        # ====================================================
        # Q-learning
        # ====================================================
        if method == "q_learning":
            # Only the current state's behavior values need noise.
            Q_behavior_noisy = Q[state] + rng.normal(
                0, behavior_sigma, size=4
            )
            action = choose_action(Q_behavior_noisy, epsilon, rng)

            next_state, reward, done = env_step(
                state,
                action,
                grid_height,
                grid_width,
                wall_reward=wall_reward,
                step_reward=step_reward,
                goal_reward=goal_reward,
            )

            if done:
                target = reward
            else:
                # Only next-state bootstrap candidates are required: shape (4, K).
                Q_boot_noisy = Q[next_state][:, None] + rng.normal(
                    0,
                    bootstrap_sigma,
                    size=(4, max_dim_multiplier),
                )
                target = reward + gamma * np.max(Q_boot_noisy)

            Q[state][action] += alpha * (target - Q[state][action])

        # ====================================================
        # Double Q-learning
        # ====================================================
        elif method == "double_q":
            Q1_behavior_noisy = Q1[state] + rng.normal(
                0, behavior_sigma, size=4
            )
            Q2_behavior_noisy = Q2[state] + rng.normal(
                0, behavior_sigma, size=4
            )

            Q_behavior = (Q1_behavior_noisy + Q2_behavior_noisy) / 2
            action = choose_action(Q_behavior, epsilon, rng)

            next_state, reward, done = env_step(
                state,
                action,
                grid_height,
                grid_width,
                wall_reward=wall_reward,
                step_reward=step_reward,
                goal_reward=goal_reward,
            )

            if done:
                target1 = reward
                target2 = reward
            else:
                # Independent next-state noisy candidates only: shape (4, K).
                Q1_boot_noisy = Q1[next_state][:, None] + rng.normal(
                    0, bootstrap_sigma, size=(4, max_dim_multiplier)
                )
                Q2_boot_noisy = Q2[next_state][:, None] + rng.normal(
                    0, bootstrap_sigma, size=(4, max_dim_multiplier)
                )

                index1 = np.unravel_index(
                    np.argmax(Q1_boot_noisy), Q1_boot_noisy.shape
                )
                target1 = reward + gamma * Q2_boot_noisy[index1]

                index2 = np.unravel_index(
                    np.argmax(Q2_boot_noisy), Q2_boot_noisy.shape
                )
                target2 = reward + gamma * Q1_boot_noisy[index2]

            Q1[state][action] += alpha * (target1 - Q1[state][action])
            Q2[state][action] += alpha * (target2 - Q2[state][action])

        # ====================================================
        # Clipped Double Q-learning
        # ====================================================
        elif method == "clipped_double_q":
            Q1_behavior_noisy = Q1[state] + rng.normal(
                0, behavior_sigma, size=4
            )
            Q2_behavior_noisy = Q2[state] + rng.normal(
                0, behavior_sigma, size=4
            )

            Q_behavior = (Q1_behavior_noisy + Q2_behavior_noisy) / 2
            action = choose_action(Q_behavior, epsilon, rng)

            next_state, reward, done = env_step(
                state,
                action,
                grid_height,
                grid_width,
                wall_reward=wall_reward,
                step_reward=step_reward,
                goal_reward=goal_reward,
            )

            if done:
                target = reward
            else:
                Q1_boot_noisy = Q1[next_state][:, None] + rng.normal(
                    0, bootstrap_sigma, size=(4, max_dim_multiplier)
                )
                Q2_boot_noisy = Q2[next_state][:, None] + rng.normal(
                    0, bootstrap_sigma, size=(4, max_dim_multiplier)
                )

                clipped_candidates = np.minimum(
                    Q1_boot_noisy,
                    Q2_boot_noisy,
                )
                target = reward + gamma * np.max(clipped_candidates)

            Q1[state][action] += alpha * (target - Q1[state][action])
            Q2[state][action] += alpha * (target - Q2[state][action])

        # ====================================================
        # DD-Q-learning
        # ====================================================
        elif method == "dd_q":
            # Behavior policy uses only the current state from each online estimator.
            r, c = state
            Q_behavior_noisy = Qs[:, r, c, :] + rng.normal(
                0, behavior_sigma, size=(dd_num_q, 4)
            )
            Q_behavior = np.mean(Q_behavior_noisy, axis=0)
            action = choose_action(Q_behavior, epsilon, rng)

            next_state, reward, done = env_step(
                state,
                action,
                grid_height,
                grid_width,
                wall_reward=wall_reward,
                step_reward=step_reward,
                goal_reward=goal_reward,
            )

            if done:
                targets = np.full(dd_num_q, reward, dtype=float)
            else:
                # Only next-state candidates are generated: shape (N, 4, K).
                nr, nc = next_state
                Q_delayed_boot_noisy = Q_delayed[:, nr, nc, :, None] + rng.normal(
                    0,
                    bootstrap_sigma,
                    size=(dd_num_q, 4, max_dim_multiplier),
                )

                Q_online_boot_noisy = Qs[:, nr, nc, :, None] + rng.normal(
                    0,
                    bootstrap_sigma,
                    size=(dd_num_q, 4, max_dim_multiplier),
                )

                peer_indices = (
                    np.arange(dd_num_q) + peer_shift
                ) % dd_num_q

                targets = np.empty(dd_num_q, dtype=float)

                for i, j in enumerate(peer_indices):
                    selector_candidates = Q_delayed_boot_noisy[j]
                    selected_index = np.unravel_index(
                        np.argmax(selector_candidates),
                        selector_candidates.shape,
                    )

                    own_value = Q_online_boot_noisy[i][selected_index]
                    targets[i] = reward + gamma * own_value

            # Every Q_i is updated from the same environment transition.
            r, c = state
            Qs[:, r, c, action] += alpha * (
                targets - Qs[:, r, c, action]
            )

            # Refresh delayed tables at each D-step boundary.
            # The refreshed snapshot is used from the NEXT step onward.
            if t % dd_delay_steps == 0:
                Q_delayed = Qs.copy()

                if dd_num_q > 2:
                    peer_shift += 1
                    if peer_shift >= dd_num_q:
                        peer_shift = 1
                else:
                    peer_shift = 1

        # ====================================================
        # State transition
        # ====================================================
        state = start if done else next_state

        # ====================================================
        # Logging
        # ====================================================
        if t % log_every == 0:
            if method == "q_learning":
                Q_est = Q.copy()
            elif method in ("double_q", "clipped_double_q"):
                Q_est = (Q1 + Q2) / 2
            else:
                Q_est = np.mean(Qs, axis=0)

            # Evaluation uses only the 4 real actions.
            V_est = np.max(Q_est, axis=2)
            V_est[goal] = 0.0

            q_rmse = np.sqrt(np.mean((Q_est[mask] - Q_star[mask]) ** 2))
            v_rmse = np.sqrt(np.mean((V_est[mask] - V_star[mask]) ** 2))
            v_bias = np.mean(V_est[mask] - V_star[mask])

            initial_v_error = V_est[start] - V_star[start]
            initial_v_squared_error = initial_v_error ** 2

            history["step"].append(t)
            history["V"].append(V_est.copy())
            history["q_rmse"].append(q_rmse)
            history["v_rmse"].append(v_rmse)
            history["v_bias"].append(v_bias)
            history["initial_v_error"].append(initial_v_error)
            history["initial_v_squared_error"].append(initial_v_squared_error)

    for key in history:
        history[key] = np.array(history[key])

    # --------------------------------------------------------
    # Final Q tables
    # --------------------------------------------------------
    if method == "q_learning":
        history["Q_final"] = Q.copy()

    elif method in ("double_q", "clipped_double_q"):
        history["Q1_final"] = Q1.copy()
        history["Q2_final"] = Q2.copy()
        history["Q_final"] = (Q1 + Q2) / 2

    else:
        history["Q_ensemble_final"] = Qs.copy()
        history["Q_delayed_final"] = Q_delayed.copy()
        history["Q_final"] = np.mean(Qs, axis=0)

    return history


# ============================================================
# 5. Experiment parameters
# ============================================================

# Grid size
# A long, thin rectangle increases bootstrap depth with few states.
GRID_HEIGHT = 2
GRID_WIDTH = 150

NUM_UPDATES = 100_000

ALPHA = 0.1
GAMMA = 0.9999
EPSILON = 0.1

# Separate behavior noise from bootstrap noise.
# Small behavior noise keeps exploration/learning stable, while large bootstrap
# noise amplifies max/selection bias without increasing computational cost.
BEHAVIOR_SIGMA = 0.1
BOOTSTRAP_SIGMA = 10.0

# Reward shaping controls action gaps.
# More negative WALL_REWARD makes wrong wall selections much more costly,
# which can amplify Double-Q / DD-Q underestimation.
WALL_REWARD = -2.0
STEP_REWARD = 0.0
GOAL_REWARD = 1.0

# K=1 -> max over 4 candidates
# K=3 -> max over 12 candidates
# Noise is generated only for the next state, so large K is much cheaper now.
MAX_DIM_MULTIPLIER = 100

# DD-Q configurations
# Each tuple is (N, D):
#   N = number of online/delayed Q tables
#   D = delayed refresh / peer-cycle period
#
# Example below runs two separate DD-Q experiments:
#   DD-Q (N=2, D=1)
#   DD-Q (N=2, D=10)
DD_Q_CONFIGS = [
    (2, 1),
    (2, 10),
    (2, 100),
    (2, 1000),
]

LOG_EVERY = 500

# Run the same algorithms on 10 independent random seeds.
# Using the same seed set for every algorithm enables paired comparisons.
SEEDS = list(range(10))
STD_DDOF = 1  # sample standard deviation across seeds


# ============================================================
# Algorithms to run
# ============================================================

RUN_METHODS = [
    # "Q-learning",
    "Double Q",
    # "Clipped Double Q",
    "DD-Q-learning",
]


# ============================================================
# 6. Save paths
# ============================================================

SAVE_DIR = "./logs/iclr2027_dd_iql/tabular"

FIGURE_DIR = os.path.join(SAVE_DIR, "figures")
TABLE_DIR = os.path.join(SAVE_DIR, "tables")
NUMPY_DIR = os.path.join(SAVE_DIR, "numpy")


# ============================================================
# 7. Multi-seed aggregation / plotting helpers
# ============================================================

METRIC_KEYS = [
    "q_rmse",
    "v_rmse",
    "v_bias",
    "initial_v_error",
    "initial_v_squared_error",
]


def safe_result_name(name):
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(",", "")
    )


def aggregate_seed_histories(seed_histories, seeds, std_ddof=1):
    if len(seed_histories) == 0:
        raise ValueError("seed_histories must not be empty")
    if len(seed_histories) != len(seeds):
        raise ValueError("seed_histories and seeds must have the same length")

    reference_steps = seed_histories[0]["step"]
    for history in seed_histories[1:]:
        if not np.array_equal(history["step"], reference_steps):
            raise ValueError("All seeds must log at identical update steps")

    # ddof=1 is undefined for one seed, so fall back to 0 if needed.
    effective_ddof = std_ddof if len(seed_histories) > std_ddof else 0

    aggregated = {
        "step": reference_steps.copy(),
        "seeds": np.asarray(seeds, dtype=int),
        "num_seeds": len(seed_histories),
    }

    # Scalar learning-curve metrics: [seed, log_step]
    for key in METRIC_KEYS:
        values = np.stack([history[key] for history in seed_histories], axis=0)
        aggregated[f"{key}_all"] = values
        aggregated[f"{key}_mean"] = np.mean(values, axis=0)
        aggregated[f"{key}_std"] = np.std(values, axis=0, ddof=effective_ddof)

    # V histories: [seed, log_step, H, W]
    v_values = np.stack([history["V"] for history in seed_histories], axis=0)
    aggregated["V_mean"] = np.mean(v_values, axis=0)
    aggregated["V_std"] = np.std(v_values, axis=0, ddof=effective_ddof)

    # Final clean-Q estimates across seeds.
    q_final_values = np.stack([history["Q_final"] for history in seed_histories], axis=0)
    aggregated["Q_final_mean"] = np.mean(q_final_values, axis=0)
    aggregated["Q_final_std"] = np.std(q_final_values, axis=0, ddof=effective_ddof)

    # Keep method-specific final tables when available.
    for optional_key in (
        "Q1_final",
        "Q2_final",
        "Q_ensemble_final",
        "Q_delayed_final",
    ):
        if optional_key in seed_histories[0]:
            values = np.stack(
                [history[optional_key] for history in seed_histories],
                axis=0,
            )
            aggregated[f"{optional_key}_mean"] = np.mean(values, axis=0)
            aggregated[f"{optional_key}_std"] = np.std(
                values,
                axis=0,
                ddof=effective_ddof,
            )

    return aggregated


def save_plot(results, metric, ylabel, title, filename, figure_dir, zero_line=False):
    """Plot seed mean with a shaded +/- 1 standard-deviation band."""
    plt.figure(figsize=(7, 4))

    for name, history in results.items():
        steps = history["step"]
        mean = history[f"{metric}_mean"]
        std = history[f"{metric}_std"]

        line, = plt.plot(steps, mean, label=name)
        plt.fill_between(
            steps,
            mean - std,
            mean + std,
            alpha=0.20,
            color=line.get_color(),
        )

    if zero_line:
        plt.axhline(0, linestyle="--")

    plt.xlabel("Update step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(figure_dir, filename),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def save_seed_aggregate(name, history, table_dir, numpy_dir):
    """Save aggregated curves plus the raw per-seed metric curves."""
    safe_name = safe_result_name(name)

    # Aggregated V/Q tables.
    np.save(
        os.path.join(numpy_dir, f"{safe_name}_V_mean.npy"),
        history["V_mean"],
    )
    np.save(
        os.path.join(numpy_dir, f"{safe_name}_V_std.npy"),
        history["V_std"],
    )
    np.save(
        os.path.join(numpy_dir, f"{safe_name}_Q_final_mean.npy"),
        history["Q_final_mean"],
    )
    np.save(
        os.path.join(numpy_dir, f"{safe_name}_Q_final_std.npy"),
        history["Q_final_std"],
    )

    # Raw metric arrays have shape [num_seeds, num_log_points].
    for key in METRIC_KEYS:
        np.save(
            os.path.join(numpy_dir, f"{safe_name}_{key}_all_seeds.npy"),
            history[f"{key}_all"],
        )

    np.save(
        os.path.join(numpy_dir, f"{safe_name}_seeds.npy"),
        history["seeds"],
    )

    # Optional method-specific mean/std final tables.
    for optional_key in (
        "Q1_final",
        "Q2_final",
        "Q_ensemble_final",
        "Q_delayed_final",
    ):
        mean_key = f"{optional_key}_mean"
        std_key = f"{optional_key}_std"
        if mean_key in history:
            np.save(
                os.path.join(numpy_dir, f"{safe_name}_{mean_key}.npy"),
                history[mean_key],
            )
            np.save(
                os.path.join(numpy_dir, f"{safe_name}_{std_key}.npy"),
                history[std_key],
            )

    # Final V mean/std as CSV tables.
    np.savetxt(
        os.path.join(table_dir, f"{safe_name}_final_V_mean.csv"),
        history["V_mean"][-1],
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(table_dir, f"{safe_name}_final_V_std.csv"),
        history["V_std"][-1],
        delimiter=",",
        fmt="%.6f",
    )

    # Aggregated metric CSV: mean and std for every logged step.
    metric_path = os.path.join(table_dir, f"{safe_name}_metrics_mean_std.csv")
    with open(metric_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step"]
        for key in METRIC_KEYS:
            header.extend([f"{key}_mean", f"{key}_std"])
        writer.writerow(header)

        for idx, step in enumerate(history["step"]):
            row = [step]
            for key in METRIC_KEYS:
                row.extend([
                    history[f"{key}_mean"][idx],
                    history[f"{key}_std"][idx],
                ])
            writer.writerow(row)

    # Long-form per-seed CSV for downstream statistics / plotting.
    raw_path = os.path.join(table_dir, f"{safe_name}_metrics_all_seeds.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "step", *METRIC_KEYS])
        for seed_idx, seed in enumerate(history["seeds"]):
            for log_idx, step in enumerate(history["step"]):
                writer.writerow([
                    int(seed),
                    step,
                    *[
                        history[f"{key}_all"][seed_idx, log_idx]
                        for key in METRIC_KEYS
                    ],
                ])


def run_method_over_seeds(
    result_name,
    method,
    common_args,
    seeds,
    dd_num_q=None,
    dd_delay_steps=None,
):
    seed_histories = []
    total = len(seeds)

    for seed_idx, seed in enumerate(seeds, start=1):
        print(
            f"Training: {result_name} | "
            f"seed={seed} ({seed_idx}/{total})"
        )

        args = dict(common_args)
        args["seed"] = int(seed)

        if method == "dd_q":
            args["dd_num_q"] = int(dd_num_q)
            args["dd_delay_steps"] = int(dd_delay_steps)

        seed_histories.append(train(method, **args))

    return aggregate_seed_histories(
        seed_histories,
        seeds=seeds,
        std_ddof=STD_DDOF,
    )


# ============================================================
# 8. Main experiment
# ============================================================

def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(NUMPY_DIR, exist_ok=True)

    print("Save directory:", os.path.abspath(SAVE_DIR))
    print("Seeds:", SEEDS)

    common_args = dict(
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        num_updates=NUM_UPDATES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        behavior_sigma=BEHAVIOR_SIGMA,
        bootstrap_sigma=BOOTSTRAP_SIGMA,
        wall_reward=WALL_REWARD,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        max_dim_multiplier=MAX_DIM_MULTIPLIER,
        log_every=LOG_EVERY,
    )

    methods = {
        "Q-learning": "q_learning",
        "Double Q": "double_q",
        "Clipped Double Q": "clipped_double_q",
    }

    results = {}

    for name in RUN_METHODS:
        if name == "DD-Q-learning":
            for dd_num_q, dd_delay_steps in DD_Q_CONFIGS:
                if dd_num_q < 2:
                    raise ValueError(
                        f"Invalid DD-Q config {(dd_num_q, dd_delay_steps)}: N must be >= 2"
                    )
                if dd_delay_steps < 1:
                    raise ValueError(
                        f"Invalid DD-Q config {(dd_num_q, dd_delay_steps)}: D must be >= 1"
                    )

                result_name = f"DD-Q (N={dd_num_q}, D={dd_delay_steps})"
                results[result_name] = run_method_over_seeds(
                    result_name=result_name,
                    method="dd_q",
                    common_args=common_args,
                    seeds=SEEDS,
                    dd_num_q=dd_num_q,
                    dd_delay_steps=dd_delay_steps,
                )
            continue

        if name not in methods:
            raise ValueError(f"Unknown method in RUN_METHODS: {name}")

        results[name] = run_method_over_seeds(
            result_name=name,
            method=methods[name],
            common_args=common_args,
            seeds=SEEDS,
        )

    Q_star, V_star = get_optimal_q(
        GRID_HEIGHT,
        GRID_WIDTH,
        GAMMA,
        wall_reward=WALL_REWARD,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
    )

    np.save(os.path.join(NUMPY_DIR, "Q_star.npy"), Q_star)
    np.save(os.path.join(NUMPY_DIR, "V_star.npy"), V_star)
    np.savetxt(
        os.path.join(TABLE_DIR, "V_star.csv"),
        V_star,
        delimiter=",",
        fmt="%.6f",
    )

    # Save aggregated and raw seed results.
    for name, history in results.items():
        save_seed_aggregate(
            name=name,
            history=history,
            table_dir=TABLE_DIR,
            numpy_dir=NUMPY_DIR,
        )

    suffix = (
        f"H{GRID_HEIGHT}_W{GRID_WIDTH}"
        f"_K{MAX_DIM_MULTIPLIER}"
        f"_bs{BEHAVIOR_SIGMA}"
        f"_boot{BOOTSTRAP_SIGMA}"
        f"_seeds{len(SEEDS)}"
    )

    candidates = 4 * MAX_DIM_MULTIPLIER
    band_label = f"mean +/- 1 SD over {len(SEEDS)} seeds"

    save_plot(
        results,
        "v_rmse",
        "V RMSE",
        f"V RMSE ({band_label})",
        f"v_rmse_{suffix}.png",
        FIGURE_DIR,
    )

    save_plot(
        results,
        "q_rmse",
        "Q RMSE",
        f"Q RMSE ({band_label})",
        f"q_rmse_{suffix}.png",
        FIGURE_DIR,
    )

    save_plot(
        results,
        "v_bias",
        "Mean(V - V*)",
        f"Global V Bias ({band_label})",
        f"v_bias_{suffix}.png",
        FIGURE_DIR,
        zero_line=True,
    )

    save_plot(
        results,
        "initial_v_error",
        "V(s0) - V*(s0)",
        f"Initial State Value Error ({band_label})",
        f"initial_v_error_{suffix}.png",
        FIGURE_DIR,
        zero_line=True,
    )

    save_plot(
        results,
        "initial_v_squared_error",
        "(V(s0) - V*(s0))^2",
        f"Initial State Squared Error ({band_label})",
        f"initial_v_squared_error_{suffix}.png",
        FIGURE_DIR,
    )

    # --------------------------------------------------------
    # Print final mean +/- std
    # --------------------------------------------------------
    start = (GRID_HEIGHT - 1, 0)

    print("\n========================================")
    print("Experiment")
    print("========================================")
    print("Grid                     :", (GRID_HEIGHT, GRID_WIDTH))
    print("Real actions             :", 4)
    print("Max dimension multiplier :", MAX_DIM_MULTIPLIER)
    print("Bootstrap candidates     :", candidates)
    print("Behavior sigma           :", BEHAVIOR_SIGMA)
    print("Bootstrap sigma          :", BOOTSTRAP_SIGMA)
    print("Wall reward              :", WALL_REWARD)
    print("Step reward              :", STEP_REWARD)
    print("Goal reward              :", GOAL_REWARD)
    print("Number of seeds          :", len(SEEDS))
    print("Seeds                    :", SEEDS)
    print("Std ddof                 :", STD_DDOF)
    if "DD-Q-learning" in RUN_METHODS:
        print("DD-Q configurations      :", DD_Q_CONFIGS)
    print("Initial state            :", start)
    print("True V*(initial state)   :", V_star[start])

    for name, history in results.items():
        print("\n========================================")
        print(name)
        print("========================================")
        print("Final V mean table:")
        print(np.round(history["V_mean"][-1], 4))
        print("Final V std table:")
        print(np.round(history["V_std"][-1], 4))

        for metric, label in (
            ("q_rmse", "Final Q RMSE"),
            ("v_rmse", "Final V RMSE"),
            ("v_bias", "Final global V bias"),
            ("initial_v_error", "Final initial V error"),
            ("initial_v_squared_error", "Final initial squared err"),
        ):
            mean = history[f"{metric}_mean"][-1]
            std = history[f"{metric}_std"][-1]
            print(f"{label:<28}: {mean:.6f} +/- {std:.6f}")

    print("\nAll results saved to:")
    print(os.path.abspath(SAVE_DIR))


if __name__ == "__main__":
    main()
