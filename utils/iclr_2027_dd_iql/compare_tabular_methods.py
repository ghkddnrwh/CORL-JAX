"""Tabular GridWorld: two noise modes + generalized N-member Double Q.

Modes:
    shared_target  : one per-step noisy Q view (shared by behavior and TD target)
    table_additive : permanently perturb every online Q-table cell after TD

The full experiment configuration is near section 5 at the bottom.
"""

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
    """Epsilon-greedy over the 4 REAL actions.

    Accept either [4] (one value per action) or [4, K] (virtual noisy
    candidates per action).  When given [4, K], choose the best (action, k)
    candidate and return only its REAL action index (0..3).
    Random exploration remains uniform over just the 4 real actions.
    """
    q_values = np.asarray(q_values)
    if q_values.ndim == 1:
        if q_values.shape != (4,):
            raise ValueError("1D behavior values must have shape (4,)")
        action_scores = q_values
    elif q_values.ndim == 2:
        if q_values.shape[0] != 4 or q_values.shape[1] < 1:
            raise ValueError("2D behavior candidates must have shape (4, K), K>=1")
        action_scores = np.max(q_values, axis=1)
    else:
        raise ValueError("behavior values must be shape (4,) or (4,K)")

    if rng.random() < epsilon:
        return int(rng.integers(4))
    return int(np.argmax(action_scores))


# ============================================================
# 4. Noise models / Training
# ============================================================

NOISE_MODES = ("shared_target", "table_additive")


class StepQView:
    """Q-value view fixed within one environment update.

    shared_target:
      For each estimator i and queried state s, sample ONE (4, K) array:
          noisy_Q[i, s, a, k] = Q[i, s, a] + eps[i, s, a, k]
      Reuse the SAME array for behavior (all K by default) AND bootstrap. Different
      states/estimators/candidates have independent Gaussian errors.  Instead
      of allocating the entire [N,H,W,4,K] tensor at each step, create only
      queried state slices and cache them for the duration of the step.

    table_additive:
      The table already contains all previously injected errors.  Return its
      current real-action values with a singleton candidate axis (K=1).
      No read-time noise is added.

    IMPORTANT: Create a new StepQView before every training update.  All
    accesses to this view occur BEFORE that update changes the Q parameters.
    """

    def __init__(self, tables, noise_mode, noise_sigma, num_candidates, rng):
        self.tables = tables  # shape [N, H, W, 4]
        self.noise_mode = noise_mode
        self.noise_sigma = noise_sigma
        self.num_candidates = num_candidates if noise_mode == "shared_target" else 1
        self.rng = rng
        self._cache = {}

    def at(self, state):
        state = tuple(state)
        if state not in self._cache:
            r, c = state
            base = self.tables[:, r, c, :, None]
            if self.noise_mode == "shared_target":
                # All members get independent noise; same sampled noise is
                # reused whenever the same state is queried in this step.
                eps = self.rng.normal(
                    0.0, self.noise_sigma,
                    size=(self.tables.shape[0], 4, self.num_candidates),
                )
                self._cache[state] = base + eps
            else:
                # Freeze pre-update values for this one step.
                self._cache[state] = base.copy()
        return self._cache[state]


def train(
    method,
    grid_height=5,
    grid_width=5,
    num_updates=100_000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    noise_mode="shared_target",
    noise_sigma=10.0,
    wall_reward=-1.0,
    step_reward=0.0,
    goal_reward=1.0,
    max_dim_multiplier=1,
    behavior_use_all_k=True,
    double_num_q=2,
    dd_num_q=2,
    dd_delay_steps=1,
    log_every=500,
    seed=0,
):
    """Train one tabular algorithm with one of two noise mechanisms.

    noise_mode='shared_target':
      Per step, create *logically* one noisy view of each CURRENT online
      Q table. Behavior uses all K by default (or only k=0 if
      behavior_use_all_k=False); bootstrap always uses all K.
      The exact SAME noisy value is reused for the same (i,s,a,k) within the
      step (not independently resampled for behavior and target).
      DD-Q delayed selector tables get separate per-step noisy views.
      The noisy views are discarded after the TD update.

    noise_mode='table_additive':
      Behavior and TD targets directly read the stored Q tables (NO read-time
      noise).  After EVERY TD update, independently sample N(0,sigma^2) for
      EVERY stored [estimator,state,real_action] entry and add it in-place.
      Delayed DD-Q tables are snapshots, not separately perturbed each step.
      In this mode max_dim_multiplier is ignored: there are 4 real actions.

    Generalized Double Q when N>2:
      Q_i chooses argmax using its own view; estimator (i+shift)%N evaluates
      the chosen candidate, where shift cycles over 1..N-1 every step.
      All N members update from the same transition.  At N=2 this reduces
      exactly to the prior cross-evaluation/update rule.

    Generalized Clipped Double Q when N>2:
      For each candidate, take the MIN over all N estimators; then MAX over
      candidates; apply that shared target to each of the N estimators.
    """
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {NOISE_MODES}")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be >= 0")
    if max_dim_multiplier < 1:
        raise ValueError("max_dim_multiplier must be >= 1")
    if num_updates < 1 or log_every < 1:
        raise ValueError("num_updates and log_every must be >= 1")
    if method in ("double_q", "clipped_double_q") and double_num_q < 2:
        raise ValueError("Double/Clipped Double Q require double_num_q >= 2")
    if method == "dd_q" and (dd_num_q < 2 or dd_delay_steps < 1):
        raise ValueError("DD-Q requires dd_num_q >= 2 and dd_delay_steps >= 1")

    if method == "q_learning":
        n = 1
    elif method in ("double_q", "clipped_double_q"):
        n = double_num_q
    elif method == "dd_q":
        n = dd_num_q
    else:
        raise ValueError(f"Unknown method: {method}")

    rng = np.random.default_rng(seed)
    start = (grid_height - 1, 0)
    goal = (0, grid_width - 1)
    Q_star, V_star = get_optimal_q(
        grid_height, grid_width, gamma,
        wall_reward=wall_reward,
        step_reward=step_reward,
        goal_reward=goal_reward,
    )

    # All trainable estimators have uniform shape [N, H, W, 4].
    Qs = np.zeros((n, grid_height, grid_width, 4), dtype=float)

    if method == "dd_q":
        # Only online tables are learned/perturbed; snapshots are periodically
        # copied AFTER the online TD update and table noise (if selected).
        Q_delayed = Qs.copy()
        peer_shift = 1

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

    for t in range(1, num_updates + 1):
        # Construct the shared view BEFORE acting/updating.  Lazy sampling
        # gives the same independent per-cell distribution as constructing
        # a full [N,H,W,4,K] tensor once per update, much more efficiently.
        online_view = StepQView(
            Qs, noise_mode, noise_sigma, max_dim_multiplier, rng
        )

        # The online noisy view is shared with the subsequent TD target.
        # Default: average estimators per (real action, virtual candidate),
        # then choose the best candidate over all 4*K positions; execute
        # ONLY the associated real action.  Optional k=0 behavior supports
        # apples-to-apples ablations against the previous implementation.
        current_candidates = online_view.at(state)  # [N,4,K]
        if behavior_use_all_k:
            Q_behavior = np.mean(current_candidates, axis=0)  # [4,K]
        else:
            Q_behavior = np.mean(current_candidates[:, :, 0], axis=0)  # [4]
        action = choose_action(Q_behavior, epsilon, rng)

        next_state, reward, done = env_step(
            state, action, grid_height, grid_width,
            wall_reward=wall_reward,
            step_reward=step_reward,
            goal_reward=goal_reward,
        )

        if done:
            # No bootstrap/noise on terminal targets.
            targets = np.full(n, reward, dtype=float)
        else:
            online_next = online_view.at(next_state)  # [N, 4, K] or [N,4,1]

            if method == "q_learning":
                targets = np.array([reward + gamma * np.max(online_next[0])])

            elif method == "double_q":
                # Every Q_i selects using itself, and is evaluated by one
                # DIFFERENT Q_j. Cycle peers so N>2 uses all other members.
                shift = 1 + ((t - 1) % (n - 1))
                targets = np.empty(n, dtype=float)
                for i in range(n):
                    j = (i + shift) % n
                    selected = np.unravel_index(
                        np.argmax(online_next[i]), online_next[i].shape
                    )
                    targets[i] = reward + gamma * online_next[j][selected]

            elif method == "clipped_double_q":
                # N=2 is exactly the original pairwise minimum.
                clipped_candidates = np.min(online_next, axis=0)
                common_target = reward + gamma * np.max(clipped_candidates)
                targets = np.full(n, common_target, dtype=float)

            else:  # DD-Q
                # For shared_target, selector uses a separate noisy view of
                # the frozen delayed Q parameters.  For table_additive, the
                # delayed table is read as-is (past noisy online snapshot).
                delayed_view = StepQView(
                    Q_delayed, noise_mode, noise_sigma, max_dim_multiplier, rng
                )
                delayed_next = delayed_view.at(next_state)
                targets = np.empty(n, dtype=float)
                for i in range(n):
                    j = (i + peer_shift) % n
                    selected = np.unravel_index(
                        np.argmax(delayed_next[j]), delayed_next[j].shape
                    )
                    # Own CURRENT online member evaluates the peer's choice.
                    targets[i] = reward + gamma * online_next[i][selected]

        # All Q_i updated on the SAME actual transition (s,a,r,s').
        r, c = state
        Qs[:, r, c, action] += alpha * (targets - Qs[:, r, c, action])

        # Mode 2: after learning, perturb EVERY trainable table entry with
        # independent fresh noise.  No extra noise on behavior/TD reads.
        if noise_mode == "table_additive":
            Qs += rng.normal(0.0, noise_sigma, size=Qs.shape)

        # Delayed snapshots are refreshed only on D-step boundaries.
        # In table_additive mode the snapshot includes noise already written
        # to the online table, and is NOT separately noised on other steps.
        if method == "dd_q" and t % dd_delay_steps == 0:
            Q_delayed = Qs.copy()
            peer_shift = (peer_shift % (n - 1)) + 1 if n > 2 else 1

        state = start if done else next_state

        if t % log_every == 0:
            Q_est = Qs[0].copy() if n == 1 else np.mean(Qs, axis=0)
            V_est = np.max(Q_est, axis=2)
            V_est[goal] = 0.0
            history["step"].append(t)
            history["V"].append(V_est.copy())
            history["q_rmse"].append(
                np.sqrt(np.mean((Q_est[mask] - Q_star[mask]) ** 2))
            )
            history["v_rmse"].append(
                np.sqrt(np.mean((V_est[mask] - V_star[mask]) ** 2))
            )
            history["v_bias"].append(np.mean(V_est[mask] - V_star[mask]))
            initial_error = V_est[start] - V_star[start]
            history["initial_v_error"].append(initial_error)
            history["initial_v_squared_error"].append(initial_error ** 2)

    for key in history:
        history[key] = np.array(history[key])

    history["Q_final"] = Qs[0].copy() if n == 1 else np.mean(Qs, axis=0)
    if n > 1:
        history["Q_ensemble_final"] = Qs.copy()
        # Backward-compatible convenience outputs for first two estimators.
        history["Q1_final"] = Qs[0].copy()
        history["Q2_final"] = Qs[1].copy()
    if method == "dd_q":
        history["Q_delayed_final"] = Q_delayed.copy()

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

# Select ONE of the two mutually exclusive noise injection mechanisms:
#   "shared_target": per-step shared noisy view for behavior AND bootstrap;
#                    does NOT permanently overwrite the Q tables.
#   "table_additive": directly add iid Gaussian noise to ALL online Q-table
#                     entries after EACH learning update.
NOISE_MODE = "shared_target"  # change to "table_additive" to run mode 2
NOISE_SIGMA = 10.0           # same Gaussian std used in either mode

# Reward shaping controls action gaps.
# More negative WALL_REWARD makes wrong wall selections much more costly,
# which can amplify Double-Q / DD-Q underestimation.
WALL_REWARD = -2.0
STEP_REWARD = 0.0
GOAL_REWARD = 1.0

# Virtual noisy candidates for shared_target only.
# By default BOTH behavior and bootstrap consider all 4*K candidate values.
# Behavior maps the best (real action, virtual candidate) to its real action.
# Set to False to reproduce the old behavior policy (only candidate k=0).
# In table_additive mode K is ignored and only 4 real actions are compared.
MAX_DIM_MULTIPLIER = 100
BEHAVIOR_USE_ALL_K = True

# Generalized Double / Clipped Double ensemble sizes.
# N=2 preserves the original Double Q update rule.
DOUBLE_Q_CONFIGS = [2, 3]
CLIPPED_DOUBLE_Q_CONFIGS = [2, 3]

# DD-Q configurations
# Each tuple is (N, D):
#   N = number of online/delayed Q tables
#   D = delayed refresh / peer-cycle period
#
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

# Keep both modes and sigma/K settings in separate output directories.
SAVE_DIR = os.path.join(
    "./logs/iclr2027_dd_iql/tabular_dual_noise",
    NOISE_MODE,
    f"sigma{NOISE_SIGMA:g}",
    (f"K{MAX_DIM_MULTIPLIER}_behavior_" + ("allK" if BEHAVIOR_USE_ALL_K else "k0"))
    if NOISE_MODE == "shared_target" else "real_actions_only",
)

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

    # Final stored Q estimates across seeds (perturbed in table_additive mode).
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
    double_num_q=None,
    dd_num_q=None,
    dd_delay_steps=None,
):
    seed_histories = []
    total = len(seeds)
    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"Training: {result_name} | seed={seed} ({seed_idx}/{total})")
        args = dict(common_args)
        args["seed"] = int(seed)
        if method in ("double_q", "clipped_double_q"):
            args["double_num_q"] = int(double_num_q)
        if method == "dd_q":
            args["dd_num_q"] = int(dd_num_q)
            args["dd_delay_steps"] = int(dd_delay_steps)
        seed_histories.append(train(method, **args))
    return aggregate_seed_histories(
        seed_histories, seeds=seeds, std_ddof=STD_DDOF,
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
        noise_mode=NOISE_MODE,
        noise_sigma=NOISE_SIGMA,
        wall_reward=WALL_REWARD,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        max_dim_multiplier=MAX_DIM_MULTIPLIER,
        behavior_use_all_k=BEHAVIOR_USE_ALL_K,
        log_every=LOG_EVERY,
    )

    results = {}
    for name in RUN_METHODS:
        if name == "Q-learning":
            results[name] = run_method_over_seeds(
                result_name=name, method="q_learning",
                common_args=common_args, seeds=SEEDS,
            )
        elif name in ("Double Q", "Clipped Double Q"):
            method = "double_q" if name == "Double Q" else "clipped_double_q"
            configs = (
                DOUBLE_Q_CONFIGS if name == "Double Q"
                else CLIPPED_DOUBLE_Q_CONFIGS
            )
            for n in configs:
                result_name = f"{name} (N={n})"
                results[result_name] = run_method_over_seeds(
                    result_name=result_name, method=method,
                    common_args=common_args, seeds=SEEDS,
                    double_num_q=n,
                )
        elif name == "DD-Q-learning":
            for n, delay in DD_Q_CONFIGS:
                result_name = f"DD-Q (N={n}, D={delay})"
                results[result_name] = run_method_over_seeds(
                    result_name=result_name, method="dd_q",
                    common_args=common_args, seeds=SEEDS,
                    dd_num_q=n, dd_delay_steps=delay,
                )
        else:
            raise ValueError(f"Unknown method in RUN_METHODS: {name}")

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

    actual_k = MAX_DIM_MULTIPLIER if NOISE_MODE == "shared_target" else 1
    suffix = (
        f"H{GRID_HEIGHT}_W{GRID_WIDTH}"
        f"_mode{NOISE_MODE}_sigma{NOISE_SIGMA:g}"
        f"_K{actual_k}_behavior{'allK' if BEHAVIOR_USE_ALL_K else 'k0'}"
        f"_seeds{len(SEEDS)}"
    )
    candidates = 4 * actual_k
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
    print("Noise mode               :", NOISE_MODE)
    print("Noise sigma              :", NOISE_SIGMA)
    print("Effective target K       :", actual_k)
    print("Double Q N configs       :", DOUBLE_Q_CONFIGS)
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
