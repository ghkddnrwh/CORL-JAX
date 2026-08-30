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
# 3.5 Persistent function-approximation error
# ============================================================

class PersistentAR1Noise:
    """Lazy stationary AR(1) approximation-error field.

    Each estimator has a persistent Gaussian error for every
    (state, real action, virtual candidate):

        eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * sigma * xi_t

    with stationary marginal N(0, sigma^2).

    To keep computation cheap on a large GridWorld, the full tensor is NOT
    advanced every environment step.  A state's slice is advanced only when
    that state is queried.  If it has not been queried for delta steps, the
    exact delta-step AR(1) transition is sampled in one shot.
    """

    def __init__(
        self,
        num_estimators,
        grid_height,
        grid_width,
        num_actions,
        max_dim_multiplier,
        sigma,
        rho,
        rng,
    ):
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho must satisfy 0 <= rho < 1")

        self.num_estimators = int(num_estimators)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.num_actions = int(num_actions)
        self.max_dim_multiplier = int(max_dim_multiplier)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.rng = rng

        shape = (
            self.num_estimators,
            self.grid_height,
            self.grid_width,
            self.num_actions,
            self.max_dim_multiplier,
        )

        # Start directly from the stationary distribution.
        self.values = self.rng.normal(0.0, self.sigma, size=shape)

        # All candidates of a state are advanced together, so one timestamp
        # per state is sufficient.
        self.last_step = np.zeros((self.grid_height, self.grid_width), dtype=np.int64)

    def at(self, state, step):
        """Return persistent errors at `state` at integer time `step`."""
        r, c = state
        step = int(step)
        previous_step = int(self.last_step[r, c])
        delta = step - previous_step

        if delta < 0:
            raise ValueError("PersistentAR1Noise cannot move backward in time")

        if delta > 0:
            rho_delta = self.rho ** delta
            innovation_scale = self.sigma * np.sqrt(max(0.0, 1.0 - rho_delta ** 2))

            old = self.values[:, r, c, :, :]
            innovation = self.rng.normal(0.0, 1.0, size=old.shape)
            self.values[:, r, c, :, :] = (
                rho_delta * old + innovation_scale * innovation
            )
            self.last_step[r, c] = step

        return self.values[:, r, c, :, :]


def sample_delayed_noise_snapshot(
    current_noise,
    current_step,
    snapshot_step,
    sigma,
    rho,
    rng,
):
    """Sample the AR(1) error that the same approximator had at snapshot_step.

    A stationary Gaussian AR(1) process is time reversible.  Therefore, if
    eps_t is known, eps_{t-L} | eps_t has

        mean = rho^L eps_t
        var  = sigma^2 (1-rho^(2L)).

    DD-Q uses this to lazily construct the delayed approximation-error
    snapshot only for states that are actually queried, avoiding an
    O(N*H*W*4*K) copy/update every D steps.
    """
    lag = int(current_step) - int(snapshot_step)
    if lag < 0:
        raise ValueError("snapshot_step cannot be in the future")
    if lag == 0:
        return current_noise.copy()

    rho_lag = rho ** lag
    innovation_scale = sigma * np.sqrt(max(0.0, 1.0 - rho_lag ** 2))
    return (
        rho_lag * current_noise
        + innovation_scale * rng.normal(0.0, 1.0, size=current_noise.shape)
    )


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
    approx_sigma=0.1,
    noise_rho=0.99,
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
    if approx_sigma < 0:
        raise ValueError("approx_sigma must be >= 0")
    if not (0.0 <= noise_rho < 1.0):
        raise ValueError("noise_rho must satisfy 0 <= noise_rho < 1")

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
    # Clean Q parameters + persistent approximation errors
    # --------------------------------------------------------
    if method == "q_learning":
        Q = np.zeros(q_shape)
        approx_noise = PersistentAR1Noise(
            num_estimators=1,
            grid_height=grid_height,
            grid_width=grid_width,
            num_actions=4,
            max_dim_multiplier=max_dim_multiplier,
            sigma=approx_sigma,
            rho=noise_rho,
            rng=rng,
        )

    elif method in ("double_q", "clipped_double_q"):
        Q1 = np.zeros(q_shape)
        Q2 = np.zeros(q_shape)
        approx_noise = PersistentAR1Noise(
            num_estimators=2,
            grid_height=grid_height,
            grid_width=grid_width,
            num_actions=4,
            max_dim_multiplier=max_dim_multiplier,
            sigma=approx_sigma,
            rho=noise_rho,
            rng=rng,
        )

    elif method == "dd_q":
        Qs = np.zeros((dd_num_q, *q_shape))
        Q_delayed = Qs.copy()

        approx_noise = PersistentAR1Noise(
            num_estimators=dd_num_q,
            grid_height=grid_height,
            grid_width=grid_width,
            num_actions=4,
            max_dim_multiplier=max_dim_multiplier,
            sigma=approx_sigma,
            rho=noise_rho,
            rng=rng,
        )

        # Member i is selected by delayed member (i + peer_shift) % N.
        peer_shift = 1

        # The clean delayed Q snapshot is refreshed every D updates.
        # delayed_snapshot_step records the time represented by Q_delayed.
        delayed_snapshot_step = 0

        # Persistent approximation error is also delayed.  We construct its
        # snapshot lazily per state during the current D-step block.
        delayed_noise_cache = {}

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
            # The same persistent approximation-error field is used by
            # behavior and bootstrap.  Behavior uses virtual copy k=0 so
            # the environment still has exactly four real actions.
            noise_state = approx_noise.at(state, t)[0]  # [4, K]
            Q_behavior = Q[state] + noise_state[:, 0]
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
                noise_next = approx_noise.at(next_state, t)[0]  # [4, K]
                noisy_candidates = Q[next_state][:, None] + noise_next
                target = reward + gamma * np.max(noisy_candidates)

            # Approximation error is NOT accumulated into the clean table.
            Q[state][action] += alpha * (target - Q[state][action])

        # ====================================================
        # Double Q-learning
        # ====================================================
        elif method == "double_q":
            noise_state = approx_noise.at(state, t)  # [2, 4, K]
            Q1_behavior = Q1[state] + noise_state[0, :, 0]
            Q2_behavior = Q2[state] + noise_state[1, :, 0]
            Q_behavior = (Q1_behavior + Q2_behavior) / 2.0
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
                noise_next = approx_noise.at(next_state, t)  # [2, 4, K]
                Q1_candidates = Q1[next_state][:, None] + noise_next[0]
                Q2_candidates = Q2[next_state][:, None] + noise_next[1]

                index1 = np.unravel_index(
                    np.argmax(Q1_candidates), Q1_candidates.shape
                )
                target1 = reward + gamma * Q2_candidates[index1]

                index2 = np.unravel_index(
                    np.argmax(Q2_candidates), Q2_candidates.shape
                )
                target2 = reward + gamma * Q1_candidates[index2]

            Q1[state][action] += alpha * (target1 - Q1[state][action])
            Q2[state][action] += alpha * (target2 - Q2[state][action])

        # ====================================================
        # Clipped Double Q-learning
        # ====================================================
        elif method == "clipped_double_q":
            noise_state = approx_noise.at(state, t)  # [2, 4, K]
            Q1_behavior = Q1[state] + noise_state[0, :, 0]
            Q2_behavior = Q2[state] + noise_state[1, :, 0]
            Q_behavior = (Q1_behavior + Q2_behavior) / 2.0
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
                noise_next = approx_noise.at(next_state, t)  # [2, 4, K]
                Q1_candidates = Q1[next_state][:, None] + noise_next[0]
                Q2_candidates = Q2[next_state][:, None] + noise_next[1]

                clipped_candidates = np.minimum(Q1_candidates, Q2_candidates)
                target = reward + gamma * np.max(clipped_candidates)

            Q1[state][action] += alpha * (target - Q1[state][action])
            Q2[state][action] += alpha * (target - Q2[state][action])

        # ====================================================
        # DD-Q-learning
        # ====================================================
        elif method == "dd_q":
            r, c = state
            noise_state = approx_noise.at(state, t)  # [N, 4, K]

            # Behavior uses the ensemble mean over the N online estimators,
            # with k=0 as the four-real-action approximation output.
            Q_behavior_members = Qs[:, r, c, :] + noise_state[:, :, 0]
            Q_behavior = np.mean(Q_behavior_members, axis=0)
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
                nr, nc = next_state

                # Current online approximation errors at s'.
                online_noise_next = approx_noise.at(next_state, t)  # [N, 4, K]
                online_candidates = (
                    Qs[:, nr, nc, :, None] + online_noise_next
                )

                # Delayed selector must use both the old clean Q snapshot and
                # the old approximation-error snapshot.  The latter is lazily
                # sampled once per state per D-step block and then frozen.
                if next_state not in delayed_noise_cache:
                    delayed_noise_cache[next_state] = sample_delayed_noise_snapshot(
                        current_noise=online_noise_next,
                        current_step=t,
                        snapshot_step=delayed_snapshot_step,
                        sigma=approx_sigma,
                        rho=noise_rho,
                        rng=rng,
                    )

                delayed_candidates = (
                    Q_delayed[:, nr, nc, :, None]
                    + delayed_noise_cache[next_state]
                )

                peer_indices = (
                    np.arange(dd_num_q) + peer_shift
                ) % dd_num_q

                targets = np.empty(dd_num_q, dtype=float)

                for i, j in enumerate(peer_indices):
                    # Other delayed estimator selects the candidate.
                    selected_index = np.unravel_index(
                        np.argmax(delayed_candidates[j]),
                        delayed_candidates[j].shape,
                    )

                    # Own CURRENT online estimator evaluates exactly that
                    # same (real action, virtual candidate).
                    own_value = online_candidates[i][selected_index]
                    targets[i] = reward + gamma * own_value

            Qs[:, r, c, action] += alpha * (
                targets - Qs[:, r, c, action]
            )

            # Refresh the delayed clean table and the delayed error-snapshot
            # time together.  The new snapshot is used from the NEXT update.
            if t % dd_delay_steps == 0:
                Q_delayed = Qs.copy()
                delayed_snapshot_step = t
                delayed_noise_cache = {}

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
        # Logging: evaluate the CLEAN learned Q parameters.
        # ====================================================
        if t % log_every == 0:
            if method == "q_learning":
                Q_est = Q.copy()
            elif method in ("double_q", "clipped_double_q"):
                Q_est = (Q1 + Q2) / 2.0
            else:
                Q_est = np.mean(Qs, axis=0)

            # Only the four real actions define V_est.
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
    # Final clean Q tables
    # --------------------------------------------------------
    if method == "q_learning":
        history["Q_final"] = Q.copy()

    elif method in ("double_q", "clipped_double_q"):
        history["Q1_final"] = Q1.copy()
        history["Q2_final"] = Q2.copy()
        history["Q_final"] = (Q1 + Q2) / 2.0

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

# Persistent function-approximation error.
# Each Q estimator carries a temporally correlated AR(1) error field:
#   eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * sigma * xi_t
# APPROX_SIGMA controls the stationary standard deviation.
# NOISE_RHO=0.0 gives temporally fresh error; values close to 1 persist longer.
APPROX_SIGMA = 2.0
NOISE_RHO = 0.99

# Reward shaping controls action gaps.
# More negative WALL_REWARD makes wrong wall selections much more costly,
# which can amplify Double-Q / DD-Q underestimation.
WALL_REWARD = -2.0
STEP_REWARD = 0.0
GOAL_REWARD = 1.0

# K=1 -> max over 4 candidates
# K=3 -> max over 12 candidates
# Persistent noise is advanced lazily only for states that are queried, so
# large K remains much cheaper than advancing a full H*W*4*K tensor every step.
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

# Run each algorithm with 10 independent random seeds.
NUM_SEEDS = 10
BASE_SEED = 0
SEEDS = list(range(BASE_SEED, BASE_SEED + NUM_SEEDS))


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

SAVE_DIR = "./logs/iclr2027_dd_iql/tabular_persistent_ar1"

FIGURE_DIR = os.path.join(SAVE_DIR, "figures")
TABLE_DIR = os.path.join(SAVE_DIR, "tables")
NUMPY_DIR = os.path.join(SAVE_DIR, "numpy")


# ============================================================
# 7. Plot helper
# ============================================================

def aggregate_seed_histories(seed_histories):
    """Aggregate histories from independent seeds.

    The central curve is the seed mean.  Variance is computed with ddof=1,
    and plots use mean +/- sqrt(variance) (= one sample standard deviation)
    so the shaded region has the same units as the plotted metric.
    """
    if len(seed_histories) == 0:
        raise ValueError("At least one seed history is required.")

    reference_steps = np.asarray(seed_histories[0]["step"])
    for history in seed_histories[1:]:
        if not np.array_equal(reference_steps, np.asarray(history["step"])):
            raise ValueError("All seeds must have identical logging steps.")

    aggregated = {
        "step": reference_steps.copy(),
        "num_seeds": len(seed_histories),
    }

    # Time-series quantities logged during training.
    history_keys = [
        "V",
        "q_rmse",
        "v_rmse",
        "v_bias",
        "initial_v_error",
        "initial_v_squared_error",
    ]

    # Final learned parameters.  Only keys present for the method are used.
    final_keys = [
        "Q_final",
        "Q1_final",
        "Q2_final",
        "Q_ensemble_final",
        "Q_delayed_final",
    ]

    ddof = 1 if len(seed_histories) > 1 else 0

    for key in history_keys + final_keys:
        if key not in seed_histories[0]:
            continue
        values = np.stack([np.asarray(history[key]) for history in seed_histories], axis=0)
        aggregated[f"{key}_per_seed"] = values
        aggregated[key] = np.mean(values, axis=0)
        aggregated[f"{key}_var"] = np.var(values, axis=0, ddof=ddof)
        aggregated[f"{key}_std"] = np.sqrt(aggregated[f"{key}_var"])

    return aggregated


def run_across_seeds(method, common_args, **method_specific_args):
    """Train one method independently for every seed and aggregate results."""
    seed_histories = []
    for seed_index, seed in enumerate(SEEDS, start=1):
        print(f"  Seed {seed_index}/{NUM_SEEDS}: {seed}")
        history = train(
            method,
            seed=seed,
            **method_specific_args,
            **common_args,
        )
        seed_histories.append(history)
    return aggregate_seed_histories(seed_histories)


def save_plot(results, metric, ylabel, title, filename, figure_dir, zero_line=False):
    plt.figure(figsize=(7, 4))

    for name, history in results.items():
        steps = history["step"]
        mean = history[metric]
        std = history[f"{metric}_std"]

        line, = plt.plot(steps, mean, label=name)
        plt.fill_between(
            steps,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.20,
            linewidth=0.0,
        )

    if zero_line:
        plt.axhline(0, linestyle="--")

    plt.xlabel("Update step")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n(mean +/- 1 std over {NUM_SEEDS} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(figure_dir, filename),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


# ============================================================
# 8. Main experiment
# ============================================================

def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(NUMPY_DIR, exist_ok=True)

    print("Save directory:", os.path.abspath(SAVE_DIR))

    common_args = dict(
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        num_updates=NUM_UPDATES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        approx_sigma=APPROX_SIGMA,
        noise_rho=NOISE_RHO,
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
                print(f"Training: {result_name} over {NUM_SEEDS} seeds")
                results[result_name] = run_across_seeds(
                    "dd_q",
                    common_args,
                    dd_num_q=dd_num_q,
                    dd_delay_steps=dd_delay_steps,
                )
            continue

        if name not in methods:
            raise ValueError(f"Unknown method in RUN_METHODS: {name}")

        print(f"Training: {name} over {NUM_SEEDS} seeds")
        results[name] = run_across_seeds(methods[name], common_args)

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

    # --------------------------------------------------------
    # Save seed-aggregated histories / final tables
    # --------------------------------------------------------
    scalar_metrics = [
        "q_rmse",
        "v_rmse",
        "v_bias",
        "initial_v_error",
        "initial_v_squared_error",
    ]

    for name, history in results.items():
        safe_name = (
            name.replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            .replace(",", "")
        )

        # Keep the original filenames for the seed mean so existing analysis
        # scripts continue to work, and save std/variance/per-seed arrays too.
        np.save(
            os.path.join(NUMPY_DIR, f"{safe_name}_V_history.npy"),
            history["V"],
        )
        np.save(
            os.path.join(NUMPY_DIR, f"{safe_name}_V_history_std.npy"),
            history["V_std"],
        )
        np.save(
            os.path.join(NUMPY_DIR, f"{safe_name}_V_history_var.npy"),
            history["V_var"],
        )
        np.save(
            os.path.join(NUMPY_DIR, f"{safe_name}_V_history_per_seed.npy"),
            history["V_per_seed"],
        )

        final_keys = [
            "Q_final",
            "Q1_final",
            "Q2_final",
            "Q_ensemble_final",
            "Q_delayed_final",
        ]
        for key in final_keys:
            if key not in history:
                continue
            np.save(
                os.path.join(NUMPY_DIR, f"{safe_name}_{key}.npy"),
                history[key],
            )
            np.save(
                os.path.join(NUMPY_DIR, f"{safe_name}_{key}_std.npy"),
                history[f"{key}_std"],
            )
            np.save(
                os.path.join(NUMPY_DIR, f"{safe_name}_{key}_var.npy"),
                history[f"{key}_var"],
            )
            np.save(
                os.path.join(NUMPY_DIR, f"{safe_name}_{key}_per_seed.npy"),
                history[f"{key}_per_seed"],
            )

        # Final V table: seed mean, standard deviation, and variance.
        np.savetxt(
            os.path.join(TABLE_DIR, f"{safe_name}_final_V.csv"),
            history["V"][-1],
            delimiter=",",
            fmt="%.6f",
        )
        np.savetxt(
            os.path.join(TABLE_DIR, f"{safe_name}_final_V_std.csv"),
            history["V_std"][-1],
            delimiter=",",
            fmt="%.6f",
        )
        np.savetxt(
            os.path.join(TABLE_DIR, f"{safe_name}_final_V_var.csv"),
            history["V_var"][-1],
            delimiter=",",
            fmt="%.6f",
        )

        # CSV contains mean/std/variance at every logging step.
        metric_path = os.path.join(TABLE_DIR, f"{safe_name}_metrics.csv")
        with open(metric_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"]
            for metric in scalar_metrics:
                header.extend([
                    f"{metric}_mean",
                    f"{metric}_std",
                    f"{metric}_var",
                ])
            writer.writerow(header)

            for i in range(len(history["step"])):
                row = [history["step"][i]]
                for metric in scalar_metrics:
                    row.extend([
                        history[metric][i],
                        history[f"{metric}_std"][i],
                        history[f"{metric}_var"][i],
                    ])
                writer.writerow(row)

        # Raw per-seed scalar histories make downstream statistical tests easy.
        for metric in scalar_metrics:
            np.save(
                os.path.join(NUMPY_DIR, f"{safe_name}_{metric}_per_seed.npy"),
                history[f"{metric}_per_seed"],
            )

    suffix = (
        f"H{GRID_HEIGHT}_W{GRID_WIDTH}"
        f"_K{MAX_DIM_MULTIPLIER}"
        f"_sigma{APPROX_SIGMA}"
        f"_rho{NOISE_RHO}"
    )

    candidates = 4 * MAX_DIM_MULTIPLIER

    save_plot(
        results,
        "v_rmse",
        "V RMSE",
        f"V RMSE (K={MAX_DIM_MULTIPLIER}, candidates={candidates})",
        f"v_rmse_{suffix}.png",
        FIGURE_DIR,
    )

    save_plot(
        results,
        "q_rmse",
        "Q RMSE",
        f"Q RMSE (K={MAX_DIM_MULTIPLIER}, candidates={candidates})",
        f"q_rmse_{suffix}.png",
        FIGURE_DIR,
    )

    save_plot(
        results,
        "v_bias",
        "Mean(V - V*)",
        f"Global V Bias (K={MAX_DIM_MULTIPLIER}, candidates={candidates})",
        f"v_bias_{suffix}.png",
        FIGURE_DIR,
        zero_line=True,
    )

    save_plot(
        results,
        "initial_v_error",
        "V(s0) - V*(s0)",
        f"Initial State Value Error (K={MAX_DIM_MULTIPLIER})",
        f"initial_v_error_{suffix}.png",
        FIGURE_DIR,
        zero_line=True,
    )

    save_plot(
        results,
        "initial_v_squared_error",
        "(V(s0) - V*(s0))^2",
        f"Initial State Squared Error (K={MAX_DIM_MULTIPLIER})",
        f"initial_v_squared_error_{suffix}.png",
        FIGURE_DIR,
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------
    start = (GRID_HEIGHT - 1, 0)

    print("\n========================================")
    print("Experiment")
    print("========================================")
    print("Grid                     :", (GRID_HEIGHT, GRID_WIDTH))
    print("Real actions             :", 4)
    print("Max dimension multiplier :", MAX_DIM_MULTIPLIER)
    print("Bootstrap candidates     :", candidates)
    print("Approximation sigma      :", APPROX_SIGMA)
    print("Noise AR(1) rho          :", NOISE_RHO)
    print("Wall reward              :", WALL_REWARD)
    print("Step reward              :", STEP_REWARD)
    print("Goal reward              :", GOAL_REWARD)
    print("Number of seeds          :", NUM_SEEDS)
    print("Seeds                    :", SEEDS)
    if "DD-Q-learning" in RUN_METHODS:
        print("DD-Q configurations      :", DD_Q_CONFIGS)
    print("Initial state            :", start)
    print("True V*(initial state)   :", V_star[start])

    for name, history in results.items():
        print("\n========================================")
        print(name)
        print("========================================")
        print("Final mean V table:")
        print(np.round(history["V"][-1], 4))
        print(
            f"Final Q RMSE             : {history['q_rmse'][-1]:.6f} "
            f"+/- {history['q_rmse_std'][-1]:.6f}"
        )
        print(
            f"Final V RMSE             : {history['v_rmse'][-1]:.6f} "
            f"+/- {history['v_rmse_std'][-1]:.6f}"
        )
        print(
            f"Final global V bias      : {history['v_bias'][-1]:.6f} "
            f"+/- {history['v_bias_std'][-1]:.6f}"
        )
        print(
            f"Final initial V error    : {history['initial_v_error'][-1]:.6f} "
            f"+/- {history['initial_v_error_std'][-1]:.6f}"
        )
        print(
            f"Final initial squared err: {history['initial_v_squared_error'][-1]:.6f} "
            f"+/- {history['initial_v_squared_error_std'][-1]:.6f}"
        )

    print("\nAll results saved to:")
    print(os.path.abspath(SAVE_DIR))


if __name__ == "__main__":
    main()
