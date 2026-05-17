import copy
import os
import pickle
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
import yaml
from flax import linen as nn
from flax import serialization, struct

TensorBatch = Dict[str, jnp.ndarray]

# ALGORITHM_NAME = "CDAF"
ALGORITHM_FULL_NAME = "Conservative Delayed Advantage Filtering"


@dataclass
class TrainConfig:
    # Experiment
    device: str = "gpu"  # one of: cpu, gpu, tpu. JAX selects the matching backend when available.
    env: str = "halfcheetah-medium-expert-v2"
    seed: int = 0
    eval_freq: int = int(25e3)
    n_episodes: int = 10
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    hyperparams_path: Optional[str] = "hyperparams/cdaf_jax.yml"
    use_hyperparams: bool = True

    # Dataset
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False

    # CDAF value learning
    discount: float = 0.99
    tau: float = 0.005
    qf_lr: float = 3e-4
    vf_lr: float = 3e-4
    delayed_update_period: int = 250
    min_weight_exponent: float = 0.0
    max_weight_exponent: float = 2.0
    weight_logit_clip: float = 10.0
    beta_min: float = 0.0

    # Coverage-aware CDAF filtering
    # beta_adv performs the original negative-advantage filtering.
    # coverage_confidence c(s) controls how much we trust that filtering:
    #   sparse state -> c(s) ~= 0 -> beta ~= 1, so V(s) is still trained from Q(s, a)
    #   dense state  -> c(s) ~= 1 -> beta ~= beta_adv
    use_coverage_aware_beta: bool = True
    coverage_knn_k: int = 10
    coverage_reference_size: int = 100_000
    coverage_low_quantile: float = 0.20
    coverage_high_quantile: float = 0.80
    adv_margin: float = 0.0

    # Online actor update. The actor is optimized inside the same training step
    # as Q and V instead of being trained in a separate post-hoc phase.
    actor_lr: float = 3e-4
    alpha: float = 2.5
    bc_coef: float = 1.0

    # Actor update objective.
    # td3_bc uses the TD3+BC actor objective.
    # weighted_bc is fully in-sample: weighted behavior cloning from dataset actions.
    # td3_weighted_bc keeps the TD3+BC Q-improvement term, but replaces the plain
    # BC regularizer with advantage-weighted BC from dataset actions.
    actor_update_method: str = "td3_bc"  # one of: td3_bc, weighted_bc, td3_weighted_bc
    policy_weight_exponent: float = 1.0
    policy_weight_clip: float = 20.0

    # Logging
    project: str = "ORL-BIAS"
    group: str = "CDAF-JAX"
    name: str = "CDAF-JAX"
    log_wandb: bool = True
    log_every: int = 100

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    # config.project = "ORL-BIAS"
    # config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{config.group}-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.min_weight_exponent >= 0.0, "min_weight_exponent must be >= 0"
    assert config.max_weight_exponent >= 0.0, "max_weight_exponent must be >= 0"
    assert config.beta_min >= 0.0, "beta_min must be >= 0"
    assert config.weight_logit_clip > 0.0, "weight_logit_clip must be > 0"
    assert config.coverage_knn_k > 0, "coverage_knn_k must be > 0"
    assert config.coverage_reference_size > 0, "coverage_reference_size must be > 0"
    assert 0.0 <= config.coverage_low_quantile <= 1.0, "coverage_low_quantile must be in [0, 1]"
    assert 0.0 <= config.coverage_high_quantile <= 1.0, "coverage_high_quantile must be in [0, 1]"
    assert config.coverage_high_quantile >= config.coverage_low_quantile
    assert config.adv_margin >= 0.0, "adv_margin must be >= 0"
    assert config.max_weight_exponent >= config.min_weight_exponent
    assert config.delayed_update_period > 0
    assert config.bc_coef >= 0.0
    assert config.policy_weight_exponent >= 0.0, "policy_weight_exponent must be >= 0"
    assert config.policy_weight_clip > 0.0, "policy_weight_clip must be > 0"
    assert config.actor_update_method in (
        "td3_bc",
        "weighted_bc",
        "td3_weighted_bc",
    ), "actor_update_method must be td3_bc, weighted_bc, or td3_weighted_bc"
    assert config.batch_size > 0


def _cli_overridden_fields(argv: Optional[List[str]] = None) -> set:
    argv = sys.argv[1:] if argv is None else argv
    overridden = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        if key:
            overridden.add(key)
    return overridden


def _coerce_hparam_value(value: Any) -> Any:
    # YAML often loads !!float 1e6 as float, but step counts should be ints in this script.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    """Load RL-Zoo-style env-specific hyperparameters and merge them into config.

    Priority is:
        dataclass defaults < hyperparams YAML < explicit CLI flags

    Supported alias:
        n_timesteps -> max_timesteps
    """
    if not config.use_hyperparams or config.hyperparams_path is None:
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    hparam_path = Path(config.hyperparams_path)
    if not hparam_path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}.")

    with open(hparam_path, "r") as f:
        all_hyperparams = yaml.safe_load(f) or {}

    if config.env not in all_hyperparams:
        print(f"No hyperparameters found for env '{config.env}' in {hparam_path}. Using dataclass/CLI values.")
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    env_hyperparams = all_hyperparams[config.env] or {}
    cli_overrides = _cli_overridden_fields()
    aliases = {
        "n_timesteps": "max_timesteps",
        # Backward compatibility for older hyperparameter files.
        "actor_fit_method": "actor_update_method",
    }
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []

    for raw_key, raw_value in env_hyperparams.items():
        key = aliases.get(raw_key, raw_key)
        if key not in config_fields:
            skipped_unknown.append(raw_key)
            continue
        if key in cli_overrides or raw_key in cli_overrides:
            skipped_cli.append(raw_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(f"{raw_key}->{key}" if raw_key != key else key)

    refresh_algorithm_names(config)
    validate_config(config)

    if applied:
        print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
    if skipped_cli:
        print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
    if skipped_unknown:
        print(f"Ignored unknown hyperparameter keys for CDAF: {', '.join(skipped_unknown)}")
    return config


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def select_jax_device(device: str):
    backend = device.lower()
    if backend == "cuda":
        backend = "gpu"
    try:
        dev = jax.devices(backend)[0]
    except Exception:
        print(f"Requested JAX backend '{device}' is not available. Falling back to default device.")
        dev = jax.devices()[0]
    print(f"Using JAX device: {dev}")
    return dev


def tree_to_device(tree, device):
    return jax.device_put(tree, device)


def soft_update(params, target_params, tau: float):
    return optax.incremental_update(params, target_params, tau)


def hard_update(params):
    return params


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
    return (states - mean) / std


def _query_knn_distances_numpy(
    query_states: np.ndarray,
    reference_states: np.ndarray,
    k: int,
    query_chunk_size: int = 4096,
    reference_chunk_size: int = 32768,
) -> np.ndarray:
    """Memory-safe NumPy fallback for approximate kNN distances.

    This is only used when scikit-learn is unavailable. It computes distances to a
    reference subset in chunks, so it is slower than sklearn but avoids adding a hard
    dependency to the training script.
    """
    k = min(k, reference_states.shape[0])
    kth_distances = np.empty(query_states.shape[0], dtype=np.float32)

    for q_start in range(0, query_states.shape[0], query_chunk_size):
        q_end = min(q_start + query_chunk_size, query_states.shape[0])
        q = query_states[q_start:q_end].astype(np.float32, copy=False)
        best = np.full((q.shape[0], k), np.inf, dtype=np.float32)

        q_norm = np.sum(q * q, axis=1, keepdims=True)
        for r_start in range(0, reference_states.shape[0], reference_chunk_size):
            r_end = min(r_start + reference_chunk_size, reference_states.shape[0])
            r = reference_states[r_start:r_end].astype(np.float32, copy=False)
            r_norm = np.sum(r * r, axis=1)[None, :]
            dist_sq = np.maximum(q_norm + r_norm - 2.0 * (q @ r.T), 0.0)

            # Keep the current k smallest squared distances for this query chunk.
            merged = np.concatenate([best, dist_sq], axis=1)
            kth_idx = min(k - 1, merged.shape[1] - 1)
            best = np.partition(merged, kth_idx, axis=1)[:, :k]

        kth_distances[q_start:q_end] = np.sqrt(np.max(best, axis=1))

    return kth_distances


def compute_state_coverage_confidence(
    states: np.ndarray,
    k: int = 10,
    reference_size: int = 100_000,
    low_quantile: float = 0.20,
    high_quantile: float = 0.80,
    seed: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Estimate state coverage confidence c(s) in [0, 1] from local density.

    The input states should already be normalized. We use the kNN distance as an
    inverse density estimate:
      small kth-neighbor distance -> dense state  -> confidence near 1
      large kth-neighbor distance -> sparse state -> confidence near 0

    For large D4RL datasets, kNN is computed against a random reference subset.
    This gives a cheap and stable density proxy without changing the online JAX
    training step.
    """
    states = np.asarray(states, dtype=np.float32)
    n_states = states.shape[0]
    if n_states == 0:
        return np.empty((0,), dtype=np.float32)
    if n_states <= 2:
        return np.ones((n_states,), dtype=np.float32)

    reference_size = min(int(reference_size), n_states)
    rng = np.random.default_rng(seed)
    if reference_size < n_states:
        reference_indices = rng.choice(n_states, size=reference_size, replace=False)
        reference_states = states[reference_indices]
        # With a reference subset, the query state may not be in the reference set.
        n_neighbors = min(int(k), reference_size)
    else:
        reference_states = states
        # Exact self-query includes the point itself as the nearest neighbor, so
        # use k + 1 neighbors and take the last distance.
        n_neighbors = min(int(k) + 1, reference_size)

    if n_neighbors <= 1:
        return np.ones((n_states,), dtype=np.float32)

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", n_jobs=-1)
        nn.fit(reference_states)
        distances, _ = nn.kneighbors(states, return_distance=True)
        kth_distances = distances[:, -1].astype(np.float32)
    except Exception as exc:
        print(
            "scikit-learn kNN is unavailable or failed. Falling back to a slower "
            f"NumPy kNN implementation. Original error: {exc}"
        )
        kth_distances = _query_knn_distances_numpy(
            query_states=states,
            reference_states=reference_states,
            k=n_neighbors,
        )

    q_low, q_high = np.quantile(kth_distances, [low_quantile, high_quantile])
    if q_high <= q_low + eps:
        coverage_conf = np.ones_like(kth_distances, dtype=np.float32)
    else:
        # distance <= q_low  -> dense  -> 1
        # distance >= q_high -> sparse -> 0
        coverage_conf = 1.0 - (kth_distances - q_low) / (q_high - q_low)
        coverage_conf = np.clip(coverage_conf, 0.0, 1.0).astype(np.float32)

    print(
        "State coverage confidence: "
        f"mean={coverage_conf.mean():.4f}, min={coverage_conf.min():.4f}, "
        f"max={coverage_conf.max():.4f}, "
        f"k={k}, reference_size={reference_size}, "
        f"distance_q_low={q_low:.6f}, distance_q_high={q_high:.6f}"
    )
    return coverage_conf.astype(np.float32)



def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: Any,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self._state_coverage_conf = np.ones((buffer_size, 1), dtype=np.float32)
        self._device = device

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

        self._states[:n_transitions] = data["observations"].astype(np.float32)
        self._actions[:n_transitions] = data["actions"].astype(np.float32)
        self._rewards[:n_transitions] = data["rewards"][..., None].astype(np.float32)
        self._next_states[:n_transitions] = data["next_observations"].astype(np.float32)
        self._dones[:n_transitions] = data["terminals"][..., None].astype(np.float32)
        if "state_coverage_conf" in data:
            self._state_coverage_conf[:n_transitions] = data["state_coverage_conf"][..., None].astype(np.float32)
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
            "state_coverage_conf": self._state_coverage_conf[indices],
        }
        return tree_to_device({k: jnp.asarray(v) for k, v in batch.items()}, self._device)


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_scalar_value(value: Any) -> bool:
    if isinstance(value, (int, float, bool, np.number)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return True
    return False


def to_python_scalar(value: Any) -> Union[int, float, bool]:
    if isinstance(value, np.ndarray):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jnp.ndarray) and value.ndim == 0:
        return float(value)
    return value


def prefix_log_keys(log: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in log.items()}


def save_logs_npz(logs: List[Dict[str, Any]], path: str) -> None:
    if len(logs) == 0:
        return
    keys = logs[0].keys()
    data_to_save: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [log[key] for log in logs]
        try:
            data_to_save[key] = np.asarray(values)
        except ValueError:
            data_to_save[key] = np.asarray(values, dtype=object)
    np.savez(path, **data_to_save)


def save_and_upload_eval_logs(
    eval_logs: List[Dict[str, Any]],
    checkpoints_path: Optional[str],
    log_wandb: bool,
):
    if checkpoints_path is None or len(eval_logs) == 0:
        return
    eval_logs_path = os.path.join(checkpoints_path, "eval_logs.npz")
    save_logs_npz(eval_logs, eval_logs_path)
    if log_wandb and wandb.run is not None:
        wandb.save(eval_logs_path, policy="now")


def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
        dtype=np.float32,
    )


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        # Preserves the reward preprocessing in the original main training code.
        dataset["rewards"] *= 100.0


class Actor(nn.Module):
    action_dim: int
    max_action: float
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return self.max_action * x


class QFunction(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class ValueFunction(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


@struct.dataclass
class CDAFState:
    total_it: jnp.ndarray
    q_params: Any
    q_target_params: Any
    q_delayed_params: Any
    q_opt_state: Any
    v_params: Any
    v_target_params: Any
    v_delayed_params: Any
    v_opt_state: Any


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any


class CDAFJAX:
    """Conservative Delayed Advantage Filtering (CDAF) for offline RL.

    Q, V, and policy are optimized during the same online training loop.
    The policy update uses a TD3+BC-style or advantage-weighted objective.
    """
    def __init__(
        self,
        max_action: float,
        state_dim: int,
        action_dim: int,
        max_steps: int,
        qf_lr: float = 3e-4,
        vf_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        delayed_update_period: int = 250,
        min_weight_exponent: float = 0.0,
        max_weight_exponent: float = 2.0,
        weight_logit_clip: float = 10.0,
        beta_min: float = 0.0,
        use_coverage_aware_beta: bool = True,
        adv_margin: float = 0.0,
        actor_update_method: str = "td3_bc",
        policy_weight_exponent: float = 1.0,
        policy_weight_clip: float = 20.0,
        alpha: float = 2.5,
        bc_coef: float = 1.0,
        seed: int = 0,
        device: Any = None,
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.discount = discount
        self.tau = tau
        self.delayed_update_period = delayed_update_period
        self.min_weight_exponent = min_weight_exponent
        self.max_weight_exponent = max_weight_exponent
        self.weight_logit_clip = weight_logit_clip
        self.beta_min = beta_min
        self.use_coverage_aware_beta = use_coverage_aware_beta
        self.adv_margin = adv_margin
        self.actor_update_method = actor_update_method
        self.policy_weight_exponent = policy_weight_exponent
        self.policy_weight_clip = policy_weight_clip
        self.alpha = alpha
        self.bc_coef = bc_coef
        self.device = device if device is not None else jax.devices()[0]

        self.actor_def = Actor(action_dim=action_dim, max_action=max_action)
        self.q_def = QFunction()
        self.v_def = ValueFunction()

        self.q_tx = optax.adam(qf_lr)
        self.v_tx = optax.adam(vf_lr)
        self.actor_tx = optax.adam(actor_lr)

        key = jax.random.PRNGKey(seed)
        key_actor, key_q, key_v = jax.random.split(key, 3)
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state)["params"]
        q_params = self.q_def.init(key_q, dummy_state, dummy_action)["params"]
        v_params = self.v_def.init(key_v, dummy_state)["params"]

        self.state = CDAFState(
            total_it=jnp.asarray(0, dtype=jnp.int32),
            q_params=q_params,
            q_target_params=copy.deepcopy(q_params),
            q_delayed_params=copy.deepcopy(q_params),
            q_opt_state=self.q_tx.init(q_params),
            v_params=v_params,
            v_target_params=copy.deepcopy(v_params),
            v_delayed_params=copy.deepcopy(v_params),
            v_opt_state=self.v_tx.init(v_params),
        )
        self.actor_state = ActorState(
            params=actor_params,
            opt_state=self.actor_tx.init(actor_params),
        )

        self.state = tree_to_device(self.state, self.device)
        self.actor_state = tree_to_device(self.actor_state, self.device)

        self._train_step = self._build_train_step()

    def _build_train_step(self):
        actor_apply = self.actor_def.apply
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        actor_tx = self.actor_tx
        discount = self.discount
        tau = self.tau
        delayed_update_period = self.delayed_update_period
        min_weight_exponent = self.min_weight_exponent
        max_weight_exponent = self.max_weight_exponent
        weight_logit_clip = self.weight_logit_clip
        beta_min = self.beta_min
        use_coverage_aware_beta = self.use_coverage_aware_beta
        adv_margin = self.adv_margin
        max_steps = self.max_steps
        actor_update_method = self.actor_update_method
        policy_weight_exponent = self.policy_weight_exponent
        policy_weight_clip = self.policy_weight_clip
        alpha = self.alpha
        bc_coef = self.bc_coef

        @jax.jit
        def train_step(state: CDAFState, actor_state: ActorState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            def q_loss_fn(q_params):
                next_v = v_apply({"params": state.v_target_params}, next_observations)
                target_q = rewards + (1.0 - dones) * discount * next_v
                q = q_apply({"params": q_params}, observations, actions)
                q_loss = jnp.mean((q - jax.lax.stop_gradient(target_q)) ** 2)
                return q_loss, (q, target_q)

            (q_loss, (q, target_q)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
            q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
            q_params = optax.apply_updates(state.q_params, q_updates)

            progress = jnp.minimum(total_it.astype(jnp.float32) / jnp.maximum(float(max_steps), 1.0), 1.0)
            exponent = min_weight_exponent + (max_weight_exponent - min_weight_exponent) * progress

            delayed_q = q_apply({"params": state.q_delayed_params}, observations, actions)
            delayed_v = v_apply({"params": state.v_delayed_params}, observations)
            raw_delayed_adv = delayed_q - delayed_v
            delayed_adv = jnp.clip(raw_delayed_adv, -weight_logit_clip, weight_logit_clip)

            # Original CDAF filters every negative advantage. The margin delays
            # filtering until the negative advantage is large enough to be trusted:
            #   delayed_adv >= -adv_margin -> beta_adv = 1
            #   delayed_adv <  -adv_margin -> beta_adv decreases smoothly
            shifted_negative_adv = delayed_adv + adv_margin
            beta_adv = jnp.where(
                delayed_adv < -adv_margin,
                jnp.exp(exponent * shifted_negative_adv),
                jnp.ones_like(delayed_adv),
            )
            beta_adv = jnp.maximum(beta_adv, beta_min)

            if use_coverage_aware_beta:
                coverage_conf = jnp.clip(jnp.squeeze(batch["state_coverage_conf"], axis=-1), 0.0, 1.0)
            else:
                coverage_conf = jnp.ones_like(beta_adv)

            # coverage_conf c(s) gates how much we trust advantage filtering.
            #   sparse state: c(s) ~= 0 -> beta ~= 1 -> V(s) learns from Q(s, a)
            #   dense state:  c(s) ~= 1 -> beta ~= beta_adv -> original CDAF
            filter_strength = coverage_conf * progress
            beta = 1.0 - filter_strength * (1.0 - beta_adv)
            beta = jnp.maximum(beta, beta_min)

            def v_loss_fn(v_params):
                target_v_q = q_apply({"params": state.q_target_params}, observations, actions)
                v = v_apply({"params": v_params}, observations)
                value_residual = v - jax.lax.stop_gradient(target_v_q)
                value_loss = jnp.mean(jax.lax.stop_gradient(beta) * value_residual ** 2)
                return value_loss, (v, target_v_q)

            (value_loss, (v, target_v_q)), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            def actor_loss_fn(actor_params):
                pi = actor_apply({"params": actor_params}, observations)
                bc_per_sample = jnp.mean((pi - actions) ** 2, axis=-1)
                bc_loss = jnp.mean(bc_per_sample)

                # Dataset-action advantage weights for weighted BC variants.
                # We use the learned CDAF value V(s) as the baseline, consistent
                # with the training-time advantage Q(s,a)-V(s).
                if actor_update_method in ("weighted_bc", "td3_weighted_bc"):
                    data_q = q_apply({"params": q_params}, observations, actions)
                    data_v = v_apply({"params": v_params}, observations)
                    data_adv = jnp.clip(data_q - data_v, -weight_logit_clip, weight_logit_clip)
                    policy_weight = jnp.exp(policy_weight_exponent * data_adv)
                    policy_weight = policy_weight / jnp.maximum(jnp.mean(policy_weight), 1e-6)
                    policy_weight = jnp.minimum(policy_weight, policy_weight_clip)
                    weighted_bc_loss = jnp.mean(jax.lax.stop_gradient(policy_weight) * bc_per_sample)
                    weight_mean = jnp.mean(policy_weight)
                    weight_max = jnp.max(policy_weight)
                else:
                    data_q = jnp.zeros_like(bc_per_sample)
                    weighted_bc_loss = bc_loss
                    weight_mean = jnp.asarray(1.0)
                    weight_max = jnp.asarray(1.0)

                if actor_update_method == "weighted_bc":
                    actor_loss = weighted_bc_loss
                    q_for_log = data_q
                    lmbda = jnp.asarray(0.0)
                else:
                    q_pi = q_apply({"params": q_params}, observations, pi)
                    lmbda = jax.lax.stop_gradient(alpha / jnp.maximum(jnp.mean(jnp.abs(q_pi)), 1e-6))
                    if actor_update_method == "td3_weighted_bc":
                        bc_regularizer = weighted_bc_loss
                    else:
                        bc_regularizer = bc_loss
                    actor_loss = -lmbda * jnp.mean(q_pi) + bc_coef * bc_regularizer
                    q_for_log = q_pi

                return actor_loss, (q_for_log, lmbda, bc_loss, weighted_bc_loss, weight_mean, weight_max)

            (actor_loss, (actor_q, actor_lmbda, actor_bc_loss, actor_weighted_bc_loss, actor_weight_mean, actor_weight_max)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
            actor_updates, actor_opt_state = actor_tx.update(actor_grads, actor_state.opt_state, actor_state.params)
            actor_params = optax.apply_updates(actor_state.params, actor_updates)
            new_actor_state = ActorState(params=actor_params, opt_state=actor_opt_state)

            q_target_params = soft_update(q_params, state.q_target_params, tau)
            v_target_params = soft_update(v_params, state.v_target_params, tau)

            should_update_delayed = (total_it % delayed_update_period) == 0
            q_delayed_params = jax.lax.cond(
                should_update_delayed,
                lambda _: q_target_params,
                lambda _: state.q_delayed_params,
                operand=None,
            )
            v_delayed_params = jax.lax.cond(
                should_update_delayed,
                lambda _: v_target_params,
                lambda _: state.v_delayed_params,
                operand=None,
            )

            new_state = CDAFState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_delayed_params=q_delayed_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_target_params=v_target_params,
                v_delayed_params=v_delayed_params,
                v_opt_state=v_opt_state,
            )

            log_dict = {
                "q_loss": q_loss,
                "q_mean": jnp.mean(q),
                "target_q_mean": jnp.mean(target_q),
                "value_loss": value_loss,
                "v_mean": jnp.mean(v),
                "target_v_q_mean": jnp.mean(target_v_q),
                "actor_loss": actor_loss,
                "actor_q_mean": jnp.mean(actor_q),
                "actor_lambda": actor_lmbda,
                "actor_bc_loss": actor_bc_loss,
                "actor_weighted_bc_loss": actor_weighted_bc_loss,
                "actor_policy_weight_mean": actor_weight_mean,
                "actor_policy_weight_max": actor_weight_max,
                "actor_bc_coef": jnp.asarray(bc_coef),
                "beta_mean": jnp.mean(beta),
                "beta_min": jnp.min(beta),
                "beta_max": jnp.max(beta),
                "beta_adv_mean": jnp.mean(beta_adv),
                "coverage_conf_mean": jnp.mean(coverage_conf),
                "coverage_conf_min": jnp.min(coverage_conf),
                "coverage_conf_max": jnp.max(coverage_conf),
                "filter_strength_mean": jnp.mean(filter_strength),
                "adv_margin": jnp.asarray(adv_margin),
                "weight_exponent": exponent,
                "delayed_adv_mean": jnp.mean(delayed_adv),
                "delayed_adv_min": jnp.min(delayed_adv),
                "delayed_adv_max": jnp.max(delayed_adv),
                "raw_delayed_adv_mean": jnp.mean(raw_delayed_adv),
                "raw_delayed_adv_min": jnp.min(raw_delayed_adv),
                "raw_delayed_adv_max": jnp.max(raw_delayed_adv),
                "clipped_low_frac": jnp.mean((raw_delayed_adv <= -weight_logit_clip).astype(jnp.float32)),
                "clipped_high_frac": jnp.mean((raw_delayed_adv >= weight_logit_clip).astype(jnp.float32)),
                "negative_adv_frac": jnp.mean((raw_delayed_adv < 0.0).astype(jnp.float32)),
                "filtered_adv_frac": jnp.mean((delayed_adv < -adv_margin).astype(jnp.float32)),
            }
            return new_state, new_actor_state, log_dict

        return train_step

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.state, self.actor_state, log_dict = self._train_step(self.state, self.actor_state, batch)
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
        action = self.actor_def.apply({"params": actor_params}, state_jnp)
        return np.asarray(jax.device_get(action))[0]

    def eval_actor(self, env: gym.Env, actor_params: Any, n_episodes: int, seed: int) -> np.ndarray:
        env.seed(seed)
        episode_rewards = []
        for _ in range(n_episodes):
            state, done = env.reset(), False
            episode_reward = 0.0
            while not done:
                action = self.actor_act(actor_params, state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return np.asarray(episode_rewards, dtype=np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cdaf_state": serialization.to_state_dict(self.state),
            "actor_state": serialization.to_state_dict(self.actor_state),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state = serialization.from_state_dict(self.state, state_dict["cdaf_state"])
        self.actor_state = serialization.from_state_dict(self.actor_state, state_dict["actor_state"])
        self.state = tree_to_device(self.state, self.device)
        self.actor_state = tree_to_device(self.actor_state, self.device)


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def resolve_checkpoint_path(
    load_model: Union[str, Path],
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Return (run_dir, checkpoint_path) for a saved CDAF checkpoint.

    Supported load_model formats:

    1. Direct checkpoint file:
       path/to/checkpoint.pkl

    2. Direct run directory:
       path/to/run_dir/
       where path/to/run_dir/checkpoint.pkl exists

    3. Parent directory that contains env/seed subdirectory:
       path/to/base_dir/
       where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists

    Example:
       --load_model logs/tuning/cdaf_jax/0.2/0.1

       resolves to:
       logs/tuning/cdaf_jax/0.2/0.1/CDAF-JAX-antmaze-medium-play-v2/0/checkpoint.pkl
    """
    load_path = Path(load_model)

    if load_path.is_file():
        if load_path.name != "checkpoint.pkl":
            raise FileNotFoundError(
                f"load_model points to a file, but it is not checkpoint.pkl: {load_path}"
            )
        return load_path.parent, load_path

    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")

    candidates: List[Path] = []

    # Case 1: load_model is already the run directory.
    candidates.append(load_path / "checkpoint.pkl")

    # Case 2: load_model is a parent directory containing {run_name}/{seed}/checkpoint.pkl.
    if run_name is not None and seed is not None:
        candidates.append(load_path / run_name / str(seed) / "checkpoint.pkl")

    # Case 3: load_model contains {run_name}/*/checkpoint.pkl.
    if run_name is not None:
        run_name_dir = load_path / run_name
        if run_name_dir.exists():
            candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pkl")))

    # Case 4: fallback search, but only 2 levels deep to avoid accidentally
    # scanning unrelated large folders.
    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))

    # Deduplicate while preserving order.
    seen = set()
    existing_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            existing_candidates.append(candidate)

    if len(existing_candidates) == 0:
        tried = "\n".join(str(p) for p in candidates[:20])
        raise FileNotFoundError(
            f"checkpoint file not found under: {load_path}\n"
            f"Tried candidates:\n{tried}"
        )

    # Prefer exact {run_name}/{seed}/checkpoint.pkl if available.
    if run_name is not None and seed is not None:
        exact = (load_path / run_name / str(seed) / "checkpoint.pkl").resolve()
        if exact in existing_candidates:
            return exact.parent, exact

    if len(existing_candidates) > 1:
        found = "\n".join(str(p) for p in existing_candidates)
        raise FileNotFoundError(
            f"Multiple checkpoint.pkl files found under {load_path}.\n"
            f"Please provide a more specific --load_model path.\n"
            f"Found:\n{found}"
        )

    checkpoint_path = existing_candidates[0]
    return checkpoint_path.parent, checkpoint_path


@pyrallis.wrap()
def train(config: TrainConfig):
    config = apply_env_hyperparams(config)
    config = finalize_checkpoint_path(config)

    jax_device = select_jax_device(config.device)
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)

    if config.use_coverage_aware_beta:
        dataset["state_coverage_conf"] = compute_state_coverage_confidence(
            dataset["observations"],
            k=config.coverage_knn_k,
            reference_size=config.coverage_reference_size,
            low_quantile=config.coverage_low_quantile,
            high_quantile=config.coverage_high_quantile,
            seed=config.seed,
        )
    else:
        dataset["state_coverage_conf"] = np.ones(dataset["observations"].shape[0], dtype=np.float32)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=jax_device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        config_path = os.path.join(config.checkpoints_path, "config.yaml")
        if os.path.exists(config_path):
            print(f"Error: The file '{config_path}' already exists.")
            exit(1)
        with open(config_path, "w") as f:
            pyrallis.dump(config, f)

    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    print(f"Training {config.name}-JAX, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = CDAFJAX(
        max_action=max_action,
        state_dim=state_dim,
        action_dim=action_dim,
        max_steps=config.max_timesteps,
        qf_lr=config.qf_lr,
        vf_lr=config.vf_lr,
        actor_lr=config.actor_lr,
        discount=config.discount,
        tau=config.tau,
        delayed_update_period=config.delayed_update_period,
        min_weight_exponent=config.min_weight_exponent,
        max_weight_exponent=config.max_weight_exponent,
        weight_logit_clip=config.weight_logit_clip,
        beta_min=config.beta_min,
        use_coverage_aware_beta=config.use_coverage_aware_beta,
        adv_margin=config.adv_margin,
        actor_update_method=config.actor_update_method,
        policy_weight_exponent=config.policy_weight_exponent,
        policy_weight_clip=config.policy_weight_clip,
        alpha=config.alpha,
        bc_coef=config.bc_coef,
        seed=seed,
        device=jax_device,
    )

    if config.load_model != "":
        _, checkpoint_path = resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        trainer.load_state_dict(checkpoint)

    if config.log_wandb:
        wandb_init(asdict(config))


    eval_logs: List[Dict[str, Any]] = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)

        if config.log_wandb and (t + 1) % config.log_every == 0:
            wandb.log(log_dict, step=int(jax.device_get(trainer.state.total_it)))

        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = trainer.eval_actor(
                env,
                trainer.actor_state.params,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            normalized_eval_scores = normalize_episode_scores(env, eval_scores)
            eval_log: Dict[str, Any] = {
                "timestep": int(t + 1),
                "eval/reward_mean": float(np.mean(eval_scores)),
                "eval/reward_std": float(np.std(eval_scores)),
                "eval/normalized_score_mean": float(np.mean(normalized_eval_scores)),
                "eval/normalized_score_std": float(np.std(normalized_eval_scores)),
            }
            eval_logs.append(eval_log.copy())
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"reward_mean={eval_log['eval/reward_mean']:.3f}, "
                f"reward_std={eval_log['eval/reward_std']:.3f}, "
                f"D4RL_mean={eval_log['eval/normalized_score_mean']:.3f}, "
                f"D4RL_std={eval_log['eval/normalized_score_std']:.3f}"
            )

            if config.log_wandb:
                wandb_eval_log = {
                    key: to_python_scalar(value)
                    for key, value in eval_log.items()
                    if is_scalar_value(value)
                }
                wandb.log(wandb_eval_log, step=int(jax.device_get(trainer.state.total_it)))

            save_and_upload_eval_logs(
                eval_logs=eval_logs,
                checkpoints_path=config.checkpoints_path,
                log_wandb=config.log_wandb,
            )

    if config.checkpoints_path is not None:
        save_pickle(
            os.path.join(config.checkpoints_path, "checkpoint.pkl"),
            trainer.state_dict(),
        )

        if config.log_wandb and wandb.run is not None:
            wandb.save(os.path.join(config.checkpoints_path, "checkpoint.pkl"), policy="now")

        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()