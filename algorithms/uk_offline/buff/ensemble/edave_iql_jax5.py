# JAX/Flax EDAVE-IQL implementation with CDAF_JAX-style experiment plumbing.
# EDAVE-IQL = Ensemble Double Advantage-filtered Value Estimation IQL.
#
# Main idea:
#   Use an ensemble of n coupled (Q_i, V_i) pairs.
#   - Q_i is always backed up with its own V_i target.
#   - V_i is regressed to its own Q_i target.
#   - The V_i filtering weight is computed using a delayed pair (Q_j^d, V_j^d)
#     where j != i whenever n > 1. This extends Double Q-style decoupling
#     to an ensemble without min/max over critics.
#   - The actor is updated using the mean ensemble advantage.
#
# No max/min over critics is used anywhere.

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

ALGORITHM_NAME = "EDAVE-IQL"
ALGORITHM_FULL_NAME = "Ensemble Double Advantage-filtered Value Estimation IQL"

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

IQL_ACTOR_METHODS = ("iql", "iql_awbc")
ACTOR_UPDATE_METHODS = ("td3_bc", "weighted_bc", "td3_weighted_bc", *IQL_ACTOR_METHODS)
ACTOR_REFIT_METHODS = IQL_ACTOR_METHODS


@dataclass
class TrainConfig:
    # Experiment
    device: str = "gpu"  # one of: cpu, gpu, tpu. JAX selects the matching backend when available.
    env: str = "halfcheetah-medium-expert-v2"
    seed: int = 0
    # Shared by both modes:
    #   mode="train": evaluate pi every eval_freq joint Q/V/pi training steps.
    #   mode="refit": evaluate pi every eval_freq actor-only refit steps.
    eval_freq: int = int(25e3)
    n_episodes: int = 10
    # Shared by both modes:
    #   mode="train": number of joint Q/V/pi training steps.
    #   mode="refit": number of actor-only refit steps using frozen loaded Q/V.
    max_timesteps: int = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"  # one of: train, refit. refit loads Q/V and trains only pi.
    hyperparams_path: Optional[str] = "hyperparams/edave_iql_jax.yml"
    use_hyperparams: bool = True

    # Dataset
    buffer_size: int = 2_000_000
    # Shared by both modes:
    #   mode="train": minibatch size for joint Q/V/pi updates.
    #   mode="refit": minibatch size for actor-only refit updates.
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False

    # EDAVE-IQL value/Q learning
    discount: float = 0.99
    tau: float = 0.005
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    iql_deterministic: bool = False
    actor_dropout: Optional[float] = None

    # Ensemble double Q/V learning.
    # There are n (Q_i, V_i) pairs. Q_i is backed up with V_i, and V_i is
    # regressed to Q_i. The filtering weight for V_i is computed from delayed
    # Q_j/V_j with j != i when n > 1.
    ensemble_size: int = 4
    delayed_update_freq: int = 1000
    v_filter_type: str = "exp"  # one of: sigmoid, exp
    # Inverse-temperature style multiplier for V filtering logits.
    # Old v_filter_temperature=T is equivalent to v_filter_exponent=1/T.
    v_filter_exponent: float = 1.0 / 3.0
    v_filter_clip: float = 3.0
    v_filter_floor: float = 0.2
    # If true, positive filtered advantages are not upweighted: A_filter >= 0 -> w_V = 1.
    # Negative advantages are still downweighted by the selected filter.
    v_filter_positive_adv_weight_one: bool = False

    # Actor policy learning. In mode="train", actor is updated jointly with Q/V.
    # In mode="refit", Q/V are frozen from a checkpoint and only pi is optimized.
    # Training supports the full CDAF-style actor set below. Refit intentionally
    # supports only iql / iql_awbc.
    #   td3_bc:          -lambda * mean_i Q_i(s, pi(s)) + plain BC
    #   weighted_bc:     delayed ensemble-advantage weighted BC only
    #   td3_weighted_bc: -lambda * mean_i Q_i(s, pi(s)) + weighted BC
    #   iql / iql_awbc:  IQL-style advantage-weighted BC from dataset actions
    actor_update_method: str = "iql"
    policy_weight_exponent: float = 1.0
    policy_weight_clip: float = 20.0
    alpha: float = 2.5
    bc_coef: float = 1.0


    # Stability. This is not a min/max critic trick; it only clips gradients.
    grad_clip_norm: float = 10.0

    # Standalone actor refit output directory.
    # Refit reuses the shared training schedule fields above:
    #   max_timesteps -> actor-only refit steps
    #   batch_size    -> actor-only refit batch size
    #   eval_freq     -> actor-only refit evaluation interval
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-BIAS"
    group: str = "EDAVE-IQL-JAX"
    name: str = "EDAVE-IQL-JAX"
    log_wandb: bool = True
    log_every: int = 500

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)

def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-BIAS"
    config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.mode in ("train", "refit"), "mode must be train or refit"
    assert config.batch_size > 0
    assert config.buffer_size > 0
    assert config.eval_freq > 0
    assert config.n_episodes > 0
    assert config.max_timesteps >= 0
    assert config.log_every > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    if config.actor_dropout is not None:
        assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
    assert config.ensemble_size >= 1
    assert config.delayed_update_freq > 0
    assert config.v_filter_type in ("sigmoid", "exp")
    assert config.v_filter_exponent > 0.0
    assert config.v_filter_clip > 0.0
    assert config.v_filter_floor >= 0.0 and config.v_filter_floor <= 1.0
    assert isinstance(config.v_filter_positive_adv_weight_one, bool)
    assert config.actor_update_method in ACTOR_UPDATE_METHODS, (
        "actor_update_method must be one of: "
        f"{', '.join(ACTOR_UPDATE_METHODS)}"
    )
    assert config.policy_weight_exponent >= 0.0
    assert config.policy_weight_clip > 0.0
    assert config.alpha >= 0.0
    assert config.bc_coef >= 0.0
    assert config.grad_clip_norm > 0.0
    assert config.actor_refit_dir_name != ""
    if config.mode == "refit":
        assert config.load_model != "", "mode='refit' requires --load_model"
        assert config.actor_update_method in ACTOR_REFIT_METHODS, (
            "mode='refit' supports only actor_update_method in: "
            f"{', '.join(ACTOR_REFIT_METHODS)}"
        )

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
    """Load env-specific hyperparameters and merge them into config.

    Priority is:
        dataclass defaults < hyperparams YAML < explicit CLI flags
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
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []

    for raw_key, raw_value in env_hyperparams.items():
        key = raw_key
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
        print(f"Ignored unknown hyperparameter keys for EDAVE-IQL: {', '.join(skipped_unknown)}")
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


def stack_ensemble_params(params_sequence):
    """Convert a sequence of per-head parameter PyTrees into a batched PyTree.

    Each leaf receives a leading ensemble dimension. This lets JAX map a single
    Q/V module over all ensemble heads with vmap, rather than storing the
    ensemble as a Python tuple of independent parameter trees.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_sequence)


def _is_serialized_ensemble_sequence(value: Any) -> bool:
    """Detect Flax's serialized form for an old tuple/list ensemble."""
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(str(key).isdigit() for key in value.keys())
    )


def _stack_serialized_ensemble_sequence(value: Dict[Any, Any]):
    ordered_keys = sorted(value.keys(), key=lambda key: int(str(key)))
    ordered_values = [value[key] for key in ordered_keys]
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *ordered_values,
    )


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
    return (states - mean) / std


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
    # Preserves the reward preprocessing from the provided PyTorch IQL code.
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class PolicyMLP(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    n_hidden: int = 2
    dropout: Optional[float] = None

    @nn.compact
    def __call__(self, state: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = state
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            if self.dropout is not None:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)


class GaussianPolicy(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    n_hidden: int = 2
    dropout: Optional[float] = None

    @nn.compact
    def __call__(self, state: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean = PolicyMLP(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
        )(state, training=training)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class DeterministicPolicy(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    n_hidden: int = 2
    dropout: Optional[float] = None

    @nn.compact
    def __call__(self, state: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return PolicyMLP(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
        )(state, training=training)


class QFunction(nn.Module):
    hidden_dim: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)



class ValueFunction(nn.Module):
    hidden_dim: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)



@struct.dataclass
class EDAVEState:
    total_it: jnp.ndarray
    q_params: Any
    q_target_params: Any
    q_delay_params: Any
    q_opt_state: Any
    v_params: Any
    v_target_params: Any
    v_delay_params: Any
    v_opt_state: Any
    filter_indices: jnp.ndarray
    filter_key: jnp.ndarray


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


class EDAVEIQLJAX:
    """Ensemble Double Advantage-filtered Value Estimation IQL in JAX/Flax.

    Main algorithm:
      - Maintain n pairs (Q_i, V_i).
      - Q_i target: r + gamma * V_i_target(s').
      - V_i target: Q_i_target(s, a).
      - V_i filtering weight: computed from delayed Q_j/V_j where j != i for n > 1.
      - The mapping i -> j is one-to-one and refreshed with delayed networks.
      - Actor update is performed jointly with Q/V in mode="train" and actor-only in mode="refit".
      - Training actor losses support td3_bc, weighted_bc, td3_weighted_bc, iql, and iql_awbc.
      - Actor refit supports only iql and iql_awbc.

    No max/min over critics is used anywhere.
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
        iql_deterministic: bool = False,
        actor_dropout: Optional[float] = None,
        ensemble_size: int = 4,
        delayed_update_freq: int = 1000,
        v_filter_type: str = "sigmoid",
        v_filter_exponent: float = 1.0 / 3.0,
        v_filter_clip: float = 3.0,
        v_filter_floor: float = 0.2,
        v_filter_positive_adv_weight_one: bool = False,
        actor_update_method: str = "td3_weighted_bc",
        policy_weight_exponent: float = 1.0,
        policy_weight_clip: float = 20.0,
        alpha: float = 2.5,
        bc_coef: float = 1.0,
        grad_clip_norm: float = 10.0,
        seed: int = 0,
        device: Any = None,
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.discount = discount
        self.tau = tau
        self.iql_deterministic = iql_deterministic
        self.actor_dropout = actor_dropout
        self.ensemble_size = int(ensemble_size)
        self.delayed_update_freq = int(delayed_update_freq)
        self.v_filter_type = v_filter_type
        self.v_filter_exponent = float(v_filter_exponent)
        self.v_filter_clip = float(v_filter_clip)
        self.v_filter_floor = float(v_filter_floor)
        self.v_filter_positive_adv_weight_one = bool(v_filter_positive_adv_weight_one)
        if actor_update_method not in ACTOR_UPDATE_METHODS:
            raise ValueError(f"actor_update_method must be one of: {', '.join(ACTOR_UPDATE_METHODS)}")
        self.actor_update_method = actor_update_method
        self.policy_weight_exponent = float(policy_weight_exponent)
        self.policy_weight_clip = float(policy_weight_clip)
        self.alpha = float(alpha)
        self.bc_coef = float(bc_coef)
        self.grad_clip_norm = float(grad_clip_norm)
        self.device = device if device is not None else jax.devices()[0]

        if iql_deterministic:
            self.actor_def = DeterministicPolicy(action_dim=action_dim, dropout=actor_dropout)
        else:
            self.actor_def = GaussianPolicy(action_dim=action_dim, dropout=actor_dropout)
        self.q_def = QFunction()
        self.v_def = ValueFunction()

        def make_tx(lr):
            return optax.chain(optax.clip_by_global_norm(self.grad_clip_norm), optax.adam(lr))

        self.q_tx = make_tx(qf_lr)
        self.v_tx = make_tx(vf_lr)
        actor_lr_schedule = optax.cosine_decay_schedule(
            init_value=actor_lr,
            decay_steps=max(int(max_steps), 1),
            alpha=0.0,
        )
        self.actor_tx = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adam(actor_lr_schedule),
        )

        key = jax.random.PRNGKey(seed)
        key_actor, key_q, key_v, actor_key, filter_key = jax.random.split(key, 5)
        q_keys = jax.random.split(key_q, self.ensemble_size)
        v_keys = jax.random.split(key_v, self.ensemble_size)
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
        q_params = stack_ensemble_params(
            [
                self.q_def.init(q_keys[i], dummy_state, dummy_action)["params"]
                for i in range(self.ensemble_size)
            ]
        )
        v_params = stack_ensemble_params(
            [
                self.v_def.init(v_keys[i], dummy_state)["params"]
                for i in range(self.ensemble_size)
            ]
        )

        self.initial_actor_params = copy.deepcopy(actor_params)
        self.initial_actor_opt_state = self.actor_tx.init(actor_params)
        self.initial_actor_key = actor_key

        self.state = EDAVEState(
            total_it=jnp.asarray(0, dtype=jnp.int32),
            q_params=q_params,
            q_target_params=copy.deepcopy(q_params),
            q_delay_params=copy.deepcopy(q_params),
            q_opt_state=self.q_tx.init(q_params),
            v_params=v_params,
            v_target_params=copy.deepcopy(v_params),
            v_delay_params=copy.deepcopy(v_params),
            v_opt_state=self.v_tx.init(v_params),
            filter_indices=self._initial_filter_indices(),
            filter_key=filter_key,
        )
        self.actor_state = ActorState(
            params=actor_params,
            opt_state=copy.deepcopy(self.initial_actor_opt_state),
            key=actor_key,
        )

        self.state = tree_to_device(self.state, self.device)
        self.actor_state = tree_to_device(self.actor_state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
        self._train_step = self._build_train_step()
        self._actor_fit_step = self._build_actor_fit_step()

    def _initial_filter_indices(self):
        if self.ensemble_size == 1:
            return jnp.asarray([0], dtype=jnp.int32)
        if self.ensemble_size == 2:
            return jnp.asarray([1, 0], dtype=jnp.int32)
        return (jnp.arange(self.ensemble_size, dtype=jnp.int32) + 1) % self.ensemble_size

    def _apply_actor(self, actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
        if self.actor_dropout is not None and training:
            return self.actor_def.apply(
                {"params": actor_params},
                observations,
                training=training,
                rngs={"dropout": rng},
            )
        return self.actor_def.apply({"params": actor_params}, observations, training=training)

    def _build_common_fns(self):
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        actor_apply_fn = self.actor_def.apply
        ensemble_size = self.ensemble_size
        use_dropout = self.actor_dropout is not None
        iql_deterministic = self.iql_deterministic
        actor_update_method = self.actor_update_method
        policy_weight_exponent = self.policy_weight_exponent
        policy_weight_clip = self.policy_weight_clip
        v_filter_clip = self.v_filter_clip
        alpha = self.alpha
        bc_coef = self.bc_coef

        def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
            if use_dropout and training:
                return actor_apply_fn(
                    {"params": actor_params},
                    observations,
                    training=training,
                    rngs={"dropout": rng},
                )
            return actor_apply_fn({"params": actor_params}, observations, training=training)

        def stack_q(params, observations, actions):
            def apply_one(single_params):
                return q_apply({"params": single_params}, observations, actions)

            return jax.vmap(apply_one)(params)

        def stack_v(params, observations):
            def apply_one(single_params):
                return v_apply({"params": single_params}, observations)

            return jax.vmap(apply_one)(params)

        def policy_action_and_bc(policy_out, actions):
            if iql_deterministic:
                action_for_q = policy_out
                deterministic_bc = jnp.mean((action_for_q - actions) ** 2, axis=-1)
                iql_bc = jnp.sum((action_for_q - actions) ** 2, axis=-1)
                policy_mean = action_for_q
                log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
            else:
                mean, log_std = policy_out
                std = jnp.exp(log_std)
                log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
                action_for_q = mean
                deterministic_bc = jnp.mean((mean - actions) ** 2, axis=-1)
                iql_bc = -jnp.sum(log_prob, axis=-1)
                policy_mean = mean
                log_std_mean = jnp.mean(log_std)
            return action_for_q, deterministic_bc, iql_bc, policy_mean, log_std_mean

        def compute_actor_loss(actor_params, observations, actions, q_params, q_delayed_params, v_delayed_params, rng):
            eps = jnp.asarray(1e-6, dtype=jnp.float32)
            policy_out = apply_actor(actor_params, observations, training=True, rng=rng)
            pi_action, deterministic_bc, iql_bc, policy_mean, log_std_mean = policy_action_and_bc(policy_out, actions)
            bc_per_sample = jnp.where(
                jnp.asarray(actor_update_method in IQL_ACTOR_METHODS),
                iql_bc,
                deterministic_bc,
            )

            data_q_all = stack_q(q_delayed_params, observations, actions)
            data_v_all = stack_v(v_delayed_params, observations)
            data_q = jnp.mean(data_q_all, axis=0)
            data_v = jnp.mean(data_v_all, axis=0)
            data_adv = data_q - data_v

            if actor_update_method in IQL_ACTOR_METHODS:
                logits = jnp.clip(
                    policy_weight_exponent * jax.lax.stop_gradient(data_adv),
                    a_min=-60.0,
                    a_max=jnp.log(policy_weight_clip),
                )
                policy_weight = jnp.exp(logits)
                weighted_bc_loss = jnp.mean(jax.lax.stop_gradient(policy_weight) * bc_per_sample)
                weight_mean = jnp.mean(policy_weight)
                weight_max = jnp.max(policy_weight)
            elif actor_update_method in ("weighted_bc", "td3_weighted_bc"):
                logits = jnp.clip(
                    policy_weight_exponent * jax.lax.stop_gradient(data_adv),
                    -v_filter_clip,
                    v_filter_clip,
                )
                policy_weight = jnp.exp(logits)
                policy_weight = policy_weight / (jnp.mean(policy_weight) + eps)
                policy_weight = jnp.minimum(policy_weight, policy_weight_clip)
                weighted_bc_loss = jnp.mean(jax.lax.stop_gradient(policy_weight) * bc_per_sample)
                weight_mean = jnp.mean(policy_weight)
                weight_max = jnp.max(policy_weight)
            else:
                policy_weight = jnp.ones_like(bc_per_sample)
                weighted_bc_loss = jnp.mean(bc_per_sample)
                weight_mean = jnp.asarray(1.0, dtype=jnp.float32)
                weight_max = jnp.asarray(1.0, dtype=jnp.float32)

            bc_loss = jnp.mean(bc_per_sample)

            if actor_update_method == "weighted_bc" or actor_update_method in IQL_ACTOR_METHODS:
                actor_loss = weighted_bc_loss
                q_for_log = data_q
                lmbda = jnp.asarray(0.0, dtype=jnp.float32)
            else:
                q_pi_all = stack_q(q_params, observations, pi_action)
                q_pi = jnp.mean(q_pi_all, axis=0)
                lmbda = jax.lax.stop_gradient(alpha / jnp.maximum(jnp.mean(jnp.abs(q_pi)), eps))
                bc_regularizer = jnp.where(
                    jnp.asarray(actor_update_method == "td3_weighted_bc"),
                    weighted_bc_loss,
                    bc_loss,
                )
                actor_loss = -lmbda * jnp.mean(q_pi) + bc_coef * bc_regularizer
                q_for_log = q_pi

            actor_weight_sum = jnp.sum(policy_weight) + eps
            actor_ess = (actor_weight_sum ** 2) / (jnp.sum(policy_weight ** 2) + eps)
            actor_ess_ratio = actor_ess / jnp.asarray(observations.shape[0], dtype=jnp.float32)
            return actor_loss, (
                q_for_log,
                lmbda,
                bc_loss,
                weighted_bc_loss,
                weight_mean,
                weight_max,
                actor_ess_ratio,
                policy_mean,
                log_std_mean,
                data_adv,
            )

        return apply_actor, stack_q, stack_v, compute_actor_loss

    def _build_train_step(self):
        q_tx = self.q_tx
        v_tx = self.v_tx
        actor_tx = self.actor_tx
        discount = self.discount
        tau = self.tau
        ensemble_size = self.ensemble_size
        delayed_update_freq = self.delayed_update_freq
        v_filter_type = self.v_filter_type
        v_filter_exponent = self.v_filter_exponent
        v_filter_clip = self.v_filter_clip
        v_filter_floor = self.v_filter_floor
        v_filter_positive_adv_weight_one = self.v_filter_positive_adv_weight_one
        apply_actor, stack_q, stack_v, compute_actor_loss = self._build_common_fns()

        def maybe_sync(new_params: Any, old_delay_params: Any, should_sync: jnp.ndarray):
            return jax.lax.cond(
                should_sync,
                lambda _: new_params,
                lambda _: old_delay_params,
                operand=jnp.asarray(0, dtype=jnp.int32),
            )

        def propose_filter_indices(filter_key: jnp.ndarray):
            if ensemble_size == 1:
                return filter_key, jnp.asarray([0], dtype=jnp.int32)
            if ensemble_size == 2:
                return filter_key, jnp.asarray([1, 0], dtype=jnp.int32)
            new_key, shift_key = jax.random.split(filter_key)
            shift = jax.random.randint(shift_key, shape=(), minval=1, maxval=ensemble_size)
            indices = (jnp.arange(ensemble_size, dtype=jnp.int32) + shift) % ensemble_size
            return new_key, indices

        @jax.jit
        def train_step(state: EDAVEState, actor_state: ActorState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)
            eps = jnp.asarray(1e-6, dtype=jnp.float32)

            q_delay_all = stack_q(state.q_delay_params, observations, actions)
            v_delay_all = stack_v(state.v_delay_params, observations)
            adv_delay_all = q_delay_all - v_delay_all

            filter_indices = state.filter_indices
            adv_filter_all = q_delay_all[filter_indices, :] - v_delay_all[filter_indices, :]
            # Inverse-temperature parameterization, consistent with actor IQL/AWBC:
            #   z_V = eta_V * A_filter
            # where eta_V = v_filter_exponent.
            v_weight_logits = jnp.clip(
                v_filter_exponent * jax.lax.stop_gradient(adv_filter_all),
                -v_filter_clip,
                v_filter_clip,
            )
            if v_filter_type == "sigmoid":
                raw_v_weights = 2.0 * jax.nn.sigmoid(v_weight_logits)
            else:
                raw_v_weights = jnp.exp(v_weight_logits)
            v_weights = v_filter_floor + (1.0 - v_filter_floor) * raw_v_weights
            if v_filter_positive_adv_weight_one:
                # Optional one-sided filter: keep all nonnegative-advantage samples at neutral
                # weight 1 and only downweight negative-advantage samples. This avoids
                # amplifying optimistic positive-advantage estimates in V regression.
                v_weights = jnp.where(adv_filter_all >= 0.0, jnp.ones_like(v_weights), v_weights)
            v_weights = jax.lax.stop_gradient(v_weights)
            v_weight_sum = jnp.sum(v_weights, axis=1) + eps
            v_ess = (jnp.sum(v_weights, axis=1) ** 2) / (jnp.sum(v_weights ** 2, axis=1) + eps)
            v_ess_ratio = jnp.mean(v_ess / jnp.asarray(observations.shape[0], dtype=jnp.float32))

            next_v_target_all = stack_v(state.v_target_params, next_observations)
            target_q_all = rewards[None, :] + (1.0 - dones[None, :]) * discount * next_v_target_all
            target_q_all = jax.lax.stop_gradient(target_q_all)

            def q_loss_fn(q_params):
                q_all = stack_q(q_params, observations, actions)
                per_q_loss = jnp.mean((q_all - target_q_all) ** 2, axis=1)
                q_loss = jnp.mean(per_q_loss)
                return q_loss, (q_all, per_q_loss)

            (q_loss, (q_all, per_q_loss)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
            q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
            q_params = optax.apply_updates(state.q_params, q_updates)

            q_target_for_v_all = stack_q(state.q_target_params, observations, actions)
            q_target_for_v_all = jax.lax.stop_gradient(q_target_for_v_all)

            def v_loss_fn(v_params):
                v_all = stack_v(v_params, observations)
                per_v_loss = jnp.sum(v_weights * (v_all - q_target_for_v_all) ** 2, axis=1) / v_weight_sum
                value_loss = jnp.mean(per_v_loss)
                return value_loss, (v_all, per_v_loss)

            (value_loss, (v_all, per_v_loss)), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                return compute_actor_loss(
                    actor_params,
                    observations,
                    actions,
                    q_params,
                    state.q_delay_params,
                    state.v_delay_params,
                    dropout_key,
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
            (
                actor_q,
                actor_lmbda,
                actor_bc_loss,
                actor_weighted_bc_loss,
                policy_weight_mean,
                policy_weight_max,
                actor_weight_ess_ratio,
                policy_mean,
                log_std_mean,
                actor_data_adv,
            ) = actor_aux
            actor_updates, actor_opt_state = actor_tx.update(actor_grads, actor_state.opt_state, actor_state.params)
            actor_params = optax.apply_updates(actor_state.params, actor_updates)
            new_actor_state = ActorState(params=actor_params, opt_state=actor_opt_state, key=actor_key)

            q_target_params = soft_update(q_params, state.q_target_params, tau)
            v_target_params = soft_update(v_params, state.v_target_params, tau)
            should_sync = (total_it % jnp.asarray(delayed_update_freq, dtype=jnp.int32)) == 0
            q_delay_params = maybe_sync(q_target_params, state.q_delay_params, should_sync)
            v_delay_params = maybe_sync(v_target_params, state.v_delay_params, should_sync)

            proposed_filter_key, proposed_filter_indices = propose_filter_indices(state.filter_key)
            filter_indices_new = jax.lax.cond(
                should_sync,
                lambda _: proposed_filter_indices,
                lambda _: state.filter_indices,
                operand=jnp.asarray(0, dtype=jnp.int32),
            )
            filter_key_new = jax.lax.cond(
                should_sync,
                lambda _: proposed_filter_key,
                lambda _: state.filter_key,
                operand=jnp.asarray(0, dtype=jnp.int32),
            )

            new_state = EDAVEState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_delay_params=q_delay_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_target_params=v_target_params,
                v_delay_params=v_delay_params,
                v_opt_state=v_opt_state,
                filter_indices=filter_indices_new,
                filter_key=filter_key_new,
            )

            q_target_gap = q_delay_all - q_target_for_v_all
            log_dict = {
                "q_loss": q_loss,
                "q_loss_mean": jnp.mean(per_q_loss),
                "q_loss_max": jnp.max(per_q_loss),
                "q_mean": jnp.mean(q_all),
                "q_std_across_ensemble": jnp.mean(jnp.std(q_all, axis=0)),
                "q_delay_mean": jnp.mean(q_delay_all),
                "target_q_mean": jnp.mean(target_q_all),
                "q_target_for_v_mean": jnp.mean(q_target_for_v_all),
                "q_delay_target_gap_abs_mean": jnp.mean(jnp.abs(q_target_gap)),
                "value_loss": value_loss,
                "value_loss_mean": jnp.mean(per_v_loss),
                "value_loss_max": jnp.max(per_v_loss),
                "v_mean": jnp.mean(v_all),
                "v_std_across_ensemble": jnp.mean(jnp.std(v_all, axis=0)),
                "v_delay_mean": jnp.mean(v_delay_all),
                "adv_filter_mean": jnp.mean(adv_filter_all),
                "adv_filter_min": jnp.min(adv_filter_all),
                "adv_filter_max": jnp.max(adv_filter_all),
                "actor_loss": actor_loss,
                "actor_q_mean": jnp.mean(actor_q),
                "actor_lambda": actor_lmbda,
                "actor_bc_loss": actor_bc_loss,
                "actor_weighted_bc_loss": actor_weighted_bc_loss,
                "actor_policy_weight_mean": policy_weight_mean,
                "actor_policy_weight_max": policy_weight_max,
                "actor_weight_ess_ratio": actor_weight_ess_ratio,
                "actor_data_adv_mean": jnp.mean(actor_data_adv),
                "actor_data_adv_min": jnp.min(actor_data_adv),
                "actor_data_adv_max": jnp.max(actor_data_adv),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
                "v_weight_mean": jnp.mean(v_weights),
                "v_weight_std": jnp.std(v_weights),
                "v_weight_min": jnp.min(v_weights),
                "v_weight_max": jnp.max(v_weights),
                "v_weight_ess_ratio": v_ess_ratio,
                "v_filter_positive_adv_weight_one": jnp.asarray(v_filter_positive_adv_weight_one, dtype=jnp.float32),
                "filter_indices_mean": jnp.mean(filter_indices.astype(jnp.float32)),
                "filter_self_match_frac": jnp.mean((filter_indices == jnp.arange(ensemble_size)).astype(jnp.float32)),
                "delay_synced": should_sync.astype(jnp.float32),
            }
            return new_state, new_actor_state, log_dict

        return train_step

    def _build_actor_fit_step(self):
        actor_tx = self.actor_tx
        _, _, _, compute_actor_loss = self._build_common_fns()

        @jax.jit
        def actor_fit_step(actor_state: ActorState, state: EDAVEState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                return compute_actor_loss(
                    actor_params,
                    observations,
                    actions,
                    state.q_params,
                    state.q_delay_params,
                    state.v_delay_params,
                    dropout_key,
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
            (
                actor_q,
                actor_lmbda,
                actor_bc_loss,
                actor_weighted_bc_loss,
                policy_weight_mean,
                policy_weight_max,
                actor_weight_ess_ratio,
                policy_mean,
                log_std_mean,
                actor_data_adv,
            ) = actor_aux
            actor_updates, actor_opt_state = actor_tx.update(actor_grads, actor_state.opt_state, actor_state.params)
            actor_params = optax.apply_updates(actor_state.params, actor_updates)
            new_actor_state = ActorState(params=actor_params, opt_state=actor_opt_state, key=actor_key)
            log_dict = {
                "loss": actor_loss,
                "q_mean": jnp.mean(actor_q),
                "lambda": actor_lmbda,
                "bc_loss": actor_bc_loss,
                "weighted_bc_loss": actor_weighted_bc_loss,
                "policy_weight_mean": policy_weight_mean,
                "policy_weight_max": policy_weight_max,
                "policy_weight_ess_ratio": actor_weight_ess_ratio,
                "data_adv_mean": jnp.mean(actor_data_adv),
                "data_adv_min": jnp.min(actor_data_adv),
                "data_adv_max": jnp.max(actor_data_adv),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
            }
            return new_actor_state, log_dict

        return actor_fit_step

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.state, self.actor_state, log_dict = self._train_step(
            self.state,
            self.actor_state,
            batch,
        )
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}


    def make_initial_actor_state(self) -> ActorState:
        return tree_to_device(
            ActorState(
                params=copy.deepcopy(self.initial_actor_params),
                opt_state=copy.deepcopy(self.initial_actor_opt_state),
                key=copy.deepcopy(self.initial_actor_key),
            ),
            self.device,
        )


    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
        policy_out = self._apply_actor(actor_params, state_jnp, training=False)
        action = policy_out if self.iql_deterministic else policy_out[0]
        action = jnp.clip(self.max_action * action, -self.max_action, self.max_action)
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

    def fit_actor(
        self,
        replay_buffer: ReplayBuffer,
        actor_state: ActorState,
        steps: int,
        batch_size: int,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 0,
        eval_seed: int = 0,
        eval_interval: int = 0,
        prefix: str = "actor_refit",
        save_dir: Optional[Union[str, Path]] = None,
        loaded_checkpoint: Optional[Union[str, Path]] = None,
        log_wandb: bool = False,
    ) -> Tuple[ActorState, Dict[str, Any]]:
        eval_fit_log: Dict[str, Any] = {
            f"{prefix}/final_loss": np.nan,
            f"{prefix}/final_q": np.nan,
            f"{prefix}/final_lambda": np.nan,
            f"{prefix}/final_bc_loss": np.nan,
            f"{prefix}/final_weighted_bc_loss": np.nan,
            f"{prefix}/final_policy_weight_mean": np.nan,
            f"{prefix}/final_policy_weight_max": np.nan,
            f"{prefix}/final_score_mean": np.nan,
            f"{prefix}/final_score_std": np.nan,
            f"{prefix}/final_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/final_d4rl_normalized_score_std": np.nan,
            f"{prefix}/best_score_mean": np.nan,
            f"{prefix}/best_score_std": np.nan,
            f"{prefix}/best_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/best_d4rl_normalized_score_std": np.nan,
            f"{prefix}/inner_eval_steps": [],
            f"{prefix}/inner_score_mean": [],
            f"{prefix}/inner_score_std": [],
            f"{prefix}/inner_d4rl_normalized_score_mean": [],
            f"{prefix}/inner_d4rl_normalized_score_std": [],
        }
        if self.actor_update_method not in ACTOR_REFIT_METHODS:
            raise ValueError(
                "Actor refit supports only actor_update_method in: "
                f"{', '.join(ACTOR_REFIT_METHODS)}. "
                f"Got: {self.actor_update_method}"
            )

        if steps <= 0:
            return actor_state, eval_fit_log

        best_normalized_score_mean = -np.inf
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
        loaded_checkpoint_str = str(loaded_checkpoint) if loaded_checkpoint is not None else None

        for fit_step in range(1, steps + 1):
            batch = replay_buffer.sample(batch_size)
            actor_state, step_log = self._actor_fit_step(actor_state, self.state, batch)
            step_log = {key: float(jax.device_get(value)) for key, value in step_log.items()}

            eval_fit_log[f"{prefix}/final_loss"] = step_log["loss"]
            eval_fit_log[f"{prefix}/final_q"] = step_log["q_mean"]
            eval_fit_log[f"{prefix}/final_lambda"] = step_log["lambda"]
            eval_fit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]
            eval_fit_log[f"{prefix}/final_weighted_bc_loss"] = step_log["weighted_bc_loss"]
            eval_fit_log[f"{prefix}/final_policy_weight_mean"] = step_log["policy_weight_mean"]
            eval_fit_log[f"{prefix}/final_policy_weight_max"] = step_log["policy_weight_max"]

            should_eval = (
                eval_env is not None
                and eval_episodes > 0
                and eval_interval > 0
                and (fit_step % eval_interval == 0 or fit_step == steps)
            )
            if should_eval:
                eval_scores = self.eval_actor(
                    eval_env,
                    actor_state.params,
                    n_episodes=eval_episodes,
                    seed=eval_seed,
                )
                normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

                eval_score_mean = float(np.mean(eval_scores))
                eval_score_std = float(np.std(eval_scores))
                normalized_eval_score_mean = float(np.mean(normalized_eval_scores))
                normalized_eval_score_std = float(np.std(normalized_eval_scores))

                eval_fit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                eval_fit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                eval_fit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                eval_fit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
                eval_fit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
                eval_fit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                eval_fit_log[f"{prefix}/final_score_std"] = eval_score_std
                eval_fit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                eval_fit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std

                is_best = normalized_eval_score_mean > best_normalized_score_mean
                if is_best:
                    best_normalized_score_mean = normalized_eval_score_mean
                    eval_fit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                    eval_fit_log[f"{prefix}/best_score_std"] = eval_score_std
                    eval_fit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                    eval_fit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std

                if save_dir_path is not None:
                    latest_actor_path = save_dir_path / "latest_actor.pkl"
                    fit_eval_logs_path = save_dir_path / "fit_eval_logs.npz"
                    save_pickle(latest_actor_path, serialization.to_state_dict(actor_state.params))
                    log_record: Dict[str, Any] = eval_fit_log.copy()
                    if loaded_checkpoint_str is not None:
                        log_record = {"loaded_checkpoint": loaded_checkpoint_str, **log_record}
                    save_logs_npz([log_record], str(fit_eval_logs_path))
                    if is_best:
                        best_actor_path = save_dir_path / "best_actor.pkl"
                        save_pickle(best_actor_path, serialization.to_state_dict(actor_state.params))
                    if log_wandb and wandb.run is not None:
                        wandb.save(str(latest_actor_path), policy="now")
                        wandb.save(str(fit_eval_logs_path), policy="now")
                        if is_best:
                            wandb.save(str(best_actor_path), policy="now")

                print(
                    f"[{prefix}:{self.actor_update_method}] step {fit_step}/{steps}: "
                    f"loss={step_log['loss']:.4f}, q={step_log['q_mean']:.4f}, "
                    f"lambda={step_log['lambda']:.4f}, bc_loss={step_log['bc_loss']:.4f}, "
                    f"weighted_bc_loss={step_log['weighted_bc_loss']:.4f}, "
                    f"weight_mean={step_log['policy_weight_mean']:.4f}, "
                    f"weight_max={step_log['policy_weight_max']:.4f}, "
                    f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                    f"D4RL_mean={normalized_eval_score_mean:.3f}, "
                    f"D4RL_std={normalized_eval_score_std:.3f}"
                )

        return actor_state, eval_fit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "edave_state": serialization.to_state_dict(self.state),
            "actor_state": serialization.to_state_dict(self.actor_state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
            "iql_deterministic": self.iql_deterministic,
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_full_name": ALGORITHM_FULL_NAME,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raw_edave_state = copy.deepcopy(state_dict["edave_state"])
        ensemble_param_keys = (
            "q_params",
            "q_target_params",
            "q_delay_params",
            "v_params",
            "v_target_params",
            "v_delay_params",
        )
        uses_old_tuple_ensemble = any(
            _is_serialized_ensemble_sequence(raw_edave_state.get(key))
            for key in ensemble_param_keys
        )

        if uses_old_tuple_ensemble:
            # Older EDAVE checkpoints stored the ensemble as a tuple/list of
            # parameter PyTrees. Convert those serialized per-head trees into the
            # new batched-PyTree layout and reinitialize Q/V optimizer states.
            for key in ensemble_param_keys:
                if _is_serialized_ensemble_sequence(raw_edave_state.get(key)):
                    raw_edave_state[key] = _stack_serialized_ensemble_sequence(raw_edave_state[key])

            q_params = serialization.from_state_dict(self.state.q_params, raw_edave_state["q_params"])
            v_params = serialization.from_state_dict(self.state.v_params, raw_edave_state["v_params"])
            self.state = EDAVEState(
                total_it=serialization.from_state_dict(self.state.total_it, raw_edave_state["total_it"]),
                q_params=q_params,
                q_target_params=serialization.from_state_dict(
                    self.state.q_target_params, raw_edave_state["q_target_params"]
                ),
                q_delay_params=serialization.from_state_dict(
                    self.state.q_delay_params, raw_edave_state["q_delay_params"]
                ),
                q_opt_state=self.q_tx.init(q_params),
                v_params=v_params,
                v_target_params=serialization.from_state_dict(
                    self.state.v_target_params, raw_edave_state["v_target_params"]
                ),
                v_delay_params=serialization.from_state_dict(
                    self.state.v_delay_params, raw_edave_state["v_delay_params"]
                ),
                v_opt_state=self.v_tx.init(v_params),
                filter_indices=serialization.from_state_dict(
                    self.state.filter_indices, raw_edave_state["filter_indices"]
                ),
                filter_key=serialization.from_state_dict(self.state.filter_key, raw_edave_state["filter_key"]),
            )
        else:
            self.state = serialization.from_state_dict(self.state, raw_edave_state)

        self.actor_state = serialization.from_state_dict(self.actor_state, state_dict["actor_state"])
        self.initial_actor_params = serialization.from_state_dict(
            self.initial_actor_params,
            state_dict["initial_actor_params"],
        )
        self.initial_actor_opt_state = serialization.from_state_dict(
            self.initial_actor_opt_state,
            state_dict["initial_actor_opt_state"],
        )
        self.initial_actor_key = serialization.from_state_dict(
            self.initial_actor_key,
            state_dict["initial_actor_key"],
        )
        self.state = tree_to_device(self.state, self.device)
        self.actor_state = tree_to_device(self.actor_state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)


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
    """Return (run_dir, checkpoint_path) for a saved EDAVE-IQL checkpoint.

    Supported load_model formats:

    1. Direct checkpoint file:
       path/to/checkpoint.pkl

    2. Direct run directory:
       path/to/run_dir/
       where path/to/run_dir/checkpoint.pkl exists

    3. Parent directory that contains env/seed subdirectory:
       path/to/base_dir/
       where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists
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
    candidates.append(load_path / "checkpoint.pkl")

    if run_name is not None and seed is not None:
        candidates.append(load_path / run_name / str(seed) / "checkpoint.pkl")

    if run_name is not None:
        run_name_dir = load_path / run_name
        if run_name_dir.exists():
            candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pkl")))

    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))

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
    refit_only = config.mode == "refit"
    if refit_only and config.load_model == "":
        raise ValueError("mode='refit' requires --load_model")
    if not refit_only:
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
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=jax_device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None and not refit_only:
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
    run_mode_name = "Actor refit" if refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = EDAVEIQLJAX(
        max_action=max_action,
        state_dim=state_dim,
        action_dim=action_dim,
        max_steps=max(int(config.max_timesteps), 1),
        qf_lr=config.qf_lr,
        vf_lr=config.vf_lr,
        actor_lr=config.actor_lr,
        discount=config.discount,
        tau=config.tau,
        iql_deterministic=config.iql_deterministic,
        actor_dropout=config.actor_dropout,
        ensemble_size=config.ensemble_size,
        delayed_update_freq=config.delayed_update_freq,
        v_filter_type=config.v_filter_type,
        v_filter_exponent=config.v_filter_exponent,
        v_filter_clip=config.v_filter_clip,
        v_filter_floor=config.v_filter_floor,
        v_filter_positive_adv_weight_one=config.v_filter_positive_adv_weight_one,
        actor_update_method=config.actor_update_method,
        policy_weight_exponent=config.policy_weight_exponent,
        policy_weight_clip=config.policy_weight_clip,
        alpha=config.alpha,
        bc_coef=config.bc_coef,
        grad_clip_norm=config.grad_clip_norm,
        seed=seed,
        device=jax_device,
    )

    loaded_run_dir: Optional[Path] = None
    if config.load_model != "":
        loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        trainer.load_state_dict(checkpoint)

    if config.log_wandb:
        wandb_init(asdict(config))

    if refit_only:
        if loaded_run_dir is None:
            raise ValueError("refit mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
        print("Q/V ensemble is frozen; only pi is optimized.")
        print(
            "Refit schedule uses shared fields: "
            f"max_timesteps={config.max_timesteps}, "
            f"batch_size={config.batch_size}, "
            f"eval_freq={config.eval_freq}"
        )
        print(f"Actor update method: {config.actor_update_method}")
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        fresh_actor_state = trainer.make_initial_actor_state()
        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=fresh_actor_state,
            steps=config.max_timesteps,
            batch_size=config.batch_size,
            eval_env=env,
            eval_episodes=config.n_episodes,
            eval_seed=config.seed,
            eval_interval=config.eval_freq,
            prefix="actor_refit",
            save_dir=actor_refit_dir,
            loaded_checkpoint=loaded_run_dir / "checkpoint.pkl",
            log_wandb=config.log_wandb,
        )

        save_pickle(
            actor_refit_dir / "final_actor.pkl",
            serialization.to_state_dict(refit_actor_state.params),
        )
        save_logs_npz(
            [{"loaded_checkpoint": str(loaded_run_dir / "checkpoint.pkl"), **refit_log}],
            str(actor_refit_dir / "fit_eval_logs.npz"),
        )
        with open(actor_refit_dir / "refit_config.yaml", "w") as f:
            pyrallis.dump(config, f)

        if config.log_wandb and wandb.run is not None:
            wandb.save(str(actor_refit_dir / "final_actor.pkl"), policy="now")
            wandb.save(str(actor_refit_dir / "fit_eval_logs.npz"), policy="now")
            wandb.save(str(actor_refit_dir / "refit_config.yaml"), policy="now")

        print("---------------------------------------")
        print("Actor refit finished")
        print(f"Saved final actor to: {actor_refit_dir / 'final_actor.pkl'}")
        print(f"Saved fit logs to:    {actor_refit_dir / 'fit_eval_logs.npz'}")
        print("---------------------------------------")
        return

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

            eval_score_mean = float(np.mean(eval_scores))
            eval_score_std = float(np.std(eval_scores))
            normalized_eval_score_mean = float(np.mean(normalized_eval_scores))
            normalized_eval_score_std = float(np.std(normalized_eval_scores))

            eval_log: Dict[str, Any] = {
                "timestep": int(t + 1),
                "eval/reward_mean": eval_score_mean,
                "eval/reward_std": eval_score_std,
                "eval/normalized_score_mean": normalized_eval_score_mean,
                "eval/normalized_score_std": normalized_eval_score_std,
            }
            eval_logs.append(eval_log.copy())

            print(
                f"Evaluation at step {t + 1}: "
                f"reward_mean={eval_score_mean:.3f}, reward_std={eval_score_std:.3f}, "
                f"D4RL_mean={normalized_eval_score_mean:.3f}, "
                f"D4RL_std={normalized_eval_score_std:.3f}"
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
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
        save_pickle(checkpoint_path, trainer.state_dict())

        if config.log_wandb and wandb.run is not None:
            wandb.save(checkpoint_path, policy="now")

        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()
