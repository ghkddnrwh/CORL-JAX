# JAX/Flax IQL implementation with CDAF_JAX-style experiment plumbing.
# Algorithmic losses mirror the provided PyTorch IQL code; checkpointing,
# hyperparameter merging, logging, and path handling follow the CDAF_JAX style.
#
# Current plumbing:
#   - Explicit mode switch: mode="train" or mode="refit".
#   - Refit mode uses shared schedule fields:
#       max_timesteps -> actor-only refit steps
#       batch_size    -> actor-only refit batch size
#       eval_freq     -> actor-only refit evaluation interval
#   - No backward-compatibility aliases for old refit_* keys.

import copy
import json
import os
import pickle
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg

try:
    import scipy.linalg as scipy_linalg

    if not hasattr(scipy_linalg, "tril"):
        scipy_linalg.tril = np.tril
    if not hasattr(scipy_linalg, "triu"):
        scipy_linalg.triu = np.triu
except ImportError:
    pass

import optax
import pyrallis
import yaml

try:
    import wandb
except ImportError:
    class _UnavailableWandb:
        run = None

        def init(self, *args, **kwargs):
            raise ImportError(
                "wandb is unavailable in this environment; run with --log_wandb False "
                "or install wandb with its dependencies."
            )

        def save(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _UnavailableWandb()

from flax import linen as nn
from flax import serialization, struct

d4rl = None

try:
    import ogbench
except ImportError:
    ogbench = None


# Automatic training resume utilities shared by all offline-RL algorithms.
_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "algorithms").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.uk_offline.common.checkpointing import (
    DEFAULT_IDENTITY_IGNORED_FIELDS,
    TrainingCheckpointManager,
    best_eval_metric,
    evaluation_is_due,
    find_eval_log,
    log_training_exceptions,
    upsert_eval_log,
)

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "IQL-PD"
ALGORITHM_FULL_NAME = "IQL Propagation Diagnostics"

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


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
    hyperparams_path: Optional[str] = "hyperparams/iql_jax.yml"
    use_hyperparams: bool = True

    # Dataset
    buffer_size: int = 2_000_000

    # Shared by both modes:
    #   mode="train": minibatch size for joint Q/V/pi updates.
    #   mode="refit": minibatch size for actor-only refit updates.
    batch_size: int = 256

    normalize: bool = True
    normalize_reward: bool = False

    # IQL
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = False
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = None
    hidden_dim: int = 256
    n_hidden: int = 2
    num_critics: int = 2

    # Stage-1 propagation diagnostics. These do not change Q/V/pi updates.
    enable_propagation_diagnostics: bool = True
    diag_every: int = 1000
    diag_probe_batch_size: int = 256
    diag_hidden_dim: int = 128
    diag_head_lr: float = 3e-4
    diag_kappa_floor: float = 1e-3
    diag_fd_epsilon: float = 1e-3
    diag_num_basis_directions: int = 4

    # Expensive local full-fit approximation
    # B ~= J_Z (J_X^T J_X / m + lambda I)^-1 J_X^T / m.
    # It is evaluated less frequently than the one-step diagnostics.
    diag_enable_ridge_b: bool = True
    diag_ridge_every: int = 5000
    diag_ridge_lambda: float = 1e-2
    diag_ridge_cg_iters: int = 10
    diag_ridge_cg_tol: float = 1e-4

    # Standalone actor refit output directory.
    # Refit reuses the shared training schedule fields above:
    #   max_timesteps -> actor-only refit steps
    #   batch_size    -> actor-only refit batch size
    #   eval_freq     -> actor-only refit evaluation interval
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-SMOOTH"
    group: str = "IQL-JAX"
    name: str = "IQL-JAX"
    log_wandb: bool = True
    log_every: int = 500
    save_final_model: bool = False

    checkpoint_freq: int = int(25e3)
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
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
    assert config.checkpoint_freq > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.beta >= 0.0
    assert config.iql_tau >= 0.0 and config.iql_tau <= 1.0
    if config.actor_dropout is not None:
        assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
    assert config.hidden_dim > 0
    assert config.n_hidden > 0
    assert config.num_critics > 0
    if config.enable_propagation_diagnostics:
        assert config.diag_every > 0
        assert config.diag_probe_batch_size > 0
        assert config.diag_hidden_dim > 0
        assert config.diag_head_lr > 0.0
        assert config.diag_kappa_floor > 0.0
        assert config.diag_fd_epsilon > 0.0
        assert config.diag_num_basis_directions > 0
        assert config.diag_ridge_every > 0
        assert config.diag_ridge_lambda > 0.0
        assert config.diag_ridge_cg_iters > 0
        assert config.diag_ridge_cg_tol > 0.0
    assert config.actor_refit_dir_name != ""
    if config.mode == "refit":
        assert config.load_model != "", "mode='refit' requires --load_model"


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

    Hyperparameter YAML keys must exactly match TrainConfig field names.
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
    aliases = {"n_timesteps": "max_timesteps"}
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []
    applied_fields = set()

    for raw_key, raw_value in env_hyperparams.items():
        key = aliases.get(raw_key, raw_key)
        if key not in config_fields:
            skipped_unknown.append(raw_key)
            continue
        if key in applied_fields:
            continue
        if key in cli_overrides or raw_key in cli_overrides:
            skipped_cli.append(raw_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(f"{raw_key}->{key}" if raw_key != key else key)
        applied_fields.add(key)

    refresh_algorithm_names(config)
    validate_config(config)

    if applied:
        print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
    if skipped_cli:
        print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
    if skipped_unknown:
        print(f"Ignored unknown hyperparameter keys for IQL: {', '.join(skipped_unknown)}")
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


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
    return (states - mean) / std


class TransformEnv:
    def __init__(
        self,
        env: gym.Env,
        state_mean: Union[np.ndarray, float],
        state_std: Union[np.ndarray, float],
        reward_scale: float,
    ):
        self.env = env
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def _scale_reward(self, reward):
        return self.reward_scale * reward

    def reset(self, *args, **kwargs):
        reset_out = self.env.reset(*args, **kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            state, info = reset_out
            return self._normalize_state(state), info
        return self._normalize_state(reset_out)

    def step(self, action):
        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            state, reward, terminated, truncated, info = step_out
            return self._normalize_state(state), self._scale_reward(reward), terminated, truncated, info
        state, reward, done, info = step_out
        return self._normalize_state(state), self._scale_reward(reward), done, info

    def seed(self, seed: int):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return self.env.reset(seed=seed)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    return TransformEnv(env, state_mean=state_mean, state_std=state_std, reward_scale=reward_scale)


def is_ogbench_env(env_name: str) -> bool:
    return "singletask" in env_name or "oraclerep" in env_name


def load_env_and_dataset(env_name: str) -> Tuple[gym.Env, Dict[str, np.ndarray], str]:
    if is_ogbench_env(env_name):
        if ogbench is None:
            raise ImportError(
                "OGBench environment requested, but the `ogbench` package is not installed."
            )
        env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        return env, train_dataset, "ogbench"

    global d4rl
    if d4rl is None:
        try:
            import d4rl as d4rl_module
        except Exception as exc:
            raise ImportError(
                "D4RL environment requested, but the `d4rl` package could not be imported."
            ) from exc
        d4rl = d4rl_module
    env = gym.make(env_name)
    return env, d4rl.qlearning_dataset(env), "d4rl"


def reset_env(env: gym.Env, seed: Optional[int] = None):
    if seed is not None:
        try:
            reset_out = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            reset_out = env.reset()
    else:
        reset_out = env.reset()

    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def step_env(env: gym.Env, action: np.ndarray):
    step_out = env.step(action)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        next_state, reward, terminated, truncated, info = step_out
        return next_state, reward, bool(terminated or truncated), info
    next_state, reward, done, info = step_out
    return next_state, reward, bool(done), info


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
        done_values = 1.0 - data["masks"] if "masks" in data else data["terminals"]
        self._dones[:n_transitions] = done_values[..., None].astype(np.float32)
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
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def wandb_init(config: dict) -> None:
    run = wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    run.log_code(".")


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


def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
    if not hasattr(env, "get_normalized_score"):
        return np.full_like(np.asarray(eval_scores, dtype=np.float32), np.nan, dtype=np.float32)
    return np.asarray(
        [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
        dtype=np.float32,
    )


def mean_std_or_nan(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan
    return float(np.mean(finite_values)), float(np.std(finite_values))


def extract_success(info: Any) -> float:
    if not isinstance(info, dict) or "success" not in info:
        return np.nan
    success = np.asarray(info["success"])
    if success.size == 0:
        return np.nan
    return float(success.reshape(-1)[0])


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


class TwinQ(nn.Module):
    """Configurable ensemble of independent Q-functions.

    The class name is kept as ``TwinQ`` for minimal disruption to the existing
    code, but ``num_critics`` can be any positive integer. With the default
    value of 2, the parameter names remain ``q1`` and ``q2`` as before.
    """

    hidden_dim: int = 256
    n_hidden: int = 2
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        qs = [
            QFunction(
                hidden_dim=self.hidden_dim,
                n_hidden=self.n_hidden,
                name=f"q{i + 1}",
            )(state, action)
            for i in range(self.num_critics)
        ]
        # Shape: (num_critics, batch_size).
        return jnp.stack(qs, axis=0)


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


class ExpectileDiagnosticHead(nn.Module):
    """Auxiliary state-conditional estimators used only for diagnostics.

    Outputs:
      raw[..., 0] -> G_hat(s) ~= E[psi_tau(Q_target(s,a)-V(s)) | s]
      raw[..., 1] -> unconstrained kappa logit
      raw[..., 2] -> C_Q_hat(s) ~= E[w_tau(A) * eps_lag(s,a) | s]

    The second output is mapped to the valid expectile-weight interval before use.
    This network is deliberately detached from the IQL losses: it diagnoses the
    current algorithm but does not change Q/V/pi in Stage 1.
    """

    hidden_dim: int = 128
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        return nn.Dense(3)(x)


@struct.dataclass
class IQLState:
    total_it: jnp.ndarray
    q_params: Any
    q_target_params: Any
    q_opt_state: Any
    v_params: Any
    v_opt_state: Any
    actor_params: Any
    actor_opt_state: Any
    actor_key: jnp.ndarray
    diag_params: Any
    diag_opt_state: Any


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


class IQLJAX:
    """Implicit Q-Learning in JAX/Flax.

    The value, Q, and actor update equations mirror the provided PyTorch IQL code.
    Experiment-management code follows the CDAF_JAX file style.
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
        beta: float = 3.0,
        iql_tau: float = 0.7,
        iql_deterministic: bool = False,
        actor_dropout: Optional[float] = None,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_critics: int = 2,
        enable_propagation_diagnostics: bool = True,
        diag_hidden_dim: int = 128,
        diag_head_lr: float = 3e-4,
        diag_kappa_floor: float = 1e-3,
        diag_fd_epsilon: float = 1e-3,
        diag_num_basis_directions: int = 4,
        diag_ridge_lambda: float = 1e-2,
        diag_ridge_cg_iters: int = 10,
        diag_ridge_cg_tol: float = 1e-4,
        seed: int = 0,
        device: Any = None,
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max(int(max_steps), 1)
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.iql_tau = iql_tau
        self.iql_deterministic = iql_deterministic
        self.actor_dropout = actor_dropout
        self.hidden_dim = int(hidden_dim)
        self.n_hidden = int(n_hidden)
        self.num_critics = int(num_critics)
        self.qf_lr = float(qf_lr)
        self.vf_lr = float(vf_lr)
        self.enable_propagation_diagnostics = bool(enable_propagation_diagnostics)
        self.diag_hidden_dim = int(diag_hidden_dim)
        self.diag_head_lr = float(diag_head_lr)
        self.diag_kappa_floor = float(diag_kappa_floor)
        self.diag_fd_epsilon = float(diag_fd_epsilon)
        self.diag_num_basis_directions = int(diag_num_basis_directions)
        self.diag_ridge_lambda = float(diag_ridge_lambda)
        self.diag_ridge_cg_iters = int(diag_ridge_cg_iters)
        self.diag_ridge_cg_tol = float(diag_ridge_cg_tol)
        self.diag_seed = int(seed) + 7919
        self.device = device if device is not None else jax.devices()[0]

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be > 0")
        if self.num_critics <= 0:
            raise ValueError("num_critics must be > 0")
        if self.diag_hidden_dim <= 0:
            raise ValueError("diag_hidden_dim must be > 0")
        if self.diag_kappa_floor <= 0.0:
            raise ValueError("diag_kappa_floor must be > 0")

        if iql_deterministic:
            self.actor_def = DeterministicPolicy(
                action_dim=action_dim,
                hidden_dim=self.hidden_dim,
                n_hidden=self.n_hidden,
                dropout=actor_dropout,
            )
        else:
            self.actor_def = GaussianPolicy(
                action_dim=action_dim,
                hidden_dim=self.hidden_dim,
                n_hidden=self.n_hidden,
                dropout=actor_dropout,
            )
        self.q_def = TwinQ(
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            num_critics=self.num_critics,
        )
        self.v_def = ValueFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        self.diag_def = ExpectileDiagnosticHead(hidden_dim=self.diag_hidden_dim, n_hidden=2)

        self.q_tx = optax.adam(qf_lr)
        self.v_tx = optax.adam(vf_lr)
        self.diag_tx = optax.adam(self.diag_head_lr)
        actor_lr_schedule = optax.cosine_decay_schedule(
            init_value=actor_lr,
            decay_steps=self.max_steps,
            alpha=0.0,
        )
        self.actor_tx = optax.adam(actor_lr_schedule)

        key = jax.random.PRNGKey(seed)
        key_actor, key_q, key_v, key_diag, actor_key = jax.random.split(key, 5)
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
        q_params = self.q_def.init(key_q, dummy_state, dummy_action)["params"]
        v_params = self.v_def.init(key_v, dummy_state)["params"]
        diag_params = self.diag_def.init(key_diag, dummy_state)["params"]

        self.initial_actor_params = copy.deepcopy(actor_params)
        self.initial_actor_opt_state = self.actor_tx.init(actor_params)
        self.initial_actor_key = actor_key

        self.state = IQLState(
            total_it=jnp.asarray(0, dtype=jnp.int32),
            q_params=q_params,
            q_target_params=copy.deepcopy(q_params),
            q_opt_state=self.q_tx.init(q_params),
            v_params=v_params,
            v_opt_state=self.v_tx.init(v_params),
            actor_params=actor_params,
            actor_opt_state=copy.deepcopy(self.initial_actor_opt_state),
            actor_key=actor_key,
            diag_params=diag_params,
            diag_opt_state=self.diag_tx.init(diag_params),
        )

        self.state = tree_to_device(self.state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
        self._train_step = self._build_train_step()
        self._propagation_diagnostic_step = self._build_propagation_diagnostic_step()
        self._ridge_diagnostic_step = self._build_ridge_diagnostic_step()
        self._actor_refit_step = self._build_actor_refit_step()

    def _apply_actor(self, actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
        if self.actor_dropout is not None and training:
            return self.actor_def.apply(
                {"params": actor_params},
                observations,
                training=training,
                rngs={"dropout": rng},
            )
        return self.actor_def.apply({"params": actor_params}, observations, training=training)

    def _build_train_step(self):
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        diag_apply = self.diag_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        diag_tx = self.diag_tx
        actor_tx = self.actor_tx
        discount = self.discount
        tau = self.tau
        beta = self.beta
        iql_tau = self.iql_tau
        iql_deterministic = self.iql_deterministic
        num_critics = self.num_critics
        enable_diag = self.enable_propagation_diagnostics
        diag_kappa_floor = self.diag_kappa_floor
        use_dropout = self.actor_dropout is not None
        actor_apply_fn = self.actor_def.apply

        kappa_low = min(float(iql_tau), float(1.0 - iql_tau))
        kappa_high = max(float(iql_tau), float(1.0 - iql_tau))

        def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
            if use_dropout and training:
                return actor_apply_fn(
                    {"params": actor_params},
                    observations,
                    training=training,
                    rngs={"dropout": rng},
                )
            return actor_apply_fn({"params": actor_params}, observations, training=training)

        def unpack_diag(raw: jnp.ndarray):
            g_hat = raw[..., 0]
            if abs(kappa_high - kappa_low) < 1e-12:
                kappa_hat = jnp.full_like(g_hat, kappa_low)
            else:
                kappa_hat = kappa_low + (kappa_high - kappa_low) * jax.nn.sigmoid(raw[..., 1])
            kappa_hat = jnp.maximum(kappa_hat, diag_kappa_floor)
            q_num_hat = raw[..., 2]
            return g_hat, kappa_hat, q_num_hat

        @jax.jit
        def train_step(state: IQLState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            # Values used by multiple losses are computed from the old state, matching
            # the sequencing of the provided PyTorch implementation.
            next_v = v_apply({"params": state.v_params}, next_observations)
            target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
            target_qs = q_apply({"params": state.q_target_params}, observations, actions)
            target_q_for_v = jnp.min(target_qs, axis=0)
            old_v = v_apply({"params": state.v_params}, observations)
            adv = target_q_for_v - old_v
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

            # Auxiliary diagnostic head. It estimates conditional quantities that are
            # unavailable from a single continuous-state sample:
            #   G(s)      = E[w_tau(A) A | s]
            #   kappa(s)  = E[w_tau(A) | s]
            #   C_Q(s)    = E[w_tau(A) eps_lag | s]
            # where eps_lag = Q_online - Q_target is only a practical proxy for the
            # theoretical critic perturbation epsilon_k.
            diag_params = state.diag_params
            diag_opt_state = state.diag_opt_state
            diag_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            diag_g_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            diag_kappa_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            diag_qnum_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            raw_score_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
            raw_score_abs_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
            raw_kappa_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
            q_lag_rms = jnp.asarray(jnp.nan, dtype=jnp.float32)

            if enable_diag:
                online_qs_diag = q_apply({"params": state.q_params}, observations, actions)
                online_q_min_diag = jnp.min(online_qs_diag, axis=0)
                eps_lag = jax.lax.stop_gradient(online_q_min_diag - target_q_for_v)
                diag_adv = jax.lax.stop_gradient(target_q_for_v - old_v)
                diag_weight = jnp.abs(iql_tau - (diag_adv < 0.0).astype(jnp.float32))
                diag_score = diag_weight * diag_adv
                diag_q_num = diag_weight * eps_lag

                def diag_loss_fn(params):
                    raw = diag_apply({"params": params}, observations)
                    g_hat, kappa_hat, q_num_hat = unpack_diag(raw)
                    g_loss = jnp.mean((g_hat - diag_score) ** 2)
                    kappa_loss = jnp.mean((kappa_hat - diag_weight) ** 2)
                    qnum_loss = jnp.mean((q_num_hat - diag_q_num) ** 2)
                    loss = g_loss + kappa_loss + qnum_loss
                    return loss, (g_loss, kappa_loss, qnum_loss)

                (diag_loss, (diag_g_loss, diag_kappa_loss, diag_qnum_loss)), diag_grads = jax.value_and_grad(
                    diag_loss_fn, has_aux=True
                )(state.diag_params)
                diag_updates, diag_opt_state = diag_tx.update(
                    diag_grads,
                    state.diag_opt_state,
                    state.diag_params,
                )
                diag_params = optax.apply_updates(state.diag_params, diag_updates)
                raw_score_mean = jnp.mean(diag_score)
                raw_score_abs_mean = jnp.mean(jnp.abs(diag_score))
                raw_kappa_mean = jnp.mean(diag_weight)
                q_lag_rms = jnp.sqrt(jnp.mean(eps_lag ** 2) + 1e-12)

            def v_loss_fn(v_params):
                v = v_apply({"params": v_params}, observations)
                value_adv = jax.lax.stop_gradient(target_q_for_v) - v
                value_weight = jnp.abs(iql_tau - (value_adv < 0.0).astype(jnp.float32))
                value_loss = jnp.mean(value_weight * value_adv ** 2)
                return value_loss, v

            (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            def q_loss_fn(q_params):
                qs = q_apply({"params": q_params}, observations, actions)
                target = jax.lax.stop_gradient(target_q_for_backup)
                # Average the squared TD error over both critic and batch axes.
                # For num_critics=2, this has the same scaling as the original
                # 0.5 * (MSE(q1) + MSE(q2)) formulation.
                q_loss = jnp.mean((qs - target[None, :]) ** 2)
                return q_loss, qs

            (q_loss, qs), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
            q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
            q_params = optax.apply_updates(state.q_params, q_updates)
            q_target_params = soft_update(q_params, state.q_target_params, tau)

            actor_key, dropout_key = jax.random.split(state.actor_key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_losses = jnp.sum((policy_out - actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
                    bc_losses = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_losses)
                return actor_loss, (bc_losses, policy_mean, log_std_mean)

            (actor_loss, (bc_losses, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(state.actor_params)
            actor_updates, actor_opt_state = actor_tx.update(
                actor_grads,
                state.actor_opt_state,
                state.actor_params,
            )
            actor_params = optax.apply_updates(state.actor_params, actor_updates)

            new_state = IQLState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_opt_state=v_opt_state,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                actor_key=actor_key,
                diag_params=diag_params,
                diag_opt_state=diag_opt_state,
            )

            log_dict = {
                "q_loss": q_loss,
                "q_mean": jnp.mean(qs),
                "q_min_mean": jnp.mean(jnp.min(qs, axis=0)),
                "q_max_mean": jnp.mean(jnp.max(qs, axis=0)),
                "target_q_mean": jnp.mean(target_q_for_backup),
                "value_loss": value_loss,
                "v_mean": jnp.mean(v),
                "adv_mean": jnp.mean(adv),
                "adv_min": jnp.min(adv),
                "adv_max": jnp.max(adv),
                "exp_adv_mean": jnp.mean(exp_adv),
                "actor_loss": actor_loss,
                "bc_loss_mean": jnp.mean(bc_losses),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
            }
            if enable_diag:
                log_dict.update({
                    "diag_head/loss": diag_loss,
                    "diag_head/g_loss": diag_g_loss,
                    "diag_head/kappa_loss": diag_kappa_loss,
                    "diag_head/qnum_loss": diag_qnum_loss,
                    "diag_raw/score_mean": raw_score_mean,
                    "diag_raw/score_abs_mean": raw_score_abs_mean,
                    "diag_raw/kappa_mean": raw_kappa_mean,
                    "diag_raw/q_target_lag_rms": q_lag_rms,
                })
            for i in range(num_critics):
                log_dict[f"q{i + 1}_mean"] = jnp.mean(qs[i])
            return new_state, log_dict

        return train_step

    def _build_propagation_diagnostic_step(self):
        """Build the expensive Stage-1 diagnostic pass.

        The fitted-critic response is always the same abstract object,
            B_k = D_y F_k(y),
        but we compare three practical approximations on identical directions:
          1) paired finite difference through the *actual Adam* Q update,
          2) exact JVP through the same one-step Adam update,
          3) one-step SGD/NTK linearization eta * J_Z J_X^T (with exact loss scaling).

        h_V is estimated from the state-conditional expectile FOC defect:
            h_V ~= -gamma * G(s') / kappa(s').
        h_Q uses an explicitly labelled proxy for epsilon_k based on online-target lag:
            h_Q^lag ~= gamma * E[w * (Q_online-Q_target)|s'] / kappa(s').
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        diag_apply = self.diag_def.apply
        q_tx = self.q_tx
        discount = self.discount
        iql_tau = self.iql_tau
        num_critics = self.num_critics
        qf_lr = self.qf_lr
        fd_epsilon = self.diag_fd_epsilon
        num_basis = self.diag_num_basis_directions
        diag_seed = self.diag_seed
        kappa_floor = self.diag_kappa_floor

        kappa_low = min(float(iql_tau), float(1.0 - iql_tau))
        kappa_high = max(float(iql_tau), float(1.0 - iql_tau))

        def unpack_diag(raw: jnp.ndarray):
            g_hat = raw[..., 0]
            if abs(kappa_high - kappa_low) < 1e-12:
                kappa_hat = jnp.full_like(g_hat, kappa_low)
            else:
                kappa_hat = kappa_low + (kappa_high - kappa_low) * jax.nn.sigmoid(raw[..., 1])
            kappa_hat = jnp.maximum(kappa_hat, kappa_floor)
            q_num_hat = raw[..., 2]
            return g_hat, kappa_hat, q_num_hat

        def rms(x):
            x = jnp.asarray(x)
            return jnp.sqrt(jnp.mean(x ** 2) + 1e-12)

        def tree_l2(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves) + 1e-12)

        def cosine(a, b):
            a = jnp.ravel(a)
            b = jnp.ravel(b)
            return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)

        def corr(a, b):
            a = jnp.ravel(a)
            b = jnp.ravel(b)
            ac = a - jnp.mean(a)
            bc = b - jnp.mean(b)
            return jnp.sum(ac * bc) / (jnp.linalg.norm(ac) * jnp.linalg.norm(bc) + 1e-12)

        def normalized_direction(h):
            source_rms = rms(h)
            h_unit = h / jnp.maximum(source_rms, 1e-8)
            return h_unit, source_rms

        @jax.jit
        def diagnostic_step(old_state: IQLState, new_state: IQLState, batch: TensorBatch, probe_batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            probe_observations = probe_batch["observations"]
            probe_actions = probe_batch["actions"]

            # Exact Bellman target used by the Q update in the original training step.
            next_v_old = v_apply({"params": old_state.v_params}, next_observations)
            target = rewards + (1.0 - dones) * discount * next_v_old

            def actual_adam_output_for(target_arg, out_obs, out_actions):
                def q_loss_only(q_params):
                    qs_local = q_apply({"params": q_params}, observations, actions)
                    return jnp.mean((qs_local - target_arg[None, :]) ** 2)

                q_grads = jax.grad(q_loss_only)(old_state.q_params)
                q_updates, _ = q_tx.update(q_grads, old_state.q_opt_state, old_state.q_params)
                q_params_plus = optax.apply_updates(old_state.q_params, q_updates)
                return q_apply({"params": q_params_plus}, out_obs, out_actions)

            def actual_probe_output(target_arg):
                return actual_adam_output_for(target_arg, probe_observations, probe_actions)

            def actual_train_output(target_arg):
                return actual_adam_output_for(target_arg, observations, actions)

            # Old-parameter NTK / one-step-SGD linearization with the exact q_loss scaling.
            def ntk_output(h_unit, out_obs, out_actions):
                def train_q_fn(params):
                    return q_apply({"params": params}, observations, actions)

                def out_q_fn(params):
                    return q_apply({"params": params}, out_obs, out_actions)

                q_train, pullback = jax.vjp(train_q_fn, old_state.q_params)
                cotangent = jnp.broadcast_to(h_unit[None, :], q_train.shape)
                param_pullback = pullback(cotangent)[0]
                scale = (2.0 * qf_lr) / float(q_train.size)
                param_direction = jax.tree_util.tree_map(lambda x: scale * x, param_pullback)
                _, out_tangent = jax.jvp(out_q_fn, (old_state.q_params,), (param_direction,))
                return out_tangent

            # Auxiliary head validation on an independent probe batch.
            probe_q_target_all = q_apply(
                {"params": old_state.q_target_params}, probe_observations, probe_actions
            )
            probe_q_target = jnp.min(probe_q_target_all, axis=0)
            probe_q_online_all = q_apply(
                {"params": old_state.q_params}, probe_observations, probe_actions
            )
            probe_q_online = jnp.min(probe_q_online_all, axis=0)
            probe_v_pre = v_apply({"params": old_state.v_params}, probe_observations)
            probe_v_post = v_apply({"params": new_state.v_params}, probe_observations)

            def score_terms(v_values):
                u = jax.lax.stop_gradient(probe_q_target) - v_values
                w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                score = w * u
                loss = jnp.mean(w * u ** 2)
                return score, w, loss

            score_pre, weight_pre, probe_v_loss_pre = score_terms(probe_v_pre)
            score_post, weight_post, probe_v_loss_post = score_terms(probe_v_post)
            eps_lag_probe = probe_q_online - probe_q_target
            q_num_target_probe = weight_pre * eps_lag_probe

            raw_probe = diag_apply({"params": new_state.diag_params}, probe_observations)
            g_hat_probe, kappa_hat_probe, q_num_hat_probe = unpack_diag(raw_probe)

            g_global_pre = jnp.mean(score_pre)
            kappa_global_pre = jnp.maximum(jnp.mean(weight_pre), kappa_floor)
            d_global_pre = -g_global_pre / kappa_global_pre
            g_global_post = jnp.mean(score_post)
            kappa_global_post = jnp.maximum(jnp.mean(weight_post), kappa_floor)
            d_global_post = -g_global_post / kappa_global_post

            def probe_v_loss_fn(v_params):
                probe_v = v_apply({"params": v_params}, probe_observations)
                u = jax.lax.stop_gradient(probe_q_target) - probe_v
                w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                return jnp.mean(w * u ** 2)

            v_grad_pre = jax.grad(probe_v_loss_fn)(old_state.v_params)
            v_grad_post = jax.grad(probe_v_loss_fn)(new_state.v_params)

            # State-conditional h_V and h_Q^lag directions on the actual Q-fitting batch.
            raw_next = diag_apply({"params": new_state.diag_params}, next_observations)
            g_hat_next, kappa_hat_next, q_num_hat_next = unpack_diag(raw_next)
            residual_value_error_hat = -g_hat_next / kappa_hat_next  # V - V_exact, first order.
            h_v = (1.0 - dones) * discount * residual_value_error_hat
            critic_value_error_lag_hat = q_num_hat_next / kappa_hat_next
            h_q_lag = (1.0 - dones) * discount * critic_value_error_lag_hat

            # Deterministic random direction so the optional ridge diagnostic can reproduce it.
            base_key = jax.random.fold_in(jax.random.PRNGKey(diag_seed), old_state.total_it + 1)
            rand_key, basis_key = jax.random.split(base_key)
            h_random = jax.random.rademacher(rand_key, shape=target.shape, dtype=jnp.float32)

            base_probe = actual_probe_output(target)
            base_train = actual_train_output(target)

            def direction_outputs(h):
                h_unit, source_rms = normalized_direction(h)
                fd_probe = (actual_probe_output(target + fd_epsilon * h_unit) - base_probe) / fd_epsilon
                _, adam_jvp_probe = jax.jvp(actual_probe_output, (target,), (h_unit,))
                ntk_probe = ntk_output(h_unit, probe_observations, probe_actions)
                return h_unit, source_rms, fd_probe, adam_jvp_probe, ntk_probe

            _, random_source_rms, random_fd, random_adam_jvp, random_ntk = direction_outputs(h_random)
            _, h_v_source_rms, h_v_fd, h_v_adam_jvp, h_v_ntk = direction_outputs(h_v)
            _, h_q_source_rms, h_q_fd, h_q_adam_jvp, h_q_ntk = direction_outputs(h_q_lag)

            # Cross-sample spillover: perturb a few individual targets e_i and measure
            # how much the actual Adam update changes other training-batch Q outputs.
            batch_size = target.shape[0]
            basis_indices = jax.random.randint(
                basis_key,
                shape=(num_basis,),
                minval=0,
                maxval=batch_size,
            )
            basis = jax.nn.one_hot(basis_indices, batch_size, dtype=target.dtype)

            def one_basis_adam(h_basis):
                _, out = jax.jvp(actual_train_output, (target,), (h_basis,))
                return out

            def one_basis_ntk(h_basis):
                return ntk_output(h_basis, observations, actions)

            basis_adam = jax.vmap(one_basis_adam)(basis)  # [K, C, B]
            basis_ntk = jax.vmap(one_basis_ntk)(basis)

            def spillover_stats(outputs):
                mean_q_response = jnp.mean(outputs, axis=1)  # [K, B]
                self_response = jnp.take_along_axis(
                    mean_q_response, basis_indices[:, None], axis=1
                )[:, 0]
                total_sq = jnp.sum(mean_q_response ** 2, axis=1)
                cross_l2 = jnp.sqrt(jnp.maximum(total_sq - self_response ** 2, 0.0) + 1e-12)
                ratio = cross_l2 / (jnp.abs(self_response) + 1e-8)
                return (
                    jnp.mean(jnp.abs(self_response)),
                    jnp.mean(cross_l2),
                    jnp.mean(ratio),
                )

            adam_self, adam_spill, adam_spill_ratio = spillover_stats(basis_adam)
            ntk_self, ntk_spill, ntk_spill_ratio = spillover_stats(basis_ntk)

            ensemble_std_probe = jnp.mean(jnp.std(probe_q_online_all, axis=0))
            td_error_old = jnp.mean(
                q_apply({"params": old_state.q_params}, observations, actions), axis=0
            ) - target

            log_dict = {
                # Residual-source diagnostics. raw score magnitude is deliberately named
                # "raw_score" because |psi_tau| is NOT itself the expectile residual.
                "diag_residual/raw_score_mean_probe": jnp.mean(score_pre),
                "diag_residual/raw_score_abs_mean_probe": jnp.mean(jnp.abs(score_pre)),
                "diag_residual/kappa_batch_mean_probe": jnp.mean(weight_pre),
                "diag_residual/g_hat_rms_probe": rms(g_hat_probe),
                "diag_residual/g_hat_abs_mean_probe": jnp.mean(jnp.abs(g_hat_probe)),
                "diag_residual/kappa_hat_mean_probe": jnp.mean(kappa_hat_probe),
                "diag_residual/kappa_hat_min_probe": jnp.min(kappa_hat_probe),
                "diag_residual/value_error_hat_rms_probe": rms(-g_hat_probe / kappa_hat_probe),
                "diag_residual/head_g_mse_probe": jnp.mean((g_hat_probe - score_pre) ** 2),
                "diag_residual/head_kappa_mse_probe": jnp.mean((kappa_hat_probe - weight_pre) ** 2),
                "diag_residual/head_qnum_mse_probe": jnp.mean((q_num_hat_probe - q_num_target_probe) ** 2),
                "diag_residual/head_g_raw_corr_probe": corr(g_hat_probe, score_pre),
                "diag_residual/head_kappa_raw_corr_probe": corr(kappa_hat_probe, weight_pre),
                "diag_residual/head_qnum_raw_corr_probe": corr(q_num_hat_probe, q_num_target_probe),

                # Correctability / under-training diagnostics. These compare the same
                # frozen Q_target before and after the *actual single V update*.
                "diag_residual/probe_v_loss_pre": probe_v_loss_pre,
                "diag_residual/probe_v_loss_post": probe_v_loss_post,
                "diag_residual/probe_v_loss_ratio": probe_v_loss_post / (probe_v_loss_pre + 1e-12),
                "diag_residual/global_d_pre": d_global_pre,
                "diag_residual/global_d_post": d_global_post,
                "diag_residual/global_d_abs_ratio": jnp.abs(d_global_post) / (jnp.abs(d_global_pre) + 1e-8),
                "diag_residual/v_grad_norm_pre": tree_l2(v_grad_pre),
                "diag_residual/v_grad_norm_post": tree_l2(v_grad_post),
                "diag_residual/v_grad_norm_ratio": tree_l2(v_grad_post) / (tree_l2(v_grad_pre) + 1e-12),

                # Observable proxies entering the two theoretical channels.
                "diag_source/h_v_rms": h_v_source_rms,
                "diag_source/h_q_lag_rms": h_q_source_rms,
                "diag_source/q_target_lag_rms_probe": rms(eps_lag_probe),
                "diag_source/q_ensemble_std_mean_probe": ensemble_std_probe,
                "diag_source/td_error_rms_train": rms(td_error_old),

                # B estimators on the same random direction: this directly tests whether
                # the cheap estimators agree with paired finite differences.
                "diag_B/random_source_rms": random_source_rms,
                "diag_B/fd_adam_random_gain_probe": rms(random_fd),
                "diag_B/jvp_adam_random_gain_probe": rms(random_adam_jvp),
                "diag_B/ntk_sgd_random_gain_probe": rms(random_ntk),
                "diag_B/fd_vs_jvp_adam_cos_random": cosine(random_fd, random_adam_jvp),
                "diag_B/jvp_adam_vs_ntk_cos_random": cosine(random_adam_jvp, random_ntk),

                # Directional propagation of the estimated IQL residual channel.
                "diag_prop/h_v_fd_adam_q_rms": rms(h_v_fd) * h_v_source_rms,
                "diag_prop/h_v_jvp_adam_q_rms": rms(h_v_adam_jvp) * h_v_source_rms,
                "diag_prop/h_v_ntk_sgd_q_rms": rms(h_v_ntk) * h_v_source_rms,
                "diag_prop/h_v_fd_vs_jvp_cos": cosine(h_v_fd, h_v_adam_jvp),

                # h_Q is intentionally labelled a lag proxy, not the unobservable true epsilon channel.
                "diag_prop/h_q_lag_fd_adam_q_rms": rms(h_q_fd) * h_q_source_rms,
                "diag_prop/h_q_lag_jvp_adam_q_rms": rms(h_q_adam_jvp) * h_q_source_rms,
                "diag_prop/h_q_lag_ntk_sgd_q_rms": rms(h_q_ntk) * h_q_source_rms,
                "diag_prop/h_q_lag_fd_vs_jvp_cos": cosine(h_q_fd, h_q_adam_jvp),

                # Per-target spillover / cross-sample coupling.
                "diag_B/basis_adam_self_abs_mean": adam_self,
                "diag_B/basis_adam_spillover_l2_mean": adam_spill,
                "diag_B/basis_adam_spillover_ratio_mean": adam_spill_ratio,
                "diag_B/basis_ntk_self_abs_mean": ntk_self,
                "diag_B/basis_ntk_spillover_l2_mean": ntk_spill,
                "diag_B/basis_ntk_spillover_ratio_mean": ntk_spill_ratio,
            }
            return log_dict

        return diagnostic_step

    def _build_ridge_diagnostic_step(self):
        """Local fit-to-convergence B approximation using matrix-free CG.

        For W=I this estimates
            B_ridge h = J_Z (J_X^T J_X / m + lambda I)^-1 J_X^T h_rep / m.
        It is intentionally separate from the one-step Adam B: the point of logging both
        is to test whether short-horizon optimizer sensitivity and local full-fit sensitivity
        tell the same empirical story.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        discount = self.discount
        num_critics = self.num_critics
        ridge_lambda = self.diag_ridge_lambda
        ridge_iters = self.diag_ridge_cg_iters
        ridge_tol = self.diag_ridge_cg_tol
        diag_seed = self.diag_seed

        def rms(x):
            return jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2) + 1e-12)

        def cosine(a, b):
            a = jnp.ravel(a)
            b = jnp.ravel(b)
            return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)

        @jax.jit
        def ridge_step(old_state: IQLState, batch: TensorBatch, probe_batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)
            probe_observations = probe_batch["observations"]
            probe_actions = probe_batch["actions"]

            next_v = v_apply({"params": old_state.v_params}, next_observations)
            target = rewards + (1.0 - dones) * discount * next_v

            key = jax.random.fold_in(jax.random.PRNGKey(diag_seed), old_state.total_it + 1)
            h = jax.random.rademacher(key, shape=target.shape, dtype=jnp.float32)
            h_unit = h / jnp.maximum(rms(h), 1e-8)

            flat_params, unravel = ravel_pytree(old_state.q_params)

            def train_output_flat(flat):
                params = unravel(flat)
                return q_apply({"params": params}, observations, actions).reshape(-1)

            def probe_output_flat(flat):
                params = unravel(flat)
                return q_apply({"params": params}, probe_observations, probe_actions).reshape(-1)

            train_value, pullback = jax.vjp(train_output_flat, flat_params)
            h_rep = jnp.broadcast_to(h_unit[None, :], (num_critics, h_unit.shape[0])).reshape(-1)
            m = float(train_value.size)
            rhs = pullback(h_rep)[0] / m

            def matvec(vec):
                _, jv = jax.jvp(train_output_flat, (flat_params,), (vec,))
                return pullback(jv)[0] / m + ridge_lambda * vec

            solution, _ = jax_cg(
                matvec,
                rhs,
                tol=ridge_tol,
                maxiter=ridge_iters,
            )
            _, ridge_out_flat = jax.jvp(probe_output_flat, (flat_params,), (solution,))
            ridge_out = ridge_out_flat.reshape(num_critics, probe_observations.shape[0])
            relative_linear_residual = jnp.linalg.norm(matvec(solution) - rhs) / (jnp.linalg.norm(rhs) + 1e-12)

            # Same random direction through the actual one-step Adam update for comparison.
            def actual_probe_output(target_arg):
                def q_loss_only(q_params):
                    qs_local = q_apply({"params": q_params}, observations, actions)
                    return jnp.mean((qs_local - target_arg[None, :]) ** 2)

                q_grads = jax.grad(q_loss_only)(old_state.q_params)
                q_updates, _ = q_tx.update(q_grads, old_state.q_opt_state, old_state.q_params)
                q_params_plus = optax.apply_updates(old_state.q_params, q_updates)
                return q_apply({"params": q_params_plus}, probe_observations, probe_actions)

            _, adam_jvp = jax.jvp(actual_probe_output, (target,), (h_unit,))

            return {
                "diag_B/ridge_random_gain_probe": rms(ridge_out),
                "diag_B/ridge_vs_jvp_adam_cos_random": cosine(ridge_out, adam_jvp),
                "diag_B/ridge_cg_relative_residual": relative_linear_residual,
            }

        return ridge_step

    def train(
        self,
        batch: TensorBatch,
        diag_batch: Optional[TensorBatch] = None,
        run_ridge_diagnostic: bool = False,
    ) -> Dict[str, float]:
        old_state = self.state
        self.state, log_dict = self._train_step(self.state, batch)

        if self.enable_propagation_diagnostics and diag_batch is not None:
            diag_log = self._propagation_diagnostic_step(
                old_state,
                self.state,
                batch,
                diag_batch,
            )
            log_dict.update(diag_log)

            if run_ridge_diagnostic:
                ridge_log = self._ridge_diagnostic_step(old_state, batch, diag_batch)
                log_dict.update(ridge_log)

        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
        policy_out = self._apply_actor(actor_params, state_jnp, training=False)
        action = policy_out if self.iql_deterministic else policy_out[0]
        action = jnp.clip(self.max_action * action, -self.max_action, self.max_action)
        return np.asarray(jax.device_get(action))[0]

    def eval_actor(
        self,
        env: gym.Env,
        actor_params: Any,
        n_episodes: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        episode_rewards = []
        episode_successes = []
        for episode_idx in range(n_episodes):
            state, done = reset_env(env, seed=seed if episode_idx == 0 else None), False
            episode_reward = 0.0
            episode_success = np.nan
            while not done:
                action = self.actor_act(actor_params, state)
                state, reward, done, info = step_env(env, action)
                episode_reward += reward
                step_success = extract_success(info)
                if np.isfinite(step_success):
                    episode_success = step_success
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
        return (
            np.asarray(episode_rewards, dtype=np.float32),
            np.asarray(episode_successes, dtype=np.float32),
        )

    def _build_actor_refit_step(self):
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        actor_tx = self.actor_tx
        beta = self.beta
        iql_deterministic = self.iql_deterministic
        use_dropout = self.actor_dropout is not None
        actor_apply_fn = self.actor_def.apply

        def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
            if use_dropout and training:
                return actor_apply_fn(
                    {"params": actor_params},
                    observations,
                    training=training,
                    rngs={"dropout": rng},
                )
            return actor_apply_fn({"params": actor_params}, observations, training=training)

        @jax.jit
        def actor_refit_step(actor_state: ActorState, iql_state: IQLState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]

            qs = q_apply({"params": iql_state.q_target_params}, observations, actions)
            target_q = jnp.min(qs, axis=0)
            v = v_apply({"params": iql_state.v_params}, observations)
            adv = target_q - v
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_losses = jnp.sum((policy_out - actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
                    bc_losses = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_losses)
                return actor_loss, (bc_losses, policy_mean, log_std_mean)

            (actor_loss, (bc_losses, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_state.params)
            actor_updates, actor_opt_state = actor_tx.update(
                actor_grads,
                actor_state.opt_state,
                actor_state.params,
            )
            actor_params = optax.apply_updates(actor_state.params, actor_updates)
            new_actor_state = ActorState(
                params=actor_params,
                opt_state=actor_opt_state,
                key=actor_key,
            )
            log_dict = {
                "loss": actor_loss,
                "bc_loss": jnp.mean(bc_losses),
                "adv_mean": jnp.mean(adv),
                "adv_min": jnp.min(adv),
                "adv_max": jnp.max(adv),
                "exp_adv_mean": jnp.mean(exp_adv),
                "exp_adv_max": jnp.max(exp_adv),
                "target_q_mean": jnp.mean(target_q),
                "v_mean": jnp.mean(v),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
            }
            return new_actor_state, log_dict

        return actor_refit_step

    def make_initial_actor_state(self) -> ActorState:
        return tree_to_device(
            ActorState(
                params=copy.deepcopy(self.initial_actor_params),
                opt_state=copy.deepcopy(self.initial_actor_opt_state),
                key=copy.deepcopy(self.initial_actor_key),
            ),
            self.device,
        )

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
        log_wandb: bool = False,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ActorState, Dict[str, Any]]:
        refit_log: Dict[str, Any] = {
            f"{prefix}/final_loss": np.nan,
            f"{prefix}/final_bc_loss": np.nan,
            f"{prefix}/final_adv_mean": np.nan,
            f"{prefix}/final_exp_adv_mean": np.nan,
            f"{prefix}/final_score_mean": np.nan,
            f"{prefix}/final_score_std": np.nan,
            f"{prefix}/final_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/final_d4rl_normalized_score_std": np.nan,
            f"{prefix}/final_success_rate": np.nan,
            f"{prefix}/final_success_std": np.nan,
            f"{prefix}/best_score_mean": np.nan,
            f"{prefix}/best_score_std": np.nan,
            f"{prefix}/best_d4rl_normalized_score_mean": np.nan,
            f"{prefix}/best_d4rl_normalized_score_std": np.nan,
            f"{prefix}/best_success_rate": np.nan,
            f"{prefix}/best_success_std": np.nan,
            f"{prefix}/inner_eval_steps": [],
            f"{prefix}/inner_score_mean": [],
            f"{prefix}/inner_score_std": [],
            f"{prefix}/inner_d4rl_normalized_score_mean": [],
            f"{prefix}/inner_d4rl_normalized_score_std": [],
            f"{prefix}/inner_success_rate": [],
            f"{prefix}/inner_success_std": [],
        }
        if steps <= 0:
            return actor_state, refit_log

        best_eval_metric_mean = -np.inf
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
        log_extra = {} if log_extra is None else dict(log_extra)

        def save_refit_snapshot(
            current_actor_state: ActorState,
            current_refit_log: Dict[str, Any],
            fit_step: int,
            is_best: bool,
        ) -> None:
            if save_dir_path is None:
                return

            logs_payload = {
                **log_extra,
                "refit_step": int(fit_step),
                **current_refit_log,
            }
            logs_path = save_dir_path / "fit_eval_logs.npz"
            latest_actor_path = save_dir_path / "latest_actor.pkl"
            actor_payload = serialization.to_state_dict(current_actor_state.params)
            save_pickle(latest_actor_path, actor_payload)
            save_logs_npz([logs_payload], str(logs_path))

            if is_best:
                save_pickle(save_dir_path / "best_actor.pkl", actor_payload)

            if log_wandb and wandb.run is not None:
                wandb.save(str(logs_path), policy="now")
                wandb.save(str(latest_actor_path), policy="now")
                if is_best:
                    wandb.save(str(save_dir_path / "best_actor.pkl"), policy="now")

        for fit_step in range(1, steps + 1):
            batch = replay_buffer.sample(batch_size)
            actor_state, step_log = self._actor_refit_step(actor_state, self.state, batch)
            step_log = {key: float(jax.device_get(value)) for key, value in step_log.items()}

            refit_log[f"{prefix}/final_loss"] = step_log["loss"]
            refit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]
            refit_log[f"{prefix}/final_adv_mean"] = step_log["adv_mean"]
            refit_log[f"{prefix}/final_exp_adv_mean"] = step_log["exp_adv_mean"]

            should_eval = (
                eval_env is not None
                and eval_episodes > 0
                and eval_interval > 0
                and (fit_step % eval_interval == 0 or fit_step == steps)
            )
            if should_eval:
                eval_scores, eval_successes = self.eval_actor(
                    eval_env,
                    actor_state.params,
                    n_episodes=eval_episodes,
                    seed=eval_seed,
                )
                normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

                eval_score_mean = float(np.mean(eval_scores))
                eval_score_std = float(np.std(eval_scores))
                normalized_eval_score_mean, normalized_eval_score_std = mean_std_or_nan(normalized_eval_scores)
                success_rate, success_std = mean_std_or_nan(eval_successes)

                refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
                refit_log[f"{prefix}/inner_success_rate"].append(success_rate)
                refit_log[f"{prefix}/inner_success_std"].append(success_std)
                refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                refit_log[f"{prefix}/final_score_std"] = eval_score_std
                refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std
                refit_log[f"{prefix}/final_success_rate"] = success_rate
                refit_log[f"{prefix}/final_success_std"] = success_std

                eval_metric_mean = success_rate if np.isfinite(success_rate) else normalized_eval_score_mean
                is_best = np.isfinite(eval_metric_mean) and eval_metric_mean > best_eval_metric_mean
                if is_best:
                    best_eval_metric_mean = eval_metric_mean
                    refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                    refit_log[f"{prefix}/best_score_std"] = eval_score_std
                    refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                    refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std
                    refit_log[f"{prefix}/best_success_rate"] = success_rate
                    refit_log[f"{prefix}/best_success_std"] = success_std

                save_refit_snapshot(
                    current_actor_state=actor_state,
                    current_refit_log=refit_log,
                    fit_step=fit_step,
                    is_best=is_best,
                )

                print(
                    f"[{prefix}:iql_awbc] step {fit_step}/{steps}: "
                    f"loss={step_log['loss']:.4f}, bc={step_log['bc_loss']:.4f}, "
                    f"adv={step_log['adv_mean']:.4f}, exp_adv={step_log['exp_adv_mean']:.4f}, "
                    f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                    f"d4rl_normalized_mean={normalized_eval_score_mean:.3f}, "
                    f"d4rl_normalized_std={normalized_eval_score_std:.3f}, "
                    f"success_rate={success_rate:.3f}"
                )

        return actor_state, refit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "iql_state": serialization.to_state_dict(self.state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
            "iql_deterministic": self.iql_deterministic,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Backward-compatible with checkpoints created before the diagnostic head existed.
        loaded_iql_state = dict(state_dict["iql_state"])
        current_iql_state = serialization.to_state_dict(self.state)
        for diagnostic_field in ("diag_params", "diag_opt_state"):
            if diagnostic_field not in loaded_iql_state:
                loaded_iql_state[diagnostic_field] = current_iql_state[diagnostic_field]
        self.state = serialization.from_state_dict(self.state, loaded_iql_state)
        if "initial_actor_params" in state_dict:
            self.initial_actor_params = serialization.from_state_dict(
                self.initial_actor_params,
                state_dict["initial_actor_params"],
            )
        if "initial_actor_opt_state" in state_dict:
            self.initial_actor_opt_state = serialization.from_state_dict(
                self.initial_actor_opt_state,
                state_dict["initial_actor_opt_state"],
            )
        if "initial_actor_key" in state_dict:
            self.initial_actor_key = serialization.from_state_dict(
                self.initial_actor_key,
                state_dict["initial_actor_key"],
            )
        self.state = tree_to_device(self.state, self.device)
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
    """Return (run_dir, checkpoint_path) for a saved IQL-JAX checkpoint.

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


def load_run_config_for_refit(
    current_config: TrainConfig,
    loaded_run_dir: Union[str, Path],
) -> TrainConfig:
    """Load saved config.yaml from the checkpoint run dir for refit mode.

    Priority:
        saved run config.yaml < explicit CLI flags

    This reconstructs the original training env/model/preprocessing settings,
    while allowing any CLI-provided field to override the saved config.
    """
    loaded_run_dir = Path(loaded_run_dir)
    saved_config_path = loaded_run_dir / "config.yaml"

    if not saved_config_path.exists():
        raise FileNotFoundError(
            f"mode='refit' expects saved run config at: {saved_config_path}"
        )

    with open(saved_config_path, "r") as f:
        saved_raw = yaml.safe_load(f) or {}

    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    saved_kwargs = {
        key: _coerce_hparam_value(value)
        for key, value in saved_raw.items()
        if key in config_fields
    }

    # 1. Start from the original saved training config.
    loaded_config = TrainConfig(**saved_kwargs)

    # 2. Override with explicitly provided CLI fields.
    cli_overrides = _cli_overridden_fields()
    current_config_dict = asdict(current_config)

    applied_cli_overrides = []
    for key in sorted(cli_overrides):
        if key not in config_fields:
            continue
        setattr(loaded_config, key, current_config_dict[key])
        applied_cli_overrides.append(key)

    # 3. These must be forced for refit regardless of saved training config.
    loaded_config.mode = "refit"
    loaded_config.load_model = current_config.load_model

    # In refit mode, do not reuse the original training checkpoint output path.
    # Actor refit outputs are saved under loaded_run_dir / actor_refit_dir_name.
    loaded_config.checkpoints_path = None

    refresh_algorithm_names(loaded_config)
    validate_config(loaded_config)

    print(f"Loaded saved run config for refit from: {saved_config_path}")
    if applied_cli_overrides:
        print(
            "Applied explicit CLI overrides on top of saved config: "
            + ", ".join(applied_cli_overrides)
        )

    return loaded_config


def _train_impl(config: TrainConfig):
    refit_only = config.mode == "refit"

    loaded_run_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    if refit_only:
        if config.load_model == "":
            raise ValueError("refit mode requires --load_model")

        # First resolve the checkpoint using the current CLI config.
        # This lets --load_model be either checkpoint.pkl, a run dir, or a parent dir.
        loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )

        # Then replace config with the original saved run config,
        # while preserving refit-specific CLI/runtime fields.
        config = load_run_config_for_refit(
            current_config=config,
            loaded_run_dir=loaded_run_dir,
        )

    else:
        config = apply_env_hyperparams(config)
        config = finalize_checkpoint_path(config)

    checkpoint_manager = None
    checkpoint_preparation = None
    if config.checkpoints_path is not None and (not refit_only):
        current_config_dict = asdict(config)
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path,
            current_config=current_config_dict,
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="iql_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=getattr(config, "wandb_entity", None),
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        checkpoint_preparation = checkpoint_manager.prepare()
        print(checkpoint_preparation.message)
        if checkpoint_preparation.is_completed:
            return

    jax_device = select_jax_device(config.device)
    env, dataset, dataset_backend = load_env_and_dataset(config.env)

    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError(
            f"{ALGORITHM_NAME}-JAX currently supports vector observations/actions only; "
            f"got observation_space={env.observation_space}, action_space={env.action_space}."
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.normalize_reward and dataset_backend == "d4rl":
        modify_reward(dataset, config.env)
    elif config.normalize_reward:
        print("Skipping D4RL reward normalization for non-D4RL dataset.")

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


    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    run_mode_name = "Actor refit" if refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = IQLJAX(
        max_action=max_action,
        state_dim=state_dim,
        action_dim=action_dim,
        max_steps=max(int(config.max_timesteps), 1),
        qf_lr=config.qf_lr,
        vf_lr=config.vf_lr,
        actor_lr=config.actor_lr,
        discount=config.discount,
        tau=config.tau,
        beta=config.beta,
        iql_tau=config.iql_tau,
        iql_deterministic=config.iql_deterministic,
        actor_dropout=config.actor_dropout,
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
        num_critics=config.num_critics,
        enable_propagation_diagnostics=config.enable_propagation_diagnostics,
        diag_hidden_dim=config.diag_hidden_dim,
        diag_head_lr=config.diag_head_lr,
        diag_kappa_floor=config.diag_kappa_floor,
        diag_fd_epsilon=config.diag_fd_epsilon,
        diag_num_basis_directions=config.diag_num_basis_directions,
        diag_ridge_lambda=config.diag_ridge_lambda,
        diag_ridge_cg_iters=config.diag_ridge_cg_iters,
        diag_ridge_cg_tol=config.diag_ridge_cg_tol,
        seed=seed,
        device=jax_device,
    )

    if config.load_model != "":
        if checkpoint_path is None or loaded_run_dir is None:
            loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
                config.load_model,
                run_name=config.name,
                seed=config.seed,
            )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        trainer.load_state_dict(checkpoint)

    def _progress_state():
        return trainer.state_dict()

    def _final_state():
        return trainer.state_dict()

    def _load_progress_state(payload):
        trainer.load_state_dict(payload)

    def _training_timestep():
        return int(jax.device_get(trainer.state.total_it))

    start_timestep = _training_timestep()
    eval_logs: List[Dict[str, Any]] = []
    diagnostic_log_path: Optional[Path] = None
    if config.checkpoints_path is not None and config.enable_propagation_diagnostics:
        diagnostic_log_path = Path(config.checkpoints_path) / "propagation_diagnostics.jsonl"
        diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_manager is not None:
        if checkpoint_preparation.is_resuming:
            start_timestep, eval_logs, _ = checkpoint_manager.restore(
                load_trainer_state=_load_progress_state,
                get_restored_timestep=_training_timestep,
            )
            print(f"Restored training at timestep {start_timestep}.")
        else:
            start_timestep = _training_timestep()
            checkpoint_manager.save_progress(
                timestep=start_timestep,
                trainer_state=_progress_state(),
                eval_logs=eval_logs,
                status="running",
            )

    if config.log_wandb:
        if checkpoint_manager is not None:
            if not checkpoint_manager.initialize_wandb_with_fallback(
                wandb_module=wandb,
                config=asdict(config),
                code_root=_PROJECT_ROOT,
            ):
                config.log_wandb = False
        else:
            try:
                wandb_init(asdict(config))
            except Exception as exc:
                print(f"Warning: W&B initialization failed: {exc}")
                print("Continuing local training with W&B disabled.")
                config.log_wandb = False


    def _wandb_log(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))


    if refit_only:
        if loaded_run_dir is None:
            raise ValueError("refit mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
        print("Q/V are frozen; only pi is optimized.")
        print(
            "Refit schedule uses shared fields: "
            f"max_timesteps={config.max_timesteps}, "
            f"batch_size={config.batch_size}, "
            f"eval_freq={config.eval_freq}"
        )
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        actor_state = trainer.make_initial_actor_state()
        loaded_checkpoint_for_log = str(loaded_run_dir / "checkpoint.pkl")

        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=actor_state,
            steps=config.max_timesteps,
            batch_size=config.batch_size,
            eval_env=env,
            eval_episodes=config.n_episodes,
            eval_seed=config.seed,
            eval_interval=config.eval_freq,
            prefix="actor_refit",
            save_dir=actor_refit_dir,
            log_wandb=config.log_wandb,
            log_extra={"loaded_checkpoint": loaded_checkpoint_for_log},
        )

        save_pickle(
            actor_refit_dir / "final_actor.pkl",
            serialization.to_state_dict(refit_actor_state.params),
        )
        save_logs_npz(
            [{"loaded_checkpoint": loaded_checkpoint_for_log, **refit_log}],
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

    def _evaluation_required(step):
        return evaluation_is_due(int(step), int(config.eval_freq))

    try:
        for t in range(start_timestep, int(config.max_timesteps)):
            batch = replay_buffer.sample(config.batch_size)
            step = t + 1
            diag_due = (
                config.enable_propagation_diagnostics
                and step % int(config.diag_every) == 0
            )
            ridge_due = (
                diag_due
                and config.diag_enable_ridge_b
                and step % int(config.diag_ridge_every) == 0
            )
            diag_batch = (
                replay_buffer.sample(config.diag_probe_batch_size)
                if diag_due
                else None
            )
            log_dict = trainer.train(
                batch,
                diag_batch=diag_batch,
                run_ridge_diagnostic=ridge_due,
            )

            if diag_due:
                diagnostic_record = {
                    "timestep": int(step),
                    **{
                        key: to_python_scalar(value)
                        for key, value in log_dict.items()
                        if key.startswith("diag_")
                    },
                }
                if diagnostic_log_path is not None:
                    with open(diagnostic_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(diagnostic_record, allow_nan=True) + "\n")

            if config.log_wandb and (step % config.log_every == 0 or diag_due):
                _wandb_log(log_dict, int(jax.device_get(trainer.state.total_it)))
    
            if (t + 1) % config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                eval_scores, eval_successes = trainer.eval_actor(
                    env,
                    trainer.state.actor_params,
                    n_episodes=config.n_episodes,
                    seed=config.seed,
                )
                normalized_eval_scores = normalize_episode_scores(env, eval_scores)
                normalized_eval_score_mean, normalized_eval_score_std = mean_std_or_nan(normalized_eval_scores)
                success_rate, success_std = mean_std_or_nan(eval_successes)
    
                eval_log: Dict[str, Any] = {
                    "timestep": int(t + 1),
                    "eval/reward_mean": float(np.mean(eval_scores)),
                    "eval/reward_std": float(np.std(eval_scores)),
                    "eval/d4rl_normalized_score_mean": normalized_eval_score_mean,
                    "eval/d4rl_normalized_score_std": normalized_eval_score_std,
                    "eval/success_rate": success_rate,
                    "eval/success_std": success_std,
                }
                upsert_eval_log(eval_logs, eval_log)
    
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"reward={eval_log['eval/reward_mean']:.3f} ± {eval_log['eval/reward_std']:.3f}, "
                    f"d4rl_normalized={eval_log['eval/d4rl_normalized_score_mean']:.3f} ± "
                    f"{eval_log['eval/d4rl_normalized_score_std']:.3f}, "
                    f"success_rate={eval_log['eval/success_rate']:.3f} ± "
                    f"{eval_log['eval/success_std']:.3f}"
                )
    
                if config.log_wandb:
                    wandb_eval_log = {
                        key: to_python_scalar(value)
                        for key, value in eval_log.items()
                        if is_scalar_value(value)
                    }
                    _wandb_log(wandb_eval_log, int(jax.device_get(trainer.state.total_it)))
    
                save_and_upload_eval_logs(
                    eval_logs=eval_logs,
                    checkpoints_path=config.checkpoints_path,
                    log_wandb=config.log_wandb,
                )
    
            current_timestep = _training_timestep()
            if (
                checkpoint_manager is not None
                and current_timestep % int(config.checkpoint_freq) == 0
            ):
                checkpoint_manager.save_progress(
                    timestep=current_timestep,
                    trainer_state=_progress_state(),
                    eval_logs=eval_logs,
                    status="running",
                )
    except BaseException:
        if checkpoint_manager is not None:
            interrupted_timestep = _training_timestep()
            evaluation_complete = (
                not _evaluation_required(interrupted_timestep)
                or find_eval_log(eval_logs, interrupted_timestep) is not None
            )
            if evaluation_complete:
                checkpoint_manager.save_progress(
                    timestep=interrupted_timestep,
                    trainer_state=_progress_state(),
                    eval_logs=eval_logs,
                    status="interrupted",
                )
                print(f"Saved interrupted checkpoint at timestep {interrupted_timestep}.")
            else:
                print(
                    "Evaluation was interrupted before its result was committed; "
                    "retaining the previous safe checkpoint so evaluation cannot be skipped."
                )
        raise
    final_timestep = _training_timestep()
    if checkpoint_manager is not None:
        final_path = checkpoint_manager.complete(
            timestep=final_timestep,
            final_state=_final_state(),
            save_final_model=bool(config.save_final_model),
            eval_logs=eval_logs,
        )
        if final_path is not None:
            print("---------------------------------------")
            print(f"Saved final checkpoint to: {final_path}")
            print("---------------------------------------")



@pyrallis.wrap()
@log_training_exceptions
def train(config: TrainConfig):
    exit_code = 0
    try:
        return _train_impl(config)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if getattr(wandb, "run", None) is not None:
            wandb.finish(exit_code=exit_code)

            
if __name__ == "__main__":
    train()
