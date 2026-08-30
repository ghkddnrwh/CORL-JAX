# JAX/Flax Decoupled Delayed IQL implementation with CDAF_JAX-style experiment plumbing.
#
# Decoupled Delayed IQL idea:
#   - Use an ensemble of N paired Q_i and V_i networks.
#   - Remove the original twin-Q min operator entirely.
#   - Q_i is always bootstrapped from its own V_i.
#   - V_i is regressed to its own Q_i, but its asymmetric expectile weight is
#     computed from a delayed, decoupled pair j(i): Q_j_delayed - V_j_delayed.
#   - By default, if N=1, j(i)=i; if N=2, the two members filter each other;
#     if N>=3, the one-to-one filtering assignment cycles on delayed refreshes.
#   - Expectile filtering supports three index schedules:
#       cross         -> always use another ensemble member (for N>=2).
#       self          -> always use the same ensemble member.
#       periodic_self -> mostly cross-filter, but periodically use self-filtering.
#   - Actor updates always use ensemble-mean Q and ensemble-mean V.
#
# Experiment plumbing:
#   - Explicit mode switch: mode="train" or mode="refit".
#   - Refit mode uses shared schedule fields:
#       max_timesteps -> actor-only refit steps
#       batch_size    -> actor-only refit batch size
#       eval_freq     -> actor-only refit evaluation interval
#   - Hyperparameter YAML keys must exactly match TrainConfig field names.

import copy
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

ALGORITHM_NAME = "DDIQL"
ALGORITHM_FULL_NAME = "Decoupled Delayed Implicit Q-Learning"

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
    hyperparams_path: Optional[str] = "hyperparams/decoupled_delayed_iql_jax.yml"
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

    # Decoupled delayed ensemble learning
    ensemble_size: int = 2
    delayed_update_period: int = 250

    # Expectile-filter index schedule:
    #   "legacy"        : preserve delayed_expectile_self_index behavior below.
    #   "cross"         : always use another ensemble member (original False behavior).
    #   "self"          : always use the same ensemble member (original True behavior).
    #   "periodic_self" : use cross filtering most of the time and self filtering
    #                       once every delayed_expectile_self_period delayed phases.
    delayed_expectile_index_mode: str = "legacy"
    delayed_expectile_self_period: int = 5

    # Backward compatibility for existing YAML/checkpoints. Used only when
    # delayed_expectile_index_mode == "legacy".
    delayed_expectile_self_index: bool = False

    # Standalone actor refit output directory.
    # Refit reuses the shared training schedule fields above:
    #   max_timesteps -> actor-only refit steps
    #   batch_size    -> actor-only refit batch size
    #   eval_freq     -> actor-only refit evaluation interval
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-BIAS"
    group: str = "DDIQL-JAX"
    name: str = "DDIQL-JAX"
    log_wandb: bool = True
    log_every: int = 500
    save_final_model: bool = False

    checkpoint_freq: int = int(25e3)
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    # config.project = "ORL-BIAS"
    # config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"
    # config.name = f"{config.name}-{config.env}"


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
    assert config.ensemble_size >= 1
    assert config.delayed_update_period > 0
    assert str(config.delayed_expectile_index_mode).lower() in (
        "legacy", "cross", "self", "periodic_self"
    )
    assert config.delayed_expectile_self_period > 0
    assert isinstance(config.delayed_expectile_self_index, bool)
    if config.actor_dropout is not None:
        assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
    assert config.hidden_dim > 0
    assert config.n_hidden > 0
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
        print(f"Ignored unknown hyperparameter keys for {ALGORITHM_NAME}: {', '.join(skipped_unknown)}")
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


def stack_ensemble_params(params_list: List[Any]) -> Any:
    """Stack a list of Flax parameter PyTrees along a leading ensemble axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


def cycle_filter_indices(ensemble_size: int, shift: Union[int, jnp.ndarray]) -> jnp.ndarray:
    """Return deterministic cycle-shift mapping i -> j for decoupled filtering.

    n=1: [0]
    n=2: shift 1 gives [1, 0]
    n>=3: shift cycles through 1, 2, ..., n-1.

    For n>=2 and shift in {1, ..., n-1}, this is one-to-one and never maps
    an ensemble member to itself.
    """
    if ensemble_size == 1:
        return jnp.zeros((1,), dtype=jnp.int32)
    indices = jnp.arange(ensemble_size, dtype=jnp.int32)
    return (indices + jnp.asarray(shift, dtype=jnp.int32)) % jnp.asarray(ensemble_size, dtype=jnp.int32)


DELAYED_EXPECTILE_INDEX_MODES = ("legacy", "cross", "self", "periodic_self")


def resolve_delayed_expectile_index_mode(
    mode: str,
    delayed_expectile_self_index: bool = False,
) -> str:
    """Resolve the new schedule mode while preserving the old boolean interface."""
    mode = str(mode).lower()
    if mode not in DELAYED_EXPECTILE_INDEX_MODES:
        raise ValueError(
            f"Unknown delayed_expectile_index_mode={mode!r}. "
            f"Expected one of {DELAYED_EXPECTILE_INDEX_MODES}."
        )
    if mode == "legacy":
        return "self" if delayed_expectile_self_index else "cross"
    return mode


def scheduled_filter_indices(
    ensemble_size: int,
    delayed_round: Union[int, jnp.ndarray],
    mode: str,
    self_period: int,
) -> jnp.ndarray:
    """Return j(i) for a delayed-refresh phase.

    periodic_self starts in cross mode. With self_period=5, the phases are
    cross, cross, cross, cross, self, then repeat. Cross phases continue the
    original non-self cycle while skipping self phases in the cross counter.
    """
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")
    if self_period <= 0:
        raise ValueError("self_period must be > 0")
    if mode not in ("cross", "self", "periodic_self"):
        raise ValueError(f"scheduled_filter_indices requires a resolved mode, got {mode!r}")

    indices = jnp.arange(ensemble_size, dtype=jnp.int32)
    if ensemble_size == 1 or mode == "self":
        return indices

    delayed_round = jnp.asarray(delayed_round, dtype=jnp.int32)
    ensemble_mod = jnp.asarray(ensemble_size, dtype=jnp.int32)
    cross_mod = jnp.asarray(ensemble_size - 1, dtype=jnp.int32)

    if mode == "cross":
        shift = jnp.asarray(1, dtype=jnp.int32) + (delayed_round % cross_mod)
        return (indices + shift) % ensemble_mod

    # periodic_self: every K-th phase is self, with phase 0 starting as cross.
    period = jnp.asarray(self_period, dtype=jnp.int32)
    is_self_round = ((delayed_round + 1) % period) == 0

    # Remove completed self phases from the cross-cycle counter so that, for
    # N>=3, cross shifts continue 1,2,...,N-1 without being advanced by self.
    completed_self_rounds = (delayed_round + 1) // period
    cross_round = delayed_round - completed_self_rounds
    cross_shift = jnp.asarray(1, dtype=jnp.int32) + (cross_round % cross_mod)
    cross_indices = (indices + cross_shift) % ensemble_mod
    return jnp.where(is_self_round, indices, cross_indices)


def initial_filter_indices(
    ensemble_size: int,
    mode: str,
    self_period: int,
) -> jnp.ndarray:
    return scheduled_filter_indices(
        ensemble_size=ensemble_size,
        delayed_round=jnp.asarray(0, dtype=jnp.int32),
        mode=mode,
        self_period=self_period,
    )


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
class DDIQLState:
    total_it: jnp.ndarray
    q_params: Any
    q_target_params: Any
    q_delayed_params: Any
    q_opt_state: Any
    v_params: Any
    v_delayed_params: Any
    v_opt_state: Any
    filter_indices: jnp.ndarray
    actor_params: Any
    actor_opt_state: Any
    actor_key: jnp.ndarray


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


class DDIQLJAX:
    """Decoupled Delayed IQL in JAX/Flax.

    For each ensemble member i:

        Q_i target: r + gamma * V_i(s')
        V_i target: Q_i_target(s, a)
        V_i weight: |tau - 1[Q_j_delayed(s, a) - V_j_delayed(s) < 0]|

    where j = filter_indices[i]. The index schedule can be "cross", "self",
    or "periodic_self". Cross preserves the original one-to-one non-self cycle;
    self fixes j(i)=i; periodic_self periodically inserts a self-filter phase.

    Actor AWBC uses ensemble means:

        adv_actor = mean_i Q_i_target(s, a) - mean_i V_i(s)
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
        ensemble_size: int = 2,
        delayed_update_period: int = 250,
        delayed_expectile_index_mode: str = "legacy",
        delayed_expectile_self_period: int = 5,
        delayed_expectile_self_index: bool = False,
        iql_deterministic: bool = False,
        actor_dropout: Optional[float] = None,
        hidden_dim: int = 256,
        n_hidden: int = 2,
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
        self.ensemble_size = int(ensemble_size)
        self.delayed_update_period = int(delayed_update_period)
        self.delayed_expectile_self_period = int(delayed_expectile_self_period)
        self.delayed_expectile_self_index = bool(delayed_expectile_self_index)
        self.delayed_expectile_index_mode = resolve_delayed_expectile_index_mode(
            delayed_expectile_index_mode,
            delayed_expectile_self_index=self.delayed_expectile_self_index,
        )
        self.iql_deterministic = iql_deterministic
        self.actor_dropout = actor_dropout
        self.hidden_dim = int(hidden_dim)
        self.n_hidden = int(n_hidden)
        self.device = device if device is not None else jax.devices()[0]

        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if self.delayed_update_period <= 0:
            raise ValueError("delayed_update_period must be > 0")
        if self.delayed_expectile_self_period <= 0:
            raise ValueError("delayed_expectile_self_period must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be > 0")

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
        self.q_def = QFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
        self.v_def = ValueFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)

        self.q_tx = optax.adam(qf_lr)
        self.v_tx = optax.adam(vf_lr)
        actor_lr_schedule = optax.cosine_decay_schedule(
            init_value=actor_lr,
            decay_steps=self.max_steps,
            alpha=0.0,
        )
        self.actor_tx = optax.adam(actor_lr_schedule)

        key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(key, 2 + 2 * self.ensemble_size)
        key_actor = init_keys[0]
        actor_key = init_keys[1]
        q_keys = init_keys[2 : 2 + self.ensemble_size]
        v_keys = init_keys[2 + self.ensemble_size :]
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
        q_params = stack_ensemble_params([
            self.q_def.init(q_key, dummy_state, dummy_action)["params"]
            for q_key in q_keys
        ])
        v_params = stack_ensemble_params([
            self.v_def.init(v_key, dummy_state)["params"]
            for v_key in v_keys
        ])

        self.initial_actor_params = copy.deepcopy(actor_params)
        self.initial_actor_opt_state = self.actor_tx.init(actor_params)
        self.initial_actor_key = actor_key

        self.state = DDIQLState(
            total_it=jnp.asarray(0, dtype=jnp.int32),
            q_params=q_params,
            q_target_params=copy.deepcopy(q_params),
            q_delayed_params=copy.deepcopy(q_params),
            q_opt_state=self.q_tx.init(q_params),
            v_params=v_params,
            v_delayed_params=copy.deepcopy(v_params),
            v_opt_state=self.v_tx.init(v_params),
            filter_indices=initial_filter_indices(
                ensemble_size=self.ensemble_size,
                mode=self.delayed_expectile_index_mode,
                self_period=self.delayed_expectile_self_period,
            ),
            actor_params=actor_params,
            actor_opt_state=copy.deepcopy(self.initial_actor_opt_state),
            actor_key=actor_key,
        )

        self.state = tree_to_device(self.state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
        self._train_step = self._build_train_step()
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
        q_tx = self.q_tx
        v_tx = self.v_tx
        actor_tx = self.actor_tx
        discount = self.discount
        tau = self.tau
        beta = self.beta
        iql_tau = self.iql_tau
        ensemble_size = self.ensemble_size
        delayed_update_period = self.delayed_update_period
        delayed_expectile_index_mode = self.delayed_expectile_index_mode
        delayed_expectile_self_period = self.delayed_expectile_self_period
        iql_deterministic = self.iql_deterministic
        use_dropout = self.actor_dropout is not None
        actor_apply_fn = self.actor_def.apply
        ensemble_indices = jnp.arange(ensemble_size, dtype=jnp.int32)

        def apply_q_ensemble(params: Any, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: q_apply({"params": p}, states, actions))(params)

        def apply_v_ensemble(params: Any, states: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: v_apply({"params": p}, states))(params)

        def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
            if use_dropout and training:
                return actor_apply_fn(
                    {"params": actor_params},
                    observations,
                    training=training,
                    rngs={"dropout": rng},
                )
            return actor_apply_fn({"params": actor_params}, observations, training=training)

        def filter_indices_for_update(total_it_: jnp.ndarray) -> jnp.ndarray:
            delayed_round = total_it_ // jnp.asarray(delayed_update_period, dtype=jnp.int32)
            return scheduled_filter_indices(
                ensemble_size=ensemble_size,
                delayed_round=delayed_round,
                mode=delayed_expectile_index_mode,
                self_period=delayed_expectile_self_period,
            )

        @jax.jit
        def train_step(state: DDIQLState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            # Old-state quantities used by multiple losses, mirroring IQL-style sequencing.
            next_v_all = apply_v_ensemble(state.v_params, next_observations)  # [N, B]
            target_q_for_backup = rewards[None, :] + (1.0 - dones[None, :]) * discount * next_v_all
            target_v_q_all = apply_q_ensemble(state.q_target_params, observations, actions)  # [N, B]
            old_v_all = apply_v_ensemble(state.v_params, observations)  # [N, B]

            # Decoupled delayed filtering weights: target member i uses delayed member j(i).
            delayed_q_all = apply_q_ensemble(state.q_delayed_params, observations, actions)  # [N, B]
            delayed_v_all = apply_v_ensemble(state.v_delayed_params, observations)  # [N, B]
            delayed_adv_all = delayed_q_all - delayed_v_all

            # filter_indices is the single source of truth for cross/self scheduling.
            effective_filter_indices = state.filter_indices
            delayed_adv_for_filter = delayed_adv_all[effective_filter_indices, :]  # [N, B]
            is_self_filter = jnp.all(effective_filter_indices == ensemble_indices)
            delayed_value_weight = jnp.abs(
                iql_tau - (delayed_adv_for_filter < 0.0).astype(jnp.float32)
            )

            # Actor advantage uses ensemble-mean Q and V, not min.
            data_q_mean = jnp.mean(target_v_q_all, axis=0)
            data_v_mean = jnp.mean(old_v_all, axis=0)
            actor_adv = data_q_mean - data_v_mean
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(actor_adv)), EXP_ADV_MAX)

            def v_loss_fn(v_params):
                v_all = apply_v_ensemble(v_params, observations)  # [N, B]
                value_residual = jax.lax.stop_gradient(target_v_q_all) - v_all
                value_loss = jnp.mean(jax.lax.stop_gradient(delayed_value_weight) * value_residual ** 2)
                return value_loss, v_all

            (value_loss, v_all), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            def q_loss_fn(q_params):
                q_all = apply_q_ensemble(q_params, observations, actions)  # [N, B]
                target = jax.lax.stop_gradient(target_q_for_backup)
                q_loss = jnp.mean((q_all - target) ** 2)
                return q_loss, q_all

            (q_loss, q_all), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
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

            should_update_delayed = (total_it % jnp.asarray(delayed_update_period, dtype=jnp.int32)) == 0

            def update_delayed(carry):
                q_target_params_, v_params_ = carry
                filter_indices_ = filter_indices_for_update(total_it)
                return q_target_params_, v_params_, filter_indices_

            def keep_delayed(carry):
                _q_target_params_, _v_params_ = carry
                return state.q_delayed_params, state.v_delayed_params, state.filter_indices

            q_delayed_params, v_delayed_params, filter_indices = jax.lax.cond(
                should_update_delayed,
                update_delayed,
                keep_delayed,
                operand=(q_target_params, v_params),
            )

            new_state = DDIQLState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_delayed_params=q_delayed_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_delayed_params=v_delayed_params,
                v_opt_state=v_opt_state,
                filter_indices=filter_indices,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                actor_key=actor_key,
            )

            log_dict = {
                "q_loss": q_loss,
                "q_mean": jnp.mean(q_all),
                "q_std": jnp.std(q_all),
                "target_q_mean": jnp.mean(target_q_for_backup),
                "value_loss": value_loss,
                "v_mean": jnp.mean(v_all),
                "v_std": jnp.std(v_all),
                "target_v_q_mean": jnp.mean(target_v_q_all),
                "actor_loss": actor_loss,
                "bc_loss_mean": jnp.mean(bc_losses),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
                "actor_adv_mean": jnp.mean(actor_adv),
                "actor_adv_min": jnp.min(actor_adv),
                "actor_adv_max": jnp.max(actor_adv),
                "exp_adv_mean": jnp.mean(exp_adv),
                "exp_adv_max": jnp.max(exp_adv),
                "delayed_adv_mean": jnp.mean(delayed_adv_for_filter),
                "delayed_adv_min": jnp.min(delayed_adv_for_filter),
                "delayed_adv_max": jnp.max(delayed_adv_for_filter),
                "delayed_weight_mean": jnp.mean(delayed_value_weight),
                "delayed_weight_min": jnp.min(delayed_value_weight),
                "delayed_weight_max": jnp.max(delayed_value_weight),
                "delayed_negative_adv_frac": jnp.mean((delayed_adv_for_filter < 0.0).astype(jnp.float32)),
                "delayed_update": should_update_delayed.astype(jnp.float32),
                "ensemble_size": jnp.asarray(ensemble_size, dtype=jnp.float32),
                # Backward-compatible dashboard key: 1 only for fixed-self mode.
                "delayed_expectile_self_index": jnp.asarray(
                    delayed_expectile_index_mode == "self", dtype=jnp.float32
                ),
                # Actual phase used by the current V update. In periodic_self this
                # toggles between 0 (cross) and 1 (self).
                "delayed_expectile_is_self_filter": is_self_filter.astype(jnp.float32),
                "filter_index_mean": jnp.mean(effective_filter_indices.astype(jnp.float32)),
                "filter_shift": jnp.where(
                    jnp.asarray(ensemble_size, dtype=jnp.int32) > 1,
                    effective_filter_indices[0],
                    jnp.asarray(0, dtype=jnp.int32),
                ).astype(jnp.float32),
                # This is the filter that will be active on the next step after a
                # delayed refresh (or the same filter if no refresh occurred).
                "next_delayed_expectile_is_self_filter": jnp.all(
                    filter_indices == ensemble_indices
                ).astype(jnp.float32),
                "next_filter_shift": jnp.where(
                    jnp.asarray(ensemble_size, dtype=jnp.int32) > 1,
                    filter_indices[0],
                    jnp.asarray(0, dtype=jnp.int32),
                ).astype(jnp.float32),
            }
            return new_state, log_dict

        return train_step

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.state, log_dict = self._train_step(self.state, batch)
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

        def apply_q_ensemble(params: Any, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: q_apply({"params": p}, states, actions))(params)

        def apply_v_ensemble(params: Any, states: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: v_apply({"params": p}, states))(params)

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
        def actor_refit_step(actor_state: ActorState, iql_state: DDIQLState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]

            # Actor refit uses frozen trained ensemble-mean Q_target and V.
            q_all = apply_q_ensemble(iql_state.q_target_params, observations, actions)
            v_all = apply_v_ensemble(iql_state.v_params, observations)
            target_q = jnp.mean(q_all, axis=0)
            v = jnp.mean(v_all, axis=0)
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
        log_interval: int = 500,
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
        log_interval = int(log_interval)
        if log_interval <= 0:
            log_interval = 1

        def _wandb_log_scalar_dict(payload: Dict[str, Any], step: int) -> None:
            if not (log_wandb and wandb.run is not None):
                return

            scalar_payload = {}
            for key, value in payload.items():
                if is_scalar_value(value):
                    scalar_payload[key] = to_python_scalar(value)

            if scalar_payload:
                wandb.log(scalar_payload, step=int(step))

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
            step_log = {
                key: float(jax.device_get(value))
                for key, value in step_log.items()
            }

            refit_log[f"{prefix}/final_loss"] = step_log["loss"]
            refit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]
            refit_log[f"{prefix}/final_adv_mean"] = step_log["adv_mean"]
            refit_log[f"{prefix}/final_exp_adv_mean"] = step_log["exp_adv_mean"]

            should_log_step = (
                log_wandb
                and wandb.run is not None
                and (fit_step % log_interval == 0 or fit_step == 1 or fit_step == steps)
            )
            if should_log_step:
                _wandb_log_scalar_dict(
                    {
                        f"{prefix}/step": fit_step,
                        f"{prefix}/loss": step_log["loss"],
                        f"{prefix}/bc_loss": step_log["bc_loss"],
                        f"{prefix}/adv_mean": step_log["adv_mean"],
                        f"{prefix}/adv_min": step_log["adv_min"],
                        f"{prefix}/adv_max": step_log["adv_max"],
                        f"{prefix}/exp_adv_mean": step_log["exp_adv_mean"],
                        f"{prefix}/exp_adv_max": step_log["exp_adv_max"],
                        f"{prefix}/target_q_mean": step_log["target_q_mean"],
                        f"{prefix}/v_mean": step_log["v_mean"],
                        f"{prefix}/policy_mean": step_log["policy_mean"],
                        f"{prefix}/policy_log_std_mean": step_log["policy_log_std_mean"],
                    },
                    step=fit_step,
                )

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
                normalized_eval_score_mean, normalized_eval_score_std = mean_std_or_nan(
                    normalized_eval_scores
                )
                success_rate, success_std = mean_std_or_nan(eval_successes)

                refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(
                    normalized_eval_score_mean
                )
                refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(
                    normalized_eval_score_std
                )
                refit_log[f"{prefix}/inner_success_rate"].append(success_rate)
                refit_log[f"{prefix}/inner_success_std"].append(success_std)

                refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                refit_log[f"{prefix}/final_score_std"] = eval_score_std
                refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = (
                    normalized_eval_score_mean
                )
                refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = (
                    normalized_eval_score_std
                )
                refit_log[f"{prefix}/final_success_rate"] = success_rate
                refit_log[f"{prefix}/final_success_std"] = success_std

                eval_metric_mean = (
                    success_rate
                    if np.isfinite(success_rate)
                    else normalized_eval_score_mean
                )
                is_best = (
                    np.isfinite(eval_metric_mean)
                    and eval_metric_mean > best_eval_metric_mean
                )

                if is_best:
                    best_eval_metric_mean = eval_metric_mean
                    refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                    refit_log[f"{prefix}/best_score_std"] = eval_score_std
                    refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = (
                        normalized_eval_score_mean
                    )
                    refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = (
                        normalized_eval_score_std
                    )
                    refit_log[f"{prefix}/best_success_rate"] = success_rate
                    refit_log[f"{prefix}/best_success_std"] = success_std

                _wandb_log_scalar_dict(
                    {
                        f"{prefix}/eval_score_mean": eval_score_mean,
                        f"{prefix}/eval_score_std": eval_score_std,
                        f"{prefix}/eval_d4rl_normalized_score_mean": normalized_eval_score_mean,
                        f"{prefix}/eval_d4rl_normalized_score_std": normalized_eval_score_std,
                        f"{prefix}/eval_success_rate": success_rate,
                        f"{prefix}/eval_success_std": success_std,
                        f"{prefix}/best_score_mean": refit_log[f"{prefix}/best_score_mean"],
                        f"{prefix}/best_score_std": refit_log[f"{prefix}/best_score_std"],
                        f"{prefix}/best_d4rl_normalized_score_mean": refit_log[
                            f"{prefix}/best_d4rl_normalized_score_mean"
                        ],
                        f"{prefix}/best_d4rl_normalized_score_std": refit_log[
                            f"{prefix}/best_d4rl_normalized_score_std"
                        ],
                        f"{prefix}/best_success_rate": refit_log[
                            f"{prefix}/best_success_rate"
                        ],
                        f"{prefix}/best_success_std": refit_log[
                            f"{prefix}/best_success_std"
                        ],
                    },
                    step=fit_step,
                )

                save_refit_snapshot(
                    current_actor_state=actor_state,
                    current_refit_log=refit_log,
                    fit_step=fit_step,
                    is_best=is_best,
                )

                print(
                    f"[{prefix}:decoupled_delayed_iql_awbc] step {fit_step}/{steps}: "
                    f"loss={step_log['loss']:.4f}, "
                    f"bc={step_log['bc_loss']:.4f}, "
                    f"adv={step_log['adv_mean']:.4f}, "
                    f"exp_adv={step_log['exp_adv_mean']:.4f}, "
                    f"eval_mean={eval_score_mean:.3f}, "
                    f"eval_std={eval_score_std:.3f}, "
                    f"d4rl_normalized_mean={normalized_eval_score_mean:.3f}, "
                    f"d4rl_normalized_std={normalized_eval_score_std:.3f}, "
                    f"success_rate={success_rate:.3f}, "
                    f"is_best={is_best}"
                )

        return actor_state, refit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decoupled_delayed_iql_state": serialization.to_state_dict(self.state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
            "iql_deterministic": self.iql_deterministic,
            "ensemble_size": self.ensemble_size,
            "delayed_update_period": self.delayed_update_period,
            "delayed_expectile_index_mode": self.delayed_expectile_index_mode,
            "delayed_expectile_self_period": self.delayed_expectile_self_period,
            "delayed_expectile_self_index": self.delayed_expectile_self_index,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state = serialization.from_state_dict(
            self.state,
            state_dict["decoupled_delayed_iql_state"],
        )
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
        # Reconstruct filter_indices deterministically from total_it and the
        # current schedule. This supports old checkpoints and periodic resumes
        # without adding an extra phase counter to DDIQLState.
        delayed_round = self.state.total_it // jnp.asarray(
            self.delayed_update_period, dtype=jnp.int32
        )
        self.state = self.state.replace(
            filter_indices=scheduled_filter_indices(
                ensemble_size=self.ensemble_size,
                delayed_round=delayed_round,
                mode=self.delayed_expectile_index_mode,
                self_period=self.delayed_expectile_self_period,
            )
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
    """Return (run_dir, checkpoint_path) for a saved DDIQL-JAX checkpoint.

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
            checkpoint_type="decoupled_delayed_iql_jax_training_progress",
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
    effective_index_mode = resolve_delayed_expectile_index_mode(
        config.delayed_expectile_index_mode,
        delayed_expectile_self_index=config.delayed_expectile_self_index,
    )
    print(
        f"ensemble_size={config.ensemble_size}, "
        f"delayed_update_period={config.delayed_update_period}, "
        f"delayed_expectile_index_mode={config.delayed_expectile_index_mode} "
        f"(effective={effective_index_mode}), "
        f"delayed_expectile_self_period={config.delayed_expectile_self_period}, "
        f"legacy_delayed_expectile_self_index={config.delayed_expectile_self_index}"
    )
    print("---------------------------------------")

    trainer = DDIQLJAX(
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
        ensemble_size=config.ensemble_size,
        delayed_update_period=config.delayed_update_period,
        delayed_expectile_index_mode=config.delayed_expectile_index_mode,
        delayed_expectile_self_period=config.delayed_expectile_self_period,
        delayed_expectile_self_index=config.delayed_expectile_self_index,
        iql_deterministic=config.iql_deterministic,
        actor_dropout=config.actor_dropout,
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
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
            log_interval=config.log_every,
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
            log_dict = trainer.train(batch)
    
            if config.log_wandb and (t + 1) % config.log_every == 0:
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







