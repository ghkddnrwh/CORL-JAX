# ReBRAC JAX implementation with IQL_JAX-style experiment plumbing.
#
# Algorithmic losses, target updates, network initialization, D4RL dataset
# construction, and evaluation action selection are kept from the provided
# ReBRAC JAX script. Experiment management follows the accompanying IQL_JAX
# style: config/hyperparameter merging, deterministic run names, wandb logging,
# eval-log persistence, checkpoint saving/loading, and explicit run modes.
#
# Current plumbing:
#   - Explicit mode switch: mode="train" or mode="actor_refit".
#   - actor_refit mode loads a saved ReBRAC checkpoint, freezes the learned critic,
#     and optimizes only a freshly initialized actor with the ReBRAC actor loss.
#   - Refit mode uses shared schedule fields:
#       max_timesteps -> actor-only refit steps
#       batch_size    -> actor-only refit batch size
#       eval_freq     -> actor-only refit evaluation interval
#
# Source algorithm:
#   https://github.com/tinkoff-ai/ReBRAC
#   https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility.

import math
import pickle
import random
import sys
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chex
import flax.linen as nn
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

import optax
import pyrallis
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

        def mark_preempting(self, *args, **kwargs):
            return None

    wandb = _UnavailableWandb()

import yaml
from flax import serialization, struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

# Match the IQL implementation: do not import d4rl at module import time.
# D4RL is imported lazily only when a D4RL environment is requested, so
# OGBench runs do not trigger mujoco_py/Cython compilation.
d4rl = None

try:
    import ogbench
except ImportError:
    ogbench = None

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "ReBRAC"
ALGORITHM_FULL_NAME = "Revisiting Behavioral Regularization in Actor-Critic"


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class TrainConfig:
    # Experiment.
    device: str = "gpu"  # one of: cpu, gpu, tpu. JAX falls back if unavailable.
    env: str = "halfcheetah-medium-v2"
    seed: int = 0
    eval_seed: int = 42

    # IQL_JAX-style schedule fields.
    max_timesteps: int = int(1e6)
    eval_freq: int = int(5e3)
    n_episodes: int = 10

    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"  # one of: train, actor_refit. refit is accepted as an alias.
    hyperparams_path: Optional[str] = "hyperparams/rebrac_jax.yml"
    use_hyperparams: bool = True

    # Backward-compatible alias for the original ReBRAC script.
    # Prefer --env in new runs. If provided, dataset_name overrides env.
    dataset_name: Optional[str] = None

    # Dataset.
    batch_size: int = 1024
    normalize_reward: bool = False
    normalize_states: bool = False

    # ReBRAC / TD3+BC-style actor-critic parameters.
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    num_critics: int = 2
    discount: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True

    # Standalone actor refit output directory.
    # actor_refit reuses the shared training schedule fields above:
    #   max_timesteps -> actor-only refit steps
    #   batch_size    -> actor-only refit batch size
    #   eval_freq     -> actor-only refit evaluation interval
    actor_refit_dir_name: str = "actor_refit"

    # Logging / saving.
    project: str = "ORL-SMOOTH"
    group: str = "ReBRAC-JAX"
    name: str = "ReBRAC-JAX"
    log_wandb: bool = True
    log_every: int = 500
    save_best_model: bool = True

    def __post_init__(self):
        normalize_config_aliases(self)
        refresh_algorithm_names(self)
        validate_config(self)


def normalize_config_aliases(config: TrainConfig) -> None:
    if config.dataset_name is not None:
        config.env = config.dataset_name
    if config.mode == "refit":
        config.mode = "actor_refit"


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.mode in ("train", "actor_refit"), "mode must be train or actor_refit"
    assert config.batch_size > 0
    assert config.eval_freq > 0
    assert config.n_episodes > 0
    assert config.max_timesteps >= 0
    assert config.log_every > 0
    assert config.num_critics > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.actor_bc_coef >= 0.0
    assert config.critic_bc_coef >= 0.0
    assert config.policy_noise >= 0.0
    assert config.noise_clip >= 0.0
    assert config.policy_freq > 0
    assert config.actor_refit_dir_name != ""
    if config.mode == "actor_refit":
        assert config.load_model != "", "mode='actor_refit' requires --load_model"


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
    # YAML may load scientific notation such as 1e6 as float.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    """Load env-specific hyperparameters and merge them into config.

    Priority:
        dataclass defaults < hyperparams YAML < explicit CLI flags

    Hyperparameter YAML keys should match TrainConfig field names. The old
    ReBRAC key `dataset_name` is accepted as a backward-compatible env alias,
    and `n_timesteps` is accepted as an alias for `max_timesteps`.
    """
    normalize_config_aliases(config)
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
        print(
            f"No hyperparameters found for env '{config.env}' in {hparam_path}. "
            "Using dataclass/CLI values."
        )
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    env_hyperparams = all_hyperparams[config.env] or {}
    cli_overrides = _cli_overridden_fields()
    aliases = {"dataset_name": "env", "n_timesteps": "max_timesteps"}
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

    normalize_config_aliases(config)
    refresh_algorithm_names(config)
    validate_config(config)

    if applied:
        print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
    if skipped_cli:
        print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
    if skipped_unknown:
        print(f"Ignored unknown hyperparameter keys for ReBRAC: {', '.join(skipped_unknown)}")
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


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
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
    if run is not None and hasattr(run, "log_code"):
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


def save_logs_npz(logs: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    if len(logs) == 0:
        return
    keys = logs[0].keys()
    data_to_save: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [log.get(key, np.nan) for log in logs]
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


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


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
    """Extract a scalar success signal from Gym/OGBench info dicts.

    OGBench tasks are commonly evaluated by success rate rather than D4RL-style
    normalized score. Returning NaN when the key is absent lets D4RL evaluation
    continue to use normalized scores without special casing every environment.
    """
    if not isinstance(info, dict) or "success" not in info:
        return np.nan
    success = np.asarray(info["success"])
    if success.size == 0:
        return np.nan
    return float(success.reshape(-1)[0])


def pytorch_init(fan_in: float) -> Callable:
    """Default init for PyTorch Linear layer weights and biases."""
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )

    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )

    return _init


def identity(x: Any) -> Any:
    return x


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC/ReBRAC implementation.
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
            nn.tanh,
        ]
        return nn.Sequential(layers)(state)


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC/ReBRAC implementation.
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))]
        state_action = jnp.hstack([state, action])
        return nn.Sequential(layers)(state_action).squeeze(-1)


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 2
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        return ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(state, action)


def qlearning_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    n_transitions = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, next_action_, reward_, done_ = [], [], [], [], [], []
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(n_transitions - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "next_actions": np.array(next_action_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def compute_mean_std(states: Union[np.ndarray, jax.Array], eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(states).mean(0)
    std = np.asarray(states).std(0) + eps
    return mean, std


def normalize_states(
    states: Union[np.ndarray, jax.Array],
    mean: Union[np.ndarray, float],
    std: Union[np.ndarray, float],
) -> np.ndarray:
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


def add_next_actions_if_missing(dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a ReBRAC-compatible dataset dict.

    ReBRAC's critic regularization needs behavior actions at next states. D4RL's
    env.get_dataset path above already builds them by shifting actions by one.
    OGBench-style datasets normally provide observations/actions/next_observations
    but not next_actions, so we construct the same shifted-action proxy here.
    """
    data = dict(dataset)
    n_transitions = int(np.asarray(data["rewards"]).shape[0])

    if "next_actions" not in data:
        actions = np.asarray(data["actions"], dtype=np.float32)
        next_actions = np.empty_like(actions)
        if n_transitions > 1:
            next_actions[:-1] = actions[1:]
        next_actions[-1] = actions[-1]
        data["next_actions"] = next_actions

    if "terminals" not in data:
        if "masks" in data:
            data["terminals"] = 1.0 - np.asarray(data["masks"], dtype=np.float32)
        else:
            data["terminals"] = np.zeros(n_transitions, dtype=np.float32)

    return data


def load_env_and_dataset(env_name: str) -> Tuple[gym.Env, Dict[str, np.ndarray], str]:
    if is_ogbench_env(env_name):
        if ogbench is None:
            raise ImportError(
                "OGBench environment requested, but the `ogbench` package is not installed."
            )
        env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        return env, add_next_actions_if_missing(train_dataset), "ogbench"

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
    return env, qlearning_dataset(env), "d4rl"


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
    def __init__(self, device: Any):
        self.data: Optional[TensorBatch] = None
        self.mean: Union[np.ndarray, float] = 0.0
        self.std: Union[np.ndarray, float] = 1.0
        self.device = device

    def create_from_dataset(
        self,
        env_name: str,
        dataset: Dict[str, np.ndarray],
        normalize_reward: bool = False,
        is_normalize: bool = False,
        dataset_backend: str = "d4rl",
    ):
        dataset = add_next_actions_if_missing(dataset)
        rewards = np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1)
        terminals = np.asarray(dataset["terminals"], dtype=np.float32).reshape(-1)
        buffer_np = {
            "states": np.asarray(dataset["observations"], dtype=np.float32),
            "actions": np.asarray(dataset["actions"], dtype=np.float32),
            "rewards": rewards,
            "next_states": np.asarray(dataset["next_observations"], dtype=np.float32),
            "next_actions": np.asarray(dataset["next_actions"], dtype=np.float32),
            "dones": terminals,
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer_np["states"], eps=1e-3)
            buffer_np["states"] = normalize_states(buffer_np["states"], self.mean, self.std)
            buffer_np["next_states"] = normalize_states(buffer_np["next_states"], self.mean, self.std)
        if normalize_reward:
            if dataset_backend == "d4rl":
                buffer_np["rewards"] = ReplayBuffer.normalize_reward(env_name, buffer_np["rewards"])
            else:
                print("Skipping ReBRAC/D4RL reward normalization for non-D4RL dataset.")

        self.data = tree_to_device({k: jnp.asarray(v, dtype=jnp.float32) for k, v in buffer_np.items()}, self.device)
        print(f"Dataset size: {self.size}")

    def create_from_d4rl(
        self,
        env_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        _, dataset, dataset_backend = load_env_and_dataset(env_name)
        self.create_from_dataset(
            env_name=env_name,
            dataset=dataset,
            normalize_reward=normalize_reward,
            is_normalize=is_normalize,
            dataset_backend=dataset_backend,
        )

    @property
    def size(self) -> int:
        if self.data is None:
            return 0
        return int(self.data["states"].shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.data["states"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.data["actions"].shape[-1])

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> TensorBatch:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        return jax.tree_util.tree_map(lambda arr: arr[indices], self.data)

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: np.ndarray) -> np.ndarray:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # Like in LAPO / original ReBRAC code.
        raise NotImplementedError("Reward normalization is implemented only for AntMaze.")


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


@struct.dataclass
class ActorRefitState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


def update_actor(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: TensorBatch,
    beta: float,
    tau: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
    key, random_action_key = jax.random.split(key, 2)

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        actions = actor.apply_fn(params, batch["states"])

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1.0
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()
        random_actions = jax.random.uniform(
            random_action_key,
            shape=batch["actions"].shape,
            minval=-1.0,
            maxval=1.0,
        )
        log_dict = {
            "actor_loss": loss,
            "bc_mse_policy": bc_penalty.mean(),
            "bc_mse_random": ((random_actions - batch["actions"]) ** 2).sum(-1).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean(),
        }
        return loss, log_dict

    grads, log_dict = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(
        target_params=optax.incremental_update(actor.params, actor.target_params, tau)
    )
    new_critic = critic.replace(
        target_params=optax.incremental_update(critic.params, critic.target_params, tau)
    )
    return key, new_actor, new_critic, log_dict


def update_critic(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: TensorBatch,
    discount: float,
    beta: float,
    policy_noise: float,
    noise_clip: float,
) -> Tuple[jax.random.PRNGKey, CriticTrainState, Dict[str, jax.Array]]:
    key, actions_key = jax.random.split(key)

    next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
    noise = jnp.clip(
        jax.random.normal(actions_key, next_actions.shape) * policy_noise,
        -noise_clip,
        noise_clip,
    )
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    next_q = next_q - beta * bc_penalty

    target_q = batch["rewards"] + (1.0 - batch["dones"]) * discount * next_q

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        q_min = q.min(0).mean()
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q_min

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    log_dict = {
        "critic_loss": loss,
        "q_min": q_min,
    }
    return key, new_critic, log_dict


def update_td3(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: TensorBatch,
    discount: float,
    actor_bc_coef: float,
    critic_bc_coef: float,
    tau: float,
    policy_noise: float,
    noise_clip: float,
    normalize_q: bool,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
    key, new_critic, critic_log = update_critic(
        key,
        actor,
        critic,
        batch,
        discount,
        critic_bc_coef,
        policy_noise,
        noise_clip,
    )
    key, new_actor, new_critic, actor_log = update_actor(
        key,
        actor,
        new_critic,
        batch,
        actor_bc_coef,
        tau,
        normalize_q,
    )
    return key, new_actor, new_critic, {**critic_log, **actor_log}


def update_td3_no_targets(
    key: jax.random.PRNGKey,
    actor: ActorTrainState,
    critic: CriticTrainState,
    batch: TensorBatch,
    discount: float,
    critic_bc_coef: float,
    policy_noise: float,
    noise_clip: float,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
    key, new_critic, critic_log = update_critic(
        key,
        actor,
        critic,
        batch,
        discount,
        critic_bc_coef,
        policy_noise,
        noise_clip,
    )
    return key, actor, new_critic, critic_log


def update_actor_only(
    actor_apply_fn: Callable,
    actor_tx: optax.GradientTransformation,
    key: jax.random.PRNGKey,
    actor_state: ActorRefitState,
    critic: CriticTrainState,
    batch: TensorBatch,
    beta: float,
    normalize_q: bool,
) -> Tuple[ActorRefitState, Dict[str, jax.Array]]:
    key, random_action_key = jax.random.split(key, 2)

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        actions = actor_apply_fn(params, batch["states"])

        bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        lmbda = 1.0
        if normalize_q:
            lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())

        loss = (beta * bc_penalty - lmbda * q_values).mean()
        random_actions = jax.random.uniform(
            random_action_key,
            shape=batch["actions"].shape,
            minval=-1.0,
            maxval=1.0,
        )
        log_dict = {
            "loss": loss,
            "bc_mse_policy": bc_penalty.mean(),
            "bc_mse_random": ((random_actions - batch["actions"]) ** 2).sum(-1).mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean(),
            "q_mean": q_values.mean(),
            "q_abs_mean": jnp.abs(q_values).mean(),
            "lambda": lmbda,
        }
        return loss, log_dict

    grads, log_dict = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
    updates, actor_opt_state = actor_tx.update(grads, actor_state.opt_state, actor_state.params)
    actor_params = optax.apply_updates(actor_state.params, updates)
    new_actor_state = ActorRefitState(params=actor_params, opt_state=actor_opt_state, key=key)
    return new_actor_state, log_dict


class ReBRACJAX:
    """ReBRAC in JAX/Flax with IQL_JAX-style training plumbing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        hidden_dim: int = 256,
        actor_n_hiddens: int = 3,
        critic_n_hiddens: int = 3,
        num_critics: int = 2,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_bc_coef: float = 1.0,
        critic_bc_coef: float = 1.0,
        actor_ln: bool = False,
        critic_ln: bool = True,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        normalize_q: bool = True,
        seed: int = 0,
        device: Any = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.normalize_q = normalize_q
        self.device = device if device is not None else jax.devices()[0]

        key = jax.random.PRNGKey(seed)
        key, actor_key, critic_key, actor_refit_key = jax.random.split(key, 4)
        init_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        init_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        self.actor_module = DetActor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            layernorm=actor_ln,
            n_hiddens=actor_n_hiddens,
        )
        actor_params = self.actor_module.init(actor_key, init_state)
        self.actor_tx = optax.adam(learning_rate=actor_learning_rate)
        self.actor = ActorTrainState.create(
            apply_fn=self.actor_module.apply,
            params=actor_params,
            target_params=deepcopy(actor_params),
            tx=self.actor_tx,
        )
        self.initial_actor_params = deepcopy(actor_params)
        self.initial_actor_opt_state = deepcopy(self.actor.opt_state)
        self.initial_actor_key = actor_refit_key

        self.critic_module = EnsembleCritic(
            hidden_dim=hidden_dim,
            num_critics=num_critics,
            layernorm=critic_ln,
            n_hiddens=critic_n_hiddens,
        )
        critic_params = self.critic_module.init(critic_key, init_state, init_action)
        self.critic = CriticTrainState.create(
            apply_fn=self.critic_module.apply,
            params=critic_params,
            target_params=deepcopy(critic_params),
            tx=optax.adam(learning_rate=critic_learning_rate),
        )

        self.key = key
        self.total_it = 0
        self.actor = tree_to_device(self.actor, self.device)
        self.critic = tree_to_device(self.critic, self.device)
        self.key = tree_to_device(self.key, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)

        self._full_update_step = self._build_full_update_step()
        self._critic_update_step = self._build_critic_update_step()
        self._actor_refit_step = self._build_actor_refit_step()

    def _build_full_update_step(self):
        discount = self.discount
        actor_bc_coef = self.actor_bc_coef
        critic_bc_coef = self.critic_bc_coef
        tau = self.tau
        policy_noise = self.policy_noise
        noise_clip = self.noise_clip
        normalize_q = self.normalize_q

        @jax.jit
        def full_update_step(key, actor, critic, batch):
            return update_td3(
                key=key,
                actor=actor,
                critic=critic,
                batch=batch,
                discount=discount,
                actor_bc_coef=actor_bc_coef,
                critic_bc_coef=critic_bc_coef,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                normalize_q=normalize_q,
            )

        return full_update_step

    def _build_critic_update_step(self):
        discount = self.discount
        critic_bc_coef = self.critic_bc_coef
        policy_noise = self.policy_noise
        noise_clip = self.noise_clip

        @jax.jit
        def critic_update_step(key, actor, critic, batch):
            return update_td3_no_targets(
                key=key,
                actor=actor,
                critic=critic,
                batch=batch,
                discount=discount,
                critic_bc_coef=critic_bc_coef,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
            )

        return critic_update_step

    def train(self, batch: TensorBatch, update_actor_now: bool) -> Dict[str, float]:
        if update_actor_now:
            self.key, self.actor, self.critic, log_dict = self._full_update_step(
                self.key,
                self.actor,
                self.critic,
                batch,
            )
        else:
            self.key, self.actor, self.critic, log_dict = self._critic_update_step(
                self.key,
                self.actor,
                self.critic,
                batch,
            )
        self.total_it += 1
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
        action = self.actor.apply_fn(actor_params, state_jnp)
        return np.asarray(jax.device_get(action))[0]

    def eval_actor(
        self,
        env: gym.Env,
        actor_params: Any,
        n_episodes: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        returns = []
        successes = []
        for episode_idx in trange(n_episodes, desc="Eval", leave=False):
            obs = reset_env(env, seed=seed if episode_idx == 0 else None)
            done = False
            total_reward = 0.0
            episode_success = np.nan
            while not done:
                action = self.actor_act(actor_params, obs)
                obs, reward, done, info = step_env(env, action)
                total_reward += reward
                step_success = extract_success(info)
                if np.isfinite(step_success):
                    episode_success = step_success
            returns.append(total_reward)
            successes.append(episode_success)
        return (
            np.asarray(returns, dtype=np.float32),
            np.asarray(successes, dtype=np.float32),
        )

    def _build_actor_refit_step(self):
        actor_apply_fn = self.actor.apply_fn
        actor_tx = self.actor_tx
        actor_bc_coef = self.actor_bc_coef
        normalize_q = self.normalize_q

        @jax.jit
        def actor_refit_step(actor_state: ActorRefitState, critic: CriticTrainState, batch: TensorBatch):
            return update_actor_only(
                actor_apply_fn=actor_apply_fn,
                actor_tx=actor_tx,
                key=actor_state.key,
                actor_state=actor_state,
                critic=critic,
                batch=batch,
                beta=actor_bc_coef,
                normalize_q=normalize_q,
            )

        return actor_refit_step

    def make_initial_actor_state(self) -> ActorRefitState:
        return tree_to_device(
            ActorRefitState(
                params=deepcopy(self.initial_actor_params),
                opt_state=deepcopy(self.initial_actor_opt_state),
                key=deepcopy(self.initial_actor_key),
            ),
            self.device,
        )

    def fit_actor(
        self,
        replay_buffer: ReplayBuffer,
        actor_state: ActorRefitState,
        steps: int,
        batch_size: int,
        eval_env: Optional[gym.Env] = None,
        eval_episodes: int = 0,
        eval_seed: int = 0,
        eval_interval: int = 0,
        prefix: str = "actor_refit",
        save_dir: Optional[Union[str, Path]] = None,
        log_wandb: bool = False,
        log_every: int = 500,
        log_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ActorRefitState, Dict[str, Any]]:
        refit_log: Dict[str, Any] = {
            f"{prefix}/final_loss": np.nan,
            f"{prefix}/final_bc_mse_policy": np.nan,
            f"{prefix}/final_action_mse": np.nan,
            f"{prefix}/final_q_mean": np.nan,
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
            current_actor_state: ActorRefitState,
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

        for fit_step in trange(1, int(steps) + 1, desc="Actor refit"):
            next_key, batch_key = jax.random.split(actor_state.key, 2)
            actor_state = actor_state.replace(key=next_key)
            batch = replay_buffer.sample_batch(batch_key, batch_size=batch_size)
            actor_state, step_log = self._actor_refit_step(actor_state, self.critic, batch)
            step_log = {key: float(jax.device_get(value)) for key, value in step_log.items()}

            refit_log[f"{prefix}/final_loss"] = step_log["loss"]
            refit_log[f"{prefix}/final_bc_mse_policy"] = step_log["bc_mse_policy"]
            refit_log[f"{prefix}/final_action_mse"] = step_log["action_mse"]
            refit_log[f"{prefix}/final_q_mean"] = step_log["q_mean"]

            if log_wandb and fit_step % max(1, int(log_every)) == 0:
                wandb.log({f"{prefix}/train/{key}": value for key, value in step_log.items()}, step=fit_step)

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
                normalized_score_mean, normalized_score_std = mean_std_or_nan(normalized_eval_scores)
                success_rate, success_std = mean_std_or_nan(eval_successes)

                refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_score_mean)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_score_std)
                refit_log[f"{prefix}/inner_success_rate"].append(success_rate)
                refit_log[f"{prefix}/inner_success_std"].append(success_std)
                refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                refit_log[f"{prefix}/final_score_std"] = eval_score_std
                refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_score_mean
                refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_score_std
                refit_log[f"{prefix}/final_success_rate"] = success_rate
                refit_log[f"{prefix}/final_success_std"] = success_std

                # Prefer OGBench success rate when available; otherwise fall back to
                # D4RL normalized score. This keeps best-model selection meaningful
                # across both families of environments.
                eval_metric_mean = success_rate if np.isfinite(success_rate) else normalized_score_mean
                is_best = np.isfinite(eval_metric_mean) and eval_metric_mean > best_eval_metric_mean
                if is_best:
                    best_eval_metric_mean = eval_metric_mean
                    refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                    refit_log[f"{prefix}/best_score_std"] = eval_score_std
                    refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_score_mean
                    refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_score_std
                    refit_log[f"{prefix}/best_success_rate"] = success_rate
                    refit_log[f"{prefix}/best_success_std"] = success_std

                save_refit_snapshot(
                    current_actor_state=actor_state,
                    current_refit_log=refit_log,
                    fit_step=fit_step,
                    is_best=is_best,
                )

                print(
                    f"[{prefix}:rebrac_actor] step {fit_step}/{steps}: "
                    f"loss={step_log['loss']:.4f}, "
                    f"bc={step_log['bc_mse_policy']:.4f}, "
                    f"q={step_log['q_mean']:.4f}, "
                    f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                    f"d4rl_normalized_mean={normalized_score_mean:.3f}, "
                    f"d4rl_normalized_std={normalized_score_std:.3f}, "
                    f"success_rate={success_rate:.3f}, "
                    f"success_std={success_std:.3f}"
                )

        return actor_state, refit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": serialization.to_state_dict(self.actor),
            "critic": serialization.to_state_dict(self.critic),
            "key": serialization.to_state_dict(self.key),
            "total_it": self.total_it,
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor = serialization.from_state_dict(self.actor, state_dict["actor"])
        self.critic = serialization.from_state_dict(self.critic, state_dict["critic"])
        self.key = serialization.from_state_dict(self.key, state_dict["key"])
        self.total_it = int(state_dict.get("total_it", 0))
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
        self.actor = tree_to_device(self.actor, self.device)
        self.critic = tree_to_device(self.critic, self.device)
        self.key = tree_to_device(self.key, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)


def resolve_checkpoint_path(
    load_model: Union[str, Path],
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Return (run_dir, checkpoint_path) for a saved ReBRAC-JAX checkpoint.

    Supported load_model formats:

    1. Direct checkpoint file:
       path/to/checkpoint.pkl or path/to/best_checkpoint.pkl

    2. Direct run directory:
       path/to/run_dir/
       where path/to/run_dir/checkpoint.pkl exists

    3. Parent directory that contains env/seed subdirectory:
       path/to/base_dir/
       where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists
    """
    load_path = Path(load_model)

    if load_path.is_file():
        if load_path.name not in ("checkpoint.pkl", "best_checkpoint.pkl"):
            raise FileNotFoundError(
                f"load_model points to a file, but it is not checkpoint.pkl/best_checkpoint.pkl: {load_path}"
            )
        return load_path.parent, load_path

    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")

    # Prefer the normal checkpoint when a concrete run directory is provided.
    direct_checkpoint = (load_path / "checkpoint.pkl").resolve()
    if direct_checkpoint.exists():
        return direct_checkpoint.parent, direct_checkpoint
    direct_best_checkpoint = (load_path / "best_checkpoint.pkl").resolve()
    if direct_best_checkpoint.exists():
        return direct_best_checkpoint.parent, direct_best_checkpoint

    if run_name is not None and seed is not None:
        for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
            exact = (load_path / run_name / str(seed) / filename).resolve()
            if exact.exists():
                return exact.parent, exact

    candidates: List[Path] = []
    if run_name is not None:
        run_name_dir = load_path / run_name
        if run_name_dir.exists():
            candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pkl")))
            candidates.extend(sorted(run_name_dir.glob("*/best_checkpoint.pkl")))

    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/best_checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/best_checkpoint.pkl")))

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
        tried = "\n".join(str(path) for path in candidates[:20])
        raise FileNotFoundError(
            f"checkpoint file not found under: {load_path}\n"
            f"Tried candidates:\n{tried}"
        )

    if len(existing_candidates) > 1:
        found = "\n".join(str(path) for path in existing_candidates)
        raise FileNotFoundError(
            f"Multiple checkpoint files found under {load_path}.\n"
            f"Please provide a more specific --load_model path.\n"
            f"Found:\n{found}"
        )

    checkpoint_path = existing_candidates[0]
    return checkpoint_path.parent, checkpoint_path


def load_run_config_for_actor_refit(
    current_config: TrainConfig,
    loaded_run_dir: Union[str, Path],
) -> TrainConfig:
    """Load saved config.yaml from the checkpoint run dir for actor_refit mode.

    This mirrors the IQL implementation: refit reconstructs the original
    training env/model/preprocessing settings, while explicit CLI flags are
    allowed to override saved values. This also prevents the default ReBRAC
    D4RL env from being loaded accidentally when refitting an OGBench run.
    """
    loaded_run_dir = Path(loaded_run_dir)
    saved_config_path = loaded_run_dir / "config.yaml"

    if not saved_config_path.exists():
        raise FileNotFoundError(
            f"mode='actor_refit' expects saved run config at: {saved_config_path}"
        )

    with open(saved_config_path, "r") as f:
        saved_raw = yaml.safe_load(f) or {}

    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    saved_kwargs = {
        key: _coerce_hparam_value(value)
        for key, value in saved_raw.items()
        if key in config_fields
    }

    loaded_config = TrainConfig(**saved_kwargs)

    cli_overrides = _cli_overridden_fields()
    current_config_dict = asdict(current_config)
    applied_cli_overrides = []
    for key in sorted(cli_overrides):
        if key not in config_fields:
            continue
        setattr(loaded_config, key, current_config_dict[key])
        applied_cli_overrides.append(key)

    loaded_config.mode = "actor_refit"
    loaded_config.load_model = current_config.load_model
    loaded_config.checkpoints_path = None

    normalize_config_aliases(loaded_config)
    refresh_algorithm_names(loaded_config)
    validate_config(loaded_config)

    print(f"Loaded saved run config for actor_refit from: {saved_config_path}")
    if applied_cli_overrides:
        print(
            "Applied explicit CLI overrides on top of saved config: "
            + ", ".join(applied_cli_overrides)
        )

    return loaded_config


def make_checkpoint_payload(
    trainer: ReBRACJAX,
    config: TrainConfig,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
) -> Dict[str, Any]:
    return {
        "trainer": trainer.state_dict(),
        "config": asdict(config),
        "state_mean": state_mean,
        "state_std": state_std,
    }


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    trainer: ReBRACJAX,
    config: TrainConfig,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
    log_wandb: bool,
) -> None:
    save_pickle(
        checkpoint_path,
        make_checkpoint_payload(
            trainer=trainer,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
        ),
    )
    # if log_wandb and wandb.run is not None:
    #     wandb.save(str(checkpoint_path), policy="now")


@pyrallis.wrap()
def train(config: TrainConfig):
    normalize_config_aliases(config)
    actor_refit_only = config.mode == "actor_refit"

    loaded_run_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    if actor_refit_only:
        if config.load_model == "":
            raise ValueError("actor_refit mode requires --load_model")

        loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )
        config = load_run_config_for_actor_refit(
            current_config=config,
            loaded_run_dir=loaded_run_dir,
        )
    else:
        config = apply_env_hyperparams(config)
        config = finalize_checkpoint_path(config)

    jax_device = select_jax_device(config.device)
    raw_env, dataset, dataset_backend = load_env_and_dataset(config.env)

    if len(raw_env.observation_space.shape) != 1 or len(raw_env.action_space.shape) != 1:
        raise ValueError(
            f"{ALGORITHM_NAME}-JAX currently supports vector observations/actions only; "
            f"got observation_space={raw_env.observation_space}, action_space={raw_env.action_space}."
        )

    replay_buffer = ReplayBuffer(device=jax_device)
    replay_buffer.create_from_dataset(
        env_name=config.env,
        dataset=dataset,
        normalize_reward=config.normalize_reward,
        is_normalize=config.normalize_states,
        dataset_backend=dataset_backend,
    )

    state_mean, state_std = replay_buffer.mean, replay_buffer.std
    eval_env = wrap_env(raw_env, state_mean=state_mean, state_std=state_std)
    set_seed(config.seed, eval_env)

    if config.checkpoints_path is not None and not actor_refit_only:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        config_path = os.path.join(config.checkpoints_path, "config.yaml")
        if os.path.exists(config_path):
            print(f"Error: The file '{config_path}' already exists.")
            exit(1)
        with open(config_path, "w") as f:
            pyrallis.dump(config, f)

    print("---------------------------------------")
    run_mode_name = "Actor refit" if actor_refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    trainer = ReBRACJAX(
        state_dim=replay_buffer.state_dim,
        action_dim=replay_buffer.action_dim,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        hidden_dim=config.hidden_dim,
        actor_n_hiddens=config.actor_n_hiddens,
        critic_n_hiddens=config.critic_n_hiddens,
        num_critics=config.num_critics,
        discount=config.discount,
        tau=config.tau,
        actor_bc_coef=config.actor_bc_coef,
        critic_bc_coef=config.critic_bc_coef,
        actor_ln=config.actor_ln,
        critic_ln=config.critic_ln,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        normalize_q=config.normalize_q,
        seed=config.seed,
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
        trainer.load_state_dict(checkpoint["trainer"] if "trainer" in checkpoint else checkpoint)
        if isinstance(checkpoint, dict):
            state_mean = checkpoint.get("state_mean", state_mean)
            state_std = checkpoint.get("state_std", state_std)
            eval_env = wrap_env(raw_env, state_mean=state_mean, state_std=state_std)

    if config.log_wandb:
        wandb_init(asdict(config))

    if actor_refit_only:
        if loaded_run_dir is None:
            raise ValueError("actor_refit mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
        print("Critic is frozen; only actor is optimized with the ReBRAC actor loss.")
        print(
            "Refit schedule uses shared fields: "
            f"max_timesteps={config.max_timesteps}, "
            f"batch_size={config.batch_size}, "
            f"eval_freq={config.eval_freq}"
        )
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        actor_state = trainer.make_initial_actor_state()
        loaded_checkpoint_for_log = str(loaded_run_dir / checkpoint_path.name)

        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=actor_state,
            steps=int(config.max_timesteps),
            batch_size=config.batch_size,
            eval_env=eval_env,
            eval_episodes=config.n_episodes,
            eval_seed=config.eval_seed,
            eval_interval=config.eval_freq,
            prefix="actor_refit",
            save_dir=actor_refit_dir,
            log_wandb=config.log_wandb,
            log_every=config.log_every,
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

    eval_logs: List[Dict[str, Any]] = []
    best_eval_metric_mean = -np.inf
    last_actor_log: Dict[str, float] = {}
    last_actor_update_step = 0
    actor_log_keys = {"actor_loss", "bc_mse_policy", "bc_mse_random", "action_mse"}

    for t in trange(int(config.max_timesteps), desc=f"{ALGORITHM_NAME} Training"):
        trainer.key, batch_key = jax.random.split(trainer.key)
        batch = replay_buffer.sample_batch(batch_key, batch_size=config.batch_size)

        # Keep the original ReBRAC/TD3-style actor update schedule intact.
        # Actor metrics are cached below so wandb still receives them when
        # log_every lands on a critic-only step.
        update_actor_now = (t % config.policy_freq) == 0
        log_dict = trainer.train(batch, update_actor_now=update_actor_now)
        train_step = int(trainer.total_it)

        if update_actor_now:
            last_actor_update_step = train_step
            last_actor_log = {
                key: value
                for key, value in log_dict.items()
                if key in actor_log_keys
            }

        if config.log_wandb and train_step % config.log_every == 0:
            # Use cached actor metrics under the normal train/* names so plots do
            # not disappear when the current step is critic-only. Also emit an
            # explicit train/last_actor/* namespace and step marker for clarity.
            train_log_dict = {**last_actor_log, **log_dict}
            wandb_train_log = {
                "train/policy_update": float(update_actor_now),
                "train/actor_log_is_cached": float(not update_actor_now and len(last_actor_log) > 0),
                "train/last_actor_update_step": int(last_actor_update_step),
                **{f"train/{key}": value for key, value in train_log_dict.items()},
                **{f"train/last_actor/{key}": value for key, value in last_actor_log.items()},
            }
            wandb.log(wandb_train_log, step=train_step)

        should_eval = train_step % config.eval_freq == 0 or train_step == int(config.max_timesteps)
        if should_eval:
            print(f"Time steps: {train_step}")
            eval_scores, eval_successes = trainer.eval_actor(
                eval_env,
                trainer.actor.params,
                n_episodes=config.n_episodes,
                seed=config.eval_seed,
            )
            normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)
            normalized_score_mean, normalized_score_std = mean_std_or_nan(normalized_eval_scores)
            success_rate, success_std = mean_std_or_nan(eval_successes)

            eval_log: Dict[str, Any] = {
                "timestep": train_step,
                "eval/reward_mean": float(np.mean(eval_scores)),
                "eval/reward_std": float(np.std(eval_scores)),
                "eval/d4rl_normalized_score_mean": normalized_score_mean,
                "eval/d4rl_normalized_score_std": normalized_score_std,
                "eval/success_rate": success_rate,
                "eval/success_std": success_std,
            }
            eval_logs.append(eval_log.copy())

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
                wandb.log(wandb_eval_log, step=train_step)

            save_and_upload_eval_logs(
                eval_logs=eval_logs,
                checkpoints_path=config.checkpoints_path,
                log_wandb=config.log_wandb,
            )

            if config.checkpoints_path is not None and config.save_best_model:
                # Prefer OGBench success rate when available; otherwise fall back to
                # D4RL normalized score for D4RL-style environments.
                eval_metric_mean = success_rate if np.isfinite(success_rate) else normalized_score_mean
                is_best = np.isfinite(eval_metric_mean) and eval_metric_mean > best_eval_metric_mean
                if is_best:
                    best_eval_metric_mean = eval_metric_mean
                    best_checkpoint_path = os.path.join(config.checkpoints_path, "best_checkpoint.pkl")
                    save_checkpoint(
                        best_checkpoint_path,
                        trainer=trainer,
                        config=config,
                        state_mean=state_mean,
                        state_std=state_std,
                        log_wandb=config.log_wandb,
                    )

    if config.checkpoints_path is not None:
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
        save_checkpoint(
            checkpoint_path,
            trainer=trainer,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
            log_wandb=config.log_wandb,
        )
        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )
        print("---------------------------------------")
        print(f"Saved final checkpoint to: {checkpoint_path}")
        if config.save_best_model:
            print(f"Saved best checkpoint to:  {os.path.join(config.checkpoints_path, 'best_checkpoint.pkl')}")
        print("---------------------------------------")


if __name__ == "__main__":
    train()
