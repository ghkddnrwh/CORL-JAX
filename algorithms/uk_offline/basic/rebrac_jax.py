# ReBRAC JAX implementation with IQL_JAX-style experiment plumbing.
#
# Algorithmic losses, target updates, network initialization, D4RL dataset
# construction, and evaluation action selection are kept from the provided
# ReBRAC JAX script. Experiment management follows the accompanying IQL_JAX
# style: config/hyperparameter merging, deterministic run names, wandb logging,
# eval-log persistence, and checkpoint saving/loading.
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
import d4rl  # noqa: F401
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
import yaml
from flax import serialization
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

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
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True

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


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.batch_size > 0
    assert config.eval_freq > 0
    assert config.n_episodes > 0
    assert config.max_timesteps >= 0
    assert config.log_every > 0
    assert config.num_critics > 0
    assert config.gamma >= 0.0 and config.gamma <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.actor_bc_coef >= 0.0
    assert config.critic_bc_coef >= 0.0
    assert config.policy_noise >= 0.0
    assert config.noise_clip >= 0.0
    assert config.policy_freq > 0


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
    ReBRAC key `dataset_name` is accepted as a backward-compatible env alias.
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
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []

    for key, raw_value in env_hyperparams.items():
        key = "env" if key == "dataset_name" else key
        if key not in config_fields:
            skipped_unknown.append(key)
            continue
        if key in cli_overrides:
            skipped_cli.append(key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(key)

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
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
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
    wandb.mark_preempting()


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
    return np.asarray(
        [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
        dtype=np.float32,
    )


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


class ReplayBuffer:
    def __init__(self, device: Any):
        self.data: Optional[TensorBatch] = None
        self.mean: Union[np.ndarray, float] = 0.0
        self.std: Union[np.ndarray, float] = 1.0
        self.device = device

    def create_from_d4rl(
        self,
        env_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        d4rl_data = qlearning_dataset(gym.make(env_name))
        buffer_np = {
            "states": d4rl_data["observations"].astype(np.float32),
            "actions": d4rl_data["actions"].astype(np.float32),
            "rewards": d4rl_data["rewards"].astype(np.float32),
            "next_states": d4rl_data["next_observations"].astype(np.float32),
            "next_actions": d4rl_data["next_actions"].astype(np.float32),
            "dones": d4rl_data["terminals"].astype(np.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer_np["states"], eps=1e-3)
            buffer_np["states"] = normalize_states(buffer_np["states"], self.mean, self.std)
            buffer_np["next_states"] = normalize_states(buffer_np["next_states"], self.mean, self.std)
        if normalize_reward:
            buffer_np["rewards"] = ReplayBuffer.normalize_reward(env_name, buffer_np["rewards"])

        self.data = tree_to_device({k: jnp.asarray(v, dtype=jnp.float32) for k, v in buffer_np.items()}, self.device)
        print(f"Dataset size: {self.size}")

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


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (state - state_mean) / state_std

    def scale_reward(reward: float) -> float:
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


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
    gamma: float,
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

    target_q = batch["rewards"] + (1.0 - batch["dones"]) * gamma * next_q

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
    gamma: float,
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
        gamma,
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
    gamma: float,
    critic_bc_coef: float,
    policy_noise: float,
    noise_clip: float,
) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
    key, new_critic, critic_log = update_critic(
        key,
        actor,
        critic,
        batch,
        gamma,
        critic_bc_coef,
        policy_noise,
        noise_clip,
    )
    return key, actor, new_critic, critic_log


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
        gamma: float = 0.99,
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
        self.gamma = gamma
        self.tau = tau
        self.actor_bc_coef = actor_bc_coef
        self.critic_bc_coef = critic_bc_coef
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.normalize_q = normalize_q
        self.device = device if device is not None else jax.devices()[0]

        key = jax.random.PRNGKey(seed)
        key, actor_key, critic_key = jax.random.split(key, 3)
        init_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        init_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        self.actor_module = DetActor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            layernorm=actor_ln,
            n_hiddens=actor_n_hiddens,
        )
        actor_params = self.actor_module.init(actor_key, init_state)
        self.actor = ActorTrainState.create(
            apply_fn=self.actor_module.apply,
            params=actor_params,
            target_params=deepcopy(actor_params),
            tx=optax.adam(learning_rate=actor_learning_rate),
        )

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

        self._full_update_step = self._build_full_update_step()
        self._critic_update_step = self._build_critic_update_step()

    def _build_full_update_step(self):
        gamma = self.gamma
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
                gamma=gamma,
                actor_bc_coef=actor_bc_coef,
                critic_bc_coef=critic_bc_coef,
                tau=tau,
                policy_noise=policy_noise,
                noise_clip=noise_clip,
                normalize_q=normalize_q,
            )

        return full_update_step

    def _build_critic_update_step(self):
        gamma = self.gamma
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
                gamma=gamma,
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

    def eval_actor(self, env: gym.Env, actor_params: Any, n_episodes: int, seed: int) -> np.ndarray:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        returns = []
        for _ in trange(n_episodes, desc="Eval", leave=False):
            obs, done = env.reset(), False
            total_reward = 0.0
            while not done:
                action = self.actor_act(actor_params, obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            returns.append(total_reward)
        return np.asarray(returns, dtype=np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": serialization.to_state_dict(self.actor),
            "critic": serialization.to_state_dict(self.critic),
            "key": serialization.to_state_dict(self.key),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor = serialization.from_state_dict(self.actor, state_dict["actor"])
        self.critic = serialization.from_state_dict(self.critic, state_dict["critic"])
        self.key = serialization.from_state_dict(self.key, state_dict["key"])
        self.total_it = int(state_dict.get("total_it", 0))
        self.actor = tree_to_device(self.actor, self.device)
        self.critic = tree_to_device(self.critic, self.device)
        self.key = tree_to_device(self.key, self.device)


def resolve_checkpoint_path(load_model: Union[str, Path]) -> Path:
    load_path = Path(load_model)
    if load_path.is_file():
        return load_path
    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")

    candidates = [load_path / "checkpoint.pkl"]
    candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
    candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))
    existing = [path.resolve() for path in candidates if path.exists()]

    if len(existing) == 0:
        tried = "\n".join(str(path) for path in candidates[:20])
        raise FileNotFoundError(
            f"checkpoint file not found under: {load_path}\n"
            f"Tried candidates:\n{tried}"
        )
    if len(existing) > 1:
        found = "\n".join(str(path) for path in existing)
        raise FileNotFoundError(
            f"Multiple checkpoint.pkl files found under {load_path}.\n"
            f"Please provide a more specific --load_model path.\n"
            f"Found:\n{found}"
        )
    return existing[0]


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
    if log_wandb and wandb.run is not None:
        wandb.save(str(checkpoint_path), policy="now")


@pyrallis.wrap()
def train(config: TrainConfig):
    config = apply_env_hyperparams(config)
    config = finalize_checkpoint_path(config)

    jax_device = select_jax_device(config.device)
    eval_env = make_env(config.env, seed=config.eval_seed)
    set_seed(config.seed, eval_env)

    replay_buffer = ReplayBuffer(device=jax_device)
    replay_buffer.create_from_d4rl(
        config.env,
        normalize_reward=config.normalize_reward,
        is_normalize=config.normalize_states,
    )

    state_mean, state_std = replay_buffer.mean, replay_buffer.std
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        config_path = os.path.join(config.checkpoints_path, "config.yaml")
        if os.path.exists(config_path):
            print(f"Error: The file '{config_path}' already exists.")
            exit(1)
        with open(config_path, "w") as f:
            pyrallis.dump(config, f)

    print("---------------------------------------")
    print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
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
        gamma=config.gamma,
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
        checkpoint_path = resolve_checkpoint_path(config.load_model)
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        trainer.load_state_dict(checkpoint["trainer"] if "trainer" in checkpoint else checkpoint)

    if config.log_wandb:
        wandb_init(asdict(config))

    eval_logs: List[Dict[str, Any]] = []
    best_normalized_score_mean = -np.inf

    for t in trange(int(config.max_timesteps), desc=f"{ALGORITHM_NAME} Training"):
        trainer.key, batch_key = jax.random.split(trainer.key)
        batch = replay_buffer.sample_batch(batch_key, batch_size=config.batch_size)
        update_actor_now = (t % config.policy_freq) == 0
        log_dict = trainer.train(batch, update_actor_now=update_actor_now)
        train_step = int(trainer.total_it)

        if config.log_wandb and train_step % config.log_every == 0:
            wandb.log(
                {f"train/{key}": value for key, value in log_dict.items()},
                step=train_step,
            )

        should_eval = train_step % config.eval_freq == 0 or train_step == int(config.max_timesteps)
        if should_eval:
            print(f"Time steps: {train_step}")
            eval_scores = trainer.eval_actor(
                eval_env,
                trainer.actor.params,
                n_episodes=config.n_episodes,
                seed=config.eval_seed,
            )
            normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)
            normalized_score_mean = float(np.mean(normalized_eval_scores))

            eval_log: Dict[str, Any] = {
                "timestep": train_step,
                "eval/reward_mean": float(np.mean(eval_scores)),
                "eval/reward_std": float(np.std(eval_scores)),
                "eval/normalized_score_mean": normalized_score_mean,
                "eval/normalized_score_std": float(np.std(normalized_eval_scores)),
            }
            eval_logs.append(eval_log.copy())

            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"reward={eval_log['eval/reward_mean']:.3f} ± {eval_log['eval/reward_std']:.3f}, "
                f"D4RL={eval_log['eval/normalized_score_mean']:.3f} ± "
                f"{eval_log['eval/normalized_score_std']:.3f}"
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
                is_best = normalized_score_mean > best_normalized_score_mean
                if is_best:
                    best_normalized_score_mean = normalized_score_mean
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


# # ReBRAC JAX implementation with IQL_JAX-style experiment plumbing.
# #
# # Algorithmic losses, target updates, network initialization, D4RL dataset
# # construction, and evaluation action selection are kept from the provided
# # ReBRAC JAX script. Experiment management follows the accompanying IQL_JAX
# # style: config/hyperparameter merging, deterministic run names, wandb logging,
# # eval-log persistence, and checkpoint saving/loading.
# #
# # Source algorithm:
# #   https://github.com/tinkoff-ai/ReBRAC
# #   https://arxiv.org/abs/2305.09836

# import os

# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility.

# import math
# import pickle
# import random
# import sys
# import uuid
# from copy import deepcopy
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import chex
# import d4rl  # noqa: F401
# import flax.linen as nn
# import gym
# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# import pyrallis
# import wandb
# import yaml
# from flax import serialization
# from flax.core import FrozenDict
# from flax.training.train_state import TrainState
# from tqdm.auto import trange

# TensorBatch = Dict[str, jnp.ndarray]

# ALGORITHM_NAME = "ReBRAC"
# ALGORITHM_FULL_NAME = "Revisiting Behavioral Regularization in Actor-Critic"


# default_kernel_init = nn.initializers.lecun_normal()
# default_bias_init = nn.initializers.zeros


# @dataclass
# class TrainConfig:
#     # Experiment.
#     device: str = "gpu"  # one of: cpu, gpu, tpu. JAX falls back if unavailable.
#     env: str = "halfcheetah-medium-v2"
#     seed: int = 0
#     eval_seed: int = 42

#     # IQL_JAX-style schedule fields.
#     max_timesteps: int = int(1e6)
#     eval_freq: int = int(25e3)
#     n_episodes: int = 10

#     checkpoints_path: Optional[str] = None
#     load_model: str = ""
#     hyperparams_path: Optional[str] = "hyperparams/rebrac_jax.yml"
#     use_hyperparams: bool = True

#     # Backward-compatible alias for the original ReBRAC script.
#     # Prefer --env in new runs. If provided, dataset_name overrides env.
#     dataset_name: Optional[str] = None

#     # Dataset.
#     batch_size: int = 1024
#     normalize_reward: bool = False
#     normalize_states: bool = False

#     # ReBRAC / TD3+BC-style actor-critic parameters.
#     actor_learning_rate: float = 1e-3
#     critic_learning_rate: float = 1e-3
#     hidden_dim: int = 256
#     actor_n_hiddens: int = 3
#     critic_n_hiddens: int = 3
#     num_critics: int = 2
#     discount: float = 0.99
#     tau: float = 5e-3
#     actor_bc_coef: float = 1.0
#     critic_bc_coef: float = 1.0
#     actor_ln: bool = False
#     critic_ln: bool = True
#     policy_noise: float = 0.2
#     noise_clip: float = 0.5
#     policy_freq: int = 2
#     normalize_q: bool = True

#     # Logging / saving.
#     project: str = "ORL-SMOOTH"
#     group: str = "ReBRAC-JAX"
#     name: str = "ReBRAC-JAX"
#     log_wandb: bool = True
#     log_every: int = 500
#     save_best_model: bool = True

#     def __post_init__(self):
#         normalize_config_aliases(self)
#         refresh_algorithm_names(self)
#         validate_config(self)


# def normalize_config_aliases(config: TrainConfig) -> None:
#     if config.dataset_name is not None:
#         config.env = config.dataset_name


# def refresh_algorithm_names(config: TrainConfig) -> None:
#     config.project = "ORL-SMOOTH"
#     config.group = f"{ALGORITHM_NAME}-JAX"
#     config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


# def validate_config(config: TrainConfig) -> None:
#     assert config.batch_size > 0
#     assert config.eval_freq > 0
#     assert config.n_episodes > 0
#     assert config.max_timesteps >= 0
#     assert config.log_every > 0
#     assert config.num_critics > 0
#     assert config.discount >= 0.0 and config.discount <= 1.0
#     assert config.tau >= 0.0 and config.tau <= 1.0
#     assert config.actor_bc_coef >= 0.0
#     assert config.critic_bc_coef >= 0.0
#     assert config.policy_noise >= 0.0
#     assert config.noise_clip >= 0.0
#     assert config.policy_freq > 0


# def _cli_overridden_fields(argv: Optional[List[str]] = None) -> set:
#     argv = sys.argv[1:] if argv is None else argv
#     overridden = set()
#     for token in argv:
#         if not token.startswith("--"):
#             continue
#         key = token[2:].split("=", 1)[0].replace("-", "_")
#         if key:
#             overridden.add(key)
#     return overridden


# def _coerce_hparam_value(value: Any) -> Any:
#     # YAML may load scientific notation such as 1e6 as float.
#     if isinstance(value, float) and value.is_integer():
#         return int(value)
#     return value


# def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
#     """Load env-specific hyperparameters and merge them into config.

#     Priority:
#         dataclass defaults < hyperparams YAML < explicit CLI flags

#     Hyperparameter YAML keys should match TrainConfig field names. The old
#     ReBRAC key `dataset_name` is accepted as a backward-compatible env alias.
#     """
#     normalize_config_aliases(config)
#     if not config.use_hyperparams or config.hyperparams_path is None:
#         refresh_algorithm_names(config)
#         validate_config(config)
#         return config

#     hparam_path = Path(config.hyperparams_path)
#     if not hparam_path.exists():
#         raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}.")

#     with open(hparam_path, "r") as f:
#         all_hyperparams = yaml.safe_load(f) or {}

#     if config.env not in all_hyperparams:
#         print(
#             f"No hyperparameters found for env '{config.env}' in {hparam_path}. "
#             "Using dataclass/CLI values."
#         )
#         refresh_algorithm_names(config)
#         validate_config(config)
#         return config

#     env_hyperparams = all_hyperparams[config.env] or {}
#     cli_overrides = _cli_overridden_fields()
#     config_fields = set(TrainConfig.__dataclass_fields__.keys())
#     applied, skipped_unknown, skipped_cli = [], [], []

#     for key, raw_value in env_hyperparams.items():
#         key = "env" if key == "dataset_name" else key
#         if key not in config_fields:
#             skipped_unknown.append(key)
#             continue
#         if key in cli_overrides:
#             skipped_cli.append(key)
#             continue
#         setattr(config, key, _coerce_hparam_value(raw_value))
#         applied.append(key)

#     normalize_config_aliases(config)
#     refresh_algorithm_names(config)
#     validate_config(config)

#     if applied:
#         print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
#     if skipped_cli:
#         print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
#     if skipped_unknown:
#         print(f"Ignored unknown hyperparameter keys for ReBRAC: {', '.join(skipped_unknown)}")
#     return config


# def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
#     if config.checkpoints_path is not None:
#         config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
#     return config


# def select_jax_device(device: str):
#     backend = device.lower()
#     if backend == "cuda":
#         backend = "gpu"
#     try:
#         dev = jax.devices(backend)[0]
#     except Exception:
#         print(f"Requested JAX backend '{device}' is not available. Falling back to default device.")
#         dev = jax.devices()[0]
#     print(f"Using JAX device: {dev}")
#     return dev


# def tree_to_device(tree, device):
#     return jax.device_put(tree, device)


# def set_seed(seed: int, env: Optional[gym.Env] = None):
#     if env is not None:
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)


# def wandb_init(config: dict) -> None:
#     wandb.init(
#         config=config,
#         project=config["project"],
#         group=config["group"],
#         name=config["name"],
#         id=str(uuid.uuid4()),
#     )
#     wandb.run.save()
#     wandb.mark_preempting()


# def is_scalar_value(value: Any) -> bool:
#     if isinstance(value, (int, float, bool, np.number)):
#         return True
#     if isinstance(value, np.ndarray) and value.ndim == 0:
#         return True
#     return False


# def to_python_scalar(value: Any) -> Union[int, float, bool]:
#     if isinstance(value, np.ndarray):
#         return value.item()
#     if isinstance(value, np.generic):
#         return value.item()
#     if isinstance(value, jnp.ndarray) and value.ndim == 0:
#         return float(value)
#     return value


# def save_logs_npz(logs: List[Dict[str, Any]], path: Union[str, Path]) -> None:
#     if len(logs) == 0:
#         return
#     keys = logs[0].keys()
#     data_to_save: Dict[str, np.ndarray] = {}
#     for key in keys:
#         values = [log.get(key, np.nan) for log in logs]
#         try:
#             data_to_save[key] = np.asarray(values)
#         except ValueError:
#             data_to_save[key] = np.asarray(values, dtype=object)
#     np.savez(path, **data_to_save)


# def save_and_upload_eval_logs(
#     eval_logs: List[Dict[str, Any]],
#     checkpoints_path: Optional[str],
#     log_wandb: bool,
# ):
#     if checkpoints_path is None or len(eval_logs) == 0:
#         return
#     eval_logs_path = os.path.join(checkpoints_path, "eval_logs.npz")
#     save_logs_npz(eval_logs, eval_logs_path)
#     if log_wandb and wandb.run is not None:
#         wandb.save(eval_logs_path, policy="now")


# def save_pickle(path: Union[str, Path], obj: Any) -> None:
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)


# def load_pickle(path: Union[str, Path]) -> Any:
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
#     return np.asarray(
#         [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
#         dtype=np.float32,
#     )


# def pytorch_init(fan_in: float) -> Callable:
#     """Default init for PyTorch Linear layer weights and biases."""
#     bound = math.sqrt(1 / fan_in)

#     def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
#         return jax.random.uniform(
#             key,
#             shape=shape,
#             minval=-bound,
#             maxval=bound,
#             dtype=dtype,
#         )

#     return _init


# def uniform_init(bound: float) -> Callable:
#     def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
#         return jax.random.uniform(
#             key,
#             shape=shape,
#             minval=-bound,
#             maxval=bound,
#             dtype=dtype,
#         )

#     return _init


# def identity(x: Any) -> Any:
#     return x


# class DetActor(nn.Module):
#     action_dim: int
#     hidden_dim: int = 256
#     layernorm: bool = True
#     n_hiddens: int = 3

#     @nn.compact
#     def __call__(self, state: jax.Array) -> jax.Array:
#         s_d, h_d = state.shape[-1], self.hidden_dim
#         # Initialization as in the EDAC/ReBRAC implementation.
#         layers = [
#             nn.Dense(
#                 self.hidden_dim,
#                 kernel_init=pytorch_init(s_d),
#                 bias_init=nn.initializers.constant(0.1),
#             ),
#             nn.relu,
#             nn.LayerNorm() if self.layernorm else identity,
#         ]
#         for _ in range(self.n_hiddens - 1):
#             layers += [
#                 nn.Dense(
#                     self.hidden_dim,
#                     kernel_init=pytorch_init(h_d),
#                     bias_init=nn.initializers.constant(0.1),
#                 ),
#                 nn.relu,
#                 nn.LayerNorm() if self.layernorm else identity,
#             ]
#         layers += [
#             nn.Dense(
#                 self.action_dim,
#                 kernel_init=uniform_init(1e-3),
#                 bias_init=uniform_init(1e-3),
#             ),
#             nn.tanh,
#         ]
#         return nn.Sequential(layers)(state)


# class Critic(nn.Module):
#     hidden_dim: int = 256
#     layernorm: bool = True
#     n_hiddens: int = 3

#     @nn.compact
#     def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
#         s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
#         # Initialization as in the EDAC/ReBRAC implementation.
#         layers = [
#             nn.Dense(
#                 self.hidden_dim,
#                 kernel_init=pytorch_init(s_d + a_d),
#                 bias_init=nn.initializers.constant(0.1),
#             ),
#             nn.relu,
#             nn.LayerNorm() if self.layernorm else identity,
#         ]
#         for _ in range(self.n_hiddens - 1):
#             layers += [
#                 nn.Dense(
#                     self.hidden_dim,
#                     kernel_init=pytorch_init(h_d),
#                     bias_init=nn.initializers.constant(0.1),
#                 ),
#                 nn.relu,
#                 nn.LayerNorm() if self.layernorm else identity,
#             ]
#         layers += [nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))]
#         state_action = jnp.hstack([state, action])
#         return nn.Sequential(layers)(state_action).squeeze(-1)


# class EnsembleCritic(nn.Module):
#     hidden_dim: int = 256
#     num_critics: int = 2
#     layernorm: bool = True
#     n_hiddens: int = 3

#     @nn.compact
#     def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
#         ensemble = nn.vmap(
#             target=Critic,
#             in_axes=None,
#             out_axes=0,
#             variable_axes={"params": 0},
#             split_rngs={"params": True},
#             axis_size=self.num_critics,
#         )
#         return ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(state, action)


# def qlearning_dataset(
#     env: gym.Env,
#     dataset: Dict = None,
#     terminate_on_end: bool = False,
#     **kwargs,
# ) -> Dict:
#     if dataset is None:
#         dataset = env.get_dataset(**kwargs)

#     n_transitions = dataset["rewards"].shape[0]
#     obs_, next_obs_, action_, next_action_, reward_, done_ = [], [], [], [], [], []
#     use_timeouts = "timeouts" in dataset

#     episode_step = 0
#     for i in range(n_transitions - 1):
#         obs = dataset["observations"][i].astype(np.float32)
#         new_obs = dataset["observations"][i + 1].astype(np.float32)
#         action = dataset["actions"][i].astype(np.float32)
#         new_action = dataset["actions"][i + 1].astype(np.float32)
#         reward = dataset["rewards"][i].astype(np.float32)
#         done_bool = bool(dataset["terminals"][i])

#         if use_timeouts:
#             final_timestep = dataset["timeouts"][i]
#         else:
#             final_timestep = episode_step == env._max_episode_steps - 1
#         if (not terminate_on_end) and final_timestep:
#             episode_step = 0
#             continue
#         if done_bool or final_timestep:
#             episode_step = 0

#         obs_.append(obs)
#         next_obs_.append(new_obs)
#         action_.append(action)
#         next_action_.append(new_action)
#         reward_.append(reward)
#         done_.append(done_bool)
#         episode_step += 1

#     return {
#         "observations": np.array(obs_),
#         "actions": np.array(action_),
#         "next_observations": np.array(next_obs_),
#         "next_actions": np.array(next_action_),
#         "rewards": np.array(reward_),
#         "terminals": np.array(done_),
#     }


# def compute_mean_std(states: Union[np.ndarray, jax.Array], eps: float) -> Tuple[np.ndarray, np.ndarray]:
#     mean = np.asarray(states).mean(0)
#     std = np.asarray(states).std(0) + eps
#     return mean, std


# def normalize_states(
#     states: Union[np.ndarray, jax.Array],
#     mean: Union[np.ndarray, float],
#     std: Union[np.ndarray, float],
# ) -> np.ndarray:
#     return (states - mean) / std


# class ReplayBuffer:
#     def __init__(self, device: Any):
#         self.data: Optional[TensorBatch] = None
#         self.mean: Union[np.ndarray, float] = 0.0
#         self.std: Union[np.ndarray, float] = 1.0
#         self.device = device

#     def create_from_d4rl(
#         self,
#         env_name: str,
#         normalize_reward: bool = False,
#         is_normalize: bool = False,
#     ):
#         d4rl_data = qlearning_dataset(gym.make(env_name))
#         buffer_np = {
#             "states": d4rl_data["observations"].astype(np.float32),
#             "actions": d4rl_data["actions"].astype(np.float32),
#             "rewards": d4rl_data["rewards"].astype(np.float32),
#             "next_states": d4rl_data["next_observations"].astype(np.float32),
#             "next_actions": d4rl_data["next_actions"].astype(np.float32),
#             "dones": d4rl_data["terminals"].astype(np.float32),
#         }
#         if is_normalize:
#             self.mean, self.std = compute_mean_std(buffer_np["states"], eps=1e-3)
#             buffer_np["states"] = normalize_states(buffer_np["states"], self.mean, self.std)
#             buffer_np["next_states"] = normalize_states(buffer_np["next_states"], self.mean, self.std)
#         if normalize_reward:
#             buffer_np["rewards"] = ReplayBuffer.normalize_reward(env_name, buffer_np["rewards"])

#         self.data = tree_to_device({k: jnp.asarray(v, dtype=jnp.float32) for k, v in buffer_np.items()}, self.device)
#         print(f"Dataset size: {self.size}")

#     @property
#     def size(self) -> int:
#         if self.data is None:
#             return 0
#         return int(self.data["states"].shape[0])

#     @property
#     def state_dim(self) -> int:
#         return int(self.data["states"].shape[-1])

#     @property
#     def action_dim(self) -> int:
#         return int(self.data["actions"].shape[-1])

#     def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> TensorBatch:
#         indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
#         return jax.tree_util.tree_map(lambda arr: arr[indices], self.data)

#     @staticmethod
#     def normalize_reward(dataset_name: str, rewards: np.ndarray) -> np.ndarray:
#         if "antmaze" in dataset_name:
#             return rewards * 100.0  # Like in LAPO / original ReBRAC code.
#         raise NotImplementedError("Reward normalization is implemented only for AntMaze.")


# def make_env(env_name: str, seed: int) -> gym.Env:
#     env = gym.make(env_name)
#     env.seed(seed)
#     env.action_space.seed(seed)
#     env.observation_space.seed(seed)
#     return env


# def wrap_env(
#     env: gym.Env,
#     state_mean: Union[np.ndarray, float] = 0.0,
#     state_std: Union[np.ndarray, float] = 1.0,
#     reward_scale: float = 1.0,
# ) -> gym.Env:
#     def normalize_state(state: np.ndarray) -> np.ndarray:
#         return (state - state_mean) / state_std

#     def scale_reward(reward: float) -> float:
#         return reward_scale * reward

#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env


# class CriticTrainState(TrainState):
#     target_params: FrozenDict


# class ActorTrainState(TrainState):
#     target_params: FrozenDict


# def update_actor(
#     key: jax.random.PRNGKey,
#     actor: ActorTrainState,
#     critic: CriticTrainState,
#     batch: TensorBatch,
#     beta: float,
#     tau: float,
#     normalize_q: bool,
# ) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
#     key, random_action_key = jax.random.split(key, 2)

#     def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
#         actions = actor.apply_fn(params, batch["states"])

#         bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
#         q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
#         lmbda = 1.0
#         if normalize_q:
#             lmbda = jax.lax.stop_gradient(1.0 / jnp.abs(q_values).mean())

#         loss = (beta * bc_penalty - lmbda * q_values).mean()
#         random_actions = jax.random.uniform(
#             random_action_key,
#             shape=batch["actions"].shape,
#             minval=-1.0,
#             maxval=1.0,
#         )
#         log_dict = {
#             "actor_loss": loss,
#             "bc_mse_policy": bc_penalty.mean(),
#             "bc_mse_random": ((random_actions - batch["actions"]) ** 2).sum(-1).mean(),
#             "action_mse": ((actions - batch["actions"]) ** 2).mean(),
#         }
#         return loss, log_dict

#     grads, log_dict = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
#     new_actor = actor.apply_gradients(grads=grads)
#     new_actor = new_actor.replace(
#         target_params=optax.incremental_update(actor.params, actor.target_params, tau)
#     )
#     new_critic = critic.replace(
#         target_params=optax.incremental_update(critic.params, critic.target_params, tau)
#     )
#     return key, new_actor, new_critic, log_dict


# def update_critic(
#     key: jax.random.PRNGKey,
#     actor: ActorTrainState,
#     critic: CriticTrainState,
#     batch: TensorBatch,
#     discount: float,
#     beta: float,
#     policy_noise: float,
#     noise_clip: float,
# ) -> Tuple[jax.random.PRNGKey, CriticTrainState, Dict[str, jax.Array]]:
#     key, actions_key = jax.random.split(key)

#     next_actions = actor.apply_fn(actor.target_params, batch["next_states"])
#     noise = jnp.clip(
#         jax.random.normal(actions_key, next_actions.shape) * policy_noise,
#         -noise_clip,
#         noise_clip,
#     )
#     next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
#     bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
#     next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
#     next_q = next_q - beta * bc_penalty

#     target_q = batch["rewards"] + (1.0 - batch["dones"]) * discount * next_q

#     def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
#         q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
#         q_min = q.min(0).mean()
#         loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
#         return loss, q_min

#     (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
#     new_critic = critic.apply_gradients(grads=grads)
#     log_dict = {
#         "critic_loss": loss,
#         "q_min": q_min,
#     }
#     return key, new_critic, log_dict


# def update_td3(
#     key: jax.random.PRNGKey,
#     actor: ActorTrainState,
#     critic: CriticTrainState,
#     batch: TensorBatch,
#     discount: float,
#     actor_bc_coef: float,
#     critic_bc_coef: float,
#     tau: float,
#     policy_noise: float,
#     noise_clip: float,
#     normalize_q: bool,
# ) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
#     key, new_critic, critic_log = update_critic(
#         key,
#         actor,
#         critic,
#         batch,
#         discount,
#         critic_bc_coef,
#         policy_noise,
#         noise_clip,
#     )
#     key, new_actor, new_critic, actor_log = update_actor(
#         key,
#         actor,
#         new_critic,
#         batch,
#         actor_bc_coef,
#         tau,
#         normalize_q,
#     )
#     return key, new_actor, new_critic, {**critic_log, **actor_log}


# def update_td3_no_targets(
#     key: jax.random.PRNGKey,
#     actor: ActorTrainState,
#     critic: CriticTrainState,
#     batch: TensorBatch,
#     discount: float,
#     critic_bc_coef: float,
#     policy_noise: float,
#     noise_clip: float,
# ) -> Tuple[jax.random.PRNGKey, ActorTrainState, CriticTrainState, Dict[str, jax.Array]]:
#     key, new_critic, critic_log = update_critic(
#         key,
#         actor,
#         critic,
#         batch,
#         discount,
#         critic_bc_coef,
#         policy_noise,
#         noise_clip,
#     )
#     return key, actor, new_critic, critic_log


# class ReBRACJAX:
#     """ReBRAC in JAX/Flax with IQL_JAX-style training plumbing."""

#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         actor_learning_rate: float = 1e-3,
#         critic_learning_rate: float = 1e-3,
#         hidden_dim: int = 256,
#         actor_n_hiddens: int = 3,
#         critic_n_hiddens: int = 3,
#         num_critics: int = 2,
#         discount: float = 0.99,
#         tau: float = 5e-3,
#         actor_bc_coef: float = 1.0,
#         critic_bc_coef: float = 1.0,
#         actor_ln: bool = False,
#         critic_ln: bool = True,
#         policy_noise: float = 0.2,
#         noise_clip: float = 0.5,
#         normalize_q: bool = True,
#         seed: int = 0,
#         device: Any = None,
#     ):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.discount = discount
#         self.tau = tau
#         self.actor_bc_coef = actor_bc_coef
#         self.critic_bc_coef = critic_bc_coef
#         self.policy_noise = policy_noise
#         self.noise_clip = noise_clip
#         self.normalize_q = normalize_q
#         self.device = device if device is not None else jax.devices()[0]

#         key = jax.random.PRNGKey(seed)
#         key, actor_key, critic_key = jax.random.split(key, 3)
#         init_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
#         init_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

#         self.actor_module = DetActor(
#             action_dim=action_dim,
#             hidden_dim=hidden_dim,
#             layernorm=actor_ln,
#             n_hiddens=actor_n_hiddens,
#         )
#         actor_params = self.actor_module.init(actor_key, init_state)
#         self.actor = ActorTrainState.create(
#             apply_fn=self.actor_module.apply,
#             params=actor_params,
#             target_params=deepcopy(actor_params),
#             tx=optax.adam(learning_rate=actor_learning_rate),
#         )

#         self.critic_module = EnsembleCritic(
#             hidden_dim=hidden_dim,
#             num_critics=num_critics,
#             layernorm=critic_ln,
#             n_hiddens=critic_n_hiddens,
#         )
#         critic_params = self.critic_module.init(critic_key, init_state, init_action)
#         self.critic = CriticTrainState.create(
#             apply_fn=self.critic_module.apply,
#             params=critic_params,
#             target_params=deepcopy(critic_params),
#             tx=optax.adam(learning_rate=critic_learning_rate),
#         )

#         self.key = key
#         self.total_it = 0
#         self.actor = tree_to_device(self.actor, self.device)
#         self.critic = tree_to_device(self.critic, self.device)
#         self.key = tree_to_device(self.key, self.device)

#         self._full_update_step = self._build_full_update_step()
#         self._critic_update_step = self._build_critic_update_step()

#     def _build_full_update_step(self):
#         discount = self.discount
#         actor_bc_coef = self.actor_bc_coef
#         critic_bc_coef = self.critic_bc_coef
#         tau = self.tau
#         policy_noise = self.policy_noise
#         noise_clip = self.noise_clip
#         normalize_q = self.normalize_q

#         @jax.jit
#         def full_update_step(key, actor, critic, batch):
#             return update_td3(
#                 key=key,
#                 actor=actor,
#                 critic=critic,
#                 batch=batch,
#                 discount=discount,
#                 actor_bc_coef=actor_bc_coef,
#                 critic_bc_coef=critic_bc_coef,
#                 tau=tau,
#                 policy_noise=policy_noise,
#                 noise_clip=noise_clip,
#                 normalize_q=normalize_q,
#             )

#         return full_update_step

#     def _build_critic_update_step(self):
#         discount = self.discount
#         critic_bc_coef = self.critic_bc_coef
#         policy_noise = self.policy_noise
#         noise_clip = self.noise_clip

#         @jax.jit
#         def critic_update_step(key, actor, critic, batch):
#             return update_td3_no_targets(
#                 key=key,
#                 actor=actor,
#                 critic=critic,
#                 batch=batch,
#                 discount=discount,
#                 critic_bc_coef=critic_bc_coef,
#                 policy_noise=policy_noise,
#                 noise_clip=noise_clip,
#             )

#         return critic_update_step

#     def train(self, batch: TensorBatch, update_actor_now: bool) -> Dict[str, float]:
#         if update_actor_now:
#             self.key, self.actor, self.critic, log_dict = self._full_update_step(
#                 self.key,
#                 self.actor,
#                 self.critic,
#                 batch,
#             )
#         else:
#             self.key, self.actor, self.critic, log_dict = self._critic_update_step(
#                 self.key,
#                 self.actor,
#                 self.critic,
#                 batch,
#             )
#         self.total_it += 1
#         return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

#     def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
#         state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
#         action = self.actor.apply_fn(actor_params, state_jnp)
#         return np.asarray(jax.device_get(action))[0]

#     def eval_actor(self, env: gym.Env, actor_params: Any, n_episodes: int, seed: int) -> np.ndarray:
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         returns = []
#         for _ in trange(n_episodes, desc="Eval", leave=False):
#             obs, done = env.reset(), False
#             total_reward = 0.0
#             while not done:
#                 action = self.actor_act(actor_params, obs)
#                 obs, reward, done, _ = env.step(action)
#                 total_reward += reward
#             returns.append(total_reward)
#         return np.asarray(returns, dtype=np.float32)

#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "actor": serialization.to_state_dict(self.actor),
#             "critic": serialization.to_state_dict(self.critic),
#             "key": serialization.to_state_dict(self.key),
#             "total_it": self.total_it,
#         }

#     def load_state_dict(self, state_dict: Dict[str, Any]):
#         self.actor = serialization.from_state_dict(self.actor, state_dict["actor"])
#         self.critic = serialization.from_state_dict(self.critic, state_dict["critic"])
#         self.key = serialization.from_state_dict(self.key, state_dict["key"])
#         self.total_it = int(state_dict.get("total_it", 0))
#         self.actor = tree_to_device(self.actor, self.device)
#         self.critic = tree_to_device(self.critic, self.device)
#         self.key = tree_to_device(self.key, self.device)


# def resolve_checkpoint_path(load_model: Union[str, Path]) -> Path:
#     load_path = Path(load_model)
#     if load_path.is_file():
#         return load_path
#     if not load_path.exists():
#         raise FileNotFoundError(f"load_model path does not exist: {load_path}")

#     candidates = [load_path / "checkpoint.pkl"]
#     candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
#     candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))
#     existing = [path.resolve() for path in candidates if path.exists()]

#     if len(existing) == 0:
#         tried = "\n".join(str(path) for path in candidates[:20])
#         raise FileNotFoundError(
#             f"checkpoint file not found under: {load_path}\n"
#             f"Tried candidates:\n{tried}"
#         )
#     if len(existing) > 1:
#         found = "\n".join(str(path) for path in existing)
#         raise FileNotFoundError(
#             f"Multiple checkpoint.pkl files found under {load_path}.\n"
#             f"Please provide a more specific --load_model path.\n"
#             f"Found:\n{found}"
#         )
#     return existing[0]


# def make_checkpoint_payload(
#     trainer: ReBRACJAX,
#     config: TrainConfig,
#     state_mean: Union[np.ndarray, float],
#     state_std: Union[np.ndarray, float],
# ) -> Dict[str, Any]:
#     return {
#         "trainer": trainer.state_dict(),
#         "config": asdict(config),
#         "state_mean": state_mean,
#         "state_std": state_std,
#     }


# def save_checkpoint(
#     checkpoint_path: Union[str, Path],
#     trainer: ReBRACJAX,
#     config: TrainConfig,
#     state_mean: Union[np.ndarray, float],
#     state_std: Union[np.ndarray, float],
#     log_wandb: bool,
# ) -> None:
#     save_pickle(
#         checkpoint_path,
#         make_checkpoint_payload(
#             trainer=trainer,
#             config=config,
#             state_mean=state_mean,
#             state_std=state_std,
#         ),
#     )
#     if log_wandb and wandb.run is not None:
#         wandb.save(str(checkpoint_path), policy="now")


# @pyrallis.wrap()
# def train(config: TrainConfig):
#     config = apply_env_hyperparams(config)
#     config = finalize_checkpoint_path(config)

#     jax_device = select_jax_device(config.device)
#     eval_env = make_env(config.env, seed=config.eval_seed)
#     set_seed(config.seed, eval_env)

#     replay_buffer = ReplayBuffer(device=jax_device)
#     replay_buffer.create_from_d4rl(
#         config.env,
#         normalize_reward=config.normalize_reward,
#         is_normalize=config.normalize_states,
#     )

#     state_mean, state_std = replay_buffer.mean, replay_buffer.std
#     eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

#     if config.checkpoints_path is not None:
#         print(f"Checkpoints path: {config.checkpoints_path}")
#         os.makedirs(config.checkpoints_path, exist_ok=True)
#         config_path = os.path.join(config.checkpoints_path, "config.yaml")
#         if os.path.exists(config_path):
#             print(f"Error: The file '{config_path}' already exists.")
#             exit(1)
#         with open(config_path, "w") as f:
#             pyrallis.dump(config, f)

#     print("---------------------------------------")
#     print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
#     print("---------------------------------------")

#     trainer = ReBRACJAX(
#         state_dim=replay_buffer.state_dim,
#         action_dim=replay_buffer.action_dim,
#         actor_learning_rate=config.actor_learning_rate,
#         critic_learning_rate=config.critic_learning_rate,
#         hidden_dim=config.hidden_dim,
#         actor_n_hiddens=config.actor_n_hiddens,
#         critic_n_hiddens=config.critic_n_hiddens,
#         num_critics=config.num_critics,
#         discount=config.discount,
#         tau=config.tau,
#         actor_bc_coef=config.actor_bc_coef,
#         critic_bc_coef=config.critic_bc_coef,
#         actor_ln=config.actor_ln,
#         critic_ln=config.critic_ln,
#         policy_noise=config.policy_noise,
#         noise_clip=config.noise_clip,
#         normalize_q=config.normalize_q,
#         seed=config.seed,
#         device=jax_device,
#     )

#     if config.load_model != "":
#         checkpoint_path = resolve_checkpoint_path(config.load_model)
#         print(f"Loading checkpoint from: {checkpoint_path}")
#         checkpoint = load_pickle(checkpoint_path)
#         trainer.load_state_dict(checkpoint["trainer"] if "trainer" in checkpoint else checkpoint)

#     if config.log_wandb:
#         wandb_init(asdict(config))

#     eval_logs: List[Dict[str, Any]] = []
#     best_normalized_score_mean = -np.inf

#     for t in trange(int(config.max_timesteps), desc=f"{ALGORITHM_NAME} Training"):
#         trainer.key, batch_key = jax.random.split(trainer.key)
#         batch = replay_buffer.sample_batch(batch_key, batch_size=config.batch_size)
#         update_actor_now = (t % config.policy_freq) == 0
#         log_dict = trainer.train(batch, update_actor_now=update_actor_now)
#         train_step = int(trainer.total_it)

#         if config.log_wandb and train_step % config.log_every == 0:
#             wandb.log(
#                 {f"train/{key}": value for key, value in log_dict.items()},
#                 step=train_step,
#             )

#         should_eval = train_step % config.eval_freq == 0 or train_step == int(config.max_timesteps)
#         if should_eval:
#             print(f"Time steps: {train_step}")
#             eval_scores = trainer.eval_actor(
#                 eval_env,
#                 trainer.actor.params,
#                 n_episodes=config.n_episodes,
#                 seed=config.eval_seed,
#             )
#             normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)
#             normalized_score_mean = float(np.mean(normalized_eval_scores))

#             eval_log: Dict[str, Any] = {
#                 "timestep": train_step,
#                 "eval/reward_mean": float(np.mean(eval_scores)),
#                 "eval/reward_std": float(np.std(eval_scores)),
#                 "eval/normalized_score_mean": normalized_score_mean,
#                 "eval/normalized_score_std": float(np.std(normalized_eval_scores)),
#             }
#             eval_logs.append(eval_log.copy())

#             print(
#                 f"Evaluation over {config.n_episodes} episodes: "
#                 f"reward={eval_log['eval/reward_mean']:.3f} ± {eval_log['eval/reward_std']:.3f}, "
#                 f"D4RL={eval_log['eval/normalized_score_mean']:.3f} ± "
#                 f"{eval_log['eval/normalized_score_std']:.3f}"
#             )

#             if config.log_wandb:
#                 wandb_eval_log = {
#                     key: to_python_scalar(value)
#                     for key, value in eval_log.items()
#                     if is_scalar_value(value)
#                 }
#                 wandb.log(wandb_eval_log, step=train_step)

#             save_and_upload_eval_logs(
#                 eval_logs=eval_logs,
#                 checkpoints_path=config.checkpoints_path,
#                 log_wandb=config.log_wandb,
#             )

#             if config.checkpoints_path is not None and config.save_best_model:
#                 is_best = normalized_score_mean > best_normalized_score_mean
#                 if is_best:
#                     best_normalized_score_mean = normalized_score_mean
#                     best_checkpoint_path = os.path.join(config.checkpoints_path, "best_checkpoint.pkl")
#                     save_checkpoint(
#                         best_checkpoint_path,
#                         trainer=trainer,
#                         config=config,
#                         state_mean=state_mean,
#                         state_std=state_std,
#                         log_wandb=config.log_wandb,
#                     )

#     if config.checkpoints_path is not None:
#         checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
#         save_checkpoint(
#             checkpoint_path,
#             trainer=trainer,
#             config=config,
#             state_mean=state_mean,
#             state_std=state_std,
#             log_wandb=config.log_wandb,
#         )
#         save_and_upload_eval_logs(
#             eval_logs=eval_logs,
#             checkpoints_path=config.checkpoints_path,
#             log_wandb=config.log_wandb,
#         )
#         print("---------------------------------------")
#         print(f"Saved final checkpoint to: {checkpoint_path}")
#         if config.save_best_model:
#             print(f"Saved best checkpoint to:  {os.path.join(config.checkpoints_path, 'best_checkpoint.pkl')}")
#         print("---------------------------------------")


# if __name__ == "__main__":
#     train()
