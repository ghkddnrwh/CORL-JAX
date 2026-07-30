# JAX/Flax Delayed IQL implementation with CDAF_JAX-style experiment plumbing.
#
# Delayed IQL idea:
#   - Introduce delayed Q and delayed V networks.
#   - Every delayed_update_period steps:
#       q_delayed_params <- q_target_params
#       v_delayed_params <- v_params
#   - The expectile/asymmetric L2 value-loss weight is computed from
#       Q_delayed(s, a) - V_delayed(s)
#     while the value-regression residual is still computed from
#       Q_target(s, a) - V(s).
#   - This keeps the value-loss weight fixed between delayed-network refreshes.
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

ALGORITHM_NAME = "DelayedIQL"
ALGORITHM_FULL_NAME = "Delayed Implicit Q-Learning"

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
    hyperparams_path: Optional[str] = "hyperparams/delayed_iql_jax.yml"
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

    # Delayed IQL
    delayed_update_period: int = 250

    # Standalone actor refit output directory.
    # Refit reuses the shared training schedule fields above:
    #   max_timesteps -> actor-only refit steps
    #   batch_size    -> actor-only refit batch size
    #   eval_freq     -> actor-only refit evaluation interval
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-BIAS"
    group: str = "DelayedIQL-JAX"
    name: str = "DelayedIQL-JAX"
    log_wandb: bool = True
    log_every: int = 500

    checkpoint_freq: int = int(25e3)
    save_final_model: bool = True
    wandb_entity: Optional[str] = None

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
    assert config.checkpoint_freq > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.beta >= 0.0
    assert config.iql_tau >= 0.0 and config.iql_tau <= 1.0
    assert config.delayed_update_period > 0
    if config.actor_dropout is not None:
        assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
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
    config_fields = set(TrainConfig.__dataclass_fields__.keys())
    applied, skipped_unknown, skipped_cli = [], [], []

    for key, raw_value in env_hyperparams.items():
        if key not in config_fields:
            skipped_unknown.append(key)
            continue
        if key in cli_overrides:
            skipped_cli.append(key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(key)

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


class TwinQ(nn.Module):
    hidden_dim: int = 256
    n_hidden: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = QFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden, name="q1")(state, action)
        q2 = QFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden, name="q2")(state, action)
        return q1, q2


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
class DelayedIQLState:
    total_it: jnp.ndarray
    q_params: Any
    q_target_params: Any
    q_delayed_params: Any
    q_opt_state: Any
    v_params: Any
    v_delayed_params: Any
    v_opt_state: Any
    actor_params: Any
    actor_opt_state: Any
    actor_key: jnp.ndarray


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any
    key: jnp.ndarray


class DelayedIQLJAX:
    """Delayed Implicit Q-Learning in JAX/Flax.

    Compared with vanilla IQL, the asymmetric value-loss weight is computed
    from delayed Q/V networks:

        delayed_adv = min(Q1_delayed(s, a), Q2_delayed(s, a)) - V_delayed(s)
        weight = |tau - 1[delayed_adv < 0]|
        value_loss = E[stop(weight) * (stop(min(Q_target(s, a))) - V(s))^2]

    Q_delayed is periodically refreshed from Q_target.
    V_delayed is periodically refreshed from online V.
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
        delayed_update_period: int = 250,
        iql_deterministic: bool = False,
        actor_dropout: Optional[float] = None,
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
        self.delayed_update_period = int(delayed_update_period)
        self.iql_deterministic = iql_deterministic
        self.actor_dropout = actor_dropout
        self.device = device if device is not None else jax.devices()[0]

        if self.delayed_update_period <= 0:
            raise ValueError("delayed_update_period must be > 0")

        if iql_deterministic:
            self.actor_def = DeterministicPolicy(action_dim=action_dim, dropout=actor_dropout)
        else:
            self.actor_def = GaussianPolicy(action_dim=action_dim, dropout=actor_dropout)
        self.q_def = TwinQ()
        self.v_def = ValueFunction()

        self.q_tx = optax.adam(qf_lr)
        self.v_tx = optax.adam(vf_lr)
        actor_lr_schedule = optax.cosine_decay_schedule(
            init_value=actor_lr,
            decay_steps=self.max_steps,
            alpha=0.0,
        )
        self.actor_tx = optax.adam(actor_lr_schedule)

        key = jax.random.PRNGKey(seed)
        key_actor, key_q, key_v, actor_key = jax.random.split(key, 4)
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
        q_params = self.q_def.init(key_q, dummy_state, dummy_action)["params"]
        v_params = self.v_def.init(key_v, dummy_state)["params"]

        self.initial_actor_params = copy.deepcopy(actor_params)
        self.initial_actor_opt_state = self.actor_tx.init(actor_params)
        self.initial_actor_key = actor_key

        self.state = DelayedIQLState(
            total_it=jnp.asarray(0, dtype=jnp.int32),
            q_params=q_params,
            q_target_params=copy.deepcopy(q_params),
            q_delayed_params=copy.deepcopy(q_params),
            q_opt_state=self.q_tx.init(q_params),
            v_params=v_params,
            v_delayed_params=copy.deepcopy(v_params),
            v_opt_state=self.v_tx.init(v_params),
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
        delayed_update_period = self.delayed_update_period
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
        def train_step(state: DelayedIQLState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            # Q backup and actor advantage follow vanilla IQL.
            # Only the asymmetric value-loss weight is delayed.
            next_v = v_apply({"params": state.v_params}, next_observations)
            target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
            target_q1, target_q2 = q_apply({"params": state.q_target_params}, observations, actions)
            target_q_for_v = jnp.minimum(target_q1, target_q2)
            old_v = v_apply({"params": state.v_params}, observations)
            adv = target_q_for_v - old_v
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

            # Delayed value-loss weight.
            delayed_q1, delayed_q2 = q_apply({"params": state.q_delayed_params}, observations, actions)
            delayed_q_for_weight = jnp.minimum(delayed_q1, delayed_q2)
            delayed_v_for_weight = v_apply({"params": state.v_delayed_params}, observations)
            delayed_adv_for_weight = delayed_q_for_weight - delayed_v_for_weight
            delayed_value_weight = jnp.abs(
                iql_tau - (delayed_adv_for_weight < 0.0).astype(jnp.float32)
            )

            def v_loss_fn(v_params):
                v = v_apply({"params": v_params}, observations)
                value_adv = jax.lax.stop_gradient(target_q_for_v) - v
                value_loss = jnp.mean(jax.lax.stop_gradient(delayed_value_weight) * value_adv ** 2)
                return value_loss, v

            (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            def q_loss_fn(q_params):
                q1, q2 = q_apply({"params": q_params}, observations, actions)
                target = jax.lax.stop_gradient(target_q_for_backup)
                q_loss = 0.5 * (jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2))
                return q_loss, (q1, q2)

            (q_loss, (q1, q2)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
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
                return q_target_params_, v_params_

            def keep_delayed(carry):
                _q_target_params_, _v_params_ = carry
                return state.q_delayed_params, state.v_delayed_params

            q_delayed_params, v_delayed_params = jax.lax.cond(
                should_update_delayed,
                update_delayed,
                keep_delayed,
                operand=(q_target_params, v_params),
            )

            new_state = DelayedIQLState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_delayed_params=q_delayed_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_delayed_params=v_delayed_params,
                v_opt_state=v_opt_state,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                actor_key=actor_key,
            )

            log_dict = {
                "q_loss": q_loss,
                "q1_mean": jnp.mean(q1),
                "q2_mean": jnp.mean(q2),
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
                "delayed_adv_mean": jnp.mean(delayed_adv_for_weight),
                "delayed_adv_min": jnp.min(delayed_adv_for_weight),
                "delayed_adv_max": jnp.max(delayed_adv_for_weight),
                "delayed_weight_mean": jnp.mean(delayed_value_weight),
                "delayed_weight_min": jnp.min(delayed_value_weight),
                "delayed_weight_max": jnp.max(delayed_value_weight),
                "delayed_negative_adv_frac": jnp.mean((delayed_adv_for_weight < 0.0).astype(jnp.float32)),
                "delayed_update": should_update_delayed.astype(jnp.float32),
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
        def actor_refit_step(actor_state: ActorState, delayed_iql_state: DelayedIQLState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]

            # Actor refit uses the frozen trained Q_target and V, same as vanilla IQL AWBC.
            q1, q2 = q_apply({"params": delayed_iql_state.q_target_params}, observations, actions)
            target_q = jnp.minimum(q1, q2)
            v = v_apply({"params": delayed_iql_state.v_params}, observations)
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
        if steps <= 0:
            return actor_state, refit_log

        best_normalized_score_mean = -np.inf
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

                refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
                refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
                refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
                refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
                refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
                refit_log[f"{prefix}/final_score_std"] = eval_score_std
                refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std

                is_best = normalized_eval_score_mean > best_normalized_score_mean
                if is_best:
                    best_normalized_score_mean = normalized_eval_score_mean
                    refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
                    refit_log[f"{prefix}/best_score_std"] = eval_score_std
                    refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
                    refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std

                save_refit_snapshot(
                    current_actor_state=actor_state,
                    current_refit_log=refit_log,
                    fit_step=fit_step,
                    is_best=is_best,
                )

                print(
                    f"[{prefix}:delayed_iql_awbc] step {fit_step}/{steps}: "
                    f"loss={step_log['loss']:.4f}, bc={step_log['bc_loss']:.4f}, "
                    f"adv={step_log['adv_mean']:.4f}, exp_adv={step_log['exp_adv_mean']:.4f}, "
                    f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
                    f"D4RL_mean={normalized_eval_score_mean:.3f}, "
                    f"D4RL_std={normalized_eval_score_std:.3f}"
                )

        return actor_state, refit_log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "delayed_iql_state": serialization.to_state_dict(self.state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
            "iql_deterministic": self.iql_deterministic,
            "delayed_update_period": self.delayed_update_period,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state = serialization.from_state_dict(self.state, state_dict["delayed_iql_state"])
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
    """Return (run_dir, checkpoint_path) for a saved DelayedIQL-JAX checkpoint.

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


def _train_impl(config: TrainConfig):
    config = apply_env_hyperparams(config)
    refit_only = config.mode == "refit"
    if refit_only and config.load_model == "":
        raise ValueError("refit mode requires --load_model")
    if not refit_only:
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
            checkpoint_type="delayed_iql_jax_training_progress",
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


    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    run_mode_name = "Actor refit" if refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
    print(f"delayed_update_period={config.delayed_update_period}")
    print("---------------------------------------")

    trainer = DelayedIQLJAX(
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
        delayed_update_period=config.delayed_update_period,
        iql_deterministic=config.iql_deterministic,
        actor_dropout=config.actor_dropout,
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
                eval_scores = trainer.eval_actor(
                    env,
                    trainer.state.actor_params,
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
                upsert_eval_log(eval_logs, eval_log)
    
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
