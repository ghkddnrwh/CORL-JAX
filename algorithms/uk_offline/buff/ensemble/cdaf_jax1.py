
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

ALGORITHM_NAME = "CDAF"
ALGORITHM_FULL_NAME = "Conservative Delayed Advantage Filtering"
IQL_ACTOR_METHODS = ("iql", "iql_awbc")
ACTOR_FIT_METHODS = ("td3_bc", "weighted_bc", "td3_weighted_bc", *IQL_ACTOR_METHODS)


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

    # Ensemble double Q/V learning
    # Each Q_i is bootstrapped from V_i. Each V_i is regressed to Q_i,
    # but its CDAF filtering weight is computed from a delayed Q_j/V_j selected
    # by a deterministic cycle-shift pairing.
    num_ensembles: int = 1

    # Actor fitting during evaluation/refit.
    # td3_bc preserves the original CDAF evaluation protocol.
    # weighted_bc is fully in-sample: weighted behavior cloning from dataset actions.
    # td3_weighted_bc keeps the TD3+BC Q-improvement term, but replaces the plain
    # BC regularizer with advantage-weighted BC from dataset actions.
    # iql and iql_awbc use IQL-style advantage-weighted BC from dataset actions.
    # In all dataset-action weighted modes (weighted_bc, td3_weighted_bc, iql,
    # iql_awbc), the behavior-cloning weights are computed from delayed
    # ensemble-mean advantages: mean_i Q_i_delayed(s,a) - mean_i V_i_delayed(s).
    # For iql/iql_awbc, policy_weight_exponent is IQL beta and policy_weight_clip is EXP_ADV_MAX.
    actor_fit_method: str = "td3_bc"  # one of: td3_bc, weighted_bc, td3_weighted_bc, iql, iql_awbc
    policy_weight_exponent: float = 1.0
    policy_weight_clip: float = 20.0

    # TD3+BC actor fitting during evaluation
    actor_lr: float = 3e-4
    eval_actor_steps: int = 1000
    eval_actor_batch_size: int = 256
    eval_actor_eval_freq: int = 2000
    reset_actor_on_eval: bool = False
    alpha: float = 2.5
    bc_coef: float = 1.0

    # Standalone actor refit from a saved checkpoint.
    # Used when --load_model is provided with --max_timesteps 0.
    refit_actor_steps: int = 50000
    actor_refit_dir_name: str = "actor_refit"

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
    config.project = "ORL-BIAS"
    config.group = f"{ALGORITHM_NAME}-JAX"
    config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


def validate_config(config: TrainConfig) -> None:
    assert config.min_weight_exponent >= 0.0, "min_weight_exponent must be >= 0"
    assert config.max_weight_exponent >= 0.0, "max_weight_exponent must be >= 0"
    assert config.beta_min >= 0.0, "beta_min must be >= 0"
    assert config.num_ensembles >= 1, "num_ensembles must be >= 1"
    assert config.policy_weight_exponent >= 0.0, "policy_weight_exponent must be >= 0"
    assert config.policy_weight_clip > 0.0, "policy_weight_clip must be > 0"
    assert config.actor_fit_method in ACTOR_FIT_METHODS, (
        "actor_fit_method must be one of: "
        f"{', '.join(ACTOR_FIT_METHODS)}"
    )
    assert config.max_weight_exponent >= config.min_weight_exponent
    assert config.delayed_update_period > 0
    assert config.bc_coef >= 0.0
    assert config.batch_size > 0
    assert config.eval_actor_batch_size > 0
    assert config.eval_actor_eval_freq > 0
    assert config.refit_actor_steps >= 0
    assert config.actor_refit_dir_name != ""


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


def stack_ensemble_params(params_list: List[Any]) -> Any:
    """Stack a list of Flax parameter PyTrees along a leading ensemble axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


def cycle_filter_indices(num_ensembles: int, shift: Union[int, jnp.ndarray]) -> jnp.ndarray:
    """Return deterministic cycle-shift mapping i -> j for cross-ensemble filtering.

    n=1: [0]
    n>=2: [shift, shift+1, ..., shift-1] mod n.

    For n>=2 and shift in {1, ..., n-1}, this is a one-to-one mapping
    and never maps an ensemble member to itself.
    """
    if num_ensembles == 1:
        return jnp.zeros((1,), dtype=jnp.int32)
    indices = jnp.arange(num_ensembles, dtype=jnp.int32)
    return (indices + jnp.asarray(shift, dtype=jnp.int32)) % jnp.asarray(num_ensembles, dtype=jnp.int32)


def initial_filter_indices(num_ensembles: int) -> jnp.ndarray:
    """Initial cycle-shift mapping."""
    return cycle_filter_indices(num_ensembles, shift=1)


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
    filter_indices: jnp.ndarray


@struct.dataclass
class ActorState:
    params: Any
    opt_state: Any


class CDAFJAX:
    """Conservative Delayed Advantage Filtering (CDAF) for offline RL.

    Main training learns an ensemble of Q_i/V_i pairs with delayed
    cross-ensemble negative-advantage filtering. Policy extraction uses
    ensemble-mean Q/V values with TD3+BC-, weighted-BC-, or IQL-style actor fitting.
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
        num_ensembles: int = 1,
        actor_fit_method: str = "td3_bc",
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
        self.num_ensembles = int(num_ensembles)
        if self.num_ensembles < 1:
            raise ValueError("num_ensembles must be >= 1")
        if actor_fit_method not in ACTOR_FIT_METHODS:
            raise ValueError(f"actor_fit_method must be one of: {', '.join(ACTOR_FIT_METHODS)}")
        self.actor_fit_method = actor_fit_method
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
        init_keys = jax.random.split(key, 1 + 2 * self.num_ensembles)
        key_actor = init_keys[0]
        q_keys = init_keys[1 : 1 + self.num_ensembles]
        v_keys = init_keys[1 + self.num_ensembles :]
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state)["params"]
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
            filter_indices=initial_filter_indices(self.num_ensembles),
        )
        self.actor_state = ActorState(
            params=actor_params,
            opt_state=self.initial_actor_opt_state,
        )

        self.state = tree_to_device(self.state, self.device)
        self.actor_state = tree_to_device(self.actor_state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)

        self._train_step = self._build_train_step()
        self._actor_fit_step = self._build_actor_fit_step()

    def _build_train_step(self):
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        discount = self.discount
        tau = self.tau
        delayed_update_period = self.delayed_update_period
        min_weight_exponent = self.min_weight_exponent
        max_weight_exponent = self.max_weight_exponent
        weight_logit_clip = self.weight_logit_clip
        beta_min = self.beta_min
        max_steps = self.max_steps
        num_ensembles = self.num_ensembles
        ensemble_indices = jnp.arange(num_ensembles, dtype=jnp.int32)

        def apply_q_ensemble(params: Any, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: q_apply({"params": p}, states, actions))(params)

        def apply_v_ensemble(params: Any, states: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: v_apply({"params": p}, states))(params)

        def cycle_shift_for_update(total_it_: jnp.ndarray) -> jnp.ndarray:
            """Deterministic cyclic i -> j filtering assignment.

            n=1: [0]
            n=2: always shift by 1, i.e. [1, 0]
            n>=3: shift cycles through 1, 2, ..., n-1 whenever delayed
                  networks are refreshed. This keeps a 1-to-1 mapping and
                  never uses j == i.
            """
            if num_ensembles == 1:
                return jnp.zeros((1,), dtype=jnp.int32)
            delayed_round = total_it_ // jnp.asarray(delayed_update_period, dtype=jnp.int32)
            shift = jnp.asarray(1, dtype=jnp.int32) + (delayed_round % jnp.asarray(num_ensembles - 1, dtype=jnp.int32))
            return (ensemble_indices + shift) % jnp.asarray(num_ensembles, dtype=jnp.int32)

        @jax.jit
        def train_step(state: CDAFState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            def q_loss_fn(q_params):
                # Q_i is always bootstrapped from V_i.
                next_v = apply_v_ensemble(state.v_target_params, next_observations)  # [N, B]
                target_q = rewards[None, :] + (1.0 - dones[None, :]) * discount * next_v
                q = apply_q_ensemble(q_params, observations, actions)  # [N, B]
                q_loss = jnp.mean((q - jax.lax.stop_gradient(target_q)) ** 2)
                return q_loss, (q, target_q)

            (q_loss, (q, target_q)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
            q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
            q_params = optax.apply_updates(state.q_params, q_updates)

            progress = jnp.minimum(total_it.astype(jnp.float32) / jnp.maximum(float(max_steps), 1.0), 1.0)
            exponent = min_weight_exponent + (max_weight_exponent - min_weight_exponent) * progress

            # Compute delayed advantages for all ensemble members, then gather the paired
            # filtering network j for each update target i.
            delayed_q_all = apply_q_ensemble(state.q_delayed_params, observations, actions)  # [N, B]
            delayed_v_all = apply_v_ensemble(state.v_delayed_params, observations)  # [N, B]
            raw_delayed_adv_all = delayed_q_all - delayed_v_all
            raw_delayed_adv = raw_delayed_adv_all[state.filter_indices, :]  # beta_i uses paired j
            delayed_adv = jnp.clip(raw_delayed_adv, -weight_logit_clip, weight_logit_clip)
            beta = jnp.where(delayed_adv < 0.0, jnp.exp(exponent * delayed_adv), jnp.ones_like(delayed_adv))
            beta = jnp.maximum(beta, beta_min)

            def v_loss_fn(v_params):
                # V_i is still regressed to its own Q_i; only the filtering ratio comes from j.
                target_v_q = apply_q_ensemble(state.q_target_params, observations, actions)  # [N, B]
                v = apply_v_ensemble(v_params, observations)  # [N, B]
                value_residual = v - jax.lax.stop_gradient(target_v_q)
                value_loss = jnp.mean(jax.lax.stop_gradient(beta) * value_residual ** 2)
                return value_loss, (v, target_v_q)

            (value_loss, (v, target_v_q)), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            q_target_params = soft_update(q_params, state.q_target_params, tau)
            v_target_params = soft_update(v_params, state.v_target_params, tau)

            should_update_delayed = (total_it % delayed_update_period) == 0

            def delayed_update(carry):
                q_target_params_, v_target_params_ = carry
                filter_indices_ = cycle_shift_for_update(total_it)
                return q_target_params_, v_target_params_, filter_indices_

            def keep_delayed(carry):
                _, _ = carry
                return state.q_delayed_params, state.v_delayed_params, state.filter_indices

            q_delayed_params, v_delayed_params, filter_indices = jax.lax.cond(
                should_update_delayed,
                delayed_update,
                keep_delayed,
                operand=(q_target_params, v_target_params),
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
                filter_indices=filter_indices,
            )

            log_dict = {
                "q_loss": q_loss,
                "q_mean": jnp.mean(q),
                "target_q_mean": jnp.mean(target_q),
                "value_loss": value_loss,
                "v_mean": jnp.mean(v),
                "target_v_q_mean": jnp.mean(target_v_q),
                "beta_mean": jnp.mean(beta),
                "beta_min": jnp.min(beta),
                "beta_max": jnp.max(beta),
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
                "ensemble_size": jnp.asarray(num_ensembles, dtype=jnp.float32),
                "filter_index_mean": jnp.mean(filter_indices.astype(jnp.float32)),
                "filter_shift": jnp.where(
                    jnp.asarray(num_ensembles, dtype=jnp.int32) > 1,
                    filter_indices[0],
                    jnp.asarray(0, dtype=jnp.int32),
                ).astype(jnp.float32),
            }
            return new_state, log_dict

        return train_step

    def _build_actor_fit_step(self):
        actor_apply = self.actor_def.apply
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        actor_tx = self.actor_tx
        alpha = self.alpha
        bc_coef = self.bc_coef
        actor_fit_method = self.actor_fit_method
        policy_weight_exponent = self.policy_weight_exponent
        policy_weight_clip = self.policy_weight_clip
        weight_logit_clip = self.weight_logit_clip

        def apply_q_ensemble(params: Any, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: q_apply({"params": p}, states, actions))(params)

        def apply_v_ensemble(params: Any, states: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: v_apply({"params": p}, states))(params)

        @jax.jit
        def actor_fit_step(
            actor_state: ActorState,
            q_params: Any,
            v_params: Any,
            q_delayed_params: Any,
            v_delayed_params: Any,
            batch: TensorBatch,
        ):
            observations = batch["observations"]
            actions = batch["actions"]

            def actor_loss_fn(actor_params):
                pi = actor_apply({"params": actor_params}, observations)
                bc_error = (pi - actions) ** 2
                bc_per_sample = (
                    jnp.sum(bc_error, axis=-1)
                    if actor_fit_method in IQL_ACTOR_METHODS
                    else jnp.mean(bc_error, axis=-1)
                )

                # Dataset-action weights use delayed ensemble-mean advantages in
                # every weighted actor mode. This decouples the BC weights from
                # the current critic update while keeping TD3 Q-improvement terms
                # on the current/original Q ensemble.
                if actor_fit_method in IQL_ACTOR_METHODS:
                    data_q_all = apply_q_ensemble(q_delayed_params, observations, actions)
                    data_v_all = apply_v_ensemble(v_delayed_params, observations)
                    data_q = jnp.mean(data_q_all, axis=0)
                    data_v = jnp.mean(data_v_all, axis=0)
                    data_adv = data_q - data_v
                    policy_weight = jnp.minimum(
                        jnp.exp(policy_weight_exponent * jax.lax.stop_gradient(data_adv)),
                        policy_weight_clip,
                    )
                    weighted_bc_loss = jnp.mean(jax.lax.stop_gradient(policy_weight) * bc_per_sample)
                    weight_mean = jnp.mean(policy_weight)
                    weight_max = jnp.max(policy_weight)
                elif actor_fit_method in ("weighted_bc", "td3_weighted_bc"):
                    data_q_all = apply_q_ensemble(q_delayed_params, observations, actions)
                    data_v_all = apply_v_ensemble(v_delayed_params, observations)
                    data_q = jnp.mean(data_q_all, axis=0)
                    data_v = jnp.mean(data_v_all, axis=0)
                    data_adv = jnp.clip(data_q - data_v, -weight_logit_clip, weight_logit_clip)
                    policy_weight = jnp.exp(policy_weight_exponent * data_adv)
                    policy_weight = policy_weight / jnp.maximum(jnp.mean(policy_weight), 1e-6)
                    policy_weight = jnp.minimum(policy_weight, policy_weight_clip)
                    weighted_bc_loss = jnp.mean(jax.lax.stop_gradient(policy_weight) * bc_per_sample)
                    weight_mean = jnp.mean(policy_weight)
                    weight_max = jnp.max(policy_weight)
                else:
                    data_q = jnp.zeros_like(bc_per_sample)
                    weighted_bc_loss = jnp.mean(bc_per_sample)
                    weight_mean = jnp.asarray(1.0)
                    weight_max = jnp.asarray(1.0)

                bc_loss = jnp.mean(bc_per_sample)

                if actor_fit_method == "weighted_bc" or actor_fit_method in IQL_ACTOR_METHODS:
                    actor_loss = weighted_bc_loss
                    q_for_log = data_q
                    lmbda = jnp.asarray(0.0)
                else:
                    q_pi_all = apply_q_ensemble(q_params, observations, pi)
                    q_pi = jnp.mean(q_pi_all, axis=0)
                    lmbda = jax.lax.stop_gradient(alpha / jnp.maximum(jnp.mean(jnp.abs(q_pi)), 1e-6))
                    # td3_bc:          -lambda * mean_i Q_i_current(s, pi(s)) + bc_coef * plain BC
                    # td3_weighted_bc: -lambda * mean_i Q_i_current(s, pi(s)) + bc_coef * delayed-weighted BC
                    bc_regularizer = jnp.where(
                        actor_fit_method == "td3_weighted_bc",
                        weighted_bc_loss,
                        bc_loss,
                    )
                    actor_loss = -lmbda * jnp.mean(q_pi) + bc_coef * bc_regularizer
                    q_for_log = q_pi
                return actor_loss, (q_for_log, lmbda, bc_loss, weighted_bc_loss, weight_mean, weight_max)

            (actor_loss, (q, lmbda, bc_loss, weighted_bc_loss, weight_mean, weight_max)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
            updates, opt_state = actor_tx.update(actor_grads, actor_state.opt_state, actor_state.params)
            params = optax.apply_updates(actor_state.params, updates)
            new_actor_state = ActorState(params=params, opt_state=opt_state)
            log_dict = {
                "loss": actor_loss,
                "q_mean": jnp.mean(q),
                "lambda": lmbda,
                "bc_loss": bc_loss,
                "weighted_bc_loss": weighted_bc_loss,
                "bc_coef": jnp.asarray(bc_coef),
                "policy_weight_mean": weight_mean,
                "policy_weight_max": weight_max,
            }
            return new_actor_state, log_dict

        return actor_fit_step

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.state, log_dict = self._train_step(self.state, batch)
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def reset_actor(self):
        self.actor_state = ActorState(
            params=self.initial_actor_params,
            opt_state=self.initial_actor_opt_state,
        )
        self.actor_state = tree_to_device(self.actor_state, self.device)

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
        action = self.actor_def.apply({"params": actor_params}, state_jnp)
        return np.asarray(jax.device_get(action))[0]

    def eval_actor(self, env: gym.Env, actor_params: Any, n_episodes: int, seed: int) -> np.ndarray:
        env.seed(seed)
        episode_rewards = []
        for ep in range(n_episodes):
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
        prefix: str = "fit_actor",
        save_dir: Optional[Union[str, Path]] = None,
        save_metadata: Optional[Dict[str, Any]] = None,
        save_wandb: bool = False,
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
        if steps <= 0:
            return actor_state, eval_fit_log

        best_normalized_score_mean = -np.inf
        # Actor Q-improvement terms use the current/original Q ensemble.
        # Dataset-action weights in weighted_bc, td3_weighted_bc, iql, and
        # iql_awbc use the delayed Q/V ensemble.
        q_params = self.state.q_params
        v_params = self.state.v_params
        q_delayed_params = self.state.q_delayed_params
        v_delayed_params = self.state.v_delayed_params
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
        save_metadata = {} if save_metadata is None else dict(save_metadata)

        for fit_step in range(1, steps + 1):
            batch = replay_buffer.sample(batch_size)
            actor_state, step_log = self._actor_fit_step(
                actor_state,
                q_params,
                v_params,
                q_delayed_params,
                v_delayed_params,
                batch,
            )
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

                    save_pickle(
                        latest_actor_path,
                        serialization.to_state_dict(actor_state.params),
                    )
                    save_logs_npz(
                        [{**save_metadata, **eval_fit_log}],
                        str(fit_eval_logs_path),
                    )

                    if is_best:
                        best_actor_path = save_dir_path / "best_actor.pkl"
                        save_pickle(
                            best_actor_path,
                            serialization.to_state_dict(actor_state.params),
                        )

                    if save_wandb and wandb.run is not None:
                        wandb.save(str(latest_actor_path), policy="now")
                        wandb.save(str(fit_eval_logs_path), policy="now")
                        if is_best:
                            wandb.save(str(best_actor_path), policy="now")

                    print(
                        f"[{prefix}:{self.actor_fit_method}] saved intermediate refit outputs to "
                        f"{save_dir_path}"
                    )

                print(
                    f"[{prefix}:{self.actor_fit_method}] step {fit_step}/{steps}: "
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
            "cdaf_state": serialization.to_state_dict(self.state),
            "actor_state": serialization.to_state_dict(self.actor_state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state = serialization.from_state_dict(self.state, state_dict["cdaf_state"])
        self.actor_state = serialization.from_state_dict(self.actor_state, state_dict["actor_state"])
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
    refit_only = config.load_model != "" and int(config.max_timesteps) <= 0
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
    print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
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
        num_ensembles=config.num_ensembles,
        actor_fit_method=config.actor_fit_method,
        policy_weight_exponent=config.policy_weight_exponent,
        policy_weight_clip=config.policy_weight_clip,
        alpha=config.alpha,
        bc_coef=config.bc_coef,
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
            raise ValueError("refit_only mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        fresh_actor_state = ActorState(
            params=copy.deepcopy(trainer.initial_actor_params),
            opt_state=copy.deepcopy(trainer.initial_actor_opt_state),
        )
        fresh_actor_state = tree_to_device(fresh_actor_state, jax_device)

        refit_actor_state, refit_log = trainer.fit_actor(
            replay_buffer=replay_buffer,
            actor_state=fresh_actor_state,
            steps=config.refit_actor_steps,
            batch_size=config.eval_actor_batch_size,
            eval_env=env,
            eval_episodes=config.n_episodes,
            eval_seed=config.seed,
            eval_interval=config.eval_actor_eval_freq,
            prefix="actor_refit",
            save_dir=actor_refit_dir,
            save_metadata={"loaded_checkpoint": str(loaded_run_dir / "checkpoint.pkl")},
            save_wandb=config.log_wandb,
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
            if config.reset_actor_on_eval:
                trainer.reset_actor()

            actor_state, eval_fit_log = trainer.fit_actor(
                replay_buffer=replay_buffer,
                actor_state=trainer.actor_state,
                steps=config.eval_actor_steps,
                batch_size=config.eval_actor_batch_size,
                eval_env=env,
                eval_episodes=config.n_episodes,
                eval_seed=config.seed,
                eval_interval=config.eval_actor_eval_freq,
                prefix="fit_actor",
            )
            trainer.actor_state = actor_state

            eval_log: Dict[str, Any] = {
                "timestep": int(t + 1),
                "eval/reward_mean": eval_fit_log["fit_actor/final_score_mean"],
                "eval/reward_std": eval_fit_log["fit_actor/final_score_std"],
                "eval/normalized_score_mean": eval_fit_log["fit_actor/final_d4rl_normalized_score_mean"],
                "eval/normalized_score_std": eval_fit_log["fit_actor/final_d4rl_normalized_score_std"],
            }
            eval_log.update(eval_fit_log)
            eval_logs.append(eval_log.copy())

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