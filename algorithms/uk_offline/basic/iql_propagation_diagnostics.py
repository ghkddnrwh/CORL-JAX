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

ALGORITHM_NAME = "IQL-PD-V7"
ALGORITHM_FULL_NAME = "IQL Propagation Diagnostics v7"

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

    # Finite-difference B diagnostics. diag_fd_epsilon is interpreted as a relative
    # perturbation scale when diag_fd_relative_scale=True:
    #   delta = diag_fd_epsilon * max(1, RMS(Bellman target)).
    # Central differences and two additional scales are logged to detect float32
    # cancellation/nonlinearity rather than silently treating a zero response as small B.
    diag_fd_epsilon: float = 1e-3
    diag_fd_relative_scale: bool = True
    diag_fd_small_multiplier: float = 0.25
    diag_fd_large_multiplier: float = 4.0
    diag_num_basis_directions: int = 4

    # Conditional-head calibration and frozen-Q V-solver correctability diagnostics.
    # The multi-step V updates are virtual: they are evaluated but never committed.
    diag_num_calibration_bins: int = 8
    diag_enable_multistep_v_correctability: bool = True
    diag_v_correctability_short_steps: int = 5
    diag_v_correctability_long_steps: int = 20

    # Independent non-parametric conditional references.  These use nearby probe states
    # rather than the learned diagnostic head, so they are useful for detecting head failure.
    # They remain approximate references (state-neighborhood smoothing), not exact expectiles.
    diag_knn_k: int = 16

    # Kernel conditional-moment diagnostics.  These test the expectile FOC
    #   E[psi_tau(Q-V) | s] = 0
    # without fitting another residual network.  An RBF kernel with a median-distance
    # bandwidth is used both for a maximum-moment-style scalar violation score and for
    # a smooth state-wise witness that can be propagated through B.
    diag_enable_kernel_residual: bool = True
    diag_kernel_bandwidth_scale: float = 1.0
    diag_kernel_min_bandwidth: float = 1e-3

    # V5 local-kernel calibration.  V4's median-distance bandwidth often averaged
    # almost the whole probe batch.  The adaptive bandwidth below is chosen to
    # target a specified effective-neighbor count so localized conditional moments
    # are not silently washed out by global smoothing.
    diag_kernel_target_neff: float = 32.0
    diag_kernel_bandwidth_search_iters: int = 24

    # Ground-truth virtual residual-injection calibration.  This does not modify
    # training: V(s) is shifted only inside a diagnostic probe and the conditional
    # estimators are asked to recover that known displacement.
    diag_enable_residual_injection_calibration: bool = True
    diag_residual_calibration_every: int = 5000
    diag_residual_injection_fraction: float = 1e-2
    diag_residual_injection_small_multiplier: float = 0.25
    diag_residual_injection_large_multiplier: float = 4.0

    # V7 conditioning-aware residual diagnostics.  Raw FOC moments shrink with kappa,
    # so high-expectile residuals can be hard to detect even when the implied value
    # displacement -G/kappa is material.  These diagnostics studentize the local
    # displacement and add a cross-fitted kappa-normalized moment-energy estimate.
    diag_enable_conditioned_residual: bool = True

    # V7 local-action empirical expectile reference.  A larger fixed reference bank is
    # sampled independently of training.  For each probe state, actions from nearby
    # dataset states are re-evaluated at the *same query state* to approximate
    # a behavior-conditioned action distribution without training a behavior model.
    diag_action_ref_batch_size: int = 2048
    diag_action_ref_k: int = 32
    diag_action_ref_bisection_iters: int = 24

    # Full Bellman-loop propagation diagnostic.  Unlike B^(H), which repeatedly
    # fits a frozen target, this virtually re-runs Q/V/target-network updates on a
    # fixed future minibatch sequence after a one-time target perturbation.
    diag_enable_loop_propagation: bool = True
    diag_loop_every: int = 10000
    diag_loop_short_steps: int = 5
    diag_loop_long_steps: int = 20
    # V6: in addition to the one-shot impulse response, replay the baseline TD/lag
    # source direction at every virtual Bellman step.  This tests persistent forcing
    # accumulation in epsilon_{k+1}=M_k epsilon_k+d_k even when impulse gain < 1.
    diag_enable_forced_loop: bool = True

    # V7 actual-amplitude forced response.  V6 normalized every source direction to unit
    # RMS, which removes the source-growth effect that dominated natural collapse.  V7
    # replays the baseline source sequence with its original per-step amplitude, using a
    # single small dimensionless perturbation scale chosen to stay in the local regime.
    diag_enable_actual_source_forcing: bool = True

    # Finite-amplitude source diagnostics: evaluate the real Adam response at 25%, 50%,
    # and 100% of a safely clipped current source amplitude.
    diag_actual_source_cap_fraction: float = 0.05

    # Expensive finite-horizon critic-fitting response.  It repeats virtual Adam critic
    # updates with the same frozen batch/target, never committing them to training.
    diag_enable_horizon_b: bool = True
    diag_horizon_every: int = 5000
    diag_horizon_short_steps: int = 5
    diag_horizon_long_steps: int = 20

    # Expensive local full-fit approximation
    # B ~= J_Z (J_X^T J_X / m + lambda I)^-1 J_X^T / m.
    # It is evaluated less frequently than the one-step diagnostics.
    diag_enable_ridge_b: bool = True
    diag_ridge_every: int = 5000
    diag_ridge_lambda: float = 1e-2
    diag_ridge_cg_iters: int = 100
    diag_ridge_cg_tol: float = 1e-5

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
        assert config.diag_fd_small_multiplier > 0.0
        assert config.diag_fd_large_multiplier > config.diag_fd_small_multiplier
        assert config.diag_num_basis_directions > 0
        assert config.diag_num_calibration_bins > 1
        assert config.diag_probe_batch_size >= config.diag_num_calibration_bins
        assert config.diag_v_correctability_short_steps >= 1
        assert config.diag_v_correctability_long_steps >= config.diag_v_correctability_short_steps
        assert config.diag_knn_k >= 2
        assert config.diag_knn_k < config.diag_probe_batch_size
        assert config.diag_kernel_bandwidth_scale > 0.0
        assert config.diag_kernel_min_bandwidth > 0.0
        assert config.diag_kernel_target_neff >= 2.0
        assert config.diag_kernel_target_neff < config.diag_probe_batch_size
        assert config.diag_kernel_bandwidth_search_iters >= 4
        assert config.diag_residual_calibration_every > 0
        assert config.diag_residual_injection_fraction > 0.0
        assert config.diag_residual_injection_small_multiplier > 0.0
        assert config.diag_residual_injection_large_multiplier > 1.0
        assert config.diag_residual_injection_small_multiplier < 1.0
        assert config.diag_action_ref_batch_size > 0
        assert config.diag_action_ref_k >= 2
        assert config.diag_action_ref_k < config.diag_action_ref_batch_size
        assert config.diag_action_ref_bisection_iters >= 8
        assert config.diag_loop_every > 0
        assert config.diag_loop_short_steps >= 1
        assert config.diag_loop_long_steps >= config.diag_loop_short_steps
        assert config.diag_actual_source_cap_fraction > 0.0
        assert config.diag_horizon_every > 0
        assert config.diag_horizon_short_steps >= 1
        assert config.diag_horizon_long_steps >= config.diag_horizon_short_steps
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
        return self._batch_from_indices(indices)

    def sample_with_rng(self, batch_size: int, rng: np.random.Generator) -> TensorBatch:
        """Sample without consuming the global NumPy RNG used by training.

        Stage-1 diagnostics must not silently alter the future training minibatch
        sequence simply because extra probe batches were requested.
        """
        indices = rng.integers(0, min(self._size, self._pointer), size=batch_size)
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: np.ndarray) -> TensorBatch:
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


def sanitize_diagnostic_jsonl(path: Union[str, Path], max_timestep: int) -> None:
    """Deduplicate/prune diagnostic records when resuming from an older safe checkpoint.

    Diagnostics are written more frequently than checkpoints.  Without pruning, a resumed
    run can append the same timesteps again (or retain diagnostics beyond the restored
    trainer state), which corrupts time-series analyses.  Keep the last valid record for
    each timestep not exceeding the restored timestep.
    """
    path = Path(path)
    if not path.exists():
        return
    records_by_step: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                step = int(record.get("timestep", -1))
            except Exception:
                continue
            if 0 <= step <= int(max_timestep):
                records_by_step[step] = record
    with open(path, "w", encoding="utf-8") as f:
        for step in sorted(records_by_step):
            f.write(json.dumps(records_by_step[step], allow_nan=True) + "\n")


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
        diag_fd_relative_scale: bool = True,
        diag_fd_small_multiplier: float = 0.25,
        diag_fd_large_multiplier: float = 4.0,
        diag_num_basis_directions: int = 4,
        diag_num_calibration_bins: int = 8,
        diag_enable_multistep_v_correctability: bool = True,
        diag_v_correctability_short_steps: int = 5,
        diag_v_correctability_long_steps: int = 20,
        diag_knn_k: int = 16,
        diag_enable_kernel_residual: bool = True,
        diag_kernel_bandwidth_scale: float = 1.0,
        diag_kernel_min_bandwidth: float = 1e-3,
        diag_kernel_target_neff: float = 32.0,
        diag_kernel_bandwidth_search_iters: int = 24,
        diag_enable_residual_injection_calibration: bool = True,
        diag_residual_injection_fraction: float = 1e-2,
        diag_residual_injection_small_multiplier: float = 0.25,
        diag_residual_injection_large_multiplier: float = 4.0,
        diag_enable_conditioned_residual: bool = True,
        diag_action_ref_k: int = 32,
        diag_action_ref_bisection_iters: int = 24,
        diag_loop_short_steps: int = 5,
        diag_loop_long_steps: int = 20,
        diag_enable_forced_loop: bool = True,
        diag_enable_actual_source_forcing: bool = True,
        diag_actual_source_cap_fraction: float = 0.05,
        diag_horizon_short_steps: int = 5,
        diag_horizon_long_steps: int = 20,
        diag_ridge_lambda: float = 1e-2,
        diag_ridge_cg_iters: int = 100,
        diag_ridge_cg_tol: float = 1e-5,
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
        self.diag_fd_relative_scale = bool(diag_fd_relative_scale)
        self.diag_fd_small_multiplier = float(diag_fd_small_multiplier)
        self.diag_fd_large_multiplier = float(diag_fd_large_multiplier)
        self.diag_num_basis_directions = int(diag_num_basis_directions)
        self.diag_num_calibration_bins = int(diag_num_calibration_bins)
        self.diag_enable_multistep_v_correctability = bool(diag_enable_multistep_v_correctability)
        self.diag_v_correctability_short_steps = int(diag_v_correctability_short_steps)
        self.diag_v_correctability_long_steps = int(diag_v_correctability_long_steps)
        self.diag_knn_k = int(diag_knn_k)
        self.diag_enable_kernel_residual = bool(diag_enable_kernel_residual)
        self.diag_kernel_bandwidth_scale = float(diag_kernel_bandwidth_scale)
        self.diag_kernel_min_bandwidth = float(diag_kernel_min_bandwidth)
        self.diag_kernel_target_neff = float(diag_kernel_target_neff)
        self.diag_kernel_bandwidth_search_iters = int(diag_kernel_bandwidth_search_iters)
        self.diag_enable_residual_injection_calibration = bool(diag_enable_residual_injection_calibration)
        self.diag_residual_injection_fraction = float(diag_residual_injection_fraction)
        self.diag_residual_injection_small_multiplier = float(diag_residual_injection_small_multiplier)
        self.diag_residual_injection_large_multiplier = float(diag_residual_injection_large_multiplier)
        self.diag_enable_conditioned_residual = bool(diag_enable_conditioned_residual)
        self.diag_action_ref_k = int(diag_action_ref_k)
        self.diag_action_ref_bisection_iters = int(diag_action_ref_bisection_iters)
        self.diag_loop_short_steps = int(diag_loop_short_steps)
        self.diag_loop_long_steps = int(diag_loop_long_steps)
        self.diag_enable_forced_loop = bool(diag_enable_forced_loop)
        self.diag_enable_actual_source_forcing = bool(diag_enable_actual_source_forcing)
        self.diag_actual_source_cap_fraction = float(diag_actual_source_cap_fraction)
        self.diag_horizon_short_steps = int(diag_horizon_short_steps)
        self.diag_horizon_long_steps = int(diag_horizon_long_steps)
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
        self._residual_calibration_step = self._build_residual_calibration_step()
        self._horizon_diagnostic_step = self._build_horizon_diagnostic_step()
        self._closed_loop_diagnostic_step = self._build_closed_loop_diagnostic_step()
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

        This pass intentionally does not modify the committed IQL update.  It diagnoses:
          * source terms h_V and h_Q^lag,
          * the local target-to-fitted-critic response B_k = D_y F_k(y),
          * numerical validity of several B estimators,
          * source-direction-specific gain/alignment rather than only a global/random gain,
          * target-network lag growth,
          * calibration and correctability of the expectile-residual estimator.

        The principal local channels are
            h_V ~= -gamma * G(s') / kappa(s')
            h_Q^lag ~= gamma * E[w(A) (Q_online-Q_target) | s'] / kappa(s')
        followed by B_k h.  h_Q^lag remains an observable proxy, not the unobservable
        theoretical critic perturbation channel.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        diag_apply = self.diag_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        discount = self.discount
        iql_tau = self.iql_tau
        beta = self.beta
        iql_deterministic = self.iql_deterministic
        actor_apply_fn = self.actor_def.apply
        num_critics = self.num_critics
        qf_lr = self.qf_lr
        fd_epsilon = self.diag_fd_epsilon
        fd_relative_scale = self.diag_fd_relative_scale
        fd_small_multiplier = self.diag_fd_small_multiplier
        fd_large_multiplier = self.diag_fd_large_multiplier
        num_basis = self.diag_num_basis_directions
        num_calibration_bins = self.diag_num_calibration_bins
        enable_multistep_v = self.diag_enable_multistep_v_correctability
        v_short_steps = self.diag_v_correctability_short_steps
        v_long_steps = self.diag_v_correctability_long_steps
        knn_k = self.diag_knn_k
        enable_kernel_residual = self.diag_enable_kernel_residual
        kernel_bandwidth_scale = self.diag_kernel_bandwidth_scale
        kernel_min_bandwidth = self.diag_kernel_min_bandwidth
        kernel_target_neff = self.diag_kernel_target_neff
        kernel_search_iters = self.diag_kernel_bandwidth_search_iters
        enable_conditioned_residual = self.diag_enable_conditioned_residual
        actual_source_cap_fraction = self.diag_actual_source_cap_fraction
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
            return jnp.sqrt(sum(jnp.sum(jnp.asarray(x, dtype=jnp.float32) ** 2) for x in leaves) + 1e-12)

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

        def finite_fraction(x):
            return jnp.mean(jnp.isfinite(jnp.asarray(x)).astype(jnp.float32))

        def abs_quantile(x, q):
            return jnp.quantile(jnp.abs(jnp.asarray(x)), q)

        def normalized_direction(h):
            source_rms = rms(h)
            h_unit = h / jnp.maximum(source_rms, 1e-8)
            return h_unit, source_rms

        def binned_calibration(pred, target):
            """Low-variance calibration check for conditional-mean heads.

            Raw sample correlation can be small even for a good E[target|s] estimator because
            the action-conditioned label is noisy.  Sorting by the prediction and averaging
            within bins tests whether predicted conditional means agree with empirical group
            means, which is the more relevant diagnostic here.
            """
            order = jnp.argsort(pred)
            pred_sorted = pred[order]
            target_sorted = target[order]
            pred_bins = jnp.stack([jnp.mean(chunk) for chunk in jnp.array_split(pred_sorted, num_calibration_bins)])
            target_bins = jnp.stack([jnp.mean(chunk) for chunk in jnp.array_split(target_sorted, num_calibration_bins)])
            mse = jnp.mean((pred_bins - target_bins) ** 2)
            sign_agreement = jnp.mean((jnp.sign(pred_bins) == jnp.sign(target_bins)).astype(jnp.float32))
            return corr(pred_bins, target_bins), mse, sign_agreement

        def binned_residual_stats(score, weight, sort_key):
            order = jnp.argsort(sort_key)
            score_sorted = score[order]
            weight_sorted = weight[order]
            score_bins = jnp.stack([jnp.mean(chunk) for chunk in jnp.array_split(score_sorted, num_calibration_bins)])
            weight_bins = jnp.stack([jnp.mean(chunk) for chunk in jnp.array_split(weight_sorted, num_calibration_bins)])
            d_bins = -score_bins / jnp.maximum(weight_bins, kappa_floor)
            return rms(d_bins), jnp.mean(jnp.abs(d_bins)), jnp.max(jnp.abs(d_bins))

        def knn_conditional(query_states, ref_states, ref_values, exclude_diagonal=False):
            """Local non-parametric conditional mean and uncertainty proxy.

            This is deliberately independent of ExpectileDiagnosticHead.  State vectors are
            already dataset-normalized by the training script, so mean squared Euclidean
            distance is a usable first diagnostic metric.  The returned radius/SE are logged
            because kNN is only trustworthy when nearby support exists.
            """
            diff = query_states[:, None, :] - ref_states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            if exclude_diagonal:
                n = jnp.minimum(query_states.shape[0], ref_states.shape[0])
                diag_mask = jnp.eye(query_states.shape[0], ref_states.shape[0], dtype=dist2.dtype)
                dist2 = dist2 + diag_mask * 1e9
            neg_dist, nn_idx = jax.lax.top_k(-dist2, knn_k)
            vals = ref_values[nn_idx]
            mean_val = jnp.mean(vals, axis=1)
            se_val = jnp.std(vals, axis=1) / jnp.sqrt(float(knn_k))
            radius = jnp.sqrt(jnp.maximum(-neg_dist[:, -1], 0.0))
            return mean_val, se_val, radius

        def kernel_bandwidth(states):
            """Median-distance RBF bandwidth on already-normalized states."""
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            n = states.shape[0]
            iu = jnp.triu_indices(n, k=1)
            med = jnp.median(dist2[iu])
            sigma2 = jnp.maximum(
                med * (kernel_bandwidth_scale ** 2),
                kernel_min_bandwidth ** 2,
            )
            return sigma2

        def kernel_bandwidth_target_neff(states):
            """Choose a global RBF bandwidth whose mean leave-one-out N_eff hits a target.

            This directly addresses the V4 failure mode where the median-distance bandwidth
            assigned non-negligible weight to almost the entire 256-state probe batch.
            """
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            n = states.shape[0]
            eye = jnp.eye(n, dtype=dist2.dtype)
            offdiag = 1.0 - eye

            masked_min = jnp.where(offdiag > 0.0, jnp.maximum(dist2, 1e-12), jnp.inf)
            min_d = jnp.min(masked_min)
            max_d = jnp.max(jnp.where(offdiag > 0.0, dist2, 0.0))
            lo = jnp.maximum(min_d * 1e-3, kernel_min_bandwidth ** 2)
            hi = jnp.maximum(max_d * 10.0, lo * 10.0)
            target_neff = jnp.minimum(kernel_target_neff, float(max(n - 1, 1)))

            def mean_neff(sigma2):
                w = jnp.exp(-0.5 * dist2 / jnp.maximum(sigma2, 1e-12)) * offdiag
                wsum = jnp.sum(w, axis=1) + 1e-12
                neff = (wsum ** 2) / (jnp.sum(w ** 2, axis=1) + 1e-12)
                return jnp.mean(neff)

            def body(_, bounds):
                lo_b, hi_b = bounds
                mid = jnp.sqrt(jnp.maximum(lo_b * hi_b, 1e-24))
                n_eff = mean_neff(mid)
                # Larger bandwidth -> larger N_eff.
                lo_new = jnp.where(n_eff < target_neff, mid, lo_b)
                hi_new = jnp.where(n_eff < target_neff, hi_b, mid)
                return lo_new, hi_new

            lo, hi = jax.lax.fori_loop(0, kernel_search_iters, body, (lo, hi))
            sigma2 = jnp.sqrt(jnp.maximum(lo * hi, 1e-24))
            return sigma2, mean_neff(sigma2)

        def kernel_conditional(query_states, ref_states, ref_values, sigma2, exclude_diagonal=False):
            """RBF Nadaraya-Watson conditional mean plus effective-sample-size SE."""
            diff = query_states[:, None, :] - ref_states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            weights = jnp.exp(-0.5 * dist2 / jnp.maximum(sigma2, 1e-12))
            if exclude_diagonal:
                weights = weights * (1.0 - jnp.eye(query_states.shape[0], ref_states.shape[0], dtype=weights.dtype))
            wsum = jnp.sum(weights, axis=1) + 1e-12
            mean_val = jnp.sum(weights * ref_values[None, :], axis=1) / wsum
            centered = ref_values[None, :] - mean_val[:, None]
            var_val = jnp.sum(weights * centered ** 2, axis=1) / wsum
            neff = (wsum ** 2) / (jnp.sum(weights ** 2, axis=1) + 1e-12)
            se_val = jnp.sqrt(jnp.maximum(var_val, 0.0) / jnp.maximum(neff, 1.0))
            return mean_val, se_val, neff

        def kernel_moment_stats(states, score, sigma2):
            """Maximum-moment-style conditional-residual score using an RBF kernel.

            The diagonal is removed so the score does not reduce to E[m^2], which would
            mix conditional-mean violation with irreducible action-conditioned variance.
            """
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            kernel = jnp.exp(-0.5 * dist2 / jnp.maximum(sigma2, 1e-12))
            kernel = kernel * (1.0 - jnp.eye(states.shape[0], dtype=kernel.dtype))
            pair = score[:, None] * kernel * score[None, :]
            n = float(states.shape[0])
            raw_u = jnp.sum(pair) / jnp.maximum(n * (n - 1.0), 1.0)
            kernel_norm = jnp.sum(pair) / (jnp.sum(kernel) + 1e-12)
            score_norm = kernel_norm / (jnp.mean(score ** 2) + 1e-12)
            return raw_u, kernel_norm, score_norm, jnp.sqrt(jnp.maximum(kernel_norm, 0.0))

        def crossfit_kernel_moment(states, score, sigma2):
            """Two-fold cross-fitted estimate of systematic conditional residual energy."""
            n = states.shape[0]
            split = n // 2
            s_a, s_b = states[:split], states[split:]
            m_a, m_b = score[:split], score[split:]
            g_a_to_b, _, _ = kernel_conditional(s_b, s_a, m_a, sigma2, exclude_diagonal=False)
            g_b_to_a, _, _ = kernel_conditional(s_a, s_b, m_b, sigma2, exclude_diagonal=False)
            e_ab = jnp.mean(g_a_to_b * m_b)
            e_ba = jnp.mean(g_b_to_a * m_a)
            energy = 0.5 * (e_ab + e_ba)
            norm = energy / (jnp.mean(score ** 2) + 1e-12)
            sign_agree = 0.5 * (
                jnp.mean((jnp.sign(g_a_to_b) == jnp.sign(m_b)).astype(jnp.float32))
                + jnp.mean((jnp.sign(g_b_to_a) == jnp.sign(m_a)).astype(jnp.float32))
            )
            return energy, norm, sign_agree

        def crossfit_displacement_energy(states, score, weight, sigma2):
            """Cross-fitted conditioning-aware residual-displacement energy.

            For each fold, estimate G(s) and kappa(s) from the opposite fold, then use
                d_hat(s) = -G_hat(s) / kappa_hat(s)
            together with the held-out noisy displacement label
                y(s,a) = -psi_tau(Q-V) / kappa_hat(s).
            Their cross product estimates E[d(s)^2] without using the same sample to
            construct both factors.  This is a diagnostic approximation, not a formal CI.
            """
            n = states.shape[0]
            split = n // 2
            s_a, s_b = states[:split], states[split:]
            m_a, m_b = score[:split], score[split:]
            w_a, w_b = weight[:split], weight[split:]

            g_a_to_b, _, _ = kernel_conditional(s_b, s_a, m_a, sigma2, exclude_diagonal=False)
            k_a_to_b, _, _ = kernel_conditional(s_b, s_a, w_a, sigma2, exclude_diagonal=False)
            g_b_to_a, _, _ = kernel_conditional(s_a, s_b, m_b, sigma2, exclude_diagonal=False)
            k_b_to_a, _, _ = kernel_conditional(s_a, s_b, w_b, sigma2, exclude_diagonal=False)

            k_a_to_b = jnp.maximum(k_a_to_b, kappa_floor)
            k_b_to_a = jnp.maximum(k_b_to_a, kappa_floor)
            d_a_to_b = -g_a_to_b / k_a_to_b
            d_b_to_a = -g_b_to_a / k_b_to_a
            y_b = -m_b / k_a_to_b
            y_a = -m_a / k_b_to_a

            e_ab = jnp.mean(d_a_to_b * y_b)
            e_ba = jnp.mean(d_b_to_a * y_a)
            energy = 0.5 * (e_ab + e_ba)
            heldout_energy = 0.5 * (jnp.mean(y_b ** 2) + jnp.mean(y_a ** 2))
            normalized = energy / (heldout_energy + 1e-12)
            sign_agree = 0.5 * (
                jnp.mean((jnp.sign(d_a_to_b) == jnp.sign(y_b)).astype(jnp.float32))
                + jnp.mean((jnp.sign(d_b_to_a) == jnp.sign(y_a)).astype(jnp.float32))
            )
            return energy, jnp.sqrt(jnp.maximum(energy, 0.0)), normalized, sign_agree

        def displacement_se(g, g_se, kappa, kappa_se):
            """Delta-method SE for d=-G/kappa, ignoring G-kappa covariance."""
            k = jnp.maximum(kappa, kappa_floor)
            return jnp.sqrt(
                (g_se / k) ** 2
                + ((g * kappa_se) / (k ** 2)) ** 2
                + 1e-12
            )

        def reinforcement_stats(delta, error):
            """Signed contribution of a propagated Q change to an existing error energy."""
            delta = jnp.ravel(delta)
            error = jnp.ravel(error)
            denom = jnp.mean(error ** 2) + 1e-12
            dot = jnp.mean(delta * error)
            coeff = dot / denom
            rel_energy_change = (2.0 * dot + jnp.mean(delta ** 2)) / denom
            harm = rms(delta) * jnp.maximum(cosine(delta, error), 0.0)
            return coeff, rel_energy_change, harm

        @jax.jit
        def diagnostic_step(old_state: IQLState, new_state: IQLState, batch: TensorBatch, probe_batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            probe_observations = probe_batch["observations"]
            probe_actions = probe_batch["actions"]
            probe_rewards = jnp.squeeze(probe_batch["rewards"], axis=-1)
            probe_next_observations = probe_batch["next_observations"]
            probe_dones = jnp.squeeze(probe_batch["dones"], axis=-1)

            # Exact Bellman target used by the Q update in the committed training step.
            next_v_old = v_apply({"params": old_state.v_params}, next_observations)
            target = rewards + (1.0 - dones) * discount * next_v_old
            target_rms = rms(target)
            fd_base_delta = fd_epsilon * (jnp.maximum(target_rms, 1.0) if fd_relative_scale else 1.0)
            fd_small_delta = fd_base_delta * fd_small_multiplier
            fd_large_delta = fd_base_delta * fd_large_multiplier

            def q_loss_given_target(q_params, target_arg):
                qs_local = q_apply({"params": q_params}, observations, actions)
                return jnp.mean((qs_local - target_arg[None, :]) ** 2)

            def actual_adam_step(target_arg):
                q_grads = jax.grad(q_loss_given_target)(old_state.q_params, target_arg)
                q_updates, _ = q_tx.update(q_grads, old_state.q_opt_state, old_state.q_params)
                q_params_plus = optax.apply_updates(old_state.q_params, q_updates)
                return q_params_plus, q_grads, q_updates

            def actual_adam_output_for(target_arg, out_obs, out_actions):
                q_params_plus, _, _ = actual_adam_step(target_arg)
                return q_apply({"params": q_params_plus}, out_obs, out_actions)

            def actual_probe_output(target_arg):
                return actual_adam_output_for(target_arg, probe_observations, probe_actions)

            def actual_train_output(target_arg):
                return actual_adam_output_for(target_arg, observations, actions)

            base_q_params_plus, base_q_grads, base_q_updates = actual_adam_step(target)
            base_probe = q_apply({"params": base_q_params_plus}, probe_observations, probe_actions)
            base_train = q_apply({"params": base_q_params_plus}, observations, actions)

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

            # ----- Residual-head validation on an independent probe batch. -----
            probe_q_target_all = q_apply({"params": old_state.q_target_params}, probe_observations, probe_actions)
            probe_q_target = jnp.min(probe_q_target_all, axis=0)
            probe_q_online_all = q_apply({"params": old_state.q_params}, probe_observations, probe_actions)
            probe_q_online = jnp.min(probe_q_online_all, axis=0)
            probe_q_online_mean = jnp.mean(probe_q_online_all, axis=0)
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

            # Conditional-head quality: raw-label fit plus binned calibration and normalized baselines.
            g_mse = jnp.mean((g_hat_probe - score_pre) ** 2)
            kappa_mse = jnp.mean((kappa_hat_probe - weight_pre) ** 2)
            qnum_mse = jnp.mean((q_num_hat_probe - q_num_target_probe) ** 2)
            g_zero_mse = jnp.mean(score_pre ** 2)
            kappa_mean_baseline_mse = jnp.mean((weight_pre - jnp.mean(weight_pre)) ** 2)
            qnum_zero_mse = jnp.mean(q_num_target_probe ** 2)
            g_cal_corr, g_cal_mse, g_cal_sign = binned_calibration(g_hat_probe, score_pre)
            kappa_cal_corr, kappa_cal_mse, _ = binned_calibration(kappa_hat_probe, weight_pre)
            qnum_cal_corr, qnum_cal_mse, qnum_cal_sign = binned_calibration(q_num_hat_probe, q_num_target_probe)

            # The kappa head can be reinterpreted as a classifier for P(Q-V >= 0 | s):
            # kappa = (1-tau) + (2tau-1) p_+.  This exposes whether the head learned
            # state-specific sign probability or merely the global mean kappa.
            kappa_denom = 2.0 * iql_tau - 1.0
            ppos_hat_probe = jnp.where(
                jnp.abs(kappa_denom) > 1e-8,
                (kappa_hat_probe - (1.0 - iql_tau)) / kappa_denom,
                0.5 * jnp.ones_like(kappa_hat_probe),
            )
            ppos_hat_probe = jnp.clip(ppos_hat_probe, 1e-5, 1.0 - 1e-5)
            d_bin_pre_rms, d_bin_pre_abs, d_bin_pre_max = binned_residual_stats(score_pre, weight_pre, g_hat_probe)
            d_bin_post_rms, d_bin_post_abs, d_bin_post_max = binned_residual_stats(score_post, weight_post, g_hat_probe)

            # Independent kNN reference on the probe batch.  Leave-one-out prevents a
            # sample from trivially explaining its own noisy action-conditioned label.
            g_knn_probe, g_knn_se_probe, knn_probe_radius = knn_conditional(
                probe_observations, probe_observations, score_pre, exclude_diagonal=True
            )
            kappa_knn_probe, kappa_knn_se_probe, _ = knn_conditional(
                probe_observations, probe_observations, weight_pre, exclude_diagonal=True
            )
            qnum_knn_probe, qnum_knn_se_probe, _ = knn_conditional(
                probe_observations, probe_observations, q_num_target_probe, exclude_diagonal=True
            )
            d_knn_probe = -g_knn_probe / jnp.maximum(kappa_knn_probe, kappa_floor)

            # Kernel conditional-moment diagnostics.  Unlike pointwise regression, these
            # explicitly test the conditional estimating-equation restriction and remove
            # diagonal m_i^2 terms that confound residual with single-action noise.
            u_probe = jax.lax.stop_gradient(probe_q_target) - probe_v_pre
            pos_moment_probe = iql_tau * jnp.maximum(u_probe, 0.0)
            neg_moment_probe = (1.0 - iql_tau) * jnp.maximum(-u_probe, 0.0)
            pos_indicator_probe = (u_probe >= 0.0).astype(jnp.float32)
            ppos_brier = jnp.mean((ppos_hat_probe - pos_indicator_probe) ** 2)
            ppos_prevalence = jnp.mean(pos_indicator_probe)
            ppos_baseline_brier = jnp.mean((pos_indicator_probe - ppos_prevalence) ** 2)
            ppos_skill = 1.0 - ppos_brier / (ppos_baseline_brier + 1e-12)
            ppos_logloss = -jnp.mean(
                pos_indicator_probe * jnp.log(ppos_hat_probe)
                + (1.0 - pos_indicator_probe) * jnp.log(1.0 - ppos_hat_probe)
            )
            ppos_cal_corr, ppos_cal_mse, _ = binned_calibration(ppos_hat_probe, pos_indicator_probe)

            # Direct finite-difference validation of dG/dV = -kappa at the population/batch level.
            foc_shift = 1e-3 * jnp.maximum(rms(probe_q_target), 1.0)
            score_plus_shift, _, _ = score_terms(probe_v_pre + foc_shift)
            score_minus_shift, _, _ = score_terms(probe_v_pre - foc_shift)
            foc_dg_dv_fd = (jnp.mean(score_plus_shift) - jnp.mean(score_minus_shift)) / (2.0 * foc_shift)
            foc_slope_over_kappa = -foc_dg_dv_fd / (kappa_global_pre + 1e-12)

            if enable_kernel_residual:
                kernel_sigma2 = kernel_bandwidth(probe_observations)
                mmr_raw_u, mmr_kernel_norm, mmr_score_norm, mmr_root = kernel_moment_stats(
                    probe_observations, score_pre, kernel_sigma2
                )
                mmr_crossfit, mmr_crossfit_norm, mmr_crossfit_sign = crossfit_kernel_moment(
                    probe_observations, score_pre, kernel_sigma2
                )
                g_kernel_probe, g_kernel_se_probe, kernel_neff_probe = kernel_conditional(
                    probe_observations, probe_observations, score_pre, kernel_sigma2, exclude_diagonal=True
                )
                kappa_kernel_probe, kappa_kernel_se_probe, _ = kernel_conditional(
                    probe_observations, probe_observations, weight_pre, kernel_sigma2, exclude_diagonal=True
                )
                qnum_kernel_probe, qnum_kernel_se_probe, _ = kernel_conditional(
                    probe_observations, probe_observations, q_num_target_probe, kernel_sigma2, exclude_diagonal=True
                )
                ppos_kernel_probe, _, _ = kernel_conditional(
                    probe_observations, probe_observations, pos_indicator_probe, kernel_sigma2, exclude_diagonal=True
                )
                kappa_from_ppos_probe = (1.0 - iql_tau) + (2.0 * iql_tau - 1.0) * ppos_kernel_probe
                d_kernel_probe = -g_kernel_probe / jnp.maximum(kappa_kernel_probe, kappa_floor)
                pos_kernel_probe, _, _ = kernel_conditional(
                    probe_observations, probe_observations, pos_moment_probe, kernel_sigma2, exclude_diagonal=True
                )
                neg_kernel_probe, _, _ = kernel_conditional(
                    probe_observations, probe_observations, neg_moment_probe, kernel_sigma2, exclude_diagonal=True
                )
                balance_kernel_probe = (pos_kernel_probe - neg_kernel_probe) / (
                    pos_kernel_probe + neg_kernel_probe + 1e-12
                )

                # Adaptive local kernel with controlled effective support.
                kernel_local_sigma2, kernel_local_neff_target_achieved = kernel_bandwidth_target_neff(
                    probe_observations
                )
                mmr_local_raw, mmr_local_kernel_norm, mmr_local_score_norm, mmr_local_root = kernel_moment_stats(
                    probe_observations, score_pre, kernel_local_sigma2
                )
                mmr_local_crossfit, mmr_local_crossfit_norm, mmr_local_crossfit_sign = crossfit_kernel_moment(
                    probe_observations, score_pre, kernel_local_sigma2
                )
                g_kernel_local_probe, g_kernel_local_se_probe, kernel_local_neff_probe = kernel_conditional(
                    probe_observations, probe_observations, score_pre, kernel_local_sigma2, exclude_diagonal=True
                )
                kappa_kernel_local_probe, kappa_kernel_local_se_probe, _ = kernel_conditional(
                    probe_observations, probe_observations, weight_pre, kernel_local_sigma2, exclude_diagonal=True
                )
                qnum_kernel_local_probe, qnum_kernel_local_se_probe, _ = kernel_conditional(
                    probe_observations, probe_observations, q_num_target_probe, kernel_local_sigma2, exclude_diagonal=True
                )
                d_kernel_local_probe = -g_kernel_local_probe / jnp.maximum(kappa_kernel_local_probe, kappa_floor)
                pos_kernel_local_probe, _, _ = kernel_conditional(
                    probe_observations, probe_observations, pos_moment_probe, kernel_local_sigma2, exclude_diagonal=True
                )
                neg_kernel_local_probe, _, _ = kernel_conditional(
                    probe_observations, probe_observations, neg_moment_probe, kernel_local_sigma2, exclude_diagonal=True
                )
                balance_kernel_local_probe = (pos_kernel_local_probe - neg_kernel_local_probe) / (
                    pos_kernel_local_probe + neg_kernel_local_probe + 1e-12
                )

                # V7 conditioning-aware residual diagnostics.  Raw G shrinks with kappa,
                # so the displacement d=-G/kappa and its uncertainty are more relevant than
                # raw moment magnitude at high expectiles.
                if enable_conditioned_residual:
                    d_kernel_local_se_probe = displacement_se(
                        g_kernel_local_probe,
                        g_kernel_local_se_probe,
                        kappa_kernel_local_probe,
                        kappa_kernel_local_se_probe,
                    )
                    d_kernel_local_snr_probe = jnp.abs(d_kernel_local_probe) / (
                        d_kernel_local_se_probe + 1e-12
                    )
                    g_kernel_local_snr_probe = jnp.abs(g_kernel_local_probe) / (
                        g_kernel_local_se_probe + 1e-12
                    )
                    d_kernel_local_noise_corrected_rms = jnp.sqrt(
                        jnp.mean(jnp.maximum(
                            d_kernel_local_probe ** 2 - d_kernel_local_se_probe ** 2,
                            0.0,
                        )) + 1e-12
                    )
                    (
                        conditioned_crossfit_energy,
                        conditioned_crossfit_root,
                        conditioned_crossfit_norm,
                        conditioned_crossfit_sign,
                    ) = crossfit_displacement_energy(
                        probe_observations,
                        score_pre,
                        weight_pre,
                        kernel_local_sigma2,
                    )
                else:
                    d_kernel_local_se_probe = jnp.full_like(d_kernel_local_probe, jnp.nan)
                    d_kernel_local_snr_probe = jnp.full_like(d_kernel_local_probe, jnp.nan)
                    g_kernel_local_snr_probe = jnp.full_like(d_kernel_local_probe, jnp.nan)
                    d_kernel_local_noise_corrected_rms = jnp.asarray(jnp.nan, dtype=jnp.float32)
                    conditioned_crossfit_energy = conditioned_crossfit_root = jnp.asarray(jnp.nan, dtype=jnp.float32)
                    conditioned_crossfit_norm = conditioned_crossfit_sign = jnp.asarray(jnp.nan, dtype=jnp.float32)

                # Multi-scale scalar moment sensitivity around the original median bandwidth.
                kernel_ms_factors = jnp.asarray([0.125, 0.25, 0.5, 1.0, 2.0], dtype=jnp.float32)
                kernel_ms_sigma2 = kernel_sigma2 * (kernel_ms_factors ** 2)
                def one_ms(s2):
                    _, _, score_norm, _ = kernel_moment_stats(probe_observations, score_pre, s2)
                    _, cross_norm, _ = crossfit_kernel_moment(probe_observations, score_pre, s2)
                    _, _, neff = kernel_conditional(
                        probe_observations, probe_observations, score_pre, s2, exclude_diagonal=True
                    )
                    return score_norm, cross_norm, jnp.mean(neff)
                kernel_ms_score, kernel_ms_crossfit, kernel_ms_neff = jax.vmap(one_ms)(kernel_ms_sigma2)
            else:
                kernel_sigma2 = jnp.asarray(jnp.nan, dtype=jnp.float32)
                mmr_raw_u = mmr_kernel_norm = mmr_score_norm = mmr_root = jnp.asarray(jnp.nan, dtype=jnp.float32)
                mmr_crossfit = mmr_crossfit_norm = mmr_crossfit_sign = jnp.asarray(jnp.nan, dtype=jnp.float32)
                g_kernel_probe = jnp.full_like(score_pre, jnp.nan)
                g_kernel_se_probe = jnp.full_like(score_pre, jnp.nan)
                kernel_neff_probe = jnp.full_like(score_pre, jnp.nan)
                kappa_kernel_probe = jnp.full_like(weight_pre, jnp.nan)
                kappa_kernel_se_probe = jnp.full_like(weight_pre, jnp.nan)
                qnum_kernel_probe = jnp.full_like(q_num_target_probe, jnp.nan)
                qnum_kernel_se_probe = jnp.full_like(q_num_target_probe, jnp.nan)
                ppos_kernel_probe = jnp.full_like(weight_pre, jnp.nan)
                kappa_from_ppos_probe = jnp.full_like(weight_pre, jnp.nan)
                d_kernel_probe = jnp.full_like(score_pre, jnp.nan)
                balance_kernel_probe = jnp.full_like(score_pre, jnp.nan)
                pos_kernel_probe = jnp.full_like(score_pre, jnp.nan)
                neg_kernel_probe = jnp.full_like(score_pre, jnp.nan)
                kernel_local_sigma2 = kernel_local_neff_target_achieved = jnp.asarray(jnp.nan, dtype=jnp.float32)
                mmr_local_raw = mmr_local_kernel_norm = mmr_local_score_norm = mmr_local_root = jnp.asarray(jnp.nan, dtype=jnp.float32)
                mmr_local_crossfit = mmr_local_crossfit_norm = mmr_local_crossfit_sign = jnp.asarray(jnp.nan, dtype=jnp.float32)
                g_kernel_local_probe = jnp.full_like(score_pre, jnp.nan)
                g_kernel_local_se_probe = jnp.full_like(score_pre, jnp.nan)
                kernel_local_neff_probe = jnp.full_like(score_pre, jnp.nan)
                kappa_kernel_local_probe = jnp.full_like(weight_pre, jnp.nan)
                kappa_kernel_local_se_probe = jnp.full_like(weight_pre, jnp.nan)
                qnum_kernel_local_probe = jnp.full_like(q_num_target_probe, jnp.nan)
                qnum_kernel_local_se_probe = jnp.full_like(q_num_target_probe, jnp.nan)
                d_kernel_local_probe = jnp.full_like(score_pre, jnp.nan)
                pos_kernel_local_probe = jnp.full_like(score_pre, jnp.nan)
                neg_kernel_local_probe = jnp.full_like(score_pre, jnp.nan)
                balance_kernel_local_probe = jnp.full_like(score_pre, jnp.nan)
                d_kernel_local_se_probe = jnp.full_like(score_pre, jnp.nan)
                d_kernel_local_snr_probe = jnp.full_like(score_pre, jnp.nan)
                g_kernel_local_snr_probe = jnp.full_like(score_pre, jnp.nan)
                d_kernel_local_noise_corrected_rms = jnp.asarray(jnp.nan, dtype=jnp.float32)
                conditioned_crossfit_energy = conditioned_crossfit_root = jnp.asarray(jnp.nan, dtype=jnp.float32)
                conditioned_crossfit_norm = conditioned_crossfit_sign = jnp.asarray(jnp.nan, dtype=jnp.float32)
                kernel_ms_factors = jnp.asarray([0.125, 0.25, 0.5, 1.0, 2.0], dtype=jnp.float32)
                kernel_ms_score = jnp.full((5,), jnp.nan, dtype=jnp.float32)
                kernel_ms_crossfit = jnp.full((5,), jnp.nan, dtype=jnp.float32)
                kernel_ms_neff = jnp.full((5,), jnp.nan, dtype=jnp.float32)

            def probe_v_loss_fn(v_params):
                probe_v = v_apply({"params": v_params}, probe_observations)
                u = jax.lax.stop_gradient(probe_q_target) - probe_v
                w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                return jnp.mean(w * u ** 2)

            v_grad_pre = jax.grad(probe_v_loss_fn)(old_state.v_params)
            v_grad_post = jax.grad(probe_v_loss_fn)(new_state.v_params)

            # Frozen-Q, virtual multi-step V refinement.  This tests whether a large residual
            # is ordinary under-training (quickly correctable) or persists under additional V optimization.
            if enable_multistep_v:
                train_q_target_all = q_apply({"params": old_state.q_target_params}, observations, actions)
                train_q_target = jax.lax.stop_gradient(jnp.min(train_q_target_all, axis=0))

                def virtual_v_one(carry):
                    v_params_local, v_opt_state_local = carry

                    def local_v_loss(params):
                        vv = v_apply({"params": params}, observations)
                        uu = train_q_target - vv
                        ww = jnp.abs(iql_tau - (uu < 0.0).astype(jnp.float32))
                        return jnp.mean(ww * uu ** 2)

                    grads = jax.grad(local_v_loss)(v_params_local)
                    updates, opt_state_new = v_tx.update(grads, v_opt_state_local, v_params_local)
                    params_new = optax.apply_updates(v_params_local, updates)
                    return params_new, opt_state_new

                def run_virtual_v_steps(n_steps):
                    def body(_, carry):
                        return virtual_v_one(carry)
                    return jax.lax.fori_loop(
                        0,
                        n_steps,
                        body,
                        (old_state.v_params, old_state.v_opt_state),
                    )[0]

                v_params_short = run_virtual_v_steps(v_short_steps)
                v_params_long = run_virtual_v_steps(v_long_steps)
                probe_v_short = v_apply({"params": v_params_short}, probe_observations)
                probe_v_long = v_apply({"params": v_params_long}, probe_observations)
                score_short, weight_short, probe_v_loss_short = score_terms(probe_v_short)
                score_long, weight_long, probe_v_loss_long = score_terms(probe_v_long)
                g_short = jnp.mean(score_short)
                kappa_short = jnp.maximum(jnp.mean(weight_short), kappa_floor)
                d_short = -g_short / kappa_short
                g_long = jnp.mean(score_long)
                kappa_long = jnp.maximum(jnp.mean(weight_long), kappa_floor)
                d_long = -g_long / kappa_long
                d_bin_short_rms, _, _ = binned_residual_stats(score_short, weight_short, g_hat_probe)
                d_bin_long_rms, _, _ = binned_residual_stats(score_long, weight_long, g_hat_probe)
            else:
                probe_v_loss_short = jnp.asarray(jnp.nan, dtype=jnp.float32)
                probe_v_loss_long = jnp.asarray(jnp.nan, dtype=jnp.float32)
                d_short = jnp.asarray(jnp.nan, dtype=jnp.float32)
                d_long = jnp.asarray(jnp.nan, dtype=jnp.float32)
                d_bin_short_rms = jnp.asarray(jnp.nan, dtype=jnp.float32)
                d_bin_long_rms = jnp.asarray(jnp.nan, dtype=jnp.float32)

            # ----- State-conditional h_V and h_Q^lag on the actual Q-fitting batch. -----
            raw_next = diag_apply({"params": new_state.diag_params}, next_observations)
            g_hat_next, kappa_hat_next, q_num_hat_next = unpack_diag(raw_next)
            residual_value_error_hat = -g_hat_next / kappa_hat_next  # V - V_exact, first order.
            h_v = (1.0 - dones) * discount * residual_value_error_hat
            critic_value_error_lag_hat = q_num_hat_next / kappa_hat_next
            h_q_lag = (1.0 - dones) * discount * critic_value_error_lag_hat
            h_total_proxy = h_v + h_q_lag

            # Independent state-neighborhood source proxy evaluated at the actual next states.
            g_knn_next, g_knn_se_next, knn_next_radius = knn_conditional(
                next_observations, probe_observations, score_pre, exclude_diagonal=False
            )
            kappa_knn_next, kappa_knn_se_next, _ = knn_conditional(
                next_observations, probe_observations, weight_pre, exclude_diagonal=False
            )
            qnum_knn_next, qnum_knn_se_next, _ = knn_conditional(
                next_observations, probe_observations, q_num_target_probe, exclude_diagonal=False
            )
            d_v_knn_next = -g_knn_next / jnp.maximum(kappa_knn_next, kappa_floor)
            d_q_knn_next = qnum_knn_next / jnp.maximum(kappa_knn_next, kappa_floor)
            h_v_knn = (1.0 - dones) * discount * d_v_knn_next
            h_q_knn = (1.0 - dones) * discount * d_q_knn_next

            if enable_kernel_residual:
                g_kernel_next, g_kernel_se_next, kernel_neff_next = kernel_conditional(
                    next_observations, probe_observations, score_pre, kernel_sigma2, exclude_diagonal=False
                )
                kappa_kernel_next, kappa_kernel_se_next, _ = kernel_conditional(
                    next_observations, probe_observations, weight_pre, kernel_sigma2, exclude_diagonal=False
                )
                qnum_kernel_next, qnum_kernel_se_next, _ = kernel_conditional(
                    next_observations, probe_observations, q_num_target_probe, kernel_sigma2, exclude_diagonal=False
                )
                d_v_kernel_next = -g_kernel_next / jnp.maximum(kappa_kernel_next, kappa_floor)
                d_q_kernel_next = qnum_kernel_next / jnp.maximum(kappa_kernel_next, kappa_floor)
                h_v_kernel = (1.0 - dones) * discount * d_v_kernel_next
                h_q_kernel = (1.0 - dones) * discount * d_q_kernel_next

                g_kernel_local_next, g_kernel_local_se_next, kernel_local_neff_next = kernel_conditional(
                    next_observations, probe_observations, score_pre, kernel_local_sigma2, exclude_diagonal=False
                )
                kappa_kernel_local_next, kappa_kernel_local_se_next, _ = kernel_conditional(
                    next_observations, probe_observations, weight_pre, kernel_local_sigma2, exclude_diagonal=False
                )
                qnum_kernel_local_next, qnum_kernel_local_se_next, _ = kernel_conditional(
                    next_observations, probe_observations, q_num_target_probe, kernel_local_sigma2, exclude_diagonal=False
                )
                d_v_kernel_local_next = -g_kernel_local_next / jnp.maximum(kappa_kernel_local_next, kappa_floor)
                d_q_kernel_local_next = qnum_kernel_local_next / jnp.maximum(kappa_kernel_local_next, kappa_floor)
                h_v_kernel_local = (1.0 - dones) * discount * d_v_kernel_local_next
                h_q_kernel_local = (1.0 - dones) * discount * d_q_kernel_local_next
            else:
                g_kernel_next = jnp.full_like(h_v_knn, jnp.nan)
                g_kernel_se_next = jnp.full_like(h_v_knn, jnp.nan)
                kernel_neff_next = jnp.full_like(h_v_knn, jnp.nan)
                kappa_kernel_next = jnp.full_like(h_v_knn, jnp.nan)
                kappa_kernel_se_next = jnp.full_like(h_v_knn, jnp.nan)
                qnum_kernel_next = jnp.full_like(h_v_knn, jnp.nan)
                qnum_kernel_se_next = jnp.full_like(h_v_knn, jnp.nan)
                d_v_kernel_next = jnp.full_like(h_v_knn, jnp.nan)
                d_q_kernel_next = jnp.full_like(h_v_knn, jnp.nan)
                h_v_kernel = jnp.full_like(h_v_knn, jnp.nan)
                h_q_kernel = jnp.full_like(h_q_knn, jnp.nan)
                g_kernel_local_next = jnp.full_like(h_v_knn, jnp.nan)
                g_kernel_local_se_next = jnp.full_like(h_v_knn, jnp.nan)
                kernel_local_neff_next = jnp.full_like(h_v_knn, jnp.nan)
                kappa_kernel_local_next = jnp.full_like(h_v_knn, jnp.nan)
                kappa_kernel_local_se_next = jnp.full_like(h_v_knn, jnp.nan)
                qnum_kernel_local_next = jnp.full_like(h_v_knn, jnp.nan)
                qnum_kernel_local_se_next = jnp.full_like(h_v_knn, jnp.nan)
                d_v_kernel_local_next = jnp.full_like(h_v_knn, jnp.nan)
                d_q_kernel_local_next = jnp.full_like(h_q_knn, jnp.nan)
                h_v_kernel_local = jnp.full_like(h_v_knn, jnp.nan)
                h_q_kernel_local = jnp.full_like(h_q_knn, jnp.nan)

            if enable_conditioned_residual:
                d_v_kernel_local_se_next = displacement_se(
                    g_kernel_local_next,
                    g_kernel_local_se_next,
                    kappa_kernel_local_next,
                    kappa_kernel_local_se_next,
                )
                d_v_kernel_local_snr_next = jnp.abs(d_v_kernel_local_next) / (
                    d_v_kernel_local_se_next + 1e-12
                )
            else:
                d_v_kernel_local_se_next = jnp.full_like(h_v_knn, jnp.nan)
                d_v_kernel_local_snr_next = jnp.full_like(h_v_knn, jnp.nan)

            # Deterministic random direction so the optional ridge diagnostic can reproduce it.
            base_key = jax.random.fold_in(jax.random.PRNGKey(diag_seed), old_state.total_it + 1)
            rand_key, basis_key = jax.random.split(base_key)
            h_random = jax.random.rademacher(rand_key, shape=target.shape, dtype=jnp.float32)

            def central_fd(h_unit, delta):
                plus = actual_probe_output(target + delta * h_unit)
                minus = actual_probe_output(target - delta * h_unit)
                return (plus - minus) / (2.0 * delta), plus

            def direction_outputs(h):
                h_unit, source_rms = normalized_direction(h)
                fd_probe, plus_probe = central_fd(h_unit, fd_base_delta)
                fd_forward = (plus_probe - base_probe) / fd_base_delta
                _, adam_jvp_probe = jax.jvp(actual_probe_output, (target,), (h_unit,))
                ntk_probe = ntk_output(h_unit, probe_observations, probe_actions)
                return h_unit, source_rms, fd_probe, fd_forward, adam_jvp_probe, ntk_probe

            random_unit, random_source_rms, random_fd, random_fd_forward, random_adam_jvp, random_ntk = direction_outputs(h_random)
            h_v_unit, h_v_source_rms, h_v_fd, h_v_fd_forward, h_v_adam_jvp, h_v_ntk = direction_outputs(h_v)
            h_q_unit, h_q_source_rms, h_q_fd, h_q_fd_forward, h_q_adam_jvp, h_q_ntk = direction_outputs(h_q_lag)
            h_total_unit, h_total_source_rms, h_total_fd, _, _, h_total_ntk = direction_outputs(h_total_proxy)

            # kNN alternatives intentionally use only the robust finite-difference reference.
            h_v_knn_unit, h_v_knn_source_rms = normalized_direction(h_v_knn)
            h_q_knn_unit, h_q_knn_source_rms = normalized_direction(h_q_knn)
            h_v_knn_fd, _ = central_fd(h_v_knn_unit, fd_base_delta)
            h_q_knn_fd, _ = central_fd(h_q_knn_unit, fd_base_delta)

            h_v_kernel_unit, h_v_kernel_source_rms = normalized_direction(h_v_kernel)
            h_q_kernel_unit, h_q_kernel_source_rms = normalized_direction(h_q_kernel)
            h_v_kernel_fd, _ = central_fd(h_v_kernel_unit, fd_base_delta)
            h_q_kernel_fd, _ = central_fd(h_q_kernel_unit, fd_base_delta)

            h_v_kernel_local_unit, h_v_kernel_local_source_rms = normalized_direction(h_v_kernel_local)
            h_q_kernel_local_unit, h_q_kernel_local_source_rms = normalized_direction(h_q_kernel_local)
            h_v_kernel_local_fd, _ = central_fd(h_v_kernel_local_unit, fd_base_delta)
            h_q_kernel_local_fd, _ = central_fd(h_q_kernel_local_unit, fd_base_delta)

            random_fd_small, _ = central_fd(random_unit, fd_small_delta)
            random_fd_large, _ = central_fd(random_unit, fd_large_delta)

            # Finite-amplitude response at 25/50/100% of a safely clipped source scale.
            # This complements the infinitesimal/local B diagnostic when Adam is nonlinear.
            def finite_amplitude_curve(h, source_rms, local_fd):
                max_rms = actual_source_cap_fraction * jnp.maximum(target_rms, 1.0)
                base_alpha = jnp.minimum(1.0, max_rms / (source_rms + 1e-12))

                def one(frac):
                    alpha = base_alpha * frac
                    response = actual_probe_output(target + alpha * h) - base_probe
                    linear_pred = local_fd * (alpha * source_rms)
                    return (
                        alpha,
                        rms(response) / (alpha * source_rms + 1e-12),
                        cosine(response, linear_pred),
                        rms(response - linear_pred) / (rms(response) + 1e-12),
                    )

                return one(0.25), one(0.50), one(1.00)

            h_v_amp25, h_v_amp50, h_v_amp100 = finite_amplitude_curve(h_v, h_v_source_rms, h_v_fd)
            h_q_amp25, h_q_amp50, h_q_amp100 = finite_amplitude_curve(h_q_lag, h_q_source_rms, h_q_fd)
            h_v_actual_alpha, h_v_actual_gain, h_v_actual_cos, h_v_actual_relerr = h_v_amp100
            h_q_actual_alpha, h_q_actual_gain, h_q_actual_cos, h_q_actual_relerr = h_q_amp100

            # Cross-sample spillover: finite-difference actual Adam is the robust reference;
            # the exact Adam JVP is retained only as a numerical diagnostic because it can NaN.
            batch_size = target.shape[0]
            basis_indices = jax.random.randint(basis_key, shape=(num_basis,), minval=0, maxval=batch_size)
            basis = jax.nn.one_hot(basis_indices, batch_size, dtype=target.dtype)

            def one_basis_fd(h_basis):
                plus = actual_train_output(target + fd_base_delta * h_basis)
                minus = actual_train_output(target - fd_base_delta * h_basis)
                return (plus - minus) / (2.0 * fd_base_delta)

            def one_basis_adam_jvp(h_basis):
                _, out = jax.jvp(actual_train_output, (target,), (h_basis,))
                return out

            def one_basis_ntk(h_basis):
                return ntk_output(h_basis, observations, actions)

            basis_fd = jax.vmap(one_basis_fd)(basis)
            basis_adam_jvp = jax.vmap(one_basis_adam_jvp)(basis)
            basis_ntk = jax.vmap(one_basis_ntk)(basis)

            def spillover_stats(outputs):
                mean_q_response = jnp.mean(outputs, axis=1)  # [K, B]
                self_response = jnp.take_along_axis(mean_q_response, basis_indices[:, None], axis=1)[:, 0]
                total_sq = jnp.sum(mean_q_response ** 2, axis=1)
                cross_l2 = jnp.sqrt(jnp.maximum(total_sq - self_response ** 2, 0.0) + 1e-12)
                cross_rms_per_coord = cross_l2 / jnp.sqrt(jnp.maximum(float(batch_size - 1), 1.0))
                ratio = cross_l2 / (jnp.abs(self_response) + 1e-8)
                per_coord_ratio = cross_rms_per_coord / (jnp.abs(self_response) + 1e-8)
                return (
                    jnp.mean(jnp.abs(self_response)),
                    jnp.mean(cross_l2),
                    jnp.mean(ratio),
                    jnp.mean(cross_rms_per_coord),
                    jnp.mean(per_coord_ratio),
                )

            fd_self, fd_spill, fd_spill_ratio, fd_spill_rms_coord, fd_spill_coord_ratio = spillover_stats(basis_fd)
            adam_jvp_self, adam_jvp_spill, adam_jvp_spill_ratio, adam_jvp_spill_rms_coord, adam_jvp_spill_coord_ratio = spillover_stats(basis_adam_jvp)
            ntk_self, ntk_spill, ntk_spill_ratio, ntk_spill_rms_coord, ntk_spill_coord_ratio = spillover_stats(basis_ntk)

            # ----- Target-network lag dynamics and actual Q-update geometry. -----
            probe_q_online_new_all = q_apply({"params": new_state.q_params}, probe_observations, probe_actions)
            probe_q_target_new_all = q_apply({"params": new_state.q_target_params}, probe_observations, probe_actions)
            probe_q_online_new = jnp.min(probe_q_online_new_all, axis=0)
            probe_q_target_new = jnp.min(probe_q_target_new_all, axis=0)
            lag_pre = eps_lag_probe
            lag_post = probe_q_online_new - probe_q_target_new
            online_delta = probe_q_online_new - probe_q_online
            target_delta = probe_q_target_new - probe_q_target
            lag_delta = lag_post - lag_pre
            lag_recurrence_error = lag_post - (lag_pre + online_delta - target_delta)

            probe_next_v_old = v_apply({"params": old_state.v_params}, probe_next_observations)
            probe_backup_target = probe_rewards + (1.0 - probe_dones) * discount * probe_next_v_old
            td_error_probe = probe_q_online_mean - probe_backup_target
            q_update_probe_mean = jnp.mean(base_probe - probe_q_online_all, axis=0)
            committed_q_probe = q_apply({"params": new_state.q_params}, probe_observations, probe_actions)

            ensemble_std_probe = jnp.mean(jnp.std(probe_q_online_all, axis=0))
            td_error_old = jnp.mean(q_apply({"params": old_state.q_params}, observations, actions), axis=0) - target

            # Scaled propagated outputs.  Since local derivatives are measured on unit-RMS source
            # directions, multiplying by source RMS reconstructs B h in the linear regime.
            h_v_fd_scaled = h_v_fd * h_v_source_rms
            h_q_fd_scaled = h_q_fd * h_q_source_rms
            h_total_fd_scaled = h_total_fd * h_total_source_rms
            h_v_ntk_scaled = h_v_ntk * h_v_source_rms
            h_q_ntk_scaled = h_q_ntk * h_q_source_rms
            linear_sum_fd = h_v_fd_scaled + h_q_fd_scaled

            h_v_fd_mean = jnp.mean(h_v_fd_scaled, axis=0)
            h_q_fd_mean = jnp.mean(h_q_fd_scaled, axis=0)
            h_total_fd_mean = jnp.mean(h_total_fd_scaled, axis=0)

            h_v_knn_fd_scaled = h_v_knn_fd * h_v_knn_source_rms
            h_q_knn_fd_scaled = h_q_knn_fd * h_q_knn_source_rms
            h_v_knn_fd_mean = jnp.mean(h_v_knn_fd_scaled, axis=0)
            h_q_knn_fd_mean = jnp.mean(h_q_knn_fd_scaled, axis=0)

            h_v_kernel_fd_scaled = h_v_kernel_fd * h_v_kernel_source_rms
            h_q_kernel_fd_scaled = h_q_kernel_fd * h_q_kernel_source_rms
            h_v_kernel_fd_mean = jnp.mean(h_v_kernel_fd_scaled, axis=0)
            h_q_kernel_fd_mean = jnp.mean(h_q_kernel_fd_scaled, axis=0)

            h_v_kernel_local_fd_scaled = h_v_kernel_local_fd * h_v_kernel_local_source_rms
            h_q_kernel_local_fd_scaled = h_q_kernel_local_fd * h_q_kernel_local_source_rms
            h_v_kernel_local_fd_mean = jnp.mean(h_v_kernel_local_fd_scaled, axis=0)
            h_q_kernel_local_fd_mean = jnp.mean(h_q_kernel_local_fd_scaled, axis=0)

            hq_lag_coeff, hq_lag_energy, hq_lag_harm = reinforcement_stats(h_q_fd_mean, eps_lag_probe)
            hv_lag_coeff, hv_lag_energy, hv_lag_harm = reinforcement_stats(h_v_fd_mean, eps_lag_probe)
            hq_td_coeff, hq_td_energy, hq_td_harm = reinforcement_stats(h_q_fd_mean, td_error_probe)
            hv_td_coeff, hv_td_energy, hv_td_harm = reinforcement_stats(h_v_fd_mean, td_error_probe)
            hq_knn_td_coeff, hq_knn_td_energy, hq_knn_td_harm = reinforcement_stats(h_q_knn_fd_mean, td_error_probe)
            hv_knn_td_coeff, hv_knn_td_energy, hv_knn_td_harm = reinforcement_stats(h_v_knn_fd_mean, td_error_probe)
            hq_kernel_td_coeff, hq_kernel_td_energy, hq_kernel_td_harm = reinforcement_stats(h_q_kernel_fd_mean, td_error_probe)
            hv_kernel_td_coeff, hv_kernel_td_energy, hv_kernel_td_harm = reinforcement_stats(h_v_kernel_fd_mean, td_error_probe)
            hq_kernel_local_td_coeff, hq_kernel_local_td_energy, hq_kernel_local_td_harm = reinforcement_stats(
                h_q_kernel_local_fd_mean, td_error_probe
            )
            hv_kernel_local_td_coeff, hv_kernel_local_td_energy, hv_kernel_local_td_harm = reinforcement_stats(
                h_v_kernel_local_fd_mean, td_error_probe
            )

            # End-to-end closed-loop proxies.  Unlike diag_gain/* (which measures only B
            # along a target-error direction), these compare the propagated critic change to
            # the current observable critic-lag error.  They are practical M-direction proxies
            # for the original recursion epsilon_{k+1} ~= gamma B D epsilon_k.
            lag_rms_probe = rms(eps_lag_probe)
            m_proxy_head = rms(h_q_fd_mean) / (lag_rms_probe + 1e-12)
            m_proxy_knn = rms(h_q_knn_fd_mean) / (lag_rms_probe + 1e-12)
            m_proxy_kernel = rms(h_q_kernel_fd_mean) / (lag_rms_probe + 1e-12)
            m_proxy_kernel_local = rms(h_q_kernel_local_fd_mean) / (lag_rms_probe + 1e-12)

            # Scale-normalized sources distinguish genuine relative error growth from the
            # trivial increase caused by Q/V magnitudes exploding together.
            q_value_scale = jnp.maximum(rms(probe_q_target), 1e-8)
            bellman_scale = jnp.maximum(target_rms, 1e-8)

            # V6 policy-bridge diagnostics.  Value instability matters for control only when
            # it changes the advantage-weighted behavior-cloning distribution or the actor.
            policy_adv_probe = jax.lax.stop_gradient(probe_q_target - probe_v_pre)
            policy_weight_probe = jnp.minimum(jnp.exp(beta * policy_adv_probe), EXP_ADV_MAX)
            policy_weight_sum = jnp.sum(policy_weight_probe) + 1e-12
            policy_weight_prob = policy_weight_probe / policy_weight_sum
            policy_ess_fraction = (policy_weight_sum ** 2) / (
                float(policy_weight_probe.shape[0]) * jnp.sum(policy_weight_probe ** 2) + 1e-12
            )
            policy_entropy_fraction = -jnp.sum(
                policy_weight_prob * jnp.log(policy_weight_prob + 1e-12)
            ) / jnp.log(float(policy_weight_probe.shape[0]) + 1e-12)
            policy_saturation_fraction = jnp.mean(
                (policy_weight_probe >= (EXP_ADV_MAX * (1.0 - 1e-6))).astype(jnp.float32)
            )

            def actor_mean(params):
                out = actor_apply_fn({"params": params}, probe_observations, training=False)
                return out if iql_deterministic else out[0]

            actor_mean_old = actor_mean(old_state.actor_params)
            actor_mean_new = actor_mean(new_state.actor_params)
            actor_mean_delta = actor_mean_new - actor_mean_old

            def weight_mass_top_decile(signal):
                threshold = jnp.quantile(jnp.asarray(signal), 0.90)
                mask = (jnp.asarray(signal) >= threshold).astype(jnp.float32)
                return jnp.sum(policy_weight_probe * mask) / policy_weight_sum

            abs_lag_probe = jnp.abs(eps_lag_probe)
            abs_td_probe = jnp.abs(td_error_probe)
            abs_residual_local_probe = jnp.abs(d_kernel_local_probe)

            log_dict = {
                # ---- Residual-source diagnostics and head calibration ----
                "diag_residual/raw_score_mean_probe": jnp.mean(score_pre),
                "diag_residual/raw_score_abs_mean_probe": jnp.mean(jnp.abs(score_pre)),
                "diag_residual/kappa_batch_mean_probe": jnp.mean(weight_pre),
                "diag_residual/g_hat_rms_probe": rms(g_hat_probe),
                "diag_residual/g_hat_abs_mean_probe": jnp.mean(jnp.abs(g_hat_probe)),
                "diag_residual/kappa_hat_mean_probe": jnp.mean(kappa_hat_probe),
                "diag_residual/kappa_hat_min_probe": jnp.min(kappa_hat_probe),
                "diag_residual/value_error_hat_rms_probe": rms(-g_hat_probe / kappa_hat_probe),
                "diag_residual/value_error_hat_abs_p50_probe": abs_quantile(-g_hat_probe / kappa_hat_probe, 0.50),
                "diag_residual/value_error_hat_abs_p90_probe": abs_quantile(-g_hat_probe / kappa_hat_probe, 0.90),
                "diag_residual/value_error_hat_abs_p99_probe": abs_quantile(-g_hat_probe / kappa_hat_probe, 0.99),
                "diag_residual/head_g_mse_probe": g_mse,
                "diag_residual/head_kappa_mse_probe": kappa_mse,
                "diag_residual/head_qnum_mse_probe": qnum_mse,
                "diag_residual/head_g_nmse_vs_zero_probe": g_mse / (g_zero_mse + 1e-12),
                "diag_residual/head_g_skill_vs_zero_probe": 1.0 - g_mse / (g_zero_mse + 1e-12),
                "diag_residual/head_kappa_nmse_vs_mean_probe": kappa_mse / (kappa_mean_baseline_mse + 1e-12),
                "diag_residual/head_qnum_nmse_vs_zero_probe": qnum_mse / (qnum_zero_mse + 1e-12),
                "diag_residual/head_g_raw_corr_probe": corr(g_hat_probe, score_pre),
                "diag_residual/head_kappa_raw_corr_probe": corr(kappa_hat_probe, weight_pre),
                "diag_residual/head_qnum_raw_corr_probe": corr(q_num_hat_probe, q_num_target_probe),
                "diag_residual/head_g_binned_cal_corr_probe": g_cal_corr,
                "diag_residual/head_g_binned_cal_mse_probe": g_cal_mse,
                "diag_residual/head_g_binned_sign_agree_probe": g_cal_sign,
                "diag_residual/head_kappa_binned_cal_corr_probe": kappa_cal_corr,
                "diag_residual/head_kappa_binned_cal_mse_probe": kappa_cal_mse,
                "diag_residual/head_qnum_binned_cal_corr_probe": qnum_cal_corr,
                "diag_residual/head_qnum_binned_cal_mse_probe": qnum_cal_mse,
                "diag_residual/head_qnum_binned_sign_agree_probe": qnum_cal_sign,

                # Independent kNN conditional reference: useful when the learned head has
                # near-zero skill.  Radius/SE indicate when the non-parametric reference is weak.
                "diag_knn/probe_neighbor_radius_mean": jnp.mean(knn_probe_radius),
                "diag_knn/probe_neighbor_radius_p90": jnp.quantile(knn_probe_radius, 0.90),
                "diag_knn/next_neighbor_radius_mean": jnp.mean(knn_next_radius),
                "diag_knn/next_neighbor_radius_p90": jnp.quantile(knn_next_radius, 0.90),
                "diag_knn/g_se_mean_probe": jnp.mean(g_knn_se_probe),
                "diag_knn/kappa_se_mean_probe": jnp.mean(kappa_knn_se_probe),
                "diag_knn/qnum_se_mean_probe": jnp.mean(qnum_knn_se_probe),
                "diag_knn/g_se_mean_next": jnp.mean(g_knn_se_next),
                "diag_knn/kappa_se_mean_next": jnp.mean(kappa_knn_se_next),
                "diag_residual/g_knn_rms_probe": rms(g_knn_probe),
                "diag_residual/kappa_knn_mean_probe": jnp.mean(kappa_knn_probe),
                "diag_residual/value_error_knn_rms_probe": rms(d_knn_probe),
                "diag_residual/head_vs_knn_g_corr_probe": corr(g_hat_probe, g_knn_probe),
                "diag_residual/head_vs_knn_kappa_corr_probe": corr(kappa_hat_probe, kappa_knn_probe),
                "diag_residual/head_vs_knn_qnum_corr_probe": corr(q_num_hat_probe, qnum_knn_probe),
                "diag_residual/head_vs_knn_d_corr_probe": corr(-g_hat_probe / kappa_hat_probe, d_knn_probe),

                # Kernel conditional-moment diagnostics: direct tests of E[psi_tau(Q-V)|s]=0
                # that do not require fitting another residual head.
                "diag_kernel/bandwidth_sigma2": kernel_sigma2,
                "diag_kernel/mmr_raw_u": mmr_raw_u,
                "diag_kernel/mmr_kernel_normalized": mmr_kernel_norm,
                "diag_kernel/mmr_score_normalized": mmr_score_norm,
                "diag_kernel/mmr_positive_root": mmr_root,
                "diag_kernel/crossfit_moment_energy": mmr_crossfit,
                "diag_kernel/crossfit_moment_normalized": mmr_crossfit_norm,
                "diag_kernel/crossfit_sign_agree": mmr_crossfit_sign,
                "diag_kernel/g_rms_probe": rms(g_kernel_probe),
                "diag_kernel/g_se_mean_probe": jnp.mean(g_kernel_se_probe),
                "diag_kernel/kappa_mean_probe": jnp.mean(kappa_kernel_probe),
                "diag_kernel/kappa_se_mean_probe": jnp.mean(kappa_kernel_se_probe),
                "diag_kernel/qnum_se_mean_probe": jnp.mean(qnum_kernel_se_probe),
                "diag_kernel/effective_neighbors_mean_probe": jnp.mean(kernel_neff_probe),
                "diag_kernel/effective_neighbors_p10_probe": jnp.quantile(kernel_neff_probe, 0.10),
                "diag_kernel/value_error_rms_probe": rms(d_kernel_probe),
                "diag_kernel/value_error_abs_p90_probe": abs_quantile(d_kernel_probe, 0.90),
                "diag_kernel/head_vs_g_corr_probe": corr(g_hat_probe, g_kernel_probe),
                "diag_kernel/knn_vs_g_corr_probe": corr(g_knn_probe, g_kernel_probe),
                "diag_kernel/head_vs_d_corr_probe": corr(-g_hat_probe / kappa_hat_probe, d_kernel_probe),
                "diag_kernel/knn_vs_d_corr_probe": corr(d_knn_probe, d_kernel_probe),
                "diag_kernel/kappa_from_ppos_sanity_rms": rms(kappa_kernel_probe - kappa_from_ppos_probe),

                # Adaptive local kernel: same conditional-moment idea, but bandwidth is
                # calibrated to a fixed effective support instead of median distance.
                "diag_kernel_local/bandwidth_sigma2": kernel_local_sigma2,
                "diag_kernel_local/target_neff": jnp.asarray(kernel_target_neff, dtype=jnp.float32),
                "diag_kernel_local/achieved_neff_mean": kernel_local_neff_target_achieved,
                "diag_kernel_local/neff_p10_probe": jnp.quantile(kernel_local_neff_probe, 0.10),
                "diag_kernel_local/mmr_score_normalized": mmr_local_score_norm,
                "diag_kernel_local/crossfit_moment_normalized": mmr_local_crossfit_norm,
                "diag_kernel_local/crossfit_sign_agree": mmr_local_crossfit_sign,
                "diag_kernel_local/g_rms_probe": rms(g_kernel_local_probe),
                "diag_kernel_local/g_se_mean_probe": jnp.mean(g_kernel_local_se_probe),
                "diag_kernel_local/kappa_mean_probe": jnp.mean(kappa_kernel_local_probe),
                "diag_kernel_local/kappa_se_mean_probe": jnp.mean(kappa_kernel_local_se_probe),
                "diag_kernel_local/value_error_rms_probe": rms(d_kernel_local_probe),
                "diag_kernel_local/head_vs_d_corr_probe": corr(-g_hat_probe / kappa_hat_probe, d_kernel_local_probe),
                "diag_kernel_local/knn_vs_d_corr_probe": corr(d_knn_probe, d_kernel_local_probe),
                "diag_kernel_local/broad_vs_d_corr_probe": corr(d_kernel_probe, d_kernel_local_probe),
                "diag_kernel_local/balance_abs_mean_probe": jnp.mean(jnp.abs(balance_kernel_local_probe)),
                "diag_kernel_local/balance_abs_p90_probe": abs_quantile(balance_kernel_local_probe, 0.90),

                # V7 conditioning-aware displacement diagnostics.  The delta-method SE is
                # heuristic because G and kappa share neighbors, but it exposes whether a
                # large -G/kappa is actually distinguishable from local conditional noise.
                "diag_residual_ca/local_d_se_mean_probe": jnp.mean(d_kernel_local_se_probe),
                "diag_residual_ca/local_d_snr_p50_probe": jnp.quantile(d_kernel_local_snr_probe, 0.50),
                "diag_residual_ca/local_d_snr_p90_probe": jnp.quantile(d_kernel_local_snr_probe, 0.90),
                "diag_residual_ca/local_d_fraction_snr_gt1_probe": jnp.mean(
                    (d_kernel_local_snr_probe > 1.0).astype(jnp.float32)
                ),
                "diag_residual_ca/local_d_fraction_snr_gt2_probe": jnp.mean(
                    (d_kernel_local_snr_probe > 2.0).astype(jnp.float32)
                ),
                "diag_residual_ca/local_d_noise_corrected_rms_probe": d_kernel_local_noise_corrected_rms,
                "diag_residual_ca/local_g_snr_p50_probe": jnp.quantile(g_kernel_local_snr_probe, 0.50),
                "diag_residual_ca/local_g_snr_p90_probe": jnp.quantile(g_kernel_local_snr_probe, 0.90),
                "diag_residual_ca/local_kappa_relative_se_p50_probe": jnp.quantile(
                    kappa_kernel_local_se_probe / jnp.maximum(kappa_kernel_local_probe, kappa_floor), 0.50
                ),
                "diag_residual_ca/local_kappa_relative_se_p90_probe": jnp.quantile(
                    kappa_kernel_local_se_probe / jnp.maximum(kappa_kernel_local_probe, kappa_floor), 0.90
                ),
                "diag_residual_ca/crossfit_displacement_energy": conditioned_crossfit_energy,
                "diag_residual_ca/crossfit_displacement_root": conditioned_crossfit_root,
                "diag_residual_ca/crossfit_displacement_normalized": conditioned_crossfit_norm,
                "diag_residual_ca/crossfit_displacement_sign_agree": conditioned_crossfit_sign,
                "diag_residual_ca/next_d_se_mean": jnp.mean(d_v_kernel_local_se_next),
                "diag_residual_ca/next_d_snr_p50": jnp.quantile(d_v_kernel_local_snr_next, 0.50),
                "diag_residual_ca/next_d_snr_p90": jnp.quantile(d_v_kernel_local_snr_next, 0.90),
                "diag_residual_ca/next_d_fraction_snr_gt2": jnp.mean(
                    (d_v_kernel_local_snr_next > 2.0).astype(jnp.float32)
                ),

                # Multi-scale sensitivity.  A real conditional violation should not exist
                # only at the single over-smoothed bandwidth used in V4.
                "diag_kernel_ms/mmr_score_s0125": kernel_ms_score[0],
                "diag_kernel_ms/mmr_score_s025": kernel_ms_score[1],
                "diag_kernel_ms/mmr_score_s05": kernel_ms_score[2],
                "diag_kernel_ms/mmr_score_s1": kernel_ms_score[3],
                "diag_kernel_ms/mmr_score_s2": kernel_ms_score[4],
                "diag_kernel_ms/crossfit_s0125": kernel_ms_crossfit[0],
                "diag_kernel_ms/crossfit_s025": kernel_ms_crossfit[1],
                "diag_kernel_ms/crossfit_s05": kernel_ms_crossfit[2],
                "diag_kernel_ms/crossfit_s1": kernel_ms_crossfit[3],
                "diag_kernel_ms/crossfit_s2": kernel_ms_crossfit[4],
                "diag_kernel_ms/neff_s0125": kernel_ms_neff[0],
                "diag_kernel_ms/neff_s025": kernel_ms_neff[1],
                "diag_kernel_ms/neff_s05": kernel_ms_neff[2],
                "diag_kernel_ms/neff_s1": kernel_ms_neff[3],
                "diag_kernel_ms/neff_s2": kernel_ms_neff[4],
                "diag_kernel_ms/max_abs_crossfit": jnp.max(jnp.abs(kernel_ms_crossfit)),
                "diag_kernel_ms/max_abs_mmr_score": jnp.max(jnp.abs(kernel_ms_score)),

                # Kappa-head-as-classifier calibration.
                "diag_kappa_cls/p_pos_hat_mean": jnp.mean(ppos_hat_probe),
                "diag_kappa_cls/p_pos_hat_std": jnp.std(ppos_hat_probe),
                "diag_kappa_cls/p_pos_empirical_mean": ppos_prevalence,
                "diag_kappa_cls/brier": ppos_brier,
                "diag_kappa_cls/brier_skill_vs_constant": ppos_skill,
                "diag_kappa_cls/logloss": ppos_logloss,
                "diag_kappa_cls/binned_cal_corr": ppos_cal_corr,
                "diag_kappa_cls/binned_cal_mse": ppos_cal_mse,

                # Expectile balance decomposition: exact FOC means positive and negative
                # weighted masses match conditionally.  The normalized imbalance is scale-free.
                "diag_balance/positive_mass_mean_probe": jnp.mean(pos_moment_probe),
                "diag_balance/negative_mass_mean_probe": jnp.mean(neg_moment_probe),
                "diag_balance/global_normalized_imbalance": (
                    jnp.mean(pos_moment_probe) - jnp.mean(neg_moment_probe)
                ) / (jnp.mean(pos_moment_probe) + jnp.mean(neg_moment_probe) + 1e-12),
                "diag_balance/kernel_abs_mean_probe": jnp.mean(jnp.abs(balance_kernel_probe)),
                "diag_balance/kernel_abs_p90_probe": abs_quantile(balance_kernel_probe, 0.90),
                "diag_balance/kernel_abs_p99_probe": abs_quantile(balance_kernel_probe, 0.99),
                "diag_balance/kernel_positive_mass_mean_probe": jnp.mean(pos_kernel_probe),
                "diag_balance/kernel_negative_mass_mean_probe": jnp.mean(neg_kernel_probe),
                "diag_balance/p_pos_mean_probe": jnp.mean(pos_indicator_probe),

                # Direct theory calibration for dG/dV = -kappa.
                "diag_conditioning/foc_shift": foc_shift,
                "diag_conditioning/foc_dG_dV_fd": foc_dg_dv_fd,
                "diag_conditioning/foc_minus_slope_over_kappa": foc_slope_over_kappa,
                "diag_conditioning/inv_kappa_head_p50": jnp.quantile(1.0 / jnp.maximum(kappa_hat_probe, kappa_floor), 0.50),
                "diag_conditioning/inv_kappa_head_p90": jnp.quantile(1.0 / jnp.maximum(kappa_hat_probe, kappa_floor), 0.90),
                "diag_conditioning/inv_kappa_kernel_p50": jnp.quantile(1.0 / jnp.maximum(kappa_kernel_probe, kappa_floor), 0.50),
                "diag_conditioning/inv_kappa_kernel_p90": jnp.quantile(1.0 / jnp.maximum(kappa_kernel_probe, kappa_floor), 0.90),

                # ---- Correctability / under-training diagnostics ----
                "diag_residual/probe_v_loss_pre": probe_v_loss_pre,
                "diag_residual/probe_v_loss_post": probe_v_loss_post,
                "diag_residual/probe_v_loss_ratio": probe_v_loss_post / (probe_v_loss_pre + 1e-12),
                "diag_residual/global_d_pre": d_global_pre,
                "diag_residual/global_d_post": d_global_post,
                "diag_residual/global_d_abs_ratio": jnp.abs(d_global_post) / (jnp.abs(d_global_pre) + 1e-8),
                "diag_residual/binned_d_rms_pre": d_bin_pre_rms,
                "diag_residual/binned_d_rms_post": d_bin_post_rms,
                "diag_residual/binned_d_rms_post_over_pre": d_bin_post_rms / (d_bin_pre_rms + 1e-12),
                "diag_residual/binned_d_abs_mean_pre": d_bin_pre_abs,
                "diag_residual/binned_d_abs_max_pre": d_bin_pre_max,
                "diag_residual/binned_d_abs_max_post": d_bin_post_max,
                "diag_residual/v_grad_norm_pre": tree_l2(v_grad_pre),
                "diag_residual/v_grad_norm_post": tree_l2(v_grad_post),
                "diag_residual/v_grad_norm_ratio": tree_l2(v_grad_post) / (tree_l2(v_grad_pre) + 1e-12),
                f"diag_residual/probe_v_loss_virtual_{v_short_steps}": probe_v_loss_short,
                f"diag_residual/probe_v_loss_virtual_{v_long_steps}": probe_v_loss_long,
                f"diag_residual/probe_v_loss_virtual_{v_short_steps}_over_pre": probe_v_loss_short / (probe_v_loss_pre + 1e-12),
                f"diag_residual/probe_v_loss_virtual_{v_long_steps}_over_pre": probe_v_loss_long / (probe_v_loss_pre + 1e-12),
                f"diag_residual/global_d_virtual_{v_short_steps}": d_short,
                f"diag_residual/global_d_virtual_{v_long_steps}": d_long,
                f"diag_residual/global_d_virtual_{v_short_steps}_abs_over_pre": jnp.abs(d_short) / (jnp.abs(d_global_pre) + 1e-8),
                f"diag_residual/global_d_virtual_{v_long_steps}_abs_over_pre": jnp.abs(d_long) / (jnp.abs(d_global_pre) + 1e-8),
                f"diag_residual/binned_d_rms_virtual_{v_short_steps}": d_bin_short_rms,
                f"diag_residual/binned_d_rms_virtual_{v_long_steps}": d_bin_long_rms,
                f"diag_residual/binned_d_rms_virtual_{v_short_steps}_over_pre": d_bin_short_rms / (d_bin_pre_rms + 1e-12),
                f"diag_residual/binned_d_rms_virtual_{v_long_steps}_over_pre": d_bin_long_rms / (d_bin_pre_rms + 1e-12),

                # ---- Observable source proxies ----
                "diag_source/h_v_rms": h_v_source_rms,
                "diag_source/h_v_abs_p90": abs_quantile(h_v, 0.90),
                "diag_source/h_v_abs_p99": abs_quantile(h_v, 0.99),
                "diag_source/h_q_lag_rms": h_q_source_rms,
                "diag_source/h_q_lag_abs_p90": abs_quantile(h_q_lag, 0.90),
                "diag_source/h_q_lag_abs_p99": abs_quantile(h_q_lag, 0.99),
                "diag_source/h_total_proxy_rms": h_total_source_rms,
                "diag_source/h_q_vs_h_v_cos": cosine(h_q_lag, h_v),
                "diag_source/h_v_knn_rms": h_v_knn_source_rms,
                "diag_source/h_q_knn_rms": h_q_knn_source_rms,
                "diag_source/h_v_head_vs_knn_cos": cosine(h_v, h_v_knn),
                "diag_source/h_q_head_vs_knn_cos": cosine(h_q_lag, h_q_knn),
                "diag_source/h_v_knn_over_head_rms": h_v_knn_source_rms / (h_v_source_rms + 1e-12),
                "diag_source/h_q_knn_over_head_rms": h_q_knn_source_rms / (h_q_source_rms + 1e-12),
                "diag_source/h_v_kernel_rms": h_v_kernel_source_rms,
                "diag_source/h_q_kernel_rms": h_q_kernel_source_rms,
                "diag_source/h_v_head_vs_kernel_cos": cosine(h_v, h_v_kernel),
                "diag_source/h_q_head_vs_kernel_cos": cosine(h_q_lag, h_q_kernel),
                "diag_source/h_v_knn_vs_kernel_cos": cosine(h_v_knn, h_v_kernel),
                "diag_source/h_q_knn_vs_kernel_cos": cosine(h_q_knn, h_q_kernel),
                "diag_source/q_target_lag_rms_probe": rms(eps_lag_probe),
                "diag_source/q_ensemble_std_mean_probe": ensemble_std_probe,
                "diag_source/td_error_rms_train": rms(td_error_old),
                "diag_source/td_error_rms_probe": rms(td_error_probe),

                # ---- Scale-normalized source/propagation diagnostics ----
                "diag_scale/probe_q_target_rms": q_value_scale,
                "diag_scale/bellman_target_rms": bellman_scale,
                "diag_scale/probe_v_rms": rms(probe_v_pre),
                "diag_scale/value_error_head_over_q": rms(-g_hat_probe / kappa_hat_probe) / q_value_scale,
                "diag_scale/value_error_knn_over_q": rms(d_knn_probe) / q_value_scale,
                "diag_scale/value_error_kernel_broad_over_q": rms(d_kernel_probe) / q_value_scale,
                "diag_scale/value_error_kernel_local_over_q": rms(d_kernel_local_probe) / q_value_scale,
                "diag_scale/h_v_head_over_target": h_v_source_rms / bellman_scale,
                "diag_scale/h_q_head_over_target": h_q_source_rms / bellman_scale,
                "diag_scale/h_v_knn_over_target": h_v_knn_source_rms / bellman_scale,
                "diag_scale/h_q_knn_over_target": h_q_knn_source_rms / bellman_scale,
                "diag_scale/h_v_kernel_broad_over_target": h_v_kernel_source_rms / bellman_scale,
                "diag_scale/h_q_kernel_broad_over_target": h_q_kernel_source_rms / bellman_scale,
                "diag_scale/h_v_kernel_local_over_target": h_v_kernel_local_source_rms / bellman_scale,
                "diag_scale/h_q_kernel_local_over_target": h_q_kernel_local_source_rms / bellman_scale,
                "diag_scale/prop_h_v_head_over_target": rms(h_v_fd_scaled) / bellman_scale,
                "diag_scale/prop_h_q_head_over_target": rms(h_q_fd_scaled) / bellman_scale,
                "diag_scale/prop_h_v_kernel_local_over_target": rms(h_v_kernel_local_fd_scaled) / bellman_scale,
                "diag_scale/prop_h_q_kernel_local_over_target": rms(h_q_kernel_local_fd_scaled) / bellman_scale,

                # ---- Policy bridge: does critic/value pathology reach AWBC extraction? ----
                "diag_policy/adv_rms_probe": rms(policy_adv_probe),
                "diag_policy/exp_adv_mean_probe": jnp.mean(policy_weight_probe),
                "diag_policy/exp_adv_p90_probe": jnp.quantile(policy_weight_probe, 0.90),
                "diag_policy/exp_adv_p99_probe": jnp.quantile(policy_weight_probe, 0.99),
                "diag_policy/exp_adv_max_probe": jnp.max(policy_weight_probe),
                "diag_policy/exp_adv_saturation_fraction": policy_saturation_fraction,
                "diag_policy/awbc_ess_fraction": policy_ess_fraction,
                "diag_policy/awbc_entropy_fraction": policy_entropy_fraction,
                "diag_policy/weight_vs_abs_lag_corr": corr(policy_weight_probe, abs_lag_probe),
                "diag_policy/weight_vs_abs_td_corr": corr(policy_weight_probe, abs_td_probe),
                "diag_policy/weight_vs_abs_residual_local_corr": corr(policy_weight_probe, abs_residual_local_probe),
                "diag_policy/weight_mass_top10_abs_lag": weight_mass_top_decile(abs_lag_probe),
                "diag_policy/weight_mass_top10_abs_td": weight_mass_top_decile(abs_td_probe),
                "diag_policy/weight_mass_top10_abs_residual_local": weight_mass_top_decile(abs_residual_local_probe),
                "diag_policy/actor_mean_rms_probe": rms(actor_mean_old),
                "diag_policy/actor_mean_delta_rms_probe": rms(actor_mean_delta),
                "diag_policy/actor_mean_delta_over_mean_rms": rms(actor_mean_delta) / (rms(actor_mean_old) + 1e-12),

                # ---- B finite-difference numerical validity ----
                "diag_B/fd_target_rms": target_rms,
                "diag_B/fd_effective_delta": fd_base_delta,
                "diag_B/fd_effective_delta_over_target_rms": fd_base_delta / (target_rms + 1e-12),
                "diag_B/random_source_rms": random_source_rms,
                "diag_B/fd_adam_random_gain_probe": rms(random_fd),
                "diag_B/fd_forward_random_gain_probe": rms(random_fd_forward),
                "diag_B/fd_small_random_gain_probe": rms(random_fd_small),
                "diag_B/fd_large_random_gain_probe": rms(random_fd_large),
                "diag_B/fd_central_vs_forward_cos_random": cosine(random_fd, random_fd_forward),
                "diag_B/fd_small_vs_base_cos_random": cosine(random_fd_small, random_fd),
                "diag_B/fd_large_vs_base_cos_random": cosine(random_fd_large, random_fd),
                "diag_B/fd_small_over_base_gain_random": rms(random_fd_small) / (rms(random_fd) + 1e-12),
                "diag_B/fd_large_over_base_gain_random": rms(random_fd_large) / (rms(random_fd) + 1e-12),
                "diag_B/fd_finite_fraction_random": finite_fraction(random_fd),
                "diag_B/jvp_adam_random_gain_probe": rms(random_adam_jvp),
                "diag_B/jvp_adam_finite_fraction_random": finite_fraction(random_adam_jvp),
                "diag_B/ntk_sgd_random_gain_probe": rms(random_ntk),
                "diag_B/ntk_finite_fraction_random": finite_fraction(random_ntk),
                "diag_B/fd_vs_jvp_adam_cos_random": cosine(random_fd, random_adam_jvp),
                "diag_B/fd_vs_ntk_cos_random": cosine(random_fd, random_ntk),
                "diag_B/ntk_over_fd_gain_random": rms(random_ntk) / (rms(random_fd) + 1e-12),

                # ---- Directional source-specific gain and realized propagation ----
                "diag_gain/h_v_fd_adam": rms(h_v_fd),
                "diag_gain/h_v_ntk_sgd": rms(h_v_ntk),
                "diag_gain/h_q_lag_fd_adam": rms(h_q_fd),
                "diag_gain/h_q_lag_ntk_sgd": rms(h_q_ntk),
                "diag_gain/h_total_proxy_fd_adam": rms(h_total_fd),
                "diag_gain/h_total_proxy_ntk_sgd": rms(h_total_ntk),
                "diag_prop/h_v_fd_adam_q_rms": rms(h_v_fd_scaled),
                "diag_prop/h_v_jvp_adam_q_rms": rms(h_v_adam_jvp) * h_v_source_rms,
                "diag_prop/h_v_ntk_sgd_q_rms": rms(h_v_ntk_scaled),
                "diag_prop/h_v_fd_vs_jvp_cos": cosine(h_v_fd, h_v_adam_jvp),
                "diag_prop/h_v_fd_vs_ntk_cos": cosine(h_v_fd, h_v_ntk),
                "diag_prop/h_v_ntk_over_fd_gain": rms(h_v_ntk) / (rms(h_v_fd) + 1e-12),
                "diag_prop/h_q_lag_fd_adam_q_rms": rms(h_q_fd_scaled),
                "diag_prop/h_q_lag_jvp_adam_q_rms": rms(h_q_adam_jvp) * h_q_source_rms,
                "diag_prop/h_q_lag_ntk_sgd_q_rms": rms(h_q_ntk_scaled),
                "diag_prop/h_q_lag_fd_vs_jvp_cos": cosine(h_q_fd, h_q_adam_jvp),
                "diag_prop/h_q_lag_fd_vs_ntk_cos": cosine(h_q_fd, h_q_ntk),
                "diag_prop/h_q_lag_ntk_over_fd_gain": rms(h_q_ntk) / (rms(h_q_fd) + 1e-12),
                "diag_prop/h_total_proxy_fd_adam_q_rms": rms(h_total_fd_scaled),
                "diag_prop/h_q_vs_h_v_fd_output_cos": cosine(h_q_fd_scaled, h_v_fd_scaled),
                "diag_prop/two_channel_linear_sum_cos": cosine(h_total_fd_scaled, linear_sum_fd),
                "diag_prop/two_channel_linear_sum_relerr": rms(h_total_fd_scaled - linear_sum_fd) / (rms(h_total_fd_scaled) + 1e-12),
                "diag_prop/h_v_fd_abs_p90": abs_quantile(h_v_fd_mean, 0.90),
                "diag_prop/h_q_lag_fd_abs_p90": abs_quantile(h_q_fd_mean, 0.90),
                "diag_gain/h_v_knn_fd_adam": rms(h_v_knn_fd),
                "diag_gain/h_q_knn_fd_adam": rms(h_q_knn_fd),
                "diag_prop/h_v_knn_fd_adam_q_rms": rms(h_v_knn_fd_scaled),
                "diag_prop/h_q_knn_fd_adam_q_rms": rms(h_q_knn_fd_scaled),
                "diag_prop/h_v_head_vs_knn_output_cos": cosine(h_v_fd_scaled, h_v_knn_fd_scaled),
                "diag_prop/h_q_head_vs_knn_output_cos": cosine(h_q_fd_scaled, h_q_knn_fd_scaled),
                "diag_gain/h_v_kernel_fd_adam": rms(h_v_kernel_fd),
                "diag_gain/h_q_kernel_fd_adam": rms(h_q_kernel_fd),
                "diag_prop/h_v_kernel_fd_adam_q_rms": rms(h_v_kernel_fd_scaled),
                "diag_prop/h_q_kernel_fd_adam_q_rms": rms(h_q_kernel_fd_scaled),
                "diag_prop/h_v_head_vs_kernel_output_cos": cosine(h_v_fd_scaled, h_v_kernel_fd_scaled),
                "diag_prop/h_q_head_vs_kernel_output_cos": cosine(h_q_fd_scaled, h_q_kernel_fd_scaled),
                "diag_prop/h_v_knn_vs_kernel_output_cos": cosine(h_v_knn_fd_scaled, h_v_kernel_fd_scaled),
                "diag_prop/h_q_knn_vs_kernel_output_cos": cosine(h_q_knn_fd_scaled, h_q_kernel_fd_scaled),
                "diag_source/h_v_kernel_local_rms": h_v_kernel_local_source_rms,
                "diag_source/h_q_kernel_local_rms": h_q_kernel_local_source_rms,
                "diag_source/h_v_kernel_local_vs_knn_cos": cosine(h_v_kernel_local, h_v_knn),
                "diag_source/h_q_kernel_local_vs_knn_cos": cosine(h_q_kernel_local, h_q_knn),
                "diag_gain/h_v_kernel_local_fd_adam": rms(h_v_kernel_local_fd),
                "diag_gain/h_q_kernel_local_fd_adam": rms(h_q_kernel_local_fd),
                "diag_prop/h_v_kernel_local_fd_adam_q_rms": rms(h_v_kernel_local_fd_scaled),
                "diag_prop/h_q_kernel_local_fd_adam_q_rms": rms(h_q_kernel_local_fd_scaled),
                "diag_prop/h_v_kernel_local_vs_knn_output_cos": cosine(h_v_kernel_local_fd_scaled, h_v_knn_fd_scaled),
                "diag_prop/h_q_kernel_local_vs_knn_output_cos": cosine(h_q_kernel_local_fd_scaled, h_q_knn_fd_scaled),

                # End-to-end critic-error recursion proxies.  Values > 1 mean that the
                # propagated component alone is larger than the current observable lag proxy.
                "diag_closed_loop/h_q_head_M_proxy_gain": m_proxy_head,
                "diag_closed_loop/h_q_head_M_proxy_margin": m_proxy_head - 1.0,
                "diag_closed_loop/h_q_head_M_proxy_expansive": (m_proxy_head > 1.0).astype(jnp.float32),
                "diag_closed_loop/h_q_knn_M_proxy_gain": m_proxy_knn,
                "diag_closed_loop/h_q_knn_M_proxy_margin": m_proxy_knn - 1.0,
                "diag_closed_loop/h_q_knn_M_proxy_expansive": (m_proxy_knn > 1.0).astype(jnp.float32),
                "diag_closed_loop/h_q_kernel_M_proxy_gain": m_proxy_kernel,
                "diag_closed_loop/h_q_kernel_M_proxy_margin": m_proxy_kernel - 1.0,
                "diag_closed_loop/h_q_kernel_M_proxy_expansive": (m_proxy_kernel > 1.0).astype(jnp.float32),
                "diag_closed_loop/h_q_kernel_local_M_proxy_gain": m_proxy_kernel_local,
                "diag_closed_loop/h_q_kernel_local_M_proxy_margin": m_proxy_kernel_local - 1.0,
                "diag_closed_loop/h_q_kernel_local_M_proxy_expansive": (m_proxy_kernel_local > 1.0).astype(jnp.float32),

                # Alignment: positive cosine with current lag/TD error means the propagated
                # perturbation points in the same direction as the existing error proxy.
                "diag_alignment/h_q_prop_with_lag_cos": cosine(h_q_fd_mean, eps_lag_probe),
                "diag_alignment/h_v_prop_with_lag_cos": cosine(h_v_fd_mean, eps_lag_probe),
                "diag_alignment/h_q_prop_with_td_error_cos": cosine(h_q_fd_mean, td_error_probe),
                "diag_alignment/h_v_prop_with_td_error_cos": cosine(h_v_fd_mean, td_error_probe),
                "diag_alignment/actual_q_update_with_negative_td_cos": cosine(q_update_probe_mean, -td_error_probe),

                # Signed reinforcement / predicted error-energy change.  Positive values mean
                # the propagated perturbation reinforces the chosen existing-error proxy.
                "diag_harm/h_q_with_lag_signed_injection": hq_lag_coeff,
                "diag_harm/h_q_with_lag_pred_rel_energy_change": hq_lag_energy,
                "diag_harm/h_q_with_lag_positive_risk": hq_lag_harm,
                "diag_harm/h_v_with_lag_signed_injection": hv_lag_coeff,
                "diag_harm/h_v_with_lag_pred_rel_energy_change": hv_lag_energy,
                "diag_harm/h_v_with_lag_positive_risk": hv_lag_harm,
                "diag_harm/h_q_with_td_signed_injection": hq_td_coeff,
                "diag_harm/h_q_with_td_pred_rel_energy_change": hq_td_energy,
                "diag_harm/h_q_with_td_positive_risk": hq_td_harm,
                "diag_harm/h_v_with_td_signed_injection": hv_td_coeff,
                "diag_harm/h_v_with_td_pred_rel_energy_change": hv_td_energy,
                "diag_harm/h_v_with_td_positive_risk": hv_td_harm,
                "diag_harm/h_q_knn_with_td_signed_injection": hq_knn_td_coeff,
                "diag_harm/h_q_knn_with_td_pred_rel_energy_change": hq_knn_td_energy,
                "diag_harm/h_q_knn_with_td_positive_risk": hq_knn_td_harm,
                "diag_harm/h_v_knn_with_td_signed_injection": hv_knn_td_coeff,
                "diag_harm/h_v_knn_with_td_pred_rel_energy_change": hv_knn_td_energy,
                "diag_harm/h_v_knn_with_td_positive_risk": hv_knn_td_harm,
                "diag_harm/h_q_kernel_with_td_signed_injection": hq_kernel_td_coeff,
                "diag_harm/h_q_kernel_with_td_pred_rel_energy_change": hq_kernel_td_energy,
                "diag_harm/h_q_kernel_with_td_positive_risk": hq_kernel_td_harm,
                "diag_harm/h_v_kernel_with_td_signed_injection": hv_kernel_td_coeff,
                "diag_harm/h_v_kernel_with_td_pred_rel_energy_change": hv_kernel_td_energy,
                "diag_harm/h_v_kernel_with_td_positive_risk": hv_kernel_td_harm,
                "diag_harm/h_q_kernel_local_with_td_signed_injection": hq_kernel_local_td_coeff,
                "diag_harm/h_q_kernel_local_with_td_pred_rel_energy_change": hq_kernel_local_td_energy,
                "diag_harm/h_q_kernel_local_with_td_positive_risk": hq_kernel_local_td_harm,
                "diag_harm/h_v_kernel_local_with_td_signed_injection": hv_kernel_local_td_coeff,
                "diag_harm/h_v_kernel_local_with_td_pred_rel_energy_change": hv_kernel_local_td_energy,
                "diag_harm/h_v_kernel_local_with_td_positive_risk": hv_kernel_local_td_harm,

                # Actual-scale locality check: whether the linearized B remains predictive at
                # the magnitude of the currently estimated source (clipped for safety).
                "diag_locality/h_v_applied_alpha": h_v_actual_alpha,
                "diag_locality/h_v_actual_scale_gain": h_v_actual_gain,
                "diag_locality/h_v_actual_vs_linear_cos": h_v_actual_cos,
                "diag_locality/h_v_actual_vs_linear_relerr": h_v_actual_relerr,
                "diag_locality/h_q_lag_applied_alpha": h_q_actual_alpha,
                "diag_locality/h_q_lag_actual_scale_gain": h_q_actual_gain,
                "diag_locality/h_q_lag_actual_vs_linear_cos": h_q_actual_cos,
                "diag_locality/h_q_lag_actual_vs_linear_relerr": h_q_actual_relerr,
                "diag_amplitude/h_v_alpha25": h_v_amp25[0],
                "diag_amplitude/h_v_gain25": h_v_amp25[1],
                "diag_amplitude/h_v_cos25": h_v_amp25[2],
                "diag_amplitude/h_v_relerr25": h_v_amp25[3],
                "diag_amplitude/h_v_alpha50": h_v_amp50[0],
                "diag_amplitude/h_v_gain50": h_v_amp50[1],
                "diag_amplitude/h_v_cos50": h_v_amp50[2],
                "diag_amplitude/h_v_relerr50": h_v_amp50[3],
                "diag_amplitude/h_v_alpha100": h_v_amp100[0],
                "diag_amplitude/h_v_gain100": h_v_amp100[1],
                "diag_amplitude/h_v_cos100": h_v_amp100[2],
                "diag_amplitude/h_v_relerr100": h_v_amp100[3],
                "diag_amplitude/h_q_alpha25": h_q_amp25[0],
                "diag_amplitude/h_q_gain25": h_q_amp25[1],
                "diag_amplitude/h_q_cos25": h_q_amp25[2],
                "diag_amplitude/h_q_relerr25": h_q_amp25[3],
                "diag_amplitude/h_q_alpha50": h_q_amp50[0],
                "diag_amplitude/h_q_gain50": h_q_amp50[1],
                "diag_amplitude/h_q_cos50": h_q_amp50[2],
                "diag_amplitude/h_q_relerr50": h_q_amp50[3],
                "diag_amplitude/h_q_alpha100": h_q_amp100[0],
                "diag_amplitude/h_q_gain100": h_q_amp100[1],
                "diag_amplitude/h_q_cos100": h_q_amp100[2],
                "diag_amplitude/h_q_relerr100": h_q_amp100[3],

                # ---- Per-target spillover / cross-sample coupling ----
                "diag_B/basis_fd_adam_self_abs_mean": fd_self,
                "diag_B/basis_fd_adam_spillover_l2_mean": fd_spill,
                "diag_B/basis_fd_adam_spillover_ratio_mean": fd_spill_ratio,
                "diag_B/basis_fd_adam_spillover_rms_per_coord_mean": fd_spill_rms_coord,
                "diag_B/basis_fd_adam_spillover_per_coord_ratio_mean": fd_spill_coord_ratio,
                "diag_B/basis_jvp_adam_self_abs_mean": adam_jvp_self,
                "diag_B/basis_jvp_adam_spillover_l2_mean": adam_jvp_spill,
                "diag_B/basis_jvp_adam_spillover_ratio_mean": adam_jvp_spill_ratio,
                "diag_B/basis_jvp_adam_spillover_rms_per_coord_mean": adam_jvp_spill_rms_coord,
                "diag_B/basis_jvp_adam_spillover_per_coord_ratio_mean": adam_jvp_spill_coord_ratio,
                "diag_B/basis_jvp_adam_finite_fraction": finite_fraction(basis_adam_jvp),
                "diag_B/basis_ntk_self_abs_mean": ntk_self,
                "diag_B/basis_ntk_spillover_l2_mean": ntk_spill,
                "diag_B/basis_ntk_spillover_ratio_mean": ntk_spill_ratio,
                "diag_B/basis_ntk_spillover_rms_per_coord_mean": ntk_spill_rms_coord,
                "diag_B/basis_ntk_spillover_per_coord_ratio_mean": ntk_spill_coord_ratio,

                # ---- Optimizer and target-network dynamics ----
                "diag_optimizer/q_grad_norm": tree_l2(base_q_grads),
                "diag_optimizer/q_update_param_norm": tree_l2(base_q_updates),
                "diag_optimizer/q_param_norm": tree_l2(old_state.q_params),
                "diag_optimizer/q_opt_state_norm": tree_l2(old_state.q_opt_state),
                "diag_optimizer/q_update_per_grad": tree_l2(base_q_updates) / (tree_l2(base_q_grads) + 1e-12),
                "diag_optimizer/q_output_change_rms_train": rms(base_train - q_apply({"params": old_state.q_params}, observations, actions)),
                "diag_optimizer/q_output_change_rms_probe": rms(base_probe - probe_q_online_all),
                "diag_optimizer/diagnostic_vs_committed_q_rms_probe": rms(base_probe - committed_q_probe),
                "diag_target_lag/lag_pre_rms_probe": rms(lag_pre),
                "diag_target_lag/lag_post_rms_probe": rms(lag_post),
                "diag_target_lag/lag_post_over_pre": rms(lag_post) / (rms(lag_pre) + 1e-12),
                "diag_target_lag/lag_delta_rms_probe": rms(lag_delta),
                "diag_target_lag/online_delta_rms_probe": rms(online_delta),
                "diag_target_lag/target_delta_rms_probe": rms(target_delta),
                "diag_target_lag/target_tracking_ratio": rms(target_delta) / (rms(online_delta) + 1e-12),
                "diag_target_lag/lag_delta_with_lag_pre_cos": cosine(lag_delta, lag_pre),
                "diag_target_lag/lag_post_with_lag_pre_cos": cosine(lag_post, lag_pre),
                "diag_target_lag/online_vs_target_delta_cos": cosine(online_delta, target_delta),
                "diag_target_lag/recurrence_sanity_rms": rms(lag_recurrence_error),
            }
            return log_dict

        return diagnostic_step

    def _build_residual_calibration_step(self):
        """Controlled calibration of residual estimators across amplitude and structure.

        V5 showed that the local-kernel estimator can recover the *direction* of a smooth
        injected V-displacement, but its magnitude is attenuated.  V6 therefore adds:
          1) small/base/large amplitude sweeps to reveal a detection floor/nonlinearity;
          2) direct power tests for the local-kernel MMR/cross-fit scalar diagnostics;
          3) random-sign injection as a negative control.
        No virtual perturbation is committed to training.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        iql_tau = self.iql_tau
        kappa_floor = self.diag_kappa_floor
        knn_k = self.diag_knn_k
        kernel_min_bandwidth = self.diag_kernel_min_bandwidth
        kernel_target_neff = self.diag_kernel_target_neff
        kernel_search_iters = self.diag_kernel_bandwidth_search_iters
        enable_conditioned_residual = self.diag_enable_conditioned_residual
        action_ref_k = self.diag_action_ref_k
        action_ref_bisection_iters = self.diag_action_ref_bisection_iters
        injection_fraction = self.diag_residual_injection_fraction
        small_mult = self.diag_residual_injection_small_multiplier
        large_mult = self.diag_residual_injection_large_multiplier
        diag_seed = self.diag_seed

        def rms(x):
            return jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2) + 1e-12)

        def cosine(a, b):
            a = jnp.ravel(a); b = jnp.ravel(b)
            return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)

        def corr(a, b):
            a = jnp.ravel(a); b = jnp.ravel(b)
            aa = a - jnp.mean(a); bb = b - jnp.mean(b)
            return jnp.sum(aa * bb) / (jnp.linalg.norm(aa) * jnp.linalg.norm(bb) + 1e-12)

        def knn_conditional(states, values):
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1) + jnp.eye(states.shape[0]) * 1e9
            _, idx = jax.lax.top_k(-dist2, knn_k)
            return jnp.mean(values[idx], axis=1)

        def local_sigma2(states):
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            n = states.shape[0]
            eye = jnp.eye(n, dtype=dist2.dtype)
            off = 1.0 - eye
            min_d = jnp.min(jnp.where(off > 0, jnp.maximum(dist2, 1e-12), jnp.inf))
            max_d = jnp.max(jnp.where(off > 0, dist2, 0.0))
            lo = jnp.maximum(min_d * 1e-3, kernel_min_bandwidth ** 2)
            hi = jnp.maximum(max_d * 10.0, lo * 10.0)
            target = jnp.minimum(kernel_target_neff, float(max(n - 1, 1)))

            def neff(s2):
                w = jnp.exp(-0.5 * dist2 / jnp.maximum(s2, 1e-12)) * off
                ws = jnp.sum(w, axis=1) + 1e-12
                return jnp.mean((ws ** 2) / (jnp.sum(w ** 2, axis=1) + 1e-12))

            def body(_, bounds):
                l, h = bounds
                m = jnp.sqrt(jnp.maximum(l * h, 1e-24))
                e = neff(m)
                return (jnp.where(e < target, m, l), jnp.where(e < target, h, m))

            lo2, hi2 = jax.lax.fori_loop(0, kernel_search_iters, body, (lo, hi))
            s2 = jnp.sqrt(jnp.maximum(lo2 * hi2, 1e-24))
            return s2, neff(s2)

        def kernel_weights(states, sigma2):
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.mean(diff ** 2, axis=-1)
            w = jnp.exp(-0.5 * dist2 / jnp.maximum(sigma2, 1e-12))
            return w * (1.0 - jnp.eye(states.shape[0], dtype=w.dtype))

        def kernel_conditional(states, values, sigma2):
            w = kernel_weights(states, sigma2)
            ws = jnp.sum(w, axis=1) + 1e-12
            return jnp.sum(w * values[None, :], axis=1) / ws

        def kernel_moment_energy(states, score, sigma2):
            w = kernel_weights(states, sigma2)
            pair = score[:, None] * w * score[None, :]
            return jnp.sum(pair) / (jnp.sum(w) + 1e-12)

        def crossfit_energy(states, score, sigma2):
            n = states.shape[0]
            split = n // 2
            sa, sb = states[:split], states[split:]
            ma, mb = score[:split], score[split:]

            def cross_cond(query, ref, values):
                diff = query[:, None, :] - ref[None, :, :]
                d2 = jnp.mean(diff ** 2, axis=-1)
                w = jnp.exp(-0.5 * d2 / jnp.maximum(sigma2, 1e-12))
                ws = jnp.sum(w, axis=1) + 1e-12
                return jnp.sum(w * values[None, :], axis=1) / ws

            ga = cross_cond(sb, sa, ma)
            gb = cross_cond(sa, sb, mb)
            return 0.5 * (jnp.mean(ga * mb) + jnp.mean(gb * ma))

        def conditioned_crossfit_energy(states, score, weight, sigma2):
            """Cross-fitted energy for the displacement d=-G/kappa."""
            n = states.shape[0]
            split = n // 2
            sa, sb = states[:split], states[split:]
            ma, mb = score[:split], score[split:]
            wa, wb = weight[:split], weight[split:]

            def cross_cond(query, ref, values):
                diff = query[:, None, :] - ref[None, :, :]
                d2 = jnp.mean(diff ** 2, axis=-1)
                w = jnp.exp(-0.5 * d2 / jnp.maximum(sigma2, 1e-12))
                ws = jnp.sum(w, axis=1) + 1e-12
                return jnp.sum(w * values[None, :], axis=1) / ws

            ga = cross_cond(sb, sa, ma)
            ka = jnp.maximum(cross_cond(sb, sa, wa), kappa_floor)
            gb = cross_cond(sa, sb, mb)
            kb = jnp.maximum(cross_cond(sa, sb, wb), kappa_floor)
            da = -ga / ka
            db = -gb / kb
            yb = -mb / ka
            ya = -ma / kb
            return 0.5 * (jnp.mean(da * yb) + jnp.mean(db * ya))

        def empirical_expectile(q_values):
            """Vectorized exact empirical expectile by monotone bisection over actions."""
            lo = jnp.min(q_values, axis=1)
            hi = jnp.max(q_values, axis=1)

            def body(_, bounds):
                l, h = bounds
                mid = 0.5 * (l + h)
                u = q_values - mid[:, None]
                w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                foc = jnp.mean(w * u, axis=1)
                # F(v) is decreasing in v.  If F(mid)>0, the root lies above mid.
                return jnp.where(foc > 0.0, mid, l), jnp.where(foc > 0.0, h, mid)

            lo, hi = jax.lax.fori_loop(
                0,
                action_ref_bisection_iters,
                body,
                (lo, hi),
            )
            return 0.5 * (lo + hi)

        @jax.jit
        def calibration_step(state: IQLState, probe_batch: TensorBatch, action_ref_batch: TensorBatch):
            s = probe_batch["observations"]
            a = probe_batch["actions"]
            q_target = jnp.min(q_apply({"params": state.q_target_params}, s, a), axis=0)
            v = v_apply({"params": state.v_params}, s)
            base_amp = injection_fraction * jnp.maximum(rms(q_target), 1.0)

            key = jax.random.fold_in(jax.random.PRNGKey(diag_seed + 404), state.total_it + 1)
            k1, k2 = jax.random.split(key)
            proj = jax.random.normal(k1, shape=(s.shape[-1],), dtype=s.dtype)
            smooth = s @ proj
            smooth = (smooth - jnp.mean(smooth)) / jnp.maximum(rms(smooth - jnp.mean(smooth)), 1e-8)
            random_dir = jax.random.rademacher(k2, shape=(s.shape[0],), dtype=s.dtype)
            random_dir = random_dir / jnp.maximum(rms(random_dir), 1e-8)
            sigma2, achieved_neff = local_sigma2(s)

            def score_weight(vv):
                u = jax.lax.stop_gradient(q_target) - vv
                w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                return w * u, w

            base_score, base_weight = score_weight(v)
            base_mmr = kernel_moment_energy(s, base_score, sigma2)
            base_crossfit = crossfit_energy(s, base_score, sigma2)
            base_conditioned_crossfit = conditioned_crossfit_energy(
                s, base_score, base_weight, sigma2
            ) if enable_conditioned_residual else jnp.asarray(jnp.nan, dtype=jnp.float32)

            # V7 local-action empirical expectile reference.  Neighbor actions are selected
            # by state proximity in a larger fixed reference bank, then Q is re-evaluated
            # at the *same query state*.  This approximates the behavior-conditioned
            # action distribution more directly than smoothing scores across different states.
            ref_s = action_ref_batch["observations"]
            ref_a = action_ref_batch["actions"]
            diff_ref = s[:, None, :] - ref_s[None, :, :]
            dist2_ref = jnp.mean(diff_ref ** 2, axis=-1)
            neg_dist_ref, ref_idx = jax.lax.top_k(-dist2_ref, action_ref_k)
            neigh_actions = ref_a[ref_idx]
            query_states = jnp.broadcast_to(s[:, None, :], (s.shape[0], action_ref_k, s.shape[-1]))
            flat_s = query_states.reshape((-1, s.shape[-1]))
            flat_a = neigh_actions.reshape((-1, neigh_actions.shape[-1]))
            q_target_bank = jnp.min(
                q_apply({"params": state.q_target_params}, flat_s, flat_a),
                axis=0,
            ).reshape((s.shape[0], action_ref_k))
            q_online_bank = jnp.min(
                q_apply({"params": state.q_params}, flat_s, flat_a),
                axis=0,
            ).reshape((s.shape[0], action_ref_k))
            v_ref_target = empirical_expectile(q_target_bank)
            v_ref_online = empirical_expectile(q_online_bank)
            d_v_action_ref = v - v_ref_target
            d_q_action_ref = v_ref_online - v_ref_target

            u_action = q_target_bank - v[:, None]
            w_action = jnp.abs(iql_tau - (u_action < 0.0).astype(jnp.float32))
            g_action = jnp.mean(w_action * u_action, axis=1)
            k_action = jnp.maximum(jnp.mean(w_action, axis=1), kappa_floor)
            d_v_action_linear = -g_action / k_action

            # Compare the existing kNN/local-kernel displacement estimates to the same-state
            # local-action reference on the unperturbed probe.
            g_knn_base = knn_conditional(s, base_score)
            k_knn_base = jnp.maximum(knn_conditional(s, base_weight), kappa_floor)
            d_knn_base = -g_knn_base / k_knn_base
            g_local_base = kernel_conditional(s, base_score, sigma2)
            k_local_base = jnp.maximum(kernel_conditional(s, base_weight, sigma2), kappa_floor)
            d_local_base = -g_local_base / k_local_base

            # Critic-shift proxy on the same probe: local smooth C_Q/kappa versus the direct
            # online-vs-target empirical expectile shift under the local action bank.
            q_obs_target = jnp.min(q_apply({"params": state.q_target_params}, s, a), axis=0)
            q_obs_online = jnp.min(q_apply({"params": state.q_params}, s, a), axis=0)
            qnum_obs = base_weight * (q_obs_online - q_obs_target)
            qnum_local = kernel_conditional(s, qnum_obs, sigma2)
            d_q_local = qnum_local / k_local_base

            def recover(direction, amp):
                sp, wp = score_weight(v + amp * direction)
                sm, wm = score_weight(v - amp * direction)

                gkp = knn_conditional(s, sp); gkm = knn_conditional(s, sm)
                kkp = knn_conditional(s, wp); kkm = knn_conditional(s, wm)
                rec_knn = ((-gkp / jnp.maximum(kkp, kappa_floor))
                           - (-gkm / jnp.maximum(kkm, kappa_floor))) / (2.0 * amp)

                glp = kernel_conditional(s, sp, sigma2); glm = kernel_conditional(s, sm, sigma2)
                klp = kernel_conditional(s, wp, sigma2); klm = kernel_conditional(s, wm, sigma2)
                rec_local = ((-glp / jnp.maximum(klp, kappa_floor))
                             - (-glm / jnp.maximum(klm, kappa_floor))) / (2.0 * amp)

                mmr_sym = 0.5 * (
                    kernel_moment_energy(s, sp, sigma2) + kernel_moment_energy(s, sm, sigma2)
                )
                cf_sym = 0.5 * (crossfit_energy(s, sp, sigma2) + crossfit_energy(s, sm, sigma2))
                mmr_delta = mmr_sym - base_mmr
                cf_delta = cf_sym - base_crossfit

                if enable_conditioned_residual:
                    cfp = conditioned_crossfit_energy(s, sp, wp, sigma2)
                    cfm = conditioned_crossfit_energy(s, sm, wm, sigma2)
                    cond_delta = 0.5 * (cfp + cfm) - base_conditioned_crossfit
                else:
                    cond_delta = jnp.asarray(jnp.nan, dtype=jnp.float32)

                base_kappa_sq = jnp.maximum(jnp.mean(base_weight), kappa_floor) ** 2
                return (
                    cosine(rec_knn, direction), rms(rec_knn),
                    rms(rec_knn - direction) / (rms(direction) + 1e-12), corr(rec_knn, direction),
                    cosine(rec_local, direction), rms(rec_local),
                    rms(rec_local - direction) / (rms(direction) + 1e-12), corr(rec_local, direction),
                    mmr_delta, mmr_delta / (amp ** 2 + 1e-12),
                    cf_delta, cf_delta / (amp ** 2 + 1e-12),
                    mmr_delta / (amp ** 2 * base_kappa_sq + 1e-12),
                    cf_delta / (amp ** 2 * base_kappa_sq + 1e-12),
                    cond_delta,
                    cond_delta / (amp ** 2 + 1e-12),
                )

            amp_small = base_amp * small_mult
            amp_large = base_amp * large_mult
            smooth_small = recover(smooth, amp_small)
            smooth_base = recover(smooth, base_amp)
            smooth_large = recover(smooth, amp_large)
            random_base = recover(random_dir, base_amp)

            action_ref_radius = jnp.sqrt(jnp.maximum(-neg_dist_ref[:, -1], 0.0))
            out = {
                "diag_rescal/injection_amplitude": base_amp,
                "diag_rescal/injection_amplitude_small": amp_small,
                "diag_rescal/injection_amplitude_large": amp_large,
                "diag_rescal/local_kernel_sigma2": sigma2,
                "diag_rescal/local_kernel_neff": achieved_neff,
                "diag_rescal/base_kernel_local_moment_energy": base_mmr,
                "diag_rescal/base_kernel_local_crossfit_energy": base_crossfit,
                "diag_rescal/base_conditioned_crossfit_energy": base_conditioned_crossfit,

                # V7 local-action empirical expectile reference.
                "diag_actionref/neighbor_radius_mean": jnp.mean(action_ref_radius),
                "diag_actionref/neighbor_radius_p90": jnp.quantile(action_ref_radius, 0.90),
                "diag_actionref/value_error_rms": rms(d_v_action_ref),
                "diag_actionref/critic_shift_rms": rms(d_q_action_ref),
                "diag_actionref/linearized_value_error_rms": rms(d_v_action_linear),
                "diag_actionref/linearized_vs_exact_cos": cosine(d_v_action_linear, d_v_action_ref),
                "diag_actionref/linearized_vs_exact_corr": corr(d_v_action_linear, d_v_action_ref),
                "diag_actionref/linearized_vs_exact_relerr": rms(d_v_action_linear - d_v_action_ref) / (
                    rms(d_v_action_ref) + 1e-12
                ),
                "diag_actionref/knn_vs_value_error_cos": cosine(d_knn_base, d_v_action_ref),
                "diag_actionref/knn_vs_value_error_corr": corr(d_knn_base, d_v_action_ref),
                "diag_actionref/kernel_local_vs_value_error_cos": cosine(d_local_base, d_v_action_ref),
                "diag_actionref/kernel_local_vs_value_error_corr": corr(d_local_base, d_v_action_ref),
                "diag_actionref/kernel_local_vs_value_error_relerr": rms(d_local_base - d_v_action_ref) / (
                    rms(d_v_action_ref) + 1e-12
                ),
                "diag_actionref/kernel_local_qshift_vs_exact_cos": cosine(d_q_local, d_q_action_ref),
                "diag_actionref/kernel_local_qshift_vs_exact_corr": corr(d_q_local, d_q_action_ref),
                "diag_actionref/kernel_local_qshift_vs_exact_relerr": rms(d_q_local - d_q_action_ref) / (
                    rms(d_q_action_ref) + 1e-12
                ),
            }

            # Backward-compatible V5 aliases use the base amplitude.
            names = ("knn_cos", "knn_gain", "knn_relerr", "knn_corr",
                     "kernel_local_cos", "kernel_local_gain", "kernel_local_relerr", "kernel_local_corr",
                     "mmr_delta", "mmr_delta_over_amp2", "crossfit_delta", "crossfit_delta_over_amp2",
                     "mmr_delta_over_amp2_kappa2", "crossfit_delta_over_amp2_kappa2",
                     "conditioned_crossfit_delta", "conditioned_crossfit_delta_over_amp2")
            for name, value in zip(names, smooth_base):
                out[f"diag_rescal/smooth_{name}"] = value
            for name, value in zip(names, random_base):
                out[f"diag_rescal/random_{name}"] = value

            for tag, stats in (("small", smooth_small), ("base", smooth_base), ("large", smooth_large)):
                for name, value in zip(names, stats):
                    out[f"diag_rescal/smooth_{name}_{tag}"] = value
            return out

        return calibration_step


    def _build_closed_loop_diagnostic_step(self):
        """Bellman-loop impulse and persistent-forcing diagnostics.

        V5 established that one-shot impulse gains Phi^(H) can remain below one even when
        frozen-target B^(H) is expansive.  The original recursion, however, contains a new
        forcing d_k at every iteration.  V6 therefore keeps the impulse response and adds a
        *forced response*: at every virtual step it injects the baseline run's current TD- or
        online-target-lag direction.  The forcing directions are computed from an unperturbed
        baseline rollout and are shared by +/- branches, so the central difference remains a
        local response measurement rather than a branch-dependent intervention.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        discount = self.discount
        target_tau = self.tau
        iql_tau = self.iql_tau
        fd_epsilon = self.diag_fd_epsilon
        fd_relative_scale = self.diag_fd_relative_scale
        short_steps = self.diag_loop_short_steps
        long_steps = self.diag_loop_long_steps
        enable_forced = self.diag_enable_forced_loop
        enable_actual_source_forcing = self.diag_enable_actual_source_forcing
        diag_seed = self.diag_seed

        def rms(x):
            return jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2) + 1e-12)

        def cosine(a, b):
            a = jnp.ravel(a); b = jnp.ravel(b)
            return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)

        def normdir(h):
            return h / jnp.maximum(rms(h), 1e-8)

        def sequence_coherence(seq, n):
            x = seq[:n]
            denom = jnp.sum(jax.vmap(rms)(x)) + 1e-12
            return rms(jnp.sum(x, axis=0)) / denom

        def consecutive_cosine(seq, n):
            x = seq[:n]
            if n <= 1:
                return jnp.asarray(1.0, dtype=x.dtype)
            a, b = x[1:], x[:-1]
            dot = jnp.sum(a * b, axis=1)
            den = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1) + 1e-12
            return jnp.mean(dot / den)

        def pair_cosine_mean(a, b, n):
            aa = a[:n]
            bb = b[:n]
            dot = jnp.sum(aa * bb, axis=1)
            den = jnp.linalg.norm(aa, axis=1) * jnp.linalg.norm(bb, axis=1) + 1e-12
            return jnp.mean(dot / den)

        @jax.jit
        def loop_step(old_state: IQLState, batch_seq: TensorBatch, probe_batch: TensorBatch):
            steps = batch_seq["observations"].shape[0]
            first_obs = batch_seq["observations"][0]
            first_actions = batch_seq["actions"][0]
            first_rewards = jnp.squeeze(batch_seq["rewards"][0], axis=-1)
            first_next = batch_seq["next_observations"][0]
            first_dones = jnp.squeeze(batch_seq["dones"][0], axis=-1)
            first_next_v = v_apply({"params": old_state.v_params}, first_next)
            first_target = first_rewards + (1.0 - first_dones) * discount * first_next_v
            delta = fd_epsilon * (jnp.maximum(rms(first_target), 1.0) if fd_relative_scale else 1.0)

            probe_obs = probe_batch["observations"]
            probe_act = probe_batch["actions"]
            probe_rew = jnp.squeeze(probe_batch["rewards"], axis=-1)
            probe_next = probe_batch["next_observations"]
            probe_done = jnp.squeeze(probe_batch["dones"], axis=-1)

            carry0 = (
                old_state.q_params, old_state.q_target_params, old_state.q_opt_state,
                old_state.v_params, old_state.v_opt_state,
            )

            def update_once(carry, obs, act, rew, nxt, done, extra_target):
                q_params, q_target_params, q_opt_state, v_params, v_opt_state = carry
                rew = jnp.squeeze(rew, axis=-1)
                done = jnp.squeeze(done, axis=-1)

                next_v = v_apply({"params": v_params}, nxt)
                q_backup = rew + (1.0 - done) * discount * next_v + extra_target
                target_qs = q_apply({"params": q_target_params}, obs, act)
                target_q_for_v = jax.lax.stop_gradient(jnp.min(target_qs, axis=0))

                def v_loss_fn(p):
                    vv = v_apply({"params": p}, obs)
                    u = target_q_for_v - vv
                    w = jnp.abs(iql_tau - (u < 0.0).astype(jnp.float32))
                    return jnp.mean(w * u ** 2)

                v_grads = jax.grad(v_loss_fn)(v_params)
                v_updates, v_opt_state_new = v_tx.update(v_grads, v_opt_state, v_params)
                v_params_new = optax.apply_updates(v_params, v_updates)

                def q_loss_fn(p):
                    qq = q_apply({"params": p}, obs, act)
                    return jnp.mean((qq - jax.lax.stop_gradient(q_backup)[None, :]) ** 2)

                q_grads = jax.grad(q_loss_fn)(q_params)
                q_updates, q_opt_state_new = q_tx.update(q_grads, q_opt_state, q_params)
                q_params_new = optax.apply_updates(q_params, q_updates)
                q_target_params_new = optax.incremental_update(q_params_new, q_target_params, target_tau)
                new_carry = (q_params_new, q_target_params_new, q_opt_state_new, v_params_new, v_opt_state_new)
                return new_carry

            def probe_metrics(carry):
                q_params, q_target_params, _, v_params, _ = carry
                q_online_all = q_apply({"params": q_params}, probe_obs, probe_act)
                q_target_all = q_apply({"params": q_target_params}, probe_obs, probe_act)
                probe_online_min = jnp.min(q_online_all, axis=0)
                probe_online_mean = jnp.mean(q_online_all, axis=0)
                probe_target_min = jnp.min(q_target_all, axis=0)
                lag = probe_online_min - probe_target_min
                pnext_v = v_apply({"params": v_params}, probe_next)
                pbackup = probe_rew + (1.0 - probe_done) * discount * pnext_v
                td = probe_online_mean - pbackup
                return probe_online_min, lag, td, pbackup

            # Unperturbed baseline rollout: supplies a realistic direction at every step and
            # logs whether the observable error on a fixed probe is persistent/coherent.
            def baseline_one(carry, xs):
                obs, act, rew, nxt, done = xs
                q_params, q_target_params, _, v_params, _ = carry
                rew_s = jnp.squeeze(rew, axis=-1)
                done_s = jnp.squeeze(done, axis=-1)
                next_v = v_apply({"params": v_params}, nxt)
                q_backup = rew_s + (1.0 - done_s) * discount * next_v
                q_online = q_apply({"params": q_params}, obs, act)
                q_target = q_apply({"params": q_target_params}, obs, act)
                h_td = jnp.mean(q_online, axis=0) - q_backup
                h_lag = jnp.min(q_online, axis=0) - jnp.min(q_target, axis=0)
                new_carry = update_once(carry, obs, act, rew, nxt, done, jnp.zeros_like(q_backup))
                p_online, p_lag, p_td, p_backup = probe_metrics(new_carry)
                return new_carry, (
                    normdir(h_td), normdir(h_lag), rms(h_td), rms(h_lag), rms(q_backup),
                    p_online, p_lag, p_td, p_backup,
                )

            xs = (
                batch_seq["observations"], batch_seq["actions"], batch_seq["rewards"],
                batch_seq["next_observations"], batch_seq["dones"],
            )
            _, baseline = jax.lax.scan(baseline_one, carry0, xs)
            (
                td_dirs, lag_dirs, td_rms_seq, lag_rms_seq, target_rms_seq,
                base_probe_q, base_probe_lag, base_probe_td, base_probe_backup,
            ) = baseline

            # V7 natural-innovation decomposition on one fixed probe.  The target-network
            # lag is an observable state variable; its growth can be decomposed into an
            # inherited component plus newly generated online-Q motion.  Because soft target
            # updates occur in parameter space, the output-space recurrence is approximate;
            # the residual is logged rather than assumed away.
            pre_probe_q, pre_probe_lag, pre_probe_td, pre_probe_backup = probe_metrics(carry0)
            probe_q_seq = jnp.concatenate([pre_probe_q[None, :], base_probe_q], axis=0)
            probe_lag_seq = jnp.concatenate([pre_probe_lag[None, :], base_probe_lag], axis=0)
            probe_backup_seq = jnp.concatenate([pre_probe_backup[None, :], base_probe_backup], axis=0)
            probe_q_delta = probe_q_seq[1:] - probe_q_seq[:-1]
            probe_backup_delta = probe_backup_seq[1:] - probe_backup_seq[:-1]
            inherited_lag = (1.0 - target_tau) * probe_lag_seq[:-1]
            lag_innovation = (1.0 - target_tau) * probe_q_delta
            lag_pred = inherited_lag + lag_innovation
            lag_recurrence_resid = probe_lag_seq[1:] - lag_pred

            key = jax.random.fold_in(jax.random.PRNGKey(diag_seed + 808), old_state.total_it + 1)
            h_random = jax.random.rademacher(key, shape=first_target.shape, dtype=first_target.dtype)
            random_dirs = jnp.broadcast_to(normdir(h_random)[None, :], td_dirs.shape)

            def simulate(direction_seq, sign, persistent):
                def one(carry, xs2):
                    idx, obs, act, rew, nxt, done, direction = xs2
                    active = jnp.asarray(1.0, dtype=direction.dtype) if persistent else (idx == 0).astype(direction.dtype)
                    extra = sign * delta * active * direction
                    new_carry = update_once(carry, obs, act, rew, nxt, done, extra)
                    p_online, p_lag, _, _ = probe_metrics(new_carry)
                    return new_carry, (p_online, p_lag)

                idxs = jnp.arange(steps, dtype=jnp.int32)
                xs2 = (
                    idxs, batch_seq["observations"], batch_seq["actions"], batch_seq["rewards"],
                    batch_seq["next_observations"], batch_seq["dones"], direction_seq,
                )
                _, outs = jax.lax.scan(one, carry0, xs2)
                return outs

            def response(direction_seq, persistent=False):
                plus_q, plus_lag = simulate(direction_seq, 1.0, persistent)
                minus_q, minus_lag = simulate(direction_seq, -1.0, persistent)
                return (plus_q - minus_q) / (2.0 * delta), (plus_lag - minus_lag) / (2.0 * delta)

            def actual_source_alpha(source_rms_seq):
                desired = fd_epsilon * (
                    jnp.maximum(target_rms_seq, 1.0)
                    if fd_relative_scale
                    else jnp.ones_like(target_rms_seq)
                )
                alpha_each = desired / (source_rms_seq + 1e-8)
                return jnp.minimum(1.0, jnp.min(alpha_each))

            def simulate_actual(source_seq, sign, alpha):
                def one(carry, xs2):
                    obs, act, rew, nxt, done, source = xs2
                    extra = sign * alpha * source
                    new_carry = update_once(carry, obs, act, rew, nxt, done, extra)
                    p_online, p_lag, _, _ = probe_metrics(new_carry)
                    return new_carry, (p_online, p_lag)

                xs2 = (
                    batch_seq["observations"], batch_seq["actions"], batch_seq["rewards"],
                    batch_seq["next_observations"], batch_seq["dones"], source_seq,
                )
                _, outs = jax.lax.scan(one, carry0, xs2)
                return outs

            def actual_response(source_seq, source_rms_seq):
                alpha = actual_source_alpha(source_rms_seq)
                plus_q, plus_lag = simulate_actual(source_seq, 1.0, alpha)
                minus_q, minus_lag = simulate_actual(source_seq, -1.0, alpha)
                return (
                    (plus_q - minus_q) / (2.0 * alpha + 1e-12),
                    (plus_lag - minus_lag) / (2.0 * alpha + 1e-12),
                    alpha,
                )

            rq, rlag = response(random_dirs, persistent=False)
            tq, tlag = response(td_dirs, persistent=False)
            lq, llag = response(lag_dirs, persistent=False)

            if enable_forced:
                ftq, ftlag = response(td_dirs, persistent=True)
                flq, fllag = response(lag_dirs, persistent=True)
            else:
                ftq = jnp.full_like(tq, jnp.nan); ftlag = jnp.full_like(tlag, jnp.nan)
                flq = jnp.full_like(lq, jnp.nan); fllag = jnp.full_like(llag, jnp.nan)

            td_source_seq = td_dirs * td_rms_seq[:, None]
            lag_source_seq = lag_dirs * lag_rms_seq[:, None]
            if enable_actual_source_forcing:
                atq, atlag, td_actual_alpha = actual_response(td_source_seq, td_rms_seq)
                alq, allag, lag_actual_alpha = actual_response(lag_source_seq, lag_rms_seq)
            else:
                atq = jnp.full_like(tq, jnp.nan); atlag = jnp.full_like(tlag, jnp.nan)
                alq = jnp.full_like(lq, jnp.nan); allag = jnp.full_like(llag, jnp.nan)
                td_actual_alpha = lag_actual_alpha = jnp.asarray(jnp.nan, dtype=jnp.float32)

            i1 = 0
            ishort = short_steps - 1
            ilong = long_steps - 1

            def pack_impulse(prefix, qresp, lagresp, source_rms):
                g1 = rms(qresp[i1]); gs = rms(qresp[ishort]); gl = rms(qresp[ilong])
                return {
                    f"diag_loop/{prefix}_source_rms": source_rms,
                    f"diag_loop/{prefix}_q_gain_h1": g1,
                    f"diag_loop/{prefix}_q_gain_h{short_steps}": gs,
                    f"diag_loop/{prefix}_q_gain_h{long_steps}": gl,
                    f"diag_loop/{prefix}_q_h{short_steps}_over_h1": gs / (g1 + 1e-12),
                    f"diag_loop/{prefix}_q_h{long_steps}_over_h1": gl / (g1 + 1e-12),
                    f"diag_loop/{prefix}_q_expansive_h1": (g1 > 1.0).astype(jnp.float32),
                    f"diag_loop/{prefix}_q_expansive_h{short_steps}": (gs > 1.0).astype(jnp.float32),
                    f"diag_loop/{prefix}_q_expansive_h{long_steps}": (gl > 1.0).astype(jnp.float32),
                    f"diag_loop/{prefix}_q_h1_vs_h{short_steps}_cos": cosine(qresp[i1], qresp[ishort]),
                    f"diag_loop/{prefix}_q_h1_vs_h{long_steps}_cos": cosine(qresp[i1], qresp[ilong]),
                    f"diag_loop/{prefix}_lag_gain_h1": rms(lagresp[i1]),
                    f"diag_loop/{prefix}_lag_gain_h{short_steps}": rms(lagresp[ishort]),
                    f"diag_loop/{prefix}_lag_gain_h{long_steps}": rms(lagresp[ilong]),
                }

            def pack_forced(prefix, qresp, lagresp, impulse_qresp, source_rms_seq, baseline_error):
                g1 = rms(qresp[i1]); gs = rms(qresp[ishort]); gl = rms(qresp[ilong])
                return {
                    f"diag_forced/{prefix}_q_response_h1": g1,
                    f"diag_forced/{prefix}_q_response_h{short_steps}": gs,
                    f"diag_forced/{prefix}_q_response_h{long_steps}": gl,
                    f"diag_forced/{prefix}_q_energy_gain_h{short_steps}": gs / jnp.sqrt(float(short_steps)),
                    f"diag_forced/{prefix}_q_energy_gain_h{long_steps}": gl / jnp.sqrt(float(long_steps)),
                    f"diag_forced/{prefix}_q_per_step_gain_h{short_steps}": gs / float(short_steps),
                    f"diag_forced/{prefix}_q_per_step_gain_h{long_steps}": gl / float(long_steps),
                    f"diag_forced/{prefix}_vs_impulse_h{short_steps}": gs / (rms(impulse_qresp[ishort]) + 1e-12),
                    f"diag_forced/{prefix}_vs_impulse_h{long_steps}": gl / (rms(impulse_qresp[ilong]) + 1e-12),
                    f"diag_forced/{prefix}_q_with_baseline_error_cos_h{short_steps}": cosine(qresp[ishort], baseline_error[ishort]),
                    f"diag_forced/{prefix}_q_with_baseline_error_cos_h{long_steps}": cosine(qresp[ilong], baseline_error[ilong]),
                    f"diag_forced/{prefix}_lag_response_h{short_steps}": rms(lagresp[ishort]),
                    f"diag_forced/{prefix}_lag_response_h{long_steps}": rms(lagresp[ilong]),
                    f"diag_forced/{prefix}_natural_source_rms_h1": source_rms_seq[i1],
                    f"diag_forced/{prefix}_natural_source_rms_mean_h{short_steps}": jnp.mean(source_rms_seq[:short_steps]),
                    f"diag_forced/{prefix}_natural_source_rms_mean_h{long_steps}": jnp.mean(source_rms_seq[:long_steps]),
                    f"diag_forced/{prefix}_natural_source_rms_max_h{long_steps}": jnp.max(source_rms_seq[:long_steps]),
                    f"diag_forced/{prefix}_natural_source_rms_h{long_steps}_over_h1": source_rms_seq[ilong] / (source_rms_seq[i1] + 1e-12),
                }

            def pack_actual_forced(prefix, qresp, lagresp, unit_qresp, source_rms_seq, alpha, baseline_error):
                gs = rms(qresp[ishort]); gl = rms(qresp[ilong])
                in_energy_s = jnp.sqrt(jnp.sum(source_rms_seq[:short_steps] ** 2) + 1e-12)
                in_energy_l = jnp.sqrt(jnp.sum(source_rms_seq[:long_steps] ** 2) + 1e-12)
                in_l1_s = jnp.sum(source_rms_seq[:short_steps]) + 1e-12
                in_l1_l = jnp.sum(source_rms_seq[:long_steps]) + 1e-12
                err_s = baseline_error[ishort]
                err_l = baseline_error[ilong]
                coeff_s = jnp.mean(qresp[ishort] * err_s) / (jnp.mean(err_s ** 2) + 1e-12)
                coeff_l = jnp.mean(qresp[ilong] * err_l) / (jnp.mean(err_l ** 2) + 1e-12)
                max_rel_applied = jnp.max(
                    alpha * source_rms_seq / jnp.maximum(target_rms_seq, 1.0)
                )
                return {
                    f"diag_forced_actual/{prefix}_alpha": alpha,
                    f"diag_forced_actual/{prefix}_max_applied_over_target": max_rel_applied,
                    f"diag_forced_actual/{prefix}_input_energy_h{short_steps}": in_energy_s,
                    f"diag_forced_actual/{prefix}_input_energy_h{long_steps}": in_energy_l,
                    f"diag_forced_actual/{prefix}_input_l1_h{short_steps}": in_l1_s,
                    f"diag_forced_actual/{prefix}_input_l1_h{long_steps}": in_l1_l,
                    f"diag_forced_actual/{prefix}_q_response_h{short_steps}": gs,
                    f"diag_forced_actual/{prefix}_q_response_h{long_steps}": gl,
                    f"diag_forced_actual/{prefix}_energy_gain_h{short_steps}": gs / in_energy_s,
                    f"diag_forced_actual/{prefix}_energy_gain_h{long_steps}": gl / in_energy_l,
                    f"diag_forced_actual/{prefix}_l1_gain_h{short_steps}": gs / in_l1_s,
                    f"diag_forced_actual/{prefix}_l1_gain_h{long_steps}": gl / in_l1_l,
                    f"diag_forced_actual/{prefix}_vs_unit_forced_h{short_steps}": gs / (
                        rms(unit_qresp[ishort]) + 1e-12
                    ),
                    f"diag_forced_actual/{prefix}_vs_unit_forced_h{long_steps}": gl / (
                        rms(unit_qresp[ilong]) + 1e-12
                    ),
                    f"diag_forced_actual/{prefix}_with_baseline_error_cos_h{short_steps}": cosine(
                        qresp[ishort], err_s
                    ),
                    f"diag_forced_actual/{prefix}_with_baseline_error_cos_h{long_steps}": cosine(
                        qresp[ilong], err_l
                    ),
                    f"diag_forced_actual/{prefix}_signed_error_projection_h{short_steps}": coeff_s,
                    f"diag_forced_actual/{prefix}_signed_error_projection_h{long_steps}": coeff_l,
                    f"diag_forced_actual/{prefix}_response_over_baseline_error_h{short_steps}": gs / (
                        rms(err_s) + 1e-12
                    ),
                    f"diag_forced_actual/{prefix}_response_over_baseline_error_h{long_steps}": gl / (
                        rms(err_l) + 1e-12
                    ),
                    f"diag_forced_actual/{prefix}_lag_response_h{short_steps}": rms(lagresp[ishort]),
                    f"diag_forced_actual/{prefix}_lag_response_h{long_steps}": rms(lagresp[ilong]),
                }

            out = {"diag_loop/fd_delta": delta}
            out.update(pack_impulse("random", rq, rlag, rms(h_random)))
            out.update(pack_impulse("td", tq, tlag, td_rms_seq[i1]))
            out.update(pack_impulse("lag", lq, llag, lag_rms_seq[i1]))

            # Natural unperturbed error trajectory on one *fixed probe*, so temporal
            # direction cosines are meaningful unlike cross-batch sample-index vectors.
            out.update({
                "diag_loopbase/probe_lag_rms_h1": rms(base_probe_lag[i1]),
                f"diag_loopbase/probe_lag_rms_h{short_steps}": rms(base_probe_lag[ishort]),
                f"diag_loopbase/probe_lag_rms_h{long_steps}": rms(base_probe_lag[ilong]),
                f"diag_loopbase/probe_lag_h1_vs_h{short_steps}_cos": cosine(base_probe_lag[i1], base_probe_lag[ishort]),
                f"diag_loopbase/probe_lag_h1_vs_h{long_steps}_cos": cosine(base_probe_lag[i1], base_probe_lag[ilong]),
                f"diag_loopbase/probe_lag_coherence_h{short_steps}": sequence_coherence(base_probe_lag, short_steps),
                f"diag_loopbase/probe_lag_coherence_h{long_steps}": sequence_coherence(base_probe_lag, long_steps),
                f"diag_loopbase/probe_lag_consecutive_cos_h{short_steps}": consecutive_cosine(base_probe_lag, short_steps),
                f"diag_loopbase/probe_lag_consecutive_cos_h{long_steps}": consecutive_cosine(base_probe_lag, long_steps),
                "diag_loopbase/probe_td_rms_h1": rms(base_probe_td[i1]),
                f"diag_loopbase/probe_td_rms_h{short_steps}": rms(base_probe_td[ishort]),
                f"diag_loopbase/probe_td_rms_h{long_steps}": rms(base_probe_td[ilong]),
                f"diag_loopbase/probe_td_h1_vs_h{short_steps}_cos": cosine(base_probe_td[i1], base_probe_td[ishort]),
                f"diag_loopbase/probe_td_h1_vs_h{long_steps}_cos": cosine(base_probe_td[i1], base_probe_td[ilong]),
                f"diag_loopbase/probe_td_coherence_h{short_steps}": sequence_coherence(base_probe_td, short_steps),
                f"diag_loopbase/probe_td_coherence_h{long_steps}": sequence_coherence(base_probe_td, long_steps),
                f"diag_loopbase/probe_td_consecutive_cos_h{short_steps}": consecutive_cosine(base_probe_td, short_steps),
                f"diag_loopbase/probe_td_consecutive_cos_h{long_steps}": consecutive_cosine(base_probe_td, long_steps),

                # V7 natural innovation decomposition of target-network lag.
                f"diag_loopbase/q_delta_rms_mean_h{short_steps}": jnp.mean(jax.vmap(rms)(probe_q_delta[:short_steps])),
                f"diag_loopbase/q_delta_rms_mean_h{long_steps}": jnp.mean(jax.vmap(rms)(probe_q_delta[:long_steps])),
                f"diag_loopbase/q_delta_coherence_h{short_steps}": sequence_coherence(probe_q_delta, short_steps),
                f"diag_loopbase/q_delta_coherence_h{long_steps}": sequence_coherence(probe_q_delta, long_steps),
                f"diag_loopbase/q_delta_with_prev_lag_cos_h{short_steps}": pair_cosine_mean(
                    probe_q_delta, probe_lag_seq[:-1], short_steps
                ),
                f"diag_loopbase/q_delta_with_prev_lag_cos_h{long_steps}": pair_cosine_mean(
                    probe_q_delta, probe_lag_seq[:-1], long_steps
                ),
                f"diag_loopbase/q_delta_with_next_lag_cos_h{short_steps}": pair_cosine_mean(
                    probe_q_delta, probe_lag_seq[1:], short_steps
                ),
                f"diag_loopbase/q_delta_with_next_lag_cos_h{long_steps}": pair_cosine_mean(
                    probe_q_delta, probe_lag_seq[1:], long_steps
                ),
                f"diag_loopbase/target_drift_rms_mean_h{short_steps}": jnp.mean(
                    jax.vmap(rms)(probe_backup_delta[:short_steps])
                ),
                f"diag_loopbase/target_drift_rms_mean_h{long_steps}": jnp.mean(
                    jax.vmap(rms)(probe_backup_delta[:long_steps])
                ),
                f"diag_loopbase/target_drift_coherence_h{short_steps}": sequence_coherence(
                    probe_backup_delta, short_steps
                ),
                f"diag_loopbase/target_drift_coherence_h{long_steps}": sequence_coherence(
                    probe_backup_delta, long_steps
                ),
                f"diag_loopbase/q_delta_vs_target_drift_cos_h{short_steps}": pair_cosine_mean(
                    probe_q_delta, probe_backup_delta, short_steps
                ),
                f"diag_loopbase/q_delta_vs_target_drift_cos_h{long_steps}": pair_cosine_mean(
                    probe_q_delta, probe_backup_delta, long_steps
                ),
                f"diag_loopbase/lag_innovation_rms_mean_h{short_steps}": jnp.mean(
                    jax.vmap(rms)(lag_innovation[:short_steps])
                ),
                f"diag_loopbase/lag_innovation_rms_mean_h{long_steps}": jnp.mean(
                    jax.vmap(rms)(lag_innovation[:long_steps])
                ),
                f"diag_loopbase/lag_innovation_with_next_lag_cos_h{short_steps}": pair_cosine_mean(
                    lag_innovation, probe_lag_seq[1:], short_steps
                ),
                f"diag_loopbase/lag_innovation_with_next_lag_cos_h{long_steps}": pair_cosine_mean(
                    lag_innovation, probe_lag_seq[1:], long_steps
                ),
                f"diag_loopbase/lag_recurrence_relerr_mean_h{short_steps}": jnp.mean(
                    jax.vmap(rms)(lag_recurrence_resid[:short_steps])
                    / (jax.vmap(rms)(probe_lag_seq[1:short_steps + 1]) + 1e-12)
                ),
                f"diag_loopbase/lag_recurrence_relerr_mean_h{long_steps}": jnp.mean(
                    jax.vmap(rms)(lag_recurrence_resid[:long_steps])
                    / (jax.vmap(rms)(probe_lag_seq[1:long_steps + 1]) + 1e-12)
                ),
            })

            if enable_forced:
                out.update(pack_forced("td", ftq, ftlag, tq, td_rms_seq, base_probe_td))
                out.update(pack_forced("lag", flq, fllag, lq, lag_rms_seq, base_probe_lag))
            if enable_actual_source_forcing:
                out.update(pack_actual_forced(
                    "td", atq, atlag, ftq, td_rms_seq, td_actual_alpha, base_probe_td
                ))
                out.update(pack_actual_forced(
                    "lag", alq, allag, flq, lag_rms_seq, lag_actual_alpha, base_probe_lag
                ))
            return out

        return loop_step


    def _build_horizon_diagnostic_step(self):
        """Finite-horizon critic-fitting response B^(H) under repeated virtual Adam fits.

        The one-step B diagnostic and the ridge/full-fit response were empirically very
        different.  This diagnostic fills the missing time scale by repeatedly fitting the
        *same frozen batch and target* for H=1/short/long virtual Adam steps, never committing
        those updates.  Directions are deliberately observable and head-independent:
          random, current TD-error pattern, and current online-target lag pattern.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        discount = self.discount
        fd_epsilon = self.diag_fd_epsilon
        fd_relative_scale = self.diag_fd_relative_scale
        short_steps = self.diag_horizon_short_steps
        long_steps = self.diag_horizon_long_steps
        diag_seed = self.diag_seed

        def rms(x):
            return jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2) + 1e-12)

        def cosine(a, b):
            a = jnp.ravel(a)
            b = jnp.ravel(b)
            return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)

        def normalize(h):
            return h / jnp.maximum(rms(h), 1e-8)

        @jax.jit
        def horizon_step(old_state: IQLState, batch: TensorBatch, probe_batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)
            probe_observations = probe_batch["observations"]
            probe_actions = probe_batch["actions"]

            next_v = v_apply({"params": old_state.v_params}, next_observations)
            target = rewards + (1.0 - dones) * discount * next_v
            target_rms = rms(target)
            delta = fd_epsilon * (jnp.maximum(target_rms, 1.0) if fd_relative_scale else 1.0)

            q_online_all = q_apply({"params": old_state.q_params}, observations, actions)
            q_target_all = q_apply({"params": old_state.q_target_params}, observations, actions)
            h_td = jnp.mean(q_online_all, axis=0) - target
            h_lag = jnp.min(q_online_all, axis=0) - jnp.min(q_target_all, axis=0)
            key = jax.random.fold_in(jax.random.PRNGKey(diag_seed + 101), old_state.total_it + 1)
            h_random = jax.random.rademacher(key, shape=target.shape, dtype=target.dtype)

            directions = (normalize(h_random), normalize(h_td), normalize(h_lag))

            def virtual_probe_output(target_arg, n_steps):
                def one_step(_, carry):
                    params, opt_state = carry

                    def loss_fn(p):
                        qs = q_apply({"params": p}, observations, actions)
                        return jnp.mean((qs - target_arg[None, :]) ** 2)

                    grads = jax.grad(loss_fn)(params)
                    updates, opt_state_new = q_tx.update(grads, opt_state, params)
                    params_new = optax.apply_updates(params, updates)
                    return params_new, opt_state_new

                params_h, _ = jax.lax.fori_loop(
                    0, n_steps, one_step, (old_state.q_params, old_state.q_opt_state)
                )
                return q_apply({"params": params_h}, probe_observations, probe_actions)

            def response(h_unit, n_steps):
                plus = virtual_probe_output(target + delta * h_unit, n_steps)
                minus = virtual_probe_output(target - delta * h_unit, n_steps)
                return (plus - minus) / (2.0 * delta)

            def three_horizons(h_unit):
                b1 = response(h_unit, 1)
                bs = response(h_unit, short_steps)
                bl = response(h_unit, long_steps)
                return b1, bs, bl

            r1, rs, rl = three_horizons(directions[0])
            t1, ts, tl = three_horizons(directions[1])
            l1, ls, ll = three_horizons(directions[2])

            return {
                "diag_BH/fd_delta": delta,
                "diag_BH/random_gain_h1": rms(r1),
                f"diag_BH/random_gain_h{short_steps}": rms(rs),
                f"diag_BH/random_gain_h{long_steps}": rms(rl),
                f"diag_BH/random_h{short_steps}_over_h1": rms(rs) / (rms(r1) + 1e-12),
                f"diag_BH/random_h{long_steps}_over_h1": rms(rl) / (rms(r1) + 1e-12),
                f"diag_BH/random_h1_vs_h{short_steps}_cos": cosine(r1, rs),
                f"diag_BH/random_h1_vs_h{long_steps}_cos": cosine(r1, rl),
                "diag_BH/td_pattern_source_rms": rms(h_td),
                "diag_BH/td_pattern_gain_h1": rms(t1),
                f"diag_BH/td_pattern_gain_h{short_steps}": rms(ts),
                f"diag_BH/td_pattern_gain_h{long_steps}": rms(tl),
                f"diag_BH/td_pattern_h{short_steps}_over_h1": rms(ts) / (rms(t1) + 1e-12),
                f"diag_BH/td_pattern_h{long_steps}_over_h1": rms(tl) / (rms(t1) + 1e-12),
                f"diag_BH/td_pattern_h1_vs_h{long_steps}_cos": cosine(t1, tl),
                "diag_BH/td_pattern_gamma_B_proxy_h1": discount * rms(t1),
                f"diag_BH/td_pattern_gamma_B_proxy_h{short_steps}": discount * rms(ts),
                f"diag_BH/td_pattern_gamma_B_proxy_h{long_steps}": discount * rms(tl),
                "diag_BH/td_pattern_B_expansive_h1": (rms(t1) > 1.0).astype(jnp.float32),
                f"diag_BH/td_pattern_B_expansive_h{short_steps}": (rms(ts) > 1.0).astype(jnp.float32),
                f"diag_BH/td_pattern_B_expansive_h{long_steps}": (rms(tl) > 1.0).astype(jnp.float32),
                "diag_BH/lag_pattern_source_rms": rms(h_lag),
                "diag_BH/lag_pattern_gain_h1": rms(l1),
                f"diag_BH/lag_pattern_gain_h{short_steps}": rms(ls),
                f"diag_BH/lag_pattern_gain_h{long_steps}": rms(ll),
                f"diag_BH/lag_pattern_h{short_steps}_over_h1": rms(ls) / (rms(l1) + 1e-12),
                f"diag_BH/lag_pattern_h{long_steps}_over_h1": rms(ll) / (rms(l1) + 1e-12),
                f"diag_BH/lag_pattern_h1_vs_h{long_steps}_cos": cosine(l1, ll),
                "diag_BH/lag_pattern_gamma_B_proxy_h1": discount * rms(l1),
                f"diag_BH/lag_pattern_gamma_B_proxy_h{short_steps}": discount * rms(ls),
                f"diag_BH/lag_pattern_gamma_B_proxy_h{long_steps}": discount * rms(ll),
                "diag_BH/lag_pattern_B_expansive_h1": (rms(l1) > 1.0).astype(jnp.float32),
                f"diag_BH/lag_pattern_B_expansive_h{short_steps}": (rms(ls) > 1.0).astype(jnp.float32),
                f"diag_BH/lag_pattern_B_expansive_h{long_steps}": (rms(ll) > 1.0).astype(jnp.float32),
            }

        return horizon_step

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
        fd_epsilon = self.diag_fd_epsilon
        fd_relative_scale = self.diag_fd_relative_scale

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

            target_rms = rms(target)
            delta = fd_epsilon * (jnp.maximum(target_rms, 1.0) if fd_relative_scale else 1.0)
            fd_out = (
                actual_probe_output(target + delta * h_unit)
                - actual_probe_output(target - delta * h_unit)
            ) / (2.0 * delta)
            _, adam_jvp = jax.jvp(actual_probe_output, (target,), (h_unit,))

            return {
                "diag_B/ridge_random_gain_probe": rms(ridge_out),
                "diag_B/ridge_vs_fd_adam_cos_random": cosine(ridge_out, fd_out),
                "diag_B/ridge_over_fd_gain_random": rms(ridge_out) / (rms(fd_out) + 1e-12),
                "diag_B/ridge_vs_jvp_adam_cos_random": cosine(ridge_out, adam_jvp),
                "diag_B/ridge_cg_relative_residual": relative_linear_residual,
                "diag_B/ridge_cg_quality_pass_005": (relative_linear_residual < 0.05).astype(jnp.float32),
                "diag_B/ridge_cg_quality_pass_01": (relative_linear_residual < 0.10).astype(jnp.float32),
            }

        return ridge_step

    def train(
        self,
        batch: TensorBatch,
        diag_batch: Optional[TensorBatch] = None,
        residual_reference_batch: Optional[TensorBatch] = None,
        run_residual_calibration: bool = False,
        run_horizon_diagnostic: bool = False,
        run_loop_diagnostic: bool = False,
        loop_batch_sequence: Optional[TensorBatch] = None,
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

            if run_residual_calibration:
                ref_batch = residual_reference_batch if residual_reference_batch is not None else diag_batch
                rescal_log = self._residual_calibration_step(old_state, diag_batch, ref_batch)
                log_dict.update(rescal_log)

            if run_horizon_diagnostic:
                horizon_log = self._horizon_diagnostic_step(old_state, batch, diag_batch)
                log_dict.update(horizon_log)

            if run_loop_diagnostic and loop_batch_sequence is not None:
                loop_log = self._closed_loop_diagnostic_step(old_state, loop_batch_sequence, diag_batch)
                log_dict.update(loop_log)

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

    # Diagnostics use a separate deterministic RNG and a fixed probe batch.
    # This gives comparable time-series diagnostics and, critically, prevents
    # diagnostic sampling from changing the future training minibatch sequence.
    diagnostic_probe_batch = None
    diagnostic_action_ref_batch = None
    if config.enable_propagation_diagnostics:
        probe_rng = np.random.default_rng(seed + 424_243)
        diagnostic_probe_batch = replay_buffer.sample_with_rng(config.diag_probe_batch_size, probe_rng)
        action_ref_rng = np.random.default_rng(seed + 424_887)
        diagnostic_action_ref_batch = replay_buffer.sample_with_rng(
            config.diag_action_ref_batch_size,
            action_ref_rng,
        )

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
        diag_fd_relative_scale=config.diag_fd_relative_scale,
        diag_fd_small_multiplier=config.diag_fd_small_multiplier,
        diag_fd_large_multiplier=config.diag_fd_large_multiplier,
        diag_num_basis_directions=config.diag_num_basis_directions,
        diag_num_calibration_bins=config.diag_num_calibration_bins,
        diag_enable_multistep_v_correctability=config.diag_enable_multistep_v_correctability,
        diag_v_correctability_short_steps=config.diag_v_correctability_short_steps,
        diag_v_correctability_long_steps=config.diag_v_correctability_long_steps,
        diag_knn_k=config.diag_knn_k,
        diag_enable_kernel_residual=config.diag_enable_kernel_residual,
        diag_kernel_bandwidth_scale=config.diag_kernel_bandwidth_scale,
        diag_kernel_min_bandwidth=config.diag_kernel_min_bandwidth,
        diag_kernel_target_neff=config.diag_kernel_target_neff,
        diag_kernel_bandwidth_search_iters=config.diag_kernel_bandwidth_search_iters,
        diag_enable_residual_injection_calibration=config.diag_enable_residual_injection_calibration,
        diag_residual_injection_fraction=config.diag_residual_injection_fraction,
        diag_residual_injection_small_multiplier=config.diag_residual_injection_small_multiplier,
        diag_residual_injection_large_multiplier=config.diag_residual_injection_large_multiplier,
        diag_enable_conditioned_residual=config.diag_enable_conditioned_residual,
        diag_action_ref_k=config.diag_action_ref_k,
        diag_action_ref_bisection_iters=config.diag_action_ref_bisection_iters,
        diag_loop_short_steps=config.diag_loop_short_steps,
        diag_loop_long_steps=config.diag_loop_long_steps,
        diag_enable_forced_loop=config.diag_enable_forced_loop,
        diag_enable_actual_source_forcing=config.diag_enable_actual_source_forcing,
        diag_actual_source_cap_fraction=config.diag_actual_source_cap_fraction,
        diag_horizon_short_steps=config.diag_horizon_short_steps,
        diag_horizon_long_steps=config.diag_horizon_long_steps,
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
            if diagnostic_log_path is not None:
                sanitize_diagnostic_jsonl(diagnostic_log_path, max_timestep=start_timestep)
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
            residual_calibration_due = (
                diag_due
                and config.diag_enable_residual_injection_calibration
                and step % int(config.diag_residual_calibration_every) == 0
            )
            horizon_due = (
                diag_due
                and config.diag_enable_horizon_b
                and step % int(config.diag_horizon_every) == 0
            )
            loop_due = (
                diag_due
                and config.diag_enable_loop_propagation
                and step % int(config.diag_loop_every) == 0
            )
            ridge_due = (
                diag_due
                and config.diag_enable_ridge_b
                and step % int(config.diag_ridge_every) == 0
            )
            diag_batch = diagnostic_probe_batch if diag_due else None
            loop_batch_sequence = None
            if loop_due:
                loop_rng = np.random.default_rng(seed + 1_000_003 * int(step) + 17)
                loop_batches = [batch] + [
                    replay_buffer.sample_with_rng(config.batch_size, loop_rng)
                    for _ in range(int(config.diag_loop_long_steps) - 1)
                ]
                loop_batch_sequence = {
                    key: jnp.stack([b[key] for b in loop_batches], axis=0)
                    for key in batch.keys()
                }

            log_dict = trainer.train(
                batch,
                diag_batch=diag_batch,
                residual_reference_batch=diagnostic_action_ref_batch if residual_calibration_due else None,
                run_residual_calibration=residual_calibration_due,
                run_horizon_diagnostic=horizon_due,
                run_loop_diagnostic=loop_due,
                loop_batch_sequence=loop_batch_sequence,
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
