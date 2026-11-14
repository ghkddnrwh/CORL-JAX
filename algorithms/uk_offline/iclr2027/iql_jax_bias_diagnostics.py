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

ALGORITHM_NAME = "IQL"
ALGORITHM_FULL_NAME = "Implicit Q-Learning"

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

    # Bias diagnostics. These diagnostics never enter the optimization losses;
    # they are evaluated on a fixed dataset batch so IQL can be compared with
    # DD-IQL under the same filter--error diagnostics. The full diagnostic pass
    # is intentionally coarse because it evaluates a large fixed batch.
    enable_bias_diagnostics: bool = True
    diagnostic_freq: int = 2_500
    diagnostic_batch_size: int = 8_192
    diagnostic_seed_offset: int = 10_000
    diagnostic_compute_mc_returns: bool = True
    diagnostic_save_npz: bool = True
    diagnostic_eval_return_gap: bool = True
    # Keep diagnostic results in memory at diagnostic_freq resolution, but rewrite
    # the NPZ / upload the diagnostic artifact only at this coarse cadence.
    diagnostic_flush_freq: int = int(25e3)
    # Small fixed probes complement the post-update batch summaries. Probe
    # arrays are evaluated before the scheduled update, with stable sample IDs.
    diagnostic_probe_size: int = 128
    diagnostic_probe_freq: int = 2_500
    diagnostic_save_probes: bool = True
    diagnostic_probe_dir: Optional[str] = None

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
    assert config.diagnostic_freq > 0
    assert config.diagnostic_flush_freq > 0
    assert config.diagnostic_batch_size > 1
    assert config.diagnostic_seed_offset >= 0
    assert config.diagnostic_probe_size > 0
    assert config.diagnostic_probe_freq > 0
    assert config.diagnostic_probe_dir is None or config.diagnostic_probe_dir != ""
    assert config.checkpoint_freq > 0
    assert config.discount >= 0.0 and config.discount <= 1.0
    assert config.tau >= 0.0 and config.tau <= 1.0
    assert config.beta >= 0.0
    assert config.iql_tau >= 0.0 and config.iql_tau <= 1.0
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


def infer_episode_ends(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    """Infer trajectory boundaries for Monte-Carlo return diagnostics.

    Explicit terminal/timeout information is used when present. We additionally
    detect discontinuities between next_observations[t] and observations[t+1],
    which is useful for fixed-horizon datasets whose truncations are not stored
    as terminals. The final transition is always treated as an episode end.
    """
    n = int(np.asarray(dataset["rewards"]).shape[0])
    ends = np.zeros(n, dtype=bool)

    for key in ("terminals", "dones", "timeouts", "truncations", "episode_ends"):
        if key in dataset:
            values = np.asarray(dataset[key]).reshape(-1)
            if values.shape[0] == n:
                ends |= values.astype(bool)

    if "masks" in dataset:
        masks = np.asarray(dataset["masks"]).reshape(-1)
        if masks.shape[0] == n:
            ends |= masks <= 0.0

    if n > 1 and "observations" in dataset and "next_observations" in dataset:
        obs = np.asarray(dataset["observations"])
        next_obs = np.asarray(dataset["next_observations"])
        if obs.shape[0] == n and next_obs.shape[0] == n and obs.shape[1:] == next_obs.shape[1:]:
            chunk_size = 100_000
            for start in range(0, n - 1, chunk_size):
                stop = min(start + chunk_size, n - 1)
                lhs = next_obs[start:stop].reshape(stop - start, -1)
                rhs = obs[start + 1:stop + 1].reshape(stop - start, -1)
                continuity = np.all(
                    np.isclose(lhs, rhs, rtol=1e-5, atol=1e-6), axis=-1
                )
                ends[start:stop] |= ~continuity

    if n > 0:
        ends[-1] = True
    return ends


def compute_discounted_returns_to_go(
    rewards: np.ndarray,
    episode_ends: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Compute realized discounted return-to-go on the offline trajectories.

    This is a diagnostic reference, not an oracle Q*. It follows the actual
    continuation contained in the offline dataset.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    episode_ends = np.asarray(episode_ends, dtype=bool).reshape(-1)
    if rewards.shape[0] != episode_ends.shape[0]:
        raise ValueError("rewards and episode_ends must have the same length")

    returns = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for idx in range(rewards.shape[0] - 1, -1, -1):
        if episode_ends[idx]:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        returns[idx] = running
    return returns.astype(np.float32)


def add_mc_return_diagnostics(dataset: Dict[str, np.ndarray], discount: float) -> None:
    """Attach realized discounted returns and inferred episode ends in-place."""
    episode_ends = infer_episode_ends(dataset)
    dataset["diagnostic_episode_ends"] = episode_ends.astype(np.float32)
    dataset["mc_returns"] = compute_discounted_returns_to_go(
        dataset["rewards"], episode_ends, discount
    )
    print(
        "Bias diagnostics: computed dataset Monte-Carlo returns "
        f"using {int(np.sum(episode_ends))} inferred trajectory boundaries."
    )


def batch_covariance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Per-row population covariance for tensors shaped [N, B]."""
    x_centered = x - jnp.mean(x, axis=-1, keepdims=True)
    y_centered = y - jnp.mean(y, axis=-1, keepdims=True)
    return jnp.mean(x_centered * y_centered, axis=-1)


def batch_correlation(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Per-row Pearson correlation, with undefined rows explicitly marked NaN.

    A constant weight or error vector has no defined correlation. Do not turn
    that case into zero by adding an epsilon to the Pearson denominator.
    """
    x_centered = x - jnp.mean(x, axis=-1, keepdims=True)
    y_centered = y - jnp.mean(y, axis=-1, keepdims=True)
    cov = jnp.mean(x_centered * y_centered, axis=-1)
    x_std = jnp.sqrt(jnp.mean(x_centered ** 2, axis=-1))
    y_std = jnp.sqrt(jnp.mean(y_centered ** 2, axis=-1))
    denominator = x_std * y_std
    valid = (
        jnp.all(jnp.isfinite(x), axis=-1)
        & jnp.all(jnp.isfinite(y), axis=-1)
        & jnp.isfinite(cov)
        & jnp.isfinite(denominator)
        # Explicit range checks also reject constant float32 arrays when the
        # reduction mean rounds slightly away from the identical input value.
        & (jnp.max(x, axis=-1) > jnp.min(x, axis=-1))
        & (jnp.max(y, axis=-1) > jnp.min(y, axis=-1))
        & (x_std > eps)
        & (y_std > eps)
    )
    safe_denominator = jnp.where(valid, denominator, jnp.ones_like(denominator))
    return jnp.where(valid, jnp.clip(cov / safe_denominator, -1.0, 1.0), jnp.nan)


def weighted_error_contribution(
    weight: jnp.ndarray,
    error: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Per-row E[w * error] / E[w], matching the direct weighted-error term."""
    return jnp.mean(weight * error, axis=-1) / (jnp.mean(weight, axis=-1) + eps)


# BEGIN FIXED-PROBE DIAGNOSTIC HELPERS
# These helpers only observe model outputs. They never change a training RNG,
# gradient, optimizer, replay sample, or target-network parameter.
def distribution_summary(values, prefix):
    """JAX-compatible distribution of values across the fixed dataset batch.

    ``prefix='diag/q'`` gives ``diag/q_p99``, etc. Non-finite observations
    are not silently discarded; finite_fraction explicitly reports them.
    These are sample/state-action statistics, not disagreement across critics.
    """
    x = jnp.asarray(values).reshape(-1)
    quantiles = jnp.quantile(x, jnp.asarray([0.01, 0.05, 0.50, 0.95, 0.99]))
    result = {
        prefix + "_mean": jnp.mean(x),
        prefix + "_std": jnp.std(x),
        prefix + "_min": jnp.min(x),
        prefix + "_max": jnp.max(x),
        prefix + "_finite_fraction": jnp.mean(jnp.isfinite(x).astype(jnp.float32)),
    }
    for name, value in zip(("p01", "p05", "p50", "p95", "p99"), quantiles):
        result[prefix + "_" + name] = value
    result[prefix + "_p99_minus_p50"] = quantiles[4] - quantiles[2]
    result[prefix + "_max_minus_p50"] = jnp.max(x) - quantiles[2]
    return result


def correlation_valid_flags(metrics):
    """Validity companions for reported Pearson statistics; NaN is undefined.

    Returns only additional keys so callers can use metrics.update(...).
    """
    return {
        key + "_valid": jnp.isfinite(jnp.asarray(value)).astype(jnp.float32)
        for key, value in metrics.items()
        if ("corr" in key.rsplit("/", 1)[-1]) and not key.endswith("_valid")
    }


def _atomic_probe_npz(path, arrays):
    """Write a non-pickled compressed NPZ atomically in the same directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ProbeTraceWriter:
    """Bounded-memory histories for fixed replay examples, sampled pre-update.

    Each process/resume creates a new segment. Earlier files are preserved;
    the analysis loader discards stale continuation records after a restart.
    append() returns scalar transition diagnostics. flush() returns only paths
    not returned by an earlier flush, for optional incremental W&B uploads.
    NPZ files contain numeric/unicode arrays only (allow_pickle=False works).
    """

    ARRAY_KEYS = (
        "receiver_q1", "receiver_q2", "receiver_q", "receiver_v", "receiver_adv",
        "source_q1", "source_q2", "source_q", "source_v", "source_adv", "weight",
    )
    STEP_KEYS = ("state_step", "update_step", "source_snapshot_step", "delay_age")
    SCHEMA_VERSION = 1
    MAX_BUFFER_RECORDS = 128

    def __init__(self, root_dir, probe_indices, probe_batch, metadata, start_state_step=0):
        import hashlib
        import json
        import time

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.created_ns = time.time_ns()
        self.segment_dir = self.root_dir / (
            f"segment_{int(start_state_step):010d}_{self.created_ns}_{uuid.uuid4().hex[:8]}"
        )
        self.segment_dir.mkdir()
        self.indices = np.asarray(probe_indices, dtype=np.int64).reshape(-1).copy()
        self.probe_size = int(self.indices.size)
        if self.probe_size < 1:
            raise ValueError("A fixed probe must contain at least one sample")
        self.start_state_step = int(start_state_step)
        self._last_state_step = None
        self._previous_gate = None
        self._buffer = []
        self._pending_paths = []
        self._chunk_number = 0
        if isinstance(probe_batch, dict):
            items = sorted(probe_batch.items())
        else:
            items = [(str(i), value) for i, value in enumerate(probe_batch)]
        batch_arrays = {}
        for key, value in items:
            array = np.asarray(value)
            if array.ndim < 1 or array.shape[0] != self.probe_size:
                raise ValueError(f"Probe batch {key} must have first dimension {self.probe_size}")
            if array.dtype.hasobject:
                raise TypeError("Object arrays are not supported in probe metadata")
            batch_arrays["batch_" + str(key)] = array.copy()
        # Identity is based on indices/inputs, not rewards or MC returns, so the
        # same examples can be compared under different discount factors.
        digest = hashlib.sha256()
        digest.update(self.indices.tobytes())
        fingerprint_keys = [k for k in ("batch_observations", "batch_actions") if k in batch_arrays]
        if not fingerprint_keys:
            fingerprint_keys = sorted(batch_arrays)
        for key in fingerprint_keys:
            array = np.ascontiguousarray(batch_arrays[key])
            digest.update(key.encode("utf-8"))
            digest.update(str((array.shape, array.dtype.str)).encode("utf-8"))
            digest.update(array.tobytes())
        self.probe_fingerprint = digest.hexdigest()
        full_metadata = dict(metadata)
        full_metadata.update(
            schema_version=self.SCHEMA_VERSION,
            observation_phase="pre_update",
            start_state_step=self.start_state_step,
            created_ns=self.created_ns,
            probe_size=self.probe_size,
            probe_fingerprint=self.probe_fingerprint,
            gate_definition="1 if finite source_adv >= 0; 0 if finite source_adv < 0; -1 if invalid",
            indices_definition="row indices in the loaded replay dataset, after any dataset preprocessing",
        )
        metadata_path = self.segment_dir / "metadata.npz"
        _atomic_probe_npz(metadata_path, {
            "metadata_json": np.asarray(json.dumps(full_metadata, sort_keys=True)),
            "probe_indices": self.indices,
            **batch_arrays,
        })
        self._pending_paths.append(str(metadata_path))

    def append(self, record):
        """Record one pre-update observation; transitions are checkpoint-to-checkpoint.

        A transition can miss intermediate flips between observations. The
        first record in a new/resumed segment has undefined transition rates.
        """
        sample = {}
        for key in self.ARRAY_KEYS:
            if key not in record:
                raise KeyError(f"Missing fixed-probe array: {key}")
            array = np.asarray(record[key]).reshape(-1)
            if array.shape != (self.probe_size,):
                raise ValueError(f"Probe {key} shape {array.shape}, expected {(self.probe_size,)}")
            sample[key] = array.copy()
        for key in self.STEP_KEYS:
            sample[key] = np.asarray(int(np.asarray(record[key]).item()), dtype=np.int64)
        state_step = int(sample["state_step"])
        if state_step < self.start_state_step:
            raise ValueError("Probe observation predates this resume segment")
        if self._last_state_step is not None and state_step <= self._last_state_step:
            raise ValueError("Probe state_step must increase within each segment")
        if int(sample["update_step"]) != state_step + 1:
            raise ValueError("Pre-update probe requires update_step = state_step + 1")
        if int(sample["source_snapshot_step"]) + int(sample["delay_age"]) != state_step:
            raise ValueError("source_snapshot_step + delay_age must equal state_step")
        if int(sample["delay_age"]) < 0:
            raise ValueError("Probe delay_age must be nonnegative")
        finite_source = (
            np.isfinite(sample["source_q"]) & np.isfinite(sample["source_v"])
            & np.isfinite(sample["source_adv"]) & np.isfinite(sample["weight"])
        )
        gate = np.where(finite_source, (sample["source_adv"] >= 0).astype(np.int8), -1).astype(np.int8)
        sample["gate"] = gate
        receiver_valid = (
            np.isfinite(sample["receiver_q"]) & np.isfinite(sample["receiver_v"])
            & np.isfinite(sample["receiver_adv"])
        )
        receiver_gate = np.where(receiver_valid, (sample["receiver_adv"] >= 0).astype(np.int8), -1).astype(np.int8)
        sample["receiver_gate"] = receiver_gate
        both_valid = (gate >= 0) & (receiver_gate >= 0)
        scalar = {"diag/probe_" + key: float(sample[key]) for key in self.STEP_KEYS}
        scalar.update({
            "diag/probe_size": float(self.probe_size),
            "diag/probe_filter_weight_mean": float(np.mean(sample["weight"])),
            "diag/probe_gate_valid_fraction": float(np.mean(gate >= 0)),
            "diag/probe_high_weight_fraction": float(np.mean(gate[gate >= 0] == 1)) if np.any(gate >= 0) else float("nan"),
            "diag/probe_receiver_positive_fraction": float(np.mean(receiver_gate[receiver_valid] == 1)) if np.any(receiver_valid) else float("nan"),
            "diag/probe_source_receiver_gate_disagreement_fraction": float(np.mean(gate[both_valid] != receiver_gate[both_valid])) if np.any(both_valid) else float("nan"),
        })
        transition_names = (
            "gate_elapsed_updates", "gate_pair_valid_count", "gate_pair_valid_fraction",
            "high_to_low_count", "low_to_high_count", "high_to_low_fraction", "low_to_high_fraction",
            "gate_flip_fraction", "high_to_low_given_previous_high", "low_to_high_given_previous_low",
        )
        transition = {name: float("nan") for name in transition_names}
        if self._previous_gate is not None:
            prev = self._previous_gate
            valid = (prev >= 0) & (gate >= 0)
            count = int(np.sum(valid))
            high_to_low = int(np.sum(valid & (prev == 1) & (gate == 0)))
            low_to_high = int(np.sum(valid & (prev == 0) & (gate == 1)))
            previous_high = int(np.sum(valid & (prev == 1)))
            previous_low = int(np.sum(valid & (prev == 0)))
            transition.update({
                "gate_elapsed_updates": float(state_step - self._last_state_step),
                "gate_pair_valid_count": float(count),
                "gate_pair_valid_fraction": float(count / self.probe_size),
                "high_to_low_count": float(high_to_low),
                "low_to_high_count": float(low_to_high),
                "high_to_low_fraction": float(high_to_low / count) if count else float("nan"),
                "low_to_high_fraction": float(low_to_high / count) if count else float("nan"),
                "gate_flip_fraction": float((high_to_low + low_to_high) / count) if count else float("nan"),
                "high_to_low_given_previous_high": float(high_to_low / previous_high) if previous_high else float("nan"),
                "low_to_high_given_previous_low": float(low_to_high / previous_low) if previous_low else float("nan"),
            })
        scalar.update({"diag/probe_" + key: value for key, value in transition.items()})
        sample.update({"summary_" + key.replace("diag/probe_", ""): np.asarray(value) for key, value in scalar.items()})
        self._buffer.append(sample)
        self._previous_gate = gate.copy()
        self._last_state_step = state_step
        if len(self._buffer) >= self.MAX_BUFFER_RECORDS:
            self._write_chunk()
        return scalar

    def _write_chunk(self):
        if not self._buffer:
            return
        path = self.segment_dir / f"probe_trace_{self._chunk_number:06d}.npz"
        arrays = {
            key: np.stack([record[key] for record in self._buffer], axis=0)
            for key in self._buffer[0]
        }
        _atomic_probe_npz(path, arrays)
        self._pending_paths.append(str(path))
        self._chunk_number += 1
        self._buffer.clear()

    def flush(self):
        """Persist buffered observations; return each new path exactly once."""
        self._write_chunk()
        paths = self._pending_paths
        self._pending_paths = []
        return paths
# END FIXED-PROBE DIAGNOSTIC HELPERS



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
        self._mc_returns = np.zeros((buffer_size, 1), dtype=np.float32)
        self._has_mc_returns = False
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
        if "mc_returns" in data:
            mc_returns = np.asarray(data["mc_returns"], dtype=np.float32).reshape(-1)
            if mc_returns.shape[0] != n_transitions:
                raise ValueError("mc_returns must have one value per transition")
            self._mc_returns[:n_transitions] = mc_returns[..., None]
            self._has_mc_returns = True
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def _batch_from_indices(self, indices: np.ndarray) -> TensorBatch:
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
        }
        if self._has_mc_returns:
            batch["mc_returns"] = self._mc_returns[indices]
        return tree_to_device({k: jnp.asarray(v) for k, v in batch.items()}, self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return self._batch_from_indices(indices)

    def sample_fixed(
        self, batch_size: int, seed: int, return_indices: bool = False
    ) -> Union[TensorBatch, Tuple[TensorBatch, np.ndarray]]:
        """Return a deterministic batch, optionally with its stable replay indices.

        The private generator and its draws are unchanged. In particular, this
        method never advances the RNG used by training minibatch sampling.
        """
        size = min(self._size, self._pointer)
        if size <= 0:
            raise ValueError("Cannot sample diagnostics from an empty replay buffer")
        rng = np.random.default_rng(int(seed))
        replace = int(batch_size) > size
        indices = rng.choice(size, size=int(batch_size), replace=replace)
        batch = self._batch_from_indices(indices)
        if return_indices:
            return batch, indices.copy()
        return batch


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


def load_logs_npz(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as data:
        if len(data.files) == 0:
            return []
        n = int(np.asarray(data[data.files[0]]).shape[0])
        logs: List[Dict[str, Any]] = []
        for idx in range(n):
            item: Dict[str, Any] = {}
            for key in data.files:
                value = data[key][idx]
                item[key] = value.item() if isinstance(value, np.generic) else value
            logs.append(item)
        return logs


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
        self.device = device if device is not None else jax.devices()[0]

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
        self.q_def = TwinQ(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
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
        key_actor, key_q, key_v, actor_key = jax.random.split(key, 4)
        dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
        q_params = self.q_def.init(key_q, dummy_state, dummy_action)["params"]
        v_params = self.v_def.init(key_v, dummy_state)["params"]

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
        )

        self.state = tree_to_device(self.state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
        self._train_step = self._build_train_step()
        self._diagnostic_step = self._build_diagnostic_step()
        self._probe_step = self._build_probe_step()
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
            target_q1, target_q2 = q_apply({"params": state.q_target_params}, observations, actions)
            target_q_for_v = jnp.minimum(target_q1, target_q2)
            old_v = v_apply({"params": state.v_params}, observations)
            adv = target_q_for_v - old_v
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

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
            }
            return new_state, log_dict

        return train_step

    def _build_diagnostic_step(self):
        """Build a no-gradient diagnostic pass for IQL filter--error coupling.

        IQL uses the clipped target critic q_min=min(q1,q2) both to determine the
        expectile weight and as the value-regression target. Thus source and
        receiver are structurally the same quantity. The diagnostics below use
        the same fixed-batch definitions as DD-IQL whenever possible.

        Monte-Carlo and Bellman residuals are proxies/diagnostics, not oracle Q*
        errors. Twin-Q disagreement is also only an estimator-disagreement proxy.
        None of these metrics enter optimization.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        discount = self.discount
        iql_tau = self.iql_tau

        @jax.jit
        def diagnostic_step(state: IQLState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            q1, q2 = q_apply({"params": state.q_target_params}, observations, actions)
            q_min = jnp.minimum(q1, q2)
            q_max = jnp.maximum(q1, q2)
            q_mean = 0.5 * (q1 + q2)
            v = v_apply({"params": state.v_params}, observations)
            next_v = v_apply({"params": state.v_params}, next_observations)

            adv = q_min - v
            weight_1d = jnp.abs(iql_tau - (adv < 0.0).astype(jnp.float32))
            weight = weight_1d[None, :]
            receiver_adv = adv[None, :]

            # Intended IQL selection signal.
            signal_cov = batch_covariance(weight, receiver_adv)
            signal_corr = batch_correlation(weight, receiver_adv)

            # Twin-Q / clipped-min disagreement proxy. For two critics, q_min-q_max
            # is the selected-min critic's leave-one-out difference to the other.
            loo_error = (q_min - q_max)[None, :]
            loo_cov = batch_covariance(weight, loo_error)
            loo_corr = batch_correlation(weight, loo_error)
            loo_weighted = weighted_error_contribution(weight, loo_error)

            q_stack = jnp.stack([q1, q2], axis=0)
            twin_disagreement_std = jnp.mean(jnp.std(q_stack, axis=0))
            twin_abs_gap = jnp.mean(jnp.abs(q1 - q2))
            q1_q2_corr = batch_correlation(q1[None, :], q2[None, :])[0]
            structural_self_coupling = jnp.asarray(1.0, dtype=jnp.float32)

            # Bellman residual on a fixed batch drawn from the training dataset.
            # Fixed sampling supports temporal comparisons; it is not held out.
            bellman_target = rewards + (1.0 - dones) * discount * next_v
            q_min_td = (q_min - bellman_target)[None, :]
            q1_td = (q1 - bellman_target)[None, :]
            q2_td = (q2 - bellman_target)[None, :]
            td_cov = batch_covariance(weight, q_min_td)
            td_corr = batch_correlation(weight, q_min_td)
            td_weighted = weighted_error_contribution(weight, q_min_td)
            twin_td_error_corr = batch_correlation(q1_td, q2_td)[0]

            # Realized dataset-return residual. G_MC is not Q*.
            if "mc_returns" in batch:
                mc_return = jnp.squeeze(batch["mc_returns"], axis=-1)
                q_min_mc = (q_min - mc_return)[None, :]
                q1_mc = (q1 - mc_return)[None, :]
                q2_mc = (q2 - mc_return)[None, :]

                mc_cov = batch_covariance(weight, q_min_mc)
                mc_corr = batch_correlation(weight, q_min_mc)
                mc_weighted = weighted_error_contribution(weight, q_min_mc)
                filter_q1_mc_corr = batch_correlation(weight, q1_mc)[0]
                filter_q2_mc_corr = batch_correlation(weight, q2_mc)[0]
                twin_mc_error_corr = batch_correlation(q1_mc, q2_mc)[0]

                q_mc_mean = jnp.mean(q_min_mc)
                q_mc_rmse = jnp.sqrt(jnp.mean(q_min_mc ** 2))
                v_mc = v - mc_return
                v_mc_mean = jnp.mean(v_mc)
                v_mc_rmse = jnp.sqrt(jnp.mean(v_mc ** 2))
            else:
                nan_vec = jnp.full((1,), jnp.nan, dtype=jnp.float32)
                mc_cov = nan_vec
                mc_corr = nan_vec
                mc_weighted = nan_vec
                filter_q1_mc_corr = jnp.asarray(jnp.nan, dtype=jnp.float32)
                filter_q2_mc_corr = jnp.asarray(jnp.nan, dtype=jnp.float32)
                twin_mc_error_corr = jnp.asarray(jnp.nan, dtype=jnp.float32)
                q_mc_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
                q_mc_rmse = jnp.asarray(jnp.nan, dtype=jnp.float32)
                v_mc_mean = jnp.asarray(jnp.nan, dtype=jnp.float32)
                v_mc_rmse = jnp.asarray(jnp.nan, dtype=jnp.float32)

            nan_scalar = jnp.asarray(jnp.nan, dtype=jnp.float32)
            metrics = {
                # Common metrics with DD-IQL.
                "diag/filter_source_adv_cov": signal_cov[0],
                "diag/filter_source_adv_corr": signal_corr[0],
                "diag/filter_receiver_adv_corr": signal_corr[0],
                "diag/filter_receiver_loo_cov": loo_cov[0],
                "diag/filter_receiver_loo_cov_abs": jnp.abs(loo_cov[0]),
                "diag/filter_receiver_loo_corr": loo_corr[0],
                "diag/filter_receiver_loo_corr_abs": jnp.abs(loo_corr[0]),
                "diag/filter_source_loo_cov": loo_cov[0],
                "diag/filter_source_loo_corr": loo_corr[0],
                "diag/filter_receiver_loo_weighted_error": loo_weighted[0],
                "diag/filter_source_loo_weighted_error": loo_weighted[0],

                # Source and receiver are structurally identical, but no
                # independent source/receiver oracle errors are measured here.
                # Keep the legacy correlation keys unavailable instead of
                # presenting a structural flag as an empirical correlation.
                "diag/source_receiver_error_corr": nan_scalar,
                "diag/source_receiver_error_corr_abs": nan_scalar,
                "diag/source_receiver_same_estimator": structural_self_coupling,
                "diag/ensemble_q_disagreement_std": twin_disagreement_std,

                # Standard IQL has no delayed filter snapshot. Keep common keys
                # as NaN for plotting compatibility rather than inventing a proxy.
                "diag/filter_receiver_temporal_cov": nan_scalar,
                "diag/filter_receiver_temporal_cov_abs": nan_scalar,
                "diag/filter_receiver_temporal_corr": nan_scalar,
                "diag/filter_source_temporal_cov": nan_scalar,
                "diag/filter_source_temporal_corr": nan_scalar,
                "diag/filter_receiver_temporal_weighted_error": nan_scalar,
                "diag/temporal_error_mean": nan_scalar,
                "diag/temporal_error_std": nan_scalar,
                "diag/temporal_reference_available": jnp.asarray(0.0, dtype=jnp.float32),

                # Bellman residual diagnostics.
                "diag/filter_receiver_td_cov": td_cov[0],
                "diag/filter_receiver_td_corr": td_corr[0],
                "diag/filter_receiver_td_weighted_error": td_weighted[0],
                "diag/td_residual_mean": jnp.mean(q_min_td),
                "diag/td_residual_rmse": jnp.sqrt(jnp.mean(q_min_td ** 2)),

                # Monte-Carlo return diagnostics.
                "diag/filter_receiver_mc_cov": mc_cov[0],
                "diag/filter_receiver_mc_cov_abs": jnp.abs(mc_cov[0]),
                "diag/filter_receiver_mc_corr": mc_corr[0],
                "diag/filter_receiver_mc_corr_abs": jnp.abs(mc_corr[0]),
                "diag/filter_source_mc_cov": mc_cov[0],
                "diag/filter_source_mc_corr": mc_corr[0],
                "diag/filter_receiver_mc_weighted_error": mc_weighted[0],
                "diag/q_minus_mc_mean": q_mc_mean,
                "diag/q_minus_mc_rmse": q_mc_rmse,
                "diag/v_minus_mc_mean": v_mc_mean,
                "diag/v_minus_mc_rmse": v_mc_rmse,

                # IQL-specific twin-Q controls.
                "diag/twin_q_abs_gap": twin_abs_gap,
                "diag/twin_q_corr": q1_q2_corr,
                "diag/twin_td_error_corr": twin_td_error_corr,
                "diag/twin_mc_error_corr": twin_mc_error_corr,
                "diag/filter_q1_mc_corr": filter_q1_mc_corr,
                "diag/filter_q2_mc_corr": filter_q2_mc_corr,
                "diag/q_min_minus_q_mean": jnp.mean(q_min - q_mean),

                # Context / scales.
                "diag/q_mean": jnp.mean(q_min),
                "diag/q1_mean": jnp.mean(q1),
                "diag/q2_mean": jnp.mean(q2),
                "diag/v_mean": jnp.mean(v),
                "diag/filter_weight_mean": jnp.mean(weight),
                "diag/is_self_filter": jnp.asarray(1.0, dtype=jnp.float32),
            }
            # These batch distributions expose upper-tail changes that a mean
            # alone cannot show. They reuse the predictions already computed in
            # this pass; no extra network evaluation or optimizer update occurs.
            for name, values in (
                ("q", q_min), ("q1", q1), ("q2", q2), ("v", v),
                ("receiver_q", q_min), ("receiver_v", v),
                ("receiver_adv", adv), ("adv", adv),
                ("source_q", q_min), ("source_v", v),
                ("source_adv", adv),
            ):
                metrics.update(distribution_summary(values, f"diag/{name}"))
            # The nonnegative branch, including exact ties, is the actual
            # expectile gate; it differs from a strictly positive fraction.
            valid_adv = jnp.isfinite(adv)
            valid_count = jnp.sum(valid_adv)
            high_branch_fraction = jnp.where(
                valid_count > 0,
                jnp.sum(valid_adv & (adv >= 0.0)) / jnp.maximum(valid_count, 1),
                nan_scalar,
            )
            metrics["diag/source_adv_nonnegative_fraction"] = high_branch_fraction
            metrics["diag/receiver_adv_nonnegative_fraction"] = high_branch_fraction
            metrics["diag/filter_high_branch_fraction"] = high_branch_fraction
            metrics["diag/filter_gate_valid_fraction"] = jnp.mean(valid_adv.astype(jnp.float32))
            metrics["diag/receiver_gate_valid_fraction"] = jnp.mean(valid_adv.astype(jnp.float32))
            metrics["diag/source_receiver_gate_mismatch_fraction"] = jnp.where(valid_count > 0, 0.0, nan_scalar)
            metrics["diag/measurement_pre_update"] = jnp.asarray(0.0, dtype=jnp.float32)
            metrics["diag/state_step"] = jnp.asarray(state.total_it, dtype=jnp.float32)
            metrics.update(correlation_valid_flags(metrics))
            return metrics

        return diagnostic_step

    def compute_bias_diagnostics(self, batch: TensorBatch) -> Dict[str, float]:
        # Transfer/synchronize the whole metric pytree once instead of once per key.
        metrics = jax.device_get(self._diagnostic_step(self.state, batch))
        return {key: float(np.asarray(value)) for key, value in metrics.items()}

    def _build_probe_step(self):
        """Evaluate the IQL filter at a supplied state, without changing training.

        The caller invokes this before the scheduled training update. These are
        the weights that that pre-update state assigns to the fixed probe, not a
        claim that the probe itself is the sampled training minibatch.
        """
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        iql_tau = self.iql_tau

        @jax.jit
        def probe_step(state: IQLState, batch: TensorBatch):
            observations = batch["observations"]
            actions = batch["actions"]
            q1, q2 = q_apply({"params": state.q_target_params}, observations, actions)
            receiver_q = jnp.minimum(q1, q2)
            receiver_v = v_apply({"params": state.v_params}, observations)
            receiver_adv = receiver_q - receiver_v
            # Standard IQL constructs the filter from this same current Q/V.
            source_adv = receiver_adv
            weight = jnp.abs(iql_tau - (source_adv < 0.0).astype(jnp.float32))
            return {
                "receiver_q1": q1,
                "receiver_q2": q2,
                "receiver_q": receiver_q,
                "receiver_v": receiver_v,
                "receiver_adv": receiver_adv,
                "source_q1": q1,
                "source_q2": q2,
                "source_q": receiver_q,
                "source_v": receiver_v,
                "source_adv": source_adv,
                "weight": weight,
                "state_step": state.total_it,
                "update_step": state.total_it + 1,
                "source_snapshot_step": state.total_it,
                "delay_age": jnp.asarray(0, dtype=jnp.int32),
            }

        return probe_step

    def compute_probe_diagnostics(self, batch: TensorBatch) -> Dict[str, np.ndarray]:
        """Transfer the complete pre-update probe record to the host once."""
        record = jax.device_get(self._probe_step(self.state, batch))
        return {key: np.asarray(value) for key, value in record.items()}

    def train(self, batch: TensorBatch) -> Dict[str, Any]:
        # Keep metrics on device. The training loop transfers them to host only on
        # actual logging steps, avoiding a GPU->CPU synchronization every step.
        self.state, log_dict = self._train_step(self.state, batch)
        return log_dict

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

    def eval_actor_with_return_diagnostics(
        self,
        env: gym.Env,
        actor_params: Any,
        n_episodes: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compare IQL clipped Q_min/V predictions with realized rollout returns.

        These are empirical return-residual diagnostics, not oracle bias. They are
        most interpretable when training and evaluation rewards have the same scale.
        """
        episode_rewards = []
        episode_successes = []
        q_gaps: List[float] = []
        v_gaps: List[float] = []
        q1_gaps: List[float] = []
        q2_gaps: List[float] = []
        q_sq_errors: List[float] = []
        v_sq_errors: List[float] = []

        for episode_idx in range(n_episodes):
            state = reset_env(env, seed=seed if episode_idx == 0 else None)
            done = False
            episode_reward = 0.0
            episode_success = np.nan
            traj_states: List[np.ndarray] = []
            traj_actions: List[np.ndarray] = []
            traj_rewards: List[float] = []

            while not done:
                action = self.actor_act(actor_params, state)
                traj_states.append(np.asarray(state, dtype=np.float32).copy())
                traj_actions.append(np.asarray(action, dtype=np.float32).copy())
                next_state, reward, done, info = step_env(env, action)
                traj_rewards.append(float(reward))
                episode_reward += reward
                step_success = extract_success(info)
                if np.isfinite(step_success):
                    episode_success = step_success
                state = next_state

            realized_returns = np.zeros(len(traj_rewards), dtype=np.float64)
            running = 0.0
            for idx in range(len(traj_rewards) - 1, -1, -1):
                running = traj_rewards[idx] + self.discount * running
                realized_returns[idx] = running

            if traj_states:
                states_jnp = tree_to_device(
                    jnp.asarray(np.asarray(traj_states), dtype=jnp.float32), self.device
                )
                actions_jnp = tree_to_device(
                    jnp.asarray(np.asarray(traj_actions), dtype=jnp.float32), self.device
                )
                q1, q2 = self.q_def.apply(
                    {"params": self.state.q_target_params}, states_jnp, actions_jnp
                )
                q_min = jnp.minimum(q1, q2)
                v = self.v_def.apply({"params": self.state.v_params}, states_jnp)

                q_min_arr = np.asarray(jax.device_get(q_min), dtype=np.float64)
                q1_arr = np.asarray(jax.device_get(q1), dtype=np.float64)
                q2_arr = np.asarray(jax.device_get(q2), dtype=np.float64)
                v_arr = np.asarray(jax.device_get(v), dtype=np.float64)

                q_diff = q_min_arr - realized_returns
                v_diff = v_arr - realized_returns
                q1_diff = q1_arr - realized_returns
                q2_diff = q2_arr - realized_returns
                q_gaps.extend(q_diff.tolist())
                v_gaps.extend(v_diff.tolist())
                q1_gaps.extend(q1_diff.tolist())
                q2_gaps.extend(q2_diff.tolist())
                q_sq_errors.extend((q_diff ** 2).tolist())
                v_sq_errors.extend((v_diff ** 2).tolist())

            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)

        diagnostic = {
            # Common keys with DD-IQL. q_* refers to IQL's clipped Q_min.
            "eval/q_return_gap_mean": float(np.mean(q_gaps)) if q_gaps else np.nan,
            "eval/q_return_gap_median": float(np.median(q_gaps)) if q_gaps else np.nan,
            "eval/q_return_gap_rmse": float(np.sqrt(np.mean(q_sq_errors))) if q_sq_errors else np.nan,
            "eval/q_over_return_fraction": float(np.mean(np.asarray(q_gaps) > 0.0)) if q_gaps else np.nan,
            "eval/v_return_gap_mean": float(np.mean(v_gaps)) if v_gaps else np.nan,
            "eval/v_return_gap_median": float(np.median(v_gaps)) if v_gaps else np.nan,
            "eval/v_return_gap_rmse": float(np.sqrt(np.mean(v_sq_errors))) if v_sq_errors else np.nan,
            "eval/v_over_return_fraction": float(np.mean(np.asarray(v_gaps) > 0.0)) if v_gaps else np.nan,
            # IQL-specific twin-Q context.
            "eval/q1_return_gap_mean": float(np.mean(q1_gaps)) if q1_gaps else np.nan,
            "eval/q2_return_gap_mean": float(np.mean(q2_gaps)) if q2_gaps else np.nan,
            "eval/q1_over_return_fraction": float(np.mean(np.asarray(q1_gaps) > 0.0)) if q1_gaps else np.nan,
            "eval/q2_over_return_fraction": float(np.mean(np.asarray(q2_gaps) > 0.0)) if q2_gaps else np.nan,
        }
        return (
            np.asarray(episode_rewards, dtype=np.float32),
            np.asarray(episode_successes, dtype=np.float32),
            diagnostic,
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

            q1, q2 = q_apply({"params": iql_state.q_target_params}, observations, actions)
            target_q = jnp.minimum(q1, q2)
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
        self.state = serialization.from_state_dict(self.state, state_dict["iql_state"])
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
            identity_ignored_fields=set(DEFAULT_IDENTITY_IGNORED_FIELDS) | {
                key for key in current_config_dict
                if key.startswith("diagnostic_") or key == "enable_bias_diagnostics"
            },
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

    # Compute trajectory-return diagnostics before observation normalization so
    # boundary inference can compare raw next/current observations.
    if config.enable_bias_diagnostics and config.diagnostic_compute_mc_returns:
        add_mc_return_diagnostics(dataset, discount=config.discount)

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

    diagnostic_batch: Optional[TensorBatch] = None
    diagnostic_probe_batch: Optional[TensorBatch] = None
    diagnostic_probe_indices: Optional[np.ndarray] = None
    if config.enable_bias_diagnostics and not refit_only:
        diagnostic_batch, diagnostic_indices = replay_buffer.sample_fixed(
            batch_size=config.diagnostic_batch_size,
            seed=config.seed + config.diagnostic_seed_offset,
            return_indices=True,
        )
        # Reuse a deterministic subset of the already selected batch. No new
        # replay sample or training RNG draw is needed for the probe.
        probe_size = min(int(config.diagnostic_probe_size), len(diagnostic_indices))
        diagnostic_probe_batch = {
            key: value[:probe_size] for key, value in diagnostic_batch.items()
        }
        diagnostic_probe_indices = diagnostic_indices[:probe_size].copy()
        print(
            "Bias diagnostics enabled: fixed batch size="
            f"{config.diagnostic_batch_size}, freq={config.diagnostic_freq}, "
            f"flush_freq={config.diagnostic_flush_freq}, "
            f"probe_size={probe_size}, probe_freq={config.diagnostic_probe_freq}."
        )

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
    diagnostic_logs: List[Dict[str, Any]] = []
    diagnostic_logs_path: Optional[Path] = None
    if config.checkpoints_path is not None and config.diagnostic_save_npz:
        diagnostic_logs_path = Path(config.checkpoints_path) / "bias_diagnostics.npz"
        diagnostic_logs = load_logs_npz(diagnostic_logs_path)

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

    probe_writer: Optional[ProbeTraceWriter] = None
    if (
        config.enable_bias_diagnostics
        and config.diagnostic_save_probes
        and diagnostic_probe_batch is not None
        and diagnostic_probe_indices is not None
    ):
        if config.diagnostic_probe_dir is not None:
            probe_root = Path(config.diagnostic_probe_dir)
        elif config.checkpoints_path is not None:
            probe_root = Path(config.checkpoints_path) / "diagnostic_probes"
        elif config.log_wandb and wandb.run is not None:
            probe_root = Path(wandb.run.dir) / "diagnostic_probes"
        else:
            probe_root = Path("diagnostics") / config.name / f"seed_{config.seed}" / "diagnostic_probes"
        probe_writer = ProbeTraceWriter(
            root_dir=probe_root,
            probe_indices=diagnostic_probe_indices,
            probe_batch=jax.device_get(diagnostic_probe_batch),
            metadata={
                "algorithm": ALGORITHM_NAME,
                "environment": config.env,
                "seed": int(config.seed),
                "iql_tau": float(config.iql_tau),
                "D": 1,
                "delay_period": 1,
                "discount": float(config.discount),
                "target_update_rate": float(config.tau),
                "diagnostic_seed": int(config.seed + config.diagnostic_seed_offset),
                "diagnostic_probe_freq": int(config.diagnostic_probe_freq),
                "diagnostic_batch_size": int(config.diagnostic_batch_size),
                "probe_size": len(diagnostic_probe_indices),
                "measurement_phase": "pre_update",
                "receiver_q_kind": "minimum_of_current_target_critics",
            },
            start_state_step=int(start_timestep),
        )
        print(f"Fixed pre-update probe traces: {probe_writer.root_dir}")

    def _wandb_log(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))

    def _host_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer a metric pytree to CPU once and convert scalar leaves."""
        host_metrics = jax.device_get(metrics)
        return {key: to_python_scalar(value) for key, value in host_metrics.items()}

    def _append_or_upsert_log(logs: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
        """O(1) append on the normal monotonic path; retain resume safety."""
        step = int(item["timestep"])
        if not logs:
            logs.append(item)
            return
        last_step = int(logs[-1].get("timestep", -1))
        if step > last_step:
            logs.append(item)
        elif step == last_step:
            logs[-1] = item
        else:
            upsert_eval_log(logs, item)

    def _flush_diagnostic_logs(step: int, upload_wandb: bool = True) -> None:
        """Persist buffered diagnostics at a coarse cadence.

        Diagnostic computation remains at diagnostic_freq resolution; only the
        expensive NPZ rewrite and W&B artifact upload are throttled.
        """
        if diagnostic_logs_path is not None and diagnostic_logs:
            save_logs_npz(diagnostic_logs, str(diagnostic_logs_path))
            if upload_wandb and config.log_wandb and wandb.run is not None:
                wandb.save(str(diagnostic_logs_path), policy="now")
            print(
                "Flushed diagnostic logs: "
                f"step={int(step)}, records={len(diagnostic_logs)}"
            )
        if probe_writer is not None:
            # The writer returns only new files, including any chunks flushed
            # automatically since the last call; old chunks are not reuploaded.
            pending_files = probe_writer.flush()
            if upload_wandb and config.log_wandb and wandb.run is not None:
                for probe_path in pending_files:
                    try:
                        wandb.save(str(probe_path), base_path=str(probe_writer.root_dir), policy="now")
                    except Exception as exc:
                        print(f"Warning: probe upload failed; local file retained: {probe_path}: {exc}")


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
            step = int(t + 1)
            batch = replay_buffer.sample(config.batch_size)
            pending_wandb_metrics: Dict[str, Any] = {}
            if (
                probe_writer is not None
                and diagnostic_probe_batch is not None
                and (
                    t == start_timestep
                    or step % int(config.diagnostic_probe_freq) == 0
                    or step == int(config.max_timesteps)
                )
            ):
                # Capture the model before this training update. The smaller
                # fixed probe is independent of the sampled minibatch above.
                probe_record = trainer.compute_probe_diagnostics(diagnostic_probe_batch)
                if int(probe_record["state_step"]) != t:
                    raise RuntimeError("Probe state_step does not match the pre-update trainer state")
                probe_metrics = probe_writer.append(probe_record)
                if config.log_wandb:
                    pending_wandb_metrics.update(probe_metrics)
            device_log_dict = trainer.train(batch)

            # Only synchronize training metrics when they are actually sent to W&B.
            if config.log_wandb and step % config.log_every == 0:
                pending_wandb_metrics.update(_host_scalar_metrics(device_log_dict))

            if (
                config.enable_bias_diagnostics
                and diagnostic_batch is not None
                and step % config.diagnostic_freq == 0
            ):
                diag_metrics = trainer.compute_bias_diagnostics(diagnostic_batch)
                diag_log: Dict[str, Any] = {"timestep": step, **diag_metrics}
                _append_or_upsert_log(diagnostic_logs, diag_log)

                print(
                    "Bias diagnostics: "
                    f"step={step}, "
                    f"receiver_mc_corr={diag_metrics['diag/filter_receiver_mc_corr']:.4f}, "
                    f"receiver_td_corr={diag_metrics['diag/filter_receiver_td_corr']:.4f}, "
                    f"receiver_loo_corr={diag_metrics['diag/filter_receiver_loo_corr']:.4f}, "
                    f"twin_mc_error_corr={diag_metrics['diag/twin_mc_error_corr']:.4f}"
                )

                # Scalar W&B diagnostics remain at diagnostic_freq resolution.
                if config.log_wandb:
                    pending_wandb_metrics.update(diag_metrics)

            # Expensive full-NPZ rewrites and W&B artifact uploads occur only every
            # 25k steps by default. This does not reduce diagnostic data resolution.
            if (
                (config.diagnostic_save_npz or probe_writer is not None)
                and step % int(config.diagnostic_flush_freq) == 0
            ):
                _flush_diagnostic_logs(step, upload_wandb=True)

            if step % config.eval_freq == 0:
                print(f"Time steps: {step}")
                eval_return_diag: Dict[str, float] = {}
                if (
                    config.enable_bias_diagnostics
                    and config.diagnostic_eval_return_gap
                    and not config.normalize_reward
                ):
                    eval_scores, eval_successes, eval_return_diag = trainer.eval_actor_with_return_diagnostics(
                        env,
                        trainer.state.actor_params,
                        n_episodes=config.n_episodes,
                        seed=config.seed,
                    )
                else:
                    if (
                        config.enable_bias_diagnostics
                        and config.diagnostic_eval_return_gap
                        and config.normalize_reward
                    ):
                        print(
                            "Skipping eval return-gap diagnostics because normalize_reward=True; "
                            "critic/value predictions and raw environment returns are on different scales."
                        )
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
                    "timestep": step,
                    "eval/reward_mean": float(np.mean(eval_scores)),
                    "eval/reward_std": float(np.std(eval_scores)),
                    "eval/d4rl_normalized_score_mean": normalized_eval_score_mean,
                    "eval/d4rl_normalized_score_std": normalized_eval_score_std,
                    "eval/success_rate": success_rate,
                    "eval/success_std": success_std,
                    **eval_return_diag,
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
                    pending_wandb_metrics.update(wandb_eval_log)

                save_and_upload_eval_logs(
                    eval_logs=eval_logs,
                    checkpoints_path=config.checkpoints_path,
                    log_wandb=config.log_wandb,
                )

            # One call per step prevents the plain W&B fallback from discarding
            # later records at an already committed step. Probe keys carry their
            # own pre-update state_step/update_step; core diagnostics remain
            # post-update as in the original script.
            if config.log_wandb and pending_wandb_metrics:
                pending_wandb_metrics["training/timestep"] = step
                _wandb_log(pending_wandb_metrics, step)

            # Use the Python loop step for checkpoint scheduling. Reading
            # trainer.state.total_it here would synchronize GPU->CPU every step.
            current_timestep = step
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
        # Preserve diagnostics accumulated since the last coarse flush. This is an
        # exceptional-path disk write, not a steady-state training cost.
        try:
            _flush_diagnostic_logs(_training_timestep(), upload_wandb=True)
        except Exception as flush_exc:
            print(f"Warning: failed to flush diagnostic logs after interruption: {flush_exc}")

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
    _flush_diagnostic_logs(final_timestep, upload_wandb=True)

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


# # JAX/Flax IQL implementation with CDAF_JAX-style experiment plumbing.
# # Algorithmic losses mirror the provided PyTorch IQL code; checkpointing,
# # hyperparameter merging, logging, and path handling follow the CDAF_JAX style.

# import copy
# import os
# import pickle
# import random
# import sys
# import uuid
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union

# import d4rl
# import gym
# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# import pyrallis
# import wandb
# import yaml
# from flax import linen as nn
# from flax import serialization, struct

# TensorBatch = Dict[str, jnp.ndarray]

# ALGORITHM_NAME = "IQL"
# ALGORITHM_FULL_NAME = "Implicit Q-Learning"

# EXP_ADV_MAX = 100.0
# LOG_STD_MIN = -20.0
# LOG_STD_MAX = 2.0


# @dataclass
# class TrainConfig:
#     # Experiment
#     device: str = "gpu"  # one of: cpu, gpu, tpu. JAX selects the matching backend when available.
#     env: str = "halfcheetah-medium-expert-v2"
#     seed: int = 0
#     eval_freq: int = int(25e3)
#     n_episodes: int = 10
#     max_timesteps: int = int(1e6)
#     checkpoints_path: Optional[str] = None
#     load_model: str = ""
#     hyperparams_path: Optional[str] = "hyperparams/iql_jax.yml"
#     use_hyperparams: bool = True

#     # Dataset
#     buffer_size: int = 2_000_000
#     batch_size: int = 256
#     normalize: bool = True
#     normalize_reward: bool = False

#     # IQL
#     discount: float = 0.99
#     tau: float = 0.005
#     beta: float = 3.0
#     iql_tau: float = 0.7
#     iql_deterministic: bool = False
#     vf_lr: float = 3e-4
#     qf_lr: float = 3e-4
#     actor_lr: float = 3e-4
#     actor_dropout: Optional[float] = None

#     # Standalone actor refit from a saved checkpoint.
#     # Used when --load_model is provided with --max_timesteps 0.
#     refit_actor_steps: int = 50000
#     refit_actor_batch_size: int = 256
#     refit_actor_eval_freq: int = 2000
#     actor_refit_dir_name: str = "actor_refit"
#     refit_from_scratch: bool = True

#     # Logging
#     project: str = "ORL-SMOOTH"
#     group: str = "IQL-JAX"
#     name: str = "IQL-JAX"
#     log_wandb: bool = True
#     log_every: int = 500

#     def __post_init__(self):
#         refresh_algorithm_names(self)
#         validate_config(self)


# def refresh_algorithm_names(config: TrainConfig) -> None:
#     config.project = "ORL-SMOOTH"
#     config.group = f"{ALGORITHM_NAME}-JAX"
#     config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


# def validate_config(config: TrainConfig) -> None:
#     assert config.batch_size > 0
#     assert config.buffer_size > 0
#     assert config.eval_freq > 0
#     assert config.n_episodes > 0
#     assert config.max_timesteps >= 0
#     assert config.log_every > 0
#     assert config.discount >= 0.0 and config.discount <= 1.0
#     assert config.tau >= 0.0 and config.tau <= 1.0
#     assert config.beta >= 0.0
#     assert config.iql_tau >= 0.0 and config.iql_tau <= 1.0
#     if config.actor_dropout is not None:
#         assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
#     assert config.refit_actor_steps >= 0
#     assert config.refit_actor_batch_size > 0
#     assert config.refit_actor_eval_freq > 0
#     assert config.actor_refit_dir_name != ""


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
#     # YAML often loads !!float 1e6 as float, but step counts should be ints in this script.
#     if isinstance(value, float) and value.is_integer():
#         return int(value)
#     return value


# def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
#     """Load RL-Zoo-style env-specific hyperparameters and merge them into config.

#     Priority is:
#         dataclass defaults < hyperparams YAML < explicit CLI flags

#     Supported alias:
#         n_timesteps -> max_timesteps
#     """
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
#         print(f"No hyperparameters found for env '{config.env}' in {hparam_path}. Using dataclass/CLI values.")
#         refresh_algorithm_names(config)
#         validate_config(config)
#         return config

#     env_hyperparams = all_hyperparams[config.env] or {}
#     cli_overrides = _cli_overridden_fields()
#     aliases = {
#         "n_timesteps": "max_timesteps",
#     }
#     config_fields = set(TrainConfig.__dataclass_fields__.keys())
#     applied, skipped_unknown, skipped_cli = [], [], []

#     for raw_key, raw_value in env_hyperparams.items():
#         key = aliases.get(raw_key, raw_key)
#         if key not in config_fields:
#             skipped_unknown.append(raw_key)
#             continue
#         if key in cli_overrides or raw_key in cli_overrides:
#             skipped_cli.append(raw_key)
#             continue
#         setattr(config, key, _coerce_hparam_value(raw_value))
#         applied.append(f"{raw_key}->{key}" if raw_key != key else key)

#     refresh_algorithm_names(config)
#     validate_config(config)

#     if applied:
#         print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
#     if skipped_cli:
#         print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
#     if skipped_unknown:
#         print(f"Ignored unknown hyperparameter keys for IQL: {', '.join(skipped_unknown)}")
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


# def soft_update(params, target_params, tau: float):
#     return optax.incremental_update(params, target_params, tau)


# def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
#     mean = states.mean(0)
#     std = states.std(0) + eps
#     return mean, std


# def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
#     return (states - mean) / std


# def wrap_env(
#     env: gym.Env,
#     state_mean: Union[np.ndarray, float] = 0.0,
#     state_std: Union[np.ndarray, float] = 1.0,
#     reward_scale: float = 1.0,
# ) -> gym.Env:
#     def normalize_state(state):
#         return (state - state_mean) / state_std

#     def scale_reward(reward):
#         return reward_scale * reward

#     env = gym.wrappers.TransformObservation(env, normalize_state)
#     if reward_scale != 1.0:
#         env = gym.wrappers.TransformReward(env, scale_reward)
#     return env


# class ReplayBuffer:
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         buffer_size: int,
#         device: Any,
#     ):
#         self._buffer_size = buffer_size
#         self._pointer = 0
#         self._size = 0
#         self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
#         self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
#         self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
#         self._device = device

#     def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
#         if self._size != 0:
#             raise ValueError("Trying to load data into non-empty replay buffer")
#         n_transitions = data["observations"].shape[0]
#         if n_transitions > self._buffer_size:
#             raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

#         self._states[:n_transitions] = data["observations"].astype(np.float32)
#         self._actions[:n_transitions] = data["actions"].astype(np.float32)
#         self._rewards[:n_transitions] = data["rewards"][..., None].astype(np.float32)
#         self._next_states[:n_transitions] = data["next_observations"].astype(np.float32)
#         self._dones[:n_transitions] = data["terminals"][..., None].astype(np.float32)
#         self._size += n_transitions
#         self._pointer = min(self._size, n_transitions)
#         print(f"Dataset size: {n_transitions}")

#     def sample(self, batch_size: int) -> TensorBatch:
#         indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
#         batch = {
#             "observations": self._states[indices],
#             "actions": self._actions[indices],
#             "rewards": self._rewards[indices],
#             "next_observations": self._next_states[indices],
#             "dones": self._dones[indices],
#         }
#         return tree_to_device({k: jnp.asarray(v) for k, v in batch.items()}, self._device)


# def set_seed(seed: int, env: Optional[gym.Env] = None):
#     if env is not None:
#         env.seed(seed)
#         env.action_space.seed(seed)
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


# def save_logs_npz(logs: List[Dict[str, Any]], path: str) -> None:
#     if len(logs) == 0:
#         return
#     keys = logs[0].keys()
#     data_to_save: Dict[str, np.ndarray] = {}
#     for key in keys:
#         values = [log[key] for log in logs]
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


# def normalize_episode_scores(env: gym.Env, eval_scores: np.ndarray) -> np.ndarray:
#     return np.asarray(
#         [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
#         dtype=np.float32,
#     )


# def return_reward_range(dataset, max_episode_steps):
#     returns, lengths = [], []
#     ep_ret, ep_len = 0.0, 0
#     for r, d in zip(dataset["rewards"], dataset["terminals"]):
#         ep_ret += float(r)
#         ep_len += 1
#         if d or ep_len == max_episode_steps:
#             returns.append(ep_ret)
#             lengths.append(ep_len)
#             ep_ret, ep_len = 0.0, 0
#     lengths.append(ep_len)
#     assert sum(lengths) == len(dataset["rewards"])
#     return min(returns), max(returns)


# def modify_reward(dataset, env_name, max_episode_steps=1000):
#     # Preserves the reward preprocessing from the provided PyTorch IQL code.
#     if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
#         min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
#         dataset["rewards"] /= max_ret - min_ret
#         dataset["rewards"] *= max_episode_steps
#     elif "antmaze" in env_name:
#         dataset["rewards"] -= 1.0


# class PolicyMLP(nn.Module):
#     action_dim: int
#     hidden_dim: int = 256
#     n_hidden: int = 2
#     dropout: Optional[float] = None

#     @nn.compact
#     def __call__(self, state: jnp.ndarray, training: bool = False) -> jnp.ndarray:
#         x = state
#         for _ in range(self.n_hidden):
#             x = nn.Dense(self.hidden_dim)(x)
#             x = nn.relu(x)
#             if self.dropout is not None:
#                 x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
#         x = nn.Dense(self.action_dim)(x)
#         return nn.tanh(x)


# class GaussianPolicy(nn.Module):
#     action_dim: int
#     hidden_dim: int = 256
#     n_hidden: int = 2
#     dropout: Optional[float] = None

#     @nn.compact
#     def __call__(self, state: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         mean = PolicyMLP(
#             action_dim=self.action_dim,
#             hidden_dim=self.hidden_dim,
#             n_hidden=self.n_hidden,
#             dropout=self.dropout,
#         )(state, training=training)
#         log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
#         log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         return mean, log_std


# class DeterministicPolicy(nn.Module):
#     action_dim: int
#     hidden_dim: int = 256
#     n_hidden: int = 2
#     dropout: Optional[float] = None

#     @nn.compact
#     def __call__(self, state: jnp.ndarray, training: bool = False) -> jnp.ndarray:
#         return PolicyMLP(
#             action_dim=self.action_dim,
#             hidden_dim=self.hidden_dim,
#             n_hidden=self.n_hidden,
#             dropout=self.dropout,
#         )(state, training=training)


# class QFunction(nn.Module):
#     hidden_dim: int = 256
#     n_hidden: int = 2

#     @nn.compact
#     def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
#         x = jnp.concatenate([state, action], axis=-1)
#         for _ in range(self.n_hidden):
#             x = nn.Dense(self.hidden_dim)(x)
#             x = nn.relu(x)
#         x = nn.Dense(1)(x)
#         return jnp.squeeze(x, axis=-1)


# class TwinQ(nn.Module):
#     hidden_dim: int = 256
#     n_hidden: int = 2

#     @nn.compact
#     def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         q1 = QFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden, name="q1")(state, action)
#         q2 = QFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden, name="q2")(state, action)
#         return q1, q2


# class ValueFunction(nn.Module):
#     hidden_dim: int = 256
#     n_hidden: int = 2

#     @nn.compact
#     def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
#         x = state
#         for _ in range(self.n_hidden):
#             x = nn.Dense(self.hidden_dim)(x)
#             x = nn.relu(x)
#         x = nn.Dense(1)(x)
#         return jnp.squeeze(x, axis=-1)


# @struct.dataclass
# class IQLState:
#     total_it: jnp.ndarray
#     q_params: Any
#     q_target_params: Any
#     q_opt_state: Any
#     v_params: Any
#     v_opt_state: Any
#     actor_params: Any
#     actor_opt_state: Any
#     actor_key: jnp.ndarray


# @struct.dataclass
# class ActorState:
#     params: Any
#     opt_state: Any
#     key: jnp.ndarray


# class IQLJAX:
#     """Implicit Q-Learning in JAX/Flax.

#     The value, Q, and actor update equations mirror the provided PyTorch IQL code.
#     Experiment-management code follows the CDAF_JAX file style.
#     """

#     def __init__(
#         self,
#         max_action: float,
#         state_dim: int,
#         action_dim: int,
#         max_steps: int,
#         qf_lr: float = 3e-4,
#         vf_lr: float = 3e-4,
#         actor_lr: float = 3e-4,
#         discount: float = 0.99,
#         tau: float = 0.005,
#         beta: float = 3.0,
#         iql_tau: float = 0.7,
#         iql_deterministic: bool = False,
#         actor_dropout: Optional[float] = None,
#         seed: int = 0,
#         device: Any = None,
#     ):
#         self.max_action = max_action
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_steps = max_steps
#         self.discount = discount
#         self.tau = tau
#         self.beta = beta
#         self.iql_tau = iql_tau
#         self.iql_deterministic = iql_deterministic
#         self.actor_dropout = actor_dropout
#         self.device = device if device is not None else jax.devices()[0]

#         if iql_deterministic:
#             self.actor_def = DeterministicPolicy(action_dim=action_dim, dropout=actor_dropout)
#         else:
#             self.actor_def = GaussianPolicy(action_dim=action_dim, dropout=actor_dropout)
#         self.q_def = TwinQ()
#         self.v_def = ValueFunction()

#         self.q_tx = optax.adam(qf_lr)
#         self.v_tx = optax.adam(vf_lr)
#         actor_lr_schedule = optax.cosine_decay_schedule(
#             init_value=actor_lr,
#             decay_steps=max(int(max_steps), 1),
#             alpha=0.0,
#         )
#         self.actor_tx = optax.adam(actor_lr_schedule)

#         key = jax.random.PRNGKey(seed)
#         key_actor, key_q, key_v, actor_key = jax.random.split(key, 4)
#         dummy_state = jnp.zeros((1, state_dim), dtype=jnp.float32)
#         dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

#         actor_params = self.actor_def.init(key_actor, dummy_state, training=False)["params"]
#         q_params = self.q_def.init(key_q, dummy_state, dummy_action)["params"]
#         v_params = self.v_def.init(key_v, dummy_state)["params"]

#         self.initial_actor_params = copy.deepcopy(actor_params)
#         self.initial_actor_opt_state = self.actor_tx.init(actor_params)
#         self.initial_actor_key = actor_key

#         self.state = IQLState(
#             total_it=jnp.asarray(0, dtype=jnp.int32),
#             q_params=q_params,
#             q_target_params=copy.deepcopy(q_params),
#             q_opt_state=self.q_tx.init(q_params),
#             v_params=v_params,
#             v_opt_state=self.v_tx.init(v_params),
#             actor_params=actor_params,
#             actor_opt_state=copy.deepcopy(self.initial_actor_opt_state),
#             actor_key=actor_key,
#         )

#         self.state = tree_to_device(self.state, self.device)
#         self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
#         self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
#         self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
#         self._train_step = self._build_train_step()
#         self._actor_refit_step = self._build_actor_refit_step()

#     def _apply_actor(self, actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
#         if self.actor_dropout is not None and training:
#             return self.actor_def.apply(
#                 {"params": actor_params},
#                 observations,
#                 training=training,
#                 rngs={"dropout": rng},
#             )
#         return self.actor_def.apply({"params": actor_params}, observations, training=training)

#     def _build_train_step(self):
#         q_apply = self.q_def.apply
#         v_apply = self.v_def.apply
#         q_tx = self.q_tx
#         v_tx = self.v_tx
#         actor_tx = self.actor_tx
#         discount = self.discount
#         tau = self.tau
#         beta = self.beta
#         iql_tau = self.iql_tau
#         iql_deterministic = self.iql_deterministic
#         use_dropout = self.actor_dropout is not None
#         actor_apply_fn = self.actor_def.apply

#         def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
#             if use_dropout and training:
#                 return actor_apply_fn(
#                     {"params": actor_params},
#                     observations,
#                     training=training,
#                     rngs={"dropout": rng},
#                 )
#             return actor_apply_fn({"params": actor_params}, observations, training=training)

#         @jax.jit
#         def train_step(state: IQLState, batch: TensorBatch):
#             total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
#             observations = batch["observations"]
#             actions = batch["actions"]
#             rewards = jnp.squeeze(batch["rewards"], axis=-1)
#             next_observations = batch["next_observations"]
#             dones = jnp.squeeze(batch["dones"], axis=-1)

#             # Values used by multiple losses are computed from the old state, matching
#             # the sequencing of the provided PyTorch implementation.
#             next_v = v_apply({"params": state.v_params}, next_observations)
#             target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
#             target_q1, target_q2 = q_apply({"params": state.q_target_params}, observations, actions)
#             target_q_for_v = jnp.minimum(target_q1, target_q2)
#             old_v = v_apply({"params": state.v_params}, observations)
#             adv = target_q_for_v - old_v
#             exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

#             def v_loss_fn(v_params):
#                 v = v_apply({"params": v_params}, observations)
#                 value_adv = jax.lax.stop_gradient(target_q_for_v) - v
#                 value_weight = jnp.abs(iql_tau - (value_adv < 0.0).astype(jnp.float32))
#                 value_loss = jnp.mean(value_weight * value_adv ** 2)
#                 return value_loss, v

#             (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
#             v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
#             v_params = optax.apply_updates(state.v_params, v_updates)

#             def q_loss_fn(q_params):
#                 q1, q2 = q_apply({"params": q_params}, observations, actions)
#                 target = jax.lax.stop_gradient(target_q_for_backup)
#                 q_loss = 0.5 * (jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2))
#                 return q_loss, (q1, q2)

#             (q_loss, (q1, q2)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
#             q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
#             q_params = optax.apply_updates(state.q_params, q_updates)
#             q_target_params = soft_update(q_params, state.q_target_params, tau)

#             actor_key, dropout_key = jax.random.split(state.actor_key)

#             def actor_loss_fn(actor_params):
#                 policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
#                 if iql_deterministic:
#                     bc_losses = jnp.sum((policy_out - actions) ** 2, axis=-1)
#                     policy_mean = policy_out
#                     log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
#                 else:
#                     mean, log_std = policy_out
#                     std = jnp.exp(log_std)
#                     log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
#                     bc_losses = -jnp.sum(log_prob, axis=-1)
#                     policy_mean = mean
#                     log_std_mean = jnp.mean(log_std)
#                 actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_losses)
#                 return actor_loss, (bc_losses, policy_mean, log_std_mean)

#             (actor_loss, (bc_losses, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
#                 actor_loss_fn, has_aux=True
#             )(state.actor_params)
#             actor_updates, actor_opt_state = actor_tx.update(
#                 actor_grads,
#                 state.actor_opt_state,
#                 state.actor_params,
#             )
#             actor_params = optax.apply_updates(state.actor_params, actor_updates)

#             new_state = IQLState(
#                 total_it=total_it,
#                 q_params=q_params,
#                 q_target_params=q_target_params,
#                 q_opt_state=q_opt_state,
#                 v_params=v_params,
#                 v_opt_state=v_opt_state,
#                 actor_params=actor_params,
#                 actor_opt_state=actor_opt_state,
#                 actor_key=actor_key,
#             )

#             log_dict = {
#                 "q_loss": q_loss,
#                 "q1_mean": jnp.mean(q1),
#                 "q2_mean": jnp.mean(q2),
#                 "target_q_mean": jnp.mean(target_q_for_backup),
#                 "value_loss": value_loss,
#                 "v_mean": jnp.mean(v),
#                 "adv_mean": jnp.mean(adv),
#                 "adv_min": jnp.min(adv),
#                 "adv_max": jnp.max(adv),
#                 "exp_adv_mean": jnp.mean(exp_adv),
#                 "actor_loss": actor_loss,
#                 "bc_loss_mean": jnp.mean(bc_losses),
#                 "policy_mean": jnp.mean(policy_mean),
#                 "policy_log_std_mean": log_std_mean,
#             }
#             return new_state, log_dict

#         return train_step

#     def train(self, batch: TensorBatch) -> Dict[str, float]:
#         self.state, log_dict = self._train_step(self.state, batch)
#         return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

#     def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
#         state_jnp = tree_to_device(jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device)
#         policy_out = self._apply_actor(actor_params, state_jnp, training=False)
#         action = policy_out if self.iql_deterministic else policy_out[0]
#         action = jnp.clip(self.max_action * action, -self.max_action, self.max_action)
#         return np.asarray(jax.device_get(action))[0]

#     def eval_actor(self, env: gym.Env, actor_params: Any, n_episodes: int, seed: int) -> np.ndarray:
#         env.seed(seed)
#         episode_rewards = []
#         for _ in range(n_episodes):
#             state, done = env.reset(), False
#             episode_reward = 0.0
#             while not done:
#                 action = self.actor_act(actor_params, state)
#                 state, reward, done, _ = env.step(action)
#                 episode_reward += reward
#             episode_rewards.append(episode_reward)
#         return np.asarray(episode_rewards, dtype=np.float32)

#     def _build_actor_refit_step(self):
#         q_apply = self.q_def.apply
#         v_apply = self.v_def.apply
#         actor_tx = self.actor_tx
#         beta = self.beta
#         iql_deterministic = self.iql_deterministic
#         use_dropout = self.actor_dropout is not None
#         actor_apply_fn = self.actor_def.apply

#         def apply_actor(actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
#             if use_dropout and training:
#                 return actor_apply_fn(
#                     {"params": actor_params},
#                     observations,
#                     training=training,
#                     rngs={"dropout": rng},
#                 )
#             return actor_apply_fn({"params": actor_params}, observations, training=training)

#         @jax.jit
#         def actor_refit_step(actor_state: ActorState, iql_state: IQLState, batch: TensorBatch):
#             observations = batch["observations"]
#             actions = batch["actions"]

#             q1, q2 = q_apply({"params": iql_state.q_target_params}, observations, actions)
#             target_q = jnp.minimum(q1, q2)
#             v = v_apply({"params": iql_state.v_params}, observations)
#             adv = target_q - v
#             exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

#             actor_key, dropout_key = jax.random.split(actor_state.key)

#             def actor_loss_fn(actor_params):
#                 policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
#                 if iql_deterministic:
#                     bc_losses = jnp.sum((policy_out - actions) ** 2, axis=-1)
#                     policy_mean = policy_out
#                     log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
#                 else:
#                     mean, log_std = policy_out
#                     std = jnp.exp(log_std)
#                     log_prob = -0.5 * (((actions - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
#                     bc_losses = -jnp.sum(log_prob, axis=-1)
#                     policy_mean = mean
#                     log_std_mean = jnp.mean(log_std)
#                 actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_losses)
#                 return actor_loss, (bc_losses, policy_mean, log_std_mean)

#             (actor_loss, (bc_losses, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
#                 actor_loss_fn, has_aux=True
#             )(actor_state.params)
#             actor_updates, actor_opt_state = actor_tx.update(
#                 actor_grads,
#                 actor_state.opt_state,
#                 actor_state.params,
#             )
#             actor_params = optax.apply_updates(actor_state.params, actor_updates)
#             new_actor_state = ActorState(
#                 params=actor_params,
#                 opt_state=actor_opt_state,
#                 key=actor_key,
#             )
#             log_dict = {
#                 "loss": actor_loss,
#                 "bc_loss": jnp.mean(bc_losses),
#                 "adv_mean": jnp.mean(adv),
#                 "adv_min": jnp.min(adv),
#                 "adv_max": jnp.max(adv),
#                 "exp_adv_mean": jnp.mean(exp_adv),
#                 "exp_adv_max": jnp.max(exp_adv),
#                 "target_q_mean": jnp.mean(target_q),
#                 "v_mean": jnp.mean(v),
#                 "policy_mean": jnp.mean(policy_mean),
#                 "policy_log_std_mean": log_std_mean,
#             }
#             return new_actor_state, log_dict

#         return actor_refit_step

#     def make_initial_actor_state(self) -> ActorState:
#         return tree_to_device(
#             ActorState(
#                 params=copy.deepcopy(self.initial_actor_params),
#                 opt_state=copy.deepcopy(self.initial_actor_opt_state),
#                 key=copy.deepcopy(self.initial_actor_key),
#             ),
#             self.device,
#         )

#     def make_loaded_actor_state(self) -> ActorState:
#         return tree_to_device(
#             ActorState(
#                 params=copy.deepcopy(self.state.actor_params),
#                 opt_state=copy.deepcopy(self.state.actor_opt_state),
#                 key=copy.deepcopy(self.state.actor_key),
#             ),
#             self.device,
#         )

#     def fit_actor(
#         self,
#         replay_buffer: ReplayBuffer,
#         actor_state: ActorState,
#         steps: int,
#         batch_size: int,
#         eval_env: Optional[gym.Env] = None,
#         eval_episodes: int = 0,
#         eval_seed: int = 0,
#         eval_interval: int = 0,
#         prefix: str = "actor_refit",
#     ) -> Tuple[ActorState, Dict[str, Any]]:
#         refit_log: Dict[str, Any] = {
#             f"{prefix}/final_loss": np.nan,
#             f"{prefix}/final_bc_loss": np.nan,
#             f"{prefix}/final_adv_mean": np.nan,
#             f"{prefix}/final_exp_adv_mean": np.nan,
#             f"{prefix}/final_score_mean": np.nan,
#             f"{prefix}/final_score_std": np.nan,
#             f"{prefix}/final_d4rl_normalized_score_mean": np.nan,
#             f"{prefix}/final_d4rl_normalized_score_std": np.nan,
#             f"{prefix}/best_score_mean": np.nan,
#             f"{prefix}/best_score_std": np.nan,
#             f"{prefix}/best_d4rl_normalized_score_mean": np.nan,
#             f"{prefix}/best_d4rl_normalized_score_std": np.nan,
#             f"{prefix}/inner_eval_steps": [],
#             f"{prefix}/inner_score_mean": [],
#             f"{prefix}/inner_score_std": [],
#             f"{prefix}/inner_d4rl_normalized_score_mean": [],
#             f"{prefix}/inner_d4rl_normalized_score_std": [],
#         }
#         if steps <= 0:
#             return actor_state, refit_log

#         best_normalized_score_mean = -np.inf

#         for fit_step in range(1, steps + 1):
#             batch = replay_buffer.sample(batch_size)
#             actor_state, step_log = self._actor_refit_step(actor_state, self.state, batch)
#             step_log = {key: float(jax.device_get(value)) for key, value in step_log.items()}

#             refit_log[f"{prefix}/final_loss"] = step_log["loss"]
#             refit_log[f"{prefix}/final_bc_loss"] = step_log["bc_loss"]
#             refit_log[f"{prefix}/final_adv_mean"] = step_log["adv_mean"]
#             refit_log[f"{prefix}/final_exp_adv_mean"] = step_log["exp_adv_mean"]

#             should_eval = (
#                 eval_env is not None
#                 and eval_episodes > 0
#                 and eval_interval > 0
#                 and (fit_step % eval_interval == 0 or fit_step == steps)
#             )
#             if should_eval:
#                 eval_scores = self.eval_actor(
#                     eval_env,
#                     actor_state.params,
#                     n_episodes=eval_episodes,
#                     seed=eval_seed,
#                 )
#                 normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

#                 eval_score_mean = float(np.mean(eval_scores))
#                 eval_score_std = float(np.std(eval_scores))
#                 normalized_eval_score_mean = float(np.mean(normalized_eval_scores))
#                 normalized_eval_score_std = float(np.std(normalized_eval_scores))

#                 refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
#                 refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
#                 refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
#                 refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
#                 refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
#                 refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
#                 refit_log[f"{prefix}/final_score_std"] = eval_score_std
#                 refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
#                 refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std

#                 if normalized_eval_score_mean > best_normalized_score_mean:
#                     best_normalized_score_mean = normalized_eval_score_mean
#                     refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
#                     refit_log[f"{prefix}/best_score_std"] = eval_score_std
#                     refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
#                     refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std

#                 print(
#                     f"[{prefix}:iql_awbc] step {fit_step}/{steps}: "
#                     f"loss={step_log['loss']:.4f}, bc={step_log['bc_loss']:.4f}, "
#                     f"adv={step_log['adv_mean']:.4f}, exp_adv={step_log['exp_adv_mean']:.4f}, "
#                     f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
#                     f"D4RL_mean={normalized_eval_score_mean:.3f}, "
#                     f"D4RL_std={normalized_eval_score_std:.3f}"
#                 )

#         return actor_state, refit_log

#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "iql_state": serialization.to_state_dict(self.state),
#             "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
#             "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
#             "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
#             "iql_deterministic": self.iql_deterministic,
#         }

#     def load_state_dict(self, state_dict: Dict[str, Any]):
#         self.state = serialization.from_state_dict(self.state, state_dict["iql_state"])
#         if "initial_actor_params" in state_dict:
#             self.initial_actor_params = serialization.from_state_dict(
#                 self.initial_actor_params,
#                 state_dict["initial_actor_params"],
#             )
#         if "initial_actor_opt_state" in state_dict:
#             self.initial_actor_opt_state = serialization.from_state_dict(
#                 self.initial_actor_opt_state,
#                 state_dict["initial_actor_opt_state"],
#             )
#         if "initial_actor_key" in state_dict:
#             self.initial_actor_key = serialization.from_state_dict(
#                 self.initial_actor_key,
#                 state_dict["initial_actor_key"],
#             )
#         self.state = tree_to_device(self.state, self.device)
#         self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
#         self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
#         self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)


# def save_pickle(path: Union[str, Path], obj: Any) -> None:
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)


# def load_pickle(path: Union[str, Path]) -> Any:
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def resolve_checkpoint_path(
#     load_model: Union[str, Path],
#     run_name: Optional[str] = None,
#     seed: Optional[int] = None,
# ) -> Tuple[Path, Path]:
#     """Return (run_dir, checkpoint_path) for a saved IQL-JAX checkpoint.

#     Supported load_model formats:

#     1. Direct checkpoint file:
#        path/to/checkpoint.pkl

#     2. Direct run directory:
#        path/to/run_dir/
#        where path/to/run_dir/checkpoint.pkl exists

#     3. Parent directory that contains env/seed subdirectory:
#        path/to/base_dir/
#        where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists
#     """
#     load_path = Path(load_model)

#     if load_path.is_file():
#         if load_path.name != "checkpoint.pkl":
#             raise FileNotFoundError(
#                 f"load_model points to a file, but it is not checkpoint.pkl: {load_path}"
#             )
#         return load_path.parent, load_path

#     if not load_path.exists():
#         raise FileNotFoundError(f"load_model path does not exist: {load_path}")

#     candidates: List[Path] = []
#     candidates.append(load_path / "checkpoint.pkl")

#     if run_name is not None and seed is not None:
#         candidates.append(load_path / run_name / str(seed) / "checkpoint.pkl")

#     if run_name is not None:
#         run_name_dir = load_path / run_name
#         if run_name_dir.exists():
#             candidates.extend(sorted(run_name_dir.glob("*/checkpoint.pkl")))

#     candidates.extend(sorted(load_path.glob("*/checkpoint.pkl")))
#     candidates.extend(sorted(load_path.glob("*/*/checkpoint.pkl")))

#     seen = set()
#     existing_candidates = []
#     for candidate in candidates:
#         candidate = candidate.resolve()
#         if candidate in seen:
#             continue
#         seen.add(candidate)
#         if candidate.exists():
#             existing_candidates.append(candidate)

#     if len(existing_candidates) == 0:
#         tried = "\n".join(str(p) for p in candidates[:20])
#         raise FileNotFoundError(
#             f"checkpoint file not found under: {load_path}\n"
#             f"Tried candidates:\n{tried}"
#         )

#     if run_name is not None and seed is not None:
#         exact = (load_path / run_name / str(seed) / "checkpoint.pkl").resolve()
#         if exact in existing_candidates:
#             return exact.parent, exact

#     if len(existing_candidates) > 1:
#         found = "\n".join(str(p) for p in existing_candidates)
#         raise FileNotFoundError(
#             f"Multiple checkpoint.pkl files found under {load_path}.\n"
#             f"Please provide a more specific --load_model path.\n"
#             f"Found:\n{found}"
#         )

#     checkpoint_path = existing_candidates[0]
#     return checkpoint_path.parent, checkpoint_path


# @pyrallis.wrap()
# def train(config: TrainConfig):
#     config = apply_env_hyperparams(config)
#     refit_only = config.load_model != "" and int(config.max_timesteps) <= 0
#     if not refit_only:
#         config = finalize_checkpoint_path(config)

#     jax_device = select_jax_device(config.device)
#     env = gym.make(config.env)

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     dataset = d4rl.qlearning_dataset(env)
#     if config.normalize_reward:
#         modify_reward(dataset, config.env)

#     if config.normalize:
#         state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
#     else:
#         state_mean, state_std = 0, 1

#     dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
#     dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
#     env = wrap_env(env, state_mean=state_mean, state_std=state_std)

#     replay_buffer = ReplayBuffer(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         buffer_size=config.buffer_size,
#         device=jax_device,
#     )
#     replay_buffer.load_d4rl_dataset(dataset)

#     max_action = float(env.action_space.high[0])

#     if config.checkpoints_path is not None and not refit_only:
#         print(f"Checkpoints path: {config.checkpoints_path}")
#         os.makedirs(config.checkpoints_path, exist_ok=True)
#         config_path = os.path.join(config.checkpoints_path, "config.yaml")
#         if os.path.exists(config_path):
#             print(f"Error: The file '{config_path}' already exists.")
#             exit(1)
#         with open(config_path, "w") as f:
#             pyrallis.dump(config, f)

#     seed = config.seed
#     set_seed(seed, env)

#     print("---------------------------------------")
#     print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
#     print("---------------------------------------")

#     trainer = IQLJAX(
#         max_action=max_action,
#         state_dim=state_dim,
#         action_dim=action_dim,
#         max_steps=max(int(config.max_timesteps), int(config.refit_actor_steps), 1),
#         qf_lr=config.qf_lr,
#         vf_lr=config.vf_lr,
#         actor_lr=config.actor_lr,
#         discount=config.discount,
#         tau=config.tau,
#         beta=config.beta,
#         iql_tau=config.iql_tau,
#         iql_deterministic=config.iql_deterministic,
#         actor_dropout=config.actor_dropout,
#         seed=seed,
#         device=jax_device,
#     )

#     loaded_run_dir: Optional[Path] = None
#     if config.load_model != "":
#         loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
#             config.load_model,
#             run_name=config.name,
#             seed=config.seed,
#         )
#         print(f"Loading checkpoint from: {checkpoint_path}")
#         checkpoint = load_pickle(checkpoint_path)
#         trainer.load_state_dict(checkpoint)

#     if config.log_wandb:
#         wandb_init(asdict(config))

#     if refit_only:
#         if loaded_run_dir is None:
#             raise ValueError("refit_only mode requires --load_model")

#         actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
#         actor_refit_dir.mkdir(parents=True, exist_ok=True)
#         print("---------------------------------------")
#         print(f"Actor refit from saved {ALGORITHM_NAME} checkpoint")
#         print(f"Saving actor refit outputs to: {actor_refit_dir}")
#         print("---------------------------------------")

#         actor_state = (
#             trainer.make_initial_actor_state()
#             if config.refit_from_scratch
#             else trainer.make_loaded_actor_state()
#         )
#         refit_actor_state, refit_log = trainer.fit_actor(
#             replay_buffer=replay_buffer,
#             actor_state=actor_state,
#             steps=config.refit_actor_steps,
#             batch_size=config.refit_actor_batch_size,
#             eval_env=env,
#             eval_episodes=config.n_episodes,
#             eval_seed=config.seed,
#             eval_interval=config.refit_actor_eval_freq,
#             prefix="actor_refit",
#         )

#         save_pickle(
#             actor_refit_dir / "final_actor.pkl",
#             serialization.to_state_dict(refit_actor_state.params),
#         )
#         save_logs_npz(
#             [{"loaded_checkpoint": str(loaded_run_dir / "checkpoint.pkl"), **refit_log}],
#             str(actor_refit_dir / "fit_eval_logs.npz"),
#         )
#         with open(actor_refit_dir / "refit_config.yaml", "w") as f:
#             pyrallis.dump(config, f)

#         if config.log_wandb and wandb.run is not None:
#             wandb.save(str(actor_refit_dir / "final_actor.pkl"), policy="now")
#             wandb.save(str(actor_refit_dir / "fit_eval_logs.npz"), policy="now")
#             wandb.save(str(actor_refit_dir / "refit_config.yaml"), policy="now")

#         print("---------------------------------------")
#         print("Actor refit finished")
#         print(f"Saved final actor to: {actor_refit_dir / 'final_actor.pkl'}")
#         print(f"Saved fit logs to:    {actor_refit_dir / 'fit_eval_logs.npz'}")
#         print("---------------------------------------")
#         return

#     eval_logs: List[Dict[str, Any]] = []
#     for t in range(int(config.max_timesteps)):
#         batch = replay_buffer.sample(config.batch_size)
#         log_dict = trainer.train(batch)

#         if config.log_wandb and (t + 1) % config.log_every == 0:
#             wandb.log(log_dict, step=int(jax.device_get(trainer.state.total_it)))

#         if (t + 1) % config.eval_freq == 0:
#             print(f"Time steps: {t + 1}")
#             eval_scores = trainer.eval_actor(
#                 env,
#                 trainer.state.actor_params,
#                 n_episodes=config.n_episodes,
#                 seed=config.seed,
#             )
#             normalized_eval_scores = normalize_episode_scores(env, eval_scores)

#             eval_log: Dict[str, Any] = {
#                 "timestep": int(t + 1),
#                 "eval/reward_mean": float(np.mean(eval_scores)),
#                 "eval/reward_std": float(np.std(eval_scores)),
#                 "eval/normalized_score_mean": float(np.mean(normalized_eval_scores)),
#                 "eval/normalized_score_std": float(np.std(normalized_eval_scores)),
#             }
#             upsert_eval_log(eval_logs, eval_log)

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
#                 wandb.log(wandb_eval_log, step=int(jax.device_get(trainer.state.total_it)))

#             save_and_upload_eval_logs(
#                 eval_logs=eval_logs,
#                 checkpoints_path=config.checkpoints_path,
#                 log_wandb=config.log_wandb,
#             )

#     if config.checkpoints_path is not None:
#         checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
#         save_pickle(checkpoint_path, trainer.state_dict())

#         if config.log_wandb and wandb.run is not None:
#             wandb.save(checkpoint_path, policy="now")

#         save_and_upload_eval_logs(
#             eval_logs=eval_logs,
#             checkpoints_path=config.checkpoints_path,
#             log_wandb=config.log_wandb,
#         )


# if __name__ == "__main__":
#     train()





