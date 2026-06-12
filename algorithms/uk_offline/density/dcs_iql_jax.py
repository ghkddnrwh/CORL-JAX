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

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "DCS-IQL"
ALGORITHM_FULL_NAME = "Density-Certified Stitching Implicit Q-Learning"

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

    # DCS-IQL: Density-Certified Stitching IQL
    #
    # The previous DC-IQL version used a scalar density confidence c(s) to
    # change the expectile level tau(s). The new research direction keeps tau
    # mostly global and uses density as a certificate for explicit local
    # stitching. A one-time kNN preprocessing step builds a local coverage
    # profile:
    #   rho(s): state-density confidence from kNN radii
    #   b(s):   local action-diversity confidence
    #   J(s):   rho(s) * b(s), a junction/stitching score
    # The profile then controls which neighbor transitions are allowed to enter
    # the value expectile target pool and the actor AWR extraction pool.
    use_density_calibration: bool = True

    # Legacy ablation only. Keep False for the new method. If True, tau(s) is
    # still computed from rho(s), but this is not the recommended setting.
    dc_use_state_tau: bool = False
    dc_tau_min: float = 0.7
    dc_tau_max: float = 0.9

    # kNN/profile preprocessing. dc_density_k is the number of neighbor states
    # excluding self; the training-time pool size is therefore k+1.
    dc_density_k: int = 10
    dc_density_subsample: int = 10_000_000
    dc_density_chunk_size: int = 50_000
    dc_density_percentile_low: float = 5.0
    dc_density_percentile_high: float = 95.0
    dc_action_diversity_percentile_low: float = 5.0
    dc_action_diversity_percentile_high: float = 95.0
    dc_junction_percentile: float = 60.0
    dc_kernel_scale: float = 1.0

    # Coverage profile cache. Saved as:
    #   {dc_density_model_path}/{env}/coverage_profile.npz
    # or, if dc_density_cache_by_seed=True:
    #   {dc_density_model_path}/{env}/seed_{seed}/coverage_profile.npz
    dc_density_model_path: Optional[str] = None
    dc_density_force_recompute: bool = False
    dc_density_cache_by_seed: bool = False

    # Switches for the two core mechanisms proposed in the memo.
    dc_use_vicinal_value: bool = True
    dc_use_vicinal_actor: bool = True

    # Deprecated smoothing ablation from the previous DC-IQL version. Default 0
    # because the new proposal replaces smoothing with explicit neighbor pools.
    dc_vicinal_lambda: float = 0.0
    dc_vicinal_noise_std: float = 0.01
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
    project: str = "ORL-BIAS"
    group: str = "DC-IQL-JAX"
    name: str = "DC-IQL-JAX"
    log_wandb: bool = True
    log_every: int = 500

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    # config.project = "ORL-BIAS"
    # config.group = f"{ALGORITHM_NAME}-JAX"
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
    assert config.beta >= 0.0
    assert config.iql_tau >= 0.0 and config.iql_tau <= 1.0
    assert config.dc_tau_min >= 0.0 and config.dc_tau_min <= 1.0
    assert config.dc_tau_max >= 0.0 and config.dc_tau_max <= 1.0
    assert config.dc_tau_min <= config.dc_tau_max
    assert config.dc_density_k >= 1
    assert config.dc_density_subsample >= 1
    assert config.dc_density_chunk_size >= 1
    assert config.dc_density_percentile_low >= 0.0 and config.dc_density_percentile_low < config.dc_density_percentile_high
    assert config.dc_density_percentile_high <= 100.0
    assert config.dc_action_diversity_percentile_low >= 0.0
    assert config.dc_action_diversity_percentile_low < config.dc_action_diversity_percentile_high
    assert config.dc_action_diversity_percentile_high <= 100.0
    assert config.dc_junction_percentile >= 0.0 and config.dc_junction_percentile <= 100.0
    assert config.dc_kernel_scale > 0.0
    if config.dc_density_model_path is not None:
        assert config.dc_density_model_path != ""
    assert config.dc_vicinal_lambda >= 0.0
    assert config.dc_vicinal_noise_std >= 0.0
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




def _percentile_confidence(
    values: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    high_is_good: bool,
) -> np.ndarray:
    """Map a scalar statistic to [0, 1] by percentile clipping."""
    values = np.asarray(values, dtype=np.float32)
    lo = np.percentile(values, percentile_low)
    hi = np.percentile(values, percentile_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(values, dtype=np.float32)
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    if high_is_good:
        return scaled.astype(np.float32)
    return (1.0 - scaled).astype(np.float32)


def compute_local_coverage_profile(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    k: int = 10,
    subsample_size: int = 100_000,
    chunk_size: int = 50_000,
    density_percentile_low: float = 5.0,
    density_percentile_high: float = 95.0,
    action_diversity_percentile_low: float = 5.0,
    action_diversity_percentile_high: float = 95.0,
    junction_percentile: float = 60.0,
    kernel_scale: float = 1.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Build the one-time local coverage profile used by DCS-IQL.

    Returns a dictionary with one scalar profile per transition plus the
    training-time neighbor pool:
      - density_confidence rho(s): larger means denser local state coverage.
      - action_diversity b(s): larger means more diverse neighbor actions.
      - junction_score J(s)=rho(s)*b(s).
      - junction_gate: 1 if J(s) is above the selected percentile threshold.
      - neighbor_indices: [N, k+1], first column is always self.
      - neighbor_weights: [N, k+1], normalized RBF weights; neighbor columns are
        zeroed when junction_gate=0, so the pool collapses to self only.

    This implements the memo's proposed shift from "tau(s) as the control
    variable" to "the expectile target distribution itself as the control
    variable." Tau can remain global while the Bellman/policy target pool expands
    only at certified stitching junctions.
    """
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    n = states.shape[0]
    k = int(k)
    if n == 0:
        empty_1 = np.zeros((0, 1), dtype=np.float32)
        return {
            "density_confidence": empty_1,
            "action_diversity": empty_1,
            "junction_score": empty_1,
            "junction_gate": empty_1,
            "knn_radius": empty_1,
            "neighbor_indices": np.zeros((0, k + 1), dtype=np.int32),
            "neighbor_weights": np.zeros((0, k + 1), dtype=np.float32),
            "junction_threshold": np.asarray(np.nan, dtype=np.float32),
        }

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise ImportError(
            "DCS-IQL requires scipy.spatial.cKDTree for certified stitching profile construction."
        ) from exc

    rng = np.random.default_rng(seed)
    ref_size = int(min(n, subsample_size))
    ref_idx = rng.choice(n, size=ref_size, replace=False) if ref_size < n else np.arange(n)
    ref_states = states[ref_idx]
    tree = cKDTree(ref_states)

    # Query a few extra neighbors so self can be removed when the full dataset is
    # used as the reference set. If the reference set is tiny, padding will fill
    # missing slots with self and zero distance.
    query_k = int(min(max(k + 2, 1), ref_size))
    neighbor_indices = np.empty((n, k), dtype=np.int32)
    neighbor_distances = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        distances, ref_positions = tree.query(states[start:end], k=query_k, workers=-1)
        if query_k == 1:
            distances = distances[:, None]
            ref_positions = ref_positions[:, None]
        candidate_indices = ref_idx[np.asarray(ref_positions, dtype=np.int64)]
        distances = np.asarray(distances, dtype=np.float32)

        for row, original_i in enumerate(range(start, end)):
            cand = candidate_indices[row]
            dist = distances[row]
            keep = cand != original_i
            cand = cand[keep]
            dist = dist[keep]
            if cand.shape[0] >= k:
                neighbor_indices[original_i] = cand[:k]
                neighbor_distances[original_i] = dist[:k]
            else:
                fill_n = cand.shape[0]
                if fill_n > 0:
                    neighbor_indices[original_i, :fill_n] = cand
                    neighbor_distances[original_i, :fill_n] = dist
                neighbor_indices[original_i, fill_n:] = original_i
                neighbor_distances[original_i, fill_n:] = 0.0

    # Density: smaller kNN radius means higher confidence.
    knn_radius = np.max(neighbor_distances, axis=1).astype(np.float32)
    density_confidence = _percentile_confidence(
        knn_radius,
        percentile_low=density_percentile_low,
        percentile_high=density_percentile_high,
        high_is_good=False,
    )

    # Local action diversity: trace of the empirical neighbor action covariance.
    all_neighbor_indices = np.concatenate(
        [np.arange(n, dtype=np.int32)[:, None], neighbor_indices], axis=1
    )
    action_pool = actions[all_neighbor_indices]
    action_center = np.mean(action_pool, axis=1, keepdims=True)
    action_diversity_raw = np.mean(
        np.sum((action_pool - action_center) ** 2, axis=-1), axis=1
    ).astype(np.float32)
    action_diversity = _percentile_confidence(
        action_diversity_raw,
        percentile_low=action_diversity_percentile_low,
        percentile_high=action_diversity_percentile_high,
        high_is_good=True,
    )

    junction_score = (density_confidence * action_diversity).astype(np.float32)
    junction_threshold = np.percentile(junction_score, junction_percentile).astype(np.float32)
    junction_gate = (junction_score >= junction_threshold).astype(np.float32)

    all_distances = np.concatenate(
        [np.zeros((n, 1), dtype=np.float32), neighbor_distances], axis=1
    )
    # Local bandwidth. The max radius is stable and cheap; kernel_scale controls
    # how quickly neighbors are down-weighted inside the certified pool.
    bandwidth = np.maximum(knn_radius[:, None] * float(kernel_scale), 1e-6)
    neighbor_weights = np.exp(-((all_distances / bandwidth) ** 2)).astype(np.float32)
    neighbor_weights[:, 0] = 1.0
    neighbor_weights[:, 1:] *= junction_gate[:, None]
    neighbor_weights = neighbor_weights / np.maximum(
        np.sum(neighbor_weights, axis=1, keepdims=True), 1e-8
    )

    return {
        "density_confidence": density_confidence.reshape(-1, 1).astype(np.float32),
        "action_diversity": action_diversity.reshape(-1, 1).astype(np.float32),
        "junction_score": junction_score.reshape(-1, 1).astype(np.float32),
        "junction_gate": junction_gate.reshape(-1, 1).astype(np.float32),
        "knn_radius": knn_radius.reshape(-1, 1).astype(np.float32),
        "neighbor_indices": all_neighbor_indices.astype(np.int32),
        "neighbor_weights": neighbor_weights.astype(np.float32),
        "junction_threshold": np.asarray(junction_threshold, dtype=np.float32),
    }


def _safe_path_name(name: str) -> str:
    """Make env names safe for directory paths."""
    return name.replace("/", "_").replace(":", "_")


def get_coverage_profile_cache_path(config: TrainConfig) -> Optional[Path]:
    """Return the cache file path for the precomputed coverage profile."""
    if config.dc_density_model_path is None:
        return None

    root = Path(config.dc_density_model_path)
    env_name = _safe_path_name(config.env)

    if config.dc_density_cache_by_seed:
        return root / env_name / f"seed_{config.seed}" / "coverage_profile.npz"

    return root / env_name / "coverage_profile.npz"


def _canonicalize_coverage_profile_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata types for reliable equality checks after JSON load."""
    canonical = dict(metadata)
    for key in ("observations_shape", "actions_shape", "next_observations_shape"):
        if key in canonical:
            canonical[key] = tuple(canonical[key])
    return canonical


def build_coverage_profile_metadata(
    config: TrainConfig,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
) -> Dict[str, Any]:
    """Metadata that must match before a cached coverage profile can be reused."""
    return _canonicalize_coverage_profile_metadata(
        {
            "cache_version": 2,
            "env": config.env,
            "seed": int(config.seed) if config.dc_density_cache_by_seed else None,
            "normalize": bool(config.normalize),
            "observations_shape": tuple(observations.shape),
            "actions_shape": tuple(actions.shape),
            "next_observations_shape": tuple(next_observations.shape),
            "observations_dtype": str(np.asarray(observations).dtype),
            "actions_dtype": str(np.asarray(actions).dtype),
            "dc_density_k": int(config.dc_density_k),
            "dc_density_subsample": int(config.dc_density_subsample),
            "dc_density_chunk_size": int(config.dc_density_chunk_size),
            "dc_density_percentile_low": float(config.dc_density_percentile_low),
            "dc_density_percentile_high": float(config.dc_density_percentile_high),
            "dc_action_diversity_percentile_low": float(config.dc_action_diversity_percentile_low),
            "dc_action_diversity_percentile_high": float(config.dc_action_diversity_percentile_high),
            "dc_junction_percentile": float(config.dc_junction_percentile),
            "dc_kernel_scale": float(config.dc_kernel_scale),
        }
    )


def load_coverage_profile_cache(
    cache_path: Path,
    expected_metadata: Dict[str, Any],
) -> Optional[Dict[str, np.ndarray]]:
    """Load coverage profile if the cache exists and metadata matches."""
    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=False)
        saved_metadata_raw = payload["metadata"]
        if hasattr(saved_metadata_raw, "item"):
            saved_metadata_raw = saved_metadata_raw.item()
        saved_metadata = _canonicalize_coverage_profile_metadata(json.loads(str(saved_metadata_raw)))
        expected_metadata = _canonicalize_coverage_profile_metadata(expected_metadata)

        if saved_metadata != expected_metadata:
            print(f"Coverage profile cache metadata mismatch. Recomputing profile: {cache_path}")
            print(f"Saved metadata:    {saved_metadata}")
            print(f"Expected metadata: {expected_metadata}")
            return None

        expected_n = int(expected_metadata["observations_shape"][0])
        expected_k = int(expected_metadata["dc_density_k"]) + 1
        profile = {
            "density_confidence": np.asarray(payload["density_confidence"], dtype=np.float32),
            "action_diversity": np.asarray(payload["action_diversity"], dtype=np.float32),
            "junction_score": np.asarray(payload["junction_score"], dtype=np.float32),
            "junction_gate": np.asarray(payload["junction_gate"], dtype=np.float32),
            "knn_radius": np.asarray(payload["knn_radius"], dtype=np.float32),
            "neighbor_indices": np.asarray(payload["neighbor_indices"], dtype=np.int32),
            "neighbor_weights": np.asarray(payload["neighbor_weights"], dtype=np.float32),
            "junction_threshold": np.asarray(payload["junction_threshold"], dtype=np.float32),
        }
        if profile["density_confidence"].shape != (expected_n, 1):
            print("Coverage profile density shape mismatch. Recomputing profile.")
            return None
        if profile["neighbor_indices"].shape != (expected_n, expected_k):
            print("Coverage profile neighbor index shape mismatch. Recomputing profile.")
            return None
        if profile["neighbor_weights"].shape != (expected_n, expected_k):
            print("Coverage profile neighbor weight shape mismatch. Recomputing profile.")
            return None

        print(f"Loaded coverage profile from: {cache_path}")
        return profile

    except Exception as exc:
        print(f"Failed to load coverage profile cache from {cache_path}: {exc}")
        print("Recomputing coverage profile.")
        return None


def save_coverage_profile_cache(
    cache_path: Path,
    profile: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """Save precomputed coverage profile and the metadata needed for reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        density_confidence=np.asarray(profile["density_confidence"], dtype=np.float32),
        action_diversity=np.asarray(profile["action_diversity"], dtype=np.float32),
        junction_score=np.asarray(profile["junction_score"], dtype=np.float32),
        junction_gate=np.asarray(profile["junction_gate"], dtype=np.float32),
        knn_radius=np.asarray(profile["knn_radius"], dtype=np.float32),
        neighbor_indices=np.asarray(profile["neighbor_indices"], dtype=np.int32),
        neighbor_weights=np.asarray(profile["neighbor_weights"], dtype=np.float32),
        junction_threshold=np.asarray(profile["junction_threshold"], dtype=np.float32),
        metadata=json.dumps(_canonicalize_coverage_profile_metadata(metadata)),
    )
    print(f"Saved coverage profile to: {cache_path}")


def load_or_compute_coverage_profile(
    config: TrainConfig,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Load cached local coverage profile, or compute and save it."""
    cache_path = get_coverage_profile_cache_path(config)
    metadata = build_coverage_profile_metadata(config, observations, actions, next_observations)

    if cache_path is not None and not config.dc_density_force_recompute:
        cached_profile = load_coverage_profile_cache(
            cache_path=cache_path,
            expected_metadata=metadata,
        )
        if cached_profile is not None:
            return cached_profile

    if cache_path is not None and config.dc_density_force_recompute:
        print(f"Ignoring existing coverage profile cache because dc_density_force_recompute=True: {cache_path}")

    print("Computing one-time local coverage profile for DCS-IQL...")
    profile = compute_local_coverage_profile(
        states=observations,
        actions=actions,
        next_states=next_observations,
        k=config.dc_density_k,
        subsample_size=config.dc_density_subsample,
        chunk_size=config.dc_density_chunk_size,
        density_percentile_low=config.dc_density_percentile_low,
        density_percentile_high=config.dc_density_percentile_high,
        action_diversity_percentile_low=config.dc_action_diversity_percentile_low,
        action_diversity_percentile_high=config.dc_action_diversity_percentile_high,
        junction_percentile=config.dc_junction_percentile,
        kernel_scale=config.dc_kernel_scale,
        seed=config.seed,
    )

    if cache_path is not None:
        save_coverage_profile_cache(
            cache_path=cache_path,
            profile=profile,
            metadata=metadata,
        )

    return profile


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

        # Local coverage profile fields. Defaults collapse the vicinal pool to
        # self only, making the algorithm exactly standard IQL when no profile is
        # provided or the corresponding switches are disabled.
        self._density_confidences = np.ones((buffer_size, 1), dtype=np.float32)
        self._action_diversities = np.zeros((buffer_size, 1), dtype=np.float32)
        self._junction_scores = np.zeros((buffer_size, 1), dtype=np.float32)
        self._junction_gates = np.zeros((buffer_size, 1), dtype=np.float32)
        self._neighbor_indices = np.zeros((buffer_size, 1), dtype=np.int32)
        self._neighbor_weights = np.ones((buffer_size, 1), dtype=np.float32)
        self._device = device

    def load_d4rl_dataset(
        self,
        data: Dict[str, np.ndarray],
        coverage_profile: Optional[Dict[str, np.ndarray]] = None,
        coverage: Optional[np.ndarray] = None,
    ):
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

        self_indices = np.arange(n_transitions, dtype=np.int32).reshape(-1, 1)
        self._neighbor_indices[:n_transitions] = self_indices
        self._neighbor_weights[:n_transitions] = 1.0

        if coverage_profile is not None:
            required = [
                "density_confidence",
                "action_diversity",
                "junction_score",
                "junction_gate",
                "neighbor_indices",
                "neighbor_weights",
            ]
            for key in required:
                if key not in coverage_profile:
                    raise KeyError(f"coverage_profile is missing required key: {key}")

            density_confidence = np.asarray(coverage_profile["density_confidence"], dtype=np.float32).reshape(-1, 1)
            action_diversity = np.asarray(coverage_profile["action_diversity"], dtype=np.float32).reshape(-1, 1)
            junction_score = np.asarray(coverage_profile["junction_score"], dtype=np.float32).reshape(-1, 1)
            junction_gate = np.asarray(coverage_profile["junction_gate"], dtype=np.float32).reshape(-1, 1)
            neighbor_indices = np.asarray(coverage_profile["neighbor_indices"], dtype=np.int32)
            neighbor_weights = np.asarray(coverage_profile["neighbor_weights"], dtype=np.float32)

            if density_confidence.shape[0] != n_transitions:
                raise ValueError("density_confidence must have one scalar per transition")
            if neighbor_indices.shape[0] != n_transitions:
                raise ValueError("neighbor_indices must have one row per transition")
            if neighbor_weights.shape != neighbor_indices.shape:
                raise ValueError("neighbor_weights must have the same shape as neighbor_indices")
            if np.min(neighbor_indices) < 0 or np.max(neighbor_indices) >= n_transitions:
                raise ValueError("coverage_profile neighbor_indices contain out-of-range entries")

            neighbor_count = neighbor_indices.shape[1]
            self._neighbor_indices = np.zeros((self._buffer_size, neighbor_count), dtype=np.int32)
            self._neighbor_weights = np.zeros((self._buffer_size, neighbor_count), dtype=np.float32)
            self._neighbor_indices[:n_transitions] = neighbor_indices
            self._neighbor_weights[:n_transitions] = neighbor_weights
            self._density_confidences[:n_transitions] = density_confidence
            self._action_diversities[:n_transitions] = action_diversity
            self._junction_scores[:n_transitions] = junction_score
            self._junction_gates[:n_transitions] = junction_gate
        elif coverage is not None:
            # Backward-compatible fallback for old density-confidence arrays.
            coverage = np.asarray(coverage, dtype=np.float32).reshape(-1, 1)
            if coverage.shape[0] != n_transitions:
                raise ValueError("coverage must have one scalar per transition")
            self._density_confidences[:n_transitions] = coverage

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")
        print(f"Vicinal pool size: {self._neighbor_indices.shape[1]}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        neighbor_indices = self._neighbor_indices[indices]
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
            "density_confidences": self._density_confidences[indices],
            "action_diversities": self._action_diversities[indices],
            "junction_scores": self._junction_scores[indices],
            "junction_gates": self._junction_gates[indices],
            "neighbor_indices": neighbor_indices,
            "neighbor_weights": self._neighbor_weights[indices],
            "neighbor_observations": self._states[neighbor_indices],
            "neighbor_actions": self._actions[neighbor_indices],
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
    if log_wandb and wandb.run is not None:
        wandb.save(eval_logs_path, policy="now")


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
    """Density-Certified Stitching IQL in JAX/Flax.

    Q learning remains standard IQL, while V and actor extraction can use a
    certified local neighbor pool built from the dataset coverage profile.
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
        dc_tau_min: float = 0.7,
        dc_tau_max: float = 0.9,
        dc_use_state_tau: bool = False,
        dc_use_vicinal_value: bool = True,
        dc_use_vicinal_actor: bool = True,
        dc_vicinal_lambda: float = 0.0,
        dc_vicinal_noise_std: float = 0.01,
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
        self.dc_tau_min = dc_tau_min
        self.dc_tau_max = dc_tau_max
        self.dc_use_state_tau = dc_use_state_tau
        self.dc_use_vicinal_value = dc_use_vicinal_value
        self.dc_use_vicinal_actor = dc_use_vicinal_actor
        self.dc_vicinal_lambda = dc_vicinal_lambda
        self.dc_vicinal_noise_std = dc_vicinal_noise_std
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
        dc_tau_min = self.dc_tau_min
        dc_tau_max = self.dc_tau_max
        dc_use_state_tau = self.dc_use_state_tau
        dc_use_vicinal_value = self.dc_use_vicinal_value
        dc_use_vicinal_actor = self.dc_use_vicinal_actor
        dc_vicinal_lambda = self.dc_vicinal_lambda
        dc_vicinal_noise_std = self.dc_vicinal_noise_std
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

            density_conf = jnp.squeeze(
                batch.get("density_confidences", jnp.ones_like(batch["dones"])), axis=-1
            )
            action_diversity = jnp.squeeze(
                batch.get("action_diversities", jnp.zeros_like(batch["dones"])), axis=-1
            )
            junction_scores = jnp.squeeze(
                batch.get("junction_scores", jnp.zeros_like(batch["dones"])), axis=-1
            )
            junction_gates = jnp.squeeze(
                batch.get("junction_gates", jnp.zeros_like(batch["dones"])), axis=-1
            )
            density_conf = jnp.clip(density_conf, 0.0, 1.0)
            action_diversity = jnp.clip(action_diversity, 0.0, 1.0)
            junction_scores = jnp.clip(junction_scores, 0.0, 1.0)
            junction_gates = jnp.clip(junction_gates, 0.0, 1.0)

            neighbor_observations = batch.get("neighbor_observations", observations[:, None, :])
            neighbor_actions = batch.get("neighbor_actions", actions[:, None, :])
            neighbor_weights = batch.get(
                "neighbor_weights",
                jnp.ones((observations.shape[0], 1), dtype=observations.dtype),
            )
            neighbor_weights = neighbor_weights / jnp.maximum(
                jnp.sum(neighbor_weights, axis=1, keepdims=True), 1e-8
            )
            self_only_weights = jnp.zeros_like(neighbor_weights).at[:, 0].set(1.0)
            value_pool_weights = neighbor_weights if dc_use_vicinal_value else self_only_weights
            actor_pool_weights = neighbor_weights if dc_use_vicinal_actor else self_only_weights

            batch_iql_tau = jnp.full_like(density_conf, iql_tau)
            if dc_use_state_tau:
                batch_iql_tau = dc_tau_min + (dc_tau_max - dc_tau_min) * density_conf

            # Values used by multiple losses are computed from the old state,
            # matching the sequencing of the provided PyTorch IQL implementation.
            next_v = v_apply({"params": state.v_params}, next_observations)
            target_q_for_backup = rewards + (1.0 - dones) * discount * next_v

            batch_size, pool_size = neighbor_weights.shape
            flat_neighbor_observations = neighbor_observations.reshape((-1, neighbor_observations.shape[-1]))
            flat_neighbor_actions = neighbor_actions.reshape((-1, neighbor_actions.shape[-1]))
            neighbor_q1, neighbor_q2 = q_apply(
                {"params": state.q_target_params},
                flat_neighbor_observations,
                flat_neighbor_actions,
            )
            neighbor_target_q = jnp.minimum(neighbor_q1, neighbor_q2).reshape((batch_size, pool_size))

            target_q_for_v_self = neighbor_target_q[:, 0]
            old_v = v_apply({"params": state.v_params}, observations)
            adv = target_q_for_v_self - old_v

            # DCS-IQL value update:
            # V(s_i) is regressed against a certified pool of real dataset
            # neighbor backups Q(s_j, a_j). When J(s_i) is low, preprocessing has
            # already collapsed neighbor_weights to self-only. This implements
            # explicit local stitching without pushing tau toward 1.
            v_noise_key, actor_rng_key = jax.random.split(state.actor_key)

            def v_loss_fn(v_params):
                v = v_apply({"params": v_params}, observations)
                value_adv = jax.lax.stop_gradient(neighbor_target_q) - v[:, None]
                value_weight = jnp.abs(batch_iql_tau[:, None] - (value_adv < 0.0).astype(jnp.float32))
                per_neighbor_loss = value_weight * value_adv ** 2
                expectile_loss = jnp.mean(jnp.sum(value_pool_weights * per_neighbor_loss, axis=1))

                # Deprecated ablation from the old DC-IQL version. Kept only for
                # compatibility; the new method should normally use lambda=0.
                noise = jax.random.normal(v_noise_key, observations.shape, dtype=observations.dtype)
                noise = noise * dc_vicinal_noise_std * density_conf[:, None]
                perturbed_observations = observations + noise
                v_perturbed = v_apply({"params": v_params}, perturbed_observations)
                v_anchor = jax.lax.stop_gradient(old_v)
                smoothing_loss = jnp.mean(density_conf * (v_perturbed - v_anchor) ** 2)
                value_loss = expectile_loss + dc_vicinal_lambda * smoothing_loss
                return value_loss, (v, expectile_loss, smoothing_loss)

            (value_loss, (v, expectile_loss, smoothing_loss)), v_grads = jax.value_and_grad(
                v_loss_fn, has_aux=True
            )(state.v_params)
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

            actor_adv = neighbor_target_q - old_v[:, None]
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(actor_adv)), EXP_ADV_MAX)
            actor_key, dropout_key = jax.random.split(actor_rng_key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_losses = jnp.sum((policy_out[:, None, :] - neighbor_actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    log_prob = -0.5 * (
                        ((neighbor_actions - mean[:, None, :]) / std) ** 2
                        + 2.0 * log_std
                        + jnp.log(2.0 * jnp.pi)
                    )
                    bc_losses = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                weighted_bc = actor_pool_weights * jax.lax.stop_gradient(exp_adv) * bc_losses
                actor_loss = jnp.mean(jnp.sum(weighted_bc, axis=1))
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
                "value_expectile_loss": expectile_loss,
                "value_smoothing_loss": smoothing_loss,
                "density_conf_mean": jnp.mean(density_conf),
                "density_conf_min": jnp.min(density_conf),
                "density_conf_max": jnp.max(density_conf),
                "action_diversity_mean": jnp.mean(action_diversity),
                "junction_score_mean": jnp.mean(junction_scores),
                "junction_gate_mean": jnp.mean(junction_gates),
                "vicinal_pool_size": jnp.asarray(pool_size, dtype=jnp.float32),
                "vicinal_effective_pool_size": jnp.mean(1.0 / jnp.maximum(jnp.sum(neighbor_weights ** 2, axis=1), 1e-8)),
                "iql_tau_mean": jnp.mean(batch_iql_tau),
                "iql_tau_min": jnp.min(batch_iql_tau),
                "iql_tau_max": jnp.max(batch_iql_tau),
                "v_mean": jnp.mean(v),
                "adv_mean": jnp.mean(adv),
                "adv_min": jnp.min(adv),
                "adv_max": jnp.max(adv),
                "exp_adv_mean": jnp.mean(jnp.sum(actor_pool_weights * exp_adv, axis=1)),
                "actor_loss": actor_loss,
                "bc_loss_mean": jnp.mean(jnp.sum(actor_pool_weights * bc_losses, axis=1)),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
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
        dc_use_vicinal_actor = self.dc_use_vicinal_actor
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
            neighbor_observations = batch.get("neighbor_observations", observations[:, None, :])
            neighbor_actions = batch.get("neighbor_actions", actions[:, None, :])
            neighbor_weights = batch.get(
                "neighbor_weights",
                jnp.ones((observations.shape[0], 1), dtype=observations.dtype),
            )
            neighbor_weights = neighbor_weights / jnp.maximum(
                jnp.sum(neighbor_weights, axis=1, keepdims=True), 1e-8
            )
            self_only_weights = jnp.zeros_like(neighbor_weights).at[:, 0].set(1.0)
            actor_pool_weights = neighbor_weights if dc_use_vicinal_actor else self_only_weights

            batch_size, pool_size = neighbor_weights.shape
            flat_neighbor_observations = neighbor_observations.reshape((-1, neighbor_observations.shape[-1]))
            flat_neighbor_actions = neighbor_actions.reshape((-1, neighbor_actions.shape[-1]))
            q1, q2 = q_apply(
                {"params": iql_state.q_target_params},
                flat_neighbor_observations,
                flat_neighbor_actions,
            )
            target_q = jnp.minimum(q1, q2).reshape((batch_size, pool_size))
            v = v_apply({"params": iql_state.v_params}, observations)
            adv = target_q - v[:, None]
            exp_adv = jnp.minimum(jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_losses = jnp.sum((policy_out[:, None, :] - neighbor_actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    log_prob = -0.5 * (
                        ((neighbor_actions - mean[:, None, :]) / std) ** 2
                        + 2.0 * log_std
                        + jnp.log(2.0 * jnp.pi)
                    )
                    bc_losses = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                actor_loss = jnp.mean(
                    jnp.sum(actor_pool_weights * jax.lax.stop_gradient(exp_adv) * bc_losses, axis=1)
                )
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
                "bc_loss": jnp.mean(jnp.sum(actor_pool_weights * bc_losses, axis=1)),
                "adv_mean": jnp.mean(jnp.sum(actor_pool_weights * adv, axis=1)),
                "adv_min": jnp.min(adv),
                "adv_max": jnp.max(adv),
                "exp_adv_mean": jnp.mean(jnp.sum(actor_pool_weights * exp_adv, axis=1)),
                "exp_adv_max": jnp.max(exp_adv),
                "target_q_mean": jnp.mean(jnp.sum(actor_pool_weights * target_q, axis=1)),
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


@pyrallis.wrap()
def train(config: TrainConfig):
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

    coverage_profile = None
    if config.use_density_calibration:
        coverage_profile = load_or_compute_coverage_profile(
            config=config,
            observations=dataset["observations"],
            actions=dataset["actions"],
            next_observations=dataset["next_observations"],
        )
        print(
            f"Coverage profile: "
            f"rho_mean={float(np.mean(coverage_profile['density_confidence'])):.3f}, "
            f"rho_min={float(np.min(coverage_profile['density_confidence'])):.3f}, "
            f"rho_max={float(np.max(coverage_profile['density_confidence'])):.3f}; "
            f"b_mean={float(np.mean(coverage_profile['action_diversity'])):.3f}; "
            f"J_mean={float(np.mean(coverage_profile['junction_score'])):.3f}, "
            f"J_gate_rate={float(np.mean(coverage_profile['junction_gate'])):.3f}; "
            f"pool_size={coverage_profile['neighbor_indices'].shape[1]}"
        )
        if config.dc_use_state_tau:
            tau_values = config.dc_tau_min + (config.dc_tau_max - config.dc_tau_min) * coverage_profile["density_confidence"]
            print(
                "Legacy state-wise tau ablation is ON: "
                f"tau(s) mean={float(np.mean(tau_values)):.3f}, "
                f"min={float(np.min(tau_values)):.3f}, max={float(np.max(tau_values)):.3f}"
            )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=jax_device,
    )
    replay_buffer.load_d4rl_dataset(dataset, coverage_profile=coverage_profile)

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
        dc_tau_min=config.dc_tau_min if config.use_density_calibration else config.iql_tau,
        dc_tau_max=config.dc_tau_max if config.use_density_calibration else config.iql_tau,
        dc_use_state_tau=config.dc_use_state_tau if config.use_density_calibration else False,
        dc_use_vicinal_value=config.dc_use_vicinal_value if config.use_density_calibration else False,
        dc_use_vicinal_actor=config.dc_use_vicinal_actor if config.use_density_calibration else False,
        dc_vicinal_lambda=config.dc_vicinal_lambda if config.use_density_calibration else 0.0,
        dc_vicinal_noise_std=config.dc_vicinal_noise_std,
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

    if config.log_wandb:
        wandb_init(asdict(config))

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

    eval_logs: List[Dict[str, Any]] = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)

        if config.log_wandb and (t + 1) % config.log_every == 0:
            wandb.log(log_dict, step=int(jax.device_get(trainer.state.total_it)))

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
                wandb.log(wandb_eval_log, step=int(jax.device_get(trainer.state.total_it)))

            save_and_upload_eval_logs(
                eval_logs=eval_logs,
                checkpoints_path=config.checkpoints_path,
                log_wandb=config.log_wandb,
            )

    if config.checkpoints_path is not None:
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
        save_pickle(checkpoint_path, trainer.state_dict())

        # if config.log_wandb and wandb.run is not None:
        #     wandb.save(checkpoint_path, policy="now")

        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()

