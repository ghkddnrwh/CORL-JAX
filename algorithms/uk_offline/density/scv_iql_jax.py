# dcs_iql_jax.py
#
# DCS-IQL: Density-Certified Stitching IQL (JAX/Flax).
#
# Successor of DC-IQL. The diagnosis of DC-IQL (see the accompanying proposal
# document) showed that (i) the per-state expectile tau(s) driven by state
# density alone diverges for tau_max >= 0.95 and is inert at 0.9, and (ii) the
# correct local signal for "how aggressive may the in-sample max be at s" is
# the *diversity of actions/continuations* among neighbors, not state density.
#
# DCS-IQL therefore changes the control lever:
#   - tau is GLOBAL and fixed (0.7-0.9, as in IQL). No per-state tau.
#   - Aggressiveness is instead controlled by the COMPOSITION of the
#     distribution the expectile looks at: V(s_i) is regressed against the
#     pooled set { Q_target(s_j, a_j) : j in {i} u N_k(s_i) } of *in-sample*
#     neighbor state-action values ("Vicinal Bellman Backup"), weighted by
#       w_ij = gate(J(s_i)) * exp(-(d_ij / h_i)^2),
#     where J(s) = density(s) x action/continuation-diversity(s) is the
#     precomputed junction score. gate -> 0 recovers exact IQL (SARSA-ish);
#     gate -> 1 performs an explicit, certified cross-trajectory stitch.
#   - The same pool drives policy extraction ("Vicinal AWR"): pi(.|s_i) is
#     trained on neighbor actions a_j weighted by exp(beta * A_ij) * w_ij,
#     attacking the policy-extraction bottleneck directly.
#   - Only in-sample pairs (s_j, a_j) are ever evaluated by Q: the in-sample
#     property of IQL (no OOD action queries) is preserved exactly.
#
# All coverage statistics (kNN graph, density, diversity) are one-time
# nonparametric precomputation handled by coverage_profile.py and cached with
# metadata validation, mirroring the DC-IQL cache design. Percentile scaling,
# junction mode, gate percentiles and kernel bandwidth are post-processing
# applied at load time, so tuning them does not invalidate caches.
#
# Plumbing (modes, hyperparameter merging, checkpointing, refit, logging)
# follows the original DC-IQL/CDAF_JAX file style:
#   - mode="train":  joint Q/V/pi training.
#   - mode="refit":  load Q/V from a checkpoint, train only pi
#                    (max_timesteps/batch_size/eval_freq are reused as the
#                    refit schedule). DC-IQL checkpoints are loadable here
#                    since the network architectures are unchanged.

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

import coverage_profile as covp
import temporal_metric as tmet

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
    upsert_eval_log,
)

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "SCV-IQL"
ALGORITHM_FULL_NAME = "Support-Certified Value Implicit Q-Learning"

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
    hyperparams_path: Optional[str] = "hyperparams/dcs_iql_jax.yml"
    use_hyperparams: bool = True

    # Dataset
    buffer_size: int = 2_000_000

    # Shared by both modes:
    #   mode="train": minibatch size for joint Q/V/pi updates.
    #   mode="refit": minibatch size for actor-only refit updates.
    batch_size: int = 256

    normalize: bool = True
    normalize_reward: bool = False

    # IQL core. iql_tau is GLOBAL and fixed by design (see header).
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = False

    # ----- SCV-IQL: Support-Certified Value learning ----------------------
    # No neighbor Q pooling is used. Neighbor statistics only define c(s), a
    # support certificate that controls the expectile asymmetry in V-learning.
    scv_tau_min: float = 0.5
    scv_tau_max: float = 0.8
    scv_support_power: float = 1.0
    scv_support_eps: float = 1e-8

    # ----- DCS: Density-Certified Stitching -------------------------------
    # Master switch. False -> neighbor weights are zeroed and DCS-IQL reduces
    # exactly to standard IQL (pool collapses onto the sample itself).
    use_dcs: bool = True

    # kNN graph: number of neighbors kept per state (self excluded).
    dcs_k: int = 8

    # Relative mass of the neighbor pool vs the sample itself, separately for
    # the value expectile (Vicinal Bellman Backup) and policy extraction
    # (Vicinal AWR). 0.0 disables that side; with both at 0.0 -> exact IQL.
    dcs_value_neighbor_weight: float = 1.0
    dcs_actor_neighbor_weight: float = 1.0

    # Junction score J(s) = density_conf x diversity_conf.
    #   diversity mode: "action" (spread of neighbor actions),
    #                   "displacement" (directional dispersion of neighbor
    #                                   next-state displacements),
    #                   "product" (geometric mean of both).
    dcs_diversity_mode: str = "product"

    # Percentile scaling of raw kNN statistics into [0, 1] confidences.
    dcs_percentile_low: float = 5.0
    dcs_percentile_high: float = 95.0

    # Gate(s): 0 below the low percentile of J, 1 above the high percentile.
    # Pooling is only active on the gated (junction-like) part of the data.
    dcs_gate_low_percentile: float = 60.0
    dcs_gate_high_percentile: float = 95.0

    # Distance kernel exp(-(d/h)^2) with per-state bandwidth
    # h_i = dcs_bandwidth_scale * median_j d_ij.
    dcs_bandwidth_scale: float = 1.0

    # Bandwidth floor: h_i is floored at
    #   dcs_bandwidth_scale * dcs_bandwidth_floor_frac * (global median of
    #   POSITIVE neighbor distances).
    # This stops the kernel from saturating to all-or-nothing when neighbors
    # are (near-)duplicates -- the failure mode of a DISCRETE metric space such
    # as the oracle button_states graph, where same-button neighbors sit at
    # distance ~0. Necessary hygiene for continuous/mixed spaces; it does NOT
    # by itself tame a fully-degenerate (all-exact-match) neighborhood -- that
    # is controlled by the neighbor-mass weights below. Set 0.0 for the old
    # (unfloored) behavior.
    dcs_bandwidth_floor_frac: float = 1.0

    # ----- Per-edge dynamics-consistency gate g_ij -------------------------
    # Certifies that neighbor j's transition actually shares s_i's local
    # action->displacement dynamics before its (s_j, a_j) is pooled into the
    # vicinal backup / vicinal AWR. This is the certification that upgrades the
    # method from "density-and-diversity gated" to genuinely dynamics-certified
    # (addresses the cross-state-action contamination of the actor loss).
    #   final edge weight = gate(J(s_i)) * kernel(d_ij) * g_ij
    # g_ij = exp(-(r_ij / dcs_dynamics_scale)^2), r_ij = displacement-normalized
    # residual of neighbor j under the local linear dynamics fit.
    #
    # dcs_use_dynamics_gate=False recovers the previous density-and-diversity
    # gated behavior (g == 1) and is the clean ablation for measuring g's effect.
    # Larger dcs_dynamics_scale -> gentler gate (so it cannot silently collapse
    # the pool back to plain IQL; watch active_edge_frac / pool mass in logs).
    dcs_use_dynamics_gate: bool = True
    dcs_dynamics_scale: float = 1.0
    dcs_dynamics_ridge: float = 1e-3

    # kNN computation controls (mirroring the DC-IQL density cache).
    dcs_subsample: int = 10_000_000
    dcs_chunk_size: int = 50_000

    # Coverage profile cache (raw kNN statistics only; post-processing knobs
    # above never invalidate it). Saved as:
    #   {dcs_profile_path}/{env}/[seed_{seed}/]coverage_profile_k{dcs_k}.npz
    dcs_profile_path: Optional[str] = None
    dcs_force_recompute: bool = False
    dcs_cache_by_seed: bool = False

    # ----- METRIC SOURCE: which space the kNN graph is built in -----------
    # Selects the metric space in which the kNN graph is built:
    #   "observation" : default; neighbors from normalized observations (L2).
    #   "oracle"      : neighbors from ground-truth dataset field(s) (e.g.
    #                   puzzle button_states), loaded via add_info=True. This is
    #                   the privileged CEILING TEST -- not deployable.
    #   "temporal"    : neighbors from a LEARNED temporal representation phi(s)
    #                   trained on the dataset's (s, s') transitions (see
    #                   temporal_metric.py). Data-only and deployable: this is
    #                   the approximation of the oracle graph. Training still
    #                   uses standard observations; only neighbor selection uses
    #                   phi(s).
    dcs_metric_source: str = "observation"  # observation | oracle | temporal
    # Dataset add_info field(s) used as the oracle metric space. Auto-defaults:
    # puzzle -> button_states; cube/antmaze -> qpos,qvel. Comma-separated.
    dcs_oracle_keys: Optional[str] = None
    # Standardize the oracle space per-dim before building the kNN graph so no
    # single field (e.g. a large-magnitude qvel) dominates the distance.
    dcs_oracle_standardize: bool = True

    # ----- Temporal metric (dcs_metric_source="temporal") -----------------
    # Contrastive encoder phi(s) hyperparameters. Embeddings are cached (keyed
    # by these + env + seed) so the encoder is trained once. phi is L2-
    # normalized onto the unit sphere by default, so Euclidean kNN ranks by
    # cosine similarity; leave dcs_temporal_standardize False in that case.
    dcs_temporal_dim: int = 32
    dcs_temporal_hidden: int = 256
    dcs_temporal_layers: int = 2
    dcs_temporal_steps: int = 50_000
    dcs_temporal_batch: int = 512
    dcs_temporal_lr: float = 3e-4
    dcs_temporal_temperature: float = 0.1
    # Positive pairs are k-step successors with k in [1, horizon]. horizon=1
    # uses next_observations directly (robust, no trajectory bookkeeping);
    # horizon>1 samples within trajectories (needs terminals/masks or is
    # detected from observation continuity).
    dcs_temporal_horizon: int = 1
    dcs_temporal_normalize: bool = True
    dcs_temporal_standardize: bool = False
    dcs_temporal_force_recompute: bool = False
    # -----------------------------------------------------------------------

    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = None
    hidden_dim: int = 256
    n_hidden: int = 2

    # Standalone actor refit output directory.
    actor_refit_dir_name: str = "actor_refit"

    # Logging
    project: str = "ORL-BIAS"
    group: str = "DCS-IQL-JAX"
    name: str = "DCS-IQL-JAX"
    log_wandb: bool = True
    log_every: int = 500
    save_final_model: bool = False

    checkpoint_freq: int = int(25e3)
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
    assert 0.0 <= config.scv_tau_min <= config.scv_tau_max <= 1.0
    assert config.scv_support_power > 0.0
    assert config.scv_support_eps > 0.0
    assert config.dcs_k >= 1
    assert config.dcs_value_neighbor_weight >= 0.0
    assert config.dcs_actor_neighbor_weight >= 0.0
    assert config.dcs_diversity_mode in ("action", "displacement", "product")
    assert 0.0 <= config.dcs_percentile_low < config.dcs_percentile_high <= 100.0
    assert 0.0 <= config.dcs_gate_low_percentile <= config.dcs_gate_high_percentile <= 100.0
    assert config.dcs_bandwidth_scale > 0.0
    assert config.dcs_bandwidth_floor_frac >= 0.0
    assert config.dcs_dynamics_scale > 0.0
    assert config.dcs_dynamics_ridge >= 0.0
    assert config.dcs_metric_source in ("observation", "oracle", "temporal"), \
        "dcs_metric_source must be 'observation', 'oracle', or 'temporal'"
    if config.dcs_metric_source == "temporal":
        assert config.dcs_temporal_dim >= 1
        assert config.dcs_temporal_hidden >= 1
        assert config.dcs_temporal_layers >= 1
        assert config.dcs_temporal_steps >= 0
        assert config.dcs_temporal_batch >= 2, "InfoNCE needs batch_size >= 2"
        assert config.dcs_temporal_lr > 0.0
        assert config.dcs_temporal_temperature > 0.0
        assert config.dcs_temporal_horizon >= 1
    assert config.dcs_subsample >= 1
    assert config.dcs_chunk_size >= 1
    if config.dcs_profile_path is not None:
        assert config.dcs_profile_path != ""
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

    Hyperparameter YAML keys must exactly match TrainConfig field names
    (a few legacy DC-IQL keys are aliased for convenience).
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
        # Legacy DC-IQL keys that map cleanly onto DCS-IQL fields.
        "dc_density_model_path": "dcs_profile_path",
        "dc_density_k": "dcs_k",
        "dc_density_subsample": "dcs_subsample",
        "dc_density_chunk_size": "dcs_chunk_size",
        "dc_density_percentile_low": "dcs_percentile_low",
        "dc_density_percentile_high": "dcs_percentile_high",
        "dc_density_force_recompute": "dcs_force_recompute",
        "dc_density_cache_by_seed": "dcs_cache_by_seed",
    }
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


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: Union[np.ndarray, float], std: Union[np.ndarray, float]):
    return (states - mean) / std


# ---------------------------------------------------------------------------
# Coverage profile cache plumbing (mirrors the DC-IQL density cache design)
# ---------------------------------------------------------------------------

def _safe_path_name(name: str) -> str:
    """Make env names safe for directory paths."""
    return name.replace("/", "_").replace(":", "_")


def get_coverage_profile_cache_path(config: TrainConfig) -> Optional[Path]:
    """Return the cache file path for the precomputed coverage profile.

    The metric source is part of the path so that an oracle-graph profile and
    an observation-graph profile for the same env never collide.
    """
    if config.dcs_profile_path is None:
        return None

    root = Path(config.dcs_profile_path)
    env_name = _safe_path_name(config.env)
    source_tag = "obs" if config.dcs_metric_source == "observation" else "oracle"
    if config.dcs_metric_source == "oracle":
        keys = resolve_oracle_keys(config) or ("auto",)
        source_tag = "oracle-" + "_".join(keys)
    elif config.dcs_metric_source == "temporal":
        source_tag = "temporal-" + tmet.signature_hash(build_temporal_signature(config))
    filename = f"coverage_profile_{_safe_path_name(source_tag)}_k{int(config.dcs_k)}.npz"

    if config.dcs_cache_by_seed:
        return root / env_name / f"seed_{config.seed}" / filename
    return root / env_name / filename


def resolve_oracle_keys(config: TrainConfig) -> Optional[Tuple[str, ...]]:
    """Resolve the oracle dataset field name(s) for the oracle metric space.

    Explicit --dcs_oracle_keys wins; otherwise auto-default by env family
    (puzzle -> button_states; cube/antmaze -> qpos,qvel), mirroring the
    metric_diagnostic.py oracle selection.
    """
    if config.dcs_oracle_keys:
        return tuple(x.strip() for x in config.dcs_oracle_keys.split(",") if x.strip())
    env = config.env
    if env.startswith("puzzle-"):
        return ("button_states",)
    if env.startswith(("cube-", "antmaze-")):
        return ("qpos", "qvel")
    return None


def build_oracle_metric_space(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
) -> np.ndarray:
    """Concatenate the requested oracle field(s) into an (N, Dm) metric space,
    standardized per-dim by default so no single field dominates the distance.
    """
    keys = resolve_oracle_keys(config)
    if not keys:
        raise SystemExit(
            f"dcs_metric_source='oracle' but no oracle keys could be resolved for "
            f"env '{config.env}'. Pass --dcs_oracle_keys, e.g. 'button_states'."
        )
    missing = [key for key in keys if key not in dataset]
    if missing:
        available = ", ".join(sorted(dataset.keys()))
        raise SystemExit(
            f"Oracle key(s) {missing} not in dataset (load uses add_info=True). "
            f"Available keys: {available}"
        )
    parts = []
    for key in keys:
        arr = np.asarray(dataset[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        parts.append(arr)
    oracle = np.concatenate(parts, axis=1).astype(np.float32)

    if config.dcs_oracle_standardize:
        mu = oracle.mean(0)
        sd = oracle.std(0) + 1e-6
        oracle = ((oracle - mu) / sd).astype(np.float32)

    print(
        f"Oracle metric space: keys={list(keys)} dim={oracle.shape[1]} "
        f"standardize={config.dcs_oracle_standardize}"
    )
    return oracle


def build_temporal_signature(
    config: TrainConfig,
    observations_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Encoder-defining signature; any change invalidates cached embeddings and
    the coverage graph built from them."""
    return {
        "cache_version": tmet.TEMPORAL_CACHE_VERSION,
        "env": config.env,
        "seed": int(config.seed),
        "normalize": bool(config.dcs_temporal_normalize),
        "observations_shape": tuple(observations_shape) if observations_shape else None,
        "embed_dim": int(config.dcs_temporal_dim),
        "hidden_dim": int(config.dcs_temporal_hidden),
        "n_hidden": int(config.dcs_temporal_layers),
        "steps": int(config.dcs_temporal_steps),
        "batch_size": int(config.dcs_temporal_batch),
        "lr": float(config.dcs_temporal_lr),
        "temperature": float(config.dcs_temporal_temperature),
        "horizon": int(config.dcs_temporal_horizon),
    }


def get_temporal_embeddings_cache_path(config: TrainConfig) -> Optional[Path]:
    if config.dcs_profile_path is None:
        return None
    root = Path(config.dcs_profile_path)
    env_name = _safe_path_name(config.env)
    sig_hash = tmet.signature_hash(build_temporal_signature(config))
    filename = f"temporal_emb_{sig_hash}.npz"
    if config.dcs_cache_by_seed:
        return root / env_name / f"seed_{config.seed}" / filename
    return root / env_name / filename


def build_temporal_metric_space(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> np.ndarray:
    """Return phi(observations) as the (N, dim) temporal metric space, training
    (or loading cached) the contrastive encoder on the dataset's (s, s')
    transitions. Data-only: uses observations/next_observations, never any
    privileged field."""
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    if "next_observations" not in dataset:
        raise SystemExit(
            "dcs_metric_source='temporal' requires next_observations in the dataset."
        )
    next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)

    signature = build_temporal_signature(config, observations_shape=observations.shape)
    cache_path = get_temporal_embeddings_cache_path(config)

    embeddings = None
    if cache_path is not None and not config.dcs_temporal_force_recompute:
        embeddings = tmet.load_temporal_embeddings_cache(cache_path, signature)

    if embeddings is None:
        next_index = None
        if config.dcs_temporal_horizon > 1:
            next_index = tmet.build_successor_index(dataset)
        embeddings = tmet.train_temporal_encoder(
            observations,
            next_observations,
            embed_dim=config.dcs_temporal_dim,
            hidden_dim=config.dcs_temporal_hidden,
            n_hidden=config.dcs_temporal_layers,
            steps=config.dcs_temporal_steps,
            batch_size=config.dcs_temporal_batch,
            lr=config.dcs_temporal_lr,
            temperature=config.dcs_temporal_temperature,
            horizon=config.dcs_temporal_horizon,
            normalize=config.dcs_temporal_normalize,
            seed=config.seed,
            next_index=next_index,
            device=device,
        )
        if cache_path is not None:
            tmet.save_temporal_embeddings_cache(cache_path, embeddings, signature)

    if config.dcs_temporal_standardize:
        # Only meaningful for UNnormalized embeddings; per-dim standardizing a
        # unit-sphere embedding distorts the cosine geometry.
        embeddings = tmet.standardize_embeddings(embeddings)

    print(
        f"Temporal metric space: dim={embeddings.shape[1]} "
        f"normalize={config.dcs_temporal_normalize} "
        f"standardize={config.dcs_temporal_standardize} "
        f"horizon={config.dcs_temporal_horizon}"
    )
    return embeddings.astype(np.float32)


def build_coverage_profile_metadata(
    config: TrainConfig,
    observations: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, Any]:
    """Metadata that must match before a cached profile can be reused.

    Post-processing knobs (percentiles, diversity mode, gate, bandwidth) are
    intentionally excluded: they are applied at load time. The metric source
    and oracle keys ARE included, since they change neighbor selection.
    """
    oracle_keys = resolve_oracle_keys(config) if config.dcs_metric_source == "oracle" else None
    temporal_sig = (
        build_temporal_signature(config, observations_shape=observations.shape)
        if config.dcs_metric_source == "temporal" else None
    )
    return covp.canonicalize_metadata(
        {
            "cache_version": covp.COVERAGE_CACHE_VERSION,
            "env": config.env,
            "seed": int(config.seed) if config.dcs_cache_by_seed else None,
            "normalize": bool(config.normalize),
            "observations_shape": tuple(observations.shape),
            "observations_dtype": str(np.asarray(observations).dtype),
            "actions_shape": tuple(actions.shape),
            "dcs_k": int(config.dcs_k),
            "dcs_subsample": int(config.dcs_subsample),
            "dcs_chunk_size": int(config.dcs_chunk_size),
            "dcs_dynamics_ridge": float(config.dcs_dynamics_ridge),
            "dcs_metric_source": config.dcs_metric_source,
            "dcs_oracle_keys": list(oracle_keys) if oracle_keys else None,
            "dcs_oracle_standardize": bool(config.dcs_oracle_standardize)
            if config.dcs_metric_source == "oracle" else None,
            "dcs_temporal_signature": tmet.signature_hash(temporal_sig)
            if temporal_sig is not None else None,
        }
    )


def load_or_compute_coverage_profile(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> Dict[str, np.ndarray]:
    """Load the cached coverage profile, or compute and save it when the cache
    is missing/stale. Observations are expected to be already normalized."""
    cache_path = get_coverage_profile_cache_path(config)
    metadata = build_coverage_profile_metadata(
        config, dataset["observations"], dataset["actions"]
    )

    if cache_path is not None and not config.dcs_force_recompute:
        cached = covp.load_coverage_profile_cache(cache_path, metadata)
        if cached is not None:
            return cached

    if cache_path is not None and config.dcs_force_recompute:
        print(f"Ignoring existing coverage cache because dcs_force_recompute=True: {cache_path}")

    # Build the kNN graph in the requested metric space.
    metric_space = None
    if config.dcs_metric_source == "oracle":
        metric_space = build_oracle_metric_space(config, dataset)
        print(f"Computing oracle-graph coverage profile for {ALGORITHM_NAME} "
              f"(k={config.dcs_k}); neighbors selected in oracle space, training "
              f"still uses standard observations.")
    elif config.dcs_metric_source == "temporal":
        metric_space = build_temporal_metric_space(config, dataset, device=device)
        print(f"Computing temporal-graph coverage profile for {ALGORITHM_NAME} "
              f"(k={config.dcs_k}); neighbors selected in learned temporal space, "
              f"training still uses standard observations.")
    else:
        print(f"Computing one-time coverage profile for {ALGORITHM_NAME} (k={config.dcs_k})...")

    profile = covp.compute_coverage_profile(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        k=config.dcs_k,
        subsample_size=config.dcs_subsample,
        chunk_size=config.dcs_chunk_size,
        seed=config.seed,
        dynamics_ridge=config.dcs_dynamics_ridge,
        metric_space=metric_space,
    )

    if cache_path is not None:
        covp.save_coverage_profile_cache(cache_path, profile, metadata)

    return profile


def build_neighbor_weights(
    config: TrainConfig,
    profile: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-process raw kNN statistics into per-transition training inputs.

    Returns:
        neighbor_indices (N, k) int32
        neighbor_weights (N, k) float32
            = gate(J(s_i)) * exp(-(d_ij/h_i)^2) * g_ij
              where g_ij is the per-edge dynamics-consistency gate (or 1 when
              dcs_use_dynamics_gate is False).
        junction         (N,)   float32 (for logging/diagnostics)
    """
    density_conf = covp.percentile_confidence(
        profile["knn_radius"],
        config.dcs_percentile_low,
        config.dcs_percentile_high,
        invert=True,
    )
    action_div_conf = covp.percentile_confidence(
        profile["action_spread"],
        config.dcs_percentile_low,
        config.dcs_percentile_high,
    )
    disp_div_conf = covp.percentile_confidence(
        profile["disp_dispersion"],
        config.dcs_percentile_low,
        config.dcs_percentile_high,
    )
    junction = covp.build_junction_score(
        density_conf, action_div_conf, disp_div_conf, config.dcs_diversity_mode
    )
    gate = covp.gate_from_junction(
        junction, config.dcs_gate_low_percentile, config.dcs_gate_high_percentile
    )
    kernel = covp.neighbor_kernel_weights(
        profile["neighbor_distances"], config.dcs_bandwidth_scale,
        config.dcs_bandwidth_floor_frac,
    )

    # Kernel-saturation diagnostic: what fraction of neighbor edges sit at
    # (near-)zero distance? A high value flags a degenerate/discrete metric
    # space where the graph cannot grade neighbors by distance (e.g. the oracle
    # button_states graph). The bandwidth floor keeps mixed neighborhoods
    # graded, but a high exact-match fraction means most pooling is uniform and
    # its influence should be controlled via eta_V / eta_pi, not the kernel.
    dists = profile["neighbor_distances"]
    zero_edge_frac = float(np.mean(dists <= 1e-6))
    zero_median_row_frac = float(np.mean(np.median(dists, axis=1) <= 1e-6))
    print(
        f"Kernel | zero-distance edges={zero_edge_frac:.3f} | "
        f"rows with zero median (kernel would saturate w/o floor)="
        f"{zero_median_row_frac:.3f} | bandwidth_floor_frac="
        f"{config.dcs_bandwidth_floor_frac}"
    )
    if zero_median_row_frac > 0.5:
        print(
            "NOTE: >50% of states have a (near-)duplicate-dominated neighborhood "
            "in this metric space. This is the discrete-oracle regime: the kernel "
            "cannot discriminate these neighbors, so pooling is largely uniform. "
            "Prefer eta_V=0 (actor-only) to avoid destabilizing the value backup."
        )

    weights_before_g = (gate[:, None] * kernel).astype(np.float32)

    dynamics_g = None
    if config.dcs_use_dynamics_gate and "dynamics_residual" in profile:
        dynamics_g = covp.dynamics_consistency_weights(
            profile["dynamics_residual"], config.dcs_dynamics_scale
        )
        neighbor_weights = (weights_before_g * dynamics_g).astype(np.float32)
    else:
        if config.dcs_use_dynamics_gate:
            print(
                "dcs_use_dynamics_gate=True but the cached profile has no "
                "dynamics_residual (old cache?); proceeding without the dynamics "
                "gate. Recompute the profile with --dcs_force_recompute True."
            )
        neighbor_weights = weights_before_g

    print(
        "Coverage profile | "
        + covp.summarize_profile(
            density_conf, action_div_conf, disp_div_conf, junction, gate, dynamics_g
        )
    )

    # Collapse monitor: how much edge mass survives each gating stage. If the
    # post-g active-edge fraction is ~0, DCS has silently reverted to IQL and
    # the dynamics gate is too aggressive (raise dcs_dynamics_scale).
    active_pre = float(np.mean(weights_before_g > 1e-6))
    active_post = float(np.mean(neighbor_weights > 1e-6))
    mass_pre = float(np.mean(np.sum(weights_before_g, axis=1)))
    mass_post = float(np.mean(np.sum(neighbor_weights, axis=1)))
    print(
        f"Edge gating | active-edge frac: gate*kernel={active_pre:.3f} "
        f"-> *g={active_post:.3f} | mean per-state neighbor mass: "
        f"{mass_pre:.3f} -> {mass_post:.3f}"
        + (
            f" | dynamics gate kept {(active_post / active_pre * 100.0):.1f}% of active edges"
            if active_pre > 0
            else ""
        )
    )
    if config.dcs_use_dynamics_gate and active_pre > 0 and active_post < 0.02 * active_pre:
        print(
            "WARNING: the dynamics gate removed >98% of active edges. DCS is "
            "effectively reduced to IQL here; consider raising dcs_dynamics_scale."
        )

    return (
        profile["neighbor_indices"].astype(np.int32),
        neighbor_weights,
        junction.astype(np.float32),
    )


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


def load_env_and_dataset(env_name: str, add_info: bool = False) -> Tuple[gym.Env, Dict[str, np.ndarray], str]:
    if is_ogbench_env(env_name):
        if ogbench is None:
            raise ImportError(
                "OGBench environment requested, but the `ogbench` package is not installed."
            )
        if add_info:
            try:
                # add_info=True exposes factored oracle fields such as
                # button_states (puzzle), qpos, and qvel for the ceiling test.
                env, train_dataset, _ = ogbench.make_env_and_datasets(env_name, add_info=True)
            except TypeError:
                print("ogbench.make_env_and_datasets(..., add_info=True) unavailable; "
                      "falling back to add_info=False (oracle metric source will fail).")
                env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
        else:
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
    """Replay buffer extended with the precomputed kNN graph.

    For every stored transition i it additionally holds:
      - neighbor_indices[i]: (k,) int32 indices of i's nearest dataset states
      - neighbor_weights[i]: (k,) float32 gate(J(s_i)) * kernel(d_ij) in [0, 1]
      - junction[i]:         (1,) float32 J(s_i) for logging

    sample() gathers the neighbors' (s_j, a_j) so the trainer can evaluate
    in-sample target-Q values on the whole pool in one batched forward pass.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        neighbor_k: int,
        device: Any,
    ):
        self._buffer_size = buffer_size
        self._neighbor_k = int(neighbor_k)
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self._neighbor_indices = np.zeros((buffer_size, self._neighbor_k), dtype=np.int32)
        self._neighbor_weights = np.zeros((buffer_size, self._neighbor_k), dtype=np.float32)
        self._junction = np.zeros((buffer_size, 1), dtype=np.float32)
        self._device = device

    def load_d4rl_dataset(
        self,
        data: Dict[str, np.ndarray],
        neighbor_indices: Optional[np.ndarray] = None,
        neighbor_weights: Optional[np.ndarray] = None,
        junction: Optional[np.ndarray] = None,
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

        if neighbor_indices is None:
            # DCS disabled: pool collapses onto the sample itself (weights 0),
            # which makes the trainer reduce exactly to standard IQL.
            neighbor_indices = np.tile(
                np.arange(n_transitions, dtype=np.int32)[:, None], (1, self._neighbor_k)
            )
            neighbor_weights = np.zeros((n_transitions, self._neighbor_k), dtype=np.float32)
            junction = np.zeros((n_transitions,), dtype=np.float32)

        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
        neighbor_weights = np.asarray(neighbor_weights, dtype=np.float32)
        junction = np.asarray(junction, dtype=np.float32).reshape(-1)
        if neighbor_indices.shape != (n_transitions, self._neighbor_k):
            raise ValueError(
                f"neighbor_indices must have shape {(n_transitions, self._neighbor_k)}, "
                f"got {neighbor_indices.shape}"
            )
        if neighbor_weights.shape != (n_transitions, self._neighbor_k):
            raise ValueError(
                f"neighbor_weights must have shape {(n_transitions, self._neighbor_k)}, "
                f"got {neighbor_weights.shape}"
            )
        if junction.shape[0] != n_transitions:
            raise ValueError("junction must have one scalar per transition")
        if neighbor_indices.min() < 0 or neighbor_indices.max() >= n_transitions:
            raise ValueError("neighbor_indices out of range for the loaded dataset")

        self._neighbor_indices[:n_transitions] = neighbor_indices
        self._neighbor_weights[:n_transitions] = np.clip(neighbor_weights, 0.0, 1.0)
        self._junction[:n_transitions] = junction[:, None]

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions} (kNN pool k={self._neighbor_k})")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        nbr_idx = self._neighbor_indices[indices]  # (B, k)
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
            "neighbor_observations": self._states[nbr_idx],   # (B, k, Ds)
            "neighbor_actions": self._actions[nbr_idx],       # (B, k, Da)
            "neighbor_weights": self._neighbor_weights[indices],  # (B, k)
            "junction": self._junction[indices],              # (B, 1)
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
    # Preserves the reward preprocessing from the original IQL code.
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


class SupportCertifiedIQLJAX:
    """Density-Certified Stitching IQL in JAX/Flax.

    Q-function training and the overall update sequencing mirror IQL. The two
    departures are the Vicinal Bellman Backup in the value loss and Vicinal
    AWR in policy extraction (see the module docstring). With all neighbor
    weights at zero both reduce exactly to standard IQL.
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
        scv_tau_min: float = 0.5,
        scv_tau_max: float = 0.8,
        scv_support_power: float = 1.0,
        scv_support_eps: float = 1e-8,
        dcs_value_neighbor_weight: float = 1.0,
        dcs_actor_neighbor_weight: float = 1.0,
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
        self.dcs_value_neighbor_weight = float(dcs_value_neighbor_weight)
        self.dcs_actor_neighbor_weight = float(dcs_actor_neighbor_weight)
        self.iql_deterministic = iql_deterministic
        self.scv_tau_min = float(scv_tau_min)
        self.scv_tau_max = float(scv_tau_max)
        self.scv_support_power = float(scv_support_power)
        self.scv_support_eps = float(scv_support_eps)
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

    @staticmethod
    def _build_pool(batch: TensorBatch) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Stack the sample itself with its kNN neighbors.

        Returns:
            pool_observations (B, 1+K, Ds)  [:, 0] is the sample itself
            pool_actions      (B, 1+K, Da)
            neighbor_weights  (B, K) clipped to [0, 1]
        """
        observations = batch["observations"]
        actions = batch["actions"]
        neighbor_observations = batch["neighbor_observations"]
        neighbor_actions = batch["neighbor_actions"]
        neighbor_weights = jnp.clip(batch["neighbor_weights"], 0.0, 1.0)
        pool_observations = jnp.concatenate(
            [observations[:, None, :], neighbor_observations], axis=1
        )
        pool_actions = jnp.concatenate([actions[:, None, :], neighbor_actions], axis=1)
        return pool_observations, pool_actions, neighbor_weights

    @staticmethod
    def _pool_weights(neighbor_weights: jnp.ndarray, neighbor_scale: float) -> jnp.ndarray:
        """Normalized pool weights: self gets mass 1, neighbor j gets
        neighbor_scale * w_ij, then the row is normalized to sum to 1.
        neighbor_scale = 0 or w = 0 recovers the plain single-sample loss."""
        ones = jnp.ones_like(neighbor_weights[:, :1])
        w = jnp.concatenate([ones, neighbor_scale * neighbor_weights], axis=1)
        return w / jnp.sum(w, axis=1, keepdims=True)

    def _build_train_step(self):
        q_apply = self.q_def.apply
        v_apply = self.v_def.apply
        q_tx = self.q_tx
        v_tx = self.v_tx
        actor_tx = self.actor_tx
        discount = self.discount
        tau = self.tau
        beta = self.beta
        tau_min = self.scv_tau_min
        tau_max = self.scv_tau_max
        support_power = self.scv_support_power
        support_eps = self.scv_support_eps
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
            junction = jnp.squeeze(batch["junction"], axis=-1)
            neighbor_weights = jnp.clip(batch["neighbor_weights"], 0.0, 1.0)

            # Support certificate c(s): effective number of certified neighbors.
            # This uses the graph only as a confidence signal; it never inserts
            # neighbor Q-values into the V expectile target.
            neighbor_mass = jnp.sum(neighbor_weights, axis=1)
            ess = (neighbor_mass ** 2) / (jnp.sum(neighbor_weights ** 2, axis=1) + support_eps)
            k = jnp.maximum(jnp.asarray(neighbor_weights.shape[1], dtype=jnp.float32), 1.0)
            support_conf = jnp.where(
                neighbor_mass > support_eps,
                jnp.clip(ess / k, 0.0, 1.0),
                0.0,
            )
            support_conf = support_conf ** support_power
            tau_s = tau_min + (tau_max - tau_min) * support_conf

            tq1, tq2 = q_apply({"params": state.q_target_params}, observations, actions)
            self_target_q = jnp.minimum(tq1, tq2)

            next_v = v_apply({"params": state.v_params}, next_observations)
            target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
            old_v = v_apply({"params": state.v_params}, observations)
            adv = self_target_q - old_v
            exp_adv = jnp.minimum(
                jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
            )

            # ----- Support-certified expectile V update.
            def v_loss_fn(v_params):
                v = v_apply({"params": v_params}, observations)
                diff = jax.lax.stop_gradient(self_target_q) - v
                expectile_weight = jnp.abs(tau_s - (diff < 0.0).astype(jnp.float32))
                value_loss = jnp.mean(expectile_weight * diff ** 2)
                return value_loss, v

            (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            # ----- Q update: standard IQL TD backup.
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

            # ----- Standard IQL AWR actor: no neighbor action copying.
            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_loss = jnp.sum((policy_out - actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    diff_a = (actions - mean) / std
                    log_prob = -0.5 * (diff_a ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
                    bc_loss = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_loss)
                return actor_loss, (bc_loss, policy_mean, log_std_mean)

            (actor_loss, (bc_loss, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
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
                "bc_loss_mean": jnp.mean(bc_loss),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
                # SCV diagnostics
                "support_conf_mean": jnp.mean(support_conf),
                "support_conf_min": jnp.min(support_conf),
                "support_conf_max": jnp.max(support_conf),
                "tau_s_mean": jnp.mean(tau_s),
                "tau_s_min": jnp.min(tau_s),
                "tau_s_max": jnp.max(tau_s),
                "neighbor_effective_count_mean": jnp.mean(ess),
                "neighbor_mass_mean": jnp.mean(neighbor_mass),
                "neighbor_weight_mean": jnp.mean(neighbor_weights),
                "gate_active_frac": jnp.mean(
                    (jnp.max(neighbor_weights, axis=1) > 0.0).astype(jnp.float32)
                ),
                "active_edge_frac": jnp.mean(
                    (neighbor_weights > 0.0).astype(jnp.float32)
                ),
                "junction_mean": jnp.mean(junction),
                "junction_max": jnp.max(junction),
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

            tq1, tq2 = q_apply({"params": iql_state.q_target_params}, observations, actions)
            target_q = jnp.minimum(tq1, tq2)
            v = v_apply({"params": iql_state.v_params}, observations)
            adv = target_q - v
            exp_adv = jnp.minimum(
                jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
            )

            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(actor_params, observations, training=True, rng=dropout_key)
                if iql_deterministic:
                    bc_loss = jnp.sum((policy_out - actions) ** 2, axis=-1)
                    policy_mean = policy_out
                    log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
                else:
                    mean, log_std = policy_out
                    std = jnp.exp(log_std)
                    diff_a = (actions - mean) / std
                    log_prob = -0.5 * (diff_a ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
                    bc_loss = -jnp.sum(log_prob, axis=-1)
                    policy_mean = mean
                    log_std_mean = jnp.mean(log_std)
                actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_loss)
                return actor_loss, (bc_loss, policy_mean, log_std_mean)

            (actor_loss, (bc_loss, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
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
                "bc_loss": jnp.mean(bc_loss),
                "adv_mean": jnp.mean(adv),
                "adv_min": jnp.min(adv),
                "adv_max": jnp.max(adv),
                "exp_adv_mean": jnp.mean(exp_adv),
                "exp_adv_max": jnp.max(exp_adv),
                "target_q_mean": jnp.mean(target_q),
                "v_mean": jnp.mean(v),
                "policy_mean": jnp.mean(policy_mean),
                "policy_log_std_mean": log_std_mean,
                "pool_neighbor_mass_actor": jnp.asarray(0.0, dtype=jnp.float32),
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
                    f"[{prefix}:dcs_awbc] step {fit_step}/{steps}: "
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
    """Return (run_dir, checkpoint_path) for a saved checkpoint.

    Supported load_model formats:

    1. Direct checkpoint file:
       path/to/checkpoint.pkl

    2. Direct run directory:
       path/to/run_dir/
       where path/to/run_dir/checkpoint.pkl exists

    3. Parent directory that contains env/seed subdirectory:
       path/to/base_dir/
       where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists

    Note: DC-IQL checkpoints are loadable as well (identical architectures);
    only the actor/value training rules differ.
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
    Unknown saved keys (e.g. legacy DC-IQL fields) are silently dropped.
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

    checkpoint_manager = None
    checkpoint_preparation = None
    if config.checkpoints_path is not None and (not refit_only):
        current_config_dict = asdict(config)
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path,
            current_config=current_config_dict,
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="scv_iql_jax_training_progress",
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
    env, dataset, dataset_backend = load_env_and_dataset(
        config.env,
        add_info=(config.use_dcs and config.dcs_metric_source == "oracle"),
    )

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

    # ----- DCS coverage profile: one-time kNN graph + junction gating -------
    neighbor_indices = None
    neighbor_weights = None
    junction = None
    buffer_neighbor_k = config.dcs_k
    if config.use_dcs:
        profile = load_or_compute_coverage_profile(config, dataset, device=jax_device)
        neighbor_indices, neighbor_weights, junction = build_neighbor_weights(config, profile)
        buffer_neighbor_k = neighbor_indices.shape[1]
        if buffer_neighbor_k != config.dcs_k:
            print(
                f"Note: coverage profile holds k={buffer_neighbor_k} neighbors "
                f"(dcs_k={config.dcs_k} was clamped during computation)."
            )
    else:
        print("use_dcs=False: neighbor pool disabled; training reduces to standard IQL.")

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        neighbor_k=buffer_neighbor_k,
        device=jax_device,
    )
    replay_buffer.load_d4rl_dataset(
        dataset,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        junction=junction,
    )

    max_action = float(env.action_space.high[0])


    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    run_mode_name = "Actor refit" if refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
    print(
        f"scv_tau=[{config.scv_tau_min}, {config.scv_tau_max}] | use_dcs={config.use_dcs} | "
        f"k={buffer_neighbor_k} | eta_V={config.dcs_value_neighbor_weight} | "
        f"eta_pi={config.dcs_actor_neighbor_weight} | "
        f"gate=[{config.dcs_gate_low_percentile}, {config.dcs_gate_high_percentile}]%% of "
        f"J(mode={config.dcs_diversity_mode})"
    )
    if config.use_dcs:
        if config.dcs_use_dynamics_gate:
            print(
                f"dynamics gate: ON (scale={config.dcs_dynamics_scale}, "
                f"ridge={config.dcs_dynamics_ridge}) -> Density-Certified Stitching"
            )
        else:
            print(
                "dynamics gate: OFF (g=1) -> density-and-diversity gated vicinal IQL "
                "(ablation)"
            )
        if config.dcs_metric_source == "oracle":
            keys = resolve_oracle_keys(config)
            print(
                f"*** ORACLE CEILING TEST: kNN graph built in ORACLE space "
                f"(keys={list(keys) if keys else 'auto'}); training still uses "
                f"standard observations. ***"
            )
        elif config.dcs_metric_source == "temporal":
            print(
                f"*** TEMPORAL METRIC: kNN graph built in a LEARNED temporal "
                f"space phi(s) (dim={config.dcs_temporal_dim}, "
                f"horizon={config.dcs_temporal_horizon}, "
                f"steps={config.dcs_temporal_steps}); data-only, deployable. "
                f"Training still uses standard observations. ***"
            )
        else:
            print("metric source: observation (standard L2 kNN graph)")
    print("---------------------------------------")

    trainer = SupportCertifiedIQLJAX(
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
        scv_tau_min=config.scv_tau_min,
        scv_tau_max=config.scv_tau_max,
        scv_support_power=config.scv_support_power,
        scv_support_eps=config.scv_support_eps,
        dcs_value_neighbor_weight=0.0,
        dcs_actor_neighbor_weight=0.0,
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
            try:
                checkpoint_manager.initialize_wandb(
                    wandb_module=wandb,
                    config=asdict(config),
                    code_root=_PROJECT_ROOT,
                )
            except Exception as exc:
                print(f"Warning: W&B resume failed: {exc}")
                print("Continuing local training with a new W&B run.")
                if getattr(wandb, "run", None) is not None:
                    wandb.finish(exit_code=1)
                checkpoint_manager.initialize_fresh_wandb(
                    wandb_module=wandb,
                    config=asdict(config),
                    code_root=_PROJECT_ROOT,
                )
        else:
            wandb_init(asdict(config))


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
        print(f"Actor refit from saved checkpoint ({ALGORITHM_NAME} vicinal AWR)")
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



if __name__ == "__main__":
    train()
