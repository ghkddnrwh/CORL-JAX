# pi_jg_iql_jax.py
#
# PI-JG-IQL: Prototype-Indexed Approximate Junction-Gated Implicit Q-Learning
# (JAX/Flax).
#
# Efficient, locality-preserving approximation of JG-IQL. Instead of building
# a full-dataset exact kNN structure, this variant uses state-space prototypes
# only as a coarse inverted index:
#
#   1) fit M prototypes c_m in the selected metric space,
#   2) assign every dataset transition to its nearest prototype,
#   3) for each anchor s_i, retrieve members of the r nearest prototype cells,
#   4) compute the exact top-k nearest transitions INSIDE that candidate pool,
#   5) compute the same JG-IQL local statistics from those actual top-k neighbors.
#
# Thus prototypes do NOT define diversity and no branch-biased coreset is used.
# They only reduce the search space before an exact local top-k refinement.
#
# For each state i:
#   A_i : RMS spread of actions among the retrieved top-k neighbors,
#   P_i : directional dispersion of their next-state displacements,
#   J_i : configurable gate signal, by default sqrt(A_i * P_i),
#   G_i : percentile ramp or automatic Beta-mixture calibration,
#   tau_i = tau_min + (tau_max - tau_min) * G_i ** p.
#
# Q-learning and actor extraction remain standard IQL.
#
import copy
import hashlib
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
    from scipy.special import betaln as scipy_betaln
    from scipy.special import logsumexp as scipy_logsumexp

    if not hasattr(scipy_linalg, "tril"):
        scipy_linalg.tril = np.tril
    if not hasattr(scipy_linalg, "triu"):
        scipy_linalg.triu = np.triu
except ImportError:
    scipy_betaln = None
    scipy_logsumexp = None

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
    log_training_exceptions,
    upsert_eval_log,
)

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "PI-JG-IQL"
ALGORITHM_FULL_NAME = "Prototype-Indexed Approximate Junction-Gated Implicit Q-Learning"

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

DCS_GATE_SOURCES = (
    "density",
    "action",
    "displacement",
    "product_diversity",
    "density_x_action",
    "density_x_displacement",
    "density_x_product",
)


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

    # ----- PI-JG-IQL: prototype-indexed approximate JG value learning --------
    # Neighbor Q/action pooling is not used. A state gate G_i controls
    # c(s), which in turn controls the expectile asymmetry in V-learning.
    # The old scv_* names are kept as CLI-compatible aliases.
    scv_tau_min: float = 0.5
    scv_tau_max: float = 0.8
    scv_support_power: float = 1.0
    scv_support_eps: float = 1e-8

    # ----- Branching gate ------------------------------------------------
    # Master switch. False -> no prototype-indexed preprocessing and
    # tau(s)=scv_tau_min, equivalent to standard IQL when
    # scv_tau_min == scv_tau_max.
    use_dcs: bool = True

    # Final top-k used for JG statistics. Prototype indexing only accelerates retrieval.
    dcs_k: int = 8

    # Legacy compatibility only. JG-IQL never pools neighbor Q-values or
    # neighbor actions; these are ignored and trainer receives 0.0 internally.
    dcs_value_neighbor_weight: float = 0.0
    dcs_actor_neighbor_weight: float = 0.0

    # Junction/gate signal source used to build the state gate G_i.
    # This mirrors KES-G-IQL's ablation knob, but JG-IQL uses the resulting
    # state-level gate directly as c(s), instead of computing kernel ESS.
    #   density                : G_i from density confidence only
    #   action                 : G_i from action-spread diversity only
    #   displacement           : G_i from next-displacement diversity only
    #   product_diversity      : G_i from sqrt(action_div * disp_div), no density
    #   density_x_action       : G_i from density * action diversity
    #   density_x_displacement : G_i from density * displacement diversity
    #   density_x_product      : G_i from density * sqrt(action_div * disp_div)
    dcs_gate_source: str = "product_diversity"

    # Backward-compatible alias used only when old hyperparameter files/scripts
    # still pass --dcs_diversity_mode. If dcs_gate_source is not explicitly
    # overridden, apply_env_hyperparams maps dcs_diversity_mode to the matching
    # density_x_* gate source.
    dcs_diversity_mode: str = "product"

    # Percentile scaling of raw branch statistics into [0, 1] confidences.
    dcs_percentile_low: float = 5.0
    dcs_percentile_high: float = 95.0

    # ----- Gate calibration -----------------------------------------------
    # beta_mixture (default): fit a two-component Beta mixture to the bounded
    # gate signal J_i in [0, 1]. The component with the larger mean is treated
    # as the junction-like population, and its posterior probability is used
    # directly as the state gate:
    #
    #   G_i = P(z_i = high-mean component | J_i).
    #
    # This removes the need to manually tune low/high gate percentiles.
    # percentile: legacy ramp gate kept only for backward-compatible ablations.
    dcs_gate_calibration: str = "beta_mixture"  # beta_mixture | percentile

    # Legacy percentile-ramp settings. Ignored when dcs_gate_calibration is
    # "beta_mixture", except when dcs_beta_mixture_fallback="percentile".
    dcs_gate_low_percentile: float = 60.0
    dcs_gate_high_percentile: float = 95.0

    # Numerical controls for the automatic two-component Beta mixture. These
    # are optimization safeguards rather than task-level gate thresholds.
    dcs_beta_mixture_max_iter: int = 200
    dcs_beta_mixture_tol: float = 1e-6
    dcs_beta_mixture_eps: float = 1e-4
    dcs_beta_mixture_fit_samples: int = 200_000
    # Degenerate distributions have no identifiable two-population structure.
    # "zero" is conservative and fully automatic; "percentile" recovers the
    # old manually specified ramp as a compatibility fallback.
    dcs_beta_mixture_fallback: str = "zero"  # zero | percentile

    # ----- Prototype-indexed approximate kNN -------------------------------
    # Prototypes are used ONLY as a coarse inverted index. Every state is
    # assigned to one prototype cell. For each anchor, members of the nearest
    # prototype cells form a candidate pool, and the exact top-k nearest states
    # are selected from that pool using the same metric space as JG-IQL.
    #
    # Existing bc_* names are retained for CLI/YAML compatibility with the
    # previous BC-JG-IQL implementation, even though no branch coreset remains.

    # Number of state-space prototypes. Larger values make cells smaller and
    # improve locality, but increase prototype fitting / lookup cost.
    bc_num_prototypes: int = 512

    # Prototype fitting uses at most this many states and streaming mini-batch
    # k-means. The final approximate top-k profile is still computed for every
    # dataset state.
    bc_prototype_fit_samples: int = 100_000
    bc_kmeans_iters: int = 10
    bc_kmeans_batch_size: int = 4096

    # Initial number r of nearest prototype cells consulted for each state.
    # If those cells do not contain at least k non-self candidates, additional
    # prototype cells are added automatically until enough candidates exist.
    bc_nearest_prototypes: int = 2

    # Chunk size for all-state prototype lookup / approximate top-k inference.
    # Neighbor refinement itself uses variable-size candidate pools per state.
    bc_profile_chunk_size: int = 2048

    # Preprocessing seed, independent of RL minibatch sampling.
    bc_preprocess_seed: int = 0

    # Optional diagnostic: on this many random anchors, compute full-dataset
    # exact kNN and report recall@k of the prototype-indexed approximation.
    # Set 0 to disable because the diagnostic itself is expensive.
    bc_exact_recall_samples: int = 0

    # Cache controls. The cache stores only final per-state statistics / gate and
    # diagnostics; no full N x k graph is kept in the ReplayBuffer.
    bc_force_recompute: bool = False

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

    # ----- METRIC SOURCE: which space prototypes/local relations use -------
    # Selects the metric space in which prototypes and local soft relations are built:
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
    # Standardize the oracle space per-dim before prototype fitting so no
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

    # ----- Network input representation ablations --------------------------
    # These switches independently control whether each learned function sees
    # the frozen temporal embedding phi(s) or the normalized raw observation s.
    # They require dcs_metric_source="temporal" because only that metric has a
    # deployable encoder that can transform unseen evaluation states online.
    actor_use_embedding: bool = False
    q_use_embedding: bool = False
    v_use_embedding: bool = False
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
    # Keep the historical run name by default. Uncomment the next two lines if
    # you want separate checkpoint directories for each gate-source ablation.
    # gate_source = getattr(config, "dcs_gate_source", "density_x_product")
    # config.name = f"{ALGORITHM_NAME}-JAX-{config.env}-gate_{gate_source}"
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
    assert config.dcs_gate_source in DCS_GATE_SOURCES, (
        f"dcs_gate_source must be one of {DCS_GATE_SOURCES}, got {config.dcs_gate_source}"
    )
    assert config.dcs_diversity_mode in ("action", "displacement", "product")
    assert 0.0 <= config.dcs_percentile_low < config.dcs_percentile_high <= 100.0
    assert config.dcs_gate_calibration in ("beta_mixture", "percentile")
    assert 0.0 <= config.dcs_gate_low_percentile <= config.dcs_gate_high_percentile <= 100.0
    assert config.dcs_beta_mixture_max_iter >= 1
    assert config.dcs_beta_mixture_tol > 0.0
    assert 0.0 < config.dcs_beta_mixture_eps < 0.5
    assert config.dcs_beta_mixture_fit_samples >= 2
    assert config.dcs_beta_mixture_fallback in ("zero", "percentile")
    assert config.bc_num_prototypes >= 1
    assert config.bc_prototype_fit_samples >= 2
    assert config.bc_kmeans_iters >= 1
    assert config.bc_kmeans_batch_size >= 1
    assert config.bc_nearest_prototypes >= 1
    assert config.bc_profile_chunk_size >= 1
    assert config.bc_exact_recall_samples >= 0
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
    uses_embedding_input = (
        config.actor_use_embedding or config.q_use_embedding or config.v_use_embedding
    )
    if uses_embedding_input:
        assert config.dcs_metric_source == "temporal", (
            "actor_use_embedding/q_use_embedding/v_use_embedding require "
            "dcs_metric_source='temporal' so unseen evaluation states can be "
            "encoded by the learned temporal encoder."
        )
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

    # Backward compatibility: old YAML/scripts may only specify
    # dcs_diversity_mode. In the new ablation interface, that corresponds to
    # density_x_{action, displacement, product}. Do this only when the new knob
    # was not explicitly set by CLI/YAML.
    if "dcs_gate_source" not in applied_fields and "dcs_gate_source" not in cli_overrides:
        legacy_to_gate_source = {
            "action": "action",
            "displacement": "displacement",
            "product": "product_diversity",
        }
        config.dcs_gate_source = legacy_to_gate_source.get(
            config.dcs_diversity_mode, config.dcs_gate_source
        )

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


def get_temporal_representation_cache_path(config: TrainConfig) -> Optional[Path]:
    """Cache path for the deployable temporal representation bundle.

    Unlike the legacy embedding-only .npz cache, this bundle also stores the
    frozen encoder parameters and the embedding standardization statistics, so
    the same phi(s) can be applied to unseen evaluation states.
    """
    if config.dcs_profile_path is None:
        return None
    root = Path(config.dcs_profile_path)
    env_name = _safe_path_name(config.env)
    sig_hash = tmet.signature_hash(build_temporal_signature(config))
    filename = f"temporal_repr_{sig_hash}.pkl"
    if config.dcs_cache_by_seed:
        return root / env_name / f"seed_{config.seed}" / filename
    return root / env_name / filename


def _temporal_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(logp[jnp.arange(logits.shape[0]), labels])


def _encode_temporal_array(
    encoder: Any,
    params: Any,
    observations: np.ndarray,
    device: Any,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Encode an arbitrary array with the frozen temporal encoder."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape[0] == 0:
        return np.zeros((0, int(encoder.embed_dim)), dtype=np.float32)

    output = np.empty((observations.shape[0], int(encoder.embed_dim)), dtype=np.float32)
    apply_fn = jax.jit(lambda p, x: encoder.apply({"params": p}, x))
    for start in range(0, observations.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), observations.shape[0])
        z = apply_fn(
            params,
            jax.device_put(jnp.asarray(observations[start:end]), device),
        )
        output[start:end] = np.asarray(jax.device_get(z), dtype=np.float32)
    return output


def train_temporal_representation(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> Dict[str, Any]:
    """Train phi and return everything needed for graph construction and RL.

    The encoder is trained once and then frozen. Dataset states are pre-encoded
    for efficient minibatch training, while the saved encoder parameters are
    used to encode unseen environment states during actor evaluation.
    """
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    if "next_observations" not in dataset:
        raise SystemExit(
            "dcs_metric_source='temporal' requires next_observations in the dataset."
        )
    next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
    n = observations.shape[0]

    if device is None:
        device = jax.devices()[0]

    encoder = tmet.TemporalEncoder(
        embed_dim=config.dcs_temporal_dim,
        hidden_dim=config.dcs_temporal_hidden,
        n_hidden=config.dcs_temporal_layers,
        normalize=config.dcs_temporal_normalize,
    )
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    params = encoder.init(
        init_key,
        jnp.zeros((1, observations.shape[1]), dtype=jnp.float32),
    )["params"]

    tx = optax.adam(config.dcs_temporal_lr)
    opt_state = tx.init(params)
    inv_temp = 1.0 / max(float(config.dcs_temporal_temperature), 1e-8)

    @jax.jit
    def temporal_train_step(params, opt_state, anchor_obs, positive_obs):
        def loss_fn(p):
            za = encoder.apply({"params": p}, anchor_obs)
            zp = encoder.apply({"params": p}, positive_obs)
            logits = (za @ zp.T) * inv_temp
            labels = jnp.arange(logits.shape[0])
            loss = 0.5 * (
                _temporal_cross_entropy(logits, labels)
                + _temporal_cross_entropy(logits.T, labels)
            )
            pos_sim = jnp.mean(jnp.sum(za * zp, axis=-1))
            return loss, pos_sim

        (loss, pos_sim), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pos_sim

    next_index = None
    use_kstep = config.dcs_temporal_horizon > 1
    if use_kstep:
        next_index = tmet.build_successor_index(dataset)

    rng = np.random.default_rng(config.seed)
    print(
        f"Training deployable temporal representation: N={n} "
        f"dim={config.dcs_temporal_dim} steps={config.dcs_temporal_steps} "
        f"batch={config.dcs_temporal_batch} horizon={config.dcs_temporal_horizon} "
        f"temp={config.dcs_temporal_temperature} "
        f"positives={'k-step' if use_kstep else '1-step successor'}"
    )
    for it in range(int(config.dcs_temporal_steps)):
        if use_kstep:
            anchor_idx, positive_idx = tmet.sample_kstep_pairs(
                next_index,
                config.dcs_temporal_horizon,
                config.dcs_temporal_batch,
                rng,
            )
            anchor_obs = observations[anchor_idx]
            positive_obs = observations[positive_idx]
        else:
            anchor_idx = rng.integers(0, n, size=config.dcs_temporal_batch)
            anchor_obs = observations[anchor_idx]
            positive_obs = next_observations[anchor_idx]

        params, opt_state, loss, pos_sim = temporal_train_step(
            params,
            opt_state,
            jax.device_put(jnp.asarray(anchor_obs), device),
            jax.device_put(jnp.asarray(positive_obs), device),
        )
        if (it + 1) % 5_000 == 0:
            print(
                f"  temporal step {it + 1}/{config.dcs_temporal_steps}: "
                f"infonce={float(jax.device_get(loss)):.4f} "
                f"pos_cos={float(jax.device_get(pos_sim)):.4f}"
            )

    embeddings = _encode_temporal_array(encoder, params, observations, device)
    next_embeddings = _encode_temporal_array(encoder, params, next_observations, device)

    if config.dcs_temporal_standardize:
        embedding_mean = embeddings.mean(axis=0).astype(np.float32)
        embedding_std = (embeddings.std(axis=0) + 1e-6).astype(np.float32)
        embeddings = ((embeddings - embedding_mean) / embedding_std).astype(np.float32)
        next_embeddings = (
            (next_embeddings - embedding_mean) / embedding_std
        ).astype(np.float32)
    else:
        embedding_mean = np.zeros((config.dcs_temporal_dim,), dtype=np.float32)
        embedding_std = np.ones((config.dcs_temporal_dim,), dtype=np.float32)

    return {
        "signature": tmet.temporal_signature(
            build_temporal_signature(config, observations_shape=observations.shape)
        ),
        "embeddings": embeddings.astype(np.float32),
        "next_embeddings": next_embeddings.astype(np.float32),
        "encoder_params": serialization.to_state_dict(params),
        "embedding_mean": embedding_mean,
        "embedding_std": embedding_std,
    }


def load_or_build_temporal_representation(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> Dict[str, Any]:
    """Load or train the deployable temporal representation bundle."""
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    expected_signature = tmet.temporal_signature(
        build_temporal_signature(config, observations_shape=observations.shape)
    )
    cache_path = get_temporal_representation_cache_path(config)

    if cache_path is not None and cache_path.exists() and not config.dcs_temporal_force_recompute:
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            required = {
                "signature",
                "embeddings",
                "next_embeddings",
                "encoder_params",
                "embedding_mean",
                "embedding_std",
            }
            if payload.get("signature") == expected_signature and required.issubset(payload.keys()):
                print(f"Loaded deployable temporal representation from: {cache_path}")
                return payload
            print(f"Temporal representation cache is stale/incomplete; retraining: {cache_path}")
        except Exception as exc:
            print(f"Failed to load temporal representation cache {cache_path}: {exc}; retraining.")

    representation = train_temporal_representation(config, dataset, device=device)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(representation, f)
        print(f"Saved deployable temporal representation to: {cache_path}")

    print(
        f"Temporal representation: dim={representation['embeddings'].shape[1]} "
        f"normalize={config.dcs_temporal_normalize} "
        f"standardize={config.dcs_temporal_standardize} "
        f"horizon={config.dcs_temporal_horizon}"
    )
    return representation


def build_temporal_metric_space(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> np.ndarray:
    """Backward-compatible wrapper returning phi(observations)."""
    representation = load_or_build_temporal_representation(config, dataset, device=device)
    return np.asarray(representation["embeddings"], dtype=np.float32)


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



def _squared_distance_matrix(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Memory-efficient squared Euclidean distance matrix."""
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (x @ centers.T)
    return np.maximum(d2, 0.0).astype(np.float32)


def _normalized_displacements(
    observations: np.ndarray,
    next_observations: np.ndarray,
) -> np.ndarray:
    disp = np.asarray(next_observations, dtype=np.float32) - np.asarray(
        observations, dtype=np.float32
    )
    norms = np.linalg.norm(disp, axis=-1, keepdims=True)
    return np.where(
        norms > 1e-8,
        disp / np.maximum(norms, 1e-8),
        0.0,
    ).astype(np.float32)


def build_prototype_index_signature(
    config: TrainConfig,
    metric_space: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, Any]:
    """Cache signature for prototype-indexed approximate JG preprocessing."""
    temporal_sig = (
        tmet.signature_hash(build_temporal_signature(config))
        if config.dcs_metric_source == "temporal"
        else None
    )
    return {
        "version": 2,
        "env": config.env,
        "metric_source": config.dcs_metric_source,
        "metric_shape": tuple(metric_space.shape),
        "actions_shape": tuple(actions.shape),
        "normalize": bool(config.normalize),
        "temporal_signature": temporal_sig,
        "num_prototypes": int(config.bc_num_prototypes),
        "prototype_fit_samples": int(config.bc_prototype_fit_samples),
        "kmeans_iters": int(config.bc_kmeans_iters),
        "kmeans_batch_size": int(config.bc_kmeans_batch_size),
        "nearest_prototypes": int(config.bc_nearest_prototypes),
        "top_k": int(config.dcs_k),
        "preprocess_seed": int(config.bc_preprocess_seed),
        "exact_recall_samples": int(config.bc_exact_recall_samples),
        "gate_source": config.dcs_gate_source,
        "percentile_low": float(config.dcs_percentile_low),
        "percentile_high": float(config.dcs_percentile_high),
        "gate_calibration": config.dcs_gate_calibration,
        "gate_low_percentile": float(config.dcs_gate_low_percentile),
        "gate_high_percentile": float(config.dcs_gate_high_percentile),
        "beta_max_iter": int(config.dcs_beta_mixture_max_iter),
        "beta_tol": float(config.dcs_beta_mixture_tol),
        "beta_eps": float(config.dcs_beta_mixture_eps),
        "beta_fit_samples": int(config.dcs_beta_mixture_fit_samples),
        "beta_fallback": config.dcs_beta_mixture_fallback,
    }


def get_prototype_index_cache_path(
    config: TrainConfig,
    metric_space: np.ndarray,
    actions: np.ndarray,
) -> Optional[Path]:
    if config.dcs_profile_path is None:
        return None
    signature = build_prototype_index_signature(config, metric_space, actions)
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig_hash = hashlib.sha1(payload).hexdigest()[:12]
    root = Path(config.dcs_profile_path)
    env_name = _safe_path_name(config.env)
    return root / env_name / f"prototype_indexed_jg_profile_{sig_hash}.npz"


def _fit_streaming_prototypes(
    metric_space: np.ndarray,
    num_prototypes: int,
    fit_samples: int,
    n_iters: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Fit compact state prototypes with streaming mini-batch k-means."""
    z = np.asarray(metric_space, dtype=np.float32)
    n = z.shape[0]
    m = min(int(num_prototypes), n)
    s = min(int(fit_samples), n)
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(n, size=s, replace=False) if s < n else np.arange(n)
    fit_z = z[fit_idx]

    init_idx = rng.choice(s, size=m, replace=False)
    centers = fit_z[init_idx].copy()
    counts = np.ones((m,), dtype=np.float64)

    for epoch in range(int(n_iters)):
        order = rng.permutation(s)
        inertia_sum = 0.0
        seen = 0
        for start in range(0, s, int(batch_size)):
            batch = fit_z[order[start : start + int(batch_size)]]
            d2 = _squared_distance_matrix(batch, centers)
            labels = np.argmin(d2, axis=1)
            inertia_sum += float(np.sum(d2[np.arange(batch.shape[0]), labels]))
            seen += batch.shape[0]

            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                n_new = int(np.sum(mask))
                batch_mean = np.mean(batch[mask], axis=0)
                old_count = counts[cluster_id]
                new_count = old_count + n_new
                centers[cluster_id] = (
                    (old_count / new_count) * centers[cluster_id]
                    + (n_new / new_count) * batch_mean
                )
                counts[cluster_id] = new_count

        print(
            f"  prototype epoch {epoch + 1}/{n_iters}: "
            f"mean_fit_d2={inertia_sum / max(seen, 1):.6f}"
        )

    return centers.astype(np.float32)


def _assign_to_prototypes(
    metric_space: np.ndarray,
    centers: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = metric_space.shape[0]
    assignments = np.empty((n,), dtype=np.int32)
    nearest_d2 = np.empty((n,), dtype=np.float32)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        d2 = _squared_distance_matrix(metric_space[start:end], centers)
        labels = np.argmin(d2, axis=1)
        assignments[start:end] = labels.astype(np.int32)
        nearest_d2[start:end] = d2[np.arange(end - start), labels]
    return assignments, nearest_d2


def _build_inverted_prototype_index(
    assignments: np.ndarray,
    n_prototypes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return members sorted by prototype plus offsets for O(1) cell lookup."""
    assignments = np.asarray(assignments, dtype=np.int32)
    order = np.argsort(assignments, kind="stable").astype(np.int32)
    counts = np.bincount(assignments, minlength=n_prototypes).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return order, offsets, counts


def _gather_candidate_pool(
    prototype_order: np.ndarray,
    member_order: np.ndarray,
    offsets: np.ndarray,
    self_index: int,
    min_candidates: int,
    initial_prototypes: int,
) -> Tuple[np.ndarray, int]:
    """Gather members from nearby cells, expanding only if k candidates are absent."""
    selected_parts: List[np.ndarray] = []
    total = 0
    used = 0
    initial_prototypes = max(1, int(initial_prototypes))

    for rank, proto in enumerate(prototype_order):
        lo = int(offsets[int(proto)])
        hi = int(offsets[int(proto) + 1])
        if hi > lo:
            members = member_order[lo:hi]
            selected_parts.append(members)
            total += int(members.size)
        used = rank + 1
        # Account for the anchor itself, which may be inside the pool.
        enough_nonself = total - 1 >= min_candidates
        if used >= initial_prototypes and enough_nonself:
            break

    if not selected_parts:
        return np.empty((0,), dtype=np.int32), used

    candidates = np.concatenate(selected_parts).astype(np.int32, copy=False)
    candidates = candidates[candidates != int(self_index)]
    return candidates, used


def _approximate_neighbors_from_prototype_index(
    metric_space: np.ndarray,
    centers: np.ndarray,
    member_order: np.ndarray,
    offsets: np.ndarray,
    k: int,
    nearest_prototypes: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coarse prototype retrieval followed by exact top-k within the pool.

    This is the core approximation. Prototypes only choose a candidate pool;
    final neighbors are actual dataset transitions selected by exact metric
    distance within that pool.
    """
    z = np.asarray(metric_space, dtype=np.float32)
    n = z.shape[0]
    k_eff = int(min(k, max(n - 1, 1)))
    n_prototypes = centers.shape[0]
    initial_r = int(min(max(nearest_prototypes, 1), n_prototypes))

    neighbor_indices = np.empty((n, k_eff), dtype=np.int32)
    neighbor_distances = np.empty((n, k_eff), dtype=np.float32)
    candidate_count = np.empty((n,), dtype=np.int32)
    prototypes_used = np.empty((n,), dtype=np.int32)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        anchors = z[start:end]
        center_d2 = _squared_distance_matrix(anchors, centers)
        # Full center ordering is cheap because M << N and enables adaptive
        # expansion when the nearest r cells do not contain enough candidates.
        proto_rank = np.argsort(center_d2, axis=1, kind="stable")

        for local_row, global_i in enumerate(range(start, end)):
            candidates, used = _gather_candidate_pool(
                prototype_order=proto_rank[local_row],
                member_order=member_order,
                offsets=offsets,
                self_index=global_i,
                min_candidates=k_eff,
                initial_prototypes=initial_r,
            )
            if candidates.size < k_eff:
                raise RuntimeError(
                    f"Prototype index produced only {candidates.size} non-self "
                    f"candidates for state {global_i}, but k={k_eff}."
                )

            diff = z[candidates] - z[global_i]
            d2 = np.einsum("nd,nd->n", diff, diff)

            if candidates.size == k_eff:
                chosen_local = np.arange(k_eff)
            else:
                chosen_local = np.argpartition(d2, k_eff - 1)[:k_eff]
            # Deterministic ordering of the selected top-k by distance, then index.
            selected_idx = candidates[chosen_local]
            selected_d2 = d2[chosen_local]
            rank = np.lexsort((selected_idx, selected_d2))
            selected_idx = selected_idx[rank]
            selected_d2 = selected_d2[rank]

            neighbor_indices[global_i] = selected_idx.astype(np.int32)
            neighbor_distances[global_i] = np.sqrt(
                np.maximum(selected_d2, 0.0)
            ).astype(np.float32)
            candidate_count[global_i] = int(candidates.size)
            prototypes_used[global_i] = int(used)

        if start == 0 or end == n or (start // int(chunk_size)) % 25 == 0:
            print(f"  prototype-indexed top-k: {end}/{n} states")

    return neighbor_indices, neighbor_distances, candidate_count, prototypes_used


def _compute_jg_statistics_from_neighbors(
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    neighbor_indices: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match JG-IQL's action-spread and displacement-dispersion definitions."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    next_observations = np.asarray(next_observations, dtype=np.float32)
    idx_all = np.asarray(neighbor_indices, dtype=np.int32)
    n = observations.shape[0]

    disp = next_observations - observations
    disp_norm = np.linalg.norm(disp, axis=-1)
    unit_disp = disp / np.maximum(disp_norm[:, None], 1e-8)
    unit_disp = np.where(disp_norm[:, None] > 1e-8, unit_disp, 0.0).astype(np.float32)
    moving = (disp_norm > 1e-8).astype(np.float32)

    action_spread = np.empty((n,), dtype=np.float32)
    disp_dispersion = np.empty((n,), dtype=np.float32)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        idx = idx_all[start:end]

        nbr_actions = actions[idx]
        centroid = nbr_actions.mean(axis=1, keepdims=True)
        sq = ((nbr_actions - centroid) ** 2).sum(axis=-1)
        action_spread[start:end] = np.sqrt(sq.mean(axis=1)).astype(np.float32)

        nbr_unit = unit_disp[idx]
        resultant = np.linalg.norm(nbr_unit.sum(axis=1), axis=-1)
        total = moving[idx].sum(axis=1)
        dispersion = 1.0 - resultant / np.maximum(total, 1e-8)
        dispersion = np.where(total > 0.5, dispersion, 0.0)
        disp_dispersion[start:end] = np.clip(dispersion, 0.0, 1.0).astype(np.float32)

    return action_spread, disp_dispersion


def _prototype_index_exact_recall_diagnostic(
    metric_space: np.ndarray,
    approx_neighbors: np.ndarray,
    sample_count: int,
    seed: int,
) -> Dict[str, float]:
    """Optional expensive recall@k check against full-dataset exact search."""
    n, k = approx_neighbors.shape
    sample_count = int(min(max(sample_count, 0), n))
    if sample_count <= 0:
        return {"exact_recall_samples": 0.0, "exact_recall_at_k": float("nan")}

    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=sample_count, replace=False)
    recalls = []
    z = np.asarray(metric_space, dtype=np.float32)
    for i in anchors:
        diff = z - z[i]
        d2 = np.einsum("nd,nd->n", diff, diff)
        d2[int(i)] = np.inf
        exact = np.argpartition(d2, k - 1)[:k]
        recalls.append(len(set(exact.tolist()) & set(approx_neighbors[i].tolist())) / float(k))
    return {
        "exact_recall_samples": float(sample_count),
        "exact_recall_at_k": float(np.mean(recalls)),
    }


def compute_prototype_indexed_profile(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    metric_space: np.ndarray,
) -> Dict[str, Any]:
    """Approximate JG-IQL profile via prototype retrieval + exact local top-k."""
    metric_space = np.asarray(metric_space, dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
    n = observations.shape[0]

    n_prototypes = min(int(config.bc_num_prototypes), n)
    n_near = min(int(config.bc_nearest_prototypes), n_prototypes)
    k_eff = int(min(config.dcs_k, max(n - 1, 1)))
    print(
        f"Building prototype-indexed approximate JG profile: N={n}, "
        f"prototypes={n_prototypes}, initial_near_prototypes={n_near}, k={k_eff}"
    )

    centers = _fit_streaming_prototypes(
        metric_space,
        num_prototypes=n_prototypes,
        fit_samples=config.bc_prototype_fit_samples,
        n_iters=config.bc_kmeans_iters,
        batch_size=config.bc_kmeans_batch_size,
        seed=config.bc_preprocess_seed,
    )
    assignments, nearest_d2 = _assign_to_prototypes(
        metric_space, centers, config.bc_profile_chunk_size
    )
    member_order, offsets, cell_counts = _build_inverted_prototype_index(
        assignments, n_prototypes
    )

    neighbor_indices, neighbor_distances, candidate_count, prototypes_used = (
        _approximate_neighbors_from_prototype_index(
            metric_space=metric_space,
            centers=centers,
            member_order=member_order,
            offsets=offsets,
            k=k_eff,
            nearest_prototypes=n_near,
            chunk_size=config.bc_profile_chunk_size,
        )
    )

    action_spread, disp_dispersion = _compute_jg_statistics_from_neighbors(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        neighbor_indices=neighbor_indices,
        chunk_size=config.bc_profile_chunk_size,
    )
    knn_radius = neighbor_distances[:, -1].astype(np.float32)

    density_conf = covp.percentile_confidence(
        knn_radius,
        config.dcs_percentile_low,
        config.dcs_percentile_high,
        invert=True,
    )
    action_div_conf = covp.percentile_confidence(
        action_spread,
        config.dcs_percentile_low,
        config.dcs_percentile_high,
    )
    disp_div_conf = covp.percentile_confidence(
        disp_dispersion,
        config.dcs_percentile_low,
        config.dcs_percentile_high,
    )

    gate_signal, gate_signal_name = build_gate_signal(
        config, density_conf, action_div_conf, disp_div_conf
    )
    gate, gate_diagnostics = build_data_adaptive_gate(config, gate_signal)

    recall_diag = _prototype_index_exact_recall_diagnostic(
        metric_space=metric_space,
        approx_neighbors=neighbor_indices,
        sample_count=config.bc_exact_recall_samples,
        seed=config.bc_preprocess_seed,
    )

    initial_r = max(1, n_near)
    expanded_frac = float(np.mean(prototypes_used > initial_r))
    positive_cell_counts = cell_counts[cell_counts > 0]

    return {
        "density_conf": np.asarray(density_conf, dtype=np.float32),
        "action_div_conf": np.asarray(action_div_conf, dtype=np.float32),
        "disp_div_conf": np.asarray(disp_div_conf, dtype=np.float32),
        "gate_signal": np.asarray(gate_signal, dtype=np.float32),
        "gate": np.asarray(gate, dtype=np.float32),
        "knn_radius_raw": knn_radius,
        "action_spread_raw": action_spread,
        "disp_dispersion_raw": disp_dispersion,
        "effective_ref_count": np.full((n,), k_eff, dtype=np.float32),
        "prototype_count": np.asarray([n_prototypes], dtype=np.int32),
        "nonempty_prototype_count": np.asarray([positive_cell_counts.size], dtype=np.int32),
        "mean_cell_size": np.asarray([np.mean(positive_cell_counts) if positive_cell_counts.size else 0.0], dtype=np.float32),
        "max_cell_size": np.asarray([np.max(positive_cell_counts) if positive_cell_counts.size else 0.0], dtype=np.float32),
        "mean_candidate_count": np.asarray([np.mean(candidate_count)], dtype=np.float32),
        "median_candidate_count": np.asarray([np.median(candidate_count)], dtype=np.float32),
        "p95_candidate_count": np.asarray([np.percentile(candidate_count, 95)], dtype=np.float32),
        "max_candidate_count": np.asarray([np.max(candidate_count)], dtype=np.int32),
        "mean_prototypes_used": np.asarray([np.mean(prototypes_used)], dtype=np.float32),
        "expanded_prototype_frac": np.asarray([expanded_frac], dtype=np.float32),
        "exact_recall_samples": np.asarray([recall_diag["exact_recall_samples"]], dtype=np.float32),
        "exact_recall_at_k": np.asarray([recall_diag["exact_recall_at_k"]], dtype=np.float32),
        "gate_signal_name": np.asarray([gate_signal_name]),
        "gate_diagnostics_json": np.asarray([json.dumps(gate_diagnostics, sort_keys=True)]),
    }


def load_or_compute_prototype_indexed_profile(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
    temporal_representation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load or build the prototype-indexed approximate JG profile."""
    if config.dcs_metric_source == "oracle":
        metric_space = build_oracle_metric_space(config, dataset)
        print("PI-JG metric space: oracle (ceiling test only).")
    elif config.dcs_metric_source == "temporal":
        if temporal_representation is None:
            temporal_representation = load_or_build_temporal_representation(
                config, dataset, device=device
            )
        metric_space = np.asarray(temporal_representation["embeddings"], dtype=np.float32)
        print("PI-JG metric space: learned temporal representation.")
    else:
        metric_space = np.asarray(dataset["observations"], dtype=np.float32)
        print("PI-JG metric space: normalized observations.")

    cache_path = get_prototype_index_cache_path(config, metric_space, dataset["actions"])
    signature = build_prototype_index_signature(config, metric_space, dataset["actions"])
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))

    if cache_path is not None and cache_path.exists() and not config.bc_force_recompute:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["signature_json"].item()) == signature_json:
                    profile = {
                        key: cached[key] for key in cached.files if key != "signature_json"
                    }
                    print(f"Loaded prototype-indexed JG profile from: {cache_path}")
                    return profile
        except Exception as exc:
            print(
                f"Failed to load prototype-indexed cache {cache_path}: {exc}; "
                "recomputing."
            )

    profile = compute_prototype_indexed_profile(config, dataset, metric_space)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_payload = dict(profile)
        save_payload["signature_json"] = np.asarray(signature_json)
        np.savez_compressed(cache_path, **save_payload)
        print(f"Saved prototype-indexed JG profile to: {cache_path}")
    return profile


def build_gate_signal(
    config: TrainConfig,
    density_conf: np.ndarray,
    action_div_conf: np.ndarray,
    disp_div_conf: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Return the bounded pre-gate scalar signal J_i used to produce G_i.

    By default, a two-component Beta mixture is fitted to this signal and the
    posterior probability of the higher-mean component is used directly as G_i.
    The legacy percentile ramp remains available as an ablation.
    """
    density_conf = np.asarray(density_conf, dtype=np.float32)
    action_div_conf = np.asarray(action_div_conf, dtype=np.float32)
    disp_div_conf = np.asarray(disp_div_conf, dtype=np.float32)

    if config.dcs_gate_source == "density":
        return density_conf, "density"
    if config.dcs_gate_source == "action":
        return action_div_conf, "action_diversity"
    if config.dcs_gate_source == "displacement":
        return disp_div_conf, "displacement_diversity"
    if config.dcs_gate_source == "product_diversity":
        diversity = np.sqrt(
            np.clip(action_div_conf, 0.0, 1.0) * np.clip(disp_div_conf, 0.0, 1.0)
        ).astype(np.float32)
        return diversity, "sqrt(action_diversity * displacement_diversity)"
    if config.dcs_gate_source == "density_x_action":
        return (density_conf * action_div_conf).astype(np.float32), "density * action_diversity"
    if config.dcs_gate_source == "density_x_displacement":
        return (density_conf * disp_div_conf).astype(np.float32), "density * displacement_diversity"
    if config.dcs_gate_source == "density_x_product":
        diversity = np.sqrt(
            np.clip(action_div_conf, 0.0, 1.0) * np.clip(disp_div_conf, 0.0, 1.0)
        ).astype(np.float32)
        return (density_conf * diversity).astype(np.float32), (
            "density * sqrt(action_diversity * displacement_diversity)"
        )

    raise ValueError(
        f"Unknown dcs_gate_source={config.dcs_gate_source}. "
        f"Valid values: {DCS_GATE_SOURCES}"
    )



def _weighted_beta_moments(
    x: np.ndarray,
    weights: np.ndarray,
    eps: float,
) -> Tuple[float, float]:
    """Estimate Beta(alpha, beta) by weighted method of moments.

    This is used as the M-step inside a lightweight two-component EM routine.
    The gate signal is clipped away from exact 0/1 before fitting, so both
    shape parameters remain finite.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    mass = float(np.sum(weights))
    if mass <= 1e-12:
        return 1.0, 1.0

    mean = float(np.sum(weights * x) / mass)
    mean = float(np.clip(mean, eps, 1.0 - eps))
    var = float(np.sum(weights * (x - mean) ** 2) / mass)

    # A Beta distribution requires var < mean * (1 - mean). Keep a small
    # numerical margin and avoid an excessively concentrated component.
    max_var = max(mean * (1.0 - mean), 1e-12)
    var = float(np.clip(var, 1e-8, max_var * (1.0 - 1e-6)))
    concentration = mean * (1.0 - mean) / var - 1.0
    concentration = float(np.clip(concentration, 1e-3, 1e6))

    alpha = max(mean * concentration, eps)
    beta = max((1.0 - mean) * concentration, eps)
    return float(alpha), float(beta)


def _beta_log_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if scipy_betaln is None:
        raise ImportError(
            "dcs_gate_calibration='beta_mixture' requires scipy.special.betaln."
        )
    x = np.asarray(x, dtype=np.float64)
    return (
        (alpha - 1.0) * np.log(x)
        + (beta - 1.0) * np.log1p(-x)
        - scipy_betaln(alpha, beta)
    )


def _initialize_beta_mixture_responsibilities(
    x: np.ndarray,
    low_q: float,
    high_q: float,
) -> np.ndarray:
    """Deterministic soft initialization for the high-signal component."""
    q_lo, q_hi = np.quantile(x, [low_q, high_q])
    if q_hi <= q_lo + 1e-12:
        # Fall back to centered min-max scaling when quantiles are tied.
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max <= x_min + 1e-12:
            return np.full_like(x, 0.5, dtype=np.float64)
        return np.clip((x - x_min) / (x_max - x_min), 1e-3, 1.0 - 1e-3)
    return np.clip((x - q_lo) / (q_hi - q_lo), 1e-3, 1.0 - 1e-3)


def fit_two_component_beta_mixture_gate(
    signal: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-4,
    fit_samples: int = 200_000,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit a two-component Beta mixture and return high-component posterior G_i.

    The component with the larger Beta mean alpha / (alpha + beta) is defined
    as the junction-like component. Its posterior responsibility becomes the
    state-level gate G_i. Multiple deterministic initializations are evaluated,
    and the fit with the highest log-likelihood is retained.
    """
    if scipy_betaln is None or scipy_logsumexp is None:
        raise ImportError(
            "dcs_gate_calibration='beta_mixture' requires scipy.special."
        )

    raw = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(raw)
    if not np.all(finite_mask):
        raise ValueError("Non-finite values found in JG-IQL gate signal.")
    if raw.size < 2:
        return np.zeros_like(raw, dtype=np.float32), {
            "status": "degenerate_too_few_samples",
            "fit_size": int(raw.size),
        }

    # J_i is theoretically bounded in [0, 1]. Clip only for Beta log-density;
    # the original un-clipped values are retained for diagnostics.
    x_all = np.clip(raw, eps, 1.0 - eps)
    raw_range = float(np.max(raw) - np.min(raw))
    if raw_range <= 1e-10:
        return np.zeros_like(raw, dtype=np.float32), {
            "status": "degenerate_constant_signal",
            "fit_size": int(raw.size),
            "signal_min": float(np.min(raw)),
            "signal_max": float(np.max(raw)),
        }

    # Fitting on a capped random subset keeps preprocessing practical for very
    # large offline datasets; posterior inference is still performed on all N.
    if x_all.size > int(fit_samples):
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(x_all.size, size=int(fit_samples), replace=False)
        x_fit = x_all[fit_idx]
    else:
        x_fit = x_all

    # Different soft quantile splits make the fit less sensitive to one
    # initialization while staying deterministic.
    init_quantiles = ((0.25, 0.75), (0.35, 0.65), (0.45, 0.55))
    best = None

    for low_q, high_q in init_quantiles:
        r_high = _initialize_beta_mixture_responsibilities(x_fit, low_q, high_q)
        responsibilities = np.stack([1.0 - r_high, r_high], axis=1)
        prev_ll = -np.inf

        for iteration in range(int(max_iter)):
            mixture_weights = np.mean(responsibilities, axis=0)
            mixture_weights = np.clip(mixture_weights, 1e-6, 1.0)
            mixture_weights /= np.sum(mixture_weights)

            params = [
                _weighted_beta_moments(x_fit, responsibilities[:, k], eps)
                for k in range(2)
            ]

            log_joint = np.empty((x_fit.shape[0], 2), dtype=np.float64)
            for k, (alpha, beta) in enumerate(params):
                log_joint[:, k] = (
                    np.log(mixture_weights[k]) + _beta_log_pdf(x_fit, alpha, beta)
                )

            log_norm = scipy_logsumexp(log_joint, axis=1)
            ll = float(np.sum(log_norm))
            responsibilities = np.exp(log_joint - log_norm[:, None])

            if np.isfinite(prev_ll):
                relative_change = abs(ll - prev_ll) / max(abs(prev_ll), 1.0)
                if relative_change < tol:
                    break
            prev_ll = ll

        means = np.asarray(
            [alpha / (alpha + beta) for alpha, beta in params], dtype=np.float64
        )
        candidate = {
            "ll": ll,
            "iterations": int(iteration + 1),
            "weights": mixture_weights.copy(),
            "params": tuple(params),
            "means": means,
        }
        if best is None or candidate["ll"] > best["ll"]:
            best = candidate

    assert best is not None
    high_component = int(np.argmax(best["means"]))
    low_component = 1 - high_component

    # Infer posterior responsibilities for every dataset state.
    log_joint_all = np.empty((x_all.shape[0], 2), dtype=np.float64)
    for k, (alpha, beta) in enumerate(best["params"]):
        log_joint_all[:, k] = (
            np.log(best["weights"][k]) + _beta_log_pdf(x_all, alpha, beta)
        )
    log_norm_all = scipy_logsumexp(log_joint_all, axis=1)
    posterior = np.exp(log_joint_all - log_norm_all[:, None])
    gate = np.clip(posterior[:, high_component], 0.0, 1.0).astype(np.float32)

    low_alpha, low_beta = best["params"][low_component]
    high_alpha, high_beta = best["params"][high_component]
    diagnostics = {
        "status": "ok",
        "fit_size": int(x_fit.size),
        "full_size": int(x_all.size),
        "iterations": int(best["iterations"]),
        "log_likelihood": float(best["ll"]),
        "low_weight": float(best["weights"][low_component]),
        "high_weight": float(best["weights"][high_component]),
        "low_alpha": float(low_alpha),
        "low_beta": float(low_beta),
        "high_alpha": float(high_alpha),
        "high_beta": float(high_beta),
        "low_mean": float(best["means"][low_component]),
        "high_mean": float(best["means"][high_component]),
        "mean_gap": float(
            best["means"][high_component] - best["means"][low_component]
        ),
        "posterior_mean": float(np.mean(gate)),
        "posterior_gt_05_frac": float(np.mean(gate >= 0.5)),
        "signal_min": float(np.min(raw)),
        "signal_max": float(np.max(raw)),
    }
    return gate, diagnostics


def build_data_adaptive_gate(
    config: TrainConfig,
    gate_signal: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build G_i using either the automatic Beta mixture or legacy percentile ramp."""
    if config.dcs_gate_calibration == "percentile":
        gate = covp.gate_from_junction(
            gate_signal,
            config.dcs_gate_low_percentile,
            config.dcs_gate_high_percentile,
        ).astype(np.float32)
        return gate, {
            "status": "legacy_percentile",
            "low_percentile": float(config.dcs_gate_low_percentile),
            "high_percentile": float(config.dcs_gate_high_percentile),
        }

    try:
        gate, diagnostics = fit_two_component_beta_mixture_gate(
            gate_signal,
            max_iter=config.dcs_beta_mixture_max_iter,
            tol=config.dcs_beta_mixture_tol,
            eps=config.dcs_beta_mixture_eps,
            fit_samples=config.dcs_beta_mixture_fit_samples,
            seed=config.seed,
        )
    except Exception as exc:
        diagnostics = {
            "status": "fit_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        gate = np.zeros_like(np.asarray(gate_signal).reshape(-1), dtype=np.float32)

    if diagnostics.get("status") != "ok":
        if config.dcs_beta_mixture_fallback == "percentile":
            gate = covp.gate_from_junction(
                gate_signal,
                config.dcs_gate_low_percentile,
                config.dcs_gate_high_percentile,
            ).astype(np.float32)
            diagnostics["fallback"] = "percentile"
            diagnostics["fallback_low_percentile"] = float(
                config.dcs_gate_low_percentile
            )
            diagnostics["fallback_high_percentile"] = float(
                config.dcs_gate_high_percentile
            )
        else:
            gate = np.zeros_like(np.asarray(gate_signal).reshape(-1), dtype=np.float32)
            diagnostics["fallback"] = "zero"

    return gate, diagnostics



def build_prototype_indexed_replay_fields(
    config: TrainConfig,
    profile: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create minimal ReplayBuffer fields plus the actual state gate G_i.

    The prototype-indexed neighbor search is used only during preprocessing to
    compute JG-IQL statistics. The trainer itself still reads only
    batch["junction"] when constructing tau(s), so no N x k graph is kept in
    the ReplayBuffer.
    """
    gate = np.asarray(profile["gate"], dtype=np.float32).reshape(-1)
    n = gate.shape[0]
    dummy_indices = np.arange(n, dtype=np.int32)[:, None]
    dummy_weights = gate[:, None].astype(np.float32)

    density_conf = np.asarray(profile["density_conf"], dtype=np.float32)
    action_div_conf = np.asarray(profile["action_div_conf"], dtype=np.float32)
    disp_div_conf = np.asarray(profile["disp_div_conf"], dtype=np.float32)
    gate_signal = np.asarray(profile["gate_signal"], dtype=np.float32)

    try:
        gate_diagnostics = json.loads(str(profile["gate_diagnostics_json"].reshape(-1)[0]))
    except Exception:
        gate_diagnostics = {"status": "unavailable"}

    print(
        "PI-JG support profile | "
        f"gate_source={config.dcs_gate_source} | "
        f"gate_calibration={config.dcs_gate_calibration} | "
        f"mean_gate={float(np.mean(gate)):.4f} | "
        f"active_frac={float(np.mean(gate > 1e-6)):.4f} | "
        f"k={config.dcs_k}"
    )
    print(
        "Prototype-index diagnostics | "
        f"prototypes={int(np.asarray(profile['prototype_count']).reshape(-1)[0])} | "
        f"nonempty={int(np.asarray(profile['nonempty_prototype_count']).reshape(-1)[0])} | "
        f"mean_cell={float(np.asarray(profile['mean_cell_size']).reshape(-1)[0]):.1f} | "
        f"max_cell={float(np.asarray(profile['max_cell_size']).reshape(-1)[0]):.0f} | "
        f"candidate_mean={float(np.asarray(profile['mean_candidate_count']).reshape(-1)[0]):.1f} | "
        f"candidate_p95={float(np.asarray(profile['p95_candidate_count']).reshape(-1)[0]):.1f} | "
        f"candidate_max={float(np.asarray(profile['max_candidate_count']).reshape(-1)[0]):.0f} | "
        f"mean_prototypes_used={float(np.asarray(profile['mean_prototypes_used']).reshape(-1)[0]):.2f} | "
        f"expanded_frac={float(np.asarray(profile['expanded_prototype_frac']).reshape(-1)[0]):.4f}"
    )
    recall_samples = float(np.asarray(profile["exact_recall_samples"]).reshape(-1)[0])
    if recall_samples > 0:
        print(
            "Approximation diagnostic | "
            f"exact_recall_samples={int(recall_samples)} | "
            f"recall@{config.dcs_k}="
            f"{float(np.asarray(profile['exact_recall_at_k']).reshape(-1)[0]):.4f}"
        )
    print(
        "Branch statistics | "
        f"action_conf_mean={float(np.mean(action_div_conf)):.4f} | "
        f"disp_conf_mean={float(np.mean(disp_div_conf)):.4f} | "
        f"density_conf_mean={float(np.mean(density_conf)):.4f} | "
        f"gate_signal_mean={float(np.mean(gate_signal)):.4f}"
    )
    if gate_diagnostics.get("status") == "ok":
        print(
            "Beta-mixture gate | "
            f"low_mean={gate_diagnostics.get('low_mean', float('nan')):.4f} | "
            f"high_mean={gate_diagnostics.get('high_mean', float('nan')):.4f} | "
            f"mean_gap={gate_diagnostics.get('mean_gap', float('nan')):.4f} | "
            f"P(high)>=0.5 frac="
            f"{gate_diagnostics.get('posterior_gt_05_frac', float('nan')):.4f}"
        )
    else:
        print(
            "Gate calibration | "
            f"status={gate_diagnostics.get('status')} | "
            f"fallback={gate_diagnostics.get('fallback', 'none')}"
        )

    return dummy_indices, dummy_weights, gate


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
    """Replay buffer with minimal compatibility fields for the precomputed state gate.

    For every stored transition i it additionally holds:
      - neighbor_indices[i]: (k,) int32 indices of i's nearest dataset states
      - neighbor_weights[i]: (k,) float32 legacy diagnostic weights in [0, 1]
      - junction[i]:         (1,) float32 state gate G_i used as support confidence base

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
        embedding_dim: int = 0,
    ):
        self._buffer_size = buffer_size
        self._neighbor_k = int(neighbor_k)
        self._embedding_dim = int(embedding_dim)
        if self._embedding_dim < 0:
            raise ValueError("embedding_dim must be >= 0")
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self._embeddings = np.zeros((buffer_size, self._embedding_dim), dtype=np.float32)
        self._next_embeddings = np.zeros((buffer_size, self._embedding_dim), dtype=np.float32)
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
        embeddings: Optional[np.ndarray] = None,
        next_embeddings: Optional[np.ndarray] = None,
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

        if self._embedding_dim > 0:
            if embeddings is None or next_embeddings is None:
                raise ValueError(
                    "embedding_dim > 0 requires both embeddings and next_embeddings"
                )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            next_embeddings = np.asarray(next_embeddings, dtype=np.float32)
            expected_shape = (n_transitions, self._embedding_dim)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"embeddings must have shape {expected_shape}, got {embeddings.shape}"
                )
            if next_embeddings.shape != expected_shape:
                raise ValueError(
                    f"next_embeddings must have shape {expected_shape}, got {next_embeddings.shape}"
                )
            self._embeddings[:n_transitions] = embeddings
            self._next_embeddings[:n_transitions] = next_embeddings

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
        print(f"Dataset size: {n_transitions} (compatibility neighbor axis={self._neighbor_k}; BC-JG uses no kNN graph)")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        nbr_idx = self._neighbor_indices[indices]  # (B, k)
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
            "embeddings": self._embeddings[indices],
            "next_embeddings": self._next_embeddings[indices],
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


class PrototypeIndexedJunctionIQLJAX:
    """Prototype-Indexed Approximate Junction-Gated IQL in JAX/Flax.

    Q-function training and actor extraction mirror standard IQL. The only
    departure is value learning: tau is state-dependent and is computed from
    JG-IQL statistics using prototype-retrieved candidate pools followed by
    exact local top-k refinement.
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
        embedding_dim: int = 0,
        actor_use_embedding: bool = False,
        q_use_embedding: bool = False,
        v_use_embedding: bool = False,
        temporal_encoder_params: Optional[Any] = None,
        temporal_embedding_mean: Optional[np.ndarray] = None,
        temporal_embedding_std: Optional[np.ndarray] = None,
        temporal_hidden_dim: int = 256,
        temporal_n_hidden: int = 2,
        temporal_normalize: bool = True,
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
        self.embedding_dim = int(embedding_dim)
        self.actor_use_embedding = bool(actor_use_embedding)
        self.q_use_embedding = bool(q_use_embedding)
        self.v_use_embedding = bool(v_use_embedding)
        self.uses_embedding_input = (
            self.actor_use_embedding or self.q_use_embedding or self.v_use_embedding
        )
        self.device = device if device is not None else jax.devices()[0]

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be > 0")
        if self.uses_embedding_input and self.embedding_dim <= 0:
            raise ValueError(
                "At least one network uses embeddings, but embedding_dim <= 0."
            )
        if self.uses_embedding_input and temporal_encoder_params is None:
            raise ValueError(
                "Embedding-based network inputs require frozen temporal encoder parameters."
            )

        self.temporal_encoder_def = None
        self.temporal_encoder_params = None
        self.temporal_embedding_mean = None
        self.temporal_embedding_std = None
        if self.uses_embedding_input:
            self.temporal_encoder_def = tmet.TemporalEncoder(
                embed_dim=self.embedding_dim,
                hidden_dim=int(temporal_hidden_dim),
                n_hidden=int(temporal_n_hidden),
                normalize=bool(temporal_normalize),
            )
            encoder_template = self.temporal_encoder_def.init(
                jax.random.PRNGKey(seed + 100003),
                jnp.zeros((1, state_dim), dtype=jnp.float32),
            )["params"]
            self.temporal_encoder_params = serialization.from_state_dict(
                encoder_template,
                temporal_encoder_params,
            )
            if temporal_embedding_mean is None:
                temporal_embedding_mean = np.zeros((self.embedding_dim,), dtype=np.float32)
            if temporal_embedding_std is None:
                temporal_embedding_std = np.ones((self.embedding_dim,), dtype=np.float32)
            self.temporal_embedding_mean = jnp.asarray(
                np.asarray(temporal_embedding_mean, dtype=np.float32).reshape(1, -1)
            )
            self.temporal_embedding_std = jnp.asarray(
                np.asarray(temporal_embedding_std, dtype=np.float32).reshape(1, -1)
            )
            if self.temporal_embedding_mean.shape[-1] != self.embedding_dim:
                raise ValueError("temporal_embedding_mean has the wrong dimension")
            if self.temporal_embedding_std.shape[-1] != self.embedding_dim:
                raise ValueError("temporal_embedding_std has the wrong dimension")

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
        actor_input_dim = self.embedding_dim if self.actor_use_embedding else state_dim
        q_input_dim = self.embedding_dim if self.q_use_embedding else state_dim
        v_input_dim = self.embedding_dim if self.v_use_embedding else state_dim
        dummy_actor_state = jnp.zeros((1, actor_input_dim), dtype=jnp.float32)
        dummy_q_state = jnp.zeros((1, q_input_dim), dtype=jnp.float32)
        dummy_v_state = jnp.zeros((1, v_input_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        actor_params = self.actor_def.init(
            key_actor, dummy_actor_state, training=False
        )["params"]
        q_params = self.q_def.init(key_q, dummy_q_state, dummy_action)["params"]
        v_params = self.v_def.init(key_v, dummy_v_state)["params"]

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
        if self.uses_embedding_input:
            self.temporal_encoder_params = tree_to_device(
                self.temporal_encoder_params, self.device
            )
            self.temporal_embedding_mean = tree_to_device(
                self.temporal_embedding_mean, self.device
            )
            self.temporal_embedding_std = tree_to_device(
                self.temporal_embedding_std, self.device
            )
        self._train_step = self._build_train_step()
        self._actor_refit_step = self._build_actor_refit_step()

    def _encode_observations(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Apply the frozen temporal encoder and the training-time standardizer."""
        if not self.uses_embedding_input:
            raise RuntimeError("Temporal encoder requested but no network uses embeddings.")
        z = self.temporal_encoder_def.apply(
            {"params": self.temporal_encoder_params}, observations
        )
        return (z - self.temporal_embedding_mean) / self.temporal_embedding_std

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
        actor_use_embedding = self.actor_use_embedding
        q_use_embedding = self.q_use_embedding
        v_use_embedding = self.v_use_embedding
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
            embeddings = batch["embeddings"]
            next_embeddings = batch["next_embeddings"]
            dones = jnp.squeeze(batch["dones"], axis=-1)
            neighbor_weights = jnp.clip(batch["neighbor_weights"], 0.0, 1.0)

            q_observations = embeddings if q_use_embedding else observations
            v_observations = embeddings if v_use_embedding else observations
            next_v_observations = (
                next_embeddings if v_use_embedding else next_observations
            )
            actor_observations = (
                embeddings if actor_use_embedding else observations
            )

            # Junction-gated certificate c(s).
            # BC-JG-IQL does NOT compute a kNN graph or kernel ESS. The ReplayBuffer stores the
            # state-level gate G_i in the legacy "junction" field.
            gate_conf = jnp.clip(jnp.squeeze(batch["junction"], axis=-1), 0.0, 1.0)
            support_conf = gate_conf ** support_power
            tau_s = tau_min + (tau_max - tau_min) * support_conf

            # Legacy diagnostics only. They are not used to compute tau(s).
            k = jnp.maximum(jnp.asarray(neighbor_weights.shape[1], dtype=jnp.float32), 1.0)
            neighbor_mass = jnp.sum(neighbor_weights, axis=1)
            gate_effective_count_proxy = gate_conf * k

            tq1, tq2 = q_apply(
                {"params": state.q_target_params}, q_observations, actions
            )
            self_target_q = jnp.minimum(tq1, tq2)

            next_v = v_apply({"params": state.v_params}, next_v_observations)
            target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
            old_v = v_apply({"params": state.v_params}, v_observations)
            adv = self_target_q - old_v
            exp_adv = jnp.minimum(
                jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
            )

            # ----- Support-certified expectile V update.
            def v_loss_fn(v_params):
                v = v_apply({"params": v_params}, v_observations)
                diff = jax.lax.stop_gradient(self_target_q) - v
                expectile_weight = jnp.abs(tau_s - (diff < 0.0).astype(jnp.float32))
                value_loss = jnp.mean(expectile_weight * diff ** 2)
                return value_loss, v

            (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            # ----- Q update: standard IQL TD backup.
            def q_loss_fn(q_params):
                q1, q2 = q_apply({"params": q_params}, q_observations, actions)
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
                policy_out = apply_actor(
                    actor_params,
                    actor_observations,
                    training=True,
                    rng=dropout_key,
                )
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
                # JG diagnostics
                "support_conf_mean": jnp.mean(support_conf),
                "support_conf_min": jnp.min(support_conf),
                "support_conf_max": jnp.max(support_conf),
                "tau_s_mean": jnp.mean(tau_s),
                "tau_s_min": jnp.min(tau_s),
                "tau_s_max": jnp.max(tau_s),
                # JG diagnostics. No ESS is computed; the effective-count value is a
                # gate-based proxy kept for W&B compatibility across ablations.
                "gate_conf_mean": jnp.mean(gate_conf),
                "gate_conf_min": jnp.min(gate_conf),
                "gate_conf_max": jnp.max(gate_conf),
                "gate_effective_count_proxy_mean": jnp.mean(gate_effective_count_proxy),
                "neighbor_effective_count_mean": jnp.mean(gate_effective_count_proxy),
                "neighbor_mass_mean": jnp.mean(neighbor_mass),
                "neighbor_weight_mean": jnp.mean(neighbor_weights),
                "gate_active_frac": jnp.mean(
                    (gate_conf > 0.0).astype(jnp.float32)
                ),
                "active_edge_frac": jnp.mean(
                    (neighbor_weights > 0.0).astype(jnp.float32)
                ),
            }
            return new_state, log_dict

        return train_step

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.state, log_dict = self._train_step(self.state, batch)
        return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

    def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
        state_jnp = tree_to_device(
            jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device
        )
        actor_input = (
            self._encode_observations(state_jnp)
            if self.actor_use_embedding
            else state_jnp
        )
        policy_out = self._apply_actor(actor_params, actor_input, training=False)
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
        actor_use_embedding = self.actor_use_embedding
        q_use_embedding = self.q_use_embedding
        v_use_embedding = self.v_use_embedding
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
            embeddings = batch["embeddings"]
            actions = batch["actions"]

            q_observations = embeddings if q_use_embedding else observations
            v_observations = embeddings if v_use_embedding else observations
            actor_observations = (
                embeddings if actor_use_embedding else observations
            )

            tq1, tq2 = q_apply(
                {"params": iql_state.q_target_params}, q_observations, actions
            )
            target_q = jnp.minimum(tq1, tq2)
            v = v_apply({"params": iql_state.v_params}, v_observations)
            adv = target_q - v
            exp_adv = jnp.minimum(
                jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
            )

            actor_key, dropout_key = jax.random.split(actor_state.key)

            def actor_loss_fn(actor_params):
                policy_out = apply_actor(
                    actor_params,
                    actor_observations,
                    training=True,
                    rng=dropout_key,
                )
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
        payload = {
            "iql_state": serialization.to_state_dict(self.state),
            "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
            "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
            "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
            "iql_deterministic": self.iql_deterministic,
            "actor_use_embedding": self.actor_use_embedding,
            "q_use_embedding": self.q_use_embedding,
            "v_use_embedding": self.v_use_embedding,
            "embedding_dim": self.embedding_dim,
        }
        if self.uses_embedding_input:
            payload["temporal_encoder_params"] = serialization.to_state_dict(
                self.temporal_encoder_params
            )
            payload["temporal_embedding_mean"] = np.asarray(
                jax.device_get(self.temporal_embedding_mean), dtype=np.float32
            )
            payload["temporal_embedding_std"] = np.asarray(
                jax.device_get(self.temporal_embedding_std), dtype=np.float32
            )
        return payload

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
        if self.uses_embedding_input and "temporal_encoder_params" in state_dict:
            self.temporal_encoder_params = serialization.from_state_dict(
                self.temporal_encoder_params,
                state_dict["temporal_encoder_params"],
            )
        if self.uses_embedding_input and "temporal_embedding_mean" in state_dict:
            self.temporal_embedding_mean = jnp.asarray(
                state_dict["temporal_embedding_mean"], dtype=jnp.float32
            )
        if self.uses_embedding_input and "temporal_embedding_std" in state_dict:
            self.temporal_embedding_std = jnp.asarray(
                state_dict["temporal_embedding_std"], dtype=jnp.float32
            )
        self.state = tree_to_device(self.state, self.device)
        self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
        self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
        self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
        if self.uses_embedding_input:
            self.temporal_encoder_params = tree_to_device(
                self.temporal_encoder_params, self.device
            )
            self.temporal_embedding_mean = tree_to_device(
                self.temporal_embedding_mean, self.device
            )
            self.temporal_embedding_std = tree_to_device(
                self.temporal_embedding_std, self.device
            )


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
@log_training_exceptions
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
            checkpoint_type="pi_jg_iql_jax_training_progress",
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

    network_uses_embedding = (
        config.actor_use_embedding or config.q_use_embedding or config.v_use_embedding
    )
    temporal_representation: Optional[Dict[str, Any]] = None
    if config.dcs_metric_source == "temporal" and (config.use_dcs or network_uses_embedding):
        temporal_representation = load_or_build_temporal_representation(
            config, dataset, device=jax_device
        )

    # ----- PI-JG profile: prototype retrieval + exact local top-k refinement ---
    neighbor_indices = None
    neighbor_weights = None
    junction = None
    buffer_neighbor_k = 1
    if config.use_dcs:
        profile = load_or_compute_prototype_indexed_profile(
            config,
            dataset,
            device=jax_device,
            temporal_representation=temporal_representation,
        )
        neighbor_indices, neighbor_weights, junction = build_prototype_indexed_replay_fields(
            config, profile
        )
    else:
        print(
            "use_dcs=False: prototype-indexed gate disabled; tau(s)=scv_tau_min. "
            "Set scv_tau_min=scv_tau_max for a plain IQL baseline."
        )

    embedding_dim = (
        int(temporal_representation["embeddings"].shape[1])
        if temporal_representation is not None and network_uses_embedding
        else 0
    )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        neighbor_k=buffer_neighbor_k,
        device=jax_device,
        embedding_dim=embedding_dim,
    )
    replay_buffer.load_d4rl_dataset(
        dataset,
        neighbor_indices=neighbor_indices,
        neighbor_weights=neighbor_weights,
        junction=junction,
        embeddings=(
            temporal_representation["embeddings"]
            if temporal_representation is not None and network_uses_embedding
            else None
        ),
        next_embeddings=(
            temporal_representation["next_embeddings"]
            if temporal_representation is not None and network_uses_embedding
            else None
        ),
    )

    max_action = float(env.action_space.high[0])


    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    run_mode_name = "Actor refit" if refit_only else "Training"
    print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
    print(
        f"support_tau=[{config.scv_tau_min}, {config.scv_tau_max}] | use_prototype_index={config.use_dcs} | "
        f"prototypes={config.bc_num_prototypes} | near_prototypes={config.bc_nearest_prototypes} | "
        f"top_k={config.dcs_k} | "
        f"support_power={config.scv_support_power} | metric={config.dcs_metric_source}"
    )
    print(
        "network inputs | "
        f"actor={'embedding' if config.actor_use_embedding else 'observation'} | "
        f"Q={'embedding' if config.q_use_embedding else 'observation'} | "
        f"V={'embedding' if config.v_use_embedding else 'observation'}"
    )
    if config.use_dcs:
        print("support confidence: c_i = G_i^p; prototype index only accelerates neighbor retrieval; no neighbor Q/action pooling")
        if config.dcs_metric_source == "oracle":
            keys = resolve_oracle_keys(config)
            print(
                f"*** ORACLE CEILING TEST: prototype index built in ORACLE space "
                f"(keys={list(keys) if keys else 'auto'}); training still uses "
                f"standard observations. ***"
            )
        elif config.dcs_metric_source == "temporal":
            print(
                f"*** TEMPORAL METRIC: prototype index built in a LEARNED temporal "
                f"space phi(s) (dim={config.dcs_temporal_dim}, "
                f"horizon={config.dcs_temporal_horizon}, "
                f"steps={config.dcs_temporal_steps}); data-only, deployable. "
                f"Each of actor/Q/V can independently consume phi(s). ***"
            )
        else:
            print("metric source: observation (prototype retrieval + exact local top-k)")
    print("---------------------------------------")

    trainer = PrototypeIndexedJunctionIQLJAX(
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
        embedding_dim=embedding_dim,
        actor_use_embedding=config.actor_use_embedding,
        q_use_embedding=config.q_use_embedding,
        v_use_embedding=config.v_use_embedding,
        temporal_encoder_params=(
            temporal_representation["encoder_params"]
            if temporal_representation is not None and network_uses_embedding
            else None
        ),
        temporal_embedding_mean=(
            temporal_representation["embedding_mean"]
            if temporal_representation is not None and network_uses_embedding
            else None
        ),
        temporal_embedding_std=(
            temporal_representation["embedding_std"]
            if temporal_representation is not None and network_uses_embedding
            else None
        ),
        temporal_hidden_dim=config.dcs_temporal_hidden,
        temporal_n_hidden=config.dcs_temporal_layers,
        temporal_normalize=config.dcs_temporal_normalize,
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
        print(f"Actor refit from saved checkpoint ({ALGORITHM_NAME} standard IQL AWR)")
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


# # pi_jg_iql_jax.py
# #
# # PI-JG-IQL: Prototype-Indexed Approximate Junction-Gated Implicit Q-Learning
# # (JAX/Flax).
# #
# # Efficient, locality-preserving approximation of JG-IQL. Instead of building
# # a full-dataset exact kNN structure, this variant uses state-space prototypes
# # only as a coarse inverted index:
# #
# #   1) fit M prototypes c_m in the selected metric space,
# #   2) assign every dataset transition to its nearest prototype,
# #   3) for each anchor s_i, retrieve members of the r nearest prototype cells,
# #   4) compute the exact top-k nearest transitions INSIDE that candidate pool,
# #   5) compute the same JG-IQL local statistics from those actual top-k neighbors.
# #
# # Thus prototypes do NOT define diversity and no branch-biased coreset is used.
# # They only reduce the search space before an exact local top-k refinement.
# #
# # For each state i:
# #   A_i : RMS spread of actions among the retrieved top-k neighbors,
# #   P_i : directional dispersion of their next-state displacements,
# #   J_i : configurable gate signal, by default sqrt(A_i * P_i),
# #   G_i : percentile ramp or automatic Beta-mixture calibration,
# #   tau_i = tau_min + (tau_max - tau_min) * G_i ** p.
# #
# # Q-learning and actor extraction remain standard IQL.
# #
# import copy
# import hashlib
# import json
# import os
# import pickle
# import random
# import sys
# import uuid
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union

# import gym
# import jax
# import jax.numpy as jnp
# import numpy as np

# try:
#     import scipy.linalg as scipy_linalg
#     from scipy.special import betaln as scipy_betaln
#     from scipy.special import logsumexp as scipy_logsumexp

#     if not hasattr(scipy_linalg, "tril"):
#         scipy_linalg.tril = np.tril
#     if not hasattr(scipy_linalg, "triu"):
#         scipy_linalg.triu = np.triu
# except ImportError:
#     scipy_betaln = None
#     scipy_logsumexp = None

# import optax
# import pyrallis
# import yaml

# try:
#     import wandb
# except ImportError:
#     class _UnavailableWandb:
#         run = None

#         def init(self, *args, **kwargs):
#             raise ImportError(
#                 "wandb is unavailable in this environment; run with --log_wandb False "
#                 "or install wandb with its dependencies."
#             )

#         def save(self, *args, **kwargs):
#             return None

#         def log(self, *args, **kwargs):
#             return None

#     wandb = _UnavailableWandb()

# from flax import linen as nn
# from flax import serialization, struct

# import coverage_profile as covp
# import temporal_metric as tmet

# d4rl = None

# try:
#     import ogbench
# except ImportError:
#     ogbench = None

# TensorBatch = Dict[str, jnp.ndarray]

# ALGORITHM_NAME = "PI-JG-IQL"
# ALGORITHM_FULL_NAME = "Prototype-Indexed Approximate Junction-Gated Implicit Q-Learning"

# EXP_ADV_MAX = 100.0
# LOG_STD_MIN = -20.0
# LOG_STD_MAX = 2.0

# DCS_GATE_SOURCES = (
#     "density",
#     "action",
#     "displacement",
#     "product_diversity",
#     "density_x_action",
#     "density_x_displacement",
#     "density_x_product",
# )


# @dataclass
# class TrainConfig:
#     # Experiment
#     device: str = "gpu"  # one of: cpu, gpu, tpu. JAX selects the matching backend when available.
#     env: str = "halfcheetah-medium-expert-v2"
#     seed: int = 0

#     # Shared by both modes:
#     #   mode="train": evaluate pi every eval_freq joint Q/V/pi training steps.
#     #   mode="refit": evaluate pi every eval_freq actor-only refit steps.
#     eval_freq: int = int(25e3)
#     n_episodes: int = 10

#     # Shared by both modes:
#     #   mode="train": number of joint Q/V/pi training steps.
#     #   mode="refit": number of actor-only refit steps using frozen loaded Q/V.
#     max_timesteps: int = int(1e6)

#     checkpoints_path: Optional[str] = None
#     load_model: str = ""
#     mode: str = "train"  # one of: train, refit. refit loads Q/V and trains only pi.
#     hyperparams_path: Optional[str] = "hyperparams/dcs_iql_jax.yml"
#     use_hyperparams: bool = True

#     # Dataset
#     buffer_size: int = 2_000_000

#     # Shared by both modes:
#     #   mode="train": minibatch size for joint Q/V/pi updates.
#     #   mode="refit": minibatch size for actor-only refit updates.
#     batch_size: int = 256

#     normalize: bool = True
#     normalize_reward: bool = False

#     # IQL core. iql_tau is GLOBAL and fixed by design (see header).
#     discount: float = 0.99
#     tau: float = 0.005
#     beta: float = 3.0
#     iql_tau: float = 0.7
#     iql_deterministic: bool = False

#     # ----- PI-JG-IQL: prototype-indexed approximate JG value learning --------
#     # Neighbor Q/action pooling is not used. A state gate G_i controls
#     # c(s), which in turn controls the expectile asymmetry in V-learning.
#     # The old scv_* names are kept as CLI-compatible aliases.
#     scv_tau_min: float = 0.5
#     scv_tau_max: float = 0.8
#     scv_support_power: float = 1.0
#     scv_support_eps: float = 1e-8

#     # ----- Branching gate ------------------------------------------------
#     # Master switch. False -> no prototype-indexed preprocessing and
#     # tau(s)=scv_tau_min, equivalent to standard IQL when
#     # scv_tau_min == scv_tau_max.
#     use_dcs: bool = True

#     # Final top-k used for JG statistics. Prototype indexing only accelerates retrieval.
#     dcs_k: int = 8

#     # Legacy compatibility only. JG-IQL never pools neighbor Q-values or
#     # neighbor actions; these are ignored and trainer receives 0.0 internally.
#     dcs_value_neighbor_weight: float = 0.0
#     dcs_actor_neighbor_weight: float = 0.0

#     # Junction/gate signal source used to build the state gate G_i.
#     # This mirrors KES-G-IQL's ablation knob, but JG-IQL uses the resulting
#     # state-level gate directly as c(s), instead of computing kernel ESS.
#     #   density                : G_i from density confidence only
#     #   action                 : G_i from action-spread diversity only
#     #   displacement           : G_i from next-displacement diversity only
#     #   product_diversity      : G_i from sqrt(action_div * disp_div), no density
#     #   density_x_action       : G_i from density * action diversity
#     #   density_x_displacement : G_i from density * displacement diversity
#     #   density_x_product      : G_i from density * sqrt(action_div * disp_div)
#     dcs_gate_source: str = "product_diversity"

#     # Backward-compatible alias used only when old hyperparameter files/scripts
#     # still pass --dcs_diversity_mode. If dcs_gate_source is not explicitly
#     # overridden, apply_env_hyperparams maps dcs_diversity_mode to the matching
#     # density_x_* gate source.
#     dcs_diversity_mode: str = "product"

#     # Percentile scaling of raw branch statistics into [0, 1] confidences.
#     dcs_percentile_low: float = 5.0
#     dcs_percentile_high: float = 95.0

#     # ----- Gate calibration -----------------------------------------------
#     # beta_mixture (default): fit a two-component Beta mixture to the bounded
#     # gate signal J_i in [0, 1]. The component with the larger mean is treated
#     # as the junction-like population, and its posterior probability is used
#     # directly as the state gate:
#     #
#     #   G_i = P(z_i = high-mean component | J_i).
#     #
#     # This removes the need to manually tune low/high gate percentiles.
#     # percentile: legacy ramp gate kept only for backward-compatible ablations.
#     dcs_gate_calibration: str = "beta_mixture"  # beta_mixture | percentile

#     # Legacy percentile-ramp settings. Ignored when dcs_gate_calibration is
#     # "beta_mixture", except when dcs_beta_mixture_fallback="percentile".
#     dcs_gate_low_percentile: float = 60.0
#     dcs_gate_high_percentile: float = 95.0

#     # Numerical controls for the automatic two-component Beta mixture. These
#     # are optimization safeguards rather than task-level gate thresholds.
#     dcs_beta_mixture_max_iter: int = 200
#     dcs_beta_mixture_tol: float = 1e-6
#     dcs_beta_mixture_eps: float = 1e-4
#     dcs_beta_mixture_fit_samples: int = 200_000
#     # Degenerate distributions have no identifiable two-population structure.
#     # "zero" is conservative and fully automatic; "percentile" recovers the
#     # old manually specified ramp as a compatibility fallback.
#     dcs_beta_mixture_fallback: str = "zero"  # zero | percentile

#     # ----- Prototype-indexed approximate kNN -------------------------------
#     # Prototypes are used ONLY as a coarse inverted index. Every state is
#     # assigned to one prototype cell. For each anchor, members of the nearest
#     # prototype cells form a candidate pool, and the exact top-k nearest states
#     # are selected from that pool using the same metric space as JG-IQL.
#     #
#     # Existing bc_* names are retained for CLI/YAML compatibility with the
#     # previous BC-JG-IQL implementation, even though no branch coreset remains.

#     # Number of state-space prototypes. Larger values make cells smaller and
#     # improve locality, but increase prototype fitting / lookup cost.
#     bc_num_prototypes: int = 512

#     # Prototype fitting uses at most this many states and streaming mini-batch
#     # k-means. The final approximate top-k profile is still computed for every
#     # dataset state.
#     bc_prototype_fit_samples: int = 100_000
#     bc_kmeans_iters: int = 10
#     bc_kmeans_batch_size: int = 4096

#     # Initial number r of nearest prototype cells consulted for each state.
#     # If those cells do not contain at least k non-self candidates, additional
#     # prototype cells are added automatically until enough candidates exist.
#     bc_nearest_prototypes: int = 2

#     # Chunk size for all-state prototype lookup / approximate top-k inference.
#     # Neighbor refinement itself uses variable-size candidate pools per state.
#     bc_profile_chunk_size: int = 2048

#     # Preprocessing seed, independent of RL minibatch sampling.
#     bc_preprocess_seed: int = 0

#     # Optional diagnostic: on this many random anchors, compute full-dataset
#     # exact kNN and report recall@k of the prototype-indexed approximation.
#     # Set 0 to disable because the diagnostic itself is expensive.
#     bc_exact_recall_samples: int = 0

#     # Cache controls. The cache stores only final per-state statistics / gate and
#     # diagnostics; no full N x k graph is kept in the ReplayBuffer.
#     bc_force_recompute: bool = False

#     # Distance kernel exp(-(d/h)^2) with per-state bandwidth
#     # h_i = dcs_bandwidth_scale * median_j d_ij.
#     dcs_bandwidth_scale: float = 1.0

#     # Bandwidth floor: h_i is floored at
#     #   dcs_bandwidth_scale * dcs_bandwidth_floor_frac * (global median of
#     #   POSITIVE neighbor distances).
#     # This stops the kernel from saturating to all-or-nothing when neighbors
#     # are (near-)duplicates -- the failure mode of a DISCRETE metric space such
#     # as the oracle button_states graph, where same-button neighbors sit at
#     # distance ~0. Necessary hygiene for continuous/mixed spaces; it does NOT
#     # by itself tame a fully-degenerate (all-exact-match) neighborhood -- that
#     # is controlled by the neighbor-mass weights below. Set 0.0 for the old
#     # (unfloored) behavior.
#     dcs_bandwidth_floor_frac: float = 1.0

#     # ----- Per-edge dynamics-consistency gate g_ij -------------------------
#     # Certifies that neighbor j's transition actually shares s_i's local
#     # action->displacement dynamics before its (s_j, a_j) is pooled into the
#     # vicinal backup / vicinal AWR. This is the certification that upgrades the
#     # method from "density-and-diversity gated" to genuinely dynamics-certified
#     # (addresses the cross-state-action contamination of the actor loss).
#     #   final edge weight = gate(J(s_i)) * kernel(d_ij) * g_ij
#     # g_ij = exp(-(r_ij / dcs_dynamics_scale)^2), r_ij = displacement-normalized
#     # residual of neighbor j under the local linear dynamics fit.
#     #
#     # dcs_use_dynamics_gate=False recovers the previous density-and-diversity
#     # gated behavior (g == 1) and is the clean ablation for measuring g's effect.
#     # Larger dcs_dynamics_scale -> gentler gate (so it cannot silently collapse
#     # the pool back to plain IQL; watch active_edge_frac / pool mass in logs).
#     dcs_use_dynamics_gate: bool = True
#     dcs_dynamics_scale: float = 1.0
#     dcs_dynamics_ridge: float = 1e-3

#     # kNN computation controls (mirroring the DC-IQL density cache).
#     dcs_subsample: int = 10_000_000
#     dcs_chunk_size: int = 50_000

#     # Coverage profile cache (raw kNN statistics only; post-processing knobs
#     # above never invalidate it). Saved as:
#     #   {dcs_profile_path}/{env}/[seed_{seed}/]coverage_profile_k{dcs_k}.npz
#     dcs_profile_path: Optional[str] = None
#     dcs_force_recompute: bool = False
#     dcs_cache_by_seed: bool = False

#     # ----- METRIC SOURCE: which space prototypes/local relations use -------
#     # Selects the metric space in which prototypes and local soft relations are built:
#     #   "observation" : default; neighbors from normalized observations (L2).
#     #   "oracle"      : neighbors from ground-truth dataset field(s) (e.g.
#     #                   puzzle button_states), loaded via add_info=True. This is
#     #                   the privileged CEILING TEST -- not deployable.
#     #   "temporal"    : neighbors from a LEARNED temporal representation phi(s)
#     #                   trained on the dataset's (s, s') transitions (see
#     #                   temporal_metric.py). Data-only and deployable: this is
#     #                   the approximation of the oracle graph. Training still
#     #                   uses standard observations; only neighbor selection uses
#     #                   phi(s).
#     dcs_metric_source: str = "observation"  # observation | oracle | temporal
#     # Dataset add_info field(s) used as the oracle metric space. Auto-defaults:
#     # puzzle -> button_states; cube/antmaze -> qpos,qvel. Comma-separated.
#     dcs_oracle_keys: Optional[str] = None
#     # Standardize the oracle space per-dim before prototype fitting so no
#     # single field (e.g. a large-magnitude qvel) dominates the distance.
#     dcs_oracle_standardize: bool = True

#     # ----- Temporal metric (dcs_metric_source="temporal") -----------------
#     # Contrastive encoder phi(s) hyperparameters. Embeddings are cached (keyed
#     # by these + env + seed) so the encoder is trained once. phi is L2-
#     # normalized onto the unit sphere by default, so Euclidean kNN ranks by
#     # cosine similarity; leave dcs_temporal_standardize False in that case.
#     dcs_temporal_dim: int = 32
#     dcs_temporal_hidden: int = 256
#     dcs_temporal_layers: int = 2
#     dcs_temporal_steps: int = 50_000
#     dcs_temporal_batch: int = 512
#     dcs_temporal_lr: float = 3e-4
#     dcs_temporal_temperature: float = 0.1
#     # Positive pairs are k-step successors with k in [1, horizon]. horizon=1
#     # uses next_observations directly (robust, no trajectory bookkeeping);
#     # horizon>1 samples within trajectories (needs terminals/masks or is
#     # detected from observation continuity).
#     dcs_temporal_horizon: int = 1
#     dcs_temporal_normalize: bool = True
#     dcs_temporal_standardize: bool = False
#     dcs_temporal_force_recompute: bool = False

#     # ----- Network input representation ablations --------------------------
#     # These switches independently control whether each learned function sees
#     # the frozen temporal embedding phi(s) or the normalized raw observation s.
#     # They require dcs_metric_source="temporal" because only that metric has a
#     # deployable encoder that can transform unseen evaluation states online.
#     actor_use_embedding: bool = False
#     q_use_embedding: bool = False
#     v_use_embedding: bool = False
#     # -----------------------------------------------------------------------

#     vf_lr: float = 3e-4
#     qf_lr: float = 3e-4
#     actor_lr: float = 3e-4
#     actor_dropout: Optional[float] = None
#     hidden_dim: int = 256
#     n_hidden: int = 2

#     # Standalone actor refit output directory.
#     actor_refit_dir_name: str = "actor_refit"

#     # Logging
#     project: str = "ORL-BIAS"
#     group: str = "DCS-IQL-JAX"
#     name: str = "DCS-IQL-JAX"
#     log_wandb: bool = True
#     log_every: int = 500
#     save_final_model: bool = False

#     def __post_init__(self):
#         refresh_algorithm_names(self)
#         validate_config(self)


# def refresh_algorithm_names(config: TrainConfig) -> None:
#     config.project = "ORL-BIAS"
#     config.group = f"{ALGORITHM_NAME}-JAX"
#     # Keep the historical run name by default. Uncomment the next two lines if
#     # you want separate checkpoint directories for each gate-source ablation.
#     # gate_source = getattr(config, "dcs_gate_source", "density_x_product")
#     # config.name = f"{ALGORITHM_NAME}-JAX-{config.env}-gate_{gate_source}"
#     config.name = f"{ALGORITHM_NAME}-JAX-{config.env}"


# def validate_config(config: TrainConfig) -> None:
#     assert config.mode in ("train", "refit"), "mode must be train or refit"
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
#     assert 0.0 <= config.scv_tau_min <= config.scv_tau_max <= 1.0
#     assert config.scv_support_power > 0.0
#     assert config.scv_support_eps > 0.0
#     assert config.dcs_k >= 1
#     assert config.dcs_value_neighbor_weight >= 0.0
#     assert config.dcs_actor_neighbor_weight >= 0.0
#     assert config.dcs_gate_source in DCS_GATE_SOURCES, (
#         f"dcs_gate_source must be one of {DCS_GATE_SOURCES}, got {config.dcs_gate_source}"
#     )
#     assert config.dcs_diversity_mode in ("action", "displacement", "product")
#     assert 0.0 <= config.dcs_percentile_low < config.dcs_percentile_high <= 100.0
#     assert config.dcs_gate_calibration in ("beta_mixture", "percentile")
#     assert 0.0 <= config.dcs_gate_low_percentile <= config.dcs_gate_high_percentile <= 100.0
#     assert config.dcs_beta_mixture_max_iter >= 1
#     assert config.dcs_beta_mixture_tol > 0.0
#     assert 0.0 < config.dcs_beta_mixture_eps < 0.5
#     assert config.dcs_beta_mixture_fit_samples >= 2
#     assert config.dcs_beta_mixture_fallback in ("zero", "percentile")
#     assert config.bc_num_prototypes >= 1
#     assert config.bc_prototype_fit_samples >= 2
#     assert config.bc_kmeans_iters >= 1
#     assert config.bc_kmeans_batch_size >= 1
#     assert config.bc_nearest_prototypes >= 1
#     assert config.bc_profile_chunk_size >= 1
#     assert config.bc_exact_recall_samples >= 0
#     assert config.dcs_bandwidth_scale > 0.0
#     assert config.dcs_bandwidth_floor_frac >= 0.0
#     assert config.dcs_dynamics_scale > 0.0
#     assert config.dcs_dynamics_ridge >= 0.0
#     assert config.dcs_metric_source in ("observation", "oracle", "temporal"), \
#         "dcs_metric_source must be 'observation', 'oracle', or 'temporal'"
#     if config.dcs_metric_source == "temporal":
#         assert config.dcs_temporal_dim >= 1
#         assert config.dcs_temporal_hidden >= 1
#         assert config.dcs_temporal_layers >= 1
#         assert config.dcs_temporal_steps >= 0
#         assert config.dcs_temporal_batch >= 2, "InfoNCE needs batch_size >= 2"
#         assert config.dcs_temporal_lr > 0.0
#         assert config.dcs_temporal_temperature > 0.0
#         assert config.dcs_temporal_horizon >= 1
#     uses_embedding_input = (
#         config.actor_use_embedding or config.q_use_embedding or config.v_use_embedding
#     )
#     if uses_embedding_input:
#         assert config.dcs_metric_source == "temporal", (
#             "actor_use_embedding/q_use_embedding/v_use_embedding require "
#             "dcs_metric_source='temporal' so unseen evaluation states can be "
#             "encoded by the learned temporal encoder."
#         )
#     assert config.dcs_subsample >= 1
#     assert config.dcs_chunk_size >= 1
#     if config.dcs_profile_path is not None:
#         assert config.dcs_profile_path != ""
#     if config.actor_dropout is not None:
#         assert config.actor_dropout >= 0.0 and config.actor_dropout < 1.0
#     assert config.hidden_dim > 0
#     assert config.n_hidden > 0
#     assert config.actor_refit_dir_name != ""
#     if config.mode == "refit":
#         assert config.load_model != "", "mode='refit' requires --load_model"


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
#     """Load env-specific hyperparameters and merge them into config.

#     Priority is:
#         dataclass defaults < hyperparams YAML < explicit CLI flags

#     Hyperparameter YAML keys must exactly match TrainConfig field names
#     (a few legacy DC-IQL keys are aliased for convenience).
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
#         # Legacy DC-IQL keys that map cleanly onto DCS-IQL fields.
#         "dc_density_model_path": "dcs_profile_path",
#         "dc_density_k": "dcs_k",
#         "dc_density_subsample": "dcs_subsample",
#         "dc_density_chunk_size": "dcs_chunk_size",
#         "dc_density_percentile_low": "dcs_percentile_low",
#         "dc_density_percentile_high": "dcs_percentile_high",
#         "dc_density_force_recompute": "dcs_force_recompute",
#         "dc_density_cache_by_seed": "dcs_cache_by_seed",
#     }
#     config_fields = set(TrainConfig.__dataclass_fields__.keys())
#     applied, skipped_unknown, skipped_cli = [], [], []
#     applied_fields = set()

#     for raw_key, raw_value in env_hyperparams.items():
#         key = aliases.get(raw_key, raw_key)
#         if key not in config_fields:
#             skipped_unknown.append(raw_key)
#             continue
#         if key in applied_fields:
#             continue
#         if key in cli_overrides or raw_key in cli_overrides:
#             skipped_cli.append(raw_key)
#             continue
#         setattr(config, key, _coerce_hparam_value(raw_value))
#         applied.append(f"{raw_key}->{key}" if raw_key != key else key)
#         applied_fields.add(key)

#     # Backward compatibility: old YAML/scripts may only specify
#     # dcs_diversity_mode. In the new ablation interface, that corresponds to
#     # density_x_{action, displacement, product}. Do this only when the new knob
#     # was not explicitly set by CLI/YAML.
#     if "dcs_gate_source" not in applied_fields and "dcs_gate_source" not in cli_overrides:
#         legacy_to_gate_source = {
#             "action": "action",
#             "displacement": "displacement",
#             "product": "product_diversity",
#         }
#         config.dcs_gate_source = legacy_to_gate_source.get(
#             config.dcs_diversity_mode, config.dcs_gate_source
#         )

#     refresh_algorithm_names(config)
#     validate_config(config)

#     if applied:
#         print(f"Loaded hyperparameters for {config.env} from {hparam_path}: {', '.join(applied)}")
#     if skipped_cli:
#         print(f"Kept CLI overrides for: {', '.join(skipped_cli)}")
#     if skipped_unknown:
#         print(f"Ignored unknown hyperparameter keys for {ALGORITHM_NAME}: {', '.join(skipped_unknown)}")
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


# # ---------------------------------------------------------------------------
# # Coverage profile cache plumbing (mirrors the DC-IQL density cache design)
# # ---------------------------------------------------------------------------

# def _safe_path_name(name: str) -> str:
#     """Make env names safe for directory paths."""
#     return name.replace("/", "_").replace(":", "_")


# def get_coverage_profile_cache_path(config: TrainConfig) -> Optional[Path]:
#     """Return the cache file path for the precomputed coverage profile.

#     The metric source is part of the path so that an oracle-graph profile and
#     an observation-graph profile for the same env never collide.
#     """
#     if config.dcs_profile_path is None:
#         return None

#     root = Path(config.dcs_profile_path)
#     env_name = _safe_path_name(config.env)
#     source_tag = "obs" if config.dcs_metric_source == "observation" else "oracle"
#     if config.dcs_metric_source == "oracle":
#         keys = resolve_oracle_keys(config) or ("auto",)
#         source_tag = "oracle-" + "_".join(keys)
#     elif config.dcs_metric_source == "temporal":
#         source_tag = "temporal-" + tmet.signature_hash(build_temporal_signature(config))
#     filename = f"coverage_profile_{_safe_path_name(source_tag)}_k{int(config.dcs_k)}.npz"

#     if config.dcs_cache_by_seed:
#         return root / env_name / f"seed_{config.seed}" / filename
#     return root / env_name / filename


# def resolve_oracle_keys(config: TrainConfig) -> Optional[Tuple[str, ...]]:
#     """Resolve the oracle dataset field name(s) for the oracle metric space.

#     Explicit --dcs_oracle_keys wins; otherwise auto-default by env family
#     (puzzle -> button_states; cube/antmaze -> qpos,qvel), mirroring the
#     metric_diagnostic.py oracle selection.
#     """
#     if config.dcs_oracle_keys:
#         return tuple(x.strip() for x in config.dcs_oracle_keys.split(",") if x.strip())
#     env = config.env
#     if env.startswith("puzzle-"):
#         return ("button_states",)
#     if env.startswith(("cube-", "antmaze-")):
#         return ("qpos", "qvel")
#     return None


# def build_oracle_metric_space(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
# ) -> np.ndarray:
#     """Concatenate the requested oracle field(s) into an (N, Dm) metric space,
#     standardized per-dim by default so no single field dominates the distance.
#     """
#     keys = resolve_oracle_keys(config)
#     if not keys:
#         raise SystemExit(
#             f"dcs_metric_source='oracle' but no oracle keys could be resolved for "
#             f"env '{config.env}'. Pass --dcs_oracle_keys, e.g. 'button_states'."
#         )
#     missing = [key for key in keys if key not in dataset]
#     if missing:
#         available = ", ".join(sorted(dataset.keys()))
#         raise SystemExit(
#             f"Oracle key(s) {missing} not in dataset (load uses add_info=True). "
#             f"Available keys: {available}"
#         )
#     parts = []
#     for key in keys:
#         arr = np.asarray(dataset[key], dtype=np.float32)
#         if arr.ndim == 1:
#             arr = arr[:, None]
#         elif arr.ndim > 2:
#             arr = arr.reshape(arr.shape[0], -1)
#         parts.append(arr)
#     oracle = np.concatenate(parts, axis=1).astype(np.float32)

#     if config.dcs_oracle_standardize:
#         mu = oracle.mean(0)
#         sd = oracle.std(0) + 1e-6
#         oracle = ((oracle - mu) / sd).astype(np.float32)

#     print(
#         f"Oracle metric space: keys={list(keys)} dim={oracle.shape[1]} "
#         f"standardize={config.dcs_oracle_standardize}"
#     )
#     return oracle


# def build_temporal_signature(
#     config: TrainConfig,
#     observations_shape: Optional[Tuple[int, ...]] = None,
# ) -> Dict[str, Any]:
#     """Encoder-defining signature; any change invalidates cached embeddings and
#     the coverage graph built from them."""
#     return {
#         "cache_version": tmet.TEMPORAL_CACHE_VERSION,
#         "env": config.env,
#         "seed": int(config.seed),
#         "normalize": bool(config.dcs_temporal_normalize),
#         "observations_shape": tuple(observations_shape) if observations_shape else None,
#         "embed_dim": int(config.dcs_temporal_dim),
#         "hidden_dim": int(config.dcs_temporal_hidden),
#         "n_hidden": int(config.dcs_temporal_layers),
#         "steps": int(config.dcs_temporal_steps),
#         "batch_size": int(config.dcs_temporal_batch),
#         "lr": float(config.dcs_temporal_lr),
#         "temperature": float(config.dcs_temporal_temperature),
#         "horizon": int(config.dcs_temporal_horizon),
#     }


# def get_temporal_embeddings_cache_path(config: TrainConfig) -> Optional[Path]:
#     if config.dcs_profile_path is None:
#         return None
#     root = Path(config.dcs_profile_path)
#     env_name = _safe_path_name(config.env)
#     sig_hash = tmet.signature_hash(build_temporal_signature(config))
#     filename = f"temporal_emb_{sig_hash}.npz"
#     if config.dcs_cache_by_seed:
#         return root / env_name / f"seed_{config.seed}" / filename
#     return root / env_name / filename


# def get_temporal_representation_cache_path(config: TrainConfig) -> Optional[Path]:
#     """Cache path for the deployable temporal representation bundle.

#     Unlike the legacy embedding-only .npz cache, this bundle also stores the
#     frozen encoder parameters and the embedding standardization statistics, so
#     the same phi(s) can be applied to unseen evaluation states.
#     """
#     if config.dcs_profile_path is None:
#         return None
#     root = Path(config.dcs_profile_path)
#     env_name = _safe_path_name(config.env)
#     sig_hash = tmet.signature_hash(build_temporal_signature(config))
#     filename = f"temporal_repr_{sig_hash}.pkl"
#     if config.dcs_cache_by_seed:
#         return root / env_name / f"seed_{config.seed}" / filename
#     return root / env_name / filename


# def _temporal_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
#     logp = jax.nn.log_softmax(logits, axis=-1)
#     return -jnp.mean(logp[jnp.arange(logits.shape[0]), labels])


# def _encode_temporal_array(
#     encoder: Any,
#     params: Any,
#     observations: np.ndarray,
#     device: Any,
#     chunk_size: int = 100_000,
# ) -> np.ndarray:
#     """Encode an arbitrary array with the frozen temporal encoder."""
#     observations = np.asarray(observations, dtype=np.float32)
#     if observations.shape[0] == 0:
#         return np.zeros((0, int(encoder.embed_dim)), dtype=np.float32)

#     output = np.empty((observations.shape[0], int(encoder.embed_dim)), dtype=np.float32)
#     apply_fn = jax.jit(lambda p, x: encoder.apply({"params": p}, x))
#     for start in range(0, observations.shape[0], int(chunk_size)):
#         end = min(start + int(chunk_size), observations.shape[0])
#         z = apply_fn(
#             params,
#             jax.device_put(jnp.asarray(observations[start:end]), device),
#         )
#         output[start:end] = np.asarray(jax.device_get(z), dtype=np.float32)
#     return output


# def train_temporal_representation(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
#     device: Any = None,
# ) -> Dict[str, Any]:
#     """Train phi and return everything needed for graph construction and RL.

#     The encoder is trained once and then frozen. Dataset states are pre-encoded
#     for efficient minibatch training, while the saved encoder parameters are
#     used to encode unseen environment states during actor evaluation.
#     """
#     observations = np.asarray(dataset["observations"], dtype=np.float32)
#     if "next_observations" not in dataset:
#         raise SystemExit(
#             "dcs_metric_source='temporal' requires next_observations in the dataset."
#         )
#     next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
#     n = observations.shape[0]

#     if device is None:
#         device = jax.devices()[0]

#     encoder = tmet.TemporalEncoder(
#         embed_dim=config.dcs_temporal_dim,
#         hidden_dim=config.dcs_temporal_hidden,
#         n_hidden=config.dcs_temporal_layers,
#         normalize=config.dcs_temporal_normalize,
#     )
#     key = jax.random.PRNGKey(config.seed)
#     key, init_key = jax.random.split(key)
#     params = encoder.init(
#         init_key,
#         jnp.zeros((1, observations.shape[1]), dtype=jnp.float32),
#     )["params"]

#     tx = optax.adam(config.dcs_temporal_lr)
#     opt_state = tx.init(params)
#     inv_temp = 1.0 / max(float(config.dcs_temporal_temperature), 1e-8)

#     @jax.jit
#     def temporal_train_step(params, opt_state, anchor_obs, positive_obs):
#         def loss_fn(p):
#             za = encoder.apply({"params": p}, anchor_obs)
#             zp = encoder.apply({"params": p}, positive_obs)
#             logits = (za @ zp.T) * inv_temp
#             labels = jnp.arange(logits.shape[0])
#             loss = 0.5 * (
#                 _temporal_cross_entropy(logits, labels)
#                 + _temporal_cross_entropy(logits.T, labels)
#             )
#             pos_sim = jnp.mean(jnp.sum(za * zp, axis=-1))
#             return loss, pos_sim

#         (loss, pos_sim), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
#         updates, opt_state = tx.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss, pos_sim

#     next_index = None
#     use_kstep = config.dcs_temporal_horizon > 1
#     if use_kstep:
#         next_index = tmet.build_successor_index(dataset)

#     rng = np.random.default_rng(config.seed)
#     print(
#         f"Training deployable temporal representation: N={n} "
#         f"dim={config.dcs_temporal_dim} steps={config.dcs_temporal_steps} "
#         f"batch={config.dcs_temporal_batch} horizon={config.dcs_temporal_horizon} "
#         f"temp={config.dcs_temporal_temperature} "
#         f"positives={'k-step' if use_kstep else '1-step successor'}"
#     )
#     for it in range(int(config.dcs_temporal_steps)):
#         if use_kstep:
#             anchor_idx, positive_idx = tmet.sample_kstep_pairs(
#                 next_index,
#                 config.dcs_temporal_horizon,
#                 config.dcs_temporal_batch,
#                 rng,
#             )
#             anchor_obs = observations[anchor_idx]
#             positive_obs = observations[positive_idx]
#         else:
#             anchor_idx = rng.integers(0, n, size=config.dcs_temporal_batch)
#             anchor_obs = observations[anchor_idx]
#             positive_obs = next_observations[anchor_idx]

#         params, opt_state, loss, pos_sim = temporal_train_step(
#             params,
#             opt_state,
#             jax.device_put(jnp.asarray(anchor_obs), device),
#             jax.device_put(jnp.asarray(positive_obs), device),
#         )
#         if (it + 1) % 5_000 == 0:
#             print(
#                 f"  temporal step {it + 1}/{config.dcs_temporal_steps}: "
#                 f"infonce={float(jax.device_get(loss)):.4f} "
#                 f"pos_cos={float(jax.device_get(pos_sim)):.4f}"
#             )

#     embeddings = _encode_temporal_array(encoder, params, observations, device)
#     next_embeddings = _encode_temporal_array(encoder, params, next_observations, device)

#     if config.dcs_temporal_standardize:
#         embedding_mean = embeddings.mean(axis=0).astype(np.float32)
#         embedding_std = (embeddings.std(axis=0) + 1e-6).astype(np.float32)
#         embeddings = ((embeddings - embedding_mean) / embedding_std).astype(np.float32)
#         next_embeddings = (
#             (next_embeddings - embedding_mean) / embedding_std
#         ).astype(np.float32)
#     else:
#         embedding_mean = np.zeros((config.dcs_temporal_dim,), dtype=np.float32)
#         embedding_std = np.ones((config.dcs_temporal_dim,), dtype=np.float32)

#     return {
#         "signature": tmet.temporal_signature(
#             build_temporal_signature(config, observations_shape=observations.shape)
#         ),
#         "embeddings": embeddings.astype(np.float32),
#         "next_embeddings": next_embeddings.astype(np.float32),
#         "encoder_params": serialization.to_state_dict(params),
#         "embedding_mean": embedding_mean,
#         "embedding_std": embedding_std,
#     }


# def load_or_build_temporal_representation(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
#     device: Any = None,
# ) -> Dict[str, Any]:
#     """Load or train the deployable temporal representation bundle."""
#     observations = np.asarray(dataset["observations"], dtype=np.float32)
#     expected_signature = tmet.temporal_signature(
#         build_temporal_signature(config, observations_shape=observations.shape)
#     )
#     cache_path = get_temporal_representation_cache_path(config)

#     if cache_path is not None and cache_path.exists() and not config.dcs_temporal_force_recompute:
#         try:
#             with open(cache_path, "rb") as f:
#                 payload = pickle.load(f)
#             required = {
#                 "signature",
#                 "embeddings",
#                 "next_embeddings",
#                 "encoder_params",
#                 "embedding_mean",
#                 "embedding_std",
#             }
#             if payload.get("signature") == expected_signature and required.issubset(payload.keys()):
#                 print(f"Loaded deployable temporal representation from: {cache_path}")
#                 return payload
#             print(f"Temporal representation cache is stale/incomplete; retraining: {cache_path}")
#         except Exception as exc:
#             print(f"Failed to load temporal representation cache {cache_path}: {exc}; retraining.")

#     representation = train_temporal_representation(config, dataset, device=device)
#     if cache_path is not None:
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(cache_path, "wb") as f:
#             pickle.dump(representation, f)
#         print(f"Saved deployable temporal representation to: {cache_path}")

#     print(
#         f"Temporal representation: dim={representation['embeddings'].shape[1]} "
#         f"normalize={config.dcs_temporal_normalize} "
#         f"standardize={config.dcs_temporal_standardize} "
#         f"horizon={config.dcs_temporal_horizon}"
#     )
#     return representation


# def build_temporal_metric_space(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
#     device: Any = None,
# ) -> np.ndarray:
#     """Backward-compatible wrapper returning phi(observations)."""
#     representation = load_or_build_temporal_representation(config, dataset, device=device)
#     return np.asarray(representation["embeddings"], dtype=np.float32)


# def build_coverage_profile_metadata(
#     config: TrainConfig,
#     observations: np.ndarray,
#     actions: np.ndarray,
# ) -> Dict[str, Any]:
#     """Metadata that must match before a cached profile can be reused.

#     Post-processing knobs (percentiles, diversity mode, gate, bandwidth) are
#     intentionally excluded: they are applied at load time. The metric source
#     and oracle keys ARE included, since they change neighbor selection.
#     """
#     oracle_keys = resolve_oracle_keys(config) if config.dcs_metric_source == "oracle" else None
#     temporal_sig = (
#         build_temporal_signature(config, observations_shape=observations.shape)
#         if config.dcs_metric_source == "temporal" else None
#     )
#     return covp.canonicalize_metadata(
#         {
#             "cache_version": covp.COVERAGE_CACHE_VERSION,
#             "env": config.env,
#             "seed": int(config.seed) if config.dcs_cache_by_seed else None,
#             "normalize": bool(config.normalize),
#             "observations_shape": tuple(observations.shape),
#             "observations_dtype": str(np.asarray(observations).dtype),
#             "actions_shape": tuple(actions.shape),
#             "dcs_k": int(config.dcs_k),
#             "dcs_subsample": int(config.dcs_subsample),
#             "dcs_chunk_size": int(config.dcs_chunk_size),
#             "dcs_dynamics_ridge": float(config.dcs_dynamics_ridge),
#             "dcs_metric_source": config.dcs_metric_source,
#             "dcs_oracle_keys": list(oracle_keys) if oracle_keys else None,
#             "dcs_oracle_standardize": bool(config.dcs_oracle_standardize)
#             if config.dcs_metric_source == "oracle" else None,
#             "dcs_temporal_signature": tmet.signature_hash(temporal_sig)
#             if temporal_sig is not None else None,
#         }
#     )



# def _squared_distance_matrix(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
#     """Memory-efficient squared Euclidean distance matrix."""
#     x = np.asarray(x, dtype=np.float32)
#     centers = np.asarray(centers, dtype=np.float32)
#     x2 = np.sum(x * x, axis=1, keepdims=True)
#     c2 = np.sum(centers * centers, axis=1, keepdims=True).T
#     d2 = x2 + c2 - 2.0 * (x @ centers.T)
#     return np.maximum(d2, 0.0).astype(np.float32)


# def _normalized_displacements(
#     observations: np.ndarray,
#     next_observations: np.ndarray,
# ) -> np.ndarray:
#     disp = np.asarray(next_observations, dtype=np.float32) - np.asarray(
#         observations, dtype=np.float32
#     )
#     norms = np.linalg.norm(disp, axis=-1, keepdims=True)
#     return np.where(
#         norms > 1e-8,
#         disp / np.maximum(norms, 1e-8),
#         0.0,
#     ).astype(np.float32)


# def build_prototype_index_signature(
#     config: TrainConfig,
#     metric_space: np.ndarray,
#     actions: np.ndarray,
# ) -> Dict[str, Any]:
#     """Cache signature for prototype-indexed approximate JG preprocessing."""
#     temporal_sig = (
#         tmet.signature_hash(build_temporal_signature(config))
#         if config.dcs_metric_source == "temporal"
#         else None
#     )
#     return {
#         "version": 2,
#         "env": config.env,
#         "metric_source": config.dcs_metric_source,
#         "metric_shape": tuple(metric_space.shape),
#         "actions_shape": tuple(actions.shape),
#         "normalize": bool(config.normalize),
#         "temporal_signature": temporal_sig,
#         "num_prototypes": int(config.bc_num_prototypes),
#         "prototype_fit_samples": int(config.bc_prototype_fit_samples),
#         "kmeans_iters": int(config.bc_kmeans_iters),
#         "kmeans_batch_size": int(config.bc_kmeans_batch_size),
#         "nearest_prototypes": int(config.bc_nearest_prototypes),
#         "top_k": int(config.dcs_k),
#         "preprocess_seed": int(config.bc_preprocess_seed),
#         "exact_recall_samples": int(config.bc_exact_recall_samples),
#         "gate_source": config.dcs_gate_source,
#         "percentile_low": float(config.dcs_percentile_low),
#         "percentile_high": float(config.dcs_percentile_high),
#         "gate_calibration": config.dcs_gate_calibration,
#         "gate_low_percentile": float(config.dcs_gate_low_percentile),
#         "gate_high_percentile": float(config.dcs_gate_high_percentile),
#         "beta_max_iter": int(config.dcs_beta_mixture_max_iter),
#         "beta_tol": float(config.dcs_beta_mixture_tol),
#         "beta_eps": float(config.dcs_beta_mixture_eps),
#         "beta_fit_samples": int(config.dcs_beta_mixture_fit_samples),
#         "beta_fallback": config.dcs_beta_mixture_fallback,
#     }


# def get_prototype_index_cache_path(
#     config: TrainConfig,
#     metric_space: np.ndarray,
#     actions: np.ndarray,
# ) -> Optional[Path]:
#     if config.dcs_profile_path is None:
#         return None
#     signature = build_prototype_index_signature(config, metric_space, actions)
#     payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
#     sig_hash = hashlib.sha1(payload).hexdigest()[:12]
#     root = Path(config.dcs_profile_path)
#     env_name = _safe_path_name(config.env)
#     return root / env_name / f"prototype_indexed_jg_profile_{sig_hash}.npz"


# def _fit_streaming_prototypes(
#     metric_space: np.ndarray,
#     num_prototypes: int,
#     fit_samples: int,
#     n_iters: int,
#     batch_size: int,
#     seed: int,
# ) -> np.ndarray:
#     """Fit compact state prototypes with streaming mini-batch k-means."""
#     z = np.asarray(metric_space, dtype=np.float32)
#     n = z.shape[0]
#     m = min(int(num_prototypes), n)
#     s = min(int(fit_samples), n)
#     rng = np.random.default_rng(seed)
#     fit_idx = rng.choice(n, size=s, replace=False) if s < n else np.arange(n)
#     fit_z = z[fit_idx]

#     init_idx = rng.choice(s, size=m, replace=False)
#     centers = fit_z[init_idx].copy()
#     counts = np.ones((m,), dtype=np.float64)

#     for epoch in range(int(n_iters)):
#         order = rng.permutation(s)
#         inertia_sum = 0.0
#         seen = 0
#         for start in range(0, s, int(batch_size)):
#             batch = fit_z[order[start : start + int(batch_size)]]
#             d2 = _squared_distance_matrix(batch, centers)
#             labels = np.argmin(d2, axis=1)
#             inertia_sum += float(np.sum(d2[np.arange(batch.shape[0]), labels]))
#             seen += batch.shape[0]

#             for cluster_id in np.unique(labels):
#                 mask = labels == cluster_id
#                 n_new = int(np.sum(mask))
#                 batch_mean = np.mean(batch[mask], axis=0)
#                 old_count = counts[cluster_id]
#                 new_count = old_count + n_new
#                 centers[cluster_id] = (
#                     (old_count / new_count) * centers[cluster_id]
#                     + (n_new / new_count) * batch_mean
#                 )
#                 counts[cluster_id] = new_count

#         print(
#             f"  prototype epoch {epoch + 1}/{n_iters}: "
#             f"mean_fit_d2={inertia_sum / max(seen, 1):.6f}"
#         )

#     return centers.astype(np.float32)


# def _assign_to_prototypes(
#     metric_space: np.ndarray,
#     centers: np.ndarray,
#     chunk_size: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     n = metric_space.shape[0]
#     assignments = np.empty((n,), dtype=np.int32)
#     nearest_d2 = np.empty((n,), dtype=np.float32)
#     for start in range(0, n, int(chunk_size)):
#         end = min(start + int(chunk_size), n)
#         d2 = _squared_distance_matrix(metric_space[start:end], centers)
#         labels = np.argmin(d2, axis=1)
#         assignments[start:end] = labels.astype(np.int32)
#         nearest_d2[start:end] = d2[np.arange(end - start), labels]
#     return assignments, nearest_d2


# def _build_inverted_prototype_index(
#     assignments: np.ndarray,
#     n_prototypes: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Return members sorted by prototype plus offsets for O(1) cell lookup."""
#     assignments = np.asarray(assignments, dtype=np.int32)
#     order = np.argsort(assignments, kind="stable").astype(np.int32)
#     counts = np.bincount(assignments, minlength=n_prototypes).astype(np.int64)
#     offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
#     return order, offsets, counts


# def _gather_candidate_pool(
#     prototype_order: np.ndarray,
#     member_order: np.ndarray,
#     offsets: np.ndarray,
#     self_index: int,
#     min_candidates: int,
#     initial_prototypes: int,
# ) -> Tuple[np.ndarray, int]:
#     """Gather members from nearby cells, expanding only if k candidates are absent."""
#     selected_parts: List[np.ndarray] = []
#     total = 0
#     used = 0
#     initial_prototypes = max(1, int(initial_prototypes))

#     for rank, proto in enumerate(prototype_order):
#         lo = int(offsets[int(proto)])
#         hi = int(offsets[int(proto) + 1])
#         if hi > lo:
#             members = member_order[lo:hi]
#             selected_parts.append(members)
#             total += int(members.size)
#         used = rank + 1
#         # Account for the anchor itself, which may be inside the pool.
#         enough_nonself = total - 1 >= min_candidates
#         if used >= initial_prototypes and enough_nonself:
#             break

#     if not selected_parts:
#         return np.empty((0,), dtype=np.int32), used

#     candidates = np.concatenate(selected_parts).astype(np.int32, copy=False)
#     candidates = candidates[candidates != int(self_index)]
#     return candidates, used


# def _approximate_neighbors_from_prototype_index(
#     metric_space: np.ndarray,
#     centers: np.ndarray,
#     member_order: np.ndarray,
#     offsets: np.ndarray,
#     k: int,
#     nearest_prototypes: int,
#     chunk_size: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Coarse prototype retrieval followed by exact top-k within the pool.

#     This is the core approximation. Prototypes only choose a candidate pool;
#     final neighbors are actual dataset transitions selected by exact metric
#     distance within that pool.
#     """
#     z = np.asarray(metric_space, dtype=np.float32)
#     n = z.shape[0]
#     k_eff = int(min(k, max(n - 1, 1)))
#     n_prototypes = centers.shape[0]
#     initial_r = int(min(max(nearest_prototypes, 1), n_prototypes))

#     neighbor_indices = np.empty((n, k_eff), dtype=np.int32)
#     neighbor_distances = np.empty((n, k_eff), dtype=np.float32)
#     candidate_count = np.empty((n,), dtype=np.int32)
#     prototypes_used = np.empty((n,), dtype=np.int32)

#     for start in range(0, n, int(chunk_size)):
#         end = min(start + int(chunk_size), n)
#         anchors = z[start:end]
#         center_d2 = _squared_distance_matrix(anchors, centers)
#         # Full center ordering is cheap because M << N and enables adaptive
#         # expansion when the nearest r cells do not contain enough candidates.
#         proto_rank = np.argsort(center_d2, axis=1, kind="stable")

#         for local_row, global_i in enumerate(range(start, end)):
#             candidates, used = _gather_candidate_pool(
#                 prototype_order=proto_rank[local_row],
#                 member_order=member_order,
#                 offsets=offsets,
#                 self_index=global_i,
#                 min_candidates=k_eff,
#                 initial_prototypes=initial_r,
#             )
#             if candidates.size < k_eff:
#                 raise RuntimeError(
#                     f"Prototype index produced only {candidates.size} non-self "
#                     f"candidates for state {global_i}, but k={k_eff}."
#                 )

#             diff = z[candidates] - z[global_i]
#             d2 = np.einsum("nd,nd->n", diff, diff)

#             if candidates.size == k_eff:
#                 chosen_local = np.arange(k_eff)
#             else:
#                 chosen_local = np.argpartition(d2, k_eff - 1)[:k_eff]
#             # Deterministic ordering of the selected top-k by distance, then index.
#             selected_idx = candidates[chosen_local]
#             selected_d2 = d2[chosen_local]
#             rank = np.lexsort((selected_idx, selected_d2))
#             selected_idx = selected_idx[rank]
#             selected_d2 = selected_d2[rank]

#             neighbor_indices[global_i] = selected_idx.astype(np.int32)
#             neighbor_distances[global_i] = np.sqrt(
#                 np.maximum(selected_d2, 0.0)
#             ).astype(np.float32)
#             candidate_count[global_i] = int(candidates.size)
#             prototypes_used[global_i] = int(used)

#         if start == 0 or end == n or (start // int(chunk_size)) % 25 == 0:
#             print(f"  prototype-indexed top-k: {end}/{n} states")

#     return neighbor_indices, neighbor_distances, candidate_count, prototypes_used


# def _compute_jg_statistics_from_neighbors(
#     observations: np.ndarray,
#     actions: np.ndarray,
#     next_observations: np.ndarray,
#     neighbor_indices: np.ndarray,
#     chunk_size: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Match JG-IQL's action-spread and displacement-dispersion definitions."""
#     observations = np.asarray(observations, dtype=np.float32)
#     actions = np.asarray(actions, dtype=np.float32)
#     next_observations = np.asarray(next_observations, dtype=np.float32)
#     idx_all = np.asarray(neighbor_indices, dtype=np.int32)
#     n = observations.shape[0]

#     disp = next_observations - observations
#     disp_norm = np.linalg.norm(disp, axis=-1)
#     unit_disp = disp / np.maximum(disp_norm[:, None], 1e-8)
#     unit_disp = np.where(disp_norm[:, None] > 1e-8, unit_disp, 0.0).astype(np.float32)
#     moving = (disp_norm > 1e-8).astype(np.float32)

#     action_spread = np.empty((n,), dtype=np.float32)
#     disp_dispersion = np.empty((n,), dtype=np.float32)

#     for start in range(0, n, int(chunk_size)):
#         end = min(start + int(chunk_size), n)
#         idx = idx_all[start:end]

#         nbr_actions = actions[idx]
#         centroid = nbr_actions.mean(axis=1, keepdims=True)
#         sq = ((nbr_actions - centroid) ** 2).sum(axis=-1)
#         action_spread[start:end] = np.sqrt(sq.mean(axis=1)).astype(np.float32)

#         nbr_unit = unit_disp[idx]
#         resultant = np.linalg.norm(nbr_unit.sum(axis=1), axis=-1)
#         total = moving[idx].sum(axis=1)
#         dispersion = 1.0 - resultant / np.maximum(total, 1e-8)
#         dispersion = np.where(total > 0.5, dispersion, 0.0)
#         disp_dispersion[start:end] = np.clip(dispersion, 0.0, 1.0).astype(np.float32)

#     return action_spread, disp_dispersion


# def _prototype_index_exact_recall_diagnostic(
#     metric_space: np.ndarray,
#     approx_neighbors: np.ndarray,
#     sample_count: int,
#     seed: int,
# ) -> Dict[str, float]:
#     """Optional expensive recall@k check against full-dataset exact search."""
#     n, k = approx_neighbors.shape
#     sample_count = int(min(max(sample_count, 0), n))
#     if sample_count <= 0:
#         return {"exact_recall_samples": 0.0, "exact_recall_at_k": float("nan")}

#     rng = np.random.default_rng(seed)
#     anchors = rng.choice(n, size=sample_count, replace=False)
#     recalls = []
#     z = np.asarray(metric_space, dtype=np.float32)
#     for i in anchors:
#         diff = z - z[i]
#         d2 = np.einsum("nd,nd->n", diff, diff)
#         d2[int(i)] = np.inf
#         exact = np.argpartition(d2, k - 1)[:k]
#         recalls.append(len(set(exact.tolist()) & set(approx_neighbors[i].tolist())) / float(k))
#     return {
#         "exact_recall_samples": float(sample_count),
#         "exact_recall_at_k": float(np.mean(recalls)),
#     }


# def compute_prototype_indexed_profile(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
#     metric_space: np.ndarray,
# ) -> Dict[str, Any]:
#     """Approximate JG-IQL profile via prototype retrieval + exact local top-k."""
#     metric_space = np.asarray(metric_space, dtype=np.float32)
#     actions = np.asarray(dataset["actions"], dtype=np.float32)
#     observations = np.asarray(dataset["observations"], dtype=np.float32)
#     next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
#     n = observations.shape[0]

#     n_prototypes = min(int(config.bc_num_prototypes), n)
#     n_near = min(int(config.bc_nearest_prototypes), n_prototypes)
#     k_eff = int(min(config.dcs_k, max(n - 1, 1)))
#     print(
#         f"Building prototype-indexed approximate JG profile: N={n}, "
#         f"prototypes={n_prototypes}, initial_near_prototypes={n_near}, k={k_eff}"
#     )

#     centers = _fit_streaming_prototypes(
#         metric_space,
#         num_prototypes=n_prototypes,
#         fit_samples=config.bc_prototype_fit_samples,
#         n_iters=config.bc_kmeans_iters,
#         batch_size=config.bc_kmeans_batch_size,
#         seed=config.bc_preprocess_seed,
#     )
#     assignments, nearest_d2 = _assign_to_prototypes(
#         metric_space, centers, config.bc_profile_chunk_size
#     )
#     member_order, offsets, cell_counts = _build_inverted_prototype_index(
#         assignments, n_prototypes
#     )

#     neighbor_indices, neighbor_distances, candidate_count, prototypes_used = (
#         _approximate_neighbors_from_prototype_index(
#             metric_space=metric_space,
#             centers=centers,
#             member_order=member_order,
#             offsets=offsets,
#             k=k_eff,
#             nearest_prototypes=n_near,
#             chunk_size=config.bc_profile_chunk_size,
#         )
#     )

#     action_spread, disp_dispersion = _compute_jg_statistics_from_neighbors(
#         observations=observations,
#         actions=actions,
#         next_observations=next_observations,
#         neighbor_indices=neighbor_indices,
#         chunk_size=config.bc_profile_chunk_size,
#     )
#     knn_radius = neighbor_distances[:, -1].astype(np.float32)

#     density_conf = covp.percentile_confidence(
#         knn_radius,
#         config.dcs_percentile_low,
#         config.dcs_percentile_high,
#         invert=True,
#     )
#     action_div_conf = covp.percentile_confidence(
#         action_spread,
#         config.dcs_percentile_low,
#         config.dcs_percentile_high,
#     )
#     disp_div_conf = covp.percentile_confidence(
#         disp_dispersion,
#         config.dcs_percentile_low,
#         config.dcs_percentile_high,
#     )

#     gate_signal, gate_signal_name = build_gate_signal(
#         config, density_conf, action_div_conf, disp_div_conf
#     )
#     gate, gate_diagnostics = build_data_adaptive_gate(config, gate_signal)

#     recall_diag = _prototype_index_exact_recall_diagnostic(
#         metric_space=metric_space,
#         approx_neighbors=neighbor_indices,
#         sample_count=config.bc_exact_recall_samples,
#         seed=config.bc_preprocess_seed,
#     )

#     initial_r = max(1, n_near)
#     expanded_frac = float(np.mean(prototypes_used > initial_r))
#     positive_cell_counts = cell_counts[cell_counts > 0]

#     return {
#         "density_conf": np.asarray(density_conf, dtype=np.float32),
#         "action_div_conf": np.asarray(action_div_conf, dtype=np.float32),
#         "disp_div_conf": np.asarray(disp_div_conf, dtype=np.float32),
#         "gate_signal": np.asarray(gate_signal, dtype=np.float32),
#         "gate": np.asarray(gate, dtype=np.float32),
#         "knn_radius_raw": knn_radius,
#         "action_spread_raw": action_spread,
#         "disp_dispersion_raw": disp_dispersion,
#         "effective_ref_count": np.full((n,), k_eff, dtype=np.float32),
#         "prototype_count": np.asarray([n_prototypes], dtype=np.int32),
#         "nonempty_prototype_count": np.asarray([positive_cell_counts.size], dtype=np.int32),
#         "mean_cell_size": np.asarray([np.mean(positive_cell_counts) if positive_cell_counts.size else 0.0], dtype=np.float32),
#         "max_cell_size": np.asarray([np.max(positive_cell_counts) if positive_cell_counts.size else 0.0], dtype=np.float32),
#         "mean_candidate_count": np.asarray([np.mean(candidate_count)], dtype=np.float32),
#         "median_candidate_count": np.asarray([np.median(candidate_count)], dtype=np.float32),
#         "p95_candidate_count": np.asarray([np.percentile(candidate_count, 95)], dtype=np.float32),
#         "max_candidate_count": np.asarray([np.max(candidate_count)], dtype=np.int32),
#         "mean_prototypes_used": np.asarray([np.mean(prototypes_used)], dtype=np.float32),
#         "expanded_prototype_frac": np.asarray([expanded_frac], dtype=np.float32),
#         "exact_recall_samples": np.asarray([recall_diag["exact_recall_samples"]], dtype=np.float32),
#         "exact_recall_at_k": np.asarray([recall_diag["exact_recall_at_k"]], dtype=np.float32),
#         "gate_signal_name": np.asarray([gate_signal_name]),
#         "gate_diagnostics_json": np.asarray([json.dumps(gate_diagnostics, sort_keys=True)]),
#     }


# def load_or_compute_prototype_indexed_profile(
#     config: TrainConfig,
#     dataset: Dict[str, np.ndarray],
#     device: Any = None,
#     temporal_representation: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """Load or build the prototype-indexed approximate JG profile."""
#     if config.dcs_metric_source == "oracle":
#         metric_space = build_oracle_metric_space(config, dataset)
#         print("PI-JG metric space: oracle (ceiling test only).")
#     elif config.dcs_metric_source == "temporal":
#         if temporal_representation is None:
#             temporal_representation = load_or_build_temporal_representation(
#                 config, dataset, device=device
#             )
#         metric_space = np.asarray(temporal_representation["embeddings"], dtype=np.float32)
#         print("PI-JG metric space: learned temporal representation.")
#     else:
#         metric_space = np.asarray(dataset["observations"], dtype=np.float32)
#         print("PI-JG metric space: normalized observations.")

#     cache_path = get_prototype_index_cache_path(config, metric_space, dataset["actions"])
#     signature = build_prototype_index_signature(config, metric_space, dataset["actions"])
#     signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))

#     if cache_path is not None and cache_path.exists() and not config.bc_force_recompute:
#         try:
#             with np.load(cache_path, allow_pickle=False) as cached:
#                 if str(cached["signature_json"].item()) == signature_json:
#                     profile = {
#                         key: cached[key] for key in cached.files if key != "signature_json"
#                     }
#                     print(f"Loaded prototype-indexed JG profile from: {cache_path}")
#                     return profile
#         except Exception as exc:
#             print(
#                 f"Failed to load prototype-indexed cache {cache_path}: {exc}; "
#                 "recomputing."
#             )

#     profile = compute_prototype_indexed_profile(config, dataset, metric_space)
#     if cache_path is not None:
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         save_payload = dict(profile)
#         save_payload["signature_json"] = np.asarray(signature_json)
#         np.savez_compressed(cache_path, **save_payload)
#         print(f"Saved prototype-indexed JG profile to: {cache_path}")
#     return profile


# def build_gate_signal(
#     config: TrainConfig,
#     density_conf: np.ndarray,
#     action_div_conf: np.ndarray,
#     disp_div_conf: np.ndarray,
# ) -> Tuple[np.ndarray, str]:
#     """Return the bounded pre-gate scalar signal J_i used to produce G_i.

#     By default, a two-component Beta mixture is fitted to this signal and the
#     posterior probability of the higher-mean component is used directly as G_i.
#     The legacy percentile ramp remains available as an ablation.
#     """
#     density_conf = np.asarray(density_conf, dtype=np.float32)
#     action_div_conf = np.asarray(action_div_conf, dtype=np.float32)
#     disp_div_conf = np.asarray(disp_div_conf, dtype=np.float32)

#     if config.dcs_gate_source == "density":
#         return density_conf, "density"
#     if config.dcs_gate_source == "action":
#         return action_div_conf, "action_diversity"
#     if config.dcs_gate_source == "displacement":
#         return disp_div_conf, "displacement_diversity"
#     if config.dcs_gate_source == "product_diversity":
#         diversity = np.sqrt(
#             np.clip(action_div_conf, 0.0, 1.0) * np.clip(disp_div_conf, 0.0, 1.0)
#         ).astype(np.float32)
#         return diversity, "sqrt(action_diversity * displacement_diversity)"
#     if config.dcs_gate_source == "density_x_action":
#         return (density_conf * action_div_conf).astype(np.float32), "density * action_diversity"
#     if config.dcs_gate_source == "density_x_displacement":
#         return (density_conf * disp_div_conf).astype(np.float32), "density * displacement_diversity"
#     if config.dcs_gate_source == "density_x_product":
#         diversity = np.sqrt(
#             np.clip(action_div_conf, 0.0, 1.0) * np.clip(disp_div_conf, 0.0, 1.0)
#         ).astype(np.float32)
#         return (density_conf * diversity).astype(np.float32), (
#             "density * sqrt(action_diversity * displacement_diversity)"
#         )

#     raise ValueError(
#         f"Unknown dcs_gate_source={config.dcs_gate_source}. "
#         f"Valid values: {DCS_GATE_SOURCES}"
#     )



# def _weighted_beta_moments(
#     x: np.ndarray,
#     weights: np.ndarray,
#     eps: float,
# ) -> Tuple[float, float]:
#     """Estimate Beta(alpha, beta) by weighted method of moments.

#     This is used as the M-step inside a lightweight two-component EM routine.
#     The gate signal is clipped away from exact 0/1 before fitting, so both
#     shape parameters remain finite.
#     """
#     x = np.asarray(x, dtype=np.float64).reshape(-1)
#     weights = np.asarray(weights, dtype=np.float64).reshape(-1)
#     mass = float(np.sum(weights))
#     if mass <= 1e-12:
#         return 1.0, 1.0

#     mean = float(np.sum(weights * x) / mass)
#     mean = float(np.clip(mean, eps, 1.0 - eps))
#     var = float(np.sum(weights * (x - mean) ** 2) / mass)

#     # A Beta distribution requires var < mean * (1 - mean). Keep a small
#     # numerical margin and avoid an excessively concentrated component.
#     max_var = max(mean * (1.0 - mean), 1e-12)
#     var = float(np.clip(var, 1e-8, max_var * (1.0 - 1e-6)))
#     concentration = mean * (1.0 - mean) / var - 1.0
#     concentration = float(np.clip(concentration, 1e-3, 1e6))

#     alpha = max(mean * concentration, eps)
#     beta = max((1.0 - mean) * concentration, eps)
#     return float(alpha), float(beta)


# def _beta_log_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
#     if scipy_betaln is None:
#         raise ImportError(
#             "dcs_gate_calibration='beta_mixture' requires scipy.special.betaln."
#         )
#     x = np.asarray(x, dtype=np.float64)
#     return (
#         (alpha - 1.0) * np.log(x)
#         + (beta - 1.0) * np.log1p(-x)
#         - scipy_betaln(alpha, beta)
#     )


# def _initialize_beta_mixture_responsibilities(
#     x: np.ndarray,
#     low_q: float,
#     high_q: float,
# ) -> np.ndarray:
#     """Deterministic soft initialization for the high-signal component."""
#     q_lo, q_hi = np.quantile(x, [low_q, high_q])
#     if q_hi <= q_lo + 1e-12:
#         # Fall back to centered min-max scaling when quantiles are tied.
#         x_min, x_max = float(np.min(x)), float(np.max(x))
#         if x_max <= x_min + 1e-12:
#             return np.full_like(x, 0.5, dtype=np.float64)
#         return np.clip((x - x_min) / (x_max - x_min), 1e-3, 1.0 - 1e-3)
#     return np.clip((x - q_lo) / (q_hi - q_lo), 1e-3, 1.0 - 1e-3)


# def fit_two_component_beta_mixture_gate(
#     signal: np.ndarray,
#     max_iter: int = 200,
#     tol: float = 1e-6,
#     eps: float = 1e-4,
#     fit_samples: int = 200_000,
#     seed: int = 0,
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """Fit a two-component Beta mixture and return high-component posterior G_i.

#     The component with the larger Beta mean alpha / (alpha + beta) is defined
#     as the junction-like component. Its posterior responsibility becomes the
#     state-level gate G_i. Multiple deterministic initializations are evaluated,
#     and the fit with the highest log-likelihood is retained.
#     """
#     if scipy_betaln is None or scipy_logsumexp is None:
#         raise ImportError(
#             "dcs_gate_calibration='beta_mixture' requires scipy.special."
#         )

#     raw = np.asarray(signal, dtype=np.float64).reshape(-1)
#     finite_mask = np.isfinite(raw)
#     if not np.all(finite_mask):
#         raise ValueError("Non-finite values found in JG-IQL gate signal.")
#     if raw.size < 2:
#         return np.zeros_like(raw, dtype=np.float32), {
#             "status": "degenerate_too_few_samples",
#             "fit_size": int(raw.size),
#         }

#     # J_i is theoretically bounded in [0, 1]. Clip only for Beta log-density;
#     # the original un-clipped values are retained for diagnostics.
#     x_all = np.clip(raw, eps, 1.0 - eps)
#     raw_range = float(np.max(raw) - np.min(raw))
#     if raw_range <= 1e-10:
#         return np.zeros_like(raw, dtype=np.float32), {
#             "status": "degenerate_constant_signal",
#             "fit_size": int(raw.size),
#             "signal_min": float(np.min(raw)),
#             "signal_max": float(np.max(raw)),
#         }

#     # Fitting on a capped random subset keeps preprocessing practical for very
#     # large offline datasets; posterior inference is still performed on all N.
#     if x_all.size > int(fit_samples):
#         rng = np.random.default_rng(seed)
#         fit_idx = rng.choice(x_all.size, size=int(fit_samples), replace=False)
#         x_fit = x_all[fit_idx]
#     else:
#         x_fit = x_all

#     # Different soft quantile splits make the fit less sensitive to one
#     # initialization while staying deterministic.
#     init_quantiles = ((0.25, 0.75), (0.35, 0.65), (0.45, 0.55))
#     best = None

#     for low_q, high_q in init_quantiles:
#         r_high = _initialize_beta_mixture_responsibilities(x_fit, low_q, high_q)
#         responsibilities = np.stack([1.0 - r_high, r_high], axis=1)
#         prev_ll = -np.inf

#         for iteration in range(int(max_iter)):
#             mixture_weights = np.mean(responsibilities, axis=0)
#             mixture_weights = np.clip(mixture_weights, 1e-6, 1.0)
#             mixture_weights /= np.sum(mixture_weights)

#             params = [
#                 _weighted_beta_moments(x_fit, responsibilities[:, k], eps)
#                 for k in range(2)
#             ]

#             log_joint = np.empty((x_fit.shape[0], 2), dtype=np.float64)
#             for k, (alpha, beta) in enumerate(params):
#                 log_joint[:, k] = (
#                     np.log(mixture_weights[k]) + _beta_log_pdf(x_fit, alpha, beta)
#                 )

#             log_norm = scipy_logsumexp(log_joint, axis=1)
#             ll = float(np.sum(log_norm))
#             responsibilities = np.exp(log_joint - log_norm[:, None])

#             if np.isfinite(prev_ll):
#                 relative_change = abs(ll - prev_ll) / max(abs(prev_ll), 1.0)
#                 if relative_change < tol:
#                     break
#             prev_ll = ll

#         means = np.asarray(
#             [alpha / (alpha + beta) for alpha, beta in params], dtype=np.float64
#         )
#         candidate = {
#             "ll": ll,
#             "iterations": int(iteration + 1),
#             "weights": mixture_weights.copy(),
#             "params": tuple(params),
#             "means": means,
#         }
#         if best is None or candidate["ll"] > best["ll"]:
#             best = candidate

#     assert best is not None
#     high_component = int(np.argmax(best["means"]))
#     low_component = 1 - high_component

#     # Infer posterior responsibilities for every dataset state.
#     log_joint_all = np.empty((x_all.shape[0], 2), dtype=np.float64)
#     for k, (alpha, beta) in enumerate(best["params"]):
#         log_joint_all[:, k] = (
#             np.log(best["weights"][k]) + _beta_log_pdf(x_all, alpha, beta)
#         )
#     log_norm_all = scipy_logsumexp(log_joint_all, axis=1)
#     posterior = np.exp(log_joint_all - log_norm_all[:, None])
#     gate = np.clip(posterior[:, high_component], 0.0, 1.0).astype(np.float32)

#     low_alpha, low_beta = best["params"][low_component]
#     high_alpha, high_beta = best["params"][high_component]
#     diagnostics = {
#         "status": "ok",
#         "fit_size": int(x_fit.size),
#         "full_size": int(x_all.size),
#         "iterations": int(best["iterations"]),
#         "log_likelihood": float(best["ll"]),
#         "low_weight": float(best["weights"][low_component]),
#         "high_weight": float(best["weights"][high_component]),
#         "low_alpha": float(low_alpha),
#         "low_beta": float(low_beta),
#         "high_alpha": float(high_alpha),
#         "high_beta": float(high_beta),
#         "low_mean": float(best["means"][low_component]),
#         "high_mean": float(best["means"][high_component]),
#         "mean_gap": float(
#             best["means"][high_component] - best["means"][low_component]
#         ),
#         "posterior_mean": float(np.mean(gate)),
#         "posterior_gt_05_frac": float(np.mean(gate >= 0.5)),
#         "signal_min": float(np.min(raw)),
#         "signal_max": float(np.max(raw)),
#     }
#     return gate, diagnostics


# def build_data_adaptive_gate(
#     config: TrainConfig,
#     gate_signal: np.ndarray,
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """Build G_i using either the automatic Beta mixture or legacy percentile ramp."""
#     if config.dcs_gate_calibration == "percentile":
#         gate = covp.gate_from_junction(
#             gate_signal,
#             config.dcs_gate_low_percentile,
#             config.dcs_gate_high_percentile,
#         ).astype(np.float32)
#         return gate, {
#             "status": "legacy_percentile",
#             "low_percentile": float(config.dcs_gate_low_percentile),
#             "high_percentile": float(config.dcs_gate_high_percentile),
#         }

#     try:
#         gate, diagnostics = fit_two_component_beta_mixture_gate(
#             gate_signal,
#             max_iter=config.dcs_beta_mixture_max_iter,
#             tol=config.dcs_beta_mixture_tol,
#             eps=config.dcs_beta_mixture_eps,
#             fit_samples=config.dcs_beta_mixture_fit_samples,
#             seed=config.seed,
#         )
#     except Exception as exc:
#         diagnostics = {
#             "status": "fit_failed",
#             "error": f"{type(exc).__name__}: {exc}",
#         }
#         gate = np.zeros_like(np.asarray(gate_signal).reshape(-1), dtype=np.float32)

#     if diagnostics.get("status") != "ok":
#         if config.dcs_beta_mixture_fallback == "percentile":
#             gate = covp.gate_from_junction(
#                 gate_signal,
#                 config.dcs_gate_low_percentile,
#                 config.dcs_gate_high_percentile,
#             ).astype(np.float32)
#             diagnostics["fallback"] = "percentile"
#             diagnostics["fallback_low_percentile"] = float(
#                 config.dcs_gate_low_percentile
#             )
#             diagnostics["fallback_high_percentile"] = float(
#                 config.dcs_gate_high_percentile
#             )
#         else:
#             gate = np.zeros_like(np.asarray(gate_signal).reshape(-1), dtype=np.float32)
#             diagnostics["fallback"] = "zero"

#     return gate, diagnostics



# def build_prototype_indexed_replay_fields(
#     config: TrainConfig,
#     profile: Dict[str, Any],
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Create minimal ReplayBuffer fields plus the actual state gate G_i.

#     The prototype-indexed neighbor search is used only during preprocessing to
#     compute JG-IQL statistics. The trainer itself still reads only
#     batch["junction"] when constructing tau(s), so no N x k graph is kept in
#     the ReplayBuffer.
#     """
#     gate = np.asarray(profile["gate"], dtype=np.float32).reshape(-1)
#     n = gate.shape[0]
#     dummy_indices = np.arange(n, dtype=np.int32)[:, None]
#     dummy_weights = gate[:, None].astype(np.float32)

#     density_conf = np.asarray(profile["density_conf"], dtype=np.float32)
#     action_div_conf = np.asarray(profile["action_div_conf"], dtype=np.float32)
#     disp_div_conf = np.asarray(profile["disp_div_conf"], dtype=np.float32)
#     gate_signal = np.asarray(profile["gate_signal"], dtype=np.float32)

#     try:
#         gate_diagnostics = json.loads(str(profile["gate_diagnostics_json"].reshape(-1)[0]))
#     except Exception:
#         gate_diagnostics = {"status": "unavailable"}

#     print(
#         "PI-JG support profile | "
#         f"gate_source={config.dcs_gate_source} | "
#         f"gate_calibration={config.dcs_gate_calibration} | "
#         f"mean_gate={float(np.mean(gate)):.4f} | "
#         f"active_frac={float(np.mean(gate > 1e-6)):.4f} | "
#         f"k={config.dcs_k}"
#     )
#     print(
#         "Prototype-index diagnostics | "
#         f"prototypes={int(np.asarray(profile['prototype_count']).reshape(-1)[0])} | "
#         f"nonempty={int(np.asarray(profile['nonempty_prototype_count']).reshape(-1)[0])} | "
#         f"mean_cell={float(np.asarray(profile['mean_cell_size']).reshape(-1)[0]):.1f} | "
#         f"max_cell={float(np.asarray(profile['max_cell_size']).reshape(-1)[0]):.0f} | "
#         f"candidate_mean={float(np.asarray(profile['mean_candidate_count']).reshape(-1)[0]):.1f} | "
#         f"candidate_p95={float(np.asarray(profile['p95_candidate_count']).reshape(-1)[0]):.1f} | "
#         f"candidate_max={float(np.asarray(profile['max_candidate_count']).reshape(-1)[0]):.0f} | "
#         f"mean_prototypes_used={float(np.asarray(profile['mean_prototypes_used']).reshape(-1)[0]):.2f} | "
#         f"expanded_frac={float(np.asarray(profile['expanded_prototype_frac']).reshape(-1)[0]):.4f}"
#     )
#     recall_samples = float(np.asarray(profile["exact_recall_samples"]).reshape(-1)[0])
#     if recall_samples > 0:
#         print(
#             "Approximation diagnostic | "
#             f"exact_recall_samples={int(recall_samples)} | "
#             f"recall@{config.dcs_k}="
#             f"{float(np.asarray(profile['exact_recall_at_k']).reshape(-1)[0]):.4f}"
#         )
#     print(
#         "Branch statistics | "
#         f"action_conf_mean={float(np.mean(action_div_conf)):.4f} | "
#         f"disp_conf_mean={float(np.mean(disp_div_conf)):.4f} | "
#         f"density_conf_mean={float(np.mean(density_conf)):.4f} | "
#         f"gate_signal_mean={float(np.mean(gate_signal)):.4f}"
#     )
#     if gate_diagnostics.get("status") == "ok":
#         print(
#             "Beta-mixture gate | "
#             f"low_mean={gate_diagnostics.get('low_mean', float('nan')):.4f} | "
#             f"high_mean={gate_diagnostics.get('high_mean', float('nan')):.4f} | "
#             f"mean_gap={gate_diagnostics.get('mean_gap', float('nan')):.4f} | "
#             f"P(high)>=0.5 frac="
#             f"{gate_diagnostics.get('posterior_gt_05_frac', float('nan')):.4f}"
#         )
#     else:
#         print(
#             "Gate calibration | "
#             f"status={gate_diagnostics.get('status')} | "
#             f"fallback={gate_diagnostics.get('fallback', 'none')}"
#         )

#     return dummy_indices, dummy_weights, gate


# class TransformEnv:
#     def __init__(
#         self,
#         env: gym.Env,
#         state_mean: Union[np.ndarray, float],
#         state_std: Union[np.ndarray, float],
#         reward_scale: float,
#     ):
#         self.env = env
#         self.state_mean = state_mean
#         self.state_std = state_std
#         self.reward_scale = reward_scale
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space

#     def __getattr__(self, name: str):
#         return getattr(self.env, name)

#     def _normalize_state(self, state):
#         return (state - self.state_mean) / self.state_std

#     def _scale_reward(self, reward):
#         return self.reward_scale * reward

#     def reset(self, *args, **kwargs):
#         reset_out = self.env.reset(*args, **kwargs)
#         if isinstance(reset_out, tuple) and len(reset_out) == 2:
#             state, info = reset_out
#             return self._normalize_state(state), info
#         return self._normalize_state(reset_out)

#     def step(self, action):
#         step_out = self.env.step(action)
#         if isinstance(step_out, tuple) and len(step_out) == 5:
#             state, reward, terminated, truncated, info = step_out
#             return self._normalize_state(state), self._scale_reward(reward), terminated, truncated, info
#         state, reward, done, info = step_out
#         return self._normalize_state(state), self._scale_reward(reward), done, info

#     def seed(self, seed: int):
#         if hasattr(self.env, "seed"):
#             return self.env.seed(seed)
#         return self.env.reset(seed=seed)


# def wrap_env(
#     env: gym.Env,
#     state_mean: Union[np.ndarray, float] = 0.0,
#     state_std: Union[np.ndarray, float] = 1.0,
#     reward_scale: float = 1.0,
# ) -> gym.Env:
#     return TransformEnv(env, state_mean=state_mean, state_std=state_std, reward_scale=reward_scale)


# def is_ogbench_env(env_name: str) -> bool:
#     return "singletask" in env_name or "oraclerep" in env_name


# def load_env_and_dataset(env_name: str, add_info: bool = False) -> Tuple[gym.Env, Dict[str, np.ndarray], str]:
#     if is_ogbench_env(env_name):
#         if ogbench is None:
#             raise ImportError(
#                 "OGBench environment requested, but the `ogbench` package is not installed."
#             )
#         if add_info:
#             try:
#                 # add_info=True exposes factored oracle fields such as
#                 # button_states (puzzle), qpos, and qvel for the ceiling test.
#                 env, train_dataset, _ = ogbench.make_env_and_datasets(env_name, add_info=True)
#             except TypeError:
#                 print("ogbench.make_env_and_datasets(..., add_info=True) unavailable; "
#                       "falling back to add_info=False (oracle metric source will fail).")
#                 env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
#         else:
#             env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
#         return env, train_dataset, "ogbench"

#     global d4rl
#     if d4rl is None:
#         try:
#             import d4rl as d4rl_module
#         except Exception as exc:
#             raise ImportError(
#                 "D4RL environment requested, but the `d4rl` package could not be imported."
#             ) from exc
#         d4rl = d4rl_module
#     env = gym.make(env_name)
#     return env, d4rl.qlearning_dataset(env), "d4rl"


# def reset_env(env: gym.Env, seed: Optional[int] = None):
#     if seed is not None:
#         try:
#             reset_out = env.reset(seed=seed)
#         except TypeError:
#             if hasattr(env, "seed"):
#                 env.seed(seed)
#             reset_out = env.reset()
#     else:
#         reset_out = env.reset()

#     if isinstance(reset_out, tuple) and len(reset_out) == 2:
#         return reset_out[0]
#     return reset_out


# def step_env(env: gym.Env, action: np.ndarray):
#     step_out = env.step(action)
#     if isinstance(step_out, tuple) and len(step_out) == 5:
#         next_state, reward, terminated, truncated, info = step_out
#         return next_state, reward, bool(terminated or truncated), info
#     next_state, reward, done, info = step_out
#     return next_state, reward, bool(done), info


# class ReplayBuffer:
#     """Replay buffer with minimal compatibility fields for the precomputed state gate.

#     For every stored transition i it additionally holds:
#       - neighbor_indices[i]: (k,) int32 indices of i's nearest dataset states
#       - neighbor_weights[i]: (k,) float32 legacy diagnostic weights in [0, 1]
#       - junction[i]:         (1,) float32 state gate G_i used as support confidence base

#     sample() gathers the neighbors' (s_j, a_j) so the trainer can evaluate
#     in-sample target-Q values on the whole pool in one batched forward pass.
#     """

#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         buffer_size: int,
#         neighbor_k: int,
#         device: Any,
#         embedding_dim: int = 0,
#     ):
#         self._buffer_size = buffer_size
#         self._neighbor_k = int(neighbor_k)
#         self._embedding_dim = int(embedding_dim)
#         if self._embedding_dim < 0:
#             raise ValueError("embedding_dim must be >= 0")
#         self._pointer = 0
#         self._size = 0
#         self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
#         self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
#         self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
#         self._embeddings = np.zeros((buffer_size, self._embedding_dim), dtype=np.float32)
#         self._next_embeddings = np.zeros((buffer_size, self._embedding_dim), dtype=np.float32)
#         self._neighbor_indices = np.zeros((buffer_size, self._neighbor_k), dtype=np.int32)
#         self._neighbor_weights = np.zeros((buffer_size, self._neighbor_k), dtype=np.float32)
#         self._junction = np.zeros((buffer_size, 1), dtype=np.float32)
#         self._device = device

#     def load_d4rl_dataset(
#         self,
#         data: Dict[str, np.ndarray],
#         neighbor_indices: Optional[np.ndarray] = None,
#         neighbor_weights: Optional[np.ndarray] = None,
#         junction: Optional[np.ndarray] = None,
#         embeddings: Optional[np.ndarray] = None,
#         next_embeddings: Optional[np.ndarray] = None,
#     ):
#         if self._size != 0:
#             raise ValueError("Trying to load data into non-empty replay buffer")
#         n_transitions = data["observations"].shape[0]
#         if n_transitions > self._buffer_size:
#             raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")

#         self._states[:n_transitions] = data["observations"].astype(np.float32)
#         self._actions[:n_transitions] = data["actions"].astype(np.float32)
#         self._rewards[:n_transitions] = data["rewards"][..., None].astype(np.float32)
#         self._next_states[:n_transitions] = data["next_observations"].astype(np.float32)
#         done_values = 1.0 - data["masks"] if "masks" in data else data["terminals"]
#         self._dones[:n_transitions] = done_values[..., None].astype(np.float32)

#         if self._embedding_dim > 0:
#             if embeddings is None or next_embeddings is None:
#                 raise ValueError(
#                     "embedding_dim > 0 requires both embeddings and next_embeddings"
#                 )
#             embeddings = np.asarray(embeddings, dtype=np.float32)
#             next_embeddings = np.asarray(next_embeddings, dtype=np.float32)
#             expected_shape = (n_transitions, self._embedding_dim)
#             if embeddings.shape != expected_shape:
#                 raise ValueError(
#                     f"embeddings must have shape {expected_shape}, got {embeddings.shape}"
#                 )
#             if next_embeddings.shape != expected_shape:
#                 raise ValueError(
#                     f"next_embeddings must have shape {expected_shape}, got {next_embeddings.shape}"
#                 )
#             self._embeddings[:n_transitions] = embeddings
#             self._next_embeddings[:n_transitions] = next_embeddings

#         if neighbor_indices is None:
#             # DCS disabled: pool collapses onto the sample itself (weights 0),
#             # which makes the trainer reduce exactly to standard IQL.
#             neighbor_indices = np.tile(
#                 np.arange(n_transitions, dtype=np.int32)[:, None], (1, self._neighbor_k)
#             )
#             neighbor_weights = np.zeros((n_transitions, self._neighbor_k), dtype=np.float32)
#             junction = np.zeros((n_transitions,), dtype=np.float32)

#         neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
#         neighbor_weights = np.asarray(neighbor_weights, dtype=np.float32)
#         junction = np.asarray(junction, dtype=np.float32).reshape(-1)
#         if neighbor_indices.shape != (n_transitions, self._neighbor_k):
#             raise ValueError(
#                 f"neighbor_indices must have shape {(n_transitions, self._neighbor_k)}, "
#                 f"got {neighbor_indices.shape}"
#             )
#         if neighbor_weights.shape != (n_transitions, self._neighbor_k):
#             raise ValueError(
#                 f"neighbor_weights must have shape {(n_transitions, self._neighbor_k)}, "
#                 f"got {neighbor_weights.shape}"
#             )
#         if junction.shape[0] != n_transitions:
#             raise ValueError("junction must have one scalar per transition")
#         if neighbor_indices.min() < 0 or neighbor_indices.max() >= n_transitions:
#             raise ValueError("neighbor_indices out of range for the loaded dataset")

#         self._neighbor_indices[:n_transitions] = neighbor_indices
#         self._neighbor_weights[:n_transitions] = np.clip(neighbor_weights, 0.0, 1.0)
#         self._junction[:n_transitions] = junction[:, None]

#         self._size += n_transitions
#         self._pointer = min(self._size, n_transitions)
#         print(f"Dataset size: {n_transitions} (compatibility neighbor axis={self._neighbor_k}; BC-JG uses no kNN graph)")

#     def sample(self, batch_size: int) -> TensorBatch:
#         indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
#         nbr_idx = self._neighbor_indices[indices]  # (B, k)
#         batch = {
#             "observations": self._states[indices],
#             "actions": self._actions[indices],
#             "rewards": self._rewards[indices],
#             "next_observations": self._next_states[indices],
#             "dones": self._dones[indices],
#             "embeddings": self._embeddings[indices],
#             "next_embeddings": self._next_embeddings[indices],
#             "neighbor_observations": self._states[nbr_idx],   # (B, k, Ds)
#             "neighbor_actions": self._actions[nbr_idx],       # (B, k, Da)
#             "neighbor_weights": self._neighbor_weights[indices],  # (B, k)
#             "junction": self._junction[indices],              # (B, 1)
#         }
#         return tree_to_device({k: jnp.asarray(v) for k, v in batch.items()}, self._device)


# def set_seed(seed: int, env: Optional[gym.Env] = None):
#     if env is not None:
#         try:
#             env.reset(seed=seed)
#         except TypeError:
#             if hasattr(env, "seed"):
#                 env.seed(seed)
#         if hasattr(env.action_space, "seed"):
#             env.action_space.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)


# def wandb_init(config: dict) -> None:
#     run = wandb.init(
#         config=config,
#         project=config["project"],
#         group=config["group"],
#         name=config["name"],
#         id=str(uuid.uuid4()),
#     )
#     run.log_code(".")


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
#     if not hasattr(env, "get_normalized_score"):
#         return np.full_like(np.asarray(eval_scores, dtype=np.float32), np.nan, dtype=np.float32)
#     return np.asarray(
#         [env.get_normalized_score(float(score)) * 100.0 for score in eval_scores],
#         dtype=np.float32,
#     )


# def mean_std_or_nan(values: np.ndarray) -> Tuple[float, float]:
#     values = np.asarray(values, dtype=np.float32)
#     finite_values = values[np.isfinite(values)]
#     if finite_values.size == 0:
#         return np.nan, np.nan
#     return float(np.mean(finite_values)), float(np.std(finite_values))


# def extract_success(info: Any) -> float:
#     if not isinstance(info, dict) or "success" not in info:
#         return np.nan
#     success = np.asarray(info["success"])
#     if success.size == 0:
#         return np.nan
#     return float(success.reshape(-1)[0])


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
#     # Preserves the reward preprocessing from the original IQL code.
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


# class PrototypeIndexedJunctionIQLJAX:
#     """Prototype-Indexed Approximate Junction-Gated IQL in JAX/Flax.

#     Q-function training and actor extraction mirror standard IQL. The only
#     departure is value learning: tau is state-dependent and is computed from
#     JG-IQL statistics using prototype-retrieved candidate pools followed by
#     exact local top-k refinement.
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
#         scv_tau_min: float = 0.5,
#         scv_tau_max: float = 0.8,
#         scv_support_power: float = 1.0,
#         scv_support_eps: float = 1e-8,
#         dcs_value_neighbor_weight: float = 1.0,
#         dcs_actor_neighbor_weight: float = 1.0,
#         actor_dropout: Optional[float] = None,
#         hidden_dim: int = 256,
#         n_hidden: int = 2,
#         embedding_dim: int = 0,
#         actor_use_embedding: bool = False,
#         q_use_embedding: bool = False,
#         v_use_embedding: bool = False,
#         temporal_encoder_params: Optional[Any] = None,
#         temporal_embedding_mean: Optional[np.ndarray] = None,
#         temporal_embedding_std: Optional[np.ndarray] = None,
#         temporal_hidden_dim: int = 256,
#         temporal_n_hidden: int = 2,
#         temporal_normalize: bool = True,
#         seed: int = 0,
#         device: Any = None,
#     ):
#         self.max_action = max_action
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_steps = max(int(max_steps), 1)
#         self.discount = discount
#         self.tau = tau
#         self.beta = beta
#         self.iql_tau = iql_tau
#         self.dcs_value_neighbor_weight = float(dcs_value_neighbor_weight)
#         self.dcs_actor_neighbor_weight = float(dcs_actor_neighbor_weight)
#         self.iql_deterministic = iql_deterministic
#         self.scv_tau_min = float(scv_tau_min)
#         self.scv_tau_max = float(scv_tau_max)
#         self.scv_support_power = float(scv_support_power)
#         self.scv_support_eps = float(scv_support_eps)
#         self.actor_dropout = actor_dropout
#         self.hidden_dim = int(hidden_dim)
#         self.n_hidden = int(n_hidden)
#         self.embedding_dim = int(embedding_dim)
#         self.actor_use_embedding = bool(actor_use_embedding)
#         self.q_use_embedding = bool(q_use_embedding)
#         self.v_use_embedding = bool(v_use_embedding)
#         self.uses_embedding_input = (
#             self.actor_use_embedding or self.q_use_embedding or self.v_use_embedding
#         )
#         self.device = device if device is not None else jax.devices()[0]

#         if self.hidden_dim <= 0:
#             raise ValueError("hidden_dim must be > 0")
#         if self.n_hidden <= 0:
#             raise ValueError("n_hidden must be > 0")
#         if self.uses_embedding_input and self.embedding_dim <= 0:
#             raise ValueError(
#                 "At least one network uses embeddings, but embedding_dim <= 0."
#             )
#         if self.uses_embedding_input and temporal_encoder_params is None:
#             raise ValueError(
#                 "Embedding-based network inputs require frozen temporal encoder parameters."
#             )

#         self.temporal_encoder_def = None
#         self.temporal_encoder_params = None
#         self.temporal_embedding_mean = None
#         self.temporal_embedding_std = None
#         if self.uses_embedding_input:
#             self.temporal_encoder_def = tmet.TemporalEncoder(
#                 embed_dim=self.embedding_dim,
#                 hidden_dim=int(temporal_hidden_dim),
#                 n_hidden=int(temporal_n_hidden),
#                 normalize=bool(temporal_normalize),
#             )
#             encoder_template = self.temporal_encoder_def.init(
#                 jax.random.PRNGKey(seed + 100003),
#                 jnp.zeros((1, state_dim), dtype=jnp.float32),
#             )["params"]
#             self.temporal_encoder_params = serialization.from_state_dict(
#                 encoder_template,
#                 temporal_encoder_params,
#             )
#             if temporal_embedding_mean is None:
#                 temporal_embedding_mean = np.zeros((self.embedding_dim,), dtype=np.float32)
#             if temporal_embedding_std is None:
#                 temporal_embedding_std = np.ones((self.embedding_dim,), dtype=np.float32)
#             self.temporal_embedding_mean = jnp.asarray(
#                 np.asarray(temporal_embedding_mean, dtype=np.float32).reshape(1, -1)
#             )
#             self.temporal_embedding_std = jnp.asarray(
#                 np.asarray(temporal_embedding_std, dtype=np.float32).reshape(1, -1)
#             )
#             if self.temporal_embedding_mean.shape[-1] != self.embedding_dim:
#                 raise ValueError("temporal_embedding_mean has the wrong dimension")
#             if self.temporal_embedding_std.shape[-1] != self.embedding_dim:
#                 raise ValueError("temporal_embedding_std has the wrong dimension")

#         if iql_deterministic:
#             self.actor_def = DeterministicPolicy(
#                 action_dim=action_dim,
#                 hidden_dim=self.hidden_dim,
#                 n_hidden=self.n_hidden,
#                 dropout=actor_dropout,
#             )
#         else:
#             self.actor_def = GaussianPolicy(
#                 action_dim=action_dim,
#                 hidden_dim=self.hidden_dim,
#                 n_hidden=self.n_hidden,
#                 dropout=actor_dropout,
#             )
#         self.q_def = TwinQ(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)
#         self.v_def = ValueFunction(hidden_dim=self.hidden_dim, n_hidden=self.n_hidden)

#         self.q_tx = optax.adam(qf_lr)
#         self.v_tx = optax.adam(vf_lr)
#         actor_lr_schedule = optax.cosine_decay_schedule(
#             init_value=actor_lr,
#             decay_steps=self.max_steps,
#             alpha=0.0,
#         )
#         self.actor_tx = optax.adam(actor_lr_schedule)

#         key = jax.random.PRNGKey(seed)
#         key_actor, key_q, key_v, actor_key = jax.random.split(key, 4)
#         actor_input_dim = self.embedding_dim if self.actor_use_embedding else state_dim
#         q_input_dim = self.embedding_dim if self.q_use_embedding else state_dim
#         v_input_dim = self.embedding_dim if self.v_use_embedding else state_dim
#         dummy_actor_state = jnp.zeros((1, actor_input_dim), dtype=jnp.float32)
#         dummy_q_state = jnp.zeros((1, q_input_dim), dtype=jnp.float32)
#         dummy_v_state = jnp.zeros((1, v_input_dim), dtype=jnp.float32)
#         dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

#         actor_params = self.actor_def.init(
#             key_actor, dummy_actor_state, training=False
#         )["params"]
#         q_params = self.q_def.init(key_q, dummy_q_state, dummy_action)["params"]
#         v_params = self.v_def.init(key_v, dummy_v_state)["params"]

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
#         if self.uses_embedding_input:
#             self.temporal_encoder_params = tree_to_device(
#                 self.temporal_encoder_params, self.device
#             )
#             self.temporal_embedding_mean = tree_to_device(
#                 self.temporal_embedding_mean, self.device
#             )
#             self.temporal_embedding_std = tree_to_device(
#                 self.temporal_embedding_std, self.device
#             )
#         self._train_step = self._build_train_step()
#         self._actor_refit_step = self._build_actor_refit_step()

#     def _encode_observations(self, observations: jnp.ndarray) -> jnp.ndarray:
#         """Apply the frozen temporal encoder and the training-time standardizer."""
#         if not self.uses_embedding_input:
#             raise RuntimeError("Temporal encoder requested but no network uses embeddings.")
#         z = self.temporal_encoder_def.apply(
#             {"params": self.temporal_encoder_params}, observations
#         )
#         return (z - self.temporal_embedding_mean) / self.temporal_embedding_std

#     def _apply_actor(self, actor_params: Any, observations: jnp.ndarray, training: bool, rng: Optional[jnp.ndarray] = None):
#         if self.actor_dropout is not None and training:
#             return self.actor_def.apply(
#                 {"params": actor_params},
#                 observations,
#                 training=training,
#                 rngs={"dropout": rng},
#             )
#         return self.actor_def.apply({"params": actor_params}, observations, training=training)

#     @staticmethod
#     def _build_pool(batch: TensorBatch) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#         """Stack the sample itself with its kNN neighbors.

#         Returns:
#             pool_observations (B, 1+K, Ds)  [:, 0] is the sample itself
#             pool_actions      (B, 1+K, Da)
#             neighbor_weights  (B, K) clipped to [0, 1]
#         """
#         observations = batch["observations"]
#         actions = batch["actions"]
#         neighbor_observations = batch["neighbor_observations"]
#         neighbor_actions = batch["neighbor_actions"]
#         neighbor_weights = jnp.clip(batch["neighbor_weights"], 0.0, 1.0)
#         pool_observations = jnp.concatenate(
#             [observations[:, None, :], neighbor_observations], axis=1
#         )
#         pool_actions = jnp.concatenate([actions[:, None, :], neighbor_actions], axis=1)
#         return pool_observations, pool_actions, neighbor_weights

#     @staticmethod
#     def _pool_weights(neighbor_weights: jnp.ndarray, neighbor_scale: float) -> jnp.ndarray:
#         """Normalized pool weights: self gets mass 1, neighbor j gets
#         neighbor_scale * w_ij, then the row is normalized to sum to 1.
#         neighbor_scale = 0 or w = 0 recovers the plain single-sample loss."""
#         ones = jnp.ones_like(neighbor_weights[:, :1])
#         w = jnp.concatenate([ones, neighbor_scale * neighbor_weights], axis=1)
#         return w / jnp.sum(w, axis=1, keepdims=True)

#     def _build_train_step(self):
#         q_apply = self.q_def.apply
#         v_apply = self.v_def.apply
#         q_tx = self.q_tx
#         v_tx = self.v_tx
#         actor_tx = self.actor_tx
#         discount = self.discount
#         tau = self.tau
#         beta = self.beta
#         tau_min = self.scv_tau_min
#         tau_max = self.scv_tau_max
#         support_power = self.scv_support_power
#         support_eps = self.scv_support_eps
#         iql_deterministic = self.iql_deterministic
#         actor_use_embedding = self.actor_use_embedding
#         q_use_embedding = self.q_use_embedding
#         v_use_embedding = self.v_use_embedding
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
#             embeddings = batch["embeddings"]
#             next_embeddings = batch["next_embeddings"]
#             dones = jnp.squeeze(batch["dones"], axis=-1)
#             neighbor_weights = jnp.clip(batch["neighbor_weights"], 0.0, 1.0)

#             q_observations = embeddings if q_use_embedding else observations
#             v_observations = embeddings if v_use_embedding else observations
#             next_v_observations = (
#                 next_embeddings if v_use_embedding else next_observations
#             )
#             actor_observations = (
#                 embeddings if actor_use_embedding else observations
#             )

#             # Junction-gated certificate c(s).
#             # BC-JG-IQL does NOT compute a kNN graph or kernel ESS. The ReplayBuffer stores the
#             # state-level gate G_i in the legacy "junction" field.
#             gate_conf = jnp.clip(jnp.squeeze(batch["junction"], axis=-1), 0.0, 1.0)
#             support_conf = gate_conf ** support_power
#             tau_s = tau_min + (tau_max - tau_min) * support_conf

#             # Legacy diagnostics only. They are not used to compute tau(s).
#             k = jnp.maximum(jnp.asarray(neighbor_weights.shape[1], dtype=jnp.float32), 1.0)
#             neighbor_mass = jnp.sum(neighbor_weights, axis=1)
#             gate_effective_count_proxy = gate_conf * k

#             tq1, tq2 = q_apply(
#                 {"params": state.q_target_params}, q_observations, actions
#             )
#             self_target_q = jnp.minimum(tq1, tq2)

#             next_v = v_apply({"params": state.v_params}, next_v_observations)
#             target_q_for_backup = rewards + (1.0 - dones) * discount * next_v
#             old_v = v_apply({"params": state.v_params}, v_observations)
#             adv = self_target_q - old_v
#             exp_adv = jnp.minimum(
#                 jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
#             )

#             # ----- Support-certified expectile V update.
#             def v_loss_fn(v_params):
#                 v = v_apply({"params": v_params}, v_observations)
#                 diff = jax.lax.stop_gradient(self_target_q) - v
#                 expectile_weight = jnp.abs(tau_s - (diff < 0.0).astype(jnp.float32))
#                 value_loss = jnp.mean(expectile_weight * diff ** 2)
#                 return value_loss, v

#             (value_loss, v), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
#             v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
#             v_params = optax.apply_updates(state.v_params, v_updates)

#             # ----- Q update: standard IQL TD backup.
#             def q_loss_fn(q_params):
#                 q1, q2 = q_apply({"params": q_params}, q_observations, actions)
#                 target = jax.lax.stop_gradient(target_q_for_backup)
#                 q_loss = 0.5 * (jnp.mean((q1 - target) ** 2) + jnp.mean((q2 - target) ** 2))
#                 return q_loss, (q1, q2)

#             (q_loss, (q1, q2)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(state.q_params)
#             q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
#             q_params = optax.apply_updates(state.q_params, q_updates)
#             q_target_params = soft_update(q_params, state.q_target_params, tau)

#             actor_key, dropout_key = jax.random.split(state.actor_key)

#             # ----- Standard IQL AWR actor: no neighbor action copying.
#             def actor_loss_fn(actor_params):
#                 policy_out = apply_actor(
#                     actor_params,
#                     actor_observations,
#                     training=True,
#                     rng=dropout_key,
#                 )
#                 if iql_deterministic:
#                     bc_loss = jnp.sum((policy_out - actions) ** 2, axis=-1)
#                     policy_mean = policy_out
#                     log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
#                 else:
#                     mean, log_std = policy_out
#                     std = jnp.exp(log_std)
#                     diff_a = (actions - mean) / std
#                     log_prob = -0.5 * (diff_a ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
#                     bc_loss = -jnp.sum(log_prob, axis=-1)
#                     policy_mean = mean
#                     log_std_mean = jnp.mean(log_std)
#                 actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_loss)
#                 return actor_loss, (bc_loss, policy_mean, log_std_mean)

#             (actor_loss, (bc_loss, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
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
#                 "bc_loss_mean": jnp.mean(bc_loss),
#                 "policy_mean": jnp.mean(policy_mean),
#                 "policy_log_std_mean": log_std_mean,
#                 # JG diagnostics
#                 "support_conf_mean": jnp.mean(support_conf),
#                 "support_conf_min": jnp.min(support_conf),
#                 "support_conf_max": jnp.max(support_conf),
#                 "tau_s_mean": jnp.mean(tau_s),
#                 "tau_s_min": jnp.min(tau_s),
#                 "tau_s_max": jnp.max(tau_s),
#                 # JG diagnostics. No ESS is computed; the effective-count value is a
#                 # gate-based proxy kept for W&B compatibility across ablations.
#                 "gate_conf_mean": jnp.mean(gate_conf),
#                 "gate_conf_min": jnp.min(gate_conf),
#                 "gate_conf_max": jnp.max(gate_conf),
#                 "gate_effective_count_proxy_mean": jnp.mean(gate_effective_count_proxy),
#                 "neighbor_effective_count_mean": jnp.mean(gate_effective_count_proxy),
#                 "neighbor_mass_mean": jnp.mean(neighbor_mass),
#                 "neighbor_weight_mean": jnp.mean(neighbor_weights),
#                 "gate_active_frac": jnp.mean(
#                     (gate_conf > 0.0).astype(jnp.float32)
#                 ),
#                 "active_edge_frac": jnp.mean(
#                     (neighbor_weights > 0.0).astype(jnp.float32)
#                 ),
#             }
#             return new_state, log_dict

#         return train_step

#     def train(self, batch: TensorBatch) -> Dict[str, float]:
#         self.state, log_dict = self._train_step(self.state, batch)
#         return {key: float(jax.device_get(value)) for key, value in log_dict.items()}

#     def actor_act(self, actor_params: Any, state: np.ndarray) -> np.ndarray:
#         state_jnp = tree_to_device(
#             jnp.asarray(state.reshape(1, -1), dtype=jnp.float32), self.device
#         )
#         actor_input = (
#             self._encode_observations(state_jnp)
#             if self.actor_use_embedding
#             else state_jnp
#         )
#         policy_out = self._apply_actor(actor_params, actor_input, training=False)
#         action = policy_out if self.iql_deterministic else policy_out[0]
#         action = jnp.clip(self.max_action * action, -self.max_action, self.max_action)
#         return np.asarray(jax.device_get(action))[0]

#     def eval_actor(
#         self,
#         env: gym.Env,
#         actor_params: Any,
#         n_episodes: int,
#         seed: int,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         episode_rewards = []
#         episode_successes = []
#         for episode_idx in range(n_episodes):
#             state, done = reset_env(env, seed=seed if episode_idx == 0 else None), False
#             episode_reward = 0.0
#             episode_success = np.nan
#             while not done:
#                 action = self.actor_act(actor_params, state)
#                 state, reward, done, info = step_env(env, action)
#                 episode_reward += reward
#                 step_success = extract_success(info)
#                 if np.isfinite(step_success):
#                     episode_success = step_success
#             episode_rewards.append(episode_reward)
#             episode_successes.append(episode_success)
#         return (
#             np.asarray(episode_rewards, dtype=np.float32),
#             np.asarray(episode_successes, dtype=np.float32),
#         )

#     def _build_actor_refit_step(self):
#         q_apply = self.q_def.apply
#         v_apply = self.v_def.apply
#         actor_tx = self.actor_tx
#         beta = self.beta
#         iql_deterministic = self.iql_deterministic
#         actor_use_embedding = self.actor_use_embedding
#         q_use_embedding = self.q_use_embedding
#         v_use_embedding = self.v_use_embedding
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
#             embeddings = batch["embeddings"]
#             actions = batch["actions"]

#             q_observations = embeddings if q_use_embedding else observations
#             v_observations = embeddings if v_use_embedding else observations
#             actor_observations = (
#                 embeddings if actor_use_embedding else observations
#             )

#             tq1, tq2 = q_apply(
#                 {"params": iql_state.q_target_params}, q_observations, actions
#             )
#             target_q = jnp.minimum(tq1, tq2)
#             v = v_apply({"params": iql_state.v_params}, v_observations)
#             adv = target_q - v
#             exp_adv = jnp.minimum(
#                 jnp.exp(beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX
#             )

#             actor_key, dropout_key = jax.random.split(actor_state.key)

#             def actor_loss_fn(actor_params):
#                 policy_out = apply_actor(
#                     actor_params,
#                     actor_observations,
#                     training=True,
#                     rng=dropout_key,
#                 )
#                 if iql_deterministic:
#                     bc_loss = jnp.sum((policy_out - actions) ** 2, axis=-1)
#                     policy_mean = policy_out
#                     log_std_mean = jnp.asarray(np.nan, dtype=jnp.float32)
#                 else:
#                     mean, log_std = policy_out
#                     std = jnp.exp(log_std)
#                     diff_a = (actions - mean) / std
#                     log_prob = -0.5 * (diff_a ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
#                     bc_loss = -jnp.sum(log_prob, axis=-1)
#                     policy_mean = mean
#                     log_std_mean = jnp.mean(log_std)
#                 actor_loss = jnp.mean(jax.lax.stop_gradient(exp_adv) * bc_loss)
#                 return actor_loss, (bc_loss, policy_mean, log_std_mean)

#             (actor_loss, (bc_loss, policy_mean, log_std_mean)), actor_grads = jax.value_and_grad(
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
#                 "bc_loss": jnp.mean(bc_loss),
#                 "adv_mean": jnp.mean(adv),
#                 "adv_min": jnp.min(adv),
#                 "adv_max": jnp.max(adv),
#                 "exp_adv_mean": jnp.mean(exp_adv),
#                 "exp_adv_max": jnp.max(exp_adv),
#                 "target_q_mean": jnp.mean(target_q),
#                 "v_mean": jnp.mean(v),
#                 "policy_mean": jnp.mean(policy_mean),
#                 "policy_log_std_mean": log_std_mean,
#                 "pool_neighbor_mass_actor": jnp.asarray(0.0, dtype=jnp.float32),
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
#         save_dir: Optional[Union[str, Path]] = None,
#         log_wandb: bool = False,
#         log_extra: Optional[Dict[str, Any]] = None,
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
#             f"{prefix}/final_success_rate": np.nan,
#             f"{prefix}/final_success_std": np.nan,
#             f"{prefix}/best_score_mean": np.nan,
#             f"{prefix}/best_score_std": np.nan,
#             f"{prefix}/best_d4rl_normalized_score_mean": np.nan,
#             f"{prefix}/best_d4rl_normalized_score_std": np.nan,
#             f"{prefix}/best_success_rate": np.nan,
#             f"{prefix}/best_success_std": np.nan,
#             f"{prefix}/inner_eval_steps": [],
#             f"{prefix}/inner_score_mean": [],
#             f"{prefix}/inner_score_std": [],
#             f"{prefix}/inner_d4rl_normalized_score_mean": [],
#             f"{prefix}/inner_d4rl_normalized_score_std": [],
#             f"{prefix}/inner_success_rate": [],
#             f"{prefix}/inner_success_std": [],
#         }
#         if steps <= 0:
#             return actor_state, refit_log

#         best_eval_metric_mean = -np.inf
#         save_dir_path = Path(save_dir) if save_dir is not None else None
#         if save_dir_path is not None:
#             save_dir_path.mkdir(parents=True, exist_ok=True)
#         log_extra = {} if log_extra is None else dict(log_extra)

#         def save_refit_snapshot(
#             current_actor_state: ActorState,
#             current_refit_log: Dict[str, Any],
#             fit_step: int,
#             is_best: bool,
#         ) -> None:
#             if save_dir_path is None:
#                 return

#             logs_payload = {
#                 **log_extra,
#                 "refit_step": int(fit_step),
#                 **current_refit_log,
#             }
#             logs_path = save_dir_path / "fit_eval_logs.npz"
#             latest_actor_path = save_dir_path / "latest_actor.pkl"
#             actor_payload = serialization.to_state_dict(current_actor_state.params)
#             save_pickle(latest_actor_path, actor_payload)
#             save_logs_npz([logs_payload], str(logs_path))

#             if is_best:
#                 save_pickle(save_dir_path / "best_actor.pkl", actor_payload)

#             if log_wandb and wandb.run is not None:
#                 wandb.save(str(logs_path), policy="now")
#                 wandb.save(str(latest_actor_path), policy="now")
#                 if is_best:
#                     wandb.save(str(save_dir_path / "best_actor.pkl"), policy="now")

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
#                 eval_scores, eval_successes = self.eval_actor(
#                     eval_env,
#                     actor_state.params,
#                     n_episodes=eval_episodes,
#                     seed=eval_seed,
#                 )
#                 normalized_eval_scores = normalize_episode_scores(eval_env, eval_scores)

#                 eval_score_mean = float(np.mean(eval_scores))
#                 eval_score_std = float(np.std(eval_scores))
#                 normalized_eval_score_mean, normalized_eval_score_std = mean_std_or_nan(normalized_eval_scores)
#                 success_rate, success_std = mean_std_or_nan(eval_successes)

#                 refit_log[f"{prefix}/inner_eval_steps"].append(int(fit_step))
#                 refit_log[f"{prefix}/inner_score_mean"].append(eval_score_mean)
#                 refit_log[f"{prefix}/inner_score_std"].append(eval_score_std)
#                 refit_log[f"{prefix}/inner_d4rl_normalized_score_mean"].append(normalized_eval_score_mean)
#                 refit_log[f"{prefix}/inner_d4rl_normalized_score_std"].append(normalized_eval_score_std)
#                 refit_log[f"{prefix}/inner_success_rate"].append(success_rate)
#                 refit_log[f"{prefix}/inner_success_std"].append(success_std)
#                 refit_log[f"{prefix}/final_score_mean"] = eval_score_mean
#                 refit_log[f"{prefix}/final_score_std"] = eval_score_std
#                 refit_log[f"{prefix}/final_d4rl_normalized_score_mean"] = normalized_eval_score_mean
#                 refit_log[f"{prefix}/final_d4rl_normalized_score_std"] = normalized_eval_score_std
#                 refit_log[f"{prefix}/final_success_rate"] = success_rate
#                 refit_log[f"{prefix}/final_success_std"] = success_std

#                 eval_metric_mean = success_rate if np.isfinite(success_rate) else normalized_eval_score_mean
#                 is_best = np.isfinite(eval_metric_mean) and eval_metric_mean > best_eval_metric_mean
#                 if is_best:
#                     best_eval_metric_mean = eval_metric_mean
#                     refit_log[f"{prefix}/best_score_mean"] = eval_score_mean
#                     refit_log[f"{prefix}/best_score_std"] = eval_score_std
#                     refit_log[f"{prefix}/best_d4rl_normalized_score_mean"] = normalized_eval_score_mean
#                     refit_log[f"{prefix}/best_d4rl_normalized_score_std"] = normalized_eval_score_std
#                     refit_log[f"{prefix}/best_success_rate"] = success_rate
#                     refit_log[f"{prefix}/best_success_std"] = success_std

#                 save_refit_snapshot(
#                     current_actor_state=actor_state,
#                     current_refit_log=refit_log,
#                     fit_step=fit_step,
#                     is_best=is_best,
#                 )

#                 print(
#                     f"[{prefix}:dcs_awbc] step {fit_step}/{steps}: "
#                     f"loss={step_log['loss']:.4f}, bc={step_log['bc_loss']:.4f}, "
#                     f"adv={step_log['adv_mean']:.4f}, exp_adv={step_log['exp_adv_mean']:.4f}, "
#                     f"eval_mean={eval_score_mean:.3f}, eval_std={eval_score_std:.3f}, "
#                     f"d4rl_normalized_mean={normalized_eval_score_mean:.3f}, "
#                     f"d4rl_normalized_std={normalized_eval_score_std:.3f}, "
#                     f"success_rate={success_rate:.3f}"
#                 )

#         return actor_state, refit_log

#     def state_dict(self) -> Dict[str, Any]:
#         payload = {
#             "iql_state": serialization.to_state_dict(self.state),
#             "initial_actor_params": serialization.to_state_dict(self.initial_actor_params),
#             "initial_actor_opt_state": serialization.to_state_dict(self.initial_actor_opt_state),
#             "initial_actor_key": serialization.to_state_dict(self.initial_actor_key),
#             "iql_deterministic": self.iql_deterministic,
#             "actor_use_embedding": self.actor_use_embedding,
#             "q_use_embedding": self.q_use_embedding,
#             "v_use_embedding": self.v_use_embedding,
#             "embedding_dim": self.embedding_dim,
#         }
#         if self.uses_embedding_input:
#             payload["temporal_encoder_params"] = serialization.to_state_dict(
#                 self.temporal_encoder_params
#             )
#             payload["temporal_embedding_mean"] = np.asarray(
#                 jax.device_get(self.temporal_embedding_mean), dtype=np.float32
#             )
#             payload["temporal_embedding_std"] = np.asarray(
#                 jax.device_get(self.temporal_embedding_std), dtype=np.float32
#             )
#         return payload

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
#         if self.uses_embedding_input and "temporal_encoder_params" in state_dict:
#             self.temporal_encoder_params = serialization.from_state_dict(
#                 self.temporal_encoder_params,
#                 state_dict["temporal_encoder_params"],
#             )
#         if self.uses_embedding_input and "temporal_embedding_mean" in state_dict:
#             self.temporal_embedding_mean = jnp.asarray(
#                 state_dict["temporal_embedding_mean"], dtype=jnp.float32
#             )
#         if self.uses_embedding_input and "temporal_embedding_std" in state_dict:
#             self.temporal_embedding_std = jnp.asarray(
#                 state_dict["temporal_embedding_std"], dtype=jnp.float32
#             )
#         self.state = tree_to_device(self.state, self.device)
#         self.initial_actor_params = tree_to_device(self.initial_actor_params, self.device)
#         self.initial_actor_opt_state = tree_to_device(self.initial_actor_opt_state, self.device)
#         self.initial_actor_key = tree_to_device(self.initial_actor_key, self.device)
#         if self.uses_embedding_input:
#             self.temporal_encoder_params = tree_to_device(
#                 self.temporal_encoder_params, self.device
#             )
#             self.temporal_embedding_mean = tree_to_device(
#                 self.temporal_embedding_mean, self.device
#             )
#             self.temporal_embedding_std = tree_to_device(
#                 self.temporal_embedding_std, self.device
#             )


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
#     """Return (run_dir, checkpoint_path) for a saved checkpoint.

#     Supported load_model formats:

#     1. Direct checkpoint file:
#        path/to/checkpoint.pkl

#     2. Direct run directory:
#        path/to/run_dir/
#        where path/to/run_dir/checkpoint.pkl exists

#     3. Parent directory that contains env/seed subdirectory:
#        path/to/base_dir/
#        where path/to/base_dir/{run_name}/{seed}/checkpoint.pkl exists

#     Note: DC-IQL checkpoints are loadable as well (identical architectures);
#     only the actor/value training rules differ.
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


# def load_run_config_for_refit(
#     current_config: TrainConfig,
#     loaded_run_dir: Union[str, Path],
# ) -> TrainConfig:
#     """Load saved config.yaml from the checkpoint run dir for refit mode.

#     Priority:
#         saved run config.yaml < explicit CLI flags

#     This reconstructs the original training env/model/preprocessing settings,
#     while allowing any CLI-provided field to override the saved config.
#     Unknown saved keys (e.g. legacy DC-IQL fields) are silently dropped.
#     """
#     loaded_run_dir = Path(loaded_run_dir)
#     saved_config_path = loaded_run_dir / "config.yaml"

#     if not saved_config_path.exists():
#         raise FileNotFoundError(
#             f"mode='refit' expects saved run config at: {saved_config_path}"
#         )

#     with open(saved_config_path, "r") as f:
#         saved_raw = yaml.safe_load(f) or {}

#     config_fields = set(TrainConfig.__dataclass_fields__.keys())
#     saved_kwargs = {
#         key: _coerce_hparam_value(value)
#         for key, value in saved_raw.items()
#         if key in config_fields
#     }

#     # 1. Start from the original saved training config.
#     loaded_config = TrainConfig(**saved_kwargs)

#     # 2. Override with explicitly provided CLI fields.
#     cli_overrides = _cli_overridden_fields()
#     current_config_dict = asdict(current_config)

#     applied_cli_overrides = []
#     for key in sorted(cli_overrides):
#         if key not in config_fields:
#             continue
#         setattr(loaded_config, key, current_config_dict[key])
#         applied_cli_overrides.append(key)

#     # 3. These must be forced for refit regardless of saved training config.
#     loaded_config.mode = "refit"
#     loaded_config.load_model = current_config.load_model

#     # In refit mode, do not reuse the original training checkpoint output path.
#     # Actor refit outputs are saved under loaded_run_dir / actor_refit_dir_name.
#     loaded_config.checkpoints_path = None

#     refresh_algorithm_names(loaded_config)
#     validate_config(loaded_config)

#     print(f"Loaded saved run config for refit from: {saved_config_path}")
#     if applied_cli_overrides:
#         print(
#             "Applied explicit CLI overrides on top of saved config: "
#             + ", ".join(applied_cli_overrides)
#         )

#     return loaded_config


# @pyrallis.wrap()
# def train(config: TrainConfig):
#     refit_only = config.mode == "refit"

#     loaded_run_dir: Optional[Path] = None
#     checkpoint_path: Optional[Path] = None

#     if refit_only:
#         if config.load_model == "":
#             raise ValueError("refit mode requires --load_model")

#         # First resolve the checkpoint using the current CLI config.
#         # This lets --load_model be either checkpoint.pkl, a run dir, or a parent dir.
#         loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
#             config.load_model,
#             run_name=config.name,
#             seed=config.seed,
#         )

#         # Then replace config with the original saved run config,
#         # while preserving refit-specific CLI/runtime fields.
#         config = load_run_config_for_refit(
#             current_config=config,
#             loaded_run_dir=loaded_run_dir,
#         )

#     else:
#         config = apply_env_hyperparams(config)
#         config = finalize_checkpoint_path(config)

#     jax_device = select_jax_device(config.device)
#     env, dataset, dataset_backend = load_env_and_dataset(
#         config.env,
#         add_info=(config.use_dcs and config.dcs_metric_source == "oracle"),
#     )

#     if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
#         raise ValueError(
#             f"{ALGORITHM_NAME}-JAX currently supports vector observations/actions only; "
#             f"got observation_space={env.observation_space}, action_space={env.action_space}."
#         )

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     if config.normalize_reward and dataset_backend == "d4rl":
#         modify_reward(dataset, config.env)
#     elif config.normalize_reward:
#         print("Skipping D4RL reward normalization for non-D4RL dataset.")

#     if config.normalize:
#         state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
#     else:
#         state_mean, state_std = 0, 1

#     dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
#     dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
#     env = wrap_env(env, state_mean=state_mean, state_std=state_std)

#     network_uses_embedding = (
#         config.actor_use_embedding or config.q_use_embedding or config.v_use_embedding
#     )
#     temporal_representation: Optional[Dict[str, Any]] = None
#     if config.dcs_metric_source == "temporal" and (config.use_dcs or network_uses_embedding):
#         temporal_representation = load_or_build_temporal_representation(
#             config, dataset, device=jax_device
#         )

#     # ----- PI-JG profile: prototype retrieval + exact local top-k refinement ---
#     neighbor_indices = None
#     neighbor_weights = None
#     junction = None
#     buffer_neighbor_k = 1
#     if config.use_dcs:
#         profile = load_or_compute_prototype_indexed_profile(
#             config,
#             dataset,
#             device=jax_device,
#             temporal_representation=temporal_representation,
#         )
#         neighbor_indices, neighbor_weights, junction = build_prototype_indexed_replay_fields(
#             config, profile
#         )
#     else:
#         print(
#             "use_dcs=False: prototype-indexed gate disabled; tau(s)=scv_tau_min. "
#             "Set scv_tau_min=scv_tau_max for a plain IQL baseline."
#         )

#     embedding_dim = (
#         int(temporal_representation["embeddings"].shape[1])
#         if temporal_representation is not None and network_uses_embedding
#         else 0
#     )
#     replay_buffer = ReplayBuffer(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         buffer_size=config.buffer_size,
#         neighbor_k=buffer_neighbor_k,
#         device=jax_device,
#         embedding_dim=embedding_dim,
#     )
#     replay_buffer.load_d4rl_dataset(
#         dataset,
#         neighbor_indices=neighbor_indices,
#         neighbor_weights=neighbor_weights,
#         junction=junction,
#         embeddings=(
#             temporal_representation["embeddings"]
#             if temporal_representation is not None and network_uses_embedding
#             else None
#         ),
#         next_embeddings=(
#             temporal_representation["next_embeddings"]
#             if temporal_representation is not None and network_uses_embedding
#             else None
#         ),
#     )

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
#     run_mode_name = "Actor refit" if refit_only else "Training"
#     print(f"{run_mode_name} {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {seed}")
#     print(
#         f"support_tau=[{config.scv_tau_min}, {config.scv_tau_max}] | use_prototype_index={config.use_dcs} | "
#         f"prototypes={config.bc_num_prototypes} | near_prototypes={config.bc_nearest_prototypes} | "
#         f"top_k={config.dcs_k} | "
#         f"support_power={config.scv_support_power} | metric={config.dcs_metric_source}"
#     )
#     print(
#         "network inputs | "
#         f"actor={'embedding' if config.actor_use_embedding else 'observation'} | "
#         f"Q={'embedding' if config.q_use_embedding else 'observation'} | "
#         f"V={'embedding' if config.v_use_embedding else 'observation'}"
#     )
#     if config.use_dcs:
#         print("support confidence: c_i = G_i^p; prototype index only accelerates neighbor retrieval; no neighbor Q/action pooling")
#         if config.dcs_metric_source == "oracle":
#             keys = resolve_oracle_keys(config)
#             print(
#                 f"*** ORACLE CEILING TEST: prototype index built in ORACLE space "
#                 f"(keys={list(keys) if keys else 'auto'}); training still uses "
#                 f"standard observations. ***"
#             )
#         elif config.dcs_metric_source == "temporal":
#             print(
#                 f"*** TEMPORAL METRIC: prototype index built in a LEARNED temporal "
#                 f"space phi(s) (dim={config.dcs_temporal_dim}, "
#                 f"horizon={config.dcs_temporal_horizon}, "
#                 f"steps={config.dcs_temporal_steps}); data-only, deployable. "
#                 f"Each of actor/Q/V can independently consume phi(s). ***"
#             )
#         else:
#             print("metric source: observation (prototype retrieval + exact local top-k)")
#     print("---------------------------------------")

#     trainer = PrototypeIndexedJunctionIQLJAX(
#         max_action=max_action,
#         state_dim=state_dim,
#         action_dim=action_dim,
#         max_steps=max(int(config.max_timesteps), 1),
#         qf_lr=config.qf_lr,
#         vf_lr=config.vf_lr,
#         actor_lr=config.actor_lr,
#         discount=config.discount,
#         tau=config.tau,
#         beta=config.beta,
#         iql_tau=config.iql_tau,
#         iql_deterministic=config.iql_deterministic,
#         scv_tau_min=config.scv_tau_min,
#         scv_tau_max=config.scv_tau_max,
#         scv_support_power=config.scv_support_power,
#         scv_support_eps=config.scv_support_eps,
#         dcs_value_neighbor_weight=0.0,
#         dcs_actor_neighbor_weight=0.0,
#         actor_dropout=config.actor_dropout,
#         hidden_dim=config.hidden_dim,
#         n_hidden=config.n_hidden,
#         embedding_dim=embedding_dim,
#         actor_use_embedding=config.actor_use_embedding,
#         q_use_embedding=config.q_use_embedding,
#         v_use_embedding=config.v_use_embedding,
#         temporal_encoder_params=(
#             temporal_representation["encoder_params"]
#             if temporal_representation is not None and network_uses_embedding
#             else None
#         ),
#         temporal_embedding_mean=(
#             temporal_representation["embedding_mean"]
#             if temporal_representation is not None and network_uses_embedding
#             else None
#         ),
#         temporal_embedding_std=(
#             temporal_representation["embedding_std"]
#             if temporal_representation is not None and network_uses_embedding
#             else None
#         ),
#         temporal_hidden_dim=config.dcs_temporal_hidden,
#         temporal_n_hidden=config.dcs_temporal_layers,
#         temporal_normalize=config.dcs_temporal_normalize,
#         seed=seed,
#         device=jax_device,
#     )

#     if config.load_model != "":
#         if checkpoint_path is None or loaded_run_dir is None:
#             loaded_run_dir, checkpoint_path = resolve_checkpoint_path(
#                 config.load_model,
#                 run_name=config.name,
#                 seed=config.seed,
#             )
#         print(f"Loading checkpoint from: {checkpoint_path}")
#         checkpoint = load_pickle(checkpoint_path)
#         trainer.load_state_dict(checkpoint)

#     if config.log_wandb:
#         wandb_init(asdict(config))

#     if refit_only:
#         if loaded_run_dir is None:
#             raise ValueError("refit mode requires --load_model")

#         actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
#         actor_refit_dir.mkdir(parents=True, exist_ok=True)
#         print("---------------------------------------")
#         print(f"Actor refit from saved checkpoint ({ALGORITHM_NAME} standard IQL AWR)")
#         print("Q/V are frozen; only pi is optimized.")
#         print(
#             "Refit schedule uses shared fields: "
#             f"max_timesteps={config.max_timesteps}, "
#             f"batch_size={config.batch_size}, "
#             f"eval_freq={config.eval_freq}"
#         )
#         print(f"Saving actor refit outputs to: {actor_refit_dir}")
#         print("---------------------------------------")

#         actor_state = trainer.make_initial_actor_state()
#         loaded_checkpoint_for_log = str(loaded_run_dir / "checkpoint.pkl")

#         refit_actor_state, refit_log = trainer.fit_actor(
#             replay_buffer=replay_buffer,
#             actor_state=actor_state,
#             steps=config.max_timesteps,
#             batch_size=config.batch_size,
#             eval_env=env,
#             eval_episodes=config.n_episodes,
#             eval_seed=config.seed,
#             eval_interval=config.eval_freq,
#             prefix="actor_refit",
#             save_dir=actor_refit_dir,
#             log_wandb=config.log_wandb,
#             log_extra={"loaded_checkpoint": loaded_checkpoint_for_log},
#         )

#         save_pickle(
#             actor_refit_dir / "final_actor.pkl",
#             serialization.to_state_dict(refit_actor_state.params),
#         )
#         save_logs_npz(
#             [{"loaded_checkpoint": loaded_checkpoint_for_log, **refit_log}],
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
#             eval_scores, eval_successes = trainer.eval_actor(
#                 env,
#                 trainer.state.actor_params,
#                 n_episodes=config.n_episodes,
#                 seed=config.seed,
#             )
#             normalized_eval_scores = normalize_episode_scores(env, eval_scores)
#             normalized_eval_score_mean, normalized_eval_score_std = mean_std_or_nan(normalized_eval_scores)
#             success_rate, success_std = mean_std_or_nan(eval_successes)

#             eval_log: Dict[str, Any] = {
#                 "timestep": int(t + 1),
#                 "eval/reward_mean": float(np.mean(eval_scores)),
#                 "eval/reward_std": float(np.std(eval_scores)),
#                 "eval/d4rl_normalized_score_mean": normalized_eval_score_mean,
#                 "eval/d4rl_normalized_score_std": normalized_eval_score_std,
#                 "eval/success_rate": success_rate,
#                 "eval/success_std": success_std,
#             }
#             upsert_eval_log(eval_logs, eval_log)

#             print(
#                 f"Evaluation over {config.n_episodes} episodes: "
#                 f"reward={eval_log['eval/reward_mean']:.3f} ± {eval_log['eval/reward_std']:.3f}, "
#                 f"d4rl_normalized={eval_log['eval/d4rl_normalized_score_mean']:.3f} ± "
#                 f"{eval_log['eval/d4rl_normalized_score_std']:.3f}, "
#                 f"success_rate={eval_log['eval/success_rate']:.3f} ± "
#                 f"{eval_log['eval/success_std']:.3f}"
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

#     if config.checkpoints_path is not None and config.save_final_model:
#         checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
#         save_pickle(checkpoint_path, trainer.state_dict())
#         print("---------------------------------------")
#         print(f"Saved final checkpoint to: {checkpoint_path}")
#         print("(needed for mode='refit'; enable with --save_final_model True)")
#         print("---------------------------------------")

#     if config.checkpoints_path is not None:
#         save_and_upload_eval_logs(
#             eval_logs=eval_logs,
#             checkpoints_path=config.checkpoints_path,
#             log_wandb=config.log_wandb,
#         )


# if __name__ == "__main__":
#     train()
