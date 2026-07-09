# moment_gated_iql_jax.py
#
# MG-IQL: Moment-Gated Implicit Q-Learning (JAX/Flax).
#
# Practical kNN-free variant of junction-gated IQL.
#
# The algorithm first learns (or loads) a temporal representation
#     z_i = phi(s_i)
# from dataset transitions. It then estimates local action/transition diversity
# directly with a conditional-moment predictor, without constructing a kNN graph:
#
#     mu_a(z) ~= E[a | z]
#     q_a(z)  ~= E[||a||^2 | z]
#     mu_u(z) ~= E[u | z],
#
# where u = (s' - s) / (||s' - s|| + eps).
#
# Raw diversity estimates are
#     A_raw(z) = sqrt(max(q_a(z) - ||mu_a(z)||^2, 0))
#     P_raw(z) = clip(1 - ||mu_u(z)||, 0, 1).
#
# After percentile scaling,
#     J_i = A_i, P_i, or sqrt(A_i * P_i),
#     G_i = ramp_gate(J_i),
#     c_i = G_i^p,
#     tau_i = tau_min + (tau_max - tau_min) * c_i.
#
# No per-sample kNN search, no distance kernel K_ij, no density term D_i,
# no effective-support ESS, no dynamics edge gate, and no neighbor pooling are used.
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

import temporal_metric as tmet

d4rl = None

try:
    import ogbench
except ImportError:
    ogbench = None

TensorBatch = Dict[str, jnp.ndarray]

ALGORITHM_NAME = "MG-IQL"
ALGORITHM_FULL_NAME = "Moment-Gated Implicit Q-Learning"

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

    # ----- MG-IQL: kNN-free moment-gated value learning -------------------
    # G_i directly controls the state-dependent expectile:
    #   c_i = G_i ** scv_support_power
    #   tau_i = scv_tau_min + (scv_tau_max - scv_tau_min) * c_i
    scv_tau_min: float = 0.5
    scv_tau_max: float = 0.8
    scv_support_power: float = 1.0
    scv_support_eps: float = 1e-8

    # Master switch. False -> tau(s)=scv_tau_min.
    use_dcs: bool = True

    # Diversity signal used to build G_i. Density-based modes are deliberately
    # absent because MG-IQL is designed to remove kNN density estimation.
    #   action            : J_i = A_i
    #   displacement      : J_i = P_i
    #   product_diversity : J_i = sqrt(A_i * P_i)
    dcs_gate_source: str = "product_diversity"

    # Backward-compatible alias. If dcs_gate_source is not explicitly set,
    # dcs_diversity_mode maps action/displacement/product -> corresponding mode.
    dcs_diversity_mode: str = "product"

    # Percentile scaling of raw moment statistics into [0, 1].
    dcs_percentile_low: float = 5.0
    dcs_percentile_high: float = 95.0

    # Ramp gate over the selected diversity signal J.
    dcs_gate_low_percentile: float = 60.0
    dcs_gate_high_percentile: float = 95.0

    # Cache root for temporal embeddings and moment profiles.
    dcs_profile_path: Optional[str] = None
    dcs_force_recompute: bool = False
    dcs_cache_by_seed: bool = False

    # ----- Temporal representation phi(s) ---------------------------------
    # MG-IQL defaults to temporal embeddings. "observation" is also supported
    # as a no-encoder ablation, but neither mode constructs a kNN graph.
    dcs_metric_source: str = "temporal"  # temporal | observation
    dcs_temporal_dim: int = 32
    dcs_temporal_hidden: int = 256
    dcs_temporal_layers: int = 2
    dcs_temporal_steps: int = 50_000
    dcs_temporal_batch: int = 512
    dcs_temporal_lr: float = 3e-4
    dcs_temporal_temperature: float = 0.1
    dcs_temporal_horizon: int = 1
    dcs_temporal_normalize: bool = True
    dcs_temporal_standardize: bool = False
    dcs_temporal_force_recompute: bool = False
    # Keep representation seed independent of policy-training seed so one
    # temporal embedding/cache can be reused across multiple RL seeds.
    dcs_temporal_seed: int = 0

    # ----- Conditional-moment predictor -----------------------------------
    # A small smoothed network predicts E[a|z], E[||a||^2|z], and E[u|z].
    moment_hidden_dim: int = 128
    moment_layers: int = 2
    moment_steps: int = 30_000
    moment_batch_size: int = 1024
    moment_lr: float = 3e-4
    moment_weight_decay: float = 1e-5
    # Gaussian perturbation prevents point-wise memorization and makes the
    # predictor estimate local conditional moments around z.
    moment_input_noise_std: float = 0.05
    moment_action_mean_coef: float = 1.0
    moment_action_second_coef: float = 1.0
    moment_displacement_coef: float = 1.0
    moment_inference_chunk_size: int = 65_536
    moment_seed: int = 0
    moment_force_recompute: bool = False
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
    assert 0.0 <= config.discount <= 1.0
    assert 0.0 <= config.tau <= 1.0
    assert config.beta >= 0.0
    assert 0.0 <= config.iql_tau <= 1.0
    assert 0.0 <= config.scv_tau_min <= config.scv_tau_max <= 1.0
    assert config.scv_support_power > 0.0
    assert config.scv_support_eps > 0.0
    assert config.dcs_gate_source in ("action", "displacement", "product_diversity")
    assert config.dcs_diversity_mode in ("action", "displacement", "product")
    assert 0.0 <= config.dcs_percentile_low < config.dcs_percentile_high <= 100.0
    assert 0.0 <= config.dcs_gate_low_percentile <= config.dcs_gate_high_percentile <= 100.0
    assert config.dcs_metric_source in ("temporal", "observation")
    if config.dcs_profile_path is not None:
        assert config.dcs_profile_path != ""
    if config.dcs_metric_source == "temporal":
        assert config.dcs_temporal_dim >= 1
        assert config.dcs_temporal_hidden >= 1
        assert config.dcs_temporal_layers >= 1
        assert config.dcs_temporal_steps >= 0
        assert config.dcs_temporal_batch >= 2, "InfoNCE needs batch_size >= 2"
        assert config.dcs_temporal_lr > 0.0
        assert config.dcs_temporal_temperature > 0.0
        assert config.dcs_temporal_horizon >= 1
    assert config.moment_hidden_dim >= 1
    assert config.moment_layers >= 1
    assert config.moment_steps >= 1
    assert config.moment_batch_size >= 1
    assert config.moment_lr > 0.0
    assert config.moment_weight_decay >= 0.0
    assert config.moment_input_noise_std >= 0.0
    assert config.moment_action_mean_coef >= 0.0
    assert config.moment_action_second_coef >= 0.0
    assert config.moment_displacement_coef >= 0.0
    assert config.moment_inference_chunk_size >= 1
    if config.actor_dropout is not None:
        assert 0.0 <= config.actor_dropout < 1.0
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
        "dc_density_model_path": "dcs_profile_path",
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

    if "dcs_gate_source" not in applied_fields and "dcs_gate_source" not in cli_overrides:
        legacy_to_source = {
            "action": "action",
            "displacement": "displacement",
            "product": "product_diversity",
        }
        config.dcs_gate_source = legacy_to_source.get(
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
# kNN-free temporal moment preprocessing
# ---------------------------------------------------------------------------

MOMENT_CACHE_VERSION = 1


def _safe_path_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def percentile_confidence(
    values: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(np.percentile(values, percentile_low))
    hi = float(np.percentile(values, percentile_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite percentile threshold in moment profile.")
    if hi <= lo + 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gate_from_signal(
    signal: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    lo = float(np.percentile(signal, low_percentile))
    hi = float(np.percentile(signal, high_percentile))
    if hi <= lo + 1e-12:
        # Degenerate signal: avoid turning the entire dataset fully optimistic.
        return np.zeros_like(signal, dtype=np.float32)
    return np.clip((signal - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_temporal_signature(
    config: TrainConfig,
    observations_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    return {
        "cache_version": tmet.TEMPORAL_CACHE_VERSION,
        "env": config.env,
        "seed": int(config.dcs_temporal_seed),
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
        return root / env_name / f"seed_{config.dcs_temporal_seed}" / filename
    return root / env_name / filename


def build_metric_space(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any = None,
) -> np.ndarray:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    if config.dcs_metric_source == "observation":
        print(
            f"Moment metric space: normalized observations, dim={observations.shape[1]} "
            "(no temporal encoder, no kNN graph)."
        )
        return observations

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
            seed=config.dcs_temporal_seed,
            next_index=next_index,
            device=device,
        )
        if cache_path is not None:
            tmet.save_temporal_embeddings_cache(cache_path, embeddings, signature)

    if config.dcs_temporal_standardize:
        embeddings = tmet.standardize_embeddings(embeddings)

    print(
        f"Temporal metric space: dim={embeddings.shape[1]} | "
        f"normalize={config.dcs_temporal_normalize} | "
        f"standardize={config.dcs_temporal_standardize} | "
        f"horizon={config.dcs_temporal_horizon} | "
        f"representation_seed={config.dcs_temporal_seed} | no kNN graph"
    )
    return np.asarray(embeddings, dtype=np.float32)


class ConditionalMomentNetwork(nn.Module):
    action_dim: int
    displacement_dim: int
    hidden_dim: int = 128
    n_hidden: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = z
        for _ in range(self.n_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        action_mean = nn.Dense(self.action_dim, name="action_mean")(x)
        action_second_raw = nn.Dense(1, name="action_second")(x)
        action_second = jax.nn.softplus(jnp.squeeze(action_second_raw, axis=-1))
        displacement_mean = nn.Dense(self.displacement_dim, name="displacement_mean")(x)
        return action_mean, action_second, displacement_mean


def _normalized_displacements(
    observations: np.ndarray,
    next_observations: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    delta = np.asarray(next_observations, dtype=np.float32) - np.asarray(observations, dtype=np.float32)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.where(norm > eps, delta / np.maximum(norm, eps), 0.0).astype(np.float32)


def build_moment_signature(
    config: TrainConfig,
    observations_shape: Tuple[int, ...],
    actions_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    temporal_sig = build_temporal_signature(config, observations_shape=observations_shape)
    return {
        "cache_version": MOMENT_CACHE_VERSION,
        "env": config.env,
        "metric_source": config.dcs_metric_source,
        "temporal_signature": tmet.signature_hash(temporal_sig)
        if config.dcs_metric_source == "temporal" else None,
        "observations_shape": tuple(observations_shape),
        "actions_shape": tuple(actions_shape),
        "hidden_dim": int(config.moment_hidden_dim),
        "layers": int(config.moment_layers),
        "steps": int(config.moment_steps),
        "batch_size": int(config.moment_batch_size),
        "lr": float(config.moment_lr),
        "weight_decay": float(config.moment_weight_decay),
        "input_noise_std": float(config.moment_input_noise_std),
        "action_mean_coef": float(config.moment_action_mean_coef),
        "action_second_coef": float(config.moment_action_second_coef),
        "displacement_coef": float(config.moment_displacement_coef),
        "seed": int(config.moment_seed),
    }


def get_moment_profile_cache_path(
    config: TrainConfig,
    observations_shape: Tuple[int, ...],
    actions_shape: Tuple[int, ...],
) -> Optional[Path]:
    if config.dcs_profile_path is None:
        return None
    root = Path(config.dcs_profile_path)
    env_name = _safe_path_name(config.env)
    sig = build_moment_signature(config, observations_shape, actions_shape)
    sig_hash = tmet.signature_hash(sig)
    filename = f"moment_profile_{sig_hash}.npz"
    if config.dcs_cache_by_seed:
        return root / env_name / f"moment_seed_{config.moment_seed}" / filename
    return root / env_name / filename


def train_conditional_moment_predictor(
    config: TrainConfig,
    embeddings: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    next_observations: np.ndarray,
    device: Any,
) -> Tuple[ConditionalMomentNetwork, Any]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    disp_targets = _normalized_displacements(observations, next_observations)
    action_second_targets = np.sum(actions ** 2, axis=-1).astype(np.float32)

    n, embed_dim = embeddings.shape
    action_dim = actions.shape[1]
    displacement_dim = disp_targets.shape[1]
    network = ConditionalMomentNetwork(
        action_dim=action_dim,
        displacement_dim=displacement_dim,
        hidden_dim=config.moment_hidden_dim,
        n_hidden=config.moment_layers,
    )
    key = jax.random.PRNGKey(config.moment_seed)
    key_init, key_loop = jax.random.split(key)
    params = network.init(
        key_init,
        jnp.zeros((1, embed_dim), dtype=jnp.float32),
    )["params"]
    tx = optax.adamw(
        learning_rate=config.moment_lr,
        weight_decay=config.moment_weight_decay,
    )
    opt_state = tx.init(params)
    params = tree_to_device(params, device)
    opt_state = tree_to_device(opt_state, device)

    normalize_noisy = bool(
        config.dcs_metric_source == "temporal"
        and config.dcs_temporal_normalize
        and not config.dcs_temporal_standardize
    )
    noise_std = float(config.moment_input_noise_std)
    mean_coef = float(config.moment_action_mean_coef)
    second_coef = float(config.moment_action_second_coef)
    disp_coef = float(config.moment_displacement_coef)

    @jax.jit
    def train_step(params, opt_state, key, z, a, a2, u):
        key, noise_key = jax.random.split(key)
        if noise_std > 0.0:
            z_in = z + noise_std * jax.random.normal(noise_key, z.shape, dtype=z.dtype)
            if normalize_noisy:
                z_in = z_in / (jnp.linalg.norm(z_in, axis=-1, keepdims=True) + 1e-8)
        else:
            z_in = z

        def loss_fn(p):
            pred_mean, pred_second, pred_u = network.apply({"params": p}, z_in)
            mean_loss = jnp.mean((pred_mean - a) ** 2)
            second_loss = jnp.mean((pred_second - a2) ** 2)
            disp_loss = jnp.mean((pred_u - u) ** 2)
            total = mean_coef * mean_loss + second_coef * second_loss + disp_coef * disp_loss
            aux = (mean_loss, second_loss, disp_loss)
            return total, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, key, loss, aux

    rng = np.random.default_rng(config.moment_seed)
    batch_size = min(int(config.moment_batch_size), n)
    print(
        f"Training conditional-moment predictor: N={n}, embed_dim={embed_dim}, "
        f"action_dim={action_dim}, displacement_dim={displacement_dim}, "
        f"steps={config.moment_steps}, batch={batch_size}, noise_std={noise_std}"
    )

    for step in range(int(config.moment_steps)):
        idx = rng.integers(0, n, size=batch_size)
        z_b = tree_to_device(jnp.asarray(embeddings[idx]), device)
        a_b = tree_to_device(jnp.asarray(actions[idx]), device)
        a2_b = tree_to_device(jnp.asarray(action_second_targets[idx]), device)
        u_b = tree_to_device(jnp.asarray(disp_targets[idx]), device)
        params, opt_state, key_loop, loss, aux = train_step(
            params, opt_state, key_loop, z_b, a_b, a2_b, u_b
        )
        if step == 0 or (step + 1) % 5_000 == 0 or step + 1 == config.moment_steps:
            mean_loss, second_loss, disp_loss = [float(jax.device_get(x)) for x in aux]
            print(
                f"Moment step {step + 1}/{config.moment_steps} | "
                f"loss={float(jax.device_get(loss)):.6f} | "
                f"mean={mean_loss:.6f} | second={second_loss:.6f} | disp={disp_loss:.6f}"
            )

    return network, params


def predict_moment_statistics(
    config: TrainConfig,
    network: ConditionalMomentNetwork,
    params: Any,
    embeddings: np.ndarray,
    device: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    chunk = int(config.moment_inference_chunk_size)
    action_spread_parts: List[np.ndarray] = []
    disp_parts: List[np.ndarray] = []

    @jax.jit
    def apply_fn(z):
        return network.apply({"params": params}, z)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        z = tree_to_device(jnp.asarray(embeddings[start:end]), device)
        action_mean, action_second, displacement_mean = apply_fn(z)
        action_mean = np.asarray(jax.device_get(action_mean), dtype=np.float32)
        action_second = np.asarray(jax.device_get(action_second), dtype=np.float32)
        displacement_mean = np.asarray(jax.device_get(displacement_mean), dtype=np.float32)

        action_var_trace = np.maximum(
            action_second - np.sum(action_mean ** 2, axis=-1), 0.0
        )
        action_spread = np.sqrt(action_var_trace).astype(np.float32)
        disp_dispersion = np.clip(
            1.0 - np.linalg.norm(displacement_mean, axis=-1), 0.0, 1.0
        ).astype(np.float32)
        action_spread_parts.append(action_spread)
        disp_parts.append(disp_dispersion)

    return (
        np.concatenate(action_spread_parts, axis=0),
        np.concatenate(disp_parts, axis=0),
    )


def build_moment_gate(
    config: TrainConfig,
    action_spread: np.ndarray,
    disp_dispersion: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    action_conf = percentile_confidence(
        action_spread, config.dcs_percentile_low, config.dcs_percentile_high
    )
    disp_conf = percentile_confidence(
        disp_dispersion, config.dcs_percentile_low, config.dcs_percentile_high
    )

    if config.dcs_gate_source == "action":
        signal = action_conf
    elif config.dcs_gate_source == "displacement":
        signal = disp_conf
    elif config.dcs_gate_source == "product_diversity":
        signal = np.sqrt(
            np.clip(action_conf, 0.0, 1.0) * np.clip(disp_conf, 0.0, 1.0)
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dcs_gate_source for MG-IQL: {config.dcs_gate_source}")

    gate = gate_from_signal(
        signal,
        config.dcs_gate_low_percentile,
        config.dcs_gate_high_percentile,
    )
    profile = {
        "action_spread": np.asarray(action_spread, dtype=np.float32),
        "disp_dispersion": np.asarray(disp_dispersion, dtype=np.float32),
        "action_conf": action_conf,
        "disp_conf": disp_conf,
        "signal": signal,
        "gate": gate,
    }
    return gate, profile


def load_or_compute_moment_gate(
    config: TrainConfig,
    dataset: Dict[str, np.ndarray],
    device: Any,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    cache_path = get_moment_profile_cache_path(
        config, observations.shape, actions.shape
    )

    if (
        cache_path is not None
        and cache_path.exists()
        and not config.moment_force_recompute
        and not config.dcs_force_recompute
    ):
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                action_spread = np.asarray(data["action_spread"], dtype=np.float32)
                disp_dispersion = np.asarray(data["disp_dispersion"], dtype=np.float32)
            if action_spread.shape[0] == observations.shape[0] and disp_dispersion.shape[0] == observations.shape[0]:
                gate, profile = build_moment_gate(config, action_spread, disp_dispersion)
                print(f"Loaded cached kNN-free moment profile from: {cache_path}")
                return gate, profile
        except Exception as exc:
            print(f"Ignoring invalid moment cache {cache_path}: {exc}")

    embeddings = build_metric_space(config, dataset, device=device)
    network, params = train_conditional_moment_predictor(
        config=config,
        embeddings=embeddings,
        actions=actions,
        observations=observations,
        next_observations=np.asarray(dataset["next_observations"], dtype=np.float32),
        device=device,
    )
    action_spread, disp_dispersion = predict_moment_statistics(
        config, network, params, embeddings, device
    )
    gate, profile = build_moment_gate(config, action_spread, disp_dispersion)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            action_spread=action_spread.astype(np.float32),
            disp_dispersion=disp_dispersion.astype(np.float32),
        )
        print(f"Saved kNN-free moment profile to: {cache_path}")

    print(
        "MG-IQL moment profile | "
        f"gate_source={config.dcs_gate_source} | "
        f"action_spread mean={float(np.mean(action_spread)):.4f} | "
        f"disp_dispersion mean={float(np.mean(disp_dispersion)):.4f} | "
        f"gate mean={float(np.mean(gate)):.4f} | "
        f"active-row frac={float(np.mean(gate > 1e-6)):.4f}"
    )
    return gate, profile

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
    """Offline replay buffer with one precomputed moment gate G_i per transition."""

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
        self._gate = np.zeros((buffer_size, 1), dtype=np.float32)
        self._device = device

    def load_d4rl_dataset(
        self,
        data: Dict[str, np.ndarray],
        gate: Optional[np.ndarray] = None,
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

        if gate is None:
            gate = np.zeros((n_transitions,), dtype=np.float32)
        gate = np.asarray(gate, dtype=np.float32).reshape(-1)
        if gate.shape[0] != n_transitions:
            raise ValueError("gate must have one scalar per transition")
        self._gate[:n_transitions] = np.clip(gate, 0.0, 1.0)[:, None]

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(
            f"Dataset size: {n_transitions} | kNN-free moment gate | "
            f"mean G={float(np.mean(gate)):.4f}"
        )

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        batch = {
            "observations": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._next_states[indices],
            "dones": self._dones[indices],
            "junction": self._gate[indices],
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


class MomentGatedIQLJAX:
    """Moment-Gated IQL in JAX/Flax.

    Q-function training and actor extraction mirror standard IQL. The only
    departure is value learning: tau(s) is controlled by a precomputed, kNN-free
    conditional-diversity gate G_i.
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
            # Moment-gated certificate c(s).
            # MG-IQL does NOT compute kernel ESS. The ReplayBuffer stores the
            # state-level moment gate G_i in the legacy "junction" field.
            gate_conf = jnp.clip(jnp.squeeze(batch["junction"], axis=-1), 0.0, 1.0)
            support_conf = gate_conf ** support_power
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
                # MG diagnostics
                "support_conf_mean": jnp.mean(support_conf),
                "support_conf_min": jnp.min(support_conf),
                "support_conf_max": jnp.max(support_conf),
                "tau_s_mean": jnp.mean(tau_s),
                "tau_s_min": jnp.min(tau_s),
                "tau_s_max": jnp.max(tau_s),
                # MG-IQL diagnostics: no kNN/ESS/neighbor tensors exist.
                "gate_conf_mean": jnp.mean(gate_conf),
                "gate_conf_min": jnp.min(gate_conf),
                "gate_conf_max": jnp.max(gate_conf),
                "gate_active_frac": jnp.mean((gate_conf > 0.0).astype(jnp.float32)),
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

    jax_device = select_jax_device(config.device)
    env, dataset, dataset_backend = load_env_and_dataset(
        config.env,
        add_info=False,
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

    # ----- MG-IQL preprocessing: temporal embedding + conditional moments -----
    gate = None
    if config.use_dcs:
        gate, _moment_profile = load_or_compute_moment_gate(
            config, dataset, device=jax_device
        )
    else:
        print(
            "use_dcs=False: moment gate disabled; tau(s)=scv_tau_min. "
            "Set scv_tau_min=scv_tau_max for a plain IQL baseline."
        )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=jax_device,
    )
    replay_buffer.load_d4rl_dataset(dataset, gate=gate)

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
    print(
        f"support_tau=[{config.scv_tau_min}, {config.scv_tau_max}] | "
        f"use_moment_gate={config.use_dcs} | support_power={config.scv_support_power} | "
        f"metric={config.dcs_metric_source} | gate_source={config.dcs_gate_source}"
    )
    if config.use_dcs:
        print(
            "support confidence: c_i = G_i^p from conditional action/transition moments; "
            "no kNN, no density, no kernel ESS, no dynamics gate, no neighbor pooling"
        )
        if config.dcs_metric_source == "temporal":
            print(
                f"*** TEMPORAL MOMENT METRIC: phi(s) dim={config.dcs_temporal_dim}, "
                f"horizon={config.dcs_temporal_horizon}, steps={config.dcs_temporal_steps}, "
                f"representation_seed={config.dcs_temporal_seed}; kNN-free. ***"
            )
        else:
            print("*** OBSERVATION MOMENT METRIC: normalized observation input; kNN-free. ***")
    print("---------------------------------------")

    trainer = MomentGatedIQLJAX(
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

    if config.checkpoints_path is not None and config.save_final_model:
        checkpoint_path = os.path.join(config.checkpoints_path, "checkpoint.pkl")
        save_pickle(checkpoint_path, trainer.state_dict())
        print("---------------------------------------")
        print(f"Saved final checkpoint to: {checkpoint_path}")
        print("(needed for mode='refit'; enable with --save_final_model True)")
        print("---------------------------------------")

    if config.checkpoints_path is not None:
        save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


if __name__ == "__main__":
    train()
