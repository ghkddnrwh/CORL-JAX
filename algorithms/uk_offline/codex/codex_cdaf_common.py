import copy
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_UK_OFFLINE_DIR = Path(__file__).resolve().parents[1]
if str(_UK_OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(_UK_OFFLINE_DIR))

import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
import yaml
from flax import serialization

import cdaf_jax_coverage_margin as coverage_base
import sa_cdaf_jax as base

TensorBatch = Dict[str, jnp.ndarray]


VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "codex_01_support_floor_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "exp_margin",
        "value_update_mode": "weighted_mse",
        "dense_bad_floor": 0.05,
        "max_weight_exponent": 0.25,
        "adv_margin": 0.05,
    },
    "codex_02_local_rank_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "local_rank_exp",
        "rank_quantile": 0.60,
        "rank_temperature": 0.25,
        "dense_bad_floor": 0.03,
        "max_weight_exponent": 0.35,
    },
    "codex_03_expectile_support_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "sigmoid",
        "value_update_mode": "expectile",
        "expectile_sparse": 0.50,
        "expectile_dense": 0.80,
        "sigmoid_temperature": 0.50,
        "dense_bad_floor": 0.15,
        "actor_fit_method": "weighted_bc",
        "policy_weight_exponent": 0.5,
    },
    "codex_04_rank_expectile_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "local_rank_exp",
        "value_update_mode": "expectile",
        "rank_quantile": 0.65,
        "rank_temperature": 0.35,
        "expectile_sparse": 0.50,
        "expectile_dense": 0.85,
        "dense_bad_floor": 0.10,
    },
    "codex_05_mild_cql_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "exp_margin",
        "value_update_mode": "weighted_mse",
        "dense_bad_floor": 0.05,
        "cql_alpha": 0.05,
        "cql_temperature": 1.0,
        "cql_support_weight": 1.0,
        "adv_margin": 0.05,
    },
    "codex_06_adaptive_margin_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "exp_margin",
        "value_update_mode": "weighted_mse",
        "dense_bad_floor": 0.04,
        "adv_margin": 0.02,
        "adaptive_margin_coef": 0.75,
        "rank_temperature": 0.35,
    },
    "codex_07_pessimistic_target_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "exp_margin",
        "target_value_mode": "min_target_delayed",
        "dense_bad_floor": 0.05,
        "adv_margin": 0.05,
        "cql_alpha": 0.02,
    },
    "codex_08_td_error_trust_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "exp_margin",
        "value_update_mode": "weighted_mse",
        "dense_bad_floor": 0.05,
        "td_error_trust_scale": 5.0,
        "adv_margin": 0.05,
    },
    "codex_09_weighted_actor_cdaf": {
        "support_source": "dataset_knn",
        "filter_mode": "local_rank_exp",
        "rank_quantile": 0.60,
        "rank_temperature": 0.30,
        "dense_bad_floor": 0.05,
        "actor_fit_method": "td3_weighted_bc",
        "policy_weight_exponent": 0.8,
        "policy_weight_clip": 10.0,
    },
    "codex_10_hybrid_support_cdaf": {
        "support_source": "hybrid_knn",
        "support_mix": "max",
        "filter_mode": "local_rank_exp",
        "rank_quantile": 0.60,
        "rank_temperature": 0.30,
        "dense_bad_floor": 0.05,
        "td_error_trust_scale": 7.5,
        "cql_alpha": 0.02,
    },
}


@dataclass
class TrainConfig(base.TrainConfig):
    variant: str = "codex_01_support_floor_cdaf"

    # Dataset/global state support. c(s)=0 means singleton/sparse, c(s)=1 means
    # locally dense enough that action filtering is trusted.
    support_source: str = "dataset_knn"  # dataset_knn, batch_knn, hybrid_knn, constant
    support_mix: str = "max"  # used only by hybrid_knn: max, min, mean, product
    coverage_knn_k: int = 10
    coverage_reference_size: int = 100_000
    coverage_low_quantile: float = 0.20
    coverage_high_quantile: float = 0.80

    # Action-quality filter.
    filter_mode: str = "exp_margin"  # exp_margin, sigmoid, local_rank_exp, hard
    value_update_mode: str = "weighted_mse"  # weighted_mse, expectile
    target_value_mode: str = "target_q"  # target_q, min_target_delayed, delayed_q
    dense_bad_floor: float = 0.05
    adv_margin: float = 0.0
    adaptive_margin_coef: float = 0.0
    rank_quantile: float = 0.60
    rank_temperature: float = 0.25
    sigmoid_temperature: float = 0.50

    # Reliability/conservatism knobs.
    cql_alpha: float = 0.0
    cql_temperature: float = 1.0
    cql_support_weight: float = 1.0
    td_error_trust_scale: float = 0.0

    # IQL-style value regression, with SARSA-like tau=0.5 for sparse states.
    expectile_sparse: float = 0.50
    expectile_dense: float = 0.80

    def __post_init__(self):
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-BIAS"
    config.group = f"{config.variant}-JAX"
    config.name = f"{config.group}-{config.env}"


def validate_config(config: TrainConfig) -> None:
    base.validate_config(config)
    assert config.variant in VARIANT_PRESETS, f"unknown variant: {config.variant}"
    assert config.support_source in ("dataset_knn", "batch_knn", "hybrid_knn", "constant")
    assert config.support_mix in ("max", "min", "mean", "product")
    assert config.filter_mode in ("exp_margin", "sigmoid", "local_rank_exp", "hard")
    assert config.value_update_mode in ("weighted_mse", "expectile")
    assert config.target_value_mode in ("target_q", "min_target_delayed", "delayed_q")
    assert config.coverage_knn_k > 0
    assert config.coverage_reference_size > 0
    assert 0.0 <= config.coverage_low_quantile <= 1.0
    assert 0.0 <= config.coverage_high_quantile <= 1.0
    assert config.coverage_low_quantile <= config.coverage_high_quantile
    assert 0.0 <= config.dense_bad_floor <= 1.0
    assert config.adv_margin >= 0.0
    assert config.adaptive_margin_coef >= 0.0
    assert 0.0 <= config.rank_quantile <= 1.0
    assert config.rank_temperature > 0.0
    assert config.sigmoid_temperature > 0.0
    assert config.cql_alpha >= 0.0
    assert config.cql_temperature > 0.0
    assert 0.0 <= config.cql_support_weight <= 1.0
    assert config.td_error_trust_scale >= 0.0
    assert 0.0 < config.expectile_sparse < 1.0
    assert 0.0 < config.expectile_dense < 1.0


def _coerce_hparam_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    if not config.use_hyperparams or config.hyperparams_path is None:
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    hparam_path = Path(config.hyperparams_path)
    if not hparam_path.exists():
        print(f"Hyperparameter file not found: {hparam_path}. Using dataclass/CLI values.")
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    with open(hparam_path, "r") as f:
        all_hyperparams = yaml.safe_load(f) or {}

    if config.env not in all_hyperparams:
        print(f"No hyperparameters found for env '{config.env}' in {hparam_path}. Using dataclass/CLI values.")
        refresh_algorithm_names(config)
        validate_config(config)
        return config

    env_hyperparams = all_hyperparams[config.env] or {}
    cli_overrides = base._cli_overridden_fields()
    aliases = {"n_timesteps": "max_timesteps"}
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
        print(f"Ignored unknown hyperparameter keys for Codex CDAF: {', '.join(skipped_unknown)}")
    return config


def apply_variant_preset(config: TrainConfig, default_variant: Optional[str] = None) -> TrainConfig:
    cli_overrides = base._cli_overridden_fields()
    if default_variant is not None and "variant" not in cli_overrides:
        config.variant = default_variant
    if config.variant not in VARIANT_PRESETS:
        raise ValueError(f"Unknown Codex CDAF variant: {config.variant}")

    for key, value in VARIANT_PRESETS[config.variant].items():
        if key not in cli_overrides:
            setattr(config, key, value)

    refresh_algorithm_names(config)
    validate_config(config)
    return config


class CodexCDAFJAX(base.SACDAFJAX):
    """CDAF variants for sparse-state protection and dense-state action filtering."""

    def __init__(
        self,
        support_source: str = "dataset_knn",
        support_mix: str = "max",
        filter_mode: str = "exp_margin",
        value_update_mode: str = "weighted_mse",
        target_value_mode: str = "target_q",
        dense_bad_floor: float = 0.05,
        adv_margin: float = 0.0,
        adaptive_margin_coef: float = 0.0,
        rank_quantile: float = 0.60,
        rank_temperature: float = 0.25,
        sigmoid_temperature: float = 0.50,
        cql_alpha: float = 0.0,
        cql_temperature: float = 1.0,
        cql_support_weight: float = 1.0,
        td_error_trust_scale: float = 0.0,
        expectile_sparse: float = 0.50,
        expectile_dense: float = 0.80,
        **kwargs,
    ):
        self.support_source = support_source
        self.support_mix = support_mix
        self.filter_mode = filter_mode
        self.value_update_mode = value_update_mode
        self.target_value_mode = target_value_mode
        self.dense_bad_floor = dense_bad_floor
        self.adv_margin = adv_margin
        self.adaptive_margin_coef = adaptive_margin_coef
        self.rank_quantile = rank_quantile
        self.rank_temperature = rank_temperature
        self.sigmoid_temperature = sigmoid_temperature
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature
        self.cql_support_weight = cql_support_weight
        self.td_error_trust_scale = td_error_trust_scale
        self.expectile_sparse = expectile_sparse
        self.expectile_dense = expectile_dense
        super().__init__(**kwargs)

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
        rho_k = self.rho_k
        rho_distance_scale = self.rho_distance_scale
        rho_power = self.rho_power
        rho_min = self.rho_min
        max_steps = self.max_steps
        max_action = self.max_action

        support_source = self.support_source
        support_mix = self.support_mix
        filter_mode = self.filter_mode
        value_update_mode = self.value_update_mode
        target_value_mode = self.target_value_mode
        dense_bad_floor = self.dense_bad_floor
        adv_margin = self.adv_margin
        adaptive_margin_coef = self.adaptive_margin_coef
        rank_quantile = self.rank_quantile
        rank_temperature = self.rank_temperature
        sigmoid_temperature = self.sigmoid_temperature
        cql_alpha = self.cql_alpha
        cql_temperature = self.cql_temperature
        cql_support_weight = self.cql_support_weight
        td_error_trust_scale = self.td_error_trust_scale
        expectile_sparse = self.expectile_sparse
        expectile_dense = self.expectile_dense

        def batch_knn_support(states: jnp.ndarray) -> jnp.ndarray:
            if support_source == "constant":
                return jnp.ones((states.shape[0],), dtype=states.dtype)

            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.sum(diff ** 2, axis=-1)
            dist2 = dist2 + jnp.eye(states.shape[0], dtype=states.dtype) * 1e9
            k = max(1, min(int(rho_k), int(states.shape[0]) - 1))
            kth_dist2 = jnp.sort(dist2, axis=-1)[:, k - 1]
            kth_dist = jnp.sqrt(jnp.maximum(kth_dist2, 1e-12))
            scale = (
                jnp.asarray(rho_distance_scale, dtype=states.dtype)
                if rho_distance_scale > 0.0
                else jax.lax.stop_gradient(jnp.median(kth_dist) + 1e-6)
            )
            rho = 1.0 / (1.0 + (kth_dist / jnp.maximum(scale, 1e-6)) ** rho_power)
            rho = jnp.maximum(rho, rho_min)
            return jnp.clip(rho, 0.0, 1.0)

        def local_sorted_values(states: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
            diff = states[:, None, :] - states[None, :, :]
            dist2 = jnp.sum(diff ** 2, axis=-1)
            dist2 = dist2 + jnp.eye(states.shape[0], dtype=states.dtype) * 1e9
            k = max(1, min(int(rho_k), int(states.shape[0]) - 1))
            nn_idx = jnp.argsort(dist2, axis=-1)[:, :k]
            neighbor_values = jnp.take(values, nn_idx, axis=0)
            return jnp.sort(neighbor_values, axis=-1)

        @jax.jit
        def train_step(state: base.SACDAFState, batch: TensorBatch):
            total_it = state.total_it + jnp.asarray(1, dtype=jnp.int32)
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = jnp.squeeze(batch["rewards"], axis=-1)
            next_observations = batch["next_observations"]
            dones = jnp.squeeze(batch["dones"], axis=-1)

            dataset_support = jnp.clip(
                jnp.squeeze(batch.get("state_coverage_conf", jnp.ones_like(batch["rewards"])), axis=-1),
                0.0,
                1.0,
            )
            batch_support = batch_knn_support(observations)
            if support_source == "dataset_knn":
                support = dataset_support
            elif support_source == "batch_knn":
                support = batch_support
            elif support_source == "hybrid_knn":
                if support_mix == "min":
                    support = jnp.minimum(dataset_support, batch_support)
                elif support_mix == "mean":
                    support = 0.5 * (dataset_support + batch_support)
                elif support_mix == "product":
                    support = dataset_support * batch_support
                else:
                    support = jnp.maximum(dataset_support, batch_support)
            else:
                support = jnp.ones_like(dataset_support)
            support = jax.lax.stop_gradient(jnp.clip(support, 0.0, 1.0))

            def q_loss_fn(q_params):
                next_v = v_apply({"params": state.v_target_params}, next_observations)
                target_q = rewards + (1.0 - dones) * discount * next_v
                q = q_apply({"params": q_params}, observations, actions)
                q_mse_loss = jnp.mean((q - jax.lax.stop_gradient(target_q)) ** 2)

                if cql_alpha > 0.0:
                    rolled_actions = jnp.roll(actions, shift=1, axis=0)
                    neg_actions = jnp.clip(-actions, -max_action, max_action)
                    q_roll = q_apply({"params": q_params}, observations, rolled_actions)
                    q_neg = q_apply({"params": q_params}, observations, neg_actions)
                    ood_q = jnp.stack([q_roll, q_neg], axis=0)
                    conservative_gap = (
                        cql_temperature * jax.nn.logsumexp(ood_q / cql_temperature, axis=0)
                        - q
                    )
                    conservative_gap = jnp.maximum(conservative_gap, 0.0)
                    cql_weight = (1.0 - cql_support_weight) + cql_support_weight * support
                    cql_penalty = jnp.mean(jax.lax.stop_gradient(cql_weight) * conservative_gap)
                    q_loss = q_mse_loss + cql_alpha * cql_penalty
                else:
                    cql_penalty = jnp.asarray(0.0, dtype=observations.dtype)
                    q_loss = q_mse_loss
                return q_loss, (q, target_q, q_mse_loss, cql_penalty)

            (q_loss, (q, target_q, q_mse_loss, cql_penalty)), q_grads = jax.value_and_grad(
                q_loss_fn,
                has_aux=True,
            )(state.q_params)
            q_updates, q_opt_state = q_tx.update(q_grads, state.q_opt_state, state.q_params)
            q_params = optax.apply_updates(state.q_params, q_updates)

            progress = jnp.minimum(total_it.astype(jnp.float32) / jnp.maximum(float(max_steps), 1.0), 1.0)
            exponent = min_weight_exponent + (max_weight_exponent - min_weight_exponent) * progress

            delayed_q = q_apply({"params": state.q_delayed_params}, observations, actions)
            delayed_v = v_apply({"params": state.v_delayed_params}, observations)
            raw_delayed_adv = delayed_q - delayed_v
            delayed_adv = jnp.clip(raw_delayed_adv, -weight_logit_clip, weight_logit_clip)

            sorted_local_adv = local_sorted_values(observations, delayed_adv)
            rank_idx = int(rank_quantile * max(1, sorted_local_adv.shape[1] - 1))
            local_threshold = sorted_local_adv[:, rank_idx]
            local_adv_std = jnp.std(sorted_local_adv, axis=-1)
            adaptive_margin = adv_margin + adaptive_margin_coef * local_adv_std

            if filter_mode == "sigmoid":
                beta_raw = 2.0 * jax.nn.sigmoid((delayed_adv + adaptive_margin) / sigmoid_temperature)
                beta_raw = jnp.minimum(beta_raw, 1.0)
            elif filter_mode == "local_rank_exp":
                rank_gap = delayed_adv - local_threshold
                beta_raw = jnp.where(
                    rank_gap < 0.0,
                    jnp.exp(jnp.clip(rank_gap / rank_temperature, -weight_logit_clip, 0.0)),
                    jnp.ones_like(delayed_adv),
                )
            elif filter_mode == "hard":
                beta_raw = (delayed_adv >= -adaptive_margin).astype(observations.dtype)
            else:
                shifted_negative_adv = delayed_adv + adaptive_margin
                beta_raw = jnp.where(
                    delayed_adv < -adaptive_margin,
                    jnp.exp(exponent * shifted_negative_adv),
                    jnp.ones_like(delayed_adv),
                )
            beta_adv = dense_bad_floor + (1.0 - dense_bad_floor) * beta_raw
            beta_adv = jnp.maximum(beta_adv, beta_min)

            td_abs_error = jnp.abs(q - target_q)
            if td_error_trust_scale > 0.0:
                filter_trust = jnp.exp(-td_abs_error / td_error_trust_scale)
            else:
                filter_trust = jnp.ones_like(td_abs_error)

            filter_strength = support * progress * jax.lax.stop_gradient(filter_trust)
            value_weight = 1.0 - filter_strength * (1.0 - beta_adv)
            value_weight = jax.lax.stop_gradient(jnp.clip(value_weight, beta_min, 1.0))

            def v_loss_fn(v_params):
                target_v_q = q_apply({"params": state.q_target_params}, observations, actions)
                if target_value_mode == "min_target_delayed":
                    delayed_target_q = q_apply({"params": state.q_delayed_params}, observations, actions)
                    target_v_q = jnp.minimum(target_v_q, delayed_target_q)
                elif target_value_mode == "delayed_q":
                    target_v_q = q_apply({"params": state.q_delayed_params}, observations, actions)

                v = v_apply({"params": v_params}, observations)
                target_v_sg = jax.lax.stop_gradient(target_v_q)

                if value_update_mode == "expectile":
                    expectile_tau = expectile_sparse + support * (expectile_dense - expectile_sparse)
                    expectile_tau = jax.lax.stop_gradient(expectile_tau)
                    expectile_residual = target_v_sg - v
                    expectile_weight = jnp.where(
                        expectile_residual > 0.0,
                        expectile_tau,
                        1.0 - expectile_tau,
                    )
                    value_loss = jnp.mean(value_weight * expectile_weight * expectile_residual ** 2)
                else:
                    value_residual = v - target_v_sg
                    value_loss = jnp.mean(value_weight * value_residual ** 2)
                return value_loss, (v, target_v_q)

            (value_loss, (v, target_v_q)), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(state.v_params)
            v_updates, v_opt_state = v_tx.update(v_grads, state.v_opt_state, state.v_params)
            v_params = optax.apply_updates(state.v_params, v_updates)

            q_target_params = base.soft_update(q_params, state.q_target_params, tau)
            v_target_params = base.soft_update(v_params, state.v_target_params, tau)

            should_update_delayed = (total_it % delayed_update_period) == 0
            q_delayed_params = jax.lax.cond(
                should_update_delayed,
                lambda _: q_target_params,
                lambda _: state.q_delayed_params,
                operand=None,
            )
            v_delayed_params = jax.lax.cond(
                should_update_delayed,
                lambda _: v_target_params,
                lambda _: state.v_delayed_params,
                operand=None,
            )

            new_state = base.SACDAFState(
                total_it=total_it,
                q_params=q_params,
                q_target_params=q_target_params,
                q_delayed_params=q_delayed_params,
                q_opt_state=q_opt_state,
                v_params=v_params,
                v_target_params=v_target_params,
                v_delayed_params=v_delayed_params,
                v_opt_state=v_opt_state,
            )

            log_dict = {
                "q_loss": q_loss,
                "q_mse_loss": q_mse_loss,
                "cql_penalty": cql_penalty,
                "q_mean": jnp.mean(q),
                "target_q_mean": jnp.mean(target_q),
                "td_abs_error_mean": jnp.mean(td_abs_error),
                "value_loss": value_loss,
                "v_mean": jnp.mean(v),
                "target_v_q_mean": jnp.mean(target_v_q),
                "support_mean": jnp.mean(support),
                "support_min": jnp.min(support),
                "support_max": jnp.max(support),
                "batch_support_mean": jnp.mean(batch_support),
                "dataset_support_mean": jnp.mean(dataset_support),
                "beta_adv_mean": jnp.mean(beta_adv),
                "beta_adv_min": jnp.min(beta_adv),
                "beta_adv_max": jnp.max(beta_adv),
                "filter_strength_mean": jnp.mean(filter_strength),
                "filter_trust_mean": jnp.mean(filter_trust),
                "value_weight_mean": jnp.mean(value_weight),
                "value_weight_min": jnp.min(value_weight),
                "value_weight_max": jnp.max(value_weight),
                "weight_exponent": exponent,
                "adaptive_margin_mean": jnp.mean(adaptive_margin),
                "local_threshold_mean": jnp.mean(local_threshold),
                "delayed_adv_mean": jnp.mean(delayed_adv),
                "delayed_adv_min": jnp.min(delayed_adv),
                "delayed_adv_max": jnp.max(delayed_adv),
                "raw_delayed_adv_mean": jnp.mean(raw_delayed_adv),
                "raw_delayed_adv_min": jnp.min(raw_delayed_adv),
                "raw_delayed_adv_max": jnp.max(raw_delayed_adv),
                "negative_adv_frac": jnp.mean((raw_delayed_adv < 0.0).astype(jnp.float32)),
                "filtered_soft_frac": jnp.mean((beta_adv < 0.5).astype(jnp.float32)),
            }
            return new_state, log_dict

        return train_step


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_impl(config: TrainConfig, default_variant: Optional[str] = None):
    config = apply_env_hyperparams(config)
    config = apply_variant_preset(config, default_variant=default_variant)
    refit_only = config.load_model != "" and int(config.max_timesteps) <= 0
    if not refit_only:
        config = finalize_checkpoint_path(config)

    jax_device = base.select_jax_device(config.device)
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)
    if config.normalize_reward:
        base.modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = base.compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = base.normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = base.normalize_states(dataset["next_observations"], state_mean, state_std)

    if config.support_source in ("dataset_knn", "hybrid_knn"):
        dataset["state_coverage_conf"] = coverage_base.compute_state_coverage_confidence(
            dataset["observations"],
            k=config.coverage_knn_k,
            reference_size=config.coverage_reference_size,
            low_quantile=config.coverage_low_quantile,
            high_quantile=config.coverage_high_quantile,
            seed=config.seed,
        )
    else:
        dataset["state_coverage_conf"] = np.ones(dataset["observations"].shape[0], dtype=np.float32)

    env = base.wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = coverage_base.ReplayBuffer(
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
    base.set_seed(seed, env)

    print("---------------------------------------")
    print(f"Training {config.variant}, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    trainer = CodexCDAFJAX(
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
        rho_mode=config.rho_mode,
        rho_k=config.rho_k,
        rho_distance_scale=config.rho_distance_scale,
        rho_power=config.rho_power,
        rho_min=config.rho_min,
        actor_fit_method=config.actor_fit_method,
        policy_weight_exponent=config.policy_weight_exponent,
        policy_weight_clip=config.policy_weight_clip,
        alpha=config.alpha,
        bc_coef=config.bc_coef,
        seed=seed,
        device=jax_device,
        support_source=config.support_source,
        support_mix=config.support_mix,
        filter_mode=config.filter_mode,
        value_update_mode=config.value_update_mode,
        target_value_mode=config.target_value_mode,
        dense_bad_floor=config.dense_bad_floor,
        adv_margin=config.adv_margin,
        adaptive_margin_coef=config.adaptive_margin_coef,
        rank_quantile=config.rank_quantile,
        rank_temperature=config.rank_temperature,
        sigmoid_temperature=config.sigmoid_temperature,
        cql_alpha=config.cql_alpha,
        cql_temperature=config.cql_temperature,
        cql_support_weight=config.cql_support_weight,
        td_error_trust_scale=config.td_error_trust_scale,
        expectile_sparse=config.expectile_sparse,
        expectile_dense=config.expectile_dense,
    )

    loaded_run_dir: Optional[Path] = None
    if config.load_model != "":
        loaded_run_dir, checkpoint_path = base.resolve_checkpoint_path(
            config.load_model,
            run_name=config.name,
            seed=config.seed,
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_pickle(checkpoint_path)
        trainer.load_state_dict(checkpoint)

    if config.log_wandb:
        base.wandb_init(asdict(config))

    if refit_only:
        if loaded_run_dir is None:
            raise ValueError("refit_only mode requires --load_model")

        actor_refit_dir = loaded_run_dir / config.actor_refit_dir_name
        actor_refit_dir.mkdir(parents=True, exist_ok=True)
        print("---------------------------------------")
        print(f"Actor refit from saved {config.variant} checkpoint")
        print(f"Saving actor refit outputs to: {actor_refit_dir}")
        print("---------------------------------------")

        fresh_actor_state = base.ActorState(
            params=copy.deepcopy(trainer.initial_actor_params),
            opt_state=copy.deepcopy(trainer.initial_actor_opt_state),
        )
        fresh_actor_state = base.tree_to_device(fresh_actor_state, jax_device)

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
        )

        save_pickle(actor_refit_dir / "final_actor.pkl", serialization.to_state_dict(refit_actor_state.params))
        base.save_logs_npz(
            [{"loaded_checkpoint": str(loaded_run_dir / "checkpoint.pkl"), **refit_log}],
            str(actor_refit_dir / "fit_eval_logs.npz"),
        )
        with open(actor_refit_dir / "refit_config.yaml", "w") as f:
            pyrallis.dump(config, f)

        if config.log_wandb and wandb.run is not None:
            wandb.save(str(actor_refit_dir / "final_actor.pkl"), policy="now")
            wandb.save(str(actor_refit_dir / "fit_eval_logs.npz"), policy="now")
            wandb.save(str(actor_refit_dir / "refit_config.yaml"), policy="now")
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
                    key: base.to_python_scalar(value)
                    for key, value in eval_log.items()
                    if base.is_scalar_value(value)
                }
                wandb.log(wandb_eval_log, step=int(jax.device_get(trainer.state.total_it)))

            base.save_and_upload_eval_logs(
                eval_logs=eval_logs,
                checkpoints_path=config.checkpoints_path,
                log_wandb=config.log_wandb,
            )

    if config.checkpoints_path is not None:
        save_pickle(os.path.join(config.checkpoints_path, "checkpoint.pkl"), trainer.state_dict())
        # if config.log_wandb and wandb.run is not None:
        #     wandb.save(os.path.join(config.checkpoints_path, "checkpoint.pkl"), policy="now")
        base.save_and_upload_eval_logs(
            eval_logs=eval_logs,
            checkpoints_path=config.checkpoints_path,
            log_wandb=config.log_wandb,
        )


def main(default_variant: str):
    @pyrallis.wrap()
    def _main(config: TrainConfig):
        train_impl(config, default_variant=default_variant)

    _main()
