"""DriftQL official algorithm, in ONE Python file with ORL-SMOOTH experiment plumbing.

All core DriftQL functions/classes, network definitions, Flax TrainState, and dataset/evaluator
implementations are in this file. They were copied from user-provided driftql-main.zip,
with source AST checks. The original imports from utils.*, agents.*, and envs.* have been
removed. There is NO runtime dependency on any local DriftQL original package.

The only project-local import is algorithms.uk_offline.common.checkpointing, the user's
existing shared ORL resumable-run manager; it does not contain algorithm code.

Upstream: driftql-main.zip (2026 DriftQL Authors), MIT. See LICENSE.

This file implements original OFFLINE DriftQL only; no online fine-tuning.
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

import collections
import copy
import functools
import pickle
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from types import SimpleNamespace

import distrax
import flax
import flax.linen as nn
from flax import serialization
from flax.core.frozen_dict import FrozenDict
import gymnasium
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import pyrallis
import yaml
from tqdm.auto import trange

try:
    import ogbench
except ImportError:
    ogbench = None
try:
    import wandb
except ImportError:
    wandb = None

_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / 'algorithms').is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.uk_offline.common.checkpointing import (
    DEFAULT_IDENTITY_IGNORED_FIELDS,
    TrainingCheckpointManager,
    best_eval_metric,
    find_eval_log,
    log_training_exceptions,
    upsert_eval_log,
)


# =============================================================================
# ORIGINAL DriftQL: utils/networks.py
# =============================================================================

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


# =============================================================================
# ORIGINAL DriftQL: utils/encoders.py
# =============================================================================

class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}


# =============================================================================
# ORIGINAL DriftQL: utils/flax_utils.py
# =============================================================================

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


# =============================================================================
# ORIGINAL DriftQL: agents/driftql.py
# =============================================================================

def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a, b, eps=1e-12):
    """Pairwise L2: a [N,F], b [M,F] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    dist2 = a2 + b2.T - 2.0 * (a @ b.T)
    return jnp.sqrt(jnp.maximum(dist2, 0.0) + eps)


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def _resolve_noise_dim(cfg, action_dim):
    noise_dim = int(_cfg_get(cfg, 'noise_dim', action_dim))
    if noise_dim <= 0:
        noise_dim = int(action_dim)
    return noise_dim


def _compute_drift(
        gen_a: jnp.ndarray,  # [Ngen, A] generated actions (= q samples)
        pos_a: jnp.ndarray,  # [Npos, A] dataset actions (= p samples)
        temp: float,
        kernel: str = 'laplace',  # "laplace" or "gaussian"
        dim_scale: bool = True,  # divide distances by sqrt(d_a)
        eps: float = 1e-12,
):
    """
    Compute the drift field for one state.

    Returns:
        V:          [Ngen, A] total drift = V_p - V_q
        V_attract:  [Ngen, A] attraction component V_p(x)
        V_repel:    [Ngen, A] repulsion component V_q(x)
    """
    Ngen, action_dim = gen_a.shape

    # Pairwise distances
    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)  # [Ngen, Npos]
    dist_neg = _pairwise_l2(gen_a, gen_a, eps=eps)  # [Ngen, Ngen]

    # Self-mask: each sample cannot repel itself
    idx = jnp.arange(Ngen)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    if dim_scale:
        dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
    else:
        dim_factor = 1.0

    # Kernel logits
    if kernel == 'gaussian':
        # Gaussian: k(x,y) = exp(-||x-y||^2 / 2tau^2)
        dist_pos_scaled = dist_pos / dim_factor
        dist_neg_scaled = dist_neg / dim_factor
        logit_pos = -(dist_pos_scaled ** 2) / (2.0 * temp * temp)
        logit_neg = -(dist_neg_scaled ** 2) / (2.0 * temp * temp)
    else:
        # Laplace: k(x,y) = exp(-||x-y|| / tau)
        logit_pos = -(dist_pos / dim_factor) / temp
        logit_neg = -(dist_neg / dim_factor) / temp

    # Separate softmax. V_p and V_q are independently normalized
    W_pos = jax.nn.softmax(logit_pos, axis=-1)  # [Ngen, Npos]
    W_neg = jax.nn.softmax(logit_neg, axis=-1)  # [Ngen, Ngen]

    disp_to_pos = pos_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Npos, A]
    disp_to_neg = gen_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Ngen, A]

    # V_{pi,k}(x) = sum_j w_j (y_j - x)
    V_attract = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1)  # [Ngen, A]
    V_repel = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1)  # [Ngen, A]

    V = V_attract - V_repel

    return V, V_attract, V_repel


def _compute_drift_batched(
        gen_a: jnp.ndarray,  # [B, Ngen, A]
        pos_a: jnp.ndarray,  # [B, Npos, A]
        temp: float,
        kernel: str = 'laplace',
        dim_scale: bool = True,
        drift_normalize: bool = False,
        eps: float = 1e-12,
):
    """
    Batched drift field computation.

    Returns:
        V:          [B, Ngen, A]
        V_attract:  [B, Ngen, A]
        V_repel:    [B, Ngen, A]
    """

    def per_item(gen_i, pos_i):
        V_i, Va_i, Vr_i = _compute_drift(
            gen_a=gen_i,
            pos_a=pos_i,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            eps=eps,
        )

        if drift_normalize:
            act_dim = gen_i.shape[-1]
            raw_sq = jnp.mean(jnp.sum(V_i * V_i, axis=-1)) / act_dim
            lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + eps))
            V_i = V_i / lam
            Va_i = Va_i / lam
            Vr_i = Vr_i / lam

        return V_i, Va_i, Vr_i

    return jax.vmap(per_item)(gen_a, pos_a)


class DriftQLAgent(flax.struct.PyTreeNode):
    """
    DriftQL with theoretically-grounded drift field.

    The drift field follows the drifting model framework exactly:
      Delta_{p,q}(x) = V_{p,k}(x) - V_{q,k}(x)

    where V_{pi,k}(x) is the kernel-weighted direction.
    With Gaussian kernel, this is exact score matching on smoothed distributions.
    With Laplace kernel, it approximates score matching with O(1/D) error.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def drifting_bc_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        # Read config
        kernel = str(_cfg_get(self.config, 'kernel', 'laplace'))
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        dim_scale = bool(_cfg_get(self.config, 'dim_scale', True))
        drift_normalize = bool(self.config['drift_normalize'])
        temp = float(self.config['drift_temp'])
        eta = float(self.config['drift_eta'])
        Ngen = int(self.config['drift_ngen'])

        # Optional subsample for drift computation
        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions = batch['actions']
            drift_bs = batch_size

        # Positive actions: dataset actions for each state [B, 1, A]
        pos_a = jnp.clip(pos_actions[:, None, :], -1.0, 1.0)

        # Generate Ngen samples per state from current policy
        noises = jax.random.normal(noise_rng, (drift_bs * Ngen, noise_dim))
        obs_rep = jnp.repeat(obs, repeats=Ngen, axis=0)
        bc_raw = self.network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        gen_a = jnp.clip(bc_raw.reshape(drift_bs, Ngen, action_dim), -1.0, 1.0)

        # Compute drift field: Delta_{p,q}(x) = V_p(x) - V_q(x)
        V, V_attract, V_repel = _compute_drift_batched(
            gen_a=gen_a,
            pos_a=pos_a,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            drift_normalize=drift_normalize,
            eps=float(self.config['drift_eps']),
        )

        target = jax.lax.stop_gradient(jnp.clip(gen_a + eta * V, -1.0, 1.0))
        bc_drift_loss = jnp.mean((gen_a - target) ** 2)
        attract_norm = jnp.mean(_l2_norm(V_attract, axis=-1))
        repel_norm = jnp.mean(_l2_norm(V_repel, axis=-1))
        drift_norm = jnp.mean(_l2_norm(V, axis=-1))

        info = {
            'bc_drift_loss': bc_drift_loss,
            'drift_norm': drift_norm,
            'drift_attract_norm': attract_norm,
            'drift_repel_norm': repel_norm,
        }

        return bc_drift_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        # Q-loss: maximize Q(s, f_θ(s,z))
        noise_dim = _resolve_noise_dim(self.config, action_dim)
        noises = jax.random.normal(noise_rng, (batch_size, noise_dim))
        actor_raw = self.network.select('actor_bc_drift')(batch['observations'], noises, params=grad_params)
        actor_raw = jnp.clip(actor_raw, -1.0, 1.0)
        qs = self.network.select('critic')(batch['observations'], actions=actor_raw)

        q_agg = _cfg_get(self.config, 'q_agg_actor', self.config['q_agg'])
        if q_agg == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam_q = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam_q * q_loss

        actor_loss = self.config['alpha'] * bc_drift_loss + q_loss

        info = {
            'actor_loss': actor_loss,
            'bc_drift_loss': bc_drift_loss * self.config['alpha'],
            'bc_drift_loss_raw': bc_drift_loss,
            'drift_norm': bc_info['drift_norm'],
            'drift_attract_norm': bc_info['drift_attract_norm'],
            'drift_repel_norm': bc_info['drift_repel_norm'],
            'q_loss': q_loss,
            'q': q.mean(),
        }

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                noise_dim,
            ),
        )
        raw_actions = self.network.select('actor_bc_drift')(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        actions = jnp.where(jnp.isnan(actions), 0.0, actions)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        noise_dim = _resolve_noise_dim(config, action_dim)
        ex_noises = jnp.zeros((*ex_actions.shape[:-1], noise_dim), dtype=ex_actions.dtype)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_drift'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_bc_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_noises)),
        )

        if encoders.get('actor_bc_drift') is not None:
            network_info['actor_bc_drift_encoder'] = (
                encoders.get('actor_bc_drift'),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['noise_dim'] = noise_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='driftql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            noise_dim=0,  # defaults to action_dim when <= 0
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,  # Polyak target update rate
            q_agg='min',  # "min" or "mean" for critic target
            q_agg_actor='mean',  # "min" or "mean" for actor Q-loss
            alpha=10.0,  # drift loss weight
            normalize_q_loss=False,
            # Drift field
            kernel='laplace',  # "laplace" or "gaussian"
            drift_temp=0.2,  # kernel temperature
            dim_scale=True,  # divide distances by sqrt(d_a) before applying temp
            drift_ngen=32,  # number of generated samples per state
            drift_eta=1.0,  # drift step size eta
            drift_normalize=False,  # keep raw magnitude unless explicitly enabled
            drift_batch_size=256,  # subsample size for drift computation
            drift_eps=1e-12,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config


# =============================================================================
# ORIGINAL DriftQL: utils/datasets.py
# =============================================================================

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


# =============================================================================
# ORIGINAL DriftQL: envs/d4rl_utils.py
# =============================================================================

def make_env(env_name):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env,
    env_name,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    import d4rl  # lazy import: OGBench-only runs need no D4RL installation
    dataset = d4rl.qlearning_dataset(env)

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )

# Inline namespace replacing original `from envs import d4rl_utils`.
d4rl_utils = SimpleNamespace(make_env=make_env, get_dataset=get_dataset)


# =============================================================================
# ORIGINAL DriftQL: envs/env_utils.py
# =============================================================================

class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def load_npz_dataset(dataset_path):
    """Load a saved offline dataset from an `.npz` bundle."""
    dataset_path = os.path.expanduser(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, offline_dataset_path=None):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        offline_dataset_path: Optional path to a custom offline dataset. For
            OGBench singletask environments, this overrides the default train
            dataset while keeping the same evaluation environment.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        if offline_dataset_path is None:
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        else:
            env = ogbench.make_env_and_datasets(env_name, env_only=True)
            train_dataset = load_npz_dataset(offline_dataset_path)
            val_dataset = None
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        if val_dataset is not None:
            val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        if offline_dataset_path is not None:
            raise ValueError('Custom offline datasets are currently only supported for OGBench singletask environments.')

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        if offline_dataset_path is not None:
            raise ValueError('Custom offline datasets are currently only supported for OGBench singletask environments.')
        import d4rl.hand_manipulation_suite  # noqa

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'hopper' in env_name or 'halfcheetah' in env_name or 'walker2d' in env_name:
        # D4RL MuJoCo.
        if offline_dataset_path is not None:
            raise ValueError('Custom offline datasets are currently only supported for OGBench singletask environments.')

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset


# =============================================================================
# ORIGINAL DriftQL: utils/evaluation.py
# =============================================================================

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


# =============================================================================
# ORL-SMOOTH: the ONLY newly authored section (CLI, YAML, W&B, checkpoints)
# =============================================================================

ALGORITHM_NAME = 'DriftQL'
ALGORITHM_FULL_NAME = 'Drift Q-Learning, official update, ORL-SMOOTH experiment interface'


@dataclass
class TrainConfig:
    # ORL experiment parameters.
    device: str = 'gpu'
    env: str = 'cube-double-play-singletask-v0'
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: int = 42
    max_timesteps: int = 1_000_000
    eval_freq: int = 100_000
    n_episodes: int = 50
    checkpoints_path: Optional[str] = None
    load_model: str = ''
    mode: str = 'train'
    hyperparams_path: Optional[str] = 'hyperparams/driftql_jax.yml'
    use_hyperparams: bool = True
    run_tag: str = ''

    # Original DriftQL dataset and schedule behavior.
    batch_size: int = 256
    buffer_size: int = 2_000_000
    offline_dataset_path: Optional[str] = None
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None
    action_clip_eps: Optional[float] = 1e-5
    normalize_states: bool = False
    normalize_reward: bool = False
    online_steps: int = 0

    # Original DriftQL optimizer and network (one shared Adam optimizer).
    lr: Optional[float] = None
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4  # Compatibility field; original uses one lr.
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None
    actor_layer_norm: Optional[bool] = None
    critic_ln: bool = True
    actor_ln: bool = False
    encoder: Optional[str] = None

    # Original DriftQL mathematical hyperparameters.
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = 'min'  # Bellman target aggregation.
    actor_critic_aggregation: str = 'mean'  # original q_agg_actor
    alpha: float = 10.0
    normalize_q_loss: bool = False
    kernel: str = 'laplace'
    drift_temp: float = 0.2
    dim_scale: bool = True
    drift_ngen: int = 32
    noise_dim: int = 0  # Original: <=0 resolves to action dimension.
    drift_eta: float = 1.0
    drift_normalize: bool = False
    drift_batch_size: int = 256
    drift_eps: float = 1e-12

    # ORL W&B / checkpoint interface.
    project: str = 'ORL-SMOOTH'
    group: str = 'DriftQL-JAX'
    name: str = 'DriftQL-JAX'
    log_wandb: bool = True
    log_every: int = 5000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    eval_at_first_step: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        refresh_algorithm_names(self)
        validate_config(self)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = 'ORL-SMOOTH'
    config.group = f'{ALGORITHM_NAME}-JAX'
    if (config.q_agg, config.actor_critic_aggregation) == ('min', 'min'):
        agg_suffix = ''
    elif (config.q_agg, config.actor_critic_aggregation) == ('mean', 'mean'):
        agg_suffix = '-QMEAN'
    else:
        agg_suffix = f'-T{config.q_agg.upper()}-A{config.actor_critic_aggregation.upper()}'
    if config.run_tag:
        if not all(c.isalnum() or c in ('-', '_') for c in config.run_tag):
            raise ValueError('run_tag only allows letters, digits, underscore, hyphen')
        agg_suffix += '-' + config.run_tag
    config.name = f'{ALGORITHM_NAME}-JAX-{config.env}{agg_suffix}'


def validate_config(config: TrainConfig) -> None:
    if config.mode != 'train':
        raise NotImplementedError("The original DriftQL actor is updated jointly. Only --mode train is supported.")
    if config.online_steps:
        raise NotImplementedError('This ORL adapter supports original offline DriftQL only (online_steps=0).')
    if config.normalize_states or config.normalize_reward:
        raise ValueError('Original DriftQL offline protocol does not apply state/reward normalization; use False.')
    if config.env.strip() == '':
        raise ValueError('env must be nonempty')
    if config.max_timesteps < 0 or min(config.eval_freq, config.n_episodes, config.log_every, config.checkpoint_freq, config.batch_size, config.buffer_size) <= 0:
        raise ValueError('Training/evaluation/checkpoint intervals and batch/buffer sizes must be positive.')
    if config.q_agg not in ('mean', 'min') or config.actor_critic_aggregation not in ('mean', 'min'):
        raise ValueError('q_agg and actor_critic_aggregation must be min or mean')
    if config.kernel not in ('laplace', 'gaussian'):
        raise ValueError('kernel must be laplace or gaussian')
    if not (0 <= config.discount <= 1 and 0 <= config.tau <= 1):
        raise ValueError('discount and tau must be in [0,1]')
    if config.alpha < 0 or config.drift_temp <= 0 or config.drift_ngen < 1 or config.drift_batch_size < 1 or config.drift_eps <= 0:
        raise ValueError('Invalid DriftQL coefficient / temperature / sample count')
    if config.drift_eta < 0:
        raise ValueError('drift_eta must be >=0')
    if config.p_aug is not None and not (0 <= config.p_aug <= 1):
        raise ValueError('p_aug must be in [0,1]')
    if config.frame_stack is not None and config.frame_stack < 1:
        raise ValueError('frame_stack must be positive')
    if config.lr is not None and config.lr <= 0:
        raise ValueError('lr must be >0')
    if config.actor_learning_rate <= 0 or config.critic_learning_rate <= 0:
        raise ValueError('learning rates must be >0')


def _cli_overridden_fields(args=None):
    args = sys.argv[1:] if args is None else args
    out = set()
    for token in args:
        if token.startswith('--'):
            out.add(token[2:].split('=', 1)[0].replace('-', '_'))
    return out


def _coerce(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    """Precedence: TrainConfig < environment YAML < explicitly supplied CLI."""
    overrides = _cli_overridden_fields()
    yaml_settings = {}
    if config.use_hyperparams and config.hyperparams_path is not None:
        yaml_path = Path(config.hyperparams_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f'Hyperparameter YAML missing: {yaml_path}')
        with yaml_path.open(encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        if config.env not in data:
            print(f"Warning: no official YAML hyperparameters for '{config.env}'; using TrainConfig/CLI defaults.")
        else:
            yaml_settings = data[config.env] or {}
            aliases = {'q_agg_actor': 'actor_critic_aggregation', 'offline_steps': 'max_timesteps',
                       'eval_interval': 'eval_freq', 'eval_episodes': 'n_episodes',
                       'log_interval': 'log_every', 'env_name': 'env'}
            known = TrainConfig.__dataclass_fields__
            for raw_key, value in yaml_settings.items():
                key = aliases.get(raw_key, raw_key)
                if key not in known:
                    print(f'Ignored unknown DriftQL YAML key: {raw_key}')
                    continue
                if key in overrides or raw_key in overrides:
                    continue
                # If CLI explicitly overrides the common rate, don't let YAML's
                # equivalent original rate shadow the CLI value.
                if key == 'lr' and 'actor_learning_rate' in overrides:
                    continue
                if key in ('actor_hidden_dims', 'value_hidden_dims') and 'hidden_dim' in overrides:
                    continue
                if key == 'actor_hidden_dims' and 'actor_n_hiddens' in overrides:
                    continue
                if key == 'value_hidden_dims' and 'critic_n_hiddens' in overrides:
                    continue
                if key == 'layer_norm' and 'critic_ln' in overrides:
                    continue
                if key == 'actor_layer_norm' and 'actor_ln' in overrides:
                    continue
                setattr(config, key, _coerce(value))
    # The original uses a *single* Adam optimizer, while other ORL modules
    # expose distinct actor/critic names. Resolve aliases without changing math.
    if 'lr' in overrides:
        config.actor_learning_rate = float(config.lr)
    elif 'actor_learning_rate' in overrides:
        config.lr = float(config.actor_learning_rate)
    elif yaml_settings.get('lr') is not None and yaml_settings.get('actor_learning_rate') is None:
        config.actor_learning_rate = float(config.lr)
    else:
        config.lr = float(config.actor_learning_rate)
    if config.actor_hidden_dims is None:
        config.actor_hidden_dims = tuple([config.hidden_dim] * config.actor_n_hiddens)
    else:
        config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    if config.value_hidden_dims is None:
        config.value_hidden_dims = tuple([config.hidden_dim] * config.critic_n_hiddens)
    else:
        config.value_hidden_dims = tuple(config.value_hidden_dims)
    if config.layer_norm is None or 'critic_ln' in overrides:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None or 'actor_ln' in overrides:
        config.actor_layer_norm = bool(config.actor_ln)
    if config.critic_learning_rate != config.lr:
        print(f'Warning: original DriftQL has one shared Adam optimizer; critic_learning_rate={config.critic_learning_rate} is not used; using lr={config.lr}.')
    refresh_algorithm_names(config)
    validate_config(config)
    return config


def config_to_agent_config(config: TrainConfig) -> Dict[str, Any]:
    """Map ORL names into exact upstream get_config keys; no algorithm changes."""
    return dict(
        agent_name='driftql',
        lr=float(config.lr),
        batch_size=int(config.batch_size),
        actor_hidden_dims=tuple(config.actor_hidden_dims),
        value_hidden_dims=tuple(config.value_hidden_dims),
        noise_dim=int(config.noise_dim),
        layer_norm=bool(config.layer_norm),
        actor_layer_norm=bool(config.actor_layer_norm),
        discount=float(config.discount), tau=float(config.tau),
        q_agg=config.q_agg, q_agg_actor=config.actor_critic_aggregation,
        alpha=float(config.alpha), normalize_q_loss=bool(config.normalize_q_loss),
        kernel=config.kernel, drift_temp=float(config.drift_temp),
        dim_scale=bool(config.dim_scale), drift_ngen=int(config.drift_ngen),
        drift_eta=float(config.drift_eta), drift_normalize=bool(config.drift_normalize),
        drift_batch_size=int(config.drift_batch_size), drift_eps=float(config.drift_eps),
        encoder=config.encoder,
    )


def finalize_checkpoint_path(config):
    if config.checkpoints_path is not None:
        config.checkpoints_path = str(Path(config.checkpoints_path) / config.name / str(config.seed))
    return config


def select_jax_device(device):
    backend = 'gpu' if device.lower() == 'cuda' else device.lower()
    try:
        dev = jax.devices(backend)[0]
    except Exception:
        print(f"Requested JAX backend {backend!r} unavailable; falling back to default.")
        dev = jax.devices()[0]
    print(f'Using JAX device: {dev}')
    return dev


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, 'seed'):
                env.seed(seed)
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed)


def save_pickle(path, payload):
    with Path(path).open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def resolve_checkpoint_path(load_model, run_name=None, seed=None):
    root = Path(load_model)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f'--load_model path does not exist: {root}')
    names = ('checkpoint.pkl', 'best_checkpoint.pkl')
    for name in names:
        f = root / name
        if f.is_file():
            return f
    if run_name is not None and seed is not None:
        for name in names:
            f = root / run_name / str(seed) / name
            if f.is_file():
                return f
    candidates = sorted({p.resolve() for pat in ('*/checkpoint.pkl','*/best_checkpoint.pkl','*/*/checkpoint.pkl','*/*/best_checkpoint.pkl') for p in root.glob(pat)})
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one checkpoint under {root}, found {len(candidates)}. Pass an exact --load_model file.')
    return candidates[0]


def _finite_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.mean(values)), float(np.std(values))) if values.size else (float('nan'), float('nan'))


def _save_npz(logs: List[Dict[str, Any]], path):
    if not logs or path is None:
        return
    keys = list(dict.fromkeys(key for log in logs for key in log))
    arrays = {key: np.asarray([log.get(key, np.nan) for log in logs]) for key in keys}
    np.savez(path, **arrays)


def evaluate_agent(agent, eval_env, n_episodes, eval_seed, timestep):
    """Call the upstream evaluator unchanged; only aggregate its episode records.

    Evaluation RNG is isolated from the training RNG (a runner-level change).
    No action ranking, greedy selection, or sampling alteration is introduced.
    """
    numpy_state = np.random.get_state()
    try:
        np.random.seed((int(eval_seed) + int(timestep)) % (2 ** 32 - 1))
        metrics, trajs, _ = evaluate(
            agent=agent, env=eval_env, config=agent.config,
            num_eval_episodes=int(n_episodes), num_video_episodes=0,
            eval_temperature=0,
        )
    finally:
        np.random.set_state(numpy_state)
    returns = [sum(float(r) for r in t['reward']) for t in trajs]
    success = []
    normalized = []
    for traj in trajs:
        info = traj['info'][-1]
        ep_info = info.get('episode', {})
        success.append(info.get('success', ep_info.get('success', np.nan)))
        normalized.append(ep_info.get('normalized_return', np.nan))
    reward_mean, reward_std = _finite_mean_std(returns)
    score_mean, score_std = _finite_mean_std(normalized)
    success_mean, success_std = _finite_mean_std(success)
    if not np.isfinite(success_mean) and 'success' in metrics:
        success_mean = float(metrics['success'])
    if not np.isfinite(score_mean) and 'episode.normalized_return' in metrics:
        score_mean = float(metrics['episode.normalized_return'])
    return dict(timestep=int(timestep), **{
        'eval/reward_mean': reward_mean,
        'eval/reward_std': reward_std,
        'eval/d4rl_normalized_score_mean': score_mean,
        'eval/d4rl_normalized_score_std': score_std,
        'eval/success_rate': success_mean,
        'eval/success_std': success_std,
    })


def _train_impl(config: TrainConfig):
    config = apply_env_hyperparams(config)
    config = finalize_checkpoint_path(config)
    checkpoint_manager = None
    preparation = None
    if config.checkpoints_path is not None:
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=config.checkpoints_path, current_config=asdict(config),
            default_config=asdict(TrainConfig()), max_timesteps=int(config.max_timesteps),
            checkpoint_type='driftql_jax_training_progress',
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2, accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb), wandb_entity=config.wandb_entity,
            wandb_project=config.project, final_checkpoint_name='checkpoint.pkl',
        )
        preparation = checkpoint_manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return

    device = select_jax_device(config.device)
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config.env, frame_stack=config.frame_stack,
        action_clip_eps=config.action_clip_eps,
        offline_dataset_path=config.offline_dataset_path,
    )
    set_seed(config.seed, eval_env)

    # Exactly the original main.py offline replay-buffer construction.
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(config.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset
    for dataset in (train_dataset, val_dataset, replay_buffer):
        if dataset is not None:
            dataset.p_aug = config.p_aug
            dataset.frame_stack = config.frame_stack
            dataset.return_next_actions = False
    example_batch = train_dataset.sample(1)
    agent = DriftQLAgent.create(
        config.seed, example_batch['observations'], example_batch['actions'],
        config_to_agent_config(config),
    )
    agent = jax.device_put(agent, device)
    train_timestep = 0
    train_logs: List[Dict[str, Any]] = []

    def current_state():
        return {'agent': serialization.to_state_dict(agent),
                'config': asdict(config), '_training_timestep': int(train_timestep),
                'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate(),
                'train_logs': train_logs}

    def load_state(payload):
        nonlocal agent, train_timestep, train_logs
        if not isinstance(payload, dict) or 'agent' not in payload:
            raise ValueError('Checkpoint must contain the official DriftQL agent state.')
        agent = serialization.from_state_dict(agent, payload['agent'])
        agent = jax.device_put(agent, device)
        train_timestep = int(payload.get('_training_timestep', 0))
        train_logs = list(payload.get('train_logs', []))
        if 'numpy_rng_state' in payload:
            np.random.set_state(payload['numpy_rng_state'])
        if 'python_rng_state' in payload:
            random.setstate(payload['python_rng_state'])

    if config.load_model:
        ckpt_file = resolve_checkpoint_path(config.load_model, config.name, config.seed)
        print(f'Loading model state from: {ckpt_file}')
        load_state(load_pickle(ckpt_file))

    eval_logs: List[Dict[str, Any]] = []
    if checkpoint_manager is not None:
        if preparation.is_resuming:
            start, eval_logs, _ = checkpoint_manager.restore(
                load_trainer_state=load_state, get_restored_timestep=lambda: train_timestep,
            )
            print(f'Restored training at step {start}.')
        else:
            checkpoint_manager.save_progress(
                timestep=train_timestep, trainer_state=current_state(),
                eval_logs=eval_logs, status='running',
            )

    if config.log_wandb:
        if wandb is None:
            print('Warning: wandb not installed; continuing with local logs only.')
            config.log_wandb = False
        elif checkpoint_manager is not None:
            if not checkpoint_manager.initialize_wandb_with_fallback(
                    wandb_module=wandb, config=asdict(config), code_root=_PROJECT_ROOT):
                config.log_wandb = False
        else:
            try:
                run = wandb.init(project=config.project, group=config.group,
                                 name=config.name, entity=config.wandb_entity,
                                 config=asdict(config), id=str(uuid.uuid4()))
                if run is not None and hasattr(run, 'log_code'):
                    run.log_code(str(_PROJECT_ROOT))
            except Exception as exc:
                print(f'Warning: W&B unavailable: {exc}; continuing locally.')
                config.log_wandb = False

    def log_wandb(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))

    def eval_due(step):
        return ((config.eval_at_first_step and step == 1) or
                step % config.eval_freq == 0 or step == config.max_timesteps)

    best_score = best_eval_metric(eval_logs)
    first_time = last_time = time.time()
    try:
        for i in trange(train_timestep + 1, int(config.max_timesteps) + 1, desc='DriftQL Training'):
            batch = train_dataset.sample(config.batch_size)
            # ORIGINAL, UNMODIFIED mathematical gradient / optimizer update:
            agent, update_info = agent.update(batch)
            train_timestep = int(i)

            if i % config.log_every == 0:
                metrics = {f'training/{k}': float(jax.device_get(v))
                           for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config.batch_size)
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    metrics.update({f'validation/{k}': float(jax.device_get(v))
                                    for k, v in val_info.items()})
                metrics['time/epoch_time'] = (time.time() - last_time) / config.log_every
                metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                log_wandb(metrics, i)
                train_logs.append({'timestep': i, **metrics})
                if config.checkpoints_path is not None:
                    _save_npz(train_logs, Path(config.checkpoints_path) / 'train_logs.npz')

            if eval_due(i):
                result = evaluate_agent(agent, eval_env, config.n_episodes, config.eval_seed, i)
                upsert_eval_log(eval_logs, result)
                print(f"Evaluation at step {i}: reward={result['eval/reward_mean']:.3f} "
                      f"± {result['eval/reward_std']:.3f}; "
                      f"success={result['eval/success_rate']:.3f} "
                      f"± {result['eval/success_std']:.3f}")
                log_wandb(result, i)
                if config.checkpoints_path is not None:
                    _save_npz(eval_logs, Path(config.checkpoints_path) / 'eval_logs.npz')
                if config.checkpoints_path and config.save_best_model:
                    score = result['eval/success_rate']
                    if not np.isfinite(score): score = result['eval/d4rl_normalized_score_mean']
                    if not np.isfinite(score): score = result['eval/reward_mean']
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        save_pickle(Path(config.checkpoints_path) / 'best_checkpoint.pkl', current_state())

            if checkpoint_manager is not None and i % config.checkpoint_freq == 0:
                checkpoint_manager.save_progress(
                    timestep=i, trainer_state=current_state(), eval_logs=eval_logs,
                    status='running',
                )
    except BaseException:
        if checkpoint_manager is not None:
            interrupted_step = int(train_timestep)
            eval_committed = (not eval_due(interrupted_step) or
                              find_eval_log(eval_logs, interrupted_step) is not None)
            if eval_committed:
                checkpoint_manager.save_progress(
                    timestep=interrupted_step, trainer_state=current_state(),
                    eval_logs=eval_logs, status='interrupted',
                )
                print(f'Saved interrupted checkpoint at step {interrupted_step}.')
            else:
                print('Evaluation interrupted; retaining earlier safe progress checkpoint.')
        raise

    if checkpoint_manager is not None:
        path = checkpoint_manager.complete(
            timestep=int(train_timestep), final_state=current_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs,
        )
        if path is not None:
            print(f'Saved final model: {path}')


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
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == '__main__':
    train()


# =============================================================================
# ORIGINAL LICENSE NOTICE
# The MIT License (MIT)
# 
# Copyright (c) 2026 DriftQL Authors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
