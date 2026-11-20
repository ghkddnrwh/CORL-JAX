"""Guided Flow Policy (GFP), complete algorithm in ONE Python file.

Source: guided-flow-policy-main.zip (2025-2026 GFP Authors / FQL Authors, MIT).
The GFPAgent, network modules, neural-network TrainState, dataset, and evaluation
structures are copied from that ORIGINAL repository. No imports from any local
GFP source package are required. See VERIFY_GFP_SINGLE_FILE.py for AST tests.

Only existing project-local dependency: algorithms.uk_offline.common.checkpointing
(the user's generic ORL-SMOOTH experiment/resume manager, NOT GFP algorithm).
The offline learning algorithm, including its fused GPU update and VaBC/guidance
loss, is UNCHANGED; the newly written section is CLI, YAML, logging and resume.

This file covers OFFLINE GFP, not the upstream optional offline-to-online stage.
"""
from __future__ import annotations

import os
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

import collections
import copy
import functools
import glob
import math
import pickle
import random
import re
import sys
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import distrax
import flax
import flax.linen as nn
from flax import serialization
from flax.core.frozen_dict import FrozenDict
import gymnasium
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import optax
import pyrallis
import yaml
from tqdm.auto import trange

BENCHMARK_OGBENCH = 'OGBench'
BENCHMARK_MINARI = 'Minari'
BENCHMARK_D4RL = 'D4RL'
# Only import optional environment packages when their benchmark is used.
d4rl = None

try:
    import wandb
except ImportError:
    class _UnavailableWandb:
        run = None
        def init(self, *args, **kwargs):
            raise ImportError('wandb unavailable: use --log_wandb False or install wandb')
        def log(self, *args, **kwargs):
            return None
        def finish(self, *args, **kwargs):
            return None
    wandb = _UnavailableWandb()

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / 'algorithms').is_dir()
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
# ORIGINAL GFP: utils/networks.py
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    dim: int = 126

    @nn.compact
    def __call__(self, t: jax.Array):
        """
        Args:
            t : Array of shape (batch,1) or (1,)
        Returns:
            emb: Array of shape (batch,dim) or (dim,)
        """
        assert t.shape[-1] == 1
        half_dim = self.dim // 2

        freqs = jnp.exp(
            jnp.arange(half_dim) * -math.log(10000.0) / (half_dim - 1)
        )  

        args = t * freqs
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (batch, dim)

        # If dim is odd, pad with a zero
        if self.dim % 2 == 1:
            zeros = jnp.zeros((*emb.shape[:-1],1))
            emb = jnp.concatenate([emb, zeros], axis=-1)

        return emb


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
            x = nn.Dense(size, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
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
    need_time: bool = True
    encode_time_dim: int = 64
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.need_time and self.encode_time_dim is not None and self.encode_time_dim != 0:
            self.time_encoder = SinusoidalPosEmb(dim=self.encode_time_dim)

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
        if not self.need_time:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.encode_time_dim is not None and self.encode_time_dim != 0:
                times = self.time_encoder(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


# =============================================================================
# ORIGINAL GFP: utils/encoders.py
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
# ORIGINAL GFP: utils/flax_utils.py
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
# ORIGINAL GFP: utils/datasets.py
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
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)
        # return np.random.randint(self.size, size=num_idxs)

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


def GPU_dataset_sample(rng, dataset, batch_size, dataset_size):
    rng, sample_rng = jax.random.split(rng)
    batch_idxs = jax.random.randint(
        sample_rng, (batch_size,), 0, dataset_size
    )
    batch = jax.tree_util.tree_map(lambda x: x[batch_idxs], dataset)
    return batch, rng


# =============================================================================
# ORIGINAL GFP: envs/minari_utils.py
# =============================================================================

def convert_episodes_to_transitions(episodes):
    # Recover OGBench-like dataset
    transitions = dict(
        (key,[]) for key in [
            "actions",
            "observations",
            "next_observations",
            "rewards",
            "masks",
            "terminals",
        ])

    for episode in episodes:
        for t in range(len(episode.rewards)):
            transitions["actions"].append(episode.actions[t])
            transitions["observations"].append(episode.observations[t])
            transitions["next_observations"].append(episode.observations[t+1])
            transitions["rewards"].append(episode.rewards[t])
            transitions["masks"].append(1.0 - episode.terminations[t])
            transitions["terminals"].append(0.0)

    return dict((k,np.array(v)) for (k,v) in transitions.items())


MINARI_DICT_REF_SCORES = dict(
    hopper = (-20.272305,3234.3),
    halfcheetah = (-280.178953,12135.0),
    walker2d = (1.629008,4592.3),
    pen = (137.92955108500217,8820.50845967583),
    door = (-45.80706024169922,2940.578369140625),
    hammer = (-267.074462890625,12635.712890625),
    relocate = (9.189092636108398,4287.70458984375),
)


MINARI_REF_SCORES: tuple[float,float]


def check_ref_scores(dataset:MinariDataset):
    global MINARI_REF_SCORES
    metadata = dataset.storage.metadata
    if 'ref_min_score' in metadata:
        metadata_scores = (metadata["ref_min_score"],metadata["ref_max_score"])
    else:
        metadata_scores = None
    for key,saved_scores in MINARI_DICT_REF_SCORES.items():
        if key in dataset.id:
            MINARI_REF_SCORES = saved_scores
            if (metadata_scores is not None 
            and np.linalg.norm(np.array(metadata_scores)-np.array(saved_scores)) > 0.1
            ):
                warnings.warn(
                    "=== WARNING ===\nMinari reference scores found in dataset's metadata "\
                    "do not match manually specified ones.\nWe ignore metadata's reference "\
                    "scores as in July 2025 some were still incorrect "\
                    "https://github.com/Farama-Foundation/Minari/issues/303 or missing in "\
                    "January 2026 https://github.com/Farama-Foundation/Minari/issues/323"
                )
            return
        if metadata_scores is not None:
            MINARI_REF_SCORES = metadata_scores
        else:
            raise Exception(
                f"Reference scores for the Minari dataset {dataset.id} were neither "\
                f"found in the dataset metadata nor in our dict of reference scores.")


def minari_normalized_score(acc_reward):
    min_score, max_score = MINARI_REF_SCORES
    return 100 * (acc_reward - min_score) / (max_score - min_score)

# Original local module import replaced by this in-file namespace.
minari_utils = SimpleNamespace(
    convert_episodes_to_transitions=convert_episodes_to_transitions,
    check_ref_scores=check_ref_scores,
    minari_normalized_score=minari_normalized_score,
)


# =============================================================================
# ORIGINAL GFP: envs/d4rl_utils.py
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

# Original local module import replaced by this in-file namespace.
d4rl_utils = SimpleNamespace(make_env=make_env, get_dataset=get_dataset)


# =============================================================================
# ORIGINAL GFP: utils/evaluation.py
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


def extract_success_from_eval_info(benchmark,eval_info:dict):
    if benchmark == BENCHMARK_OGBENCH:
        return 100*eval_info["success"]
    elif benchmark == BENCHMARK_D4RL:
        return eval_info["final_info.episode.normalized_return"]
    elif benchmark == BENCHMARK_MINARI:
        # minari_normalized_score is defined in this single file.
        return minari_normalized_score(eval_info["acc_reward"])
    else:
        raise Exception("Unknown benchmark")


def evaluate(
    agent_fn,
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
    actor_fn = supply_rng(agent_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    # may either be agent.sample_actions or agent.sample_flow_actions
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        acc_reward = 0.
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            acc_reward = acc_reward + reward
            info["acc_reward"] = acc_reward
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


def get_kth_element(x, k):
    if isinstance(x, dict):
        return {
            key: get_kth_element(x[key], k) for key in filter_keys(x.keys())
            }
    else:
        return x[k]


def filter_keys(key_list):
    # Remove the <_name> keys added by gymnasium
    keys = []
    for key in key_list:
        if key[0] == '_' and key[1:] in key_list:
            pass
        else:
            keys.append(key)
    return keys


def evaluate_parallel(
    agent_fn,
    envs,
    num_eval_episodes=50,
    num_video_episodes=0,
    eval_temperature=0,
    n_eval_envs=10,
    video_env=None,
    **kwargs # past to the evaluate function for videos
):
    """Parallel version of FQL's evaluate function"""
    actor_fn_rng = supply_rng(agent_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    # may either be agent.sample_actions or agent.sample_flow_actions
    stats = defaultdict(list)
    
    iterations = num_eval_episodes // n_eval_envs
    finished = False
    counts = np.zeros(n_eval_envs, dtype=int)
    inactive = np.zeros(n_eval_envs, dtype=int)
    
    initial_seed = np.random.randint(1e6)
    rng = jax.random.PRNGKey(initial_seed)
    rng, subkey = jax.random.split(rng)
    observation, _ = envs.reset(seed=subkey[0].item())

    trajs = [defaultdict(list) for _ in range(num_eval_episodes)]

    while not finished:
        action = actor_fn_rng(observations=observation, temperature=eval_temperature)
        action = np.array(action)
        action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = envs.step(action)
        done = np.logical_or(terminated, truncated)

        for k in range(n_eval_envs):
            if not inactive[k]:
                transition = dict(
                    observation=get_kth_element(observation, k),
                    next_observation=get_kth_element(next_observation, k),
                    action=get_kth_element(action, k),
                    reward=get_kth_element(reward, k),
                    done=get_kth_element(done, k),
                    info=get_kth_element(info, k),
                )
                index = counts[k] + k * iterations
                add_to(trajs[index], transition)
        
        observation = next_observation

        if done.max():
            for k in range(n_eval_envs):
                if done[k]:
                    add_to(stats, flatten(get_kth_element(info, k)))

            rng, subkey = jax.random.split(rng)
            observation, _ = envs.reset(
                options={"reset_mask": done.flatten()}, 
                seed=subkey[0].item()
            )
            
            counts += done
            inactive = counts >= iterations
            finished = inactive.min()
    for k, v in stats.items():
        stats[k] = np.mean(v)

    if num_video_episodes > 0:
        _, _, renders = evaluate(
            agent_fn=agent_fn, 
            env=video_env,
            num_eval_episodes=0,
            num_video_episodes=num_video_episodes,
            **kwargs)
    else:
        renders = []

    # For Minari
    acc_rewards = []
    for traj in trajs:
        rewards = traj["reward"]
        acc_rewards.append(np.sum(rewards))
    stats["acc_reward"] = np.mean(acc_rewards)

    return stats, trajs, renders


# =============================================================================
# ORIGINAL GFP: envs/env_utils.py
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


def make_env_and_datasets(
        benchmark, 
        env_name, 
        frame_stack=None, 
        action_clip_eps=1e-5
    ):
    """Make offline RL environment and datasets.

    Args:
        benchmark: OGBench, Minari or D4RL
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        - the Gym environment
        - a function to spawn evaluation environment
        - training dataset
        - validation dataset for OGBench, None otherwise
    """

    if benchmark == BENCHMARK_OGBENCH:
        # OGBench.
        import ogbench
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)

        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        if frame_stack is not None:
            env = FrameStackWrapper(env, frame_stack)
        def eval_env_creator():
            eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
            eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
            if frame_stack is not None:
                eval_env = FrameStackWrapper(eval_env, frame_stack)
            eval_env.reset()
            return eval_env
        

    elif benchmark == BENCHMARK_MINARI:
        # Minari dataset
        import minari
        # minari_utils is an in-file namespace, bound above.
        minari_dataset = minari.load_dataset(env_name, download=True)
        minari_utils.check_ref_scores(minari_dataset)
        train_dataset = Dataset.create(
            **minari_utils.convert_episodes_to_transitions(
                minari_dataset.iterate_episodes()
        ))
        val_dataset = None
        env = minari_dataset.recover_environment(eval_env=False)
        def eval_env_creator():
            eval_env = minari_dataset.recover_environment(eval_env=True)
            eval_env.reset()
            return eval_env
         

    elif benchmark == BENCHMARK_D4RL:
        # D4RL
        if any(x in env_name for x in ["pen","hammer","relocate","door"]):
            import d4rl.hand_manipulation_suite  # noqa
            # TODO why ??
        # d4rl_utils is an in-file namespace, bound above.
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
        env = d4rl_utils.make_env(env_name)
        def eval_env_creator():
            eval_env = d4rl_utils.make_env(env_name)
            eval_env.reset()
            return eval_env
    else:
        raise ValueError(f'Unsupported benchmark: {benchmark}')


    env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:

        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env_creator, train_dataset, val_dataset

# Preserve access to the original benchmark loader before the ORL wrapper.
original_make_env_and_datasets = make_env_and_datasets


# =============================================================================
# ORIGINAL GFP: agents/gfp.py
# =============================================================================

class GFPAgent(flax.struct.PyTreeNode):
    """Guided Flow Policy (GFP) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_eval(self, obs, actions):
        """Compute Q(s,a) with the target network and q aggregation function."""
        q = self.network.select("target_critic")(obs, actions)
        return getattr(jnp, self.config["q_agg"])(q, axis=0)

    def value_with_all_options(self, agg_rule, obs, actor_acts, vabc_acts):
        """
        Evaluate the value of a state V(s), with a switch-case depending
        on whether we use the Actor to compute the reference action
        or the VaBC policy, or both (with an aggregate function).

        Takes as input lambda functions for lazy computations.
        """
        assert agg_rule in ["actor", "vabc", "min", "mean", "max"]
        if agg_rule != "vabc":
            q_actor = self.critic_eval(obs, actor_acts)
        if agg_rule != "actor":
            q_vabc = self.critic_eval(obs, vabc_acts)
        if agg_rule == "actor":
            return q_actor
        elif agg_rule == "vabc":
            return q_vabc
        elif agg_rule == "max":
            return jnp.maximum(q_actor, q_vabc)
        elif agg_rule == "min":
            return jnp.minimum(q_actor, q_vabc)
        else:
            return (q_actor + q_vabc) / 2

    def critic_loss(self, batch, grad_params, rng):
        """Compute GPS's critic loss"""
        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"]
        if self.config["target_use_next"]:
            vabc_obs = next_obs
        else:
            vabc_obs = batch["observations"]
        tgt_agg = self.config["target_agg"]

        # Compute the Bellman target with different rules
        # Lazy computations
        if tgt_agg != "actor":
            vabc_acts = self.sample_flow_actions(vabc_obs, sample_rng)
        else:
            vabc_acts = None
        if tgt_agg != "vabc":
            actor_acts = self.sample_actions(next_obs, sample_rng)
        else:
            actor_acts = None
        next_q = self.value_with_all_options(tgt_agg, next_obs, actor_acts, vabc_acts)
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=grad_params
        )
        critic_loss = jnp.square(q - target_q).mean()

        # Metrics:
        if self.config["log_metrics"]:
            info = {
                "critic_loss": critic_loss,
                "q_mean": q.mean(),
                "tgt_q_mean": next_q.mean(),
                "q_max": q.max(),
                "q_min": q.min(),
            }
        else:
            info = dict()
        return critic_loss, info

    def compute_guidance(self, batch, actor_acts, vabc_acts, lam):
        """Compute GFP-VaBC guidance weights"""
        obs = batch["observations"]

        # Compute the reference value with different rules
        q_ref = self.value_with_all_options(
            self.config["guidance_agg"], obs, actor_acts, vabc_acts
        )
        q_data = self.critic_eval(obs, batch["actions"])
        scale = lam / self.config["eta_temperature"]

        # Compute the guidance, either using a softmax or as an advantage
        assert self.config["guidance_fn"] in ["softmax", "advantage"]
        scl_q_data = scale * q_data
        scl_q_ref = scale * q_ref
        if self.config["guidance_fn"] == "softmax":
            log_denominator = logsumexp(
                jnp.stack([scl_q_data, scl_q_ref], axis=-1), axis=-1
            )
            guidance = jnp.exp(scl_q_data - log_denominator)
        else:
            guidance = jnp.exp(scl_q_data - scl_q_ref)

        # Metrics:
        if self.config["log_metrics"]:
            info = {
                "q_data": q_data.mean(),
                "q_ref": q_ref.mean(),
                "scaled_q_data": scl_q_data.mean(),
                "scaled_q_ref": scl_q_ref.mean(),
                "guidance": guidance.mean(),
                "guidance_above_01": jnp.mean(guidance > 0.01),
                "guidance_above_10": jnp.mean(guidance > 0.1),
                "guidance_above_25": jnp.mean(guidance > 0.25),
                "guidance_above_40": jnp.mean(guidance > 0.4),
                "guidance_above_50": jnp.mean(guidance > 0.5),
                "guidance_above_75": jnp.mean(guidance > 0.75),
            }
        else:
            info = dict()
        return guidance, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute both the actor and the VaBC losses"""
        batch_size, action_dim = batch["actions"].shape
        rng, distill_rng, flow_x_rng, flow_t_rng = jax.random.split(rng, 4)

        # Distillation loss
        noises = jax.random.normal(distill_rng, (batch_size, action_dim))
        target_vabc_acts = self.compute_flow_actions(
            batch["observations"], noises=noises
        )
        actor_acts = self.network.select("actor")(
            batch["observations"], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_acts - target_vabc_acts) ** 2)

        # Q loss
        actor_acts = jnp.clip(actor_acts, -1, 1)
        qs = self.network.select("critic")(batch["observations"], actions=actor_acts)
        q = getattr(jnp, self.config["q_agg"])(qs, axis=0)
        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        if self.config["normalize_q_loss"]:
            q_loss = lam * q_loss

        # == VaBC loss ==
        # Noising
        x_0 = jax.random.normal(flow_x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(flow_t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # Guidance * error
        pred = self.network.select("vabc")(
            batch["observations"], x_t, t, params=grad_params
        )
        flow_error = jnp.mean((pred - vel) ** 2, axis=1)
        if self.config["eta_temperature"] == 0:
            VaBC_loss = jnp.mean(flow_error)
            g_info = dict()
        else:
            guidance, g_info = self.compute_guidance(
                batch, jax.lax.stop_gradient(actor_acts), target_vabc_acts, lam
            )
            VaBC_loss = jnp.mean(guidance * flow_error)

        # Total loss
        actor_loss = VaBC_loss + self.config["alpha"] * distill_loss + q_loss
        if self.config["log_metrics"]:
            info = dict(
                actor_loss=actor_loss,
                VaBC_loss=VaBC_loss,
                distill_loss=distill_loss,
                q_loss=q_loss,
                **g_info,
            )
        else:
            info = dict()
        return actor_loss, info

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def switch_config_to_online(self):
        new_config = self.config.copy(
            {"eta_temperature": self.config["eta_temperature_online"]}
        )
        return self.replace(config=new_config)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=rng), info

    @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
    def multi_sample_and_update(
        self,
        n_steps_fused,
        dataset,
        dataset_size,
        batch_size,
    ):
        _agent = self
        for _ in range(n_steps_fused):
            batch, rng = GPU_dataset_sample(
                _agent.rng, dataset, batch_size, dataset_size
            )
            rng, loss_rng = jax.random.split(rng)

            def loss_fn(grad_params):
                return _agent.total_loss(batch, grad_params, rng=loss_rng)

            new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
            _agent.target_update(new_network, "critic")

            _agent = _agent.replace(network=new_network, rng=rng)
        return _agent, info

    def sample_noises(self, obs, seed):
        return jax.random.normal(
            seed,
            (
                *obs.shape[: -len(self.config["ob_dims"])],  # (B,) or (,)
                self.config["action_dim"],
            ),
        )

    @jax.jit
    def sample_actions(self, observations, seed, temperature=1.0):
        """Sample actions from the one-step actor policy."""
        noises = self.sample_noises(observations, seed)
        actions = self.network.select("actor")(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the VaBC flow model using the Euler method."""
        if self.config["encoder"] is not None:
            observations = self.network.select("vabc_encoder")(observations)
        actions = noises
        # Euler method
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("vabc")(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_flow_actions(self, observations, seed, temperature=1.0):
        """Sample initial noise and call compute_flow_actions to get VaBC actions"""
        noises = self.sample_noises(observations, seed)
        return self.compute_flow_actions(observations, noises)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["vabc"] = encoder_module()
            encoders["actor"] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        vabc_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("vabc"),
            need_time=True,
            encode_time_dim=config["encode_time_dim"],
        )
        actor_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor"),
            need_time=False,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            vabc=(vabc_def, (ex_observations, ex_actions, ex_times)),
            actor=(actor_def, (ex_observations, ex_actions)),
        )

        if encoders.get("vabc") is not None:
            # Add vabc_encoder to ModuleDict to make it separately callable.
            network_info["vabc_encoder"] = (encoders.get("vabc"), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        if config["eta_temperature_online"] is None:
            config["eta_temperature_online"] = config["eta_temperature"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# =============================================================================
# ORL-SMOOTH: CLI / YAML / W&B / Evaluation / Checkpointing
# =============================================================================

ALGORITHM_NAME = "GFP"
ALGORITHM_FULL_NAME = "Guided Flow Policy — official algorithm, ORL-SMOOTH runner"


@dataclass
class TrainConfig:
    # Experiment.
    device: str = "gpu"
    env: str = "cube-double-play-singletask-v0"
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: int = 42

    # Schedule.
    max_timesteps: int = 1_000_000
    eval_freq: int = 100_000
    n_episodes: int = 50
    eval_at_first_step: bool = True
    eval_flow_policy: bool = True

    # I/O and modes.
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"  # GFP original has no actor-only refit stage.
    hyperparams_path: Optional[str] = "hyperparams/gfp_jax.yml"
    use_hyperparams: bool = True
    run_tag: str = ""  # optional to isolate coefficient sweeps of same env/seed

    # Dataset / sampling.
    batch_size: int = 256
    buffer_size: int = 2_000_000
    normalize_states: bool = False  # official GFP does not perform state normalization
    normalize_reward: bool = False  # official GFP does not perform reward normalization
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None
    dataset_on_gpu: bool = True   # official state-based GFP default
    n_steps_fused: int = 4        # official state-based GFP default

    # Original GFP's shared Adam optimizer.
    lr: Optional[float] = None
    actor_learning_rate: float = 3e-4  # alias for shared lr, not separate actor tx
    critic_learning_rate: float = 3e-4  # accepted for ORL naming; shared optimizer
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None
    actor_layer_norm: Optional[bool] = None
    critic_ln: bool = True
    actor_ln: bool = False
    num_critics: int = 2  # official GFP hardcodes 2
    encoder: Optional[str] = None

    # ORIGINAL GFP algorithm knobs, unchanged and passed through by name.
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"         # critic ensemble aggregation in original GFP
    target_agg: str = "actor"   # reference in Bellman target
    guidance_agg: str = "actor" # reference in VaBC guidance
    guidance_fn: str = "softmax"
    target_use_next: bool = False  # official GFP default (intentionally False)
    alpha: float = 1.0            # flow-to-actor distillation coefficient
    eta_temperature: float = 1e-3 # value-aware BC guidance temperature
    eta_temperature_online: Optional[float] = None
    flow_steps: int = 10
    encode_time_dim: int = 64
    normalize_q_loss: bool = True
    log_metrics: bool = True

    # ORL-SMOOTH logging / checkpointing.
    project: str = "ORL-SMOOTH"
    group: str = "GFP-JAX"
    name: str = "GFP-JAX"
    log_wandb: bool = True
    log_every: int = 5_000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        if self.mode != "train":
            raise NotImplementedError("Original GFP ORL wrapper implements mode='train' only")


def _cli_overridden_fields() -> set[str]:
    return {
        a[2:].split("=", 1)[0].replace("-", "_")
        for a in sys.argv[1:] if a.startswith("--")
    }


def _coerce_hparam_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def _tuple_or_default(dims: Any, dim: int, n: int) -> Tuple[int, ...]:
    if dims is None:
        return tuple([int(dim)] * int(n))
    if not isinstance(dims, (tuple, list)):
        raise ValueError(f"Expected list/tuple for dims, got {dims!r}")
    return tuple(int(x) for x in dims)


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = "GFP-JAX"
    # Sanitize Minari's slash-separated IDs; do NOT change OGBench names.
    safe_env = re.sub(r"[^a-zA-Z0-9_.-]+", "_", config.env)
    parts = [f"GFP-JAX-{safe_env}"]
    # Preserve the ORL distinction between different aggregation experiments.
    if config.q_agg != "mean":
        parts.append("Q" + config.q_agg.upper())
    if config.target_agg != "actor":
        parts.append("T" + config.target_agg.upper())
    if config.guidance_agg != "actor":
        parts.append("G" + config.guidance_agg.upper())
    if config.flow_steps != 10:
        parts.append("F" + str(config.flow_steps))
    if config.run_tag:
        parts.append(re.sub(r"[^a-zA-Z0-9_.-]+", "_", config.run_tag))
    config.name = "-".join(parts)


def validate_config(config: TrainConfig) -> None:
    if config.mode != "train":
        raise NotImplementedError("Original GFP does not implement actor_refit")
    for k in ("eval_freq", "n_episodes", "log_every", "checkpoint_freq", "batch_size", "flow_steps", "n_steps_fused", "hidden_dim", "actor_n_hiddens", "critic_n_hiddens", "encode_time_dim"):
        if int(getattr(config, k)) <= 0:
            raise ValueError(f"{k} must be > 0")
    if config.max_timesteps < 0 or config.buffer_size <= 0:
        raise ValueError("max_timesteps must be nonnegative, buffer_size must be positive")
    if config.num_critics != 2:
        raise ValueError("Official GFP always uses 2 critics; num_critics must remain 2")
    if config.q_agg not in ("min", "mean"):
        raise ValueError("q_agg must be 'min' or 'mean'")
    for k in ("target_agg", "guidance_agg"):
        if getattr(config, k) not in ("actor", "vabc", "min", "mean", "max"):
            raise ValueError(f"Unsupported {k}: {getattr(config, k)!r}")
    if config.guidance_fn not in ("softmax", "advantage"):
        raise ValueError("guidance_fn must be 'softmax' or 'advantage'")
    if config.eta_temperature < 0 or config.alpha < 0:
        raise ValueError("alpha and eta_temperature must be >= 0")
    if not (0 <= config.discount <= 1 and 0 <= config.tau <= 1):
        raise ValueError("discount and tau must be in [0,1]")
    if config.lr is None or config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.normalize_states or config.normalize_reward:
        raise ValueError("To preserve the ORIGINAL GFP data path, normalize_states/reward must be False")
    # Original GPU fused loop only runs at multiples of n_steps_fused.
    if config.dataset_on_gpu:
        for k in ("max_timesteps", "eval_freq", "checkpoint_freq"):
            if getattr(config, k) % config.n_steps_fused:
                raise ValueError(f"With dataset_on_gpu=True, {k} must be divisible by n_steps_fused")


def resolve_config(config: TrainConfig, cli_fields: set[str]) -> TrainConfig:
    if config.dataset_name is not None:
        config.env = config.dataset_name
    # ORL-friendly 'flow' spelling maps to the original code's 'vabc'.
    if config.target_agg == "flow":
        config.target_agg = "vabc"
    if config.guidance_agg == "flow":
        config.guidance_agg = "vabc"
    if ("hidden_dim" in cli_fields or "actor_n_hiddens" in cli_fields) and "actor_hidden_dims" not in cli_fields:
        config.actor_hidden_dims = None
    if ("hidden_dim" in cli_fields or "critic_n_hiddens" in cli_fields) and "value_hidden_dims" not in cli_fields:
        config.value_hidden_dims = None
    config.actor_hidden_dims = _tuple_or_default(
        config.actor_hidden_dims, config.hidden_dim, config.actor_n_hiddens
    )
    config.value_hidden_dims = _tuple_or_default(
        config.value_hidden_dims, config.hidden_dim, config.critic_n_hiddens
    )
    if "critic_ln" in cli_fields and "layer_norm" not in cli_fields:
        config.layer_norm = bool(config.critic_ln)
    if "actor_ln" in cli_fields and "actor_layer_norm" not in cli_fields:
        config.actor_layer_norm = bool(config.actor_ln)
    if config.layer_norm is None:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None:
        config.actor_layer_norm = bool(config.actor_ln)

    # One shared GFP optimizer: explicit CLI --lr and --actor_learning_rate
    # both override the YAML shared lr (with --lr winning if both are set).
    if "lr" in cli_fields and config.lr is not None:
        pass
    elif "actor_learning_rate" in cli_fields:
        config.lr = float(config.actor_learning_rate)
    elif config.lr is None:
        config.lr = float(config.actor_learning_rate)
    config.actor_learning_rate = float(config.lr)
    if float(config.critic_learning_rate) != float(config.lr):
        print("Warning: official GFP has one shared Adam optimizer; "
              f"critic_learning_rate={config.critic_learning_rate} is ignored, using lr={config.lr}.")

    # The original main.py forces visual observations onto CPU dataset sampling.
    if config.env.startswith("visual-"):
        if config.dataset_on_gpu:
            print("Original GFP visual configuration: disabling dataset_on_gpu")
            config.dataset_on_gpu = False
        if config.frame_stack is None:
            config.frame_stack = 3
        if config.p_aug is None:
            config.p_aug = 0.5
        if config.encoder is None:
            config.encoder = "impala_small"

    refresh_algorithm_names(config)
    validate_config(config)
    return config


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    cli_fields = _cli_overridden_fields()
    if config.dataset_name is not None:
        config.env = config.dataset_name
    if not config.use_hyperparams or config.hyperparams_path is None:
        return resolve_config(config, cli_fields)
    path = Path(config.hyperparams_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hyperparameter YAML missing: {path}")
    with path.open(encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}
    selected = all_params.get(config.env)
    if selected is None:
        print(f"No GFP YAML entry for '{config.env}' — using dataclass/CLI values")
        return resolve_config(config, cli_fields)
    aliases = {
        "offline_steps": "max_timesteps", "n_timesteps": "max_timesteps",
        "eval_interval": "eval_freq", "eval_episodes": "n_episodes",
        "env_name": "env", "learning_rate": "lr",
        "n_fused_steps": "n_steps_fused",
    }
    fields = set(TrainConfig.__dataclass_fields__)
    applied, skipped_cli, skipped_unknown = [], [], []
    for old_key, raw_value in selected.items():
        key = aliases.get(old_key, old_key)
        if key not in fields:
            skipped_unknown.append(old_key)
            continue
        if key in cli_fields or old_key in cli_fields:
            skipped_cli.append(old_key)
            continue
        # An explicit --actor_learning_rate also overrides YAML 'lr'.
        if key == "lr" and "actor_learning_rate" in cli_fields:
            skipped_cli.append(old_key)
            continue
        setattr(config, key, _coerce_hparam_value(raw_value))
        applied.append(key)
    config = resolve_config(config, cli_fields)
    if applied:
        print(f"Loaded GFP hyperparameters for {config.env} from {path}: {', '.join(applied)}")
    if skipped_cli:
        print("Kept explicit CLI overrides: " + ", ".join(skipped_cli))
    if skipped_unknown:
        print("Ignored unknown keys in GFP YAML: " + ", ".join(skipped_unknown))
    return config


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name, str(config.seed))
    return config


def select_jax_device(device: str):
    backend = "gpu" if device.lower() == "cuda" else device.lower()
    try:
        out = jax.devices(backend)[0]
    except Exception:
        print(f"Requested device '{device}' unavailable; falling back to default JAX device")
        out = jax.devices()[0]
    print(f"Using JAX device: {out}")
    return out


def set_seed(seed: int, env: Any = None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        for space_name in ("action_space", "observation_space"):
            space = getattr(env, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(seed)


def agent_config(config: TrainConfig) -> Dict[str, Any]:
    """Only renamed ORL fields are mapped; algorithm arguments mirror original config/agent/gfp.yaml."""
    return {
        "agent_name": "gfp",
        "lr": float(config.lr),
        "batch_size": int(config.batch_size),
        "actor_hidden_dims": tuple(config.actor_hidden_dims),
        "value_hidden_dims": tuple(config.value_hidden_dims),
        "layer_norm": bool(config.layer_norm),
        "actor_layer_norm": bool(config.actor_layer_norm),
        "discount": float(config.discount),
        "tau": float(config.tau),
        "encoder": config.encoder,
        "alpha": float(config.alpha),
        "eta_temperature": float(config.eta_temperature),
        "eta_temperature_online": config.eta_temperature_online,
        "q_agg": config.q_agg,
        "target_agg": config.target_agg,
        "guidance_agg": config.guidance_agg,
        "guidance_fn": config.guidance_fn,
        "target_use_next": bool(config.target_use_next),
        "flow_steps": int(config.flow_steps),
        "normalize_q_loss": bool(config.normalize_q_loss),
        "encode_time_dim": int(config.encode_time_dim),
        "log_metrics": bool(config.log_metrics),
    }


def _benchmark(env: str) -> str:
    if "singletask" in env:
        return "OGBench"
    if env.startswith(("D4RL/", "mujoco/")):
        return "Minari"
    return "D4RL"


def make_env_and_datasets(config: TrainConfig):
    """Use official GFP dataset/environment code, except its D4RL init-order typo."""
    benchmark = _benchmark(config.env)
    if benchmark != "D4RL":
        return (*original_make_env_and_datasets(
            benchmark, config.env,
            frame_stack=config.frame_stack,
            action_clip_eps=config.action_clip_eps,
        ), benchmark)

    # The uploaded original env_utils.py attempts get_dataset(env,...) before
    # creating `env`; correct only that I/O initialization order, keeping its
    # ORIGINAL d4rl_utils.get_dataset() and .make_env() implementations.
    global d4rl
    if d4rl is None:
        import d4rl as d4rl_module
        d4rl = d4rl_module
    env = d4rl_utils.make_env(config.env)
    train_dataset = d4rl_utils.get_dataset(env, config.env)
    if config.action_clip_eps is not None:
        eps = float(config.action_clip_eps)
        train_dataset = train_dataset.copy(add_or_replace={
            "actions": np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
        })
    def new_eval_env():
        eval_env = d4rl_utils.make_env(config.env)
        eval_env.reset()
        return eval_env
    env.reset()
    return env, new_eval_env, train_dataset, None, benchmark


def _safe_float(x: Any) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float("nan")


def evaluate_policy(agent_fn, eval_env: Any, episodes: int, seed: int, benchmark: str):
    """FQL/ORL-style per-episode logging; same GFP.sample_actions/flow_actions."""
    rng = jax.random.PRNGKey(int(seed))
    stats: Dict[str, list] = {}
    returns: list[float] = []
    successes: list[float] = []
    for ep in trange(episodes, desc="GFP Eval", leave=False):
        try:
            observation, info = eval_env.reset(seed=seed if ep == 0 else None)
        except TypeError:
            observation, info = eval_env.reset()
        done = False
        total_reward = 0.0
        last_info: Dict[str, Any] = {}
        while not done:
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(agent_fn(observations=observation, seed=action_rng, temperature=0))
            action = np.clip(action, -1, 1)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = dict(info)
        # Same success definition as the official evaluate and EpisodeMonitor.
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = prefix + str(k)
                if hasattr(v, "items"):
                    yield from flatten(v, key + ".")
                else:
                    yield key, v
        flat = dict(flatten(last_info))
        for k, v in flat.items():
            fval = _safe_float(v)
            if np.isfinite(fval):
                stats.setdefault(k, []).append(fval)
        returns.append(float(flat.get("episode.return", total_reward)))
        suc = flat.get("success", flat.get("episode.success", float("nan")))
        successes.append(_safe_float(suc))
    arr_rew = np.asarray(returns, dtype=np.float32)
    arr_suc = np.asarray(successes, dtype=np.float32)
    output = {k: float(np.mean(vals)) for k, vals in stats.items()}
    output["reward_mean"] = float(np.mean(arr_rew))
    output["reward_std"] = float(np.std(arr_rew))
    valid = arr_suc[np.isfinite(arr_suc)]
    output["success_rate"] = float(np.mean(valid)) if valid.size else float("nan")
    output["success_std"] = float(np.std(valid)) if valid.size else float("nan")
    if benchmark == "D4RL":
        raw = np.asarray(stats.get("episode.normalized_return", []), dtype=np.float32)
        output["d4rl_normalized_score_mean"] = float(np.mean(raw)) if raw.size else float("nan")
        output["d4rl_normalized_score_std"] = float(np.std(raw)) if raw.size else float("nan")
    elif benchmark == "Minari":
        # minari_normalized_score is defined above from the official source.
        scores = np.asarray([minari_normalized_score(v) for v in arr_rew], dtype=np.float32)
        output["d4rl_normalized_score_mean"] = float(np.mean(scores))
        output["d4rl_normalized_score_std"] = float(np.std(scores))
    else:
        output["d4rl_normalized_score_mean"] = float("nan")
        output["d4rl_normalized_score_std"] = float("nan")
    return output


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pickle(path: Union[Path, str]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_checkpoint_path(load_model: str, name: str, seed: int) -> Path:
    path = Path(load_model)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"--load_model not found: {path}")
    for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
        for candidate in (path / filename, path / name / str(seed) / filename):
            if candidate.is_file():
                return candidate
    results = [c for pattern in ("*/checkpoint.pkl", "*/best_checkpoint.pkl", "*/*/checkpoint.pkl", "*/*/best_checkpoint.pkl") for c in sorted(path.glob(pattern))]
    results = list(dict.fromkeys(results))
    if len(results) == 1:
        return results[0]
    if not results:
        raise FileNotFoundError(f"Could not find checkpoint.pkl/best_checkpoint.pkl under {path}")
    raise ValueError("Multiple checkpoint candidates found; specify an exact path: " + ", ".join(map(str, results)))


def save_eval_logs(eval_logs: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path or not eval_logs:
        return
    keys = list(dict.fromkeys(k for row in eval_logs for k in row))
    arrays = {}
    for key in keys:
        vals = [row.get(key, float("nan")) for row in eval_logs]
        try:
            arrays[key] = np.asarray(vals, dtype=np.float64)
        except (ValueError, TypeError):
            arrays[key] = np.asarray(vals, dtype=object)
    np.savez(os.path.join(path, "eval_logs.npz"), **arrays)


def _make_payload(agent, cfg: TrainConfig, step: int):
    return {
        "agent": serialization.to_state_dict(agent),
        "config": asdict(cfg),
        "_training_timestep": int(step),
        "_numpy_rng_state": np.random.get_state(),
        "_python_rng_state": random.getstate(),
    }


def _restore_payload(agent, payload):
    state = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent = serialization.from_state_dict(agent, state)
    if isinstance(payload, dict):
        if "_numpy_rng_state" in payload:
            np.random.set_state(payload["_numpy_rng_state"])
        if "_python_rng_state" in payload:
            random.setstate(payload["_python_rng_state"])
    return agent


def _wandb_init(cfg: TrainConfig, manager) -> bool:
    if not cfg.log_wandb:
        return False
    if manager is not None:
        ok = manager.initialize_wandb_with_fallback(
            wandb_module=wandb, config=asdict(cfg), code_root=_PROJECT_ROOT,
        )
        if not ok:
            cfg.log_wandb = False
        return bool(ok)
    try:
        run = wandb.init(
            config=asdict(cfg), project=cfg.project, group=cfg.group,
            name=cfg.name, id=str(uuid.uuid4()),
        )
        if run is not None and hasattr(run, "log_code"):
            run.log_code(str(_PROJECT_ROOT))
        return True
    except Exception as exc:
        print(f"Warning: W&B initialization failed: {exc}; continuing locally")
        cfg.log_wandb = False
        return False


def _train_impl(config: TrainConfig):
    config = finalize_checkpoint_path(apply_env_hyperparams(config))
    run_dir = Path(config.checkpoints_path) if config.checkpoints_path else None
    manager, preparation = None, None
    if run_dir is not None:
        manager = TrainingCheckpointManager(
            run_dir=str(run_dir),
            current_config=asdict(config),
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="gfp_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=config.wandb_entity,
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        preparation = manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "config.yaml").exists():
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(asdict(config), f, sort_keys=True, allow_unicode=True)

    dev = select_jax_device(config.device)
    train_env, eval_env_creator, train_dataset, val_dataset, benchmark = make_env_and_datasets(config)
    # Match original offline GFP's replay-buffer sampling. For visual data, do
    # not allocate the unnecessary, potentially enormous 2M image replay buffer.
    if not config.env.startswith("visual-"):
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(config.buffer_size, train_dataset.size + 1)
        )
    for dataset in (train_dataset, val_dataset):
        if dataset is not None:
            dataset.p_aug = config.p_aug
            dataset.frame_stack = config.frame_stack
            dataset.return_next_actions = False

    set_seed(config.seed, train_env)
    example_batch = train_dataset.sample(1)
    original_cfg = agent_config(config)
    agent = GFPAgent.create(config.seed, example_batch["observations"], example_batch["actions"], original_cfg)
    agent = jax.device_put(agent, dev)
    print("-" * 60)
    print(f"Training GFP-JAX, Env: {config.env}, Seed: {config.seed}")
    print(f"Original GFP agent config: {original_cfg}")
    print(f"Saving under: {run_dir if run_dir else '(disabled)'}")
    print("-" * 60)

    train_step = 0
    eval_logs: List[Dict[str, Any]] = []
    if config.load_model:
        file = _resolve_checkpoint_path(config.load_model, config.name, config.seed)
        payload = _load_pickle(file)
        agent = jax.device_put(_restore_payload(agent, payload), dev)
        train_step = int(payload.get("_training_timestep", 0)) if isinstance(payload, dict) else 0
        print(f"Loaded GFP checkpoint {file} at step {train_step}")
        if run_dir is not None and file.parent == run_dir and (run_dir / "eval_logs.npz").is_file():
            with np.load(run_dir / "eval_logs.npz", allow_pickle=True) as z:
                keys = list(z.files)
                count = len(z[keys[0]]) if keys else 0
                eval_logs = [dict((k, z[k][i].item() if np.asarray(z[k][i]).ndim == 0 else z[k][i]) for k in keys) for i in range(count)]

    def current_step():
        return int(train_step)

    def progress_state():
        return _make_payload(agent, config, train_step)

    def load_progress(payload):
        nonlocal agent, train_step
        agent = jax.device_put(_restore_payload(agent, payload), dev)
        train_step = int(payload.get("_training_timestep", 0))

    if manager is not None:
        if preparation.is_resuming:
            train_step, eval_logs, _ = manager.restore(
                load_trainer_state=load_progress, get_restored_timestep=current_step,
            )
            print(f"Automatically restored GFP training at timestep {train_step}")
        else:
            manager.save_progress(
                timestep=train_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running",
            )

    # W&B initialization is I/O, so keep its RNG use from altering training draws.
    _before_wandb_np = np.random.get_state()
    _before_wandb_py = random.getstate()
    _wandb_init(config, manager)
    np.random.set_state(_before_wandb_np)
    random.setstate(_before_wandb_py)

    def wandb_log(metrics: Dict[str, float], step: int):
        if not config.log_wandb or not metrics:
            return
        if manager is not None:
            manager.log_wandb(metrics, int(step))
        else:
            wandb.log(metrics, step=int(step))

    # The official GPU fused path samples directly with the JAX RNG from a
    # device-resident dataset; the exact official multi_sample_and_update runs.
    data_gpu = None
    if config.dataset_on_gpu:
        data_gpu = {k: jax.device_put(np.asarray(v), dev) for k, v in train_dataset.items()}
        dataset_size = int(train_dataset.size)

    first_time = time.time()
    prev_time = first_time
    last_update_info: Dict[str, Any] = {}
    best_eval_metric_mean = best_eval_metric(eval_logs)
    first_eval_step = config.n_steps_fused if config.dataset_on_gpu else 1

    def evaluation_is_required(step: int) -> bool:
        return (
            (config.eval_at_first_step and step == first_eval_step)
            or step % int(config.eval_freq) == 0
            or step == int(config.max_timesteps)
        )

    try:
        for step in trange(train_step + 1, int(config.max_timesteps) + 1, desc="GFP Training"):
            if config.dataset_on_gpu and step % config.n_steps_fused:
                continue
            if config.dataset_on_gpu:
                agent, update_info = agent.multi_sample_and_update(
                    int(config.n_steps_fused), data_gpu, dataset_size, int(config.batch_size)
                )
            else:
                batch = train_dataset.sample(int(config.batch_size))
                agent, update_info = agent.update(batch)
            train_step = int(step)
            last_update_info = {k: float(jax.device_get(v)) for k, v in update_info.items()}

            if config.log_wandb and config.log_metrics and train_step % int(config.log_every) == 0:
                train_info = {f"training/{key}": val for key, val in last_update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(int(config.batch_size))
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_info.update({
                        f"validation/{k}": float(jax.device_get(v))
                        for k, v in val_info.items()
                    })
                train_info["time/epoch_time"] = (time.time() - prev_time) / int(config.log_every)
                train_info["time/total_time"] = time.time() - first_time
                prev_time = time.time()
                wandb_log(train_info, train_step)

            if evaluation_is_required(train_step):
                actor_env = eval_env_creator()
                actor_results = evaluate_policy(
                    agent.sample_actions, actor_env, int(config.n_episodes),
                    int(config.eval_seed), benchmark,
                )
                if hasattr(actor_env, "close"):
                    actor_env.close()
                eval_log: Dict[str, Any] = {
                    "timestep": int(train_step),
                    "eval/reward_mean": actor_results["reward_mean"],
                    "eval/reward_std": actor_results["reward_std"],
                    "eval/d4rl_normalized_score_mean": actor_results["d4rl_normalized_score_mean"],
                    "eval/d4rl_normalized_score_std": actor_results["d4rl_normalized_score_std"],
                    "eval/success_rate": actor_results["success_rate"],
                    "eval/success_std": actor_results["success_std"],
                }
                if config.eval_flow_policy:
                    flow_env = eval_env_creator()
                    flow_results = evaluate_policy(
                        agent.sample_flow_actions, flow_env, int(config.n_episodes),
                        int(config.eval_seed), benchmark,
                    )
                    if hasattr(flow_env, "close"):
                        flow_env.close()
                    for key in ("reward_mean", "reward_std", "success_rate", "success_std"):
                        eval_log["eval_flow/" + key] = flow_results[key]
                    eval_log["eval_flow/d4rl_normalized_score_mean"] = flow_results["d4rl_normalized_score_mean"]
                    eval_log["eval_flow/d4rl_normalized_score_std"] = flow_results["d4rl_normalized_score_std"]
                if benchmark == "OGBench" and train_step == int(config.max_timesteps):
                    previous = [
                        log.get("eval/success_rate", float("nan"))
                        for log in eval_logs
                        if log.get("timestep", -1) < train_step
                    ]
                    values = previous[-2:] + [eval_log["eval/success_rate"]]
                    if len(values) == 3 and np.all(np.isfinite(values)):
                        # Official GFP paper reports the last three evaluations.
                        eval_log["eval/final_last3_success_rate"] = float(np.mean(values))
                        eval_log["eval/final_last3_success_score"] = float(100 * np.mean(values))
                upsert_eval_log(eval_logs, eval_log)
                print(f"Step {train_step} GFP Actor: reward={eval_log['eval/reward_mean']:.3f} ± "
                      f"{eval_log['eval/reward_std']:.3f}, success={eval_log['eval/success_rate']:.3f} ± "
                      f"{eval_log['eval/success_std']:.3f}")
                if config.eval_flow_policy:
                    print(f"    GFP VaBC flow: success={eval_log['eval_flow/success_rate']:.3f}")
                wandb_log(eval_log, train_step)
                save_eval_logs(eval_logs, config.checkpoints_path)

                if config.save_best_model and run_dir is not None:
                    # ORL convention: select actor only by success, then normalized
                    # score, then reward. Flow policy is auxiliary, never used for best.
                    metric = eval_log["eval/success_rate"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/d4rl_normalized_score_mean"]
                    if not np.isfinite(metric):
                        metric = eval_log["eval/reward_mean"]
                    if np.isfinite(metric) and metric > best_eval_metric_mean:
                        best_eval_metric_mean = metric
                        _save_pickle(run_dir / "best_checkpoint.pkl", progress_state())

            if manager is not None and train_step % int(config.checkpoint_freq) == 0:
                manager.save_progress(
                    timestep=train_step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="running",
                )
    except BaseException:
        if manager is not None:
            committed = (
                not evaluation_is_required(int(train_step))
                or find_eval_log(eval_logs, int(train_step)) is not None
            )
            if committed:
                manager.save_progress(
                    timestep=int(train_step), trainer_state=progress_state(),
                    eval_logs=eval_logs, status="interrupted",
                )
                print(f"Saved interrupted GFP training at timestep {train_step}")
            else:
                print("GFP evaluation was interrupted; retaining previous safe checkpoint")
        raise

    if manager is not None:
        path = manager.complete(
            timestep=int(train_step), final_state=progress_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs,
        )
        if path is not None:
            print(f"Saved final GFP checkpoint: {path}")
    if hasattr(train_env, "close"):
        train_env.close()


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


# =============================================================================
# ORIGINAL MIT LICENSE (retained for standalone source redistribution)
# The MIT License (MIT)
# 
# Copyright (c) 2025, FQL Authors
# Copyright (c) 2025-2026, Inria
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
