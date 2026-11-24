"""Flow Actor-Critic (FAC) in ONE Python file, with ORL-SMOOTH experiment plumbing.

FAC mathematical methods, model layers, TrainState, and datasets are extracted
verbatim from the user-supplied FAC-main.zip. No imports of external local
FAC sources are needed; they are defined in THIS FILE.

Only local project dependency: algorithms.uk_offline.common.checkpointing
(a shared experiment/resume helper, NOT part of FAC mathematical algorithm).

Implementation scope: original OFFLINE FAC (BC flow proxy training, density
precomputation, and FAC actor-critic optimization). No online fine-tuning.

Code provenance: FAC-main.zip (2026 FAC Authors), see LICENSE in package.
Static structural verification: VERIFY_FAC_SINGLE_FILE.py.
"""

import os
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')

import collections
import copy
import functools
import glob
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
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import distrax
import flax
import flax.linen as nn
from flax import serialization
from flax.core.frozen_dict import FrozenDict
from flax.training import orbax_utils
import gymnasium
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax
import pyrallis
import tqdm
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


# ==============================================================================
# ORIGINAL FAC: utils/networks.py
# ==============================================================================

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


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param('value', lambda x:self.init_value)

    def __call__(self):
        return self.value


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


# ==============================================================================
# ORIGINAL FAC: utils/encoders.py
# ==============================================================================

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


# ==============================================================================
# ORIGINAL FAC: utils/flax_utils.py
# ==============================================================================

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
    
    def load(self, load_dict):
        opt_state = self.tx.init(load_dict)
        return self.replace(
            params=load_dict,
            opt_state=opt_state
        )


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent


def save_only_bcmodel(agent, save_dir, env_name, seed):
    """Save the flow BC model of the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """
    save_path = os.path.join(os.getcwd(), save_dir, f'{env_name}_seed_{seed}')
    for name in agent.bc_network.params.keys():
        save_name = name.split('modules_')[1]
        save_name = os.path.join(save_path, save_name)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = {'model': agent.bc_network.params[name]}
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(save_name, ckpt, save_args=save_args, force=True)

    print(f'Saved to {save_path}')


def restore_only_bcmodel(agent, restore_path, env_name, seed):
    """Restore the flow BC model of the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    load_dict = dict()
    load_path = os.path.join(os.getcwd(), restore_path, f'{env_name}_seed_{seed}')
    for name in agent.bc_network.params.keys():
        load_name = name.split('modules_')[1]
        load_name = os.path.join(load_path, load_name)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        target = {'model': agent.bc_network.params[name]}
        load_model = orbax_checkpointer.restore(load_name, item=target)
        load_dict[name] = load_model['model']
    bc_network = agent.bc_network.load(load_dict)
    agent = agent.replace(bc_network=bc_network)
    return agent


# ==============================================================================
# ORIGINAL FAC: utils/datasets.py
# ==============================================================================

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
        self.frame_stack = None # Number of frames to stack
        self.p_aug = None # Image augmentation probability
        self.return_next_actions = False # Whether to additionally return next actions

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

        # Compute minimum log-density
        self.estimated_logp_min = None

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
            obs = [] # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = [] # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
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
    
    def compute_and_attach_estimated_logp(self, agent, method, batch_size=8192):
        """Compute log beta(a|s) for the entire dataset and attach it to self._dict['estimated_logp'] as float32."""
        length = getattr(self, 'max_size', self.size) # max_size if ReplayBuffer else dataset.size
        if 'estimated_logp' not in self._dict:
            self._dict['estimated_logp'] = np.full((length,), np.nan, dtype=np.float32)

        old_p_aug = self.p_aug
        self.p_aug = None

        all_indices = np.arange(self.size, dtype=np.int64)
        num_chunks = (self.size + batch_size - 1) // batch_size

        local_rng = agent.rng
        for ci in tqdm.tqdm(range(num_chunks), smoothing=0.1, dynamic_ncols=True, desc='compute logp'):
            start = ci * batch_size
            end = min((ci + 1) * batch_size, self.size)
            idxs = all_indices[start:end]

            batch = self.sample(batch_size=(end - start), idxs=idxs)

            local_rng, key = jax.random.split(local_rng)
            logp = agent.logprob_given_actions(
                observations=batch['observations'],
                actions_final=batch['actions'],
                rng=key,
                mode=method,
            )

            self._dict['estimated_logp'][idxs] = np.array(logp, dtype=np.float32)

        self.p_aug = old_p_aug

        # Compute the minimum of estimated log-densities over dataset.
        logp_min = np.nanmin(self._dict['estimated_logp'][: self.size]).astype(np.float32)
        self._dict['estimated_logp_min'] = np.full((length,), logp_min, dtype=np.float32)
        self.estimated_logp_min = float(logp_min)


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


# ==============================================================================
# ORIGINAL FAC: utils/evaluation.py
# ==============================================================================

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
    for i in range(num_eval_episodes + num_video_episodes):
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


def inference_time(
    agent,
    batch,
    config=None,
    eval_temperature=0,
):
    import time 
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    observation = batch['observations'][0]

    for i in range(30000):
        action = actor_fn(observations=observation, temperature=eval_temperature)
        action = np.array(action)
        action = np.clip(action, -1, 1)

        if i == 10000:
            start = time.time()

        if i == 20000:
            delta_time = time.time() - start
            print(f'inference{delta_time}')

    return action


# ==============================================================================
# ORIGINAL FAC: envs/env_utils.py
# ==============================================================================

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


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        # d4rl_utils is an in-file namespace bound below

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite # noqa
        # d4rl_utils is an in-file namespace bound below

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'halfcheetah' in env_name or 'hopper' in env_name or 'walker2d' in env_name:
        # D4RL MuJoCo
        # d4rl_utils is an in-file namespace bound below

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


# ==============================================================================
# ORIGINAL FAC: envs/d4rl_utils.py
# ==============================================================================

def make_env(env_name):
    """Make D4RL environment."""
    import d4rl  # noqa: F401  # Register D4RL environments only when needed.
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
    import d4rl
    dataset = d4rl.qlearning_dataset(env)

    terminals = np.zeros_like(dataset['rewards']) # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards']) # Indicate whether we should bootstrap from the next state.
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

# Rebind the original `d4rl_utils` namespace inside this single file.
d4rl_utils = SimpleNamespace(make_env=make_env, get_dataset=get_dataset)


# ==============================================================================
# ORIGINAL FAC: agents/fac.py
# ==============================================================================

class FACAgent(flax.struct.PyTreeNode):
    """Flow Actor-Crtic (FAC) agent."""

    rng: Any
    ac_network: Any
    bc_network: Any 
    config: Any = nonpytree_field()

    # --- AC loss -----------------------------------------------------------
    def critic_loss(self, batch, grad_params, rng, use_cp=True, online_finetuning=False):
        """Compute the flow-based conservative critic loss."""
        # Bellman backup loss
        rng, sample_rng, est_rng = jax.random.split(rng, 3) # [num_q, ]
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng) # [batch, a_dim]
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.ac_network.select('target_critic')(batch['next_observations'], actions=next_actions) # [num_q, batch]
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q # [batch, ]
        q_pred = self.ac_network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params) # [num_q, batch]
        td_loss = jnp.square(q_pred - target_q).mean()

        if use_cp:
            # Flow-based (conservative) Q penalization
            if not online_finetuning:
                if self.config['fac_threshold'] == 'batch_adaptive': # an epsilon scheme
                    estimated_logp_beta = batch['estimated_logp'] # [batch,]
                elif self.config['fac_threshold'] == 'batch_wide_constant':
                    estimated_logp_beta = jnp.full_like(batch['estimated_logp'], jnp.min(batch['estimated_logp'])) # [batch,]
                elif self.config['fac_threshold'] == 'dataset_wide_constant':
                    estimated_logp_beta = batch['estimated_logp_min'] # [batch,]
            else:
                rng, est_rng = jax.random.split(rng, 2) # [num_q, ]
                estimated_logp_beta = jax.lax.stop_gradient(self.logprob_given_actions(observations=batch['observations'],
                                                                                       actions_final=batch['actions'],
                                                                                       rng=est_rng,
                                                                                       mode=self.config['logp_method'],
                                                                                       )) # [batch,]
                if self.config['fac_threshold'] != 'batch_adaptive':
                    # Due to additional transition during online finetuning, 
                    # 'dataset_wide_constant' threshold scheme may increases unnecessary computation.
                    # Thus, we can use 'batch_wide_constant' threshold scheme.
                    estimated_logp_beta = jnp.full_like(estimated_logp_beta, jnp.min(estimated_logp_beta)) # [batch,]
            
            rng, sample_rng, est_rng = jax.random.split(rng, 3) # [num_q, ]

            penalty_actions = jnp.clip(self.sample_actions(batch['observations'], seed=sample_rng), -1, 1) # [batch, a_dim]
            qs_penalty_actions = self.ac_network.select('critic')(batch['observations'], actions=penalty_actions, params=grad_params) # [num_q, batch]
            estimated_logp_pi = jax.lax.stop_gradient(self.logprob_given_actions(observations=batch['observations'],
                                                                                 actions_final=penalty_actions,
                                                                                 rng=est_rng,
                                                                                 mode=self.config['logp_method'],
                                                                                 )) # [batch,]
            diff_logp = estimated_logp_pi - estimated_logp_beta

            if self.config['weight_type'] == 'linear':
                weight_penalty_per_actions = 1.0 - jnp.exp(diff_logp) # control below threshold
                # weight_penalty_actions = jnp.exp(diff_logp) # control above threshold
                weight_penalty_per_actions = jnp.clip(weight_penalty_per_actions, 0., 1.)
            else:
                if self.config['weight_type'] == 'logarithmic':
                    temperature_for_weight = 1.0
                elif self.config['weight_type'] == 'convex':
                    temperature_for_weight = 0.5
                elif self.config['weight_type'] == 'concave':
                    temperature_for_weight = 2.0
                weight_penalty_per_actions = 1.0 - (jnp.log1p(jnp.exp(diff_logp)**temperature_for_weight) / jnp.log(2.0))
                weight_penalty_per_actions = jnp.clip(weight_penalty_per_actions, 0., 1.)


            qs_penalty_actions = jax.lax.stop_gradient(weight_penalty_per_actions) * qs_penalty_actions # [num_q, batch]

            critic_penalty = self.config['fac_alpha'] * (qs_penalty_actions) # [num_q, ]
            critic_penalty = critic_penalty.mean()

        else:
            # if use_cp=False, no Q-value penalizing.
            critic_penalty = 0.0

        critic_loss = td_loss + critic_penalty

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_pred.mean(),
            'q_max': q_pred.max(),
            'q_min': q_pred.min(),
            'penalty_mean': critic_penalty,
            'diff_logpi_mean': diff_logp.mean(),
            'diff_logpi_max': diff_logp.max(),
            'diff_logpi_min': diff_logp.min(),
            'est_logpi_mean': estimated_logp_pi.mean(),
            'est_logpi_max': estimated_logp_pi.max(),
            'est_logpi_min': estimated_logp_pi.min(),
            'est_logbeta_mean': estimated_logp_beta.mean(),
            'est_logbeta_max': estimated_logp_beta.max(),
            'est_logbeta_min': estimated_logp_beta.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FAC actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        if self.config['distil_loss']:
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.ac_network.select('actor_onestep')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        else:
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            actor_actions = self.ac_network.select('actor_onestep')(batch['observations'], noises, params=grad_params)
            distill_loss = 0.0

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.ac_network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss
        actor_loss = self.config['fac_lambda'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
        }
    # --------------------------------------------------------------
    
    # --- BC loss -----------------------------------------------------------
    def bc_loss(self, batch, grad_params, rng):
        """Compute the flow proxy (behavior cloning) loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.bc_network.select('bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }
    # --------------------------------------------------------------

    # --- Update -----------------------------------------------------------
    @partial(jax.jit, static_argnames=("mode", "use_cp", "online_finetuning"))
    def total_loss(self, batch, grad_params, 
                   mode,                    # mode: 'train_bc' or 'train_ac'
                   use_cp=True,             # if training Q-values conservatively or not for online finetuning
                   online_finetuning=False, # if online finetuning or not
                   rng=None):
        """Compute the AC or BC loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        if mode == 'train_bc':
            rng, bc_rng = jax.random.split(rng, 2)
            bc_loss, bc_info = self.bc_loss(batch, grad_params, bc_rng)
            for k, v in bc_info.items():
                info[f'bc/{k}'] = v

            loss = bc_loss

        elif mode == 'train_ac':
            rng, actor_rng, critic_rng = jax.random.split(rng, 3)
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, use_cp, online_finetuning)
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v

            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            loss = critic_loss + actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=("mode", "use_cp", "online_finetuning"))
    def update(self, batch, mode,       # mode: 'train_bc' or 'train_ac'
               use_cp=True,             # if training Q-values conservatively or not for online finetuning
               online_finetuning=False, # if online finetuning or not
               ):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, mode, use_cp, online_finetuning, rng=rng)

        if mode == 'train_bc':
            new_bc_network, info = self.bc_network.apply_loss_fn(loss_fn=loss_fn)

            return self.replace(bc_network=new_bc_network, rng=new_rng), info
        
        elif mode == 'train_ac':
            new_ac_network, info = self.ac_network.apply_loss_fn(loss_fn=loss_fn)
            self.target_update(new_ac_network, 'critic')

            return self.replace(ac_network=new_ac_network, rng=new_rng), info
    # --------------------------------------------------------------

    # --- Action sampling-----------------------------------------------------------
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        ) # [batch, a_dim]
        actions = self.ac_network.select('actor_onestep')(observations, noises) # [batch, a_dim]
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.bc_network.select('bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.bc_network.select('bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    # --------------------------------------------------------------
    

    # --- Compute log-denstiy (logp) -----------------------------------------------------------
    @staticmethod
    def make_divergence_exact(v_apply):
        """v_apply: (obs[B,D_s], act[B,D_a], t[B,1]) -> vel[B,D_a]"""
        def single_div(o, a, t):
            # single-sample v_fn: R^A -> R^A
            def v_fn(a_in):
                return v_apply(o[None, :], a_in[None, :], t[None, None])[0]
            J = jax.jacrev(v_fn)(a) # Jacobian; (A, A)
            return jnp.trace(J)
        return jax.vmap(single_div, in_axes=(0, 0, 0))

    @staticmethod
    def make_divergence_hutchinson(v_apply, probes=8, gaussian=False):
        """Hutchinson trace estimator: E[e^T J e], e ~ normal or Rademacher distribution"""
        def single_div(o, a, t, key):
            def v_fn(a_in):
                return v_apply(o[None, :], a_in[None, :], t[None, None])[0]

            def vJv(k):
                _, vjp = jax.vjp(v_fn, a) # vector-Jacobian product, i.e., eps^T J
                eps = (jax.random.normal(k, a.shape, dtype=a.dtype) if gaussian
                    else jax.random.rademacher(k, a.shape, dtype=a.dtype))
                return jnp.dot(vjp(eps)[0], eps) # eps^T J eps

            keys = jax.random.split(key, probes)
            vals = jax.vmap(vJv)(keys)
            return vals.mean()

        return jax.vmap(single_div, in_axes=(0, 0, 0, 0))
    
    @partial(jax.jit, static_argnames=("mode"))
    def logprob_given_actions(
        self,
        observations,   # (B, S_dim)
        actions_final,  # (B, A_dim)
        rng=None,       # for Hutchinson estimation
        mode="exact",   # "exact" or "hutch"
    ):
        """Return: logp: (B,); log-density of flow BC model"""
        N = self.config['flow_steps']
        dt = 1.0 / N
        B, A_dim = actions_final.shape

        if self.config['encoder'] is not None:
            obs_enc = self.bc_network.select('bc_flow_encoder')(observations)
        else:
            obs_enc = observations

        def v_apply(obs_b, act_b, t_b): # obs_b: (B, S_dim), act_b: (B, A_dim), t_b: (B,1)
            return self.bc_network.select('bc_flow')(obs_b, act_b, t_b, is_encoded=True)

        # 1) Define divergence
        if mode == "exact":
            div_fn = self.make_divergence_exact(v_apply)
            need_key = False
        elif "hutch" in mode:
            assert rng is not None
            if mode == "hutch-rade":
                div_fn = self.make_divergence_hutchinson(v_apply, 
                                                         probes=self.config['logp_hutch_probes'], 
                                                         gaussian=False)
            elif mode == "hutch-gaus":
                div_fn = self.make_divergence_hutchinson(v_apply, 
                                                         probes=self.config['logp_hutch_probes'], 
                                                         gaussian=True)
            need_key = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2) Initial point for reverse ODE problem
        a_t = actions_final # at t=1
        logp_acc = jnp.zeros((B,), dtype=actions_final.dtype) # initial point: log pi(a_1|s)=0

        if need_key:
            rngs = jax.random.split(rng, N * B)
            rngs = rngs.reshape(N, B, 2)  # (N,B,2)
        else:
            rngs = None

        # 3) Reverse ODE solve via Euler method
        def body_fun(k, carry):
            a_cur, logp_cur = carry
            # time index: from 1 to 0, so t_k = (k+1)/N
            t_k = (N - k) / N
            t_batch = jnp.full((B, 1), t_k, dtype=a_cur.dtype)

            # velocity at current (obs_enc, a_cur, t_k)
            vels = v_apply(obs_enc, a_cur, t_batch)

            # divergence
            if need_key:
                div_keys = rngs[k, :, :]  # (B, 2)
                divs = div_fn(obs_enc, a_cur, t_batch.squeeze(-1), div_keys)
            else:
                divs = div_fn(obs_enc, a_cur, t_batch.squeeze(-1))

            # update logp, a_u
            logp_new = logp_cur + divs * dt # actually, data_logp = base_logp - int_logp_div
                                            # this function returns base_logp - sum_logp_div, 
                                            # so sum_logp_div can be computed like that.
            a_new   = a_cur - vels * dt     # reverse Euler step

            return (a_new, logp_new)

        # lax.fori_loop(0, N, ...): k=0..N-1
        a_0, logp_div = jax.lax.fori_loop(0, N, body_fun, (a_t, logp_acc))

        # 4) base log prob at t=0 (assume standard normal distribution)
        base_logp = -0.5 * jnp.sum(a_0**2, axis=-1) - 0.5 * A_dim * jnp.log(2 * jnp.pi)

        return base_logp - logp_div # (B,)
    # --------------------------------------------------------------

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
        rng, init_rng_ac, init_rng_bc = jax.random.split(rng, 3)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['bc_flow'] = encoder_module()
            encoders['actor_onestep'] = encoder_module()

        # Define AC networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_critics'],
            encoder=encoders.get('critic'),
        )

        actor_onestep_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep'),
        )

        ac_network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep=(actor_onestep_def, (ex_observations, ex_actions)),
        )

        ac_networks = {k: v[0] for k, v in ac_network_info.items()}
        ac_network_args = {k: v[1] for k, v in ac_network_info.items()}

        ac_network_def = ModuleDict(ac_networks)
        ac_network_tx = optax.adam(learning_rate=config['lr'])
        ac_network_params = ac_network_def.init(init_rng_ac, **ac_network_args)['params']
        ac_network = TrainState.create(ac_network_def, ac_network_params, tx=ac_network_tx)

        ac_params = ac_network.params
        ac_params['modules_target_critic'] = ac_params['modules_critic']

        # Define BC network
        bc_flow_def = ActorVectorField(
            hidden_dims=config['bc_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['bc_layer_norm'],
            encoder=encoders.get('bc_flow'),
        )
        bc_network_info = dict(
            bc_flow=(bc_flow_def, (ex_observations, ex_actions, ex_times)),
        )

        if encoders.get('bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            bc_network_info['bc_flow_encoder'] = (encoders.get('bc_flow'), (ex_observations,))

        bc_networks = {k: v[0] for k, v in bc_network_info.items()}
        bc_network_args = {k: v[1] for k, v in bc_network_info.items()}

        bc_network_def = ModuleDict(bc_networks)
        bc_network_tx = optax.adam(learning_rate=config['lr_bc'])
        bc_network_params = bc_network_def.init(init_rng_bc, **bc_network_args)['params']
        bc_network = TrainState.create(bc_network_def, bc_network_params, tx=bc_network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, ac_network=ac_network, bc_network=bc_network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fac',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate for AC.
            lr_bc=3e-4, # LR for BC_flow_model
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            bc_hidden_dims=(512, 512, 512, 512),  # BC flow network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            bc_layer_norm=False,  # Whether to use layer normalization for the bc flow.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            distil_loss=True, # if the distillation loss is in the policy loss
            fac_alpha=1.0, # critic penalty coefficient
            fac_lambda=1.0,  # actor regularization coefficient
            fac_threshold='batch_adaptive', # use logp evaluations of the offline dataset; 'dataset_wide_constant' 'batch_wide_constant' 'batch_adaptive'
            logp_method='exact', # 'exact' or 'hutch-rade' or 'hutch-gaus'
            logp_hutch_probes=8,
            num_critics=2,
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values; 'min' or 'mean'
            weight_type='linear', # 'linear' or 'logarithmic' 'convex' 'concave'
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config


# ==============================================================================
# ORL-SMOOTH: experiment/CLI/YAML/metrics/restart plumbing only
# ==============================================================================

ALGORITHM_NAME = "FAC"
ALGORITHM_FULL_NAME = "Flow Actor-Critic (official FAC math, ORL-SMOOTH runner)"


@dataclass
class TrainConfig:
    # Experiment and schedule (matches ORL interface).
    device: str = "gpu"
    env: str = "cube-double-play-singletask-v0"
    dataset_name: Optional[str] = None
    seed: int = 0
    eval_seed: Optional[int] = 42
    max_timesteps: int = 1_000_000  # AC updates ONLY; BC proxy is extra.
    eval_freq: int = 100_000
    n_episodes: int = 50

    # Experiment-management I/O.
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    mode: str = "train"    # FAC official offline runner has no actor_refit.
    hyperparams_path: Optional[str] = "hyperparams/fac_jax.yml"
    use_hyperparams: bool = True

    # Data (original FAC values by default).
    batch_size: int = 256
    buffer_size: int = 2_000_000  # Reserved for online extensions, not used offline.
    normalize_states: bool = False
    normalize_reward: bool = False
    action_clip_eps: Optional[float] = 1e-5
    frame_stack: Optional[int] = None
    p_aug: Optional[float] = None

    # Network and optimizer, renamed to the user's conventions.
    lr: Optional[float] = None  # Convenience alias for actor_learning_rate.
    actor_learning_rate: float = 3e-4  # FAC's joint AC optimizer (actor+critic).
    critic_learning_rate: float = 3e-4  # Compatibility: NOT independently optimized.
    bc_learning_rate: float = 3e-4  # FAC's independently optimized flow proxy.
    hidden_dim: int = 512
    actor_n_hiddens: int = 4
    critic_n_hiddens: int = 4
    bc_n_hiddens: int = 4
    actor_hidden_dims: Optional[Tuple[int, ...]] = None
    value_hidden_dims: Optional[Tuple[int, ...]] = None
    bc_hidden_dims: Optional[Tuple[int, ...]] = None
    layer_norm: Optional[bool] = None      # FAC's value_layer_norm.
    actor_layer_norm: Optional[bool] = None
    bc_layer_norm: bool = False
    critic_ln: bool = True
    actor_ln: bool = False
    encoder: Optional[str] = None

    # ORIGINAL FAC algorithm choices. These are mapped directly (no change).
    discount: float = 0.995
    tau: float = 0.005
    num_critics: int = 2
    q_agg: str = "min"  # Bellman target: min or mean; actor Q uses mean in original.
    flow_steps: int = 10
    normalize_q_loss: bool = True
    distil_loss: bool = True
    actor_bc_coef: float = 1.0        # Original: fac_lambda (flow distillation).
    critic_penalty_coef: float = 1.0  # Original: fac_alpha (OOD Q penalty).
    fac_threshold: str = "batch_adaptive"
    logp_method: str = "exact"
    logp_hutch_probes: int = 8
    weight_type: str = "linear"

    # Original FAC proxy stage schedule and pretrained-policy cache.
    epoch_flow_proxy: int = 250
    bc_batch_size: Optional[int] = None  # None: original size/domain-dependent rule.
    pretrained_bc_dir: Optional[str] = "pretrained_bc"

    # ORL logging/checkpoint controls.
    run_tag: str = ""  # Optional suffix for user-controlled experiment isolation.
    project: str = "ORL-SMOOTH"
    group: str = "FAC-JAX"
    name: str = "FAC-JAX"
    log_wandb: bool = True
    log_every: int = 5_000
    checkpoint_freq: int = 25_000
    save_best_model: bool = True
    save_final_model: bool = True
    eval_at_first_step: bool = True
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.dataset_name is not None:
            self.env = self.dataset_name
        validate_config(self)


def cli_overridden_fields() -> set[str]:
    return {
        arg[2:].split("=", 1)[0].replace("-", "_")
        for arg in sys.argv[1:]
        if arg.startswith("--")
    }


def _tuple(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    raise ValueError(f"Hidden dimensions must be a sequence, got {value!r}")


def resolve_config(config: TrainConfig, cli_fields: set[str]) -> TrainConfig:
    """Map ORL names to the original FAC agent settings; resolve after YAML merge."""
    if config.dataset_name is not None:
        config.env = config.dataset_name
    # The YAML spells out 512-dim tuples. Honor high-level explicit CLI overrides.
    if ("hidden_dim" in cli_fields or "actor_n_hiddens" in cli_fields) and "actor_hidden_dims" not in cli_fields:
        config.actor_hidden_dims = None
    if ("hidden_dim" in cli_fields or "critic_n_hiddens" in cli_fields) and "value_hidden_dims" not in cli_fields:
        config.value_hidden_dims = None
    if ("hidden_dim" in cli_fields or "bc_n_hiddens" in cli_fields) and "bc_hidden_dims" not in cli_fields:
        config.bc_hidden_dims = None
    if "critic_ln" in cli_fields and "layer_norm" not in cli_fields:
        config.layer_norm = None
    if "actor_ln" in cli_fields and "actor_layer_norm" not in cli_fields:
        config.actor_layer_norm = None
    config.actor_hidden_dims = _tuple(config.actor_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.actor_n_hiddens)
    config.value_hidden_dims = _tuple(config.value_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.critic_n_hiddens)
    config.bc_hidden_dims = _tuple(config.bc_hidden_dims) or (
        int(config.hidden_dim),
    ) * int(config.bc_n_hiddens)
    if config.layer_norm is None:
        config.layer_norm = bool(config.critic_ln)
    if config.actor_layer_norm is None:
        config.actor_layer_norm = bool(config.actor_ln)
    if config.lr is not None:
        # Avoid accidentally overriding an explicit actor_learning_rate in CLI.
        if "actor_learning_rate" not in cli_fields:
            config.actor_learning_rate = float(config.lr)
    else:
        config.lr = float(config.actor_learning_rate)
    refresh_algorithm_names(config)
    validate_config(config)
    if config.critic_learning_rate != config.actor_learning_rate:
        print(
            "Warning: official FAC has ONE joint AC optimizer. "
            f"critic_learning_rate={config.critic_learning_rate} is ignored; "
            f"actor_learning_rate={config.actor_learning_rate} is used for both."
        )
    return config


def refresh_algorithm_names(config: TrainConfig) -> None:
    config.project = "ORL-SMOOTH"
    config.group = "FAC-JAX"
    config.name = f"FAC-JAX-{config.env}"
    if config.run_tag:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.run_tag):
            raise ValueError("run_tag may only contain letters, digits, _ . -")
        config.name += f"-{config.run_tag}"


def validate_config(config: TrainConfig) -> None:
    if config.mode != "train":
        raise NotImplementedError("Official FAC ORL wrapper supports mode='train' only.")
    for name in ("max_timesteps", "eval_freq", "n_episodes", "log_every", "checkpoint_freq", "epoch_flow_proxy"):
        if int(getattr(config, name)) < (0 if name in ("max_timesteps", "epoch_flow_proxy") else 1):
            raise ValueError(f"{name} has invalid value {getattr(config, name)!r}")
    if config.batch_size <= 0 or config.hidden_dim <= 0:
        raise ValueError("batch_size and hidden_dim must be positive")
    if config.q_agg not in ("min", "mean"):
        raise ValueError("q_agg must be 'min' or 'mean'")
    if config.num_critics < 1 or config.flow_steps < 1:
        raise ValueError("num_critics/flow_steps must be positive")
    if config.fac_threshold not in ("batch_adaptive", "batch_wide_constant", "dataset_wide_constant"):
        raise ValueError(f"Unsupported FAC threshold: {config.fac_threshold}")
    if config.logp_method not in ("exact", "hutch-rade", "hutch-gaus"):
        raise ValueError(f"Unsupported official FAC logp method: {config.logp_method}")
    if config.weight_type not in ("linear", "logarithmic", "convex", "concave"):
        raise ValueError(f"Unsupported official FAC weight: {config.weight_type}")
    if config.logp_hutch_probes < 1:
        raise ValueError("logp_hutch_probes must be positive")
    if not 0 <= config.discount <= 1 or not 0 <= config.tau <= 1:
        raise ValueError("discount and tau must lie in [0,1]")
    if config.actor_bc_coef < 0 or config.critic_penalty_coef < 0:
        raise ValueError("FAC regularization coefficients must be >= 0")
    if config.normalize_states or config.normalize_reward:
        # No silent preprocessing changes allowed under the exact-FAC contract.
        raise ValueError("Original FAC does not normalize states/rewards; both must be False")
    if config.bc_batch_size is not None and config.bc_batch_size < 1:
        raise ValueError("bc_batch_size must be positive")


def apply_env_hyperparams(config: TrainConfig) -> TrainConfig:
    cli_fields = cli_overridden_fields()
    if not config.use_hyperparams or not config.hyperparams_path:
        return resolve_config(config, cli_fields)
    path = Path(config.hyperparams_path)
    if not path.is_file():
        raise FileNotFoundError(f"FAC hyperparameter YAML was not found: {path}")
    with path.open(encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}
    params = all_params.get(config.env)
    if params is None:
        print(f"No FAC YAML entry for {config.env}; using dataclass/CLI settings")
        return resolve_config(config, cli_fields)
    aliases = {
        "fac_alpha": "critic_penalty_coef",
        "fac_lambda": "actor_bc_coef",
        "lr_bc": "bc_learning_rate",
        "value_layer_norm": "layer_norm",
        "n_timesteps": "max_timesteps",
        "offline_steps": "max_timesteps",
        "eval_interval": "eval_freq",
        "eval_episodes": "n_episodes",
        "log_interval": "log_every",
        "env_name": "env",
        "learning_rate": "lr",
    }
    fields = set(TrainConfig.__dataclass_fields__)
    applied, skipped = [], []
    for raw, value in params.items():
        field = aliases.get(raw, raw)
        if field not in fields:
            print(f"Warning: unrecognized FAC YAML parameter {raw!r}, ignoring")
            continue
        if field in cli_fields or raw in cli_fields:
            skipped.append(field)
            continue
        if field in ("actor_hidden_dims", "value_hidden_dims", "bc_hidden_dims"):
            value = _tuple(value)
        if isinstance(value, float) and value.is_integer() and field in (
            "max_timesteps", "eval_freq", "n_episodes", "log_every", "checkpoint_freq", "epoch_flow_proxy", "batch_size"
        ):
            value = int(value)
        setattr(config, field, value)
        applied.append(field)
    if applied:
        print(f"Applied FAC YAML ({config.env}): {', '.join(applied)}")
    if skipped:
        print(f"Keeping explicit FAC CLI overrides: {', '.join(skipped)}")
    return resolve_config(config, cli_fields)


def agent_config(config: TrainConfig) -> dict:
    """EXACT mapping to FAC-main/agents/fac.py:get_config, no loss modifications."""
    return dict(
        agent_name="fac",
        lr=float(config.actor_learning_rate),
        lr_bc=float(config.bc_learning_rate),
        batch_size=int(config.batch_size),
        actor_hidden_dims=tuple(config.actor_hidden_dims),
        value_hidden_dims=tuple(config.value_hidden_dims),
        bc_hidden_dims=tuple(config.bc_hidden_dims),
        value_layer_norm=bool(config.layer_norm),
        actor_layer_norm=bool(config.actor_layer_norm),
        bc_layer_norm=bool(config.bc_layer_norm),
        flow_steps=int(config.flow_steps),
        normalize_q_loss=bool(config.normalize_q_loss),
        distil_loss=bool(config.distil_loss),
        fac_alpha=float(config.critic_penalty_coef),
        fac_lambda=float(config.actor_bc_coef),
        fac_threshold=str(config.fac_threshold),
        logp_method=str(config.logp_method),
        logp_hutch_probes=int(config.logp_hutch_probes),
        num_critics=int(config.num_critics),
        discount=float(config.discount),
        tau=float(config.tau),
        q_agg=str(config.q_agg),
        weight_type=str(config.weight_type),
        encoder=config.encoder,
    )


def finalize_checkpoint_path(config: TrainConfig) -> TrainConfig:
    if config.checkpoints_path is not None:
        config.checkpoints_path = str(Path(config.checkpoints_path) / config.name / str(config.seed))
    return config


def select_jax_device(name: str):
    try:
        dev = jax.devices("gpu" if name.lower() == "cuda" else name.lower())[0]
    except Exception:
        print(f"Requested JAX device {name!r} unavailable; falling back")
        dev = jax.devices()[0]
    print(f"Using JAX device: {dev}")
    return dev


def set_seed(seed: int, env=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
        for space in (env.action_space, env.observation_space):
            if hasattr(space, "seed"):
                space.seed(seed)


def _save_pickle(path: Union[str, Path], obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def _load_pickle(path: Union[str, Path]):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_logs_npz(logs: List[dict], save_dir: Optional[str]):
    if not save_dir or not logs:
        return
    keys = sorted({key for row in logs for key in row})
    data = {}
    for key in keys:
        values = [row.get(key, np.nan) for row in logs]
        try:
            data[key] = np.asarray(values, dtype=np.float64 if key != "timestep" else np.int64)
        except (TypeError, ValueError):
            data[key] = np.asarray(values, dtype=object)
    np.savez(Path(save_dir) / "eval_logs.npz", **data)


def resolve_checkpoint_path(load_model, run_name=None, seed=None) -> Path:
    root = Path(load_model)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    for candidate_root in (root, root / str(run_name) / str(seed)):
        for filename in ("checkpoint.pkl", "best_checkpoint.pkl"):
            candidate = candidate_root / filename
            if candidate.is_file():
                return candidate
    files = sorted(set(root.glob("*/checkpoint.pkl")) | set(root.glob("*/*/checkpoint.pkl")))
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"No checkpoint.pkl found under {root}")
    raise ValueError(f"Multiple checkpoints under {root}; specify exact file:\n" + "\n".join(map(str, files)))


def _agent_payload(agent, config: TrainConfig, ac_step: int, phase: str) -> dict:
    return {
        "agent": serialization.to_state_dict(agent),
        "config": asdict(config),
        "_training_timestep": int(ac_step),
        "_fac_phase": str(phase),
        "_numpy_rng_state": np.random.get_state(),
        "_python_rng_state": random.getstate(),
    }


def _restore_agent(agent, payload):
    state = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    return serialization.from_state_dict(agent, state)


def _proxy_batch_size(config: TrainConfig, dataset_size: int) -> int:
    if config.bc_batch_size is not None:
        return int(config.bc_batch_size)
    if "visual" in config.env:
        return int(config.batch_size)
    # Exact FAC-main/main.py dataset-size rule.
    factor = 1 if dataset_size < 100_000 else 4 if dataset_size < 500_000 else 16
    return int(config.batch_size) * factor


def _eval_metrics(agent, env, episodes: int, eval_seed: Optional[int]) -> dict:
    """Original action sampling/reset semantics; adds per-episode statistics."""
    from collections import defaultdict

    # The official FAC evaluator draws a fresh seed from NumPy per evaluation.
    # An explicit eval_seed reproduces the user's FQL-JAX deterministic RNG API.
    rng_seed = (
        int(np.random.randint(0, 2 ** 32)) if eval_seed is None else int(eval_seed)
    )
    key = jax.random.PRNGKey(rng_seed)
    info_values = defaultdict(list)
    rewards, successes, normalized = [], [], []
    for _ in trange(episodes, desc="FAC Eval", leave=False):
        obs, info = env.reset()
        total_return, done = 0.0, False
        last_success = np.nan
        while not done:
            key, action_key = jax.random.split(key)
            action = np.asarray(agent.sample_actions(observations=obs, seed=action_key, temperature=0))
            action = np.clip(action, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_return += float(reward)
            if "success" in info:
                last_success = float(np.asarray(info["success"]).reshape(-1)[0])
        flat = flatten(info)
        for k, v in flat.items():
            if isinstance(v, (int, float, bool, np.number)):
                info_values[k].append(float(v))
        rewards.append(float(flat.get("episode.return", total_return)))
        successes.append(float(flat.get("success", flat.get("episode.success", last_success))))
        normalized.append(float(flat.get("episode.normalized_return", np.nan)))

    def avg_std(values):
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        return (float(x.mean()), float(x.std())) if x.size else (float("nan"), float("nan"))

    reward_mean, reward_std = avg_std(rewards)
    success_mean, success_std = avg_std(successes)
    norm_mean, norm_std = avg_std(normalized)
    metrics = {
        "eval/reward_mean": reward_mean,
        "eval/reward_std": reward_std,
        "eval/success_rate": success_mean,
        "eval/success_std": success_std,
        "eval/d4rl_normalized_score_mean": norm_mean,
        "eval/d4rl_normalized_score_std": norm_std,
    }
    metrics.update({f"eval/{k}": float(np.mean(v)) for k, v in info_values.items() if v})
    return metrics


def _save_density_cache(path: Path, train_ds, val_ds, config: TrainConfig):
    """Persist Hutchinson-generated densities: AC resumes must not resample them."""
    metadata = {
        "env": config.env,
        "seed": int(config.seed),
        "logp_method": config.logp_method,
        "flow_steps": int(config.flow_steps),
        "logp_hutch_probes": int(config.logp_hutch_probes),
        "train_count": int(train_ds.size),
        "val_count": int(val_ds.size) if val_ds is not None else 0,
    }
    payload = {"metadata": np.asarray(yaml.safe_dump(metadata)),
               "train_logp": train_ds["estimated_logp"][:train_ds.size],
               "train_logp_min": np.asarray(train_ds.estimated_logp_min, dtype=np.float32)}
    if val_ds is not None:
        payload["val_logp"] = val_ds["estimated_logp"][:val_ds.size]
        payload["val_logp_min"] = np.asarray(val_ds.estimated_logp_min, dtype=np.float32)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def _load_density_cache(path: Path, train_ds, val_ds, config: TrainConfig) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as z:
        metadata = yaml.safe_load(str(z["metadata"].item()))
        expected = dict(
            env=config.env,
            seed=int(config.seed),
            logp_method=config.logp_method,
            flow_steps=int(config.flow_steps),
            logp_hutch_probes=int(config.logp_hutch_probes),
            train_count=int(train_ds.size),
            val_count=int(val_ds.size) if val_ds is not None else 0,
        )
        if metadata != expected:
            print("FAC density-cache metadata differs; recomputing densities")
            return False
        def attach(ds, values, minimum):
            # Original FAC dataset uses a mutable internal dictionary.
            n = getattr(ds, "max_size", ds.size)
            data = np.full((n,), np.nan, dtype=np.float32)
            data[:ds.size] = values.astype(np.float32)
            ds._dict["estimated_logp"] = data
            ds._dict["estimated_logp_min"] = np.full((n,), minimum, dtype=np.float32)
            ds.estimated_logp_min = float(minimum)
        attach(train_ds, z["train_logp"], float(z["train_logp_min"].item()))
        if val_ds is not None:
            attach(val_ds, z["val_logp"], float(z["val_logp_min"].item()))
        return True


def _prepare_density_cache(agent, train_ds, val_ds, config: TrainConfig, run_dir: Optional[Path], bc_batch: int):
    cache = run_dir / "fac_density_cache.npz" if run_dir is not None else None
    if cache is not None and _load_density_cache(cache, train_ds, val_ds, config):
        print(f"Loaded FAC behavior log-densities: {cache}")
        return
    print(f"Computing original FAC {config.logp_method} flow densities")
    train_ds.compute_and_attach_estimated_logp(
        agent=agent, method=config.logp_method, batch_size=bc_batch
    )
    if val_ds is not None:
        val_ds.compute_and_attach_estimated_logp(
            agent=agent, method=config.logp_method, batch_size=bc_batch
        )
    if cache is not None:
        _save_density_cache(cache, train_ds, val_ds, config)
        print(f"Saved FAC behavior log-densities: {cache}")


def _train_impl(config: TrainConfig):
    config = finalize_checkpoint_path(apply_env_hyperparams(config))
    run_dir = Path(config.checkpoints_path) if config.checkpoints_path else None

    checkpoint_manager, preparation = None, None
    if run_dir is not None:
        checkpoint_manager = TrainingCheckpointManager(
            run_dir=str(run_dir),
            current_config=asdict(config),
            default_config=asdict(TrainConfig()),
            max_timesteps=int(config.max_timesteps),
            checkpoint_type="fac_jax_training_progress",
            identity_ignored_fields=DEFAULT_IDENTITY_IGNORED_FIELDS,
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=bool(config.log_wandb),
            wandb_entity=config.wandb_entity,
            wandb_project=config.project,
            final_checkpoint_name="checkpoint.pkl",
        )
        preparation = checkpoint_manager.prepare()
        print(preparation.message)
        if preparation.is_completed:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        saved_config = run_dir / "config.yaml"
        if not saved_config.exists():
            with saved_config.open("w", encoding="utf-8") as f:
                yaml.safe_dump(asdict(config), f, allow_unicode=True, sort_keys=True)

    dev = select_jax_device(config.device)
    # Original FAC loaders/wrappers: prevents any silent reward/observation change.
    _, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        config.env,
        frame_stack=config.frame_stack,
        action_clip_eps=config.action_clip_eps,
    )
    train_dataset = Dataset.create(**dict(train_dataset))
    # Match FAC-main/main.py exactly: use a ReplayBuffer copy in offline training.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(config.buffer_size, train_dataset.size + 1)
    )
    for ds in (train_dataset, val_dataset):
        if ds is not None:
            ds.p_aug = config.p_aug
            ds.frame_stack = config.frame_stack
            ds.return_next_actions = False
    set_seed(config.seed, eval_env)

    print(f"Training {ALGORITHM_NAME}-JAX, Env: {config.env}, Seed: {config.seed}")
    example = train_dataset.sample(1)
    agent = FACAgent.create(
        config.seed, example["observations"], example["actions"], agent_config(config)
    )
    agent = jax.device_put(agent, dev)
    ac_step = 0
    eval_logs: List[dict] = []

    def progress_state():
        return _agent_payload(agent, config, ac_step, phase="ac")

    def load_progress(payload):
        nonlocal agent, ac_step
        agent = jax.device_put(_restore_agent(agent, payload), dev)
        ac_step = int(payload.get("_training_timestep", 0))
        if "_numpy_rng_state" in payload:
            np.random.set_state(payload["_numpy_rng_state"])
        if "_python_rng_state" in payload:
            random.setstate(payload["_python_rng_state"])

    def current_step():
        return int(ac_step)

    # An explicit --load_model is a full agent restore, NOT a from-scratch refit.
    previous_file = None
    if config.load_model:
        previous_file = resolve_checkpoint_path(config.load_model, config.name, config.seed)
        print(f"Loading FAC agent from {previous_file}")
        load_progress(_load_pickle(previous_file))
        # Preserve original Hutchinson estimates across manual checkpoint loads.
        if run_dir is not None:
            source_cache = previous_file.parent / "fac_density_cache.npz"
            target_cache = run_dir / "fac_density_cache.npz"
            if source_cache.is_file() and not target_cache.is_file():
                import shutil
                shutil.copy2(source_cache, target_cache)

    bc_stage_exists = bool(run_dir is not None and (run_dir / "fac_bc_progress.pkl").is_file())
    if checkpoint_manager is not None and preparation.is_resuming and not bc_stage_exists:
        ac_step, eval_logs, _ = checkpoint_manager.restore(
            load_trainer_state=load_progress,
            get_restored_timestep=current_step,
        )
        print(f"Restored FAC AC training at timestep {ac_step}")
    elif previous_file is not None:
        # Loading a full checkpoint must NOT retrain its already learned BC proxy.
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)
        if checkpoint_manager is not None:
            checkpoint_manager.save_progress(
                timestep=ac_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running"
            )
    else:
        # Only execute the official multi-stage initialization for fresh runs.
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        bc_iters_per_epoch = (train_dataset.size + bc_batch - 1) // bc_batch
        bc_total_iters = int(config.epoch_flow_proxy) * bc_iters_per_epoch
        bc_step = 0
        stage_path = run_dir / "fac_bc_progress.pkl" if run_dir else None
        cache_root = Path(config.pretrained_bc_dir) if config.pretrained_bc_dir else None
        cache_folder = cache_root / f"{config.env}_seed_{config.seed}" if cache_root else None

        should_train_bc = True
        if stage_path and stage_path.is_file():
            stage = _load_pickle(stage_path)
            if (stage.get("bc_total_iters"), stage.get("bc_batch_size")) != (bc_total_iters, bc_batch):
                raise ValueError("FAC BC resume config mismatch: batch size or proxy-epoch count changed")
            agent = jax.device_put(_restore_agent(agent, stage), dev)
            bc_step = int(stage["bc_iteration"])
            np.random.set_state(stage["_numpy_rng_state"])
            random.setstate(stage["_python_rng_state"])
            print(f"Restoring original FAC BC flow from iteration {bc_step}/{bc_total_iters}")
        elif cache_folder and cache_folder.is_dir():
            print(f"Reusing official pretrained FAC BC flow: {cache_folder}")
            agent = restore_only_bcmodel(agent, str(cache_root), config.env, str(config.seed))
            should_train_bc = False

        if should_train_bc:
            print(f"Training original FAC BC proxy: {bc_total_iters} updates, batch_size={bc_batch}")
            def save_bc_stage():
                if stage_path is None:
                    return
                state = _agent_payload(agent, config, ac_step=0, phase="bc")
                state.update(bc_iteration=bc_step,
                             bc_batch_size=bc_batch, bc_total_iters=bc_total_iters)
                _save_pickle(stage_path, state)
            try:
                for iteration in trange(bc_step + 1, bc_total_iters + 1, desc="FAC Train BC"):
                    batch = train_dataset.sample(bc_batch)
                    agent, bc_log = agent.update(batch, mode="train_bc")
                    bc_step = iteration
                    if stage_path and iteration % config.checkpoint_freq == 0:
                        save_bc_stage()
            except BaseException:
                save_bc_stage()
                raise
            save_bc_stage()  # BC is complete; safe to retry density precomputation after interruption.
            if cache_root is not None:
                save_only_bcmodel(agent, str(cache_root), config.env, str(config.seed))

        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)
        if stage_path and stage_path.exists():
            stage_path.unlink()
        if checkpoint_manager is not None:
            checkpoint_manager.save_progress(
                timestep=ac_step, trainer_state=progress_state(),
                eval_logs=eval_logs, status="running"
            )

    # If recovering an AC progress checkpoint, log-densities need to be restored
    # separately: they are computed ONCE in official FAC, not continually refreshed.
    if checkpoint_manager is not None and preparation.is_resuming and not bc_stage_exists:
        bc_batch = _proxy_batch_size(config, train_dataset.size)
        _prepare_density_cache(agent, train_dataset, val_dataset, config, run_dir, bc_batch)

    # ORL W&B lifecycle: resume-aware when using shared checkpoint manager.
    if config.log_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: W&B unavailable; continuing local FAC training")
            config.log_wandb = False

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
                import wandb
                run = wandb.init(config=asdict(config), project=config.project,
                                 group=config.group, name=config.name, id=str(uuid.uuid4()))
                if run is not None and hasattr(run, "log_code"):
                    run.log_code(str(_PROJECT_ROOT))
            except Exception as exc:
                print(f"Warning: W&B unavailable: {exc}; local-only training")
                config.log_wandb = False

    def wandb_log(metrics, step):
        if not config.log_wandb:
            return
        if checkpoint_manager is not None:
            checkpoint_manager.log_wandb(metrics, int(step))
        else:
            import wandb
            wandb.log(metrics, step=int(step))

    best_value = best_eval_metric(eval_logs)
    last_update_info: dict = {}
    start_time = last_log_time = time.time()

    def eval_is_due(step):
        return ((config.eval_at_first_step and step == 1)
                or step % config.eval_freq == 0
                or step == config.max_timesteps)

    try:
        for step in trange(ac_step + 1, int(config.max_timesteps) + 1, desc="FAC Train AC"):
            batch = train_dataset.sample(config.batch_size)
            agent, info = agent.update(batch, mode="train_ac", use_cp=True, online_finetuning=False)
            ac_step = step
            last_update_info = {k: float(jax.device_get(v)) for k, v in info.items()}

            if step % config.log_every == 0:
                log = {f"training/{k}": value for k, value in last_update_info.items()}
                if val_dataset is not None:
                    val = val_dataset.sample(config.batch_size)
                    _, validation_info = agent.total_loss(val, grad_params=None, mode="train_ac")
                    log.update({f"validation/{k}": float(jax.device_get(v)) for k, v in validation_info.items()})
                log["time/epoch_time"] = (time.time() - last_log_time) / config.log_every
                log["time/total_time"] = time.time() - start_time
                last_log_time = time.time()
                wandb_log(log, step)

            if eval_is_due(step):
                eval_log = {"timestep": int(step)}
                eval_log.update(_eval_metrics(agent, eval_env, config.n_episodes, config.eval_seed))
                upsert_eval_log(eval_logs, eval_log)
                print(
                    f"[{step}] reward={eval_log['eval/reward_mean']:.3f} "
                    f"± {eval_log['eval/reward_std']:.3f}; "
                    f"success={eval_log['eval/success_rate']:.3f} "
                    f"± {eval_log['eval/success_std']:.3f}"
                )
                _save_logs_npz(eval_logs, config.checkpoints_path)
                wandb_log({k: v for k, v in eval_log.items() if np.isscalar(v)}, step)
                metric = eval_log["eval/success_rate"]
                if not np.isfinite(metric):
                    metric = eval_log["eval/d4rl_normalized_score_mean"]
                if not np.isfinite(metric):
                    metric = eval_log["eval/reward_mean"]
                if (run_dir is not None and config.save_best_model
                        and np.isfinite(metric) and metric > best_value):
                    best_value = metric
                    _save_pickle(run_dir / "best_checkpoint.pkl", progress_state())

            if checkpoint_manager is not None and step % config.checkpoint_freq == 0:
                checkpoint_manager.save_progress(
                    timestep=step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="running"
                )
    except BaseException:
        if checkpoint_manager is not None:
            if not eval_is_due(ac_step) or find_eval_log(eval_logs, ac_step) is not None:
                checkpoint_manager.save_progress(
                    timestep=ac_step, trainer_state=progress_state(),
                    eval_logs=eval_logs, status="interrupted"
                )
                print(f"Saved FAC interrupted checkpoint at {ac_step}")
            else:
                print("FAC evaluation incomplete; retaining last safe progress checkpoint")
        raise

    if checkpoint_manager is not None:
        final_path = checkpoint_manager.complete(
            timestep=ac_step, final_state=progress_state(),
            save_final_model=bool(config.save_final_model), eval_logs=eval_logs
        )
        print(f"FAC complete: {final_path}")


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
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=exit_code)
        except ImportError:
            pass


if __name__ == "__main__":
    train()
