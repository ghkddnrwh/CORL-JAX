"""Set-actor inference using only NumPy/JAX/Flax and an actor-only artifact.

One integrated network maps state and noise to one action. Latent features are
pooled internally; neither candidate actions nor critic/flow weights are used.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class _MLP(nn.Module):
    dimensions: tuple
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for index, width in enumerate(self.dimensions):
            x = nn.Dense(width)(x)
            if index < len(self.dimensions)-1 or self.activate_final:
                x = nn.gelu(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class _SetActor(nn.Module):
    action_dim: int
    feature_width: int
    decoder_width: int
    layer_norm: bool = False

    def setup(self):
        self.features = _MLP((self.feature_width, self.feature_width),
                             activate_final=True, layer_norm=self.layer_norm)
        self.attention = nn.Dense(1)
        self.decoder = _MLP((self.decoder_width, self.decoder_width, self.action_dim),
                            layer_norm=self.layer_norm)

    def __call__(self, observations, packed_noises):
        noise = packed_noises.reshape((*observations.shape[:-1], 9, self.action_dim))
        obs = jnp.broadcast_to(observations[..., None, :],
                               (*noise.shape[:-1], observations.shape[-1]))
        features = self.features(jnp.concatenate([obs, noise], axis=-1))
        weights = jax.nn.softmax(self.attention(features)[..., 0], axis=-1)
        pooled = (features*weights[..., None]).sum(axis=-2)
        maximum = features.max(axis=-2)
        return self.decoder(jnp.concatenate([observations, pooled, maximum], axis=-1))


def make_policy(metadata, backend='cpu'):
    assert metadata['schema'] == 'set_actor_deployment_v1'
    assert metadata['actor_forward_passes'] == 1 and metadata['flow_integration_steps'] == 0
    assert metadata['generated_candidate_actions'] == 0
    assert not metadata['requires_critic'] and not metadata['requires_BC_flow']
    assert metadata['input'] == 'bank' and metadata['latent_vectors'] == 9
    definition = _SetActor(metadata['action_dim'], metadata['feature_width'],
                           metadata['decoder_width'], metadata['layer_norm'])
    def policy(params, observations, key):
        action_key, _ = jax.random.split(key)
        noise = jax.random.normal(action_key, (9, *observations.shape[:-1], metadata['action_dim']))
        packed = jnp.moveaxis(noise, 0, -2).reshape((*observations.shape[:-1], 9*metadata['action_dim']))
        return jnp.clip(definition.apply({'params': params}, observations, packed), -1, 1)
    return jax.jit(policy, backend=backend)


@dataclass
class LoadedActor:
    params: object
    metadata: dict
    policy: object
    device: object

    def __call__(self, observation, key):
        return self.policy(self.params, jax.device_put(np.asarray(observation, np.float32), self.device),
                           jax.device_put(key, self.device))


def load_actor(path, backend='cpu'):
    with Path(path).open('rb') as stream:
        artifact = pickle.load(stream)
    assert artifact['schema'] == 'set_actor_deployment_v1'
    assert not {'critic', 'base', 'state'} & artifact.keys()
    assert all(np.isfinite(np.asarray(v)).all()
               for v in jax.tree_util.tree_leaves(artifact['actor_params']))
    device = jax.devices(backend)[0]
    metadata = artifact['metadata']
    return LoadedActor(jax.device_put(artifact['actor_params'], device), metadata,
                       make_policy(metadata, backend), device)
