"""Portable one-step inference: NumPy/JAX/Flax only; no learner or OGBench.

Usage:
    actor = load_actor('actor_1000000.pkl', backend='cpu')
    action = actor(observation, jax.random.PRNGKey(42))

Pass a fresh key for each action. The artifact contains only actor weights and
metadata; neither a training checkpoint nor critic/flow parameters are loaded.
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
    layer_norm: bool

    @nn.compact
    def __call__(self,x):
        for index,width in enumerate(self.dimensions):
            x=nn.Dense(width)(x)
            if index<len(self.dimensions)-1:
                x=nn.gelu(x)
                if self.layer_norm:x=nn.LayerNorm()(x)
        return x


class _Actor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    layer_norm: bool

    def setup(self):self.mlp=_MLP((*self.hidden_dims,self.action_dim),self.layer_norm)

    def __call__(self,observations,latents):return self.mlp(jnp.concatenate([observations,latents],axis=-1))


def make_policy(metadata,backend='cpu'):
    assert metadata['schema']=='single_actor_deployment_v1'
    assert metadata['actor_forward_passes']==1 and metadata['flow_integration_steps']==0
    assert metadata['generated_candidate_actions']==0 and not metadata['requires_critic'] and not metadata['requires_BC_flow']
    module=_Actor(tuple(metadata['hidden_dims']),metadata['action_dim'],metadata['layer_norm'])
    def policy(params,observations,key):
        count=9 if metadata['input']=='bank'else 1
        action_key,_=jax.random.split(key)
        noise=jax.random.normal(action_key,(count,*observations.shape[:-1],metadata['action_dim']))
        if metadata['input']=='bank':latents=jnp.moveaxis(noise,0,-2).reshape((*observations.shape[:-1],9*metadata['action_dim']))
        elif metadata['input']=='state':latents=jnp.zeros_like(noise[0])
        else:
            assert metadata['input']=='single'
            latents=noise[0]
        return jnp.clip(module.apply({'params':params},observations,latents),-1,1)
    return jax.jit(policy,backend=backend)


@dataclass
class LoadedActor:
    params: object
    metadata: dict
    policy: object
    device: object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),jax.device_put(key,self.device))


def load_actor(path,backend='cpu'):
    with Path(path).open('rb')as stream:artifact=pickle.load(stream)
    assert artifact['schema']=='single_actor_deployment_v1'
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(artifact['actor_params']))
    device=jax.devices(backend)[0];metadata=artifact['metadata']
    return LoadedActor(jax.device_put(artifact['actor_params'],device),metadata,make_policy(metadata,backend),device)
