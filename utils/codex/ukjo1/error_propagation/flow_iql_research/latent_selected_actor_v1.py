"""Portable latent selector plus one-step behavior decoder.

Two MLP blocks form one feed-forward stochastic actor. Only one latent and
one action are realized. No critic, candidate action bank or flow integration
is present. This architecture is exploratory, not a validated final recipe.
Latent policies and mixture density networks are prior art (PLAS2020, MDN1994).
"""
import math
from dataclasses import dataclass
from pathlib import Path
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtr
import numpy as np

from utils.flow_iql_research.mixture_onestep_actor_v1 import MixtureActor
from utils.flow_iql_research.standalone_onestep_actor_v1 import _Actor


def latent_log_probability(parameters, values):
    """Ordinary Lebesgue Gaussian-mixture density; latents are NOT clipped."""
    logits,means,scales=(parameters[k]for k in ('logits','means','scales'))
    standardized=(values[...,None,:]-means)/scales
    components=jnp.sum(-.5*standardized**2-jnp.log(scales)
                       -.5*math.log(2*math.pi),axis=-1)
    return logsumexp(jax.nn.log_softmax(logits,axis=-1)+components,axis=-1)


def sample_latent(parameters, noise):
    """Select mixture PARAMETERS before drawing one unbounded latent vector."""
    logits,means,scales=(parameters[k]for k in ('logits','means','scales'))
    assert noise.shape[-1]==means.shape[-1]+1
    uniform=ndtr(noise[...,0])
    cdf=jnp.cumsum(jax.nn.softmax(logits,axis=-1),axis=-1)
    index=jnp.minimum((uniform[...,None]>cdf).sum(-1),logits.shape[-1]-1)
    indices=jnp.broadcast_to(index[...,None,None],(*index.shape,1,means.shape[-1]))
    location=jnp.take_along_axis(means,indices,axis=-2)[...,0,:]
    scale=jnp.take_along_axis(scales,indices,axis=-2)[...,0,:]
    return location+scale*noise[...,1:]


class LatentSelectedActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    components: int=9
    standard_deviation_floor: float=.1

    def setup(self):
        self.router=MixtureActor(self.hidden_dims,self.action_dim,self.components,
                                 self.standard_deviation_floor)
        self.decoder=_Actor(self.hidden_dims,self.action_dim,False)

    def __call__(self,observations,noise):
        p=self.router(observations,noise,return_parameters=True)
        z=sample_latent(p,noise)
        return jnp.clip(self.decoder(observations,z),-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['MLP_blocks']==2 and metadata['generated_candidate_actions']==0
    assert metadata['flow_integration_steps']==0 and not metadata['requires_critic']
    assert not metadata['requires_BC_flow']
    actor=LatentSelectedActor(tuple(metadata['hidden_dims']),metadata['action_dim'],
        metadata['components'],metadata['standard_deviation_floor'])

    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        noise=jax.random.normal(action_key,(*observations.shape[:-1],metadata['action_dim']+1))
        return actor.apply({'params':params},observations,noise)

    return jax.jit(policy,backend=backend)


@dataclass
class LoadedActor:
    params: object
    metadata: dict
    policy: object
    device: object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),
                           jax.device_put(key,self.device))


def load_actor(path,backend='cpu'):
    with Path(path).open('rb')as f:artifact=pickle.load(f)
    assert artifact['schema']=='private_selected_latent_actor_v1'
    assert set(artifact['actor_params'])=={'router','decoder'}
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(artifact['actor_params']))
    device=jax.devices(backend)[0]
    return LoadedActor(jax.device_put(artifact['actor_params'],device),artifact['metadata'],
                       make_policy(artifact['metadata'],backend),device)
