"""Portable one-forward mixture with optional explicit state normalization.

The network emits distribution parameters, then one component is selected
and only one Gaussian action is generated. No critic, flow ODE or candidate
action bank is loaded. Private diagnostic artifacts require an explicit flag.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np

from utils.flow_iql_research.mixture_onestep_actor_v1 import MixtureActor


def definition(metadata):
    return MixtureActor(tuple(metadata['hidden_dims']),metadata['action_dim'],metadata['components'],metadata['std_floor'])


def raw_action(parameters,latents):
    """Gather distribution parameters first, then draw one unbounded action."""
    means,scales=parameters['means'],parameters['scales']
    cdf=jnp.cumsum(jax.nn.softmax(parameters['logits'],axis=-1),axis=-1)
    index=jnp.minimum(jnp.sum(ndtr(latents[...,0,None])>cdf,axis=-1),cdf.shape[-1]-1)
    idx=jnp.broadcast_to(index[...,None,None],(*index.shape,1,means.shape[-1]))
    location=jnp.take_along_axis(means,idx,axis=-2)[...,0,:]
    scale=jnp.take_along_axis(scales,idx,axis=-2)[...,0,:]
    return location+scale*latents[...,1:]


def make_policy(metadata,backend='cpu'):
    assert metadata['actor_forward_passes']==1 and metadata['flow_integration_steps']==0
    assert metadata['generated_candidate_actions']==0 and not metadata['requires_critic'] and not metadata['requires_BC_flow']
    assert metadata['noise_dim']==metadata['action_dim']+1
    module=definition(metadata)
    normalization=metadata.get('observation_normalization')
    if normalization is not None:
        mean=np.asarray(normalization['mean'],np.float32);scale=np.asarray(normalization['scale'],np.float32)
        assert mean.shape==scale.shape==(metadata['observation_dim'],)
        assert np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all()
        mean,scale=jnp.asarray(mean),jnp.asarray(scale)
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        latent=jax.random.normal(action_key,(*observations.shape[:-1],metadata['noise_dim']))
        if normalization is not None:observations=(observations-mean)/scale
        p=module.apply({'params':params},observations,latent,return_parameters=True)
        return jnp.clip(raw_action(p,latent),-1,1)
    return jax.jit(policy,backend=backend)


@dataclass
class LoadedActor:
    params: object
    metadata: dict
    policy: object
    device: object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),jax.device_put(key,self.device))


def load_actor(path,backend='cpu',allow_diagnostic=False):
    with Path(path).open('rb') as stream:artifact=pickle.load(stream)
    valid=artifact['schema']=='single_actor_deployment_v1'
    if allow_diagnostic:valid=valid or artifact['schema'] in ('private_frozen_teacher_student_v1','private_frozen_mixture_refinement_v1')
    assert valid,'A private frozen-teacher fixture is not a validated joint policy.'
    metadata=artifact['metadata'];params=artifact['actor_params']
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params))
    device=jax.devices(backend)[0]
    return LoadedActor(jax.device_put(params,device),metadata,make_policy(metadata,backend),device)
