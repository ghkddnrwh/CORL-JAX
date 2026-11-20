"""Route to one learned mixture location before computing the action head.

The four-layer trunk and logits are evaluated first. Gather one trained
location-head kernel, then compute one action. No(K,D)location/action bank,
scale heads, flow, or critic is evaluated. Sampling mode shares the original
D+1 Gaussian draw solely to keep its categorical random number aligned.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np


DECODERS=('sampled_component_location','highest_weight_location')


def decode(params,observations,key,metadata,decoder):
    assert decoder in DECODERS and metadata['layer_norm']is False
    x=observations
    for index in range(len(metadata['hidden_dims'])):
        layer=params[f'Dense_{index}'];x=jax.nn.gelu(jnp.dot(x,layer['kernel'])+layer['bias'])
    output=params[f"Dense_{len(metadata['hidden_dims'])}"];k=metadata['components'];d=metadata['action_dim']
    logits=jnp.dot(x,output['kernel'][:,:k])+output['bias'][:k]
    if decoder=='highest_weight_location':component=jnp.argmax(logits,axis=-1)
    else:
        ak,_=jax.random.split(key)
        epsilon=jax.random.normal(ak,(*observations.shape[:-1],metadata['noise_dim']))
        cdf=jnp.cumsum(jax.nn.softmax(logits,-1),-1)
        component=jnp.minimum(jnp.sum(ndtr(epsilon[...,0,None])>cdf,axis=-1),k-1)
    columns=k+component[...,None]*d+jnp.arange(d)
    chosen_kernel=jnp.moveaxis(jnp.take(output['kernel'],columns,axis=1),0,-2)
    chosen_bias=jnp.take(output['bias'],columns)
    return jnp.clip(jnp.einsum('...h,...hd->...d',x,chosen_kernel)+chosen_bias,-1,1)


def make_policy(metadata,decoder,backend='cpu'):
    assert metadata['architecture']=='clipped_diagonal_mixture_MLP'
    assert metadata['flow_integration_steps']==metadata['generated_candidate_actions']==0
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    return jax.jit(lambda p,x,key:decode(p,x,key,metadata,decoder),backend=backend)


@dataclass
class LoadedActor:
    params:object
    metadata:dict
    policy:object
    device:object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),jax.device_put(key,self.device))


def load_actor(path,decoder,backend='cpu'):
    with Path(path).open('rb')as f:a=pickle.load(f)
    assert a['schema']=='single_actor_deployment_v1'
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(a['actor_params']))
    device=jax.devices(backend)[0]
    return LoadedActor(jax.device_put(a['actor_params'],device),a['metadata'],make_policy(a['metadata'],decoder,backend),device)
