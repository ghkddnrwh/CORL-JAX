"""One actor, one selected Gaussian parameter head, one realized action.

Temperature multiplies the selected standard deviation, not its variance.
The zero-temperature path is exactly the frozen location decoder. This
module never evaluates a critic, Flow, or bank of generated actions.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as location


TEMPERATURES=(0.,.25,.5,1.)


def decode(params,observations,key,metadata,temperature):
    assert temperature in TEMPERATURES and metadata['layer_norm']is False
    if temperature==0.:
        return location.decode(params,observations,key,metadata,'sampled_component_location')
    x=observations
    for i in range(len(metadata['hidden_dims'])):
        layer=params[f'Dense_{i}'];x=jax.nn.gelu(jnp.dot(x,layer['kernel'])+layer['bias'])
    head=params[f"Dense_{len(metadata['hidden_dims'])}"];k=metadata['components'];d=metadata['action_dim']
    logits=jnp.dot(x,head['kernel'][:,:k])+head['bias'][:k]
    noise=jax.random.normal(jax.random.split(key)[0],(*observations.shape[:-1],metadata['noise_dim']))
    cdf=jnp.cumsum(jax.nn.softmax(logits,-1),-1)
    component=jnp.minimum(jnp.sum(ndtr(noise[...,0,None])>cdf,-1),k-1)
    # Gather one component's mean AND scale kernels before the action head.
    means=k+component[...,None]*d+jnp.arange(d)
    scales=k+k*d+component[...,None]*d+jnp.arange(d)
    columns=jnp.concatenate([means,scales],axis=-1)
    kernel=jnp.moveaxis(jnp.take(head['kernel'],columns,axis=1),0,-2)
    values=jnp.einsum('...h,...hd->...d',x,kernel)+jnp.take(head['bias'],columns)
    mu=values[...,:d];sigma=jax.nn.softplus(values[...,d:])+metadata['std_floor']
    return jnp.clip(mu+temperature*sigma*noise[...,1:],-1,1)


def make_policy(metadata,temperature,backend='cpu'):
    assert metadata['architecture']=='clipped_diagonal_mixture_MLP'
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    assert metadata['flow_integration_steps']==metadata['generated_candidate_actions']==0
    return jax.jit(lambda params,obs,key:decode(params,obs,key,metadata,temperature),backend=backend)


@dataclass
class LoadedActor:
    params:object
    metadata:dict
    policy:object
    device:object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),
                           jax.device_put(key,self.device))


def load_actor(path,temperature,backend='cpu'):
    with Path(path).open('rb')as stream:artifact=pickle.load(stream)
    assert artifact['schema']=='single_actor_deployment_v1'and artifact['identity']['recipe']=='mdn9_bc'
    params=artifact['actor_params'];m=artifact['metadata']
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(params))
    device=jax.devices(backend)[0]
    metadata=dict(m,runtime_temperature=float(temperature),computed_action_heads=1,
        output='clip_one_selected_location_plus_temperature_times_scale_times_noise',
        training_weights_changed=False)
    return LoadedActor(jax.device_put(params,device),metadata,make_policy(m,temperature,backend),device)
