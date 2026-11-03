"""One actor: shared latent classification, then decode exactly one action.

Nine Gaussian latent vectors are retained; no action candidates or Bellman
critic are evaluated. This differs from a single-latent density sampler.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
from utils.flow_iql_research.standalone_onestep_actor_v1 import _Actor


class SharedLatentGate(nn.Module):
    hidden_dims: tuple

    @nn.compact
    def __call__(self,observations,latents):
        expanded=jnp.broadcast_to(observations[...,None,:],(*latents.shape[:-1],observations.shape[-1]))
        x=jnp.concatenate([expanded,latents],axis=-1)
        for index,width in enumerate((*self.hidden_dims,1)):
            x=nn.Dense(width,kernel_init=nn.initializers.variance_scaling(1.,'fan_avg','uniform'))(x)
            if index<len(self.hidden_dims):x=nn.gelu(x)
        return x[...,0]


def select_latent(latents,logits):
    index=jnp.argmax(logits,axis=-1)
    gather=jnp.broadcast_to(index[...,None,None],(*index.shape,1,latents.shape[-1]))
    return jnp.take_along_axis(latents,gather,axis=-2)[...,0,:]


class LatentGateActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    proposals: int=9

    def setup(self):
        self.router=SharedLatentGate(self.hidden_dims)
        self.decoder=_Actor(self.hidden_dims,self.action_dim,False)

    def __call__(self,observations,noise):
        latents=noise.reshape((*observations.shape[:-1],self.proposals,self.action_dim))
        chosen=select_latent(latents,self.router(observations,latents))
        return jnp.clip(self.decoder(observations,chosen),-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['MLP_blocks']==2 and metadata['realized_action_samples']==1
    assert metadata['latent_draws']==9 and metadata['generated_candidate_actions']==0
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    model=LatentGateActor(tuple(metadata['hidden_dims']),metadata['action_dim'],9)
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        noise=jax.random.normal(action_key,(*observations.shape[:-1],9*metadata['action_dim']))
        return model.apply({'params':params},observations,noise)
    return jax.jit(policy,backend=backend)
