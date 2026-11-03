"""Conditional Real-NVP latent policy followed by one fast action decoder.

Four affine coupling blocks form one finite feed-forward graph. There is no
numerical ODE integration, rejection, candidate generation, or critic query.
The coupling construction is standard Real NVP (Dinh et al., 2017).
"""
import math
import flax.linen as nn
import jax
import jax.numpy as jnp
from utils.flow_iql_research.standalone_onestep_actor_v1 import _Actor


class Conditioner(nn.Module):
    hidden_dims: tuple
    dimension: int

    @nn.compact
    def __call__(self,observations,masked):
        x=jnp.concatenate([observations,masked],axis=-1)
        for width in self.hidden_dims:
            x=nn.gelu(nn.Dense(width,kernel_init=nn.initializers.variance_scaling(1.,'fan_avg','uniform'))(x))
        # Identity at initialization: no arbitrary initial distortion of the prior.
        return nn.Dense(2*self.dimension,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x)


class ConditionalAffineRouter(nn.Module):
    hidden_dims: tuple
    dimension: int
    layers: int=4
    log_scale_bound: float=1.

    @nn.compact
    def __call__(self,observations,values,inverse=False):
        assert self.dimension>=2 and values.shape[-1]==self.dimension
        modules=[Conditioner(self.hidden_dims,self.dimension,name=f'coupling_{i}')for i in range(self.layers)]
        logdet=jnp.zeros(values.shape[:-1],values.dtype);x=values
        order=range(self.layers-1,-1,-1)if inverse else range(self.layers)
        for index in order:
            mask=((jnp.arange(self.dimension)+index)%2).astype(values.dtype)
            raw_scale,raw_shift=jnp.split(modules[index](observations,x*mask),2,axis=-1)
            scale=self.log_scale_bound*jnp.tanh(raw_scale)*(1-mask)
            shift=raw_shift*(1-mask)
            if inverse:
                x=x*mask+(1-mask)*(x-shift)*jnp.exp(-scale)
                logdet=logdet-scale.sum(-1)
            else:
                x=x*mask+(1-mask)*(x*jnp.exp(scale)+shift)
                logdet=logdet+scale.sum(-1)
        return dict(latent=x,logdet=logdet)


def log_probability(definition,params,observations,values):
    reverse=definition.apply({'params':params},observations,values,inverse=True)
    epsilon=reverse['latent'];d=values.shape[-1]
    return -.5*jnp.sum(epsilon**2,axis=-1)-.5*d*math.log(2*math.pi)+reverse['logdet']


class AffineLatentActor(nn.Module):
    router_hidden_dims: tuple
    decoder_hidden_dims: tuple
    dimension: int
    layers: int=4
    log_scale_bound: float=1.

    def setup(self):
        self.router=ConditionalAffineRouter(self.router_hidden_dims,self.dimension,self.layers,self.log_scale_bound)
        self.decoder=_Actor(self.decoder_hidden_dims,self.dimension,False)

    def __call__(self,observations,noise):
        z=self.router(observations,noise)['latent']
        return jnp.clip(self.decoder(observations,z),-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['latent_draws']==1 and metadata['generated_candidate_actions']==0
    assert metadata['flow_integration_steps']==0 and metadata['actor_forward_passes']==1
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    model=AffineLatentActor(tuple(metadata['router_hidden_dims']),tuple(metadata['decoder_hidden_dims']),
        metadata['action_dim'],metadata['coupling_layers'],metadata['log_scale_bound'])
    def policy(params,observations,key):
        ak,_=jax.random.split(key)
        epsilon=jax.random.normal(ak,(*observations.shape[:-1],metadata['action_dim']))
        return model.apply({'params':params},observations,epsilon)
    return jax.jit(policy,backend=backend)
