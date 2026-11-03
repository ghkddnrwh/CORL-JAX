"""Gaussian order-statistic mixture selector plus a one-step decoder.

For a unit direction v, selecting max_i v^T Z_i among N Gaussians gives
q_v(z)=N phi_D(z) Phi(v^T z)^(N-1). Mix these normalized components with
the prior. The ideal model obeys q<=N*prior. Sampling transforms ONE
Gaussian using a scalar inverse CDF; no candidate bank/critic/ODE is used.
"""
import math
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr,ndtr,ndtri,logsumexp
from utils.flow_iql_research.standalone_onestep_actor_v1 import _Actor


class OrderStatisticRouter(nn.Module):
    hidden_dims: tuple
    dimension: int
    directions: int=9
    best_of: int=9

    @nn.compact
    def __call__(self,observations):
        k,d=self.directions,self.dimension
        x=observations
        for i,width in enumerate((*self.hidden_dims,k+1+k*d)):
            x=nn.Dense(width,kernel_init=nn.initializers.variance_scaling(1.,'fan_avg','uniform'))(x)
            if i<len(self.hidden_dims):x=nn.gelu(x)
        logits=x[...,:k+1]
        raw=x[...,k+1:].reshape((*x.shape[:-1],k,d))
        squared=jnp.sum(raw**2,axis=-1,keepdims=True)
        normalized=raw/jnp.sqrt(jnp.maximum(squared,1e-12))
        fallback=jnp.zeros_like(raw).at[...,0].set(1.)
        unit=jnp.where(squared>1e-12,normalized,fallback)
        return dict(logits=logits,directions=unit)


def log_probability(parameters,values,best_of=9):
    unit,logits=parameters['directions'],parameters['logits']
    projection=jnp.sum(unit*values[...,None,:],axis=-1)
    logweights=jax.nn.log_softmax(logits,axis=-1)
    directional=logweights[...,1:]+math.log(best_of)+(best_of-1)*log_ndtr(projection)
    relative=logsumexp(jnp.concatenate([logweights[...,:1],directional],axis=-1),axis=-1)
    base=-.5*jnp.sum(values**2,axis=-1)-.5*values.shape[-1]*math.log(2*math.pi)
    return base+relative


def maximum_normal_quantile(normal,best_of=9):
    # The ideal transform is Phi^-1(Phi(u)^(1/N)). This log-domain form
    # avoids 1-p cancellation. |u|<=10 is a floating-point tail guard,
    # with ideal Gaussian probability <=2 Phi(-10)~1.53e-23 outside it.
    u=jnp.clip(normal,-10.,10.)
    upper_probability=-jnp.expm1(log_ndtr(u)/best_of)
    return -ndtri(upper_probability)


def sample_latent(parameters,noise,best_of=9):
    unit,logits=parameters['directions'],parameters['logits']
    d=unit.shape[-1];assert noise.shape[-1]==d+1
    cdf=jnp.cumsum(jax.nn.softmax(logits,axis=-1),axis=-1)
    component=jnp.minimum(jnp.sum(ndtr(noise[...,0,None])>cdf,axis=-1),logits.shape[-1]-1)
    direction_index=jnp.maximum(component-1,0)
    indices=jnp.broadcast_to(direction_index[...,None,None],(*component.shape,1,d))
    v=jnp.take_along_axis(unit,indices,axis=-2)[...,0,:]
    epsilon=noise[...,1:]
    projection=jnp.sum(epsilon*v,axis=-1)
    transformed=epsilon+(maximum_normal_quantile(projection,best_of)-projection)[...,None]*v
    return jnp.where((component==0)[...,None],epsilon,transformed)


class OrderStatisticSelectedActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    directions: int=9
    best_of: int=9

    def setup(self):
        self.router=OrderStatisticRouter(self.hidden_dims,self.action_dim,self.directions,self.best_of)
        self.decoder=_Actor(self.hidden_dims,self.action_dim,False)

    def __call__(self,observations,noise):
        z=sample_latent(self.router(observations),noise,self.best_of)
        return jnp.clip(self.decoder(observations,z),-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['MLP_blocks']==2 and metadata['generated_candidate_actions']==0
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    assert metadata['flow_integration_steps']==0
    model=OrderStatisticSelectedActor(tuple(metadata['hidden_dims']),metadata['action_dim'],
        metadata['directions'],metadata['best_of'])
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        epsilon=jax.random.normal(action_key,(*observations.shape[:-1],metadata['action_dim']+1))
        return model.apply({'params':params},observations,epsilon)
    return jax.jit(policy,backend=backend)
