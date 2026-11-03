"""Single-forward mixture-density actor, with exact clipping likelihood.

One MLP outputs distribution parameters. A categorical component and one
Gaussian action are drawn; no bank of candidate actions or Q is generated.
MDNs are prior art: Bishop1994 NCRG/94/004. Clipped-Gaussian likelihood
accounts for endpoint point masses rather than treating them as densities.
"""
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr, logsumexp


class MixtureActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    components: int = 9
    standard_deviation_floor: float = .001

    @nn.compact
    def __call__(self,observations,latents,return_parameters=False):
        # Latents are used only after the one MLP has produced its parameters.
        x=observations
        dimensions=(*self.hidden_dims,self.components*(1+2*self.action_dim))
        for index,width in enumerate(dimensions):
            x=nn.Dense(width,kernel_init=nn.initializers.variance_scaling(1.,'fan_avg','uniform'))(x)
            if index<len(dimensions)-1:x=nn.gelu(x)
        k,d=self.components,self.action_dim
        logits=x[...,:k]
        means=x[...,k:k+k*d].reshape((*x.shape[:-1],k,d))
        scales=jax.nn.softplus(x[...,k+k*d:].reshape((*x.shape[:-1],k,d)))+self.standard_deviation_floor
        parameters=dict(logits=logits,means=means,scales=scales)
        if return_parameters:return parameters
        return sample_one(parameters,latents)


def sample_one(parameters,latents):
    """D+1iidnormal latents: one uniform categorical draw and D normal draws."""
    logits,means,scales=(parameters[k] for k in ('logits','means','scales'))
    assert latents.shape[-1]==means.shape[-1]+1
    u=ndtr(latents[...,0])
    cdf=jnp.cumsum(jax.nn.softmax(logits,axis=-1),axis=-1)
    index=jnp.minimum(jnp.sum(u[...,None]>cdf,axis=-1),logits.shape[-1]-1)
    indices=jnp.broadcast_to(index[...,None,None],(*index.shape,1,means.shape[-1]))
    # Gather parameters first; only one Gaussian action is ever generated.
    location=jnp.take_along_axis(means,indices,axis=-2)[...,0,:]
    scale=jnp.take_along_axis(scales,indices,axis=-2)[...,0,:]
    return jnp.clip(location+scale*latents[...,1:],-1,1)


def log_probability(parameters,actions):
    """Mixed Lebesgue+endpoint measure on each coordinate of[-1,1]^D."""
    logits,means,scales=(parameters[k] for k in ('logits','means','scales'))
    z=(actions[...,None,:]-means)/scales
    density=-.5*z*z-jnp.log(scales)-.5*math.log(2*math.pi)
    lower=log_ndtr((-1-means)/scales)
    upper=log_ndtr((means-1)/scales)
    per_coordinate=jnp.where(actions[...,None,:]<=-1,lower,
                            jnp.where(actions[...,None,:]>=1,upper,density))
    component=jnp.sum(per_coordinate,axis=-1)
    return logsumexp(jax.nn.log_softmax(logits,axis=-1)+component,axis=-1)
