"""One sampled latent from a full-covariance mixture, then one fast decoder.

Portable Flax/JAX inference: two MLP blocks and a small Cholesky-vector
product. No action candidate bank, critic or numerical flow integration.
Full-covariance mixtures are standard density models, not a novelty claim.
"""
import math
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp,ndtr
import numpy as np
from utils.flow_iql_research.standalone_onestep_actor_v1 import _Actor


class FullCovarianceRouter(nn.Module):
    hidden_dims: tuple
    dimension: int
    components: int=9
    standard_deviation_floor: float=.1

    @nn.compact
    def __call__(self,observations):
        d,k=self.dimension,self.components
        lower_size=d*(d+1)//2
        x=observations
        for i,width in enumerate((*self.hidden_dims,k*(1+d+lower_size))):
            x=nn.Dense(width,kernel_init=nn.initializers.variance_scaling(1.,'fan_avg','uniform'))(x)
            if i<len(self.hidden_dims):x=nn.gelu(x)
        logits=x[...,:k]
        means=x[...,k:k+k*d].reshape((*x.shape[:-1],k,d))
        raw=x[...,k+k*d:].reshape((*x.shape[:-1],k,lower_size))
        row,col=np.tril_indices(d)
        chol=jnp.zeros((*x.shape[:-1],k,d,d),dtype=x.dtype).at[...,row,col].set(raw)
        diag=jnp.arange(d)
        chol=chol.at[...,diag,diag].set(jax.nn.softplus(chol[...,diag,diag])+self.standard_deviation_floor)
        return dict(logits=logits,means=means,cholesky=chol)


def log_probability(parameters,values):
    logits,means,chol=(parameters[k]for k in ('logits','means','cholesky'))
    delta=values[...,None,:]-means
    whitened=jax.lax.linalg.triangular_solve(chol,delta[...,None],left_side=True,lower=True)[...,0]
    diagonal=jnp.diagonal(chol,axis1=-2,axis2=-1)
    component=-.5*jnp.sum(whitened**2,axis=-1)-jnp.log(diagonal).sum(-1)-.5*means.shape[-1]*math.log(2*math.pi)
    return logsumexp(jax.nn.log_softmax(logits,axis=-1)+component,axis=-1)


def sample_latent(parameters,noise):
    logits,means,chol=(parameters[k]for k in ('logits','means','cholesky'))
    d=means.shape[-1];assert noise.shape[-1]==d+1
    cdf=jnp.cumsum(jax.nn.softmax(logits,axis=-1),axis=-1)
    index=jnp.minimum(jnp.sum(ndtr(noise[...,0,None])>cdf,axis=-1),logits.shape[-1]-1)
    mean_index=jnp.broadcast_to(index[...,None,None],(*index.shape,1,d))
    chol_index=jnp.broadcast_to(index[...,None,None,None],(*index.shape,1,d,d))
    mu=jnp.take_along_axis(means,mean_index,axis=-2)[...,0,:]
    factor=jnp.take_along_axis(chol,chol_index,axis=-3)[...,0,:,:]
    # Select distribution parameters first; realize just one latent vector.
    return mu+jnp.einsum('...ij,...j->...i',factor,noise[...,1:])


class FullCovSelectedActor(nn.Module):
    hidden_dims: tuple
    action_dim: int
    components: int=9
    standard_deviation_floor: float=.1

    def setup(self):
        self.router=FullCovarianceRouter(self.hidden_dims,self.action_dim,self.components,self.standard_deviation_floor)
        self.decoder=_Actor(self.hidden_dims,self.action_dim,False)

    def __call__(self,observations,noise):
        z=sample_latent(self.router(observations),noise)
        return jnp.clip(self.decoder(observations,z),-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['MLP_blocks']==2 and metadata['generated_candidate_actions']==0
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    assert metadata['flow_integration_steps']==0
    model=FullCovSelectedActor(tuple(metadata['hidden_dims']),metadata['action_dim'],
        metadata['components'],metadata['standard_deviation_floor'])
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        z=jax.random.normal(action_key,(*observations.shape[:-1],metadata['action_dim']+1))
        return model.apply({'params':params},observations,z)
    return jax.jit(policy,backend=backend)
