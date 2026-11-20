"""Training-only bestN-rank policy improvement for an invertible latent actor.

Ideal continuous-score target: p*(z|s)=N*p0(z)*F_s(score(z))**(N-1).
Finite bank, smooth CDF, and an approximate fast decoder make the implemented
objective a surrogate, not an exactly measured reverse KL. Its parameters
are absent from deployment; the actor remains one Gaussian and one action.
"""
import math
from utils.flow_iql_research import affine_latent_actor_v1 as affine
import jax
import jax.numpy as jnp


def objective(router,router_params,observations,epsilon,decoder,decoder_params,critic,
              bank_scores,selected_latents,forward_coefficient,best_of=9,smoothing=.3):
    assert bank_scores.shape[-1]==9 and best_of==9 and smoothing>0
    transformed=router.apply({'params':router_params},observations,epsilon)
    z=transformed['latent']
    frozen=jax.tree_util.tree_map(jax.lax.stop_gradient,decoder_params)
    action=jnp.clip(decoder.apply({'params':frozen},observations,z),-1,1)
    q=critic(observations,actions=action).min(0)
    scale=jax.lax.stop_gradient(jnp.maximum(bank_scores.std(-1),1.))
    smooth_ranks=jax.nn.sigmoid((q[...,None]-bank_scores)/(smoothing*scale[...,None]))
    cdf=(1.+smooth_ranks.sum(-1))/(bank_scores.shape[-1]+1.)
    log_ratio=.5*jnp.sum(z*z-epsilon*epsilon,-1)-transformed['logdet']
    reverse=jnp.mean(log_ratio-math.log(best_of)-(best_of-1)*jnp.log(cdf))
    log_model=affine.log_probability(router,router_params,observations,selected_latents)
    log_prior=-.5*jnp.sum(selected_latents**2,-1)-.5*selected_latents.shape[-1]*math.log(2*math.pi)
    forward=jnp.mean(log_prior-log_model)
    total=reverse+forward_coefficient*forward
    return total,dict(loss=total,reverse_rank_KL_surrogate=reverse,
        forward_relative_cross_entropy=forward,forward_coefficient=jnp.asarray(forward_coefficient),
        estimated_prior_KL=jnp.mean(log_ratio),sampled_log_ratio_max=log_ratio.max(),
        rank_CDF_mean=cdf.mean(),rank_CDF_min=cdf.min(),actor_model_Q=q.mean(),
        BC_bank_Q_mean=bank_scores.mean(),Q_spread_normalizer=scale.mean(),
        sampling_latent_norm_squared=jnp.mean(jnp.sum(z*z,-1)))
