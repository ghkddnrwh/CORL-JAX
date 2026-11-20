"""Best9 likelihood anchor plus FQL-style normalized expected-Q guidance.

Marginalize the categorical component exactly in training; reparameterize
one Gaussian draw per component. This does not sample/rerank a bank atdeploy.
The likelihood teacher uses online minQ-best9; actor guidance uses online
meanQ as in official FQL. The critic is supplied as a frozen callable.
"""
import jax
import jax.numpy as jnp

from utils.flow_iql_research.mixture_onestep_actor_v1 import log_probability


def component_actions(parameters,noises):
    means,scales=parameters['means'],parameters['scales']
    assert means.ndim==3 and noises.shape==(means.shape[1],means.shape[0],means.shape[2])
    return jnp.clip(jnp.moveaxis(means,1,0)+jnp.moveaxis(scales,1,0)*noises,-1,1)


def loss(parameters,target,observations,noises,critic,alpha,use_q):
    nll=-log_probability(parameters,target).mean()/target.shape[-1]
    weights=jax.nn.softmax(parameters['logits'],axis=-1)
    entropy=-jnp.sum(weights*jax.nn.log_softmax(parameters['logits'],axis=-1),axis=-1)
    info=dict(distill_objective=nll,distill_NLL_per_dimension=nll,
        weighted_mean_scale=jnp.mean(jnp.sum(weights[...,None]*parameters['scales'],axis=1)),
        component_entropy=entropy.mean(),effective_components=jnp.exp(entropy).mean())
    if use_q:
        actions=component_actions(parameters,noises)
        broadcast=jnp.broadcast_to(observations,(actions.shape[0],*observations.shape))
        q=critic(broadcast,actions=actions).mean(0)
        w=jnp.moveaxis(weights,1,0)
        expected_q=jnp.sum(w*q,axis=0)
        normalizer=jax.lax.stop_gradient(jnp.maximum(jnp.mean(jnp.sum(w*jnp.abs(q),axis=0)),1e-6))
        q_loss=-expected_q.mean()/normalizer
        target_q=critic(observations,actions=target).mean(0)
        info.update(actor_expected_Q=expected_q.mean(),teacher_action_Q=target_q.mean(),
            predicted_Q_regret=(target_q-expected_q).mean(),Q_normalizer=normalizer)
    else:q_loss=jnp.asarray(0.)
    total=alpha*nll+q_loss
    return total,dict(info,loss=total,Q_loss=q_loss)
