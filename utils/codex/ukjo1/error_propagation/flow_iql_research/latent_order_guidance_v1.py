"""FQL-style one-step value guidance inside a density-capped latent family.

Enumerate mixture components only in TRAINING. All components use the same
Gaussian draw to reduce variance. The decoder is frozen for the Q term;
only selector parameters receive policy-improvement gradients.
"""
from utils.flow_iql_research import orderstat_latent_actor_v1 as order
import jax
import jax.numpy as jnp


def component_latents(parameters,gaussian,best_of=9):
    directions=jnp.moveaxis(parameters['directions'],-2,0)
    projection=jnp.sum(directions*gaussian[None],axis=-1)
    shifted=gaussian[None]+(order.maximum_normal_quantile(projection,best_of)-projection)[...,None]*directions
    return jnp.concatenate([gaussian[None],shifted],axis=0)


def expected_Q_loss(parameters,gaussian,observations,decoder,decoder_params,critic,proposal_heads,best_of=9):
    z=component_latents(parameters,gaussian,best_of)
    repeated=jnp.broadcast_to(observations,(z.shape[0],*observations.shape))
    frozen_decoder=jax.tree_util.tree_map(jax.lax.stop_gradient,decoder_params)
    actions=jnp.clip(decoder.apply({'params':frozen_decoder},repeated,z),-1,1)
    q=critic(repeated,actions=actions).mean(0)
    weights=jnp.moveaxis(jax.nn.softmax(parameters['logits'],axis=-1),-1,0)
    expected=(weights*q).sum(0)
    reference=proposal_heads.mean(0)
    baseline=jax.lax.stop_gradient(reference.mean(0))
    scale=jax.lax.stop_gradient(jnp.maximum(reference.std(0),1.))
    loss=-jnp.mean((expected-baseline)/scale)
    return loss,dict(Q_loss=loss,actor_expected_Q=expected.mean(),BC_reference_Q=baseline.mean(),
        Q_normalizer=scale.mean(),training_guidance_components=jnp.asarray(z.shape[0],jnp.float32))
