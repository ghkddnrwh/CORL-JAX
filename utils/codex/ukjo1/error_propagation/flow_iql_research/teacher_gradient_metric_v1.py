"""Anchored behavior imitation with a frozen teacher-critic action metric.

Uses gradients at the TEACHER action, never critic values or gradients at a
student action. Both nonnegative losses vanish at the same teacher target.
The learned critic supplies a local geometry, not an oracle of task success.
"""
import jax
import jax.numpy as jnp


def normalized_gradients(gradients,training_mean_square,cap=10.):
    """Common RMS scale; bound each head's normalized squared norm by cap*D."""
    d=gradients.shape[-1]
    normalized=gradients/jnp.sqrt(jnp.maximum(training_mean_square,1e-12))
    squared=jnp.sum(normalized**2,axis=-1,keepdims=True)
    factor=jnp.minimum(1.,jnp.sqrt(cap*d/jnp.maximum(squared,1e-12)))
    return jax.lax.stop_gradient(normalized*factor)


def loss(raw,target,teacher_metric,coefficient):
    target=jax.lax.stop_gradient(target)
    geometry=jax.lax.stop_gradient(teacher_metric)
    mse=jnp.mean((raw-target)**2)
    error=jnp.clip(raw,-1,1)-target
    projected=jnp.sum(geometry*error[...,None,:],axis=-1)
    metric_penalty=jnp.mean(projected**2)/raw.shape[-1]
    total=mse+coefficient*metric_penalty
    return total,dict(raw_MSE=mse,teacher_linearized_penalty=metric_penalty,
                     clipped_MSE=jnp.mean(error**2),loss=total)
