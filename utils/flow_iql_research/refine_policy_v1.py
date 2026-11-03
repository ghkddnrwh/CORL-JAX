"""One local action correction toward the fitted IQL value threshold.

The starting action is the same best-of-N behavior-flow proposal. When its
minimum-head Q is below V, the linearized minimum-distance correction is
(V-Q) grad(Q) / ||grad(Q)||^2. Cap its RMS action displacement by epsilon,
clip to the action box, and accept only if the actual fitted minimum Q rises.
This is a model-value safeguard, not a true-return improvement guarantee.
"""
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


def make_policy(agent, samples=9, epsilon=.1):
    assert samples > 0 and epsilon >= 0
    assert 'modules_value' in agent.network.params

    def corrected(params, observation, key):
        model = agent.replace(network=agent.network.replace(params=params))
        actions, values = selection_v2.proposals(model, observation, key, samples, 'behavior')
        index = jnp.argmax(values, axis=0)
        original = actions[index]
        q0 = values[index]
        v = model.network.select('value')(observation)
        score = lambda action: model.network.select('critic')(observation, actions=action).min(0)
        gradient = jax.grad(score)(original)
        norm2 = jnp.sum(gradient**2)
        rms = jnp.sqrt(jnp.mean(gradient**2))
        gap = jnp.maximum(v-q0, 0.)
        scale = jnp.minimum(gap/jnp.maximum(norm2, 1e-16), epsilon/jnp.maximum(rms, 1e-16))
        trial = jnp.clip(original+scale*gradient, -1., 1.)
        q1 = score(trial)
        active = (gap > 0.) & (norm2 > 1e-16)
        accept = active & (q1 >= q0) & jnp.isfinite(q1)
        action = jnp.where(accept, trial, original)
        q_final = jnp.where(accept, q1, q0)
        statistics = jnp.stack([
            active.astype(jnp.float32), accept.astype(jnp.float32), gap,
            q_final-q0, jnp.sqrt(jnp.mean((action-original)**2)), rms,
            (q_final < v).astype(jnp.float32), jnp.mean((jnp.abs(action)>=1.).astype(jnp.float32))])
        return action, statistics

    return jax.jit(corrected, backend='cpu')


STATISTICS = ('active_fraction', 'accepted_fraction', 'initial_value_deficit_mean',
              'accepted_predicted_Q_gain_mean', 'action_displacement_RMS_mean',
              'initial_action_gradient_RMS_mean', 'final_Q_below_V_fraction',
              'action_box_boundary_fraction')
