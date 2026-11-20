"""Change only the deployed twin-Q reducer on the same nine BC proposals."""
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


def make_policy(agent, aggregation='mean', samples=9):
    assert aggregation in ('min', 'mean') and samples == 9

    def policy(params, observations, key):
        model = agent.replace(network=agent.network.replace(params=params))
        actions, minimum = selection_v2.proposals(model, observations, key, samples, 'behavior')
        if aggregation == 'min':
            values = minimum
        else:
            repeated = jnp.broadcast_to(observations, (samples, *observations.shape))
            values = model.network.select('critic')(repeated, actions=actions).mean(0)
        best = jnp.argmax(values, axis=0)
        index = jnp.broadcast_to(best[None,...,None], (1,*actions.shape[1:]))
        return jnp.take_along_axis(actions, index, axis=0)[0]

    return jax.jit(policy, backend='cpu')
