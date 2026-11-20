"""Select behavior-flow actions by a mean of retained critic predictions.

Each member first takes its minimum twin head, as IFQL does. Average these
predictions across the last three100K checkpoints. Common state-value offsets
cancel from this argmax; no true error cancellation or improvement is assumed.
"""
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


def make_selector(agent, samples=9):
    def select(params, critic_members, observation, key):
        model = agent.replace(network=agent.network.replace(params=params))
        actions, _ = selection_v2.proposals(model, observation, key, samples, 'behavior')
        observations = jnp.broadcast_to(observation, (samples, *observation.shape))
        values = []
        for critic in critic_members:
            member_params = {**params, 'modules_critic': critic}
            values.append(model.network.select('critic')(
                observations, actions=actions, params=member_params).min(0))
        q = jnp.stack(values)
        index = jnp.argmax(q.mean(0), axis=0)
        return actions[index], q
    return jax.jit(select, backend='cpu')
