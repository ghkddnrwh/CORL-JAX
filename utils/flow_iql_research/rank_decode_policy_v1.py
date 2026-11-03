"""Refine only the student's Euler grid; leave the BC flow at ten steps."""
import jax
import jax.numpy as jnp


def decode(agent, observations, seed, steps):
    assert steps in (10,20,50)
    action_key,_=jax.random.split(seed)
    actions=jax.random.normal(action_key,(*observations.shape[:-1],agent.config['action_dim']))
    for index in range(steps):
        time=jnp.full((*observations.shape[:-1],1),index/steps)
        actions=actions+agent.network.select('actor_policy_flow')(observations,actions,time)/steps
    return jnp.clip(actions,-1,1)


def make_policy(agent,steps):
    assert agent.config['flow_steps']==10
    def policy(params,observations,key):
        model=agent.replace(network=agent.network.replace(params=params))
        return decode(model,observations,key,steps)
    return jax.jit(policy,backend='cpu')
