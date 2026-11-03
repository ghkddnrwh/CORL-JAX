"""Matched2x2 extraction ablation: raw/rank coordinates x mean/median."""
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research.independent_critic_policy_v1 import midranks
import jax
import jax.numpy as jnp


RULES=('mean','median_rank','median_value','mean_rank')


def scores(values,rule):
    assert rule in RULES
    if rule=='mean':return values.mean(0)
    if rule=='median_rank':return jnp.median(midranks(values),axis=0)
    if rule=='median_value':return jnp.median(values,axis=0)
    return midranks(values).mean(0)


def make_selector(agent,rule='median_rank',samples=9):
    assert rule in RULES and samples==9
    def select(params,critic_members,observations,key):
        assert len(critic_members)in(1,3)
        model=agent.replace(network=agent.network.replace(params=params))
        actions,_=selection_v2.proposals(model,observations,key,samples,'behavior')
        repeated=jnp.broadcast_to(observations,(samples,*observations.shape))
        values=jnp.stack([model.network.select('critic')(repeated,actions=actions,
            params={**params,'modules_critic':member}).min(0)for member in critic_members])
        combined=values[0]if len(critic_members)==1 else scores(values,rule)
        best=jnp.argmax(combined,axis=0)
        index=jnp.broadcast_to(best[None,...,None],(1,*actions.shape[1:]))
        action=jnp.take_along_axis(actions,index,axis=0)[0]
        return action,values,combined
    return jax.jit(select,backend='cpu')
