"""Best9 BC action selection with three independent learned critics."""
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


def midranks(values):
    """Member/sample axes0/1; average ranks for ties, with .5 as lowest rank."""
    own=values[:,:,None]
    other=values[:,None,:]
    return jnp.sum(other<own,axis=2)+.5*jnp.sum(other==own,axis=2)


def scores(values,rule):
    assert rule in('mean','median_rank')
    return values.mean(0)if rule=='mean'else jnp.median(midranks(values),axis=0)


def make_selector(agent,rule='median_rank',samples=9):
    assert rule in('mean','median_rank')and samples==9
    def select(params,critic_members,observations,key):
        assert len(critic_members)in(1,3)
        model=agent.replace(network=agent.network.replace(params=params))
        actions,_=selection_v2.proposals(model,observations,key,samples,'behavior')
        repeated=jnp.broadcast_to(observations,(samples,*observations.shape))
        values=jnp.stack([model.network.select('critic')(repeated,actions=actions,
            params={**params,'modules_critic':member}).min(0)for member in critic_members])
        # A single member preserves the original max policy even on exact ties.
        combined=values[0]if len(critic_members)==1 else scores(values,rule)
        best=jnp.argmax(combined,axis=0)
        index=jnp.broadcast_to(best[None,...,None],(1,*actions.shape[1:]))
        action=jnp.take_along_axis(actions,index,axis=0)[0]
        return action,values,combined
    return jax.jit(select,backend='cpu')
