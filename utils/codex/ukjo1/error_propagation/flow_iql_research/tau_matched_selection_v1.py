"""Nested-prefix best-of-N, with N derived from the expectile odds.

This is a policy extraction experiment, not a new training objective. The
implicit expectile actor is from IDQL (arxiv:2304.10573). No novelty claim.
The ideal comparison assumes exact Q, its exact behavior expectile, and IID
behavior draws. Learned critics/behavior flows need not meet those assumptions.
"""
from fractions import Fraction
import math
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


EXTRA_PROPOSAL_FOLD = 91837


def proposal_count(expectile):
    tau = Fraction(str(expectile))
    assert Fraction(1, 2) <= tau < 1
    return math.ceil(tau / (1-tau))


def decision(agent, observations, key, prefix=9):
    n = max(prefix, proposal_count(agent.config['expectile']))
    actions, q = selection_v2.proposals(agent, observations, key, prefix, 'behavior')
    q_prefix = q.max(axis=0)
    if n > prefix:
        extra, extra_q = selection_v2.proposals(agent, observations,
            jax.random.fold_in(key, EXTRA_PROPOSAL_FOLD), n-prefix, 'behavior')
        actions = jnp.concatenate((actions, extra), axis=0)
        q = jnp.concatenate((q, extra_q), axis=0)
    index = jnp.argmax(q, axis=0)
    selected = jnp.broadcast_to(index[None, ..., None], (1, *actions.shape[1:]))
    action = jnp.take_along_axis(actions, selected, axis=0)[0]
    return action, q_prefix, q.max(axis=0), index


def make_decision(agent):
    def policy(params, observations, key):
        current = agent.replace(network=agent.network.replace(params=params))
        return decision(current, observations, key)
    return jax.jit(policy, backend='cpu')
