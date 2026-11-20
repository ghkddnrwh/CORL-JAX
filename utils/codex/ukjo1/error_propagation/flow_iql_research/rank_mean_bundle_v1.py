"""Load deployment parameters for the fixed three-critic rank-mean policy.

There are no optimizer/replay states in this artifact. It cannot resume training.
Checkpoint paths are provenance only and need not exist on a deployment host.
"""
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

from utils.flow_iql_research import policy_core, independent_critic_policy_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with Path(path).open('rb') as stream:
        bundle = pickle.load(stream)
    assert bundle['kind'] == 'independent_critic_rank_mean_deployment_v1'
    assert not bundle['resumable_training_checkpoint']
    assert bundle['recipe'] == dict(rule='mean_rank', samples=9, learners=3,
        member_twin_head_reduction='min', common_proposal_actor='first_member_BC', tie_rule='first_proposal')
    assert len(bundle['members']) == 3 and len(bundle['other_critic_params']) == 2
    identities = [m['identity'] for m in bundle['members']]
    first = identities[0]
    assert len({i['seed'] for i in identities}) == 3
    for identity in identities:
        assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
        assert identity['environment'] == first['environment']
        assert identity['steps'] == 1000000 and identity['episodes'] == 50
        assert {k: v for k, v in identity['config'].items() if k != 'num_samples'} == {
            k: v for k, v in first['config'].items() if k != 'num_samples'}
    for source, digest in bundle['source_sha256'].items():
        assert sha(source) == digest, source
    cfg = dict(first['config'])
    assert cfg['discount'] == .999 and cfg['expectile'] == .9
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    template = policy_core.create(first['seed'], jnp.zeros((1, bundle['observation_dim'])),
        jnp.zeros((1, bundle['action_dim'])), cfg, 'ifql')
    params = flax.serialization.from_state_dict(template.agent.network.params, bundle['first_network_params'])
    params = jax.device_put(params, jax.devices('cpu')[0])
    agent = template.agent.replace(network=template.agent.network.replace(params=params))
    members = (params['modules_critic'],) + tuple(
        jax.device_put(flax.serialization.from_state_dict(params['modules_critic'], p), jax.devices('cpu')[0])
        for p in bundle['other_critic_params'])
    for tree in (params, members):
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))
    compiled = independent_critic_policy_v2.make_selector(agent, 'mean_rank', 9)

    def policy(parameters, observations, key):
        return compiled(parameters, members, observations, key)[0]

    # evaluate() requires an agent container only. No synthetic learning counters
    # or optimizer moments are presented as trained state.
    return bundle, SimpleNamespace(agent=agent), policy
