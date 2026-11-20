"""IFQL value correction with a rolling critic history for deployment only.

Learning is exactly the reference bounded_v2 update. The current BC flow makes
nine proposals and past/current critics rank them by mean minimum-twin Q.
History snapshots do not enter a training loss or the replay random stream.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research import temporal_selection_v1, selection_v2
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, create, make_scan, probe, probe_summary,
)
import jax
import numpy as np


PROFILES = {'bounded_v2_temporal3': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=.999, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(num_samples=9, critic_history_size=3, critic_snapshot_stride=100000)
    return config


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'bounded_v2_temporal3' and kind == 'ifql'
    return reference.build_update('bounded_v2', bounds, kind)


def capture(history, state, step):
    """Retain at most three actual positive-step critic parameter snapshots."""
    assert step > 0
    if history and history[-1]['step'] == step:
        return history
    assert not history or history[-1]['step'] < step
    entry = dict(step=step, params=jax.device_get(state.agent.network.params['modules_critic']))
    return (list(history)+[entry])[-3:]


def validate_history(history, state, step):
    stride = state.agent.config['critic_snapshot_stride']
    k = step//stride
    expected = [stride*i for i in range(max(1, k-2), k+1)]
    assert [entry['step'] for entry in history] == expected
    if history and history[-1]['step'] == step:
        saved = jax.tree_util.tree_leaves(history[-1]['params'])
        current = jax.tree_util.tree_leaves(state.agent.network.params['modules_critic'])
        assert len(saved) == len(current)
        assert all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
                   for a, b in zip(saved, current))


def make_policy(agent, history, step):
    """Dynamic array arguments avoid embedding old critic weights in XLA."""
    assert all(entry['step'] <= step for entry in history)
    assert [h['step'] for h in history] == sorted({h['step'] for h in history})
    older = [entry for entry in history if entry['step'] < step][-2:]
    member_steps = [entry['step'] for entry in older]+[step]
    if not older:
        return selection_v2.make_policy(agent, 'max', agent.config['num_samples'], 'behavior'), member_steps
    device = jax.devices('cpu')[0]
    old_parameters = tuple(jax.device_put(entry['params'], device) for entry in older)
    selector = temporal_selection_v1.make_selector(agent, agent.config['num_samples'])

    def policy(params, observation, key):
        action, _ = selector(params, old_parameters+(params['modules_critic'],), observation, key)
        return action
    return policy, member_steps
