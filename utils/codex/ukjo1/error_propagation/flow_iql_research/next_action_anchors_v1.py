"""Recover observed next actions without crossing OGBench trajectory boundaries."""
import numpy as np


def augment(raw, data):
    count = len(data['actions'])
    terminal = np.asarray(raw['terminals'])
    assert terminal.shape == (count,) and np.isin(terminal, [0, 1]).all()
    valid = np.zeros(count, dtype=bool)
    valid[:-1] = terminal[:-1] == 0
    # OGBench regular datasets drop the final dummy action of each trajectory.
    # Only nonterminal transitions have an observed next action in this array.
    for start in range(0, max(count-1, 0), 65536):
        end = min(start+65536, count-1)
        selected = valid[start:end]
        assert np.array_equal(data['next_observations'][start:end][selected],
                              data['observations'][start+1:end+1][selected]), 'Next-state alignment failed'
    next_actions = np.zeros_like(data['actions'])
    next_actions[:-1] = data['actions'][1:]
    next_actions[~valid] = 0
    return dict(data, next_actions=next_actions, next_action_valid=valid.astype(np.float32))
