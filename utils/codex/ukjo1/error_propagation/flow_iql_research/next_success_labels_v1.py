"""Recover only next-state success labels certified by dataset adjacency.

Regular OGBench retains original trajectory-cut terminals separately from
state-based success masks. Never shift labels across an original trajectory cut.
The final next state of a trajectory is not a retained current-state row, so
its label is deliberately left unknown. No observations are approximately matched.
"""
import hashlib
import numpy as np


def labels(data, chunk=65536):
    obs = np.asarray(data['observations']); nxt = np.asarray(data['next_observations'])
    terminals = np.asarray(data['terminals']); masks = np.asarray(data['masks'])
    rewards = np.asarray(data['rewards']); n = len(masks)
    assert obs.shape == nxt.shape and len(obs) == len(terminals) == len(rewards) == n
    assert terminals.shape == masks.shape == rewards.shape == (n,)
    assert np.isin(terminals,[0,1]).all() and np.isin(masks,[0,1]).all()
    assert np.all(rewards[masks == 0] == 0), 'Needs state-based r0, mask0 success boundary'
    known = np.zeros(n,dtype=bool); success = np.zeros(n,dtype=bool)
    for start in range(0,max(n-1,0),chunk):
        end = min(start+chunk,n-1)
        same_trajectory = terminals[start:end] == 0
        exact = np.all(nxt[start:end] == obs[start+1:end+1],axis=tuple(range(1,obs.ndim)))
        assert np.all(exact[same_trajectory]), 'Nonterminal adjacency mismatch: cannot shift success labels safely'
        known[start:end] = same_trajectory
        success[start:end] = same_trajectory & (masks[start+1:end+1] == 0)
    return success.astype(np.float32), dict(transitions=n,
        adjacency_known_count=int(known.sum()), next_success_count=int(success.sum()),
        active_next_success_count=int(np.sum(success & (masks == 1))),
        unknown_next_count=int((~known).sum()),
        active_next_success_fraction=float(np.mean(success & (masks == 1))) if n else 0.,
        all_noncut_adjacency_exact=True, cross_trajectory_labels_used=False,
        next_success_label_sha256=hashlib.sha256(success.tobytes()).hexdigest())
