"""Frozen single-observation policy outputs under explicitly recorded affinity."""
import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from utils.flow_iql_research import sample_budget_core_v1 as core
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import read_source, atomic, sha


def main(args):
    plan = json.loads(args.plan.read_text())
    arrays, evidence = {}, {}
    for pair_id, name in (('hm18_s1_time', 'hm18_s1_800K'), ('pz27_seed01', 'pz27_s0_1M')):
        path = args.replay / f'{pair_id}.npz'
        with np.load(path) as data:
            observations = data['observations'][:64]
        _, state = read_source(plan['sources'][name])
        policy = core.make_teacher_policy(state)
        parameters = core.lean.policy_parameters(state.base)
        actions = [np.asarray(policy(parameters, obs, jax.random.PRNGKey(42 + i)))
                   for i, obs in enumerate(observations)]
        arrays[name + '_actions'] = np.stack(actions)
        evidence[name] = dict(checkpoint_sha256=plan['sources'][name]['checkpoint_sha256'],
                             replay_arrays_sha256=sha(path))
    assert not args.output.exists()
    np.savez_compressed(args.output, **arrays)
    atomic(args.output.with_suffix('.json'), dict(affinity=sorted(os.sched_getaffinity(0)),
        backend=jax.default_backend(), single_observation_calls_per_model=64,
        source_sha256=sha(__file__), arrays_sha256=sha(args.output), evidence=evidence,
        scope='Frozen numerical probe; no new training or environment episodes.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
