"""Export the fixed independent-critic policy without optimizer/replay states."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import policy_core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.rank_mean_bundle_v1 import sha
import jax
import numpy as np
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--step', type=int, choices=(800000, 900000, 1000000), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    verifier = Verifier()
    members, params, source_hashes = [], [], {}
    for run in args.critic_runs:
        assert (run / 'completed.json').exists() and not (run / 'stopped.json').exists()
        conf, saved, path = read(run, args.step)
        identity = conf['identity']
        assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
        assert identity['steps'] == 1000000 and identity['episodes'] == 50
        assert identity['config']['discount'] == .999 and identity['config']['expectile'] == .9
        done = json.loads((run / 'completed.json').read_text())
        assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
        digest = verifier.checkpoint(path, conf, args.step)
        for source, expected in conf['source_sha256'].items():
            assert verifier.digest(source) == expected
            relative = str(Path(source).resolve().relative_to(Path.cwd().resolve()))
            assert relative not in source_hashes or source_hashes[relative] == expected
            source_hashes[relative] = expected
        network = saved['state']['agent']['network']['params']
        params.append(network if not params else network['modules_critic'])
        members.append(dict(run=run.name, identity=identity, checkpoint=str(path), checkpoint_sha256=digest,
            config_sha256=verifier.digest(run / 'config.json'), actual_Q_Adam_count=args.step,
            actual_V_Adam_count=2 * args.step))
    first = members[0]['identity']
    assert len({m['identity']['seed'] for m in members}) == 3
    for member in members[1:]:
        identity = member['identity']
        assert identity['environment'] == first['environment']
        assert {k: v for k, v in identity['config'].items() if k != 'num_samples'} == {
            k: v for k, v in first['config'].items() if k != 'num_samples'}
    env = ogbench.make_env_and_datasets(first['environment'], env_only=True)
    observation_dim, action_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
    env.close()
    for source in ('rank_mean_bundle_v1.py', 'export_rank_mean_bundle_v1.py', 'evaluate_rank_mean_bundle_v1.py',
                   'independent_critic_policy_v2.py', 'independent_critic_policy_v1.py', 'selection_v2.py',
                   'evaluate_cross_seed_temporal_v2.py', 'audit_temporal_cohort_v1.py'):
        path = 'utils/flow_iql_research/' + source
        source_hashes[path] = sha(path)
    # Parameters remain CPU NumPy arrays; no optimizer/PRNG/replay state is copied.
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params))
    bundle = dict(kind='independent_critic_rank_mean_deployment_v1', time=time.time(),
        resumable_training_checkpoint=False, step=args.step, environment=first['environment'],
        observation_dim=observation_dim, action_dim=action_dim, members=members,
        first_network_params=params[0], other_critic_params=params[1:], source_sha256=source_hashes,
        recipe=dict(rule='mean_rank', samples=9, learners=3, member_twin_head_reduction='min',
            common_proposal_actor='first_member_BC', tie_rule='first_proposal'),
        cost=dict(total_Q_updates=3 * args.step, total_V_updates=6 * args.step),
        note='Deployment export of existing parameters. No learning, new training seed, checkpoint selection or benchmark replicate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), bytes=args.output.stat().st_size,
        seeds=[m['identity']['seed'] for m in members], step=args.step)), flush=True)


if __name__ == '__main__':
    main()
