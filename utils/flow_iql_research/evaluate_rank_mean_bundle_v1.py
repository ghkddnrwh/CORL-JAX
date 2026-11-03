"""CPU evaluation of a deployment bundle, optionally exact against its reference."""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research.rank_mean_bundle_v1 import load, sha
from utils.flow_iql_research import train
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and args.episodes > 0
    bundle, state, policy = load(args.bundle)
    env = ogbench.make_env_and_datasets(bundle['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    result = train.evaluate(state, env, policy, bundle['step'], args.episodes)
    env.close()
    if args.episodes != 50:
        result['protocol'] = 'CPU_smoke_nonbenchmark_v1'
    exact = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        assert args.episodes == reference['n_episodes'] == 50
        assert reference['environment'] == bundle['environment'] and reference['step'] == bundle['step']
        assert reference['aggregation_rule'] == 'mean_rank' and reference['selection_samples'] == 9
        assert [m['checkpoint_sha256'] for m in bundle['members']] == [m['checkpoint_sha256'] for m in reference['critic_members']]
        assert bundle['members'][0]['checkpoint_sha256'] == reference['actor_checkpoint_sha256']
        exact = {field: result[field] == reference[field] for field in (
            'episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(exact.values()), exact
    result.update(kind='self_contained_independent_critic_rank_mean_evaluation', time=time.time(),
        environment=bundle['environment'], bundle=str(args.bundle), bundle_sha256=sha(args.bundle),
        source_sha256=sha(__file__), reset_metadata=reset, reference_episode_arrays_exact=exact,
        recipe=bundle['recipe'], learner_seeds=[m['identity']['seed'] for m in bundle['members']],
        cost=bundle['cost'], primary_evaluations_modified=False,
        note='Re-evaluation of retained parameters; not a new training replicate or additional benchmark observation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(args.output)
    print(json.dumps(dict(success=result['success_rate'], step=bundle['step'], reference=exact)), flush=True)


if __name__ == '__main__':
    main()
