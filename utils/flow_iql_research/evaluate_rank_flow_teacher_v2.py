"""Evaluate the same rank-flow model's student or best9 BC teacher policy."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import rank_flow_core_v1 as core
from utils.flow_iql_research import train_rank_flow_v2 as primary
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--mode', choices=('teacher', 'student'), default='teacher')
    parser.add_argument('--episodes', type=int, choices=(1, 50), default=50)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Explicit immutable retained/archive snapshot')
    args = parser.parse_args(); assert not args.output.exists()
    config = json.loads((args.run/'config.json').read_text()); identity = config['identity']
    assert identity['kind'] == 'rank_flow'
    for source, digest in config['source_sha256'].items():
        assert sha(source) == digest, source
    checkpoint = args.checkpoint or args.run/f'checkpoint_{args.step}.pkl'
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert json.loads(json.dumps(saved['identity'])) == identity
    assert saved['source_sha256'] == config['source_sha256']
    assert saved['step'] == int(saved['state']['q_updates']) == args.step
    assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2*args.step
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == args.step
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env = ogbench.make_env_and_datasets(identity['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = core.create(identity['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                           jnp.zeros((1, env.action_space.shape[-1])), cfg)
    state = flax.serialization.from_state_dict(template, saved['state'])

    def policy(params, observations, key):
        agent = state.agent.replace(network=state.agent.network.replace(params=params))
        if args.mode == 'student':
            return agent.sample_actions(observations, key, 0.)
        action_key, _ = jax.random.split(key)
        selected, _, _ = agent.selected_teacher_targets(observations, action_key)
        return selected

    result = primary.evaluate(state, env, jax.jit(policy, backend='cpu'), args.step, args.episodes)
    comparison = None
    if args.reference is not None:
        assert args.mode == 'student'
        matches = [json.loads(line) for line in args.reference.read_text().splitlines()
                   if json.loads(line)['step'] == args.step]
        assert len(matches) == 1
        comparison = {key:result[key] == matches[0][key]
                      for key in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values())
    result.update(time=time.time(), kind='auxiliary_same_model_policy_comparison',
        policy_mode=args.mode, environment=identity['environment'], training_seed=identity['seed'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), config_sha256=sha(args.run/'config.json'),
        evaluator_source_sha256=sha(__file__), reset_metadata=reset,
        self_reference_episode_arrays_exact=comparison, primary_evaluations_modified=False,
        limits=['Teacher and student use exactly the same retained model and current BC/Q parameters.',
                'This is an auxiliary deployed-policy comparison, not another training seed.',
                'One-episode short-run checks are excluded from benchmark scores.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)); temporary.replace(args.output)
    print(json.dumps(dict(mode=args.mode, step=args.step, success=result['success_rate'], self_reference=comparison)), flush=True)


if __name__ == '__main__':
    main()
