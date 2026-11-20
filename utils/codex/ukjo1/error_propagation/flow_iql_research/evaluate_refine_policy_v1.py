"""Frozen-checkpoint local action-refinement screen; primary files untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core, policy_core, benchmark_core, selection_v2
from utils.flow_iql_research import train as primary
from utils.flow_iql_research import refine_policy_v1 as refine
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(); assert not args.output.exists()
    conf = json.loads((args.run/'config.json').read_text()); ident = conf['identity']
    for source, digest in conf['source_sha256'].items():
        assert sha(source) == digest, source
    cfg = dict(ident['config'])
    for field in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    module = benchmark_core if ident['kind'] == 'fql' else policy_core
    env = ogbench.make_env_and_datasets(ident['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = module.create(ident['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                             jnp.zeros((1, env.action_space.shape[-1])), cfg, ident['kind'])
    with args.checkpoint.open('rb') as file:
        saved = pickle.load(file)
    state = flax.serialization.from_state_dict(template, saved['state'])
    assert int(state.q_updates) == saved['step']
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    statistics = []
    if args.epsilon == 0.:
        policy = selection_v2.make_policy(state.agent, 'max', args.samples, 'behavior')
    else:
        compiled = refine.make_policy(state.agent, args.samples, args.epsilon)
        def policy(params, observation, key):
            action, metrics = compiled(params, observation, key)
            metrics = np.asarray(metrics)
            assert np.isfinite(metrics).all()
            assert metrics[4] <= args.epsilon+2e-6, metrics[4]
            assert metrics[3] >= -2e-5, metrics[3]
            statistics.append(metrics.astype(np.float64))
            return action
    result = primary.evaluate(state, env, policy, saved['step'], args.episodes)
    comparison = None
    if args.reference is not None:
        assert args.epsilon == 0.
        reference = json.loads(args.reference.read_text())
        assert reference['checkpoint_sha256'] == sha(args.checkpoint)
        assert reference['selection_samples'] == args.samples
        comparison = {name: result[name] == reference[name]
                      for name in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='frozen_one_step_value_threshold_action_projection', time=time.time(),
        run=args.run.name, environment=ident['environment'], training_seed=ident['seed'],
        selection_samples=args.samples, max_action_displacement_RMS=args.epsilon,
        proposal_source='behavior', checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        config_sha256=sha(args.run/'config.json'), evaluator_source_sha256=sha(__file__),
        policy_source_sha256=sha(refine.__file__), selection_source_sha256=sha(selection_v2.__file__),
        primary_evaluation_helper_sha256=sha(primary.__file__), reset_metadata=reset,
        zero_correction_reference_comparison=comparison, primary_evaluations_modified=False,
        refinement_statistics=dict(zip(refine.STATISTICS, np.mean(statistics, axis=0).tolist())) if statistics else None,
        limits=['One linearized minimum-distance step; the nonlinear value constraint need not be reached.',
                'The RMS cap and acceptance safeguard constrain fitted-model extrapolation, not true Q error.',
                'These frozen exploratory screens are not a nine-task or independent-seed result.'])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2, allow_nan=False)); tmp.replace(args.output)
    print(json.dumps(dict(run=args.run.name, step=saved['step'], epsilon=args.epsilon,
                         success=result['success_rate'], comparison=comparison)), flush=True)


if __name__ == '__main__':
    main()
