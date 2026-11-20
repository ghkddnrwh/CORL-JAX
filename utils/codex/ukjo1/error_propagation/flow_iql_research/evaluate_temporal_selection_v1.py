"""Frozen, causal-in-time temporal-critic policy variant; no primary edits."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, selection_v2, temporal_selection_v1
from utils.flow_iql_research import train as primary
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
    parser.add_argument('--member-checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args(); assert not args.output.exists()
    conf = json.loads((args.run/'config.json').read_text()); ident = conf['identity']
    assert ident['kind'] == 'ifql'
    for source, digest in conf['source_sha256'].items():
        assert sha(source) == digest, source
    cfg = dict(ident['config'])
    for name in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[name] = tuple(cfg[name])
    env = ogbench.make_env_and_datasets(ident['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = policy_core.create(ident['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                                   jnp.zeros((1, env.action_space.shape[-1])), cfg, 'ifql')
    def load(path):
        with path.open('rb') as file:
            saved = pickle.load(file)
        assert json.loads(json.dumps(saved['identity'])) == ident
        state = flax.serialization.from_state_dict(template, saved['state'])
        assert int(state.q_updates) == saved['step']
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        return saved['step'], state
    step, state = load(args.checkpoint)
    members, sources = [], []
    for path in args.member_checkpoints:
        member_step, member = load(path)
        assert member_step <= step, 'No future critic checkpoints'
        members.append(member.agent.network.params['modules_critic'])
        sources.append(dict(step=member_step, checkpoint=str(path), sha256=sha(path)))
    assert sources[-1]['step'] == step
    assert sources[-1]['sha256'] == sha(args.checkpoint)
    if len(sources) == 3:
        assert [x['step'] for x in sources] == [step-200000, step-100000, step]
    else:
        assert len(sources) == 1
    metrics = []
    if len(members) == 1:
        policy = selection_v2.make_policy(state.agent, 'max', args.samples, 'behavior')
    else:
        compiled = temporal_selection_v1.make_selector(state.agent, args.samples)
        def policy(params, observation, key):
            action, q = compiled(params, tuple(members), observation, key)
            q = np.asarray(q, np.float64)
            assert np.isfinite(q).all()
            centered = q-q.mean(1, keepdims=True)
            metrics.append([float(np.mean(np.argmax(q, axis=1) == np.argmax(q[-1]))),
                float(np.sqrt(np.mean(np.var(centered, axis=0)))), float(np.std(q.mean(1)))])
            return action
    result = primary.evaluate(state, env, policy, step, args.episodes)
    comparison = None
    if args.reference is not None:
        assert len(sources) == 1
        reference = json.loads(args.reference.read_text())
        assert reference['checkpoint_sha256'] == sha(args.checkpoint)
        assert reference['selection_samples'] == args.samples
        comparison = {k: result[k] == reference[k]
                      for k in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='frozen_temporal_critic_mean_selection_variant', time=time.time(),
        environment=ident['environment'], run=args.run.name, training_seed=ident['seed'],
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        critic_members=sources, selection_samples=args.samples,
        selection_rule='temporal_mean_min_heads',
        critic_combination='mean_of_per_member_minimum_twin_Q',
        actor_training_step=step, config_sha256=sha(args.run/'config.json'),
        evaluator_source_sha256=sha(__file__), selector_source_sha256=sha(temporal_selection_v1.__file__),
        proposal_source_sha256=sha(selection_v2.__file__),
        primary_evaluation_helper_sha256=sha(primary.__file__), reset_metadata=reset,
        single_member_reference_comparison=comparison,
        temporal_selection_diagnostics=dict(zip(
            ['member_argmax_agreement_with_newest_mean', 'action_centered_member_std_RMS_mean',
             'state_mean_member_Q_std_mean'], np.mean(metrics, axis=0).tolist())) if metrics else None,
        primary_evaluations_modified=False,
        limits=['Every critic member was trained no later than the evaluated checkpoint; actor is always the current BC flow.',
                'Three temporally correlated critics are not independent training seeds.',
                'Averaging may retain or increase systematic error; no true bias-reduction guarantee.',
                'Q/V training is unchanged; this tests only policy extraction and adds inference cost.'])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2, allow_nan=False)); tmp.replace(args.output)
    print(json.dumps(dict(step=step, member_steps=[x['step'] for x in sources],
                         success=result['success_rate'], reference=comparison)), flush=True)


if __name__ == '__main__':
    main()
