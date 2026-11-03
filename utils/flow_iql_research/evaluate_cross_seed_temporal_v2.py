"""Cross training-seed BC/critic parameters for frozen diagnostic inference.

This uses parameter sets only for frozen inference. It never assembles or
resumes a mixed optimizer state and never rewrites a primary score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, temporal_selection_v1
from utils.flow_iql_research import train as primary
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(run, step):
    config = json.loads((run/'config.json').read_text())
    for source, digest in config['source_sha256'].items():
        assert sha(source) == digest, source
    path = run/f'checkpoint_{step}.pkl'
    if not path.exists():
        path = run.parent.parent/'analysis/checkpoint_archive'/run.name/f'checkpoint_{step}.pkl'
    with path.open('rb') as file:
        saved = pickle.load(file)
    assert saved['step'] == int(saved['state']['q_updates']) == step
    assert json.loads(json.dumps(saved['identity'])) == config['identity']
    assert saved['source_sha256'] == config['source_sha256']
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == step
    assert int(saved['state']['v_opt']['0']['count']) == int(saved['state']['v_updates'])
    return config, saved, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic-run', type=Path, required=True)
    parser.add_argument('--actor-run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--members', type=int, choices=(1, 3), required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--allow-cross-training-seed', action='store_true')
    args = parser.parse_args(); assert not args.output.exists()
    ac, actor_saved, actor_path = read(args.actor_run, args.step)
    qc, current_q, qpath = read(args.critic_run, args.step)
    ai, qi = ac['identity'], qc['identity']
    assert ai['kind'] == qi['kind'] == 'ifql'
    assert ai['environment'] == qi['environment']
    assert ai['profile'] == qi['profile'] == 'bounded_v2'
    cross_seed = ai['seed'] != qi['seed']
    assert not cross_seed or args.allow_cross_training_seed
    assert {k:v for k,v in ai['config'].items() if k != 'num_samples'} == {k:v for k,v in qi['config'].items() if k != 'num_samples'}
    for key in ('discount', 'expectile', 'flow_steps', 'lr', 'batch_size', 'actor_hidden_dims',
                'value_hidden_dims', 'layer_norm', 'actor_layer_norm'):
        assert ai['config'][key] == qi['config'][key], key
    replay_equal = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(actor_saved['replay_rng'], current_q['replay_rng']))
    if not cross_seed:
        assert replay_equal
    cfg = dict(ai['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env = ogbench.make_env_and_datasets(ai['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = policy_core.create(ai['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                                   jnp.zeros((1, env.action_space.shape[-1])), cfg, 'ifql')
    # The actor state supplies the definition and behavior parameters only. The
    # selector below obtains every deployed critic from the separate tuple.
    state = flax.serialization.from_state_dict(template, actor_saved['state'])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    steps = [args.step] if args.members == 1 else [args.step-200000, args.step-100000, args.step]
    members, evidence = [], []
    for step in steps:
        config, saved, path = read(args.critic_run, step)
        assert config == qc
        vpq = 2 if 'v2' in qi['profile'] else 1
        assert int(saved['state']['v_updates']) == vpq*step
        member = saved['state']['agent']['network']['params']['modules_critic']
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(member))
        members.append(jax.device_put(member, jax.devices('cpu')[0]))
        evidence.append(dict(step=step, checkpoint=str(path), sha256=sha(path),
                             Q_updates=step, V_updates=int(saved['state']['v_updates'])))
    selector = temporal_selection_v1.make_selector(state.agent, args.samples)
    def policy(params, observation, key):
        action, _ = selector(params, tuple(members), observation, key)
        return action
    result = primary.evaluate(state, env, policy, args.step, 50)
    comparison = None
    if args.reference is not None:
        assert args.actor_run.resolve() == args.critic_run.resolve()
        reference = json.loads(args.reference.read_text())
        assert reference['checkpoint_sha256'] == sha(actor_path)
        assert reference['selection_samples'] == args.samples
        comparison = {key: result[key] == reference[key]
                      for key in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='frozen_cross_training_seed_actor_critic_diagnostic', time=time.time(),
        environment=ai['environment'], actor_training_seed=ai['seed'], critic_training_seed=qi['seed'],
        cross_training_seed_intervention=cross_seed, replay_RNG_equal=replay_equal,
        critic_run=args.critic_run.name, actor_run=args.actor_run.name,
        critic_controller_profile=qi['profile'], critic_members=evidence,
        critic_combination='mean_of_per_member_minimum_twin_Q', selection_samples=args.samples,
        actor_training_step=args.step, actor_checkpoint_sha256=sha(actor_path),
        actor_config_sha256=sha(args.actor_run/'config.json'),
        critic_config_sha256=sha(args.critic_run/'config.json'),
        self_reference_episode_arrays_exact=comparison, evaluator_source_sha256=sha(__file__),
        selector_source_sha256=sha(temporal_selection_v1.__file__), reset_metadata=reset,
        primary_evaluations_modified=False, mixed_learning_state_saved_or_trained=False,
        limits=['The exact same current BC parameters are held fixed across critic controllers.',
                'This tests critic and behavior generator sources; replay equality is deliberately not required across training seeds.',
                'This is a frozen diagnostic selected after observing seed heterogeneity; it is not another training replicate or a registered performance candidate.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2, allow_nan=False)); temp.replace(args.output)
    print(json.dumps(dict(critic=args.critic_run.name, actor=args.actor_run.name,
                         step=args.step, members=args.members, success=result['success_rate'],
                         reference=comparison)), flush=True)


if __name__ == '__main__':
    main()
