"""Frozen 2x2 critic/behavior-flow checkpoint diagnostic, never resumed.

Only actor-flow parameters are swapped for inference. Optimizer moments are
not transplanted and this artificial combination is never serialized or trained.
Self combinations can be checked against full existing evaluation episodes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, selection_v2
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
    parser.add_argument('--critic-checkpoint', type=Path, required=True)
    parser.add_argument('--actor-checkpoint', type=Path, required=True)
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
    with args.critic_checkpoint.open('rb') as file:
        qs = pickle.load(file)
    with args.actor_checkpoint.open('rb') as file:
        acs = pickle.load(file)
    assert json.loads(json.dumps(qs['identity'])) == json.loads(json.dumps(acs['identity'])) == ident
    qstate = flax.serialization.from_state_dict(template, qs['state'])
    astate = flax.serialization.from_state_dict(template, acs['state'])
    assert int(qstate.q_updates) == qs['step'] and int(astate.q_updates) == acs['step']
    params = dict(qstate.agent.network.params)
    params['modules_actor_flow'] = astate.agent.network.params['modules_actor_flow']
    state = qstate.replace(agent=qstate.agent.replace(network=qstate.agent.network.replace(params=params)))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    result = primary.evaluate(state, env,
        selection_v2.make_policy(state.agent, 'max', args.samples, 'behavior'), qs['step'], args.episodes)
    comparison = None
    if args.reference is not None:
        assert qs['step'] == acs['step']
        reference = json.loads(args.reference.read_text())
        assert reference['checkpoint_sha256'] == sha(args.critic_checkpoint)
        assert reference['selection_samples'] == args.samples
        comparison = {k: result[k] == reference[k]
                      for k in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='frozen_cross_checkpoint_critic_behavior_actor_diagnostic', time=time.time(),
        environment=ident['environment'], run=args.run.name, training_seed=ident['seed'],
        critic_training_step=qs['step'], behavior_actor_training_step=acs['step'],
        critic_checkpoint=str(args.critic_checkpoint), actor_checkpoint=str(args.actor_checkpoint),
        critic_checkpoint_sha256=sha(args.critic_checkpoint), actor_checkpoint_sha256=sha(args.actor_checkpoint),
        selection_samples=args.samples, config_sha256=sha(args.run/'config.json'),
        evaluator_source_sha256=sha(__file__), selection_source_sha256=sha(selection_v2.__file__),
        primary_evaluation_helper_sha256=sha(primary.__file__), reset_metadata=reset,
        self_combination_episode_reference_comparison=comparison,
        primary_evaluations_modified=False, artificial_state_saved_or_trained=False,
        limits=['This is an inference-only combination of actual retained parameter sets at different training steps.',
                'Only actor-flow parameters are swapped; Q, target-Q and V come from the critic checkpoint.',
                'No claim is made that optimizer moments or clocks form a trained full state.',
                'This paired diagnostic is not a new benchmark run or independent-seed confirmation.'])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    temp = args.output.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2, allow_nan=False)); temp.replace(args.output)
    print(json.dumps(dict(critic_step=qs['step'], actor_step=acs['step'],
                         success=result['success_rate'], reference=comparison)), flush=True)


if __name__ == '__main__':
    main()
