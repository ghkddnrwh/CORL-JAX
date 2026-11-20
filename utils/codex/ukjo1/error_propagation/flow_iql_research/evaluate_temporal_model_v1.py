"""Evaluate one self-contained temporal model, including its old critics."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import temporal_core_v1 as core
from utils.flow_iql_research import train_temporal_policy_v1 as runner
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(); assert not args.output.exists()
    with args.model.open('rb') as file:
        saved = pickle.load(file)
    exported = saved.get('kind') == 'temporal_IFQL_deployment_bundle_v1'
    if exported:
        identity = saved['training_identity']; cfg = dict(saved['inference_config'])
        for path, digest in {**saved['training_source_sha256'], **saved['deployment_source_sha256']}.items():
            assert sha(path) == digest, path
    else:
        identity = saved['identity']; cfg = dict(identity['config'])
        assert identity['profile'] == 'bounded_v2_temporal3'
        for path, digest in saved['source_sha256'].items():
            assert sha(path) == digest, path
    assert identity['kind'] == 'ifql'
    for field in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    env = ogbench.make_env_and_datasets(identity['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = core.create(identity['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                           jnp.zeros((1, env.action_space.shape[-1])), cfg, 'ifql')
    state = flax.serialization.from_state_dict(template, saved['state'])
    assert int(state.q_updates) == saved['step']
    assert int(state.v_updates) == 2*saved['step']
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    core.validate_history(saved['critic_history'], state, saved['step'])
    policy, member_steps = core.make_policy(state.agent, saved['critic_history'], saved['step'])
    result = runner.evaluate(state, env, policy, saved['step'], args.episodes)
    if args.episodes != 50:
        result['protocol'] = 'CPU_smoke_nonbenchmark_v1'
    comparison = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        assert exported and args.episodes == reference['n_episodes'] == 50
        assert saved['source_checkpoints'][-1]['sha256'] == reference['checkpoint_sha256']
        assert [m['sha256'] for m in saved['source_checkpoints']] == [m['sha256'] for m in reference['critic_members']]
        comparison = {name: result[name] == reference[name]
                      for name in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='self_contained_temporal_model_evaluation', time=time.time(),
        environment=identity['environment'], training_seed=identity['seed'], model=str(args.model),
        model_sha256=sha(args.model), critic_member_steps=member_steps,
        selection_samples=cfg['num_samples'], reference_episode_arrays_exact=comparison,
        evaluator_source_sha256=sha(__file__), deployment_core_sha256=sha(core.__file__),
        reset_metadata=reset, primary_evaluations_modified=False,
        limits='A re-evaluation of one retained model, not a new training seed or extra benchmark replicate.')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    temp = args.output.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2, allow_nan=False)); temp.replace(args.output)
    print(json.dumps(dict(step=saved['step'], success=result['success_rate'],
                         members=member_steps, reference=comparison)), flush=True)


if __name__ == '__main__':
    main()
