"""Read-only finite BC-sample proximity and predicted-Q diagnostics on CPU.

These are sampling proxies, not behavior-density estimates, true-return errors,
or proofs that a policy action is out of distribution.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--step', type=int, default=300000)
    parser.add_argument('--states', type=int, default=64)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and args.states > 0 and args.samples >= 2
    assert all(device.platform == 'cpu' for device in jax.devices())
    result = dict(time=time.time(), status='running', rows=[],
                  source_sha256=sha(__file__), core_source_sha256=sha(core.__file__),
                  states=args.states, BC_reference_samples=args.samples,
                  dataset_index_seed=21931, BC_reference_noise_seed=51931,
                  policy_noise_seed=51932, CPU_affinity=sorted(os.sched_getaffinity(0)),
                  optimizer_updates_performed=0,
                  scope='Fixed offline observations; independent finite BC reference bank; '
                        'actual actor export and same-state best9 teacher. '
                        'No density, true-Q-error, or OOD proof; not rollout performance.')
    references = {}
    for run in args.runs:
        conf = json.loads((run/'config.json').read_text())
        for path, digest in conf['source_sha256'].items():
            assert sha(path) == digest, path
        cp = run/f'checkpoint_{args.step}.pkl'
        artifact_path = run/f'actor_{args.step}.pkl'
        cp_hash, artifact_hash = sha(cp), sha(artifact_path)
        with cp.open('rb') as stream:
            saved = pickle.load(stream)
        with artifact_path.open('rb') as stream:
            artifact = pickle.load(stream)
        identity = conf['identity']
        assert saved['identity'] == artifact['identity'] == identity
        assert saved['step'] == artifact['step'] == args.step
        assert artifact['checkpoint_sha256'] == cp_hash
        assert saved['source_sha256'] == artifact['source_sha256'] == conf['source_sha256']
        assert artifact['config_sha256'] == sha(run/'config.json')
        assert artifact['actor_updates'] == artifact['Q_updates'] == args.step
        assert all(np.isfinite(np.asarray(x)).all()
                   for x in jax.tree_util.tree_leaves(artifact['actor_params']))
        cfg = dict(identity['config'])
        for key in ('actor_hidden_dims', 'value_hidden_dims'):
            cfg[key] = tuple(cfg[key])
        metadata = artifact['metadata']
        template = core.create(identity['seed'], jnp.zeros((1, metadata['observation_dim'])),
                               jnp.zeros((1, metadata['action_dim'])), cfg, identity['recipe'])
        state = flax.serialization.from_state_dict(template, saved['state'])
        assert int(state.base.q_updates) == int(state.actor_updates) == args.step
        assert int(state.base.v_updates) == 0
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        # Cache identical critics/BCs across actor-only alpha variants.
        signature = hashlib.sha256(flax.serialization.to_bytes(state.base.agent.network.params))
        base_cfg = {k: v for k, v in identity['config'].items()
                    if k != 'onestep_specification'}
        signature.update(json.dumps([identity['environment'], base_cfg], sort_keys=True).encode())
        group = signature.hexdigest()
        if group not in references:
            env, data, _ = ogbench.make_env_and_datasets(identity['environment'])
            env.close()
            indices = np.random.RandomState(21931).randint(len(data['actions']), size=args.states)
            observations = jnp.asarray(data['observations'][indices], jnp.float32)
            del data
            def reference_values(agent, obs):
                ob, noise = core.lean.bank(agent, obs, jax.random.PRNGKey(51931), args.samples)
                actions = core.lean.flow(agent, ob, noise, 10)
                heads = agent.network.select('critic')(ob, actions=actions)
                teacher_ob, teacher_noise = core.lean.bank(agent, obs, jax.random.PRNGKey(51932), 9)
                proposals = core.lean.flow(agent, teacher_ob, teacher_noise, 10)
                teacher_heads = agent.network.select('critic')(teacher_ob, actions=proposals)
                teacher = core.gather(proposals, teacher_heads.min(0).argmax(0))
                return actions, heads, teacher
            actions, heads, teacher = jax.device_get(jax.jit(reference_values)(state.base.agent, observations))
            references[group] = (observations, actions, heads, teacher)
        obs, behavior, behavior_heads, teacher = references[group]
        policy = core.make_actor_policy(metadata, 'cpu')
        actor = policy(artifact['actor_params'], obs, jax.random.PRNGKey(51932))
        critic = jax.jit(lambda agent, ob, action: agent.network.select('critic')(ob, actions=action))
        actor_heads = np.asarray(critic(state.base.agent, obs, actor))
        teacher_heads = np.asarray(critic(state.base.agent, obs, jnp.asarray(teacher)))
        actor = np.asarray(actor)
        row = dict(run=run.name, environment=identity['environment'], recipe=identity['recipe'],
                   training_seed=identity['seed'], step=args.step, checkpoint_sha256=cp_hash,
                   actor_artifact_sha256=artifact_hash, config_sha256=sha(run/'config.json'),
                   identical_base_reference_group=group)
        for label, action, q_heads in [('actor', actor, actor_heads),
                                        ('best9_teacher', teacher, teacher_heads)]:
            distance = np.mean((behavior - action[None])**2, axis=-1).min(0)
            row[label] = dict(nearest_independent_BC_action_MSE=summary(distance),
                              distance_over_BC_action_variance=summary(
                                  distance / np.maximum(behavior.var(0).mean(-1), 1e-6)),
                              predicted_min_Q=summary(q_heads.min(0)),
                              predicted_mean_Q=summary(q_heads.mean(0)),
                              predicted_min_Q_BC_percentile=summary(
                                  (behavior_heads.min(0) <= q_heads.min(0)[None]).mean(0)),
                              twin_Q_abs_disagreement=summary(np.abs(q_heads[0]-q_heads[1])))
        assert sha(cp) == cp_hash and sha(artifact_path) == artifact_hash
        result['rows'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(args.output)
        print(json.dumps(dict(run=run.name, group=group, diagnostics=row['actor'])), flush=True)
        del template, saved, artifact, state, policy, critic
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        gc.collect()
    result['status'] = 'complete'
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
