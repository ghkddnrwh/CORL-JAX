"""CPU feasibility: deterministic fixed32 teacher -> state-only fast actors.

The original lean critic/flow is frozen. All observations come from the
offline replay, and all learning precedes evaluation. Direct regression and
two shared-trunk routed regressions use identical data and minibatch keys.
This is NOT a joint learner or evidence for the registered nine-task goal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

# Import before optax: the existing environment needs its legacy SciPy shim.
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import routed_fixed_actor_v1 as routed
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-case', type=Path, required=True)
    parser.add_argument('--fixed-teacher-case', type=Path, required=True)
    parser.add_argument('--scaling-case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path = args.study_case/'result.json'
    parent = json.loads(parent_path.read_text())
    fixed_path = args.fixed_teacher_case/'result.json'
    fixed = json.loads(fixed_path.read_text())
    assert parent['status'] == fixed['status'] == 'complete'
    assert parent['teacher_checkpoint_sha256'] == fixed['teacher_checkpoint_sha256']
    cache = args.study_case/'frozen_replay_pairs_v1.npz'
    assert sha(cache) == parent['pair_cache_sha256']
    with np.load(cache) as archive:
        old_obs = archive['observations'].copy()
        old_indices = archive['selected_replay_indices'].copy()[:65536]
        dim = archive['targets'].shape[-1]
        assert int(archive['train_size']) == 65536
    train = 524288
    checkpoint = Path(parent['teacher_checkpoint'])
    assert sha(checkpoint) == parent['teacher_checkpoint_sha256']
    conf = json.loads((Path(parent['teacher_run'])/'config.json').read_text())
    for p, digest in conf['source_sha256'].items():
        assert sha(p) == digest, p
    cfg = dict(conf['identity']['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    template = flat.create(conf['identity']['seed'], jnp.asarray(old_obs[:1]),
                           jnp.zeros((1, dim)), cfg, conf['identity']['recipe'])
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    teacher = flax.serialization.from_state_dict(template, saved['state'])
    assert int(teacher.base.q_updates) == int(teacher.actor_updates) == 1000000
    assert int(teacher.base.v_updates) == 0
    bank_path = args.fixed_teacher_case/'fixed_noise_bank32_v1.npy'
    assert sha(bank_path) == fixed['bank32_sha256']
    bank = jnp.asarray(np.load(bank_path))
    expected = jax.random.normal(jax.random.fold_in(
        jax.random.PRNGKey(conf['identity']['seed']), 49501), (32, dim))
    assert np.array_equal(np.asarray(bank), np.asarray(expected))
    result = dict(
        time=time.time(), status='preparing_replay', environment=parent['environment'],
        teacher_checkpoint=str(checkpoint), teacher_checkpoint_sha256=sha(checkpoint),
        parent_result=str(parent_path), parent_result_sha256=sha(parent_path),
        original_cache=str(cache), original_cache_sha256=sha(cache),
        fixed_teacher_result=str(fixed_path), fixed_teacher_result_sha256=sha(fixed_path),
        fixed_noise_bank_sha256=sha(bank_path),
        fixed32_teacher_CPU50=fixed['evaluations']['32']['success'],
        stochastic_best9_teacher_CPU50=fixed['original_stochastic_best9_CPU50'],
        training_states=train, heldout_states=4096, replay_only=True,
        actor_initialization_seed=34210, minibatch_seed=34211,
        methods=['direct_MSE', 'routed_CE_a01', 'routed_CE_a1'],
        Adam_per_method=100000, batch_size=256, learning_rate=.0003,
        hidden_dims=[256]*4, route_classes=32,
        routed_loss='mean MSE(all32raw heads,32Flowactions) + alpha*CE(logits,best32index)',
        direct_loss='mean MSE(rawsingleaction,best32action)',
        teacher_or_GPU_updates=0, evaluation_data_used_for_learning=False,
        source_sha256={str(Path(p).resolve()): sha(p) for p in
                       [__file__, routed.__file__, *flat.source_paths()]}, actors={},
        scope='Private frozen-teacher CPU524K replay/100Kactor diagnosis. '
              'No joint1M, last3/nine-task/fresh-seed or isolated speed claim.',
    )
    write(args.output/'status.json', result)
    if args.scaling_case:
        scaling_path = args.scaling_case/'result.json'
        scaling = json.loads(scaling_path.read_text())
        large_cache = args.scaling_case/'nested_replay_cache_v1.npz'
        assert sha(large_cache) == scaling['nested_cache_sha256']
        assert scaling['original_cache_sha256'] == sha(cache)
        assert scaling['teacher_checkpoint_sha256'] == sha(checkpoint)
        with np.load(large_cache) as archive:
            obs = archive['observations'].copy()
            indices = archive['training_indices'].copy()
            assert int(archive['train_size']) == train
        result['reused_observation_cache_sha256'] = sha(large_cache)
    else:
        env, data, _ = ogbench.make_env_and_datasets(parent['environment'])
        env.close()
        seen = {hashlib.sha256(row.tobytes()).digest() for row in old_obs}
        extra, indices_extra = [], []
        random = np.random.RandomState(49311)
        attempted = 0
        while len(extra) < train-65536:
            candidate = random.choice(len(data['actions']), 8192, replace=False)
            attempted += len(candidate)
            assert attempted < 100*train
            for index in candidate:
                row = np.asarray(data['next_observations'][index], np.float32)
                key = hashlib.sha256(row.tobytes()).digest()
                if key in seen:
                    continue
                seen.add(key)
                extra.append(row.copy())
                indices_extra.append(int(index))
                if len(extra) == train-65536:
                    break
        del data, seen
        obs = np.concatenate([old_obs[:65536], np.stack(extra), old_obs[65536:]])
        indices = np.concatenate([old_indices, np.asarray(indices_extra)])
    assert len(obs) == train+4096
    assert np.array_equal(obs[:65536], old_obs[:65536])
    assert np.array_equal(obs[train:], old_obs[65536:])
    held_hashes = {hashlib.sha256(row.tobytes()).digest() for row in obs[train:]}
    assert not any(hashlib.sha256(row.tobytes()).digest() in held_hashes
                   for row in obs[:train])
    agent = teacher.base.agent

    @jax.jit
    def label(observations):
        broad = jnp.broadcast_to(observations, (32, *observations.shape))
        noise = jnp.broadcast_to(bank[:, None], (32, len(observations), dim))
        actions = flat.lean.flow(agent, broad, noise, 10)
        q = agent.network.select('critic')(broad, actions=actions).min(0)
        return jnp.moveaxis(actions, 0, 1), jnp.moveaxis(q, 0, 1)

    result['status'] = 'labeling_fixed32'
    write(args.output/'status.json', result)
    actions, scores = [], []
    for chunk, offset in enumerate(range(0, len(obs), 256)):
        a, q = label(jnp.asarray(obs[offset:offset+256]))
        actions.append(np.asarray(a)); scores.append(np.asarray(q))
        if chunk % 128 == 0:
            print(json.dumps(dict(stage='fixed32_labels', states=offset+len(a))), flush=True)
    actions = np.concatenate(actions); scores = np.concatenate(scores)
    assert actions.shape == (train+4096, 32, dim)
    assert np.isfinite(actions).all() and np.isfinite(scores).all()
    selected = scores.argmax(-1)
    targets = actions[np.arange(len(obs)), selected]
    labeled_cache = args.output/'fixed32_replay_cache_v1.npz'
    np.savez_compressed(labeled_cache, observations=obs, actions=actions,
                        scores=scores, best_index=selected, targets=targets,
                        training_indices=indices, train_size=train)
    result.update(status='fitting', labeled_cache_sha256=sha(labeled_cache),
                  byte_exact_heldout_training_overlap=False,
                  labeling_states=len(obs), teacher_flow_actions_per_state=32)
    write(args.output/'status.json', result)
    observations = jnp.asarray(obs)
    all_actions = jnp.asarray(actions)
    targets = jnp.asarray(targets)
    selected = jnp.asarray(selected)
    shared_initial = None
    for method in result['methods']:
        direct = method == 'direct_MSE'
        alpha = .1 if method.endswith('a01') else 1.
        definition = flat.actor_definition(dict(hidden_dims=[256]*4,
                      action_dim=dim, layer_norm=False)) if direct else routed.RoutedFixedActor((256,)*4, dim, 32)
        call_args = (observations[:1], jnp.zeros((1, dim))) if direct else (observations[:1],)
        initial = definition.init(jax.random.PRNGKey(34210), *call_args)['params']
        if not direct:
            if shared_initial is None:
                shared_initial = initial
            else:
                assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                           zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(shared_initial)))
        actor = TrainState.create(definition, initial, optax.adam(.0003))

        def prediction(parameters, x):
            if direct:
                raw = definition.apply({'params': parameters}, x,
                                       jnp.zeros((*x.shape[:-1], dim)))
                return dict(raw=raw)
            return definition.apply({'params': parameters}, x, return_components=True)

        def fit(actor, key, steps):
            def update(carry, _):
                actor, key = carry
                key, index_key, _ = jax.random.split(key, 3)
                idx = jax.random.randint(index_key, (256,), 0, train)

                def loss(parameters):
                    p = prediction(parameters, observations[idx])
                    if direct:
                        total = jnp.mean((p['raw']-targets[idx])**2)
                    else:
                        mse = jnp.mean((p['means']-all_actions[idx])**2)
                        ce = optax.softmax_cross_entropy_with_integer_labels(
                            p['logits'], selected[idx]).mean()
                        total = mse+alpha*ce
                    return total, dict(loss=total)

                actor, _ = actor.apply_loss_fn(loss)
                return (actor, key), None

            return jax.lax.scan(update, (actor, key), None, length=steps)[0]

        fit = jax.jit(fit, static_argnums=2)
        snapshots = []
        key = jax.random.PRNGKey(34211)
        previous = 0
        for count in (0, 1000, 5000, 20000, 50000, 100000):
            if count:
                actor, key = fit(actor, key, count-previous)
            assert int(actor.opt_state[0].count) == count
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(actor.params))
            p = prediction(actor.params, observations[train:])
            row = dict(actual_Adam=count)
            if direct:
                raw = p['raw']
            else:
                route = p['logits'].argmax(-1)
                raw = p['means'][jnp.arange(4096), route]
                row.update(route_accuracy=float(jnp.mean(route == selected[train:])),
                           all_head_MSE=float(jnp.mean((p['means']-all_actions[train:])**2)),
                           route_CE=float(optax.softmax_cross_entropy_with_integer_labels(
                               p['logits'], selected[train:]).mean()))
                if count == 0:
                    selected_action = definition.apply({'params': actor.params}, observations[train:])
                    assert np.max(np.abs(np.asarray(selected_action-jnp.clip(raw,-1,1)))) < 1e-6
            row.update(heldout_selected_action_MSE=float(jnp.mean((jnp.clip(raw,-1,1)-targets[train:])**2)),
                       elapsed_seconds=time.time()-result['time'])
            snapshots.append(row)
            result['actors'][method] = dict(status='fitting', snapshots=snapshots)
            write(args.output/'status.json', result)
            print(json.dumps(dict(method=method, **row)), flush=True)
            previous = count
        params = flax.serialization.to_state_dict(jax.device_get(actor.params))
        metadata = dict(schema='single_actor_deployment_v1', observation_dim=obs.shape[-1],
                        action_dim=dim, hidden_dims=[256]*4, layer_norm=False,
                        input='state', noise_dim=0, requires_critic=False,
                        requires_BC_flow=False, flow_integration_steps=0,
                        generated_candidate_actions=0, actor_forward_passes=1,
                        realized_action_samples=1, components=1 if direct else 32,
                        architecture='state_only_MLP' if direct else 'state_only_route_then_one_linear_head')
        artifact = args.output/(method+'_private_actor_100000.pkl')
        with artifact.open('xb') as stream:
            pickle.dump(dict(schema='private_frozen_teacher_student_v1', method=method,
                             training_mode='CPU_fixed_teacher_diagnostic_only',
                             actor_params=params, metadata=metadata,
                             actor_Adam_updates=100000, teacher_optimizer_updates_performed=0,
                             teacher_checkpoint_sha256=sha(checkpoint),
                             source_sha256=result['source_sha256']), stream, pickle.HIGHEST_PROTOCOL)
        if direct:
            def policy(p, x, key):
                return jnp.clip(definition.apply({'params': p}, x, jnp.zeros((*x.shape[:-1],dim))), -1, 1)
            policy = jax.jit(policy, backend='cpu')
        else:
            policy = routed.make_policy(metadata)
        namespace = SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env = ogbench.make_env_and_datasets(parent['environment'], env_only=True)
        reset = prepare_seeded_action_space(env, 42)
        evaluated = evaluate(namespace, env, policy, 100000, 50)
        env.close()
        evaluated.update(time=time.time(), protocol='private_fixed_teacher_student_CPU50_v1',
                         actor_artifact_sha256=sha(artifact), teacher_checkpoint_sha256=sha(checkpoint),
                         reset_metadata=reset, no_optimizer_updates=True,
                         counts_toward_joint_or_goal=False)
        ep = args.output/(method+'_100K_CPU50_v1.json')
        write(ep, evaluated)
        result['actors'][method] = dict(status='complete', snapshots=snapshots,
            success=evaluated['success_rate'], evaluation_path=str(ep), evaluation_sha256=sha(ep),
            private_actor_path=str(artifact), private_actor_sha256=sha(artifact), metadata=metadata,
            parameter_count=sum(v.size for v in jax.tree_util.tree_leaves(initial)))
        write(args.output/'status.json', result)
        print(json.dumps(dict(method=method, success=evaluated['success_rate'])), flush=True)
    assert sha(checkpoint) == parent['teacher_checkpoint_sha256']
    assert sha(cache) == parent['pair_cache_sha256']
    result.update(status='complete', completed=time.time())
    write(args.output/'result.json', result)
    write(args.output/'status.json', result)


if __name__ == '__main__':
    main()
