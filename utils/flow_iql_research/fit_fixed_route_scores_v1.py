"""Frozen CPU control: distill smooth candidate score differences, not argmax CE.

The deployment graph is exactly the same route-then-single-head actor used
by the CE controls. These scores depend only on state; no Q(s,a) or realized
action bank is evaluated at deployment. No critic gradient is used in fitting.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import routed_fixed_actor_v1 as routed
from utils.flow_iql_research.fit_fixed_teacher_students_v1 import write
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    manifest = json.loads(args.manifest.read_text())
    for p, digest in manifest['source_sha256'].items():
        assert sha(p) == digest, p
    case = manifest['cases'][args.case]
    assert sha(case['cache']) == case['cache_sha256']
    assert sha(case['status_snapshot']) == case['status_snapshot_sha256']
    parent = json.loads(Path(case['status_snapshot']).read_text())
    args.output.mkdir(parents=True)
    with np.load(case['cache']) as z:
        observations = jnp.asarray(z['observations'])
        actions = jnp.asarray(z['actions'])
        scores = jnp.asarray(z['scores'])
        indices = jnp.asarray(z['best_index'])
        targets = jnp.asarray(z['targets'])
        train = int(z['train_size'])
    assert train == 524288 and len(observations) == train+4096
    assert actions.shape[1] == 32
    dim = actions.shape[-1]
    # Shift and positive scaling preserve the teacher's candidate ordering.
    # A unit floor prevents amplification of tiny, insignificant differences.
    centered = scores-scores.mean(-1, keepdims=True)
    rank_targets = centered/jnp.maximum(scores.std(-1, keepdims=True), 1.)
    assert np.array_equal(np.asarray(scores.argmax(-1)), np.asarray(rank_targets.argmax(-1)))
    result = dict(time=time.time(), status='fitting', environment=parent['environment'],
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        cache_sha256=case['cache_sha256'], teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
        fixed32_teacher_CPU50=parent['fixed32_teacher_CPU50'], training_states=train,
        heldout_states=4096, actor_initialization_seed=34210, minibatch_seed=34211,
        Adam_per_method=100000, batch_size=256, learning_rate=.0003,
        methods=['routed_rankMSE_a01', 'routed_rankMSE_a1'],
        score_target='(Qmin_i - mean_i Qmin)/max(std_i Qmin,1)',
        loss='mean(all raw action heads - fixedFlow32 actions)^2 + alpha*mean(route scores - centered normalized teacher Q)^2',
        same_architecture_and_initial_parameters_and_batches_as_CE_controls=True,
        teacher_queries_or_updates=0, registered_GPU_updates=0,
        evaluation_data_used_for_learning=False, source_sha256=manifest['source_sha256'], actors={},
        scope='Frozen teacher CPU100Kstudent feasibility only; not joint1M/last3/nine-task/fresh-seed/isolatedspeed.')
    write(args.output/'status.json', result)
    definition = routed.RoutedFixedActor((256,)*4, dim, 32)
    initial = definition.init(jax.random.PRNGKey(34210), observations[:1])['params']
    for method in result['methods']:
        alpha = .1 if method.endswith('a01') else 1.
        actor = TrainState.create(definition, initial, optax.adam(.0003))

        def fit(actor, key, steps):
            def update(carry, _):
                actor, key = carry
                key, index_key, _ = jax.random.split(key, 3)
                idx = jax.random.randint(index_key, (256,), 0, train)

                def loss(params):
                    p = actor(observations[idx], params=params, return_components=True)
                    action_mse = jnp.mean((p['means']-actions[idx])**2)
                    score_mse = jnp.mean((p['logits']-rank_targets[idx])**2)
                    total = action_mse+alpha*score_mse
                    return total, dict(loss=total)

                actor, _ = actor.apply_loss_fn(loss)
                return (actor, key), None

            return jax.lax.scan(update, (actor, key), None, length=steps)[0]

        fit = jax.jit(fit, static_argnums=2)
        key = jax.random.PRNGKey(34211)
        previous = 0
        snapshots = []
        for count in (0, 1000, 5000, 20000, 50000, 100000):
            if count:
                actor, key = fit(actor, key, count-previous)
            assert int(actor.opt_state[0].count) == count
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(actor.params))
            p = actor(observations[train:], return_components=True)
            route = p['logits'].argmax(-1)
            chosen = jnp.clip(p['means'][jnp.arange(4096), route], -1, 1)
            regret = scores[train:].max(-1)-scores[train:][jnp.arange(4096), route]
            row = dict(actual_Adam=count, route_accuracy=float(jnp.mean(route == indices[train:])),
                all_head_MSE=float(jnp.mean((p['means']-actions[train:])**2)),
                score_MSE=float(jnp.mean((p['logits']-rank_targets[train:])**2)),
                within_teacher_bank_selection_regret_mean=float(regret.mean()),
                within_teacher_bank_selection_regret_p95=float(jnp.quantile(regret,.95)),
                heldout_selected_action_MSE=float(jnp.mean((chosen-targets[train:])**2)),
                elapsed_seconds=time.time()-result['time'])
            snapshots.append(row)
            result['actors'][method] = dict(status='fitting', snapshots=snapshots)
            write(args.output/'status.json', result)
            print(json.dumps(dict(method=method, **row)), flush=True)
            previous = count
        params = flax.serialization.to_state_dict(jax.device_get(actor.params))
        metadata = dict(schema='single_actor_deployment_v1', observation_dim=observations.shape[-1],
            action_dim=dim, hidden_dims=[256]*4, layer_norm=False, input='state', noise_dim=0,
            requires_critic=False, requires_BC_flow=False, flow_integration_steps=0,
            generated_candidate_actions=0, actor_forward_passes=1, realized_action_samples=1,
            components=32, architecture='state_only_route_then_one_linear_head')
        artifact = args.output/(method+'_private_actor_100000.pkl')
        with artifact.open('xb') as f:
            pickle.dump(dict(schema='private_frozen_teacher_student_v1', method=method,
                training_mode='CPU_fixed_teacher_score_diagnostic_only', actor_params=params,
                metadata=metadata, actor_Adam_updates=100000, teacher_optimizer_updates_performed=0,
                teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
                source_sha256=result['source_sha256']), f, pickle.HIGHEST_PROTOCOL)
        namespace = SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env = ogbench.make_env_and_datasets(parent['environment'], env_only=True)
        reset = prepare_seeded_action_space(env, 42)
        evaluated = evaluate(namespace, env, routed.make_policy(metadata), 100000, 50)
        env.close()
        evaluated.update(time=time.time(), protocol='private_fixed_teacher_rank_student_CPU50_v1',
            actor_artifact_sha256=sha(artifact), teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
            reset_metadata=reset, no_optimizer_updates=True, counts_toward_joint_or_goal=False)
        ep = args.output/(method+'_100K_CPU50_v1.json')
        write(ep, evaluated)
        result['actors'][method] = dict(status='complete', snapshots=snapshots,
            success=evaluated['success_rate'], evaluation_path=str(ep), evaluation_sha256=sha(ep),
            private_actor_path=str(artifact), private_actor_sha256=sha(artifact), metadata=metadata)
        write(args.output/'status.json', result)
        print(json.dumps(dict(method=method, success=evaluated['success_rate'])), flush=True)
    assert sha(case['cache']) == case['cache_sha256']
    result.update(status='complete', completed=time.time())
    write(args.output/'result.json', result)
    write(args.output/'status.json', result)


if __name__ == '__main__':
    main()
