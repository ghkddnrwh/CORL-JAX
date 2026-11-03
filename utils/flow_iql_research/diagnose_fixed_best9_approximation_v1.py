"""CPU scratch approximation study; never a jointly trained policy evaluation.

Freeze an existing learner, cache best9 input/target pairs on disjoint replay
observations, then fit disposable actors. No learner weights are written or
updated. No deployable actor artifact is emitted. The cached held-out MSE is
an approximation diagnostic, not an RL performance measurement.
"""
import argparse
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

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import set_onestep_deploy_core_v1 as invariant
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, default=1000000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(d.platform == 'cpu' for d in jax.devices())
    assert len(os.sched_getaffinity(0)) == 6
    args.output.mkdir(parents=True)
    started = time.time()
    conf = json.loads((args.run / 'config.json').read_text())
    identity = conf['identity']
    for path, digest in conf['source_sha256'].items():
        assert sha(path) == digest, path
    checkpoint = args.run / f'checkpoint_{args.step}.pkl'
    checkpoint_digest = sha(checkpoint)
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['identity'] == identity and saved['step'] == args.step
    assert saved['source_sha256'] == conf['source_sha256']
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env, data, _ = ogbench.make_env_and_datasets(identity['environment'])
    env.close()
    # Distinct float32 observations, not merely distinct replay indices.
    # This still is not a trajectory-disjoint or distribution-shift holdout.
    random = np.random.RandomState(31831)
    selected, seen = [], set()
    while len(selected) < 1280:
        for index in random.choice(len(data['actions']), size=4096, replace=False):
            digest = hashlib.sha256(np.asarray(data['next_observations'][index], np.float32).tobytes()).digest()
            if digest not in seen:
                seen.add(digest)
                selected.append(int(index))
                if len(selected) == 1280:
                    break
    train_indices = np.asarray(selected[:1024])
    validation_indices = np.asarray(selected[1024:])
    observations = np.asarray(data['next_observations'][selected], np.float32)
    action_dim = data['actions'].shape[-1]
    example_actions = jnp.asarray(data['actions'][selected[:1]], jnp.float32)
    template = flat.create(identity['seed'], jnp.asarray(observations[:1]), example_actions, cfg, identity['recipe'])
    teacher = flax.serialization.from_state_dict(template, saved['state'])
    assert int(teacher.base.q_updates) == int(teacher.actor_updates) == args.step
    assert int(teacher.base.v_updates) == 0
    del data, saved, template
    agent = teacher.base.agent

    @jax.jit
    def label(obs, key):
        broadcast, noises = flat.lean.bank(agent, obs, key, 9)
        proposals = flat.lean.flow(agent, broadcast, noises, 10)
        q = agent.network.select('critic')(broadcast, actions=proposals).min(0)
        target = flat.gather(proposals, q.argmax(0))
        return flat.packed_input(noises, {'input': 'bank'}), target

    # Four independent banks for each replay observation. No bank is shared
    # between training and validation because their observations are disjoint.
    repeated = np.repeat(observations, 4, axis=0)
    noise_chunks, target_chunks = [], []
    for chunk, offset in enumerate(range(0, len(repeated), 256)):
        packed, target = jax.device_get(label(jnp.asarray(repeated[offset:offset + 256]),
                                             jax.random.fold_in(jax.random.PRNGKey(31832), chunk)))
        noise_chunks.append(packed)
        target_chunks.append(target)
    packed = np.concatenate(noise_chunks)
    targets = np.concatenate(target_chunks)
    assert np.isfinite(packed).all() and np.isfinite(targets).all()
    train_size = 1024 * 4
    cache = args.output / 'fixed_pairs.npz'
    np.savez_compressed(cache, observations=repeated, packed_noises=packed,
                        targets=targets, train_size=train_size,
                        train_replay_indices=train_indices, validation_replay_indices=validation_indices)
    train = tuple(jnp.asarray(x[:train_size]) for x in (repeated, packed, targets))
    heldout = tuple(jnp.asarray(x[train_size:]) for x in (repeated, packed, targets))
    result = dict(schema='fixed_best9_scratch_approximation_v1', started=started,
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_digest,
                  environment=identity['environment'], original_recipe=identity['recipe'],
                  teacher_step=args.step, teacher_Q_aggregation='online_min',
                  teacher_proposals=9, teacher_flow_Euler_steps=10,
                  dataset_index_seed=31831, teacher_bank_seed=31832,
                  train_distinct_observations=1024, heldout_distinct_observations=256,
                  independent_banks_per_observation=4, batch_size=256,
                  heldout_exact_observations_disjoint=True,
                  heldout_trajectory_disjoint=False, heldout_distribution_shift=False,
                  pair_cache_sha256=sha(cache), teacher_optimizer_updates_performed=0,
                  registered_GPU_learner_updates_performed=0, scratch_student_Adam_updates={},
                  no_policy_success_evaluation=True, no_deployable_artifact_saved=True,
                  source_sha256={str(Path(__file__).resolve()): sha(__file__),
                                 **{str(p): sha(p) for p in flat.source_paths()},
                                 str(Path(invariant.__file__).resolve()): sha(invariant.__file__)},
                  scope='Disposable actor capacity/optimization diagnostic. Neither a joint-training result nor a policy performance claim.',
                  actors={})
    write(args.output / 'status.json', result)
    for recipe, core in [('bank_bc', flat), ('set_bc', invariant)]:
        student_cfg = core.config_for(identity['environment'], recipe)
        student = core.create(0, jnp.asarray(observations[:1]), example_actions, student_cfg, recipe).actor
        metadata = core.actor_metadata(student_cfg, observations.shape[-1], action_dim)

        @jax.jit
        def fit(state, rng, steps):
            def update(carry, _):
                actor, key = carry
                key, sample_key = jax.random.split(key)
                indices = jax.random.randint(sample_key, (256,), 0, train_size)
                obs, noise, target = (x[indices] for x in train)
                def loss(params):
                    mse = jnp.mean((actor(obs, noise, params=params) - target) ** 2)
                    return mse, {'MSE': mse}
                actor, _ = actor.apply_loss_fn(loss)
                return (actor, key), None
            return jax.lax.scan(update, (state, rng), None, length=steps)[0]

        # The loop length is static for each reporting interval.
        fit = jax.jit(fit.__wrapped__, static_argnums=2)

        @jax.jit
        def metrics(actor, triples):
            obs, noise, target = triples
            raw = actor(obs, noise)
            action = jnp.clip(raw, -1, 1)
            q = agent.network.select('critic')(obs, actions=action).min(0)
            target_q = agent.network.select('critic')(obs, actions=target).min(0)
            return dict(raw_MSE=jnp.mean((raw - target) ** 2),
                        clipped_MSE=jnp.mean((action - target) ** 2),
                        raw_saturation_fraction=jnp.mean((jnp.abs(raw) > 1).astype(jnp.float32)),
                        teacher_critic_predicted_regret=jnp.mean(target_q - q))

        record = dict(metadata=metadata, parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(student.params)),
                      learning_rate=student_cfg['lr'], snapshots=[])
        result['actors'][recipe] = record
        key = jax.random.PRNGKey(31833)
        previous = 0
        for updates in (0, 100, 500, 1000, 2000, 5000):
            if updates:
                student, key = fit(student, key, updates - previous)
            assert int(student.opt_state[0].count) == updates
            snapshot = dict(Adam_updates=updates, elapsed_seconds=time.time() - started,
                            train={k:float(v) for k,v in jax.device_get(metrics(student, train)).items()},
                            heldout={k:float(v) for k,v in jax.device_get(metrics(student, heldout)).items()})
            record['snapshots'].append(snapshot)
            result['scratch_student_Adam_updates'][recipe] = updates
            write(args.output / 'status.json', result)
            print(json.dumps({'recipe':recipe, **snapshot}), flush=True)
            previous = updates
    assert sha(checkpoint) == checkpoint_digest
    result.update(completed=time.time(), checkpoint_unchanged=True, status='complete')
    write(args.output / 'result.json', result)
    write(args.output / 'status.json', result)


if __name__ == '__main__':
    main()
