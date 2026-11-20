"""Disposable frozen-teacher CPU study: average best9 energy targets exactly.

No joint learner, environment rollout, export, or GPU queue change occurs.
The additional nine training candidates integrate subset-selection noise;
they do not redefine the desired policy as best18.
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

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import energy_onestep_deploy_core_v1 as energy
from utils.flow_iql_research import best9_subset_weights_v1 as subset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def tree_digest(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        value = np.asarray(leaf)
        digest.update(str((value.shape,str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    started = time.time()
    parent_path = args.cache_directory / 'result.json'
    parent = json.loads(parent_path.read_text())
    cache_path = args.cache_directory / 'fixed_pairs.npz'
    assert sha(cache_path) == parent['pair_cache_sha256']
    for path,digest in parent['source_sha256'].items():
        assert sha(path) == digest, path
    checkpoint = Path(parent['checkpoint'])
    checkpoint_digest = sha(checkpoint)
    assert checkpoint_digest == parent['checkpoint_sha256']
    conf = json.loads((checkpoint.parent / 'config.json').read_text())
    identity = conf['identity']
    for path,digest in conf['source_sha256'].items():
        assert sha(path) == digest, path
    cache = np.load(cache_path)
    observations = np.asarray(cache['observations'])
    old_targets = np.asarray(cache['targets'])
    count = int(cache['train_size'])
    action_dim = old_targets.shape[-1]
    assert len(observations) == 5120 and count == 4096
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    template = flat.create(identity['seed'],jnp.asarray(observations[:1]),
                           jnp.zeros((1,action_dim)),cfg,identity['recipe'])
    teacher = flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates) == int(teacher.actor_updates) == 1000000
    assert int(teacher.base.v_updates) == 0
    del template,saved
    agent = teacher.base.agent
    result = dict(schema='best9_RaoBlackwell_energy_CPU_prototype_v1',started=started,
        environment=parent['environment'], checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_digest,
        parent_result=str(parent_path),parent_result_sha256=sha(parent_path),pair_cache_sha256=sha(cache_path),
        source_sha256={str(Path(p).resolve()):sha(p) for p in
            [__file__,subset.__file__]+[str(p) for p in energy.source_paths()]},
        mathematical_checks=subset.exact_checks(), teacher_bestN=9,training_bank_size=18,
        teacher_Q_aggregation='online_min',teacher_flow_Euler_steps=10,
        teacher_optimizer_updates_performed=0,registered_GPU_learner_updates_performed=0,
        no_policy_success_evaluation=True,no_deployable_artifact_saved=True,
        train_batch_size=256,learning_rate=.0003,student_noise_seed=32011,heldout_noise_seed=32012,
        extra_teacher_bank_seed=33021,student_draws_independent_of_teacher=True,
        scope='Fixed replay/teacher approximation study, not a joint-learning or nine-task performance claim.',
        prior_art='Best-of-N distribution distillation predates this experiment (BOND2024); energy-score one-step policies also predate it (ESP2026). No novelty claim for these components.',
        actors={})
    write(args.output / 'status.json',result)

    @jax.jit
    def proposals(obs,noise):
        broadcast = jnp.broadcast_to(obs,(9,*obs.shape))
        actions = flat.lean.flow(agent,broadcast,noise,10)
        q = agent.network.select('critic')(broadcast,actions=actions).min(0)
        return actions,q

    actions_chunks,q_chunks,baseline_chunks = [],[],[]
    for chunk,offset in enumerate(range(0,len(observations),256)):
        obs = jnp.asarray(observations[offset:offset+256])
        first_noise = jnp.moveaxis(jnp.asarray(cache['packed_noises'][offset:offset+256]).reshape((256,9,action_dim)),1,0)
        extra_noise = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(33021),chunk),(9,256,action_dim))
        first,q_first = proposals(obs,first_noise)
        extra,q_extra = proposals(obs,extra_noise)
        bank,q = np.asarray(jnp.concatenate((first,extra))),np.asarray(jnp.concatenate((q_first,q_extra)))
        baseline = np.asarray(flat.gather(first,jnp.argmax(q_first,axis=0)))
        actions_chunks.append(bank)
        q_chunks.append(q)
        baseline_chunks.append(baseline)
        print(json.dumps(dict(stage='teacher_bank_cache',offset=offset,elapsed_seconds=time.time()-started)),flush=True)
    actions = np.concatenate(actions_chunks,axis=1)
    qs = np.concatenate(q_chunks,axis=1)
    baseline = np.concatenate(baseline_chunks)
    weights = subset.weights_numpy(qs,9).astype(np.float32)
    w_jax = np.asarray(jax.jit(subset.weights_jax)(jnp.asarray(qs)))
    weight_difference = float(np.max(np.abs(w_jax-weights)))
    assert weight_difference < 1e-7
    assert np.isfinite(actions).all() and np.isfinite(qs).all()
    # Original first9 floating-point evaluations can differ after recompiling
    # with explicit noises. Record this; both new fits use the reconstructed
    # bank so their ablation still has identical candidate realizations.
    target_difference = float(np.max(np.abs(baseline-old_targets)))
    changed_rows = int(np.any(np.abs(baseline-old_targets)>1e-5,axis=-1).sum())
    result.update(original_cached_target_max_absolute_difference=target_difference,
                  original_cached_target_rows_difference_above_1e_minus5=changed_rows,
                  subset_weight_numpy_JAX_max_difference=weight_difference,
                  exact_Q_tie_fraction=float(np.mean(np.sum(qs[:,None,:] == qs[None,:,:],axis=1)>1)),
                  average_effective_teacher_rank_samples=float(np.mean(1/np.sum(weights**2,axis=0))))
    assert changed_rows == 0, 'Do not silently change the first9 teacher labels.'
    teacher_cache = args.output / 'rb_training_pairs.npz'
    np.savez_compressed(teacher_cache,actions=actions,q=qs,weights=weights,baseline_targets=baseline)
    result['RB_training_cache_sha256'] = sha(teacher_cache)

    # Fixed action x=0 is independent of every teacher bank. This estimates
    # conditional ACTION-space gradient variance, not full parameter variance.
    differences = -actions.astype(np.float64)
    distances = np.sqrt(np.mean(differences**2,axis=-1))
    gradients = np.where(distances[...,None] > 0,
        differences / np.maximum(distances[...,None]*action_dim,1e-30),0.)
    raw_gradient = np.take_along_axis(gradients,
        np.broadcast_to(qs[:9].argmax(0)[None,:,None],(1,len(observations),action_dim)),axis=0)[0]
    rb_gradient = np.sum(weights[...,None]*gradients,axis=0)
    raw_loss = np.sqrt(np.mean(baseline.astype(np.float64)**2,axis=-1))
    rb_loss = np.sum(weights*distances,axis=0)
    variance = lambda x:float(np.var(x.reshape((1280,4,*x.shape[1:])),axis=1,ddof=1).mean())
    result['fixed_zero_action_teacher_sampling_variance'] = dict(
        observations=1280,independent_banks_per_observation=4,
        raw_action_gradient_variance=variance(raw_gradient),RB_action_gradient_variance=variance(rb_gradient),
        action_gradient_variance_ratio=variance(rb_gradient)/variance(raw_gradient),
        raw_loss_variance=variance(raw_loss),RB_loss_variance=variance(rb_loss),
        loss_variance_ratio=variance(rb_loss)/variance(raw_loss),
        scope='Empirical average conditional variance at a fixed zero action; four banks/state, no confidence bound or full actor-gradient claim.')
    write(args.output / 'status.json',result)
    train_obs = jnp.asarray(observations[:count])
    train_y = jnp.asarray(baseline[:count])
    train_bank = jnp.asarray(np.moveaxis(actions[:,:count],0,1))
    train_weights = jnp.asarray(weights[:,:count].T)
    evaluation_obs = jnp.asarray(observations[count::4])
    evaluation_targets = jnp.moveaxis(jnp.asarray(old_targets[count:]).reshape((256,4,-1)),1,0)
    student_cfg = energy.config_for(parent['environment'],'energy_bc')
    for name in ('energy_original_best9','energy_RB18_same_best9_law'):
        student = energy.create(0,jnp.asarray(observations[:1]),jnp.zeros((1,action_dim)),student_cfg,'energy_bc').actor
        use_rb = name.startswith('energy_RB')
        record = dict(initial_actor_params_sha256=tree_digest(student.params),
                      parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(student.params)),snapshots=[])
        result['actors'][name] = record

        def fit(actor,key,steps):
            def update(carry,_):
                actor,key = carry
                key,index_key,noise_key = jax.random.split(key,3)
                indices = jax.random.randint(index_key,(256,),0,count)
                obs,y,bank,w = train_obs[indices],train_y[indices],train_bank[indices],train_weights[indices]
                noises = jax.random.normal(noise_key,(2,256,action_dim))
                def objective(params):
                    first = actor(obs,noises[0],params=params)
                    second = actor(obs,noises[1],params=params)
                    if use_rb:
                        cross = .5*jnp.mean(jnp.sum(w*(energy.norm(first[:,None]-bank)+energy.norm(second[:,None]-bank)),axis=1))
                    else:
                        cross = .5*jnp.mean(energy.norm(first-y)+energy.norm(second-y))
                    value = cross-.5*jnp.mean(energy.norm(first-second))
                    return value,dict(energy_score=value)
                actor,_ = actor.apply_loss_fn(objective)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit = jax.jit(fit,static_argnums=2)

        @jax.jit
        def metrics(actor):
            obs = jnp.broadcast_to(evaluation_obs,(32,*evaluation_obs.shape))
            z = jax.random.normal(jax.random.PRNGKey(32012),(32,256,action_dim))
            raw = actor(obs,z)
            x,y = jnp.clip(raw,-1,1),evaluation_targets
            cross = energy.norm(x[:,None]-y[None,:]).mean()
            xx = energy.norm(x[:,None]-x[None,:]).sum()/(32*31*256)
            yy = energy.norm(y[:,None]-y[None,:]).sum()/(4*3*256)
            vx,vy = jnp.var(x,axis=0,ddof=1).mean(),jnp.var(y,axis=0,ddof=1).mean()
            return dict(heldout_energy_distance_Ustat=2*cross-xx-yy,
                heldout_energy_score=cross-.5*xx,student_same_state_variance=vx,
                teacher_same_state_variance=vy,student_to_teacher_variance_ratio=vx/jnp.maximum(vy,1e-12),
                raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
        previous,key = 0,jax.random.PRNGKey(32011)
        for updates in (0,100,1000,5000):
            if updates:
                student,key = fit(student,key,updates-previous)
            assert int(student.opt_state[0].count) == updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(student.params))
            snapshot = dict(Adam_updates=updates,elapsed_seconds=time.time()-started,
                           **{k:float(v) for k,v in jax.device_get(metrics(student)).items()})
            record['snapshots'].append(snapshot)
            write(args.output / 'status.json',result)
            print(json.dumps(dict(actor=name,**snapshot)),flush=True)
            previous = updates
    assert len({a['initial_actor_params_sha256'] for a in result['actors'].values()}) == 1
    assert sha(checkpoint) == checkpoint_digest and sha(cache_path) == parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_actor_total_Adam_updates=10000,
                  checkpoint_unchanged=True)
    write(args.output / 'result.json',result)
    write(args.output / 'status.json',result)


if __name__ == '__main__':
    main()
