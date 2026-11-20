"""Disposable CPU distribution-distillation prototype over frozen replay labels.

No policy is exported or evaluated in an environment. Independent student
noise is essential for the product-expectation energy loss; antithetic or
teacher-coupled student draws would change its meaning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import set_onestep_deploy_core_v1 as invariant


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


@jax.custom_jvp
def norm(diff):
    """Exact RMS Euclidean norm, with the zero subgradient at equality."""
    return jnp.sqrt(jnp.mean(diff * diff,axis=-1))


@norm.defjvp
def norm_jvp(primals,tangents):
    (diff,), (tangent,) = primals,tangents
    value = norm(diff)
    denominator = jnp.where(value > 0,value,1.)
    derivative = jnp.where(value > 0,jnp.mean(diff * tangent,axis=-1) / denominator,0.)
    return value,derivative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    started = time.time()
    parent = json.loads((args.cache_directory / 'result.json').read_text())
    cache_path = args.cache_directory / 'fixed_pairs.npz'
    assert sha(cache_path) == parent['pair_cache_sha256']
    for path,digest in parent['source_sha256'].items():
        assert sha(path) == digest,path
    cache = np.load(cache_path)
    count = int(cache['train_size'])
    assert count == 4096 and len(cache['targets']) == 5120
    assert np.isfinite(np.asarray(jax.grad(lambda x:norm(x))(jnp.zeros(5)))).all()
    train = tuple(jnp.asarray(cache[k][:count]) for k in ('observations','packed_noises','targets'))
    # Held-out observations and all4teacher targets per observation.
    evaluation_obs = jnp.asarray(cache['observations'][count::4])
    evaluation_targets = jnp.moveaxis(jnp.asarray(cache['targets'][count:]).reshape((256,4,-1)),1,0)
    observation_dim = evaluation_obs.shape[-1]
    action_dim = evaluation_targets.shape[-1]
    result = dict(schema='energy_student_CPU_prototype_v1',started=started,
        environment=parent['environment'],parent_result=str(args.cache_directory / 'result.json'),
        parent_result_sha256=sha(args.cache_directory / 'result.json'),pair_cache_sha256=sha(cache_path),
        source_sha256={str(Path(__file__).resolve()):sha(__file__),
            **{str(p):sha(p) for p in flat.source_paths()},str(Path(invariant.__file__).resolve()):sha(invariant.__file__)},
        teacher_optimizer_updates_performed=0,registered_GPU_learner_updates_performed=0,
        no_policy_success_evaluation=True,no_deployable_artifact_saved=True,
        train_batch_size=256,learning_rate=.0003,teacher_targets_per_observation=4,
        heldout_distinct_observations=256,heldout_student_samples_per_observation=32,
        training_student_noise_seed=32011,heldout_student_noise_seed=32012,
        energy_norm='Exact Euclidean norm divided by sqrt(action_dim); zero subgradient at equality.',
        energy_training_loss='0.5*(d(X,Y)+d(Xprime,Y)-d(X,Xprime)); X/Xprime independent of each other and cached Y given s.',
        energy_evaluation='2*cross_mean-off_diagonal_student_Ustat-off_diagonal_teacher_Ustat; finite estimate may be negative.',
        primary_reference='https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf',
        reference_scope='Section4.3Eq22, sign reversed for minimization, beta1. Proper-score theory is not an RL success guarantee.',
        scope='Frozen-teacher replay prototype only, one scratch fit per case. No joint learner result or nine-task claim.',actors={})
    cases = [('coupled_MSE_bank','bank_bc',flat),('coupled_MSE_set','set_bc',invariant),
             ('independent_MSE_single','independent_MSE_single',flat),('energy_single','energy_single',flat)]
    for name,recipe,core in cases:
        if core is invariant or recipe == 'bank_bc':
            cfg = core.config_for(parent['environment'],recipe)
        else:
            cfg = flat.config_for(parent['environment'],'bank_bc')
            cfg['onestep_specification'].update(name=recipe,input='single',objective='scratch_distribution_only')
        student = core.create(0,jnp.zeros((1,observation_dim)),jnp.zeros((1,action_dim)),cfg,recipe).actor
        metadata = core.actor_metadata(cfg,observation_dim,action_dim)
        coupled = name.startswith('coupled_')
        use_energy = name == 'energy_single'

        def fit(state,rng,steps):
            def update(carry,_):
                actor,key = carry
                key,index_key,noise_key = jax.random.split(key,3)
                indices = jax.random.randint(index_key,(256,),0,count)
                obs,packed,target = (x[indices] for x in train)
                z = jax.random.normal(noise_key,(2,256,action_dim))
                def loss(params):
                    if coupled:
                        value = jnp.mean((actor(obs,packed,params=params)-target)**2)
                    else:
                        x = actor(obs,z[0],params=params)
                        xp = actor(obs,z[1],params=params)
                        if use_energy:
                            value = .5*jnp.mean(norm(x-target)+norm(xp-target)-norm(x-xp))
                        else:
                            value = .5*jnp.mean((x-target)**2+(xp-target)**2)
                    return value,{'objective':value}
                actor,_ = actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(state,rng),None,length=steps)[0]
        fit = jax.jit(fit,static_argnums=2)

        @jax.jit
        def metrics(actor):
            obs = jnp.broadcast_to(evaluation_obs,(32,*evaluation_obs.shape))
            n = 9 if coupled else 1
            z = jax.random.normal(jax.random.PRNGKey(32012),(n,32,256,action_dim))
            inputs = flat.packed_input(z,{'input':'bank' if coupled else 'single'})
            raw = actor(obs,inputs)
            x = jnp.clip(raw,-1,1)
            y = evaluation_targets
            cross = norm(x[:,None]-y[None,:]).mean()
            xx = norm(x[:,None]-x[None,:])
            yy = norm(y[:,None]-y[None,:])
            self_x = jnp.sum(xx)/ (32*31*256)
            self_y = jnp.sum(yy)/ (4*3*256)
            variance_x = jnp.var(x,axis=0,ddof=1).mean()
            variance_y = jnp.var(y,axis=0,ddof=1).mean()
            return dict(heldout_energy_distance_Ustat=2*cross-self_x-self_y,
                heldout_energy_score=cross-.5*self_x,
                heldout_independent_action_MSE=jnp.mean((x[:,None]-y[None,:])**2),
                student_same_state_variance=variance_x,teacher_same_state_variance=variance_y,
                student_to_teacher_variance_ratio=variance_x/jnp.maximum(variance_y,1e-12),
                raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
        initialization = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(student.params):
            array = np.asarray(leaf)
            initialization.update(str((array.shape,str(array.dtype))).encode())
            initialization.update(array.tobytes())
        record = dict(metadata=metadata,parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(student.params)),
                      initial_actor_params_sha256=initialization.hexdigest(),snapshots=[])
        result['actors'][name] = record
        previous,key = 0,jax.random.PRNGKey(32011)
        for updates in (0,100,1000,5000):
            if updates:
                student,key = fit(student,key,updates-previous)
            assert int(student.opt_state[0].count) == updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(student.params))
            values = {k:float(v) for k,v in jax.device_get(metrics(student)).items()}
            snapshot = dict(Adam_updates=updates,elapsed_seconds=time.time()-started,**values)
            record['snapshots'].append(snapshot)
            write(args.output / 'status.json',result)
            print(json.dumps(dict(actor=name,**snapshot)),flush=True)
            previous = updates
    assert sha(cache_path) == parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_actor_total_Adam_updates=20000)
    write(args.output / 'result.json',result)
    write(args.output / 'status.json',result)


if __name__ == '__main__':
    main()
