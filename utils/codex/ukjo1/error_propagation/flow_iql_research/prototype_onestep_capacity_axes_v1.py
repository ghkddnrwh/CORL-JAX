"""Frozen-label capacity/conditioning axes; disposable CPU actors, no RL.

Four existing, separately supported flat-actor specifications are tested.
This is a replay approximation study, not a joint-training ablation. Both
paired target MSE and independent conditional distribution error are logged.
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

from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.energy_onestep_deploy_core_v1 import norm


RECIPES=('bank_bc','bank_bc__d6','bank_bc__w2','bank_bc__ln')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def digest_tree(tree):
    h=hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        x=np.asarray(leaf);h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.cache_directory/'result.json';parent=json.loads(parent_path.read_text())
    for p,d in parent['source_sha256'].items():assert sha(p)==d,p
    cache_path=args.cache_directory/'fixed_pairs.npz'
    assert sha(cache_path)==parent['pair_cache_sha256']
    cache=np.load(cache_path);count=int(cache['train_size'])
    assert count==4096 and len(cache['targets'])==5120
    triples=tuple(jnp.asarray(cache[k]) for k in ('observations','packed_noises','targets'))
    train=tuple(x[:count] for x in triples);heldout=tuple(x[count:] for x in triples)
    eval_obs=heldout[0][::4]
    eval_targets=jnp.moveaxis(heldout[2].reshape((256,4,-1)),1,0)
    action_dim=eval_targets.shape[-1];started=time.time()
    result=dict(schema='frozen_best9_CPU_capacity_axes_v1',time=started,
        environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        pair_cache_sha256=sha(cache_path),source_sha256={str(Path(p).resolve()):sha(p)
            for p in [__file__]+[str(p) for p in core.source_paths()]},
        planned_recipes=list(RECIPES),Adam_updates_per_actor=5000,learning_rate=.0003,batch_size=256,
        replay_minibatch_seed=31833,heldout_student_noise_seed=32012,
        train_distinct_observations=1024,heldout_distinct_observations=256,teacher_draws_per_state=4,
        heldout_trajectory_disjoint=False,teacher_updates=0,registered_GPU_updates=0,
        environment_rollouts=0,deployable_exports=0,
        scope='Each architecture has its own default random initialization; no claim of matched parameter count/function. Same replay pairs, Adam hyperparameters, minibatches and5000updates. No joint or RL result.',actors={})
    for recipe in RECIPES:
        cfg=core.config_for(parent['environment'],recipe)
        student=core.create(0,train[0][:1],train[2][:1],cfg,recipe).actor
        metadata=core.actor_metadata(cfg,train[0].shape[-1],action_dim)
        record=dict(metadata=metadata,parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(student.params)),
                    initial_params_sha256=digest_tree(student.params),snapshots=[])
        result['actors'][recipe]=record

        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry
                key,index_key=jax.random.split(key)
                idx=jax.random.randint(index_key,(256,),0,count)
                obs,z,target=(x[idx] for x in train)
                def objective(params):
                    value=jnp.mean((actor(obs,z,params=params)-target)**2)
                    return value,dict(MSE=value)
                actor,_=actor.apply_loss_fn(objective)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)

        @jax.jit
        def metrics(actor):
            result={}
            for label,data in [('train',train),('heldout',heldout)]:
                obs,z,target=data
                raw=actor(obs,z);action=jnp.clip(raw,-1,1)
                result[label+'_paired_raw_MSE']=jnp.mean((raw-target)**2)
                result[label+'_paired_clipped_MSE']=jnp.mean((action-target)**2)
            obs=jnp.broadcast_to(eval_obs,(32,*eval_obs.shape))
            noises=jax.random.normal(jax.random.PRNGKey(32012),(9,32,256,action_dim))
            raw=actor(obs,core.packed_input(noises,{'input':'bank'}))
            x,y=jnp.clip(raw,-1,1),eval_targets
            cross=norm(x[:,None]-y[None,:]).mean()
            xx=norm(x[:,None]-x[None,:]).sum()/(32*31*256)
            yy=norm(y[:,None]-y[None,:]).sum()/(4*3*256)
            result.update(heldout_energy_distance_Ustat=2*cross-xx-yy,
                same_state_variance_ratio=jnp.var(x,axis=0,ddof=1).mean()/jnp.maximum(jnp.var(y,axis=0,ddof=1).mean(),1e-12),
                independent_raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
            return result
        previous,key=0,jax.random.PRNGKey(31833)
        for updates in (0,100,1000,5000):
            if updates:student,key=fit(student,key,updates-previous)
            assert int(student.opt_state[0].count)==updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(student.params))
            snapshot=dict(Adam_updates=updates,elapsed_seconds=time.time()-started,
                          **{k:float(v) for k,v in jax.device_get(metrics(student)).items()})
            record['snapshots'].append(snapshot);write(args.output/'status.json',result)
            print(json.dumps(dict(recipe=recipe,**snapshot)),flush=True)
            previous=updates
    assert sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_Adam_updates=20000)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
