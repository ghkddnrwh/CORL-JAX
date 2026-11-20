"""Disposable CPU comparison of2vs8iid student draws for the same score.

The8-draw choice follows the existing ESP energy-score policy setup; no
novelty claim. Deployment and the population objective are unchanged.
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

from utils.flow_iql_research import energy_onestep_deploy_core_v1 as core


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def score(x,y):
    """x=(K,B,D), y=(B,D), exclude diagonal from independent-pair mean."""
    k=x.shape[0]
    cross=core.norm(x-y[None]).mean()
    self_term=core.norm(x[:,None]-x[None,:]).sum()/(k*(k-1)*y.shape[0])
    return cross-.5*self_term


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.cache_directory/'result.json';parent=json.loads(parent_path.read_text())
    for p,d in parent['source_sha256'].items():assert sha(p)==d,p
    cache_path=args.cache_directory/'fixed_pairs.npz';assert sha(cache_path)==parent['pair_cache_sha256']
    cache=np.load(cache_path);count=int(cache['train_size']);assert count==4096
    obs=jnp.asarray(cache['observations']);y=jnp.asarray(cache['targets']);dim=y.shape[-1]
    eval_obs=obs[count::4];eval_y=jnp.moveaxis(y[count:].reshape((256,4,dim)),1,0)
    started=time.time()
    result=dict(schema='energy_student_sample_count_CPU_prototype_v1',started=started,
        environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        pair_cache_sha256=sha(cache_path),source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__]+[str(p) for p in core.source_paths()]},
        cases=[2,8],updates_per_actor=5000,learning_rate=.0003,batch_size=256,
        training_index_noise_seed=32011,heldout_noise_seed=32012,
        noise_coupling='Generate8iidstudentdraws;K2usesfirst2, bothindependentofteacherY.',
        teacher_updates=0,registered_GPU_updates=0,RL_rollouts=0,exports=0,
        teacher_target='Unchanged cached best9online-minQ actions, four independent banks per observation.',
        primary_diagnostic='Heldout energy distance after5000updates, allearlierpointsalsoreported.',
        related_work='https://arxiv.org/html/2610.07696v1 SectionIII-C: ESP uses8trainingstudentsamples; one atdeployment. No novelty claim.',
        scope='Frozen replay CPU fits; actual joint training/return not measured. Only estimator sample count changes, sameactorparameterization/population score.',actors={})
    # Two-sample implementation equals the registered energy objective.
    fixture=jnp.arange(30,dtype=jnp.float32).reshape((2,3,5))/30
    target=jnp.arange(15,dtype=jnp.float32).reshape((3,5))/15
    original=.5*jnp.mean(core.norm(fixture[0]-target)+core.norm(fixture[1]-target)-core.norm(fixture[0]-fixture[1]))
    assert abs(float(score(fixture,target)-original))<1e-7
    assert np.isfinite(np.asarray(jax.grad(lambda x:score(x,jnp.zeros((3,5))))(jnp.zeros((8,3,5))))).all()
    result['K2_formula_and_zero_norm_gradient_checks_passed']=True
    for draws in (2,8):
        cfg=core.config_for(parent['environment'],'energy_bc')
        actor=core.create(0,obs[:1],y[:1],cfg,'energy_bc').actor
        digest=hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(actor.params):digest.update(np.asarray(leaf).tobytes())
        record=dict(initial_actor_params_sha256=digest.hexdigest(),parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(actor.params)),snapshots=[])
        result['actors'][str(draws)]=record

        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,noise_key=jax.random.split(key,3)
                idx=jax.random.randint(index_key,(256,),0,count);b,t=obs[idx],y[idx]
                noises=jax.random.normal(noise_key,(8,256,dim))[:draws]
                broadcast=jnp.broadcast_to(b,(draws,*b.shape))
                def loss(params):
                    x=actor(broadcast,noises,params=params);value=score(x,t)
                    return value,dict(energy_score=value)
                actor,_=actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)

        @jax.jit
        def metrics(actor):
            broadcast=jnp.broadcast_to(eval_obs,(32,*eval_obs.shape))
            z=jax.random.normal(jax.random.PRNGKey(32012),(32,256,dim))
            raw=actor(broadcast,z);x=jnp.clip(raw,-1,1)
            cross=core.norm(x[:,None]-eval_y[None,:]).mean()
            self_x=core.norm(x[:,None]-x[None,:]).sum()/(32*31*256)
            self_y=core.norm(eval_y[:,None]-eval_y[None,:]).sum()/(4*3*256)
            return dict(heldout_energy_distance_Ustat=2*cross-self_x-self_y,
                student_to_teacher_variance_ratio=jnp.var(x,axis=0,ddof=1).mean()/jnp.maximum(jnp.var(eval_y,axis=0,ddof=1).mean(),1e-12),
                raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
        previous,key=0,jax.random.PRNGKey(32011)
        for updates in (0,100,1000,5000):
            if updates:actor,key=fit(actor,key,updates-previous)
            assert int(actor.opt_state[0].count)==updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(actor.params))
            snapshot=dict(Adam_updates=updates,elapsed_seconds=time.time()-started,**{k:float(v) for k,v in jax.device_get(metrics(actor)).items()})
            record['snapshots'].append(snapshot);write(args.output/'status.json',result)
            print(json.dumps(dict(draws=draws,**snapshot)),flush=True);previous=updates
    assert len({x['initial_actor_params_sha256'] for x in result['actors'].values()})==1
    assert sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_Adam_updates=10000)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
