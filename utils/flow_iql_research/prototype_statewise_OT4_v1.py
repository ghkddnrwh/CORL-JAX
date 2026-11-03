"""Frozen-best9 conditional coupling comparison; not a joint RL algorithm.

IndependentMSE, latentOT, current-actionOT and proper energy score share
one architecture, initialization, quartets, minibatches and Adam budget.
Related prior art: OT-NFM2604.06413, OptiFlow2609.15883, C2OT2503.10636.
This is not a reproduction of full OptiFlow/VaBC/Sinkhorn/value marginals.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import energy_onestep_deploy_core_v1 as energy
from utils.flow_iql_research import conditional_assignment4_v1 as assignment
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


METHODS=('action_OT4','latent_OT4','energy4','independent_MSE4')


def objective(method,prediction,noise,target):
    if method=='energy4':
        cross=energy.norm(prediction[:,None]-target[None,:]).mean()
        self_distance=energy.norm(prediction[:,None]-prediction[None,:])
        batch=np.prod(prediction.shape[1:-1])
        return cross-.5*self_distance.sum()/(4*3*batch)
    if method=='latent_OT4':target,_,_=assignment.match(noise,target)
    elif method=='action_OT4':target,_,_=assignment.match(prediction,target)
    else:assert method=='independent_MSE4'
    return jnp.mean((prediction-target)**2)


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--label-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    args.output.mkdir(parents=True)
    pp=args.study_case/'result.json';parent=json.loads(pp.read_text());assert parent['status']=='complete'
    original=args.study_case/'frozen_replay_pairs_v1.npz';assert sha(original)==parent['pair_cache_sha256']
    with np.load(original)as c:
        observations=c['observations'].copy();old_targets=c['targets'].copy();train=int(c['train_size']);eval_targets=c['distribution_eval_targets'].copy()
    manifest=json.loads(args.label_manifest.read_text());entry=manifest['cases'][args.study_case.name];cache=Path(entry['path']);assert sha(cache)==entry['sha256']
    with np.load(cache)as c:target_quartets=c['labels'].copy()
    assert target_quartets.shape==(4,65536,old_targets.shape[-1])and np.array_equal(target_quartets[0],old_targets[:train])
    obs,y=jnp.asarray(observations),jnp.asarray(target_quartets);eval_obs=obs[train:train+256];eval_y=jnp.asarray(eval_targets)
    dim=old_targets.shape[-1];cfg=energy.config_for(parent['environment'],'energy_bc')
    metadata=energy.actor_metadata(cfg,observations.shape[-1],dim);assert metadata['input']=='single'and metadata['noise_dim']==dim
    definition=flat.actor_definition(metadata)
    initial=definition.init(jax.random.fold_in(jax.random.PRNGKey(0),99212),obs[:1],jnp.zeros((1,dim)))['params']
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],parent_result=str(pp),parent_result_sha256=sha(pp),
        original_cache_sha256=sha(original),four_label_cache_sha256=sha(cache),label_manifest_sha256=sha(args.label_manifest),
        methods=list(METHODS),training_states=train,heldout_states=4096,teacher_samples_per_state=4,student_draws=4,
        actor_initialization='Same as existing single-noise energy actor:seed0/fold99212.',minibatch_seed=34211,batch_size=256,learning_rate=.0003,Adam_per_method=20000,
        metadata=metadata,parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(initial)),
        teacher_or_GPU_updates=0,evaluation_interactions_used_for_learning=False,all_methods_same_initial_parameters_and_batches=True,
        matching_scope='Independently within each state. Never permute targets across observations; prior Gaussian quartet remains unchanged.',
        clipping='All losses use raw outputs, evaluation clips[-1,1], same as existing raw energy student. Saturation logged.',
        prior_art='OT-NFM/arxiv2604.06413; OptiFlow/arxiv2609.15883; C2OT/arxiv2503.10636. Exact4permutation experiment is not fullOptiFlow/Sinkhorn/VaBC.',
        limitation='Finite-batch OT-coupled squared regression can be biased/underdispersed; no exact population distribution or RL guarantee.',
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,assignment.__file__,*energy.source_paths()]},actors={},
        scope='Frozen1Mteacher/CPU20Kstudents/finalCPU50 only; not joint1M,last3,nine-task,fresh-trainingseeds or speed.')
    write(args.output/'status.json',result)
    for method in METHODS:
        actor=TrainState.create(definition,initial,optax.adam(.0003))
        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,noise_key=jax.random.split(key,3);idx=jax.random.randint(index_key,(256,),0,train)
                b=jnp.broadcast_to(obs[idx],(4,256,obs.shape[-1]));z=jax.random.normal(noise_key,(4,256,dim));t=y[:,idx]
                def loss(params):
                    prediction=actor(b,z,params=params)
                    value=objective(method,prediction,z,t)
                    return value,dict(objective=value)
                actor,info=actor.apply_loss_fn(loss)
                return (actor,key),info['objective']
            return jax.lax.scan(update,(actor,key),None,length=steps)
        fit=jax.jit(fit,static_argnums=2)
        @jax.jit
        def metrics(actor):
            b=jnp.broadcast_to(eval_obs,(32,*eval_obs.shape));z=jax.random.normal(jax.random.PRNGKey(49411),(32,256,dim))
            raw=actor(b,z);actions=jnp.clip(raw,-1,1)
            cross=energy.norm(actions[:,None]-eval_y[None,:]).mean()
            xx=energy.norm(actions[:,None]-actions[None,:]).sum()/(32*31*256)
            yy=energy.norm(eval_y[:,None]-eval_y[None,:]).sum()/(4*3*256)
            return dict(heldout_energy_distance_Ustat=2*cross-xx-yy,
                student_to_teacher_variance_ratio=jnp.var(actions,0,ddof=1).mean()/jnp.maximum(jnp.var(eval_y,0,ddof=1).mean(),1e-12),
                teacher_mean_action_MSE=jnp.mean((actions.mean(0)-eval_y.mean(0))**2),
                raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
        key=jax.random.PRNGKey(34211);previous=0;snapshots=[]
        for count in (0,1000,5000,20000):
            tail=None
            if count:
                (actor,key),history=fit(actor,key,count-previous);tail=float(jnp.mean(history[-min(100,count-previous):]))
            assert int(actor.opt_state[0].count)==count
            assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor.params))
            row=dict(actual_Adam=count,last100_training_objective=tail,**{k:float(v)for k,v in jax.device_get(metrics(actor)).items()})
            snapshots.append(row);previous=count;print(json.dumps(dict(method=method,**row)),flush=True)
        params=flax.serialization.to_state_dict(jax.device_get(actor.params));artifact=args.output/(method+'_private_actor_20000.pkl')
        with artifact.open('xb')as f:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_statewise_coupling_diagnostic_only',method=method,
            metadata=metadata,actor_params=params,actor_Adam_updates=20000,teacher_optimizer_updates_performed=0,
            teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
        policy=flat.make_actor_policy(metadata,'cpu');namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,20000,50);env.close()
        evaluated.update(time=time.time(),method=method,environment=parent['environment'],protocol='private_statewise_coupling_CPU50_v1',
            actor_artifact_sha256=sha(artifact),teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],reset_metadata=reset,
            no_optimizer_updates=True,counts_toward_joint_or_goal=False)
        ep=args.output/(method+'_CPU50_v1.json');write(ep,evaluated)
        result['actors'][method]=dict(snapshots=snapshots,success=evaluated['success_rate'],private_actor_path=str(artifact),private_actor_sha256=sha(artifact),evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(args.output/'status.json',result);print(json.dumps(dict(method=method,success=evaluated['success_rate'])),flush=True)
    assert sha(original)==parent['pair_cache_sha256']and sha(cache)==entry['sha256']
    result.update(status='complete',completed=time.time(),total_scratch_Adam=80000,total_evaluation_episodes=200)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
