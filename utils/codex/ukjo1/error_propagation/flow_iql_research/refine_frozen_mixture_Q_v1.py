"""Warm CPU mixture refinement: NLL-only control versus3global Q weights.

Start from the identical frozen20K mixture in every branch; reset Adam, fit
5K further steps on the same cached replay, then evaluate final50episodes.
This remains a frozen-teacher diagnostic, not a joint/last3 goal result.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research import mixture_guidance_losses_v1 as guidance
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


CASES=(('NLL_control',1.,False),('NLL_Q_a003',.03,True),('NLL_Q_a03',.3,True),('NLL_Q_a3',3.,True))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def tree_norm(tree):return math.sqrt(sum(float(np.square(np.asarray(x,np.float64)).sum()) for x in jax.tree_util.tree_leaves(tree)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True);started=time.time()
    parent_path=args.study_case/'result.json';parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    for p,d in parent['source_sha256'].items():assert sha(p)==d,p
    cache_path=args.study_case/'frozen_replay_pairs_v1.npz';assert sha(cache_path)==parent['pair_cache_sha256']
    cache=np.load(cache_path);count=int(cache['train_size']);assert count==65536
    obs,y=jnp.asarray(cache['observations']),jnp.asarray(cache['targets']);d=y.shape[-1]
    entry=parent['actors']['mixture9_NLL'];model_path=Path(entry['private_actor_path']);assert sha(model_path)==entry['private_actor_sha256']
    with model_path.open('rb') as stream:model=pickle.load(stream)
    metadata=model['metadata'];definition=mixture.MixtureActor(tuple(metadata['hidden_dims']),d,9,metadata['std_floor'])
    initial_params=model['actor_params'];assert model['actor_Adam_updates']==20000
    teacher_run=Path(parent['teacher_run']);conf=json.loads((teacher_run/'config.json').read_text());identity=conf['identity']
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    with checkpoint.open('rb') as stream:saved=pickle.load(stream)
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=flat.create(identity['seed'],obs[:1],jnp.zeros((1,d)),cfg,identity['recipe'])
    teacher=flax.serialization.from_state_dict(template,saved['state']);assert int(teacher.base.q_updates)==1000000
    critic=teacher.base.agent.network.select('critic')
    del saved,template
    result=dict(schema='frozen_mixture_Q_refinement_v1',started=started,environment=parent['environment'],
        parent_result=str(parent_path),parent_result_sha256=sha(parent_path),parent_actor_path=str(model_path),parent_actor_sha256=sha(model_path),
        teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],cases=[dict(name=n,alpha=a,use_Q=q) for n,a,q in CASES],
        source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__,mixture.__file__,guidance.__file__]+[str(p) for p in flat.source_paths()]},
        source_cache_sha256=parent['pair_cache_sha256'],initial_parameters_identical=True,
        parent_actor_updates=20000,fresh_Adam_updates_per_branch=5000,Adam_reset_before_branch=True,
        learning_rate=.0003,minibatch_noise_seed=34311,gradient_probe_seed=34312,training_label_Q_aggregation='min',actor_Q_guidance_aggregation='mean',
        q_normalizer='stopgrad(mean ofcategoricallyweightedabsmeanQ),floor1e-6',
        objective='alpha*NLL/actiondim - E_mixture(meanQ)/Q_normalizer. Enumeratecategoricalcomponents; reparameterize1Gaussian/component.',
        teacher_optimizer_updates=0,registered_GPU_updates=0,GPU_queue_change=False,
        primary='Final5Krefinement actorCPU50, all4branchesinbothcases, no per-envorcheckpointselection.',
        scope='Warm frozen-teacher CPU refinement only. No joint/last3/9taskclaim; labels andofflinecachefixed, no evaluationinteraction learning.',actors={})
    write(args.output/'status.json',result)
    for name,alpha,use_q in CASES:
        actor=TrainState.create(definition,initial_params,optax.adam(.0003))
        record=dict(alpha=alpha,use_Q=use_q,snapshots=[],gradient_probes=[])
        result['actors'][name]=record

        def info(actor,params,b,t,z):
            p=actor(b,jnp.zeros((*b.shape[:-1],d+1)),params=params,return_parameters=True)
            return guidance.loss(p,t,b,z,critic,alpha,use_q)

        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,noise_key=jax.random.split(key,3)
                idx=jax.random.randint(index_key,(256,),0,count)
                noises=jax.random.normal(noise_key,(9,256,d))
                actor,_=actor.apply_loss_fn(lambda p:info(actor,p,obs[idx],y[idx],noises))
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)

        def gradients(actor):
            b,t=obs[count:count+256],y[count:count+256]
            z=jax.random.normal(jax.random.PRNGKey(34312),(9,256,d))
            fn=lambda p:info(actor,p,b,t,z)
            gn=jax.jit(jax.grad(lambda p:fn(p)[1]['distill_objective']))(actor.params)
            gq=jax.jit(jax.grad(lambda p:fn(p)[1]['Q_loss']))(actor.params)
            gt=jax.jit(jax.grad(lambda p:fn(p)[0]))(actor.params)
            n,q,total=tree_norm(gn),tree_norm(gq),tree_norm(gt)
            dot=sum(float((np.asarray(a,np.float64)*np.asarray(b,np.float64)).sum()) for a,b in zip(jax.tree_util.tree_leaves(gn),jax.tree_util.tree_leaves(gq)))
            residual=tree_norm(jax.tree_util.tree_map(lambda t,a,b:t-alpha*a-b,gt,gn,gq))
            assert all(math.isfinite(x) for x in [n,q,total,residual])
            return dict(distillation_gradient_L2=n,weighted_distillation_gradient_L2=alpha*n,Q_gradient_L2=q,
                Q_to_weighted_distillation_gradient_ratio=q/max(alpha*n,1e-20),
                gradient_cosine=dot/(n*q) if n*q else None,total_gradient_L2=total,
                gradient_additivity_residual_L2=residual)

        @jax.jit
        def metrics(actor):
            b,t=obs[count:count+256],y[count:count+256]
            z=jax.random.normal(jax.random.PRNGKey(34312),(9,256,d))
            _,values=info(actor,actor.params,b,t,z)
            return values
        previous,key=0,jax.random.PRNGKey(34311)
        for steps in (0,1000,5000):
            if steps:actor,key=fit(actor,key,steps-previous)
            assert int(actor.opt_state[0].count)==steps
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(actor.params))
            snapshot=dict(fresh_Adam_updates=steps,elapsed_seconds=time.time()-started,**{k:float(v) for k,v in jax.device_get(metrics(actor)).items()})
            record['snapshots'].append(snapshot)
            if steps in (0,5000):record['gradient_probes'].append(dict(fresh_Adam_updates=steps,**gradients(actor)))
            write(args.output/'status.json',result);print(json.dumps(dict(case=name,**snapshot)),flush=True);previous=steps
        payload=flax.serialization.to_state_dict(jax.device_get(actor.params));artifact_path=args.output/(name+'_private_actor.pkl')
        with artifact_path.open('xb') as stream:pickle.dump(dict(schema='private_frozen_mixture_refinement_v1',method=name,metadata=metadata,actor_params=payload,
            parent_actor_updates=20000,additional_fresh_Adam_updates=5000,parent_actor_sha256=sha(model_path),teacher_updates=0,source_sha256=result['source_sha256']),stream,pickle.HIGHEST_PROTOCOL)
        def policy(params,b,key):
            action_key,_=jax.random.split(key);z=jax.random.normal(action_key,(*b.shape[:-1],d+1))
            return definition.apply({'params':params},b,z)
        policy=jax.jit(policy,backend='cpu')
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        ev=evaluate(namespace,env,policy,5000,50);env.close()
        ev.update(protocol='warm_frozen_mixture_refinement_CPU50_v1',method=name,alpha=alpha,use_Q=use_q,
            optimizer_updates_during_evaluation=0,parent_actor_updates=20000,additional_fresh_Adam_updates=5000,
            actor_artifact_sha256=sha(artifact_path),reset_metadata=reset,not_joint_not_goal_result=True)
        ev_path=args.output/(name+'_CPU50_v1.json');write(ev_path,ev)
        record.update(final_CPU50=ev['success_rate'],evaluation_path=str(ev_path),evaluation_sha256=sha(ev_path),
            private_actor_path=str(artifact_path),private_actor_sha256=sha(artifact_path))
        write(args.output/'status.json',result);print(json.dumps(dict(case=name,success=ev['success_rate'])),flush=True)
    assert sha(checkpoint)==parent['teacher_checkpoint_sha256'] and sha(cache_path)==parent['pair_cache_sha256'] and sha(model_path)==entry['private_actor_sha256']
    result.update(status='complete',completed=time.time(),additional_student_Adam_updates=20000,evaluation_episodes=200)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
