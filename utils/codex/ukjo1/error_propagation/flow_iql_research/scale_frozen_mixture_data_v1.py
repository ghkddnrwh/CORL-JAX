"""Plain MDN scaling diagnostic: 65K or524Kreplay states,100KCPUupdates.

Same original heldout4096states and20Kreproduction control. No new loss,
normalization, critic gradient, or evaluation-data training is introduced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
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


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-states',type=int,choices=(65536,524288),required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.study_case/'result.json';parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    cache=args.study_case/'frozen_replay_pairs_v1.npz';assert sha(cache)==parent['pair_cache_sha256']
    with np.load(cache)as x:
        old_train=int(x['train_size']);old_obs=x['observations'].copy();old_targets=x['targets'].copy();old_indices=x['selected_replay_indices'].copy()
    assert old_train==65536 and len(old_obs)==65536+4096
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    conf=json.loads((Path(parent['teacher_run'])/'config.json').read_text())
    for p,d in conf['source_sha256'].items():assert sha(p)==d
    cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    dim=old_targets.shape[-1]
    template=flat.create(conf['identity']['seed'],jnp.asarray(old_obs[:1]),jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    teacher=flax.serialization.from_state_dict(template,saved['state']);agent=teacher.base.agent
    assert int(teacher.base.q_updates)==int(teacher.actor_updates)==1000000 and int(teacher.base.v_updates)==0
    result=dict(time=time.time(),status='preparing',environment=parent['environment'],training_states=args.training_states,
        heldout_states=4096,heldout_exactly_original=True,old65536_training_states_nested=True,training_labels_per_state=1,
        actor_seed=34210,minibatch_seed=34211,Adam_total=100000,batch_size=256,learning_rate=.0003,
        teacher_checkpoint_sha256=sha(checkpoint),original_cache_sha256=sha(cache),parent_result_sha256=sha(parent_path),
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,mixture.__file__,*flat.source_paths()]},
        no_teacher_or_GPU_updates=True,evaluation_data_used_for_learning=False,
        scope='Data/budget scaling feasibility of plain clippedMDN frozen extraction on Puzzle only. Not joint1M, nine-task, last3, fresh-seed or speed. Same global actor architecture/loss; final100KCPU50only.',snapshots=[])
    write(args.output/'status.json',result)
    observations=old_obs[:old_train];targets=old_targets[:old_train];indices=old_indices[:old_train]
    if args.training_states>old_train:
        env,data,_=ogbench.make_env_and_datasets(parent['environment']);env.close()
        assert len(data['actions'])>=args.training_states+4096
        seen={hashlib.sha256(row.tobytes()).digest()for row in old_obs}
        selected=[];extra=[];attempted=0;random=np.random.RandomState(49311)
        while len(extra)<args.training_states-old_train:
            batch=random.choice(len(data['actions']),size=8192,replace=False);attempted+=len(batch)
            assert attempted<100*args.training_states,'Insufficient unique replay states'
            for index in batch:
                row=np.asarray(data['next_observations'][index],np.float32);key=hashlib.sha256(row.tobytes()).digest()
                if key in seen:continue
                seen.add(key);extra.append(row.copy());selected.append(int(index))
                if len(extra)==args.training_states-old_train:break
        del data,seen
        extra=np.stack(extra);extra_targets=[]
        @jax.jit
        def label(observation,key):
            broad,noise=flat.lean.bank(agent,observation,key,9)
            actions=flat.lean.flow(agent,broad,noise,10);q=agent.network.select('critic')(broad,actions=actions).min(0)
            return flat.gather(actions,q.argmax(0))
        for chunk,offset in enumerate(range(0,len(extra),256)):
            t=np.asarray(label(jnp.asarray(extra[offset:offset+256]),jax.random.fold_in(jax.random.PRNGKey(49312),chunk)))
            extra_targets.append(t)
            if chunk%128==0:print(json.dumps(dict(stage='additional_labels',states=offset+len(t))),flush=True)
        observations=np.concatenate([observations,extra]);targets=np.concatenate([targets,*extra_targets]);indices=np.concatenate([indices,np.asarray(selected)])
    observations=np.concatenate([observations,old_obs[old_train:]]);targets=np.concatenate([targets,old_targets[old_train:]])
    assert len(observations)==args.training_states+4096 and np.isfinite(targets).all()
    assert np.array_equal(observations[args.training_states:],old_obs[old_train:])
    assert np.array_equal(observations[:old_train],old_obs[:old_train]) and np.array_equal(targets[:old_train],old_targets[:old_train])
    nested=args.output/'nested_replay_cache_v1.npz';np.savez_compressed(nested,observations=observations,targets=targets,training_indices=indices,train_size=args.training_states)
    result.update(status='fitting',nested_cache_sha256=sha(nested),teacher_queries_for_new_states=args.training_states-old_train,heldout_training_overlap_exact=False)
    write(args.output/'status.json',result)
    obs,y=jnp.asarray(observations),jnp.asarray(targets);train=args.training_states
    definition=mixture.MixtureActor(tuple(cfg['actor_hidden_dims']),dim,9,.001)
    initial=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,dim+1)))['params']
    actor=TrainState.create(definition,initial,optax.adam(.0003))
    def fit(actor,key,steps):
        def update(carry,_):
            actor,key=carry;key,index_key,_=jax.random.split(key,3);idx=jax.random.randint(index_key,(256,),0,train)
            def loss(parameters):
                p=actor(obs[idx],jnp.zeros((256,dim+1)),params=parameters,return_parameters=True)
                nll=-mixture.log_probability(p,y[idx]).mean()/dim
                return nll,dict(NLL=nll)
            actor,_=actor.apply_loss_fn(loss)
            return (actor,key),None
        return jax.lax.scan(update,(actor,key),None,length=steps)[0]
    fit=jax.jit(fit,static_argnums=2)
    previous=0;key=jax.random.PRNGKey(34211)
    for count in (0,1000,5000,20000,50000,100000):
        if count:actor,key=fit(actor,key,count-previous)
        assert int(actor.opt_state[0].count)==count
        assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor.params))
        p=actor(obs[train:],jnp.zeros((4096,dim+1)),return_parameters=True)
        row=dict(actual_Adam=count,heldout_NLL_per_dimension=float(-mixture.log_probability(p,y[train:]).mean()/dim))
        if train==65536 and count==20000:
            with Path(parent['actors']['mixture9_NLL']['private_actor_path']).open('rb')as stream:control=pickle.load(stream)
            delta=max(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))for a,b in zip(jax.tree_util.tree_leaves(actor.params),jax.tree_util.tree_leaves(control['actor_params'])))
            assert delta<1e-6,delta;row['parameter_difference_from_prior_exact20K_control']=delta
        result['snapshots'].append(row);write(args.output/'status.json',result);print(json.dumps(row),flush=True);previous=count
    payload=flax.serialization.to_state_dict(jax.device_get(actor.params));metadata=parent['actors']['mixture9_NLL']['metadata']
    artifact=args.output/'plain_MDN_private_actor_100000.pkl'
    with artifact.open('xb')as stream:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_scaling_diagnostic_only',method='plain_MDN',
        metadata=metadata,actor_params=payload,actor_Adam_updates=100000,teacher_optimizer_updates_performed=0,teacher_checkpoint_sha256=sha(checkpoint),source_sha256=result['source_sha256']),stream,pickle.HIGHEST_PROTOCOL)
    def policy(p,observation,key):
        action_key,_=jax.random.split(key);latent=jax.random.normal(action_key,(*observation.shape[:-1],dim+1))
        return definition.apply({'params':p},observation,latent)
    policy=jax.jit(policy,backend='cpu');namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    evaluated=evaluate(namespace,env,policy,100000,50);env.close()
    evaluated.update(time=time.time(),protocol='private_plainMDN_scaling_CPU50_v1',training_states=train,actor_artifact_sha256=sha(artifact),teacher_checkpoint_sha256=sha(checkpoint),reset_metadata=reset,no_optimizer_updates=True,counts_toward_joint_or_goal=False)
    ep=args.output/'plain_MDN_100K_CPU50_v1.json';write(ep,evaluated)
    assert sha(checkpoint)==parent['teacher_checkpoint_sha256'] and sha(cache)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep),private_actor_path=str(artifact),private_actor_sha256=sha(artifact))
    write(args.output/'result.json',result);write(args.output/'status.json',result)
    print(json.dumps(dict(success=evaluated['success_rate'],training_states=train)),flush=True)


if __name__=='__main__':main()
