"""Frozen CPU extraction: fresh labels, input conditioning, or ranking loss.

Students share four iid best9 labels per replay state. No rollout
state trains any model. None of these disposable models counts as joint RL.
"""
import argparse
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


METHODS=('NLL4_input_standardized','NLL4_rank1')


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6 and all(x.platform=='cpu'for x in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.study_case/'result.json';parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    cache_path=args.study_case/'frozen_replay_pairs_v1.npz';assert sha(cache_path)==parent['pair_cache_sha256']
    with np.load(cache_path)as cache:
        raw_obs=cache['observations'].copy();old_targets=cache['targets'].copy();packed=cache['packed_noises'].copy();train=int(cache['train_size'])
    obs=jnp.asarray(raw_obs);dim=old_targets.shape[-1];heldout=len(obs)-train
    conf=json.loads((Path(parent['teacher_run'])/'config.json').read_text())
    for p,d in conf['source_sha256'].items():assert sha(p)==d
    cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=flat.create(conf['identity']['seed'],obs[:1],jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==int(teacher.actor_updates)==1000000 and int(teacher.base.v_updates)==0
    agent=teacher.base.agent
    result=dict(time=time.time(),status='caching',environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        teacher_checkpoint_sha256=sha(checkpoint),parent_pair_cache_sha256=sha(cache_path),
        methods=list(args.methods),teacher_labels_per_state=4,train_states=train,heldout_states=heldout,
        same_random_initial_actor_seed=34210,same_minibatch_seed=34211,batch_size=256,learning_rate=.0003,Adam_per_method=20000,
        rank_loss='beta1 cross_entropy over log_pi(candidate)/action_dim; target uniform across exactly maximal online-minQ ties. NLL best9 label anchor retained. No Q derivative.',
        rank_limits='Auxiliary ranking changes the population objective; not a proper-score guarantee or exact best9-law matching. Candidate density ranking also depends on behavior density.',
        standardization='Training65536 states mean/std, floor.001; explicit affine preprocessing followed by exactly one state-only MLP. No float32 folding.',
        prior_control=parent['actors']['mixture9_NLL'],teacher_optimizer_updates=0,registered_GPU_updates=0,evaluation_data_used_for_learning=False,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,mixture.__file__,*flat.source_paths()]},actors={},
        scope='Frozen1Mteacher CPU20Kstudents, final50episode only. Not joint, last3, nine-task, fresh training seeds or isolated latency. No GPU recipe registered by this script.')
    write(args.output/'status.json',result)
    manifest_path=args.output.parent/'four_label_cache_manifest_v1.json'
    manifest=json.loads(manifest_path.read_text())
    case=manifest['cases'][args.study_case.name]
    extra_cache=Path(case['path']);assert sha(extra_cache)==case['sha256'] and case['draw0_exact_old_labels']
    with np.load(extra_cache)as cached:
        pools,values,labels=(cached[k].copy()for k in ('actions','online_minQ','labels'))
    assert pools.shape==(4,train,9,dim) and values.shape==(4,train,9) and labels.shape==(4,train,dim)
    assert all(np.isfinite(v).all()for v in (pools,values,labels))
    assert np.array_equal(labels[0],old_targets[:train])
    result.update(status='fitting',four_label_cache_sha256=sha(extra_cache),draw0_exact_old_labels=True,
        previous_cache_manifest=str(manifest_path),previous_cache_manifest_sha256=sha(manifest_path),
        teacher_queries_performed=0,previous_raw_NLL4_control=case['raw_CPU50_path'],
        previous_raw_NLL4_control_sha256=case['raw_CPU50_sha256'])
    write(args.output/'status.json',result)
    bank,qbank,target=(jnp.asarray(v)for v in (pools,values,labels))
    mean=jnp.asarray(raw_obs[:train].mean(0));scale=jnp.asarray(np.maximum(raw_obs[:train].std(0),.001))
    definition=mixture.MixtureActor(tuple(cfg['actor_hidden_dims']),dim,9,.001)
    initial=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,dim+1)))['params']
    for method in args.methods:
        standardized=method=='NLL4_input_standardized';ranking=method=='NLL4_rank1'
        transformed=(obs-mean)/scale if standardized else obs
        actor=TrainState.create(definition,initial,optax.adam(.0003))
        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,label_key=jax.random.split(key,3)
                indices=jax.random.randint(index_key,(256,),0,train)
                draw=jax.random.randint(label_key,(256,),0,4)
                t=target[draw,indices]
                def loss(parameters):
                    p=actor(transformed[indices],jnp.zeros((256,dim+1)),params=parameters,return_parameters=True)
                    nll=-mixture.log_probability(p,t).mean()/dim
                    if ranking:
                        candidates=jnp.moveaxis(bank[draw,indices],1,0)
                        logits=jnp.moveaxis(mixture.log_probability(p,candidates)/dim,0,1)
                        scores=qbank[draw,indices];ties=scores==scores.max(-1,keepdims=True)
                        weights=ties/ties.sum(-1,keepdims=True)
                        rank=-jnp.mean(jnp.sum(weights*jax.nn.log_softmax(logits,axis=-1),axis=-1))
                    else:rank=jnp.asarray(0.)
                    return nll+rank,dict(NLL=nll,rank_CE=rank)
                actor,_=actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)
        key=jax.random.PRNGKey(34211);previous=0;snapshots=[]
        for count in (0,1000,5000,20000):
            if count:actor,key=fit(actor,key,count-previous)
            assert int(actor.opt_state[0].count)==count
            assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor.params))
            p=actor(transformed[train:],jnp.zeros((heldout,dim+1)),return_parameters=True)
            nll=float(-mixture.log_probability(p,jnp.asarray(old_targets[train:])).mean()/dim)
            snapshots.append(dict(actual_Adam=count,heldout_NLL_per_dimension=nll));previous=count
            print(json.dumps(dict(stage='fit',method=method,count=count,heldout_NLL=nll)),flush=True)
        parameters=flax.serialization.to_state_dict(jax.device_get(actor.params))
        check_obs=transformed[train:train+256]
        old=actor(check_obs,jnp.zeros((256,dim+1)),return_parameters=True)
        new=definition.apply({'params':parameters},check_obs,jnp.zeros((256,dim+1)),return_parameters=True)
        export_difference=max(float(jnp.max(jnp.abs(old[k]-new[k])))for k in old)
        assert export_difference==0,export_difference
        metadata=dict(parent['actors']['mixture9_NLL']['metadata'])
        if standardized:metadata['observation_normalization']=dict(mean=np.asarray(mean).tolist(),scale=np.asarray(scale).tolist())
        artifact=args.output/(method+'_private_actor_20000.pkl')
        with artifact.open('xb')as stream:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_multilabel_diagnostic_only',
            method=method,metadata=metadata,actor_params=parameters,actor_Adam_updates=20000,
            teacher_optimizer_updates_performed=0,teacher_checkpoint_sha256=sha(checkpoint),source_sha256=result['source_sha256']),stream,pickle.HIGHEST_PROTOCOL)
        def policy(p,observation,key):
            observation=(observation-mean)/scale if standardized else observation
            action_key,_=jax.random.split(key);latent=jax.random.normal(action_key,(*observation.shape[:-1],dim+1))
            return definition.apply({'params':p},observation,latent)
        policy=jax.jit(policy,backend='cpu')
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=parameters)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,20000,50);env.close()
        evaluated.update(time=time.time(),method=method,environment=parent['environment'],protocol='private_multilabel_frozen_CPU50_v1',
            actor_artifact_sha256=sha(artifact),teacher_checkpoint_sha256=sha(checkpoint),optimizer_updates_during_evaluation=0,reset_metadata=reset,counts_toward_joint_or_goal=False)
        ep=args.output/(method+'_CPU50_v1.json');write(ep,evaluated)
        result['actors'][method]=dict(snapshots=snapshots,success=evaluated['success_rate'],private_actor_path=str(artifact),private_actor_sha256=sha(artifact),
            evaluation_path=str(ep),evaluation_sha256=sha(ep),explicit_export_max_parameter_difference=export_difference)
        write(args.output/'status.json',result);print(json.dumps(dict(stage='CPU50',method=method,success=evaluated['success_rate'])),flush=True)
    assert sha(checkpoint)==parent['teacher_checkpoint_sha256'] and sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),total_scratch_Adam=20000*len(args.methods),total_evaluation_episodes=50*len(args.methods))
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
