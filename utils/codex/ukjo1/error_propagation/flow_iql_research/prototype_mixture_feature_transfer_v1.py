"""Frozen-teacher CPU feasibility of reusing BCFlow state features.

This is not a jointly trained algorithm. No rollout observations enter fitting.
Both branches retain one state-only MLP and the same clipped-mixture NLL.
"""
import argparse
import copy
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


METHODS=('flow_init_train_all','flow_encoder_frozen')


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6 and all(x.platform=='cpu'for x in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.study_case/'result.json';parent=json.loads(parent_path.read_text())
    assert parent['status']=='complete'
    cp=Path(parent['teacher_checkpoint']);assert sha(cp)==parent['teacher_checkpoint_sha256']
    cache_path=args.study_case/'frozen_replay_pairs_v1.npz';assert sha(cache_path)==parent['pair_cache_sha256']
    with np.load(cache_path)as cache:
        observations=cache['observations'].copy();targets=cache['targets'].copy();train=int(cache['train_size']);distribution_targets=cache['distribution_eval_targets'].copy()
    obs,y=jnp.asarray(observations),jnp.asarray(targets);dim=targets.shape[-1];heldout=len(obs)-train
    conf=json.loads((Path(parent['teacher_run'])/'config.json').read_text())
    for p,d in conf['source_sha256'].items():assert sha(p)==d
    cfg=conf['identity']['config'];assert cfg['encoder']is None and cfg['actor_layer_norm']is False
    with cp.open('rb')as stream:saved=pickle.load(stream)
    assert saved['step']==1000000 and int(saved['state']['base']['q_updates'])==1000000 and int(saved['state']['base']['v_updates'])==0
    flow=saved['state']['base']['agent']['network']['params']['modules_actor_flow']['mlp']
    definition=mixture.MixtureActor(tuple(cfg['actor_hidden_dims']),dim,9,.001)
    random_params=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,dim+1)))['params']
    initial=copy.deepcopy(flax.serialization.to_state_dict(random_params))
    for i in range(4):
        source=flow[f'Dense_{i}'];initial[f'Dense_{i}']={k:jnp.asarray(v)for k,v in source.items()}
        if i==0:
            assert source['kernel'].shape[0]==obs.shape[-1]+dim+1
            initial['Dense_0']['kernel']=initial['Dense_0']['kernel'][:obs.shape[-1]]
    # Same output head as the previously evaluated random-initialized MDN.
    assert all(np.array_equal(np.asarray(v),np.asarray(random_params['Dense_4'][k]))for k,v in initial['Dense_4'].items())
    a,b=obs[:256],jnp.concatenate([obs[:256],jnp.zeros((256,dim+1))],axis=-1)
    for i in range(4):
        a=jax.nn.gelu(a@initial[f'Dense_{i}']['kernel']+initial[f'Dense_{i}']['bias'])
        b=jax.nn.gelu(b@flow[f'Dense_{i}']['kernel']+flow[f'Dense_{i}']['bias'])
    diff=float(jnp.max(jnp.abs(a-b)));assert diff<1e-4,diff
    result=dict(time=time.time(),status='running',environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        teacher_checkpoint_sha256=sha(cp),cache_sha256=sha(cache_path),methods=list(METHODS),
        random_init20K_control=parent['actors']['mixture9_NLL'],feature_initialization='Four BCFlow hidden layers evaluated at zero action-noise/time. First kernel uses observation rows only; output mixture head unchanged.',
        zero_auxiliary_hidden_feature_max_difference=diff,training_observations=train,heldout_observations=heldout,updates_per_method=20000,
        minibatch_noise_seed=34211,batch_size=256,learning_rate=.0003,likelihood='same clippedGaussian mixture NLL/action_dimension',
        teacher_optimizer_updates=0,registered_GPU_updates=0,evaluation_interactions_used_for_learning=False,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,mixture.__file__,*flat.source_paths()]},
        scope='CPU frozen1Mteacher/20KfreshAdam each, finalCPU50. Not joint, last3, nine-task or goal evidence. Same head/architecture/cache/minibatches, encoder initialization differs.',actors={})
    write(args.output/'status.json',result)
    for method in METHODS:
        actor=TrainState.create(definition,initial,optax.adam(.0003))
        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,_=jax.random.split(key,3)
                idx=jax.random.randint(index_key,(256,),0,train)
                def loss(params):
                    if method=='flow_encoder_frozen':params={k:v if k=='Dense_4'else jax.tree_util.tree_map(jax.lax.stop_gradient,v)for k,v in params.items()}
                    p=actor(obs[idx],jnp.zeros((256,dim+1)),params=params,return_parameters=True)
                    value=-mixture.log_probability(p,y[idx]).mean()/dim
                    return value,dict(NLL_per_dimension=value)
                actor,_=actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)
        snapshots=[];key=jax.random.PRNGKey(34211);previous=0
        for count in (0,1000,5000,20000):
            if count:actor,key=fit(actor,key,count-previous)
            assert int(actor.opt_state[0].count)==count
            assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(actor.params))
            if method=='flow_encoder_frozen':
                for i in range(4):assert all(np.array_equal(np.asarray(v),np.asarray(initial[f'Dense_{i}'][k]))for k,v in actor.params[f'Dense_{i}'].items())
            p=actor(obs[train:],jnp.zeros((heldout,dim+1)),return_parameters=True)
            nll=float(-mixture.log_probability(p,y[train:]).mean()/dim)
            snapshots.append(dict(actual_Adam=count,heldout_NLL_per_dimension=nll));previous=count
            print(json.dumps(dict(method=method,step=count,heldout_NLL=nll)),flush=True)
        payload=flax.serialization.to_state_dict(jax.device_get(actor.params))
        metadata=parent['actors']['mixture9_NLL']['metadata']
        path=args.output/(method+'_private_actor_20000.pkl')
        with path.open('xb')as stream:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_teacher_feature_transfer_diagnostic_only',
            method=method,metadata=metadata,actor_params=payload,actor_Adam_updates=20000,teacher_optimizer_updates_performed=0,
            teacher_checkpoint_sha256=sha(cp),source_sha256=result['source_sha256']),stream,pickle.HIGHEST_PROTOCOL)
        def policy(params,observation,key):
            action_key,_=jax.random.split(key);latent=jax.random.normal(action_key,(*observation.shape[:-1],dim+1))
            return definition.apply({'params':params},observation,latent)
        policy=jax.jit(policy,backend='cpu')
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,20000,50);env.close()
        evaluated.update(time=time.time(),method=method,environment=parent['environment'],protocol='private_frozen_teacher_feature_transfer_CPU50_v1',
            actor_artifact_sha256=sha(path),teacher_checkpoint_sha256=sha(cp),optimizer_updates_during_evaluation=0,reset_metadata=reset,counts_toward_joint_or_goal=False)
        ep=args.output/(method+'_CPU50_v1.json');write(ep,evaluated)
        result['actors'][method]=dict(snapshots=snapshots,success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep),private_actor_path=str(path),private_actor_sha256=sha(path))
        write(args.output/'status.json',result);print(json.dumps(dict(method=method,success=evaluated['success_rate'])),flush=True)
    assert sha(cp)==parent['teacher_checkpoint_sha256']and sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),total_scratch_Adam=40000,total_evaluation_episodes=100)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
