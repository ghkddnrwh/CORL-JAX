"""Plain MDN architecture comparison on the exact audited524Kpair cache."""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import standalone_mixture_actor_v2 as portable
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
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
    parser=argparse.ArgumentParser();parser.add_argument('--scaling-case',type=Path,required=True);parser.add_argument('--depth',type=int,choices=(4,6),required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    args.output.mkdir(parents=True)
    pp=args.scaling_case/'result.json';parent=json.loads(pp.read_text());assert parent['status']=='complete'and parent['training_states']==524288
    for p,d in parent['source_sha256'].items():assert sha(p)==d,p
    cache=args.scaling_case/'nested_replay_cache_v1.npz';assert sha(cache)==parent['nested_cache_sha256']
    with np.load(cache)as x:obs=jnp.asarray(x['observations']);target=jnp.asarray(x['targets']);train=int(x['train_size'])
    with Path(parent['private_actor_path']).open('rb')as f:prior=pickle.load(f)
    metadata=dict(prior['metadata']);assert'observation_normalization'not in metadata
    metadata['hidden_dims']=[512]*args.depth;assert metadata['actor_forward_passes']==1 and metadata['generated_candidate_actions']==0
    definition=portable.definition(metadata);dim=metadata['action_dim']
    initial=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,dim+1)))['params']
    actor=TrainState.create(definition,initial,optax.adam(.0003))
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],parent_result=str(pp),parent_result_sha256=sha(pp),
        pair_cache_sha256=sha(cache),training_states=train,heldout_states=4096,hidden_dims=metadata['hidden_dims'],parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(initial)),
        actor_initialization_seed=34210,minibatch_seed=34211,learning_rate=.0003,batch_size=256,Adam_total=100000,
        loss='Unchanged clipped-mixture NLL/action_dimension, noQgradient/normalization/extraobjective.',
        baseline_256x4_CPU50=parent['success'],teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
        teacher_queries_or_updates=0,registered_GPU_updates=0,evaluation_interactions_used_for_learning=False,
        same_input_pair_labels_and_minibatch_indices_as_256x4_control=True,same_initial_function_claim=False,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,portable.__file__,mixture.__file__,*flat.source_paths()]},snapshots=[],
        scope='Frozen-teacher/Puzzleonly524Kstates/100KCPUactor, largerMLP only. Not joint1M, last3, nine-task, fresh-seed or isolatedspeed. Same fixed-label cache limitations apply.')
    write(args.output/'status.json',result)
    def fit(actor,key,steps):
        def update(carry,_):
            actor,key=carry;key,index_key,_=jax.random.split(key,3);idx=jax.random.randint(index_key,(256,),0,train)
            def loss(params):
                p=actor(obs[idx],jnp.zeros((256,dim+1)),params=params,return_parameters=True)
                nll=-mixture.log_probability(p,target[idx]).mean()/dim
                return nll,dict(NLL=nll)
            actor,_=actor.apply_loss_fn(loss)
            return (actor,key),None
        return jax.lax.scan(update,(actor,key),None,length=steps)[0]
    fit=jax.jit(fit,static_argnums=2);key=jax.random.PRNGKey(34211);previous=0
    for count in (0,1000,5000,20000,50000,100000):
        if count:actor,key=fit(actor,key,count-previous)
        assert int(actor.opt_state[0].count)==count
        assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor.params))
        p=actor(obs[train:],jnp.zeros((4096,dim+1)),return_parameters=True);weights=jax.nn.softmax(p['logits'],-1)
        entropy=-jnp.sum(weights*jax.nn.log_softmax(p['logits'],-1),-1)
        row=dict(actual_Adam=count,heldout_NLL_per_dimension=float(-mixture.log_probability(p,target[train:]).mean()/dim),
            effective_components_mean=float(jnp.exp(entropy).mean()),weighted_component_sigma_mean=float(jnp.mean(jnp.sum(weights[...,None]*p['scales'],axis=-2))),elapsed_seconds=time.time()-result['time'])
        result['snapshots'].append(row);write(args.output/'status.json',result);print(json.dumps(row),flush=True);previous=count
    params=flax.serialization.to_state_dict(jax.device_get(actor.params));artifact=args.output/'plain_MDN_private_actor_100000.pkl'
    with artifact.open('xb')as f:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_capacity_diagnostic_only',method='plain_MDN_512x'+str(args.depth),
        metadata=metadata,actor_params=params,actor_Adam_updates=100000,teacher_optimizer_updates_performed=0,teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    policy=portable.make_policy(metadata);namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
    env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    evaluated=evaluate(namespace,env,policy,100000,50);env.close()
    evaluated.update(time=time.time(),protocol='private_frozen_capacity_CPU50_v1',actor_artifact_sha256=sha(artifact),teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
        reset_metadata=reset,no_optimizer_updates=True,counts_toward_joint_or_goal=False)
    ep=args.output/'plain_MDN_100K_CPU50_v1.json';write(ep,evaluated)
    assert sha(cache)==parent['nested_cache_sha256']
    result.update(status='complete',completed=time.time(),success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep),private_actor_path=str(artifact),private_actor_sha256=sha(artifact))
    write(args.output/'result.json',result);write(args.output/'status.json',result);print(json.dumps(dict(success=evaluated['success_rate'],depth=args.depth)),flush=True)


if __name__=='__main__':main()
