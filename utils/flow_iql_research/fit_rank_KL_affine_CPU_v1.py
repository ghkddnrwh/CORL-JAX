"""Continue actual100K NLL-router state for100K rank-KL improvement updates.

The optimizer moments and minibatch stream are continued, not reset. Teacher
and decoder remain frozen; four-way controls share each baseline snapshot.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import affine_latent_actor_v1 as affine
from utils.flow_iql_research import rank_latent_KL_loss_v1 as loss_core
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
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    proposal=json.loads(a.proposal.read_text());case=proposal['cases'][a.case]
    for path,digest in proposal['source_sha256'].items():assert sha(path)==digest,path
    parentpath=Path(case['parent_result']);assert sha(parentpath)==case['parent_result_sha256']
    parent=json.loads(parentpath.read_text());assert parent['status']=='complete'and parent['actual_router_Adam']==100000
    assert sha(parent['private_artifact'])==parent['private_artifact_sha256']and sha(parent['optimizer_artifact'])==parent['optimizer_sha256']
    with Path(parent['private_artifact']).open('rb')as f:base_actor=pickle.load(f)
    with Path(parent['optimizer_artifact']).open('rb')as f:base_optimizer=pickle.load(f)
    cache=Path(case['cache']);assert sha(cache)==parent['cache_sha256']
    with np.load(cache)as f:
        obs=jnp.asarray(f['observations'].copy());selected=jnp.asarray(f['selected_latents'].copy())
        bank_scores=jnp.asarray(f['scores'].copy());train=int(f['train_size'])
    assert train==524288 and len(obs)==train+4096
    metadata=base_actor['metadata'];d=metadata['action_dim'];decoder_params=base_actor['actor_params']['decoder']
    router=affine.ConditionalAffineRouter(tuple(metadata['router_hidden_dims']),d,4,1.)
    decoder=affine._Actor(tuple(metadata['decoder_hidden_dims']),d,False)
    template=TrainState.create(router,base_actor['actor_params']['router'],optax.adam(.0003))
    state=flax.serialization.from_state_dict(template,base_optimizer)
    assert int(state.opt_state[0].count)==100000
    for u,v in zip(jax.tree_util.tree_leaves(state.params),jax.tree_util.tree_leaves(base_actor['actor_params']['router'])):np.testing.assert_array_equal(u,v)
    checkpoint=Path(case['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    conf=json.loads((checkpoint.parent/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
    template=legacy.create(conf['identity']['seed'],obs[:1],jnp.zeros((1,d)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==1000000 and int(teacher.base.v_updates)==0
    critic=teacher.base.agent.network.select('critic');coefficient=case['forward_coefficient']
    kind=case.get('objective','rank_KL');assert kind in ('rank_KL','continued_NLL_only')
    a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],proposal_sha256=sha(a.proposal),
        source_sha256=proposal['source_sha256'],parent_result_sha256=sha(parentpath),
        baseline_actor_sha256=parent['private_artifact_sha256'],baseline_optimizer_sha256=parent['optimizer_sha256'],
        cache_sha256=sha(cache),teacher_checkpoint_sha256=sha(checkpoint),
        baseline_router_Adam=100000,additional_router_target_Adam=100000,decoder_Adam_performed=0,teacher_Adam_performed=0,
        objective_kind=kind,forward_coefficient=coefficient,smoothing=.3,best_of=9,bank_size=9,learning_rate=.0003,batch_size=256,
        full_optimizer_moments_resumed=True,minibatch_seed=34211,minibatch_RNG_advanced_updates=100000,
        evaluation_data_used_for_learning=False,snapshots=[],evaluations={},
        reused_controls=parent['reused_controls'],baseline_NLL_evaluations=parent['evaluations'],
        scope='Frozen-teacher posthoc CPU100Kadditional policy improvement, actualtotal200K. Finite smooth-rank surrogate, not exact measured KL; not nativejoint1M/9task/fresh/speed.')
    write(a.output/'status.json',result)
    @jax.jit
    def advance(key):return jax.lax.fori_loop(0,100000,lambda _,k:jax.random.split(k,3)[0],key)
    key=advance(jax.random.PRNGKey(34211))
    @jax.jit
    def fit1000(state,key):
        def step(carry,_):
            s,k=carry;k,ik,_=jax.random.split(k,3);index=jax.random.randint(ik,(256,),0,train)
            epsilon=jax.random.normal(jax.random.fold_in(k,88201),(256,d))
            def loss(params):
                if kind=='continued_NLL_only':
                    value=-affine.log_probability(router,params,obs[index],selected[index]).mean()/d
                    return value,dict(loss=value)
                return loss_core.objective(router,params,obs[index],epsilon,decoder,decoder_params,
                    critic,bank_scores[index],selected[index],coefficient,9,.3)
            s,_=s.apply_loss_fn(loss);return(s,k),None
        return jax.lax.scan(step,(state,key),None,length=1000)[0]
    @jax.jit
    def diagnose(params,x,z,qs,key):
        epsilon=jax.random.normal(key,z.shape)
        return loss_core.objective(router,params,x,epsilon,decoder,decoder_params,critic,qs,z,coefficient,9,.3)[1]
    for additional in range(0,100001,1000):
        if additional:state,key=fit1000(state,key)
        assert int(state.opt_state[0].count)==100000+additional
        if additional%5000==0:
            assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
            batches=[]
            for start in range(train,train+4096,256):
                batches.append(jax.device_get(diagnose(state.params,obs[start:start+256],selected[start:start+256],bank_scores[start:start+256],jax.random.PRNGKey(start+413))))
            fields={k:float(np.mean([v[k]for v in batches]))for k in batches[0]}
            fields['sampled_log_ratio_max']=float(max(v['sampled_log_ratio_max']for v in batches))
            fields['diagnostic_rank_objective']=fields.pop('loss')
            fields['training_objective_kind']=kind
            fields.update(actual_router_Adam=100000+additional,additional_Adam=additional,elapsed_seconds=time.time()-result['time'])
            result['snapshots'].append(fields);write(a.output/'status.json',result);print(json.dumps(fields),flush=True)
    params=dict(router=flax.serialization.to_state_dict(jax.device_get(state.params)),decoder=decoder_params)
    metadata=dict(metadata,schema='private_rank_KL_affine_latent_actor_v1',objective_kind=kind,forward_coefficient=coefficient)
    artifact=a.output/'rank_KL_actor_200K_private_v1.pkl'
    with artifact.open('xb')as f:pickle.dump(dict(schema=metadata['schema'],metadata=metadata,actor_params=params,
        actual_router_Adam=200000,baseline_router_Adam=100000,additional_router_Adam=100000,decoder_Adam_performed=0,
        source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    opt=a.output/'rank_KL_optimizer_200K_private_v1.pkl'
    with opt.open('xb')as f:pickle.dump(flax.serialization.to_state_dict(jax.device_get(state)),f,pickle.HIGHEST_PROTOCOL)
    def exact_flow(_,x,key):
        ak,_=jax.random.split(key);epsilon=jax.random.normal(ak,(*x.shape[:-1],d))
        z=router.apply({'params':state.params},x,epsilon)['latent']
        return legacy.lean.flow(teacher.base.agent,x,z,10)
    policies=dict(composite=affine.make_policy(metadata),learned_router_exact_Flow=jax.jit(exact_flow,backend='cpu'))
    for mode,policy in policies.items():
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,200000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_rank_KL_affine_CPU50_v1',mode=mode,reset_metadata=reset,
            actor_artifact_sha256=sha(artifact),no_optimizer_updates=True,counts_toward_native_goal=False,
            deployable_actor_only=mode=='composite',hybrid_uses_teacher=mode!='composite')
        ep=a.output/(mode+'_CPU50_v1.json');write(ep,evaluated)
        result['evaluations'][mode]=dict(success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(a.output/'status.json',result);print(json.dumps(dict(mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(checkpoint)==result['teacher_checkpoint_sha256']and sha(cache)==result['cache_sha256']
    result.update(status='complete',completed=time.time(),actual_router_Adam=200000,additional_router_Adam=100000,
        private_artifact=str(artifact),private_artifact_sha256=sha(artifact),optimizer_artifact=str(opt),optimizer_sha256=sha(opt))
    write(a.output/'result.json',result);write(a.output/'status.json',result)


if __name__=='__main__':main()
