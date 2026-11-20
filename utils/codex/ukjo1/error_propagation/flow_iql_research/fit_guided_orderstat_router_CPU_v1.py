"""Q-guided maximum-law selector: freeze the already fitted fast decoder.

Reuse identical fixed-teacher labels, initial hidden trunk, minibatch order,
Adam count and scalar settings. Only the router density family changes; its ideal density obeys q<=9*prior.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import orderstat_latent_actor_v1 as order
from utils.flow_iql_research import latent_selected_actor_v1 as diagonal
from utils.flow_iql_research import latent_order_guidance_v1 as guidance
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.flow_iql_research.vendor.flax_utils import TrainState
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
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    proposal=json.loads(a.proposal.read_text());case=proposal['cases'][a.case]
    for path,digest in proposal['source_sha256'].items():assert sha(path)==digest,path
    parent_path=Path(case['parent_result']);assert sha(parent_path)==case['parent_result_sha256']
    parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    assert sha(parent['private_artifact'])==parent['private_artifact_sha256']
    with Path(parent['private_artifact']).open('rb')as f:old_actor=pickle.load(f)
    cache=Path(case['cache']);assert sha(cache)==parent['label_cache_sha256']
    with np.load(cache)as f:
        obs=f['observations'].copy();target=f['selected_latents'].copy();train=int(f['train_size'])
        proposal_actions=jnp.asarray(f['actions'].copy())
    assert train==524288 and len(obs)==train+4096
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    conf=json.loads((checkpoint.parent/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    d=parent['metadata']['action_dim'];hidden=tuple(parent['metadata']['hidden_dims'])
    template=flat.create(0,jnp.asarray(obs[:1]),jnp.zeros((1,d)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==1000000 and int(teacher.base.v_updates)==0
    agent=teacher.base.agent;observations=jnp.asarray(obs);targets=jnp.asarray(target)
    decoder_def=order._Actor(hidden,d,False);decoder_params=old_actor['actor_params']['decoder']
    alpha=case['selector_alpha']
    router_def=order.OrderStatisticRouter(hidden,d,9,9)
    initial=router_def.init(jax.random.PRNGKey(34210),observations[:1])['params']
    diag_def=diagonal.MixtureActor(hidden,d,9,.1)
    diag_initial=diag_def.init(jax.random.PRNGKey(34210),observations[:1],jnp.zeros((1,d+1)),return_parameters=True)['params']
    for layer in range(len(hidden)):
        for field in ('kernel','bias'):
            assert np.array_equal(np.asarray(initial['Dense_'+str(layer)][field]),np.asarray(diag_initial['Dense_'+str(layer)][field]))
    state=TrainState.create(router_def,initial,optax.adam(.0003))
    a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],
        proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        parent_result_sha256=sha(parent_path),cache_sha256=sha(cache),
        teacher_checkpoint_sha256=sha(checkpoint),frozen_decoder_parent_sha256=sha(parent['private_artifact']),
        initial_hidden_trunk_matches_diagonal=True,training_states=train,heldout_states=4096,
        selector_alpha=alpha,guidance_updates_decoder=False,training_guidance_components=10,
        unguided_orderstat_control=case['unguided_orderstat_control'],
        actual_router_target_Adam=100000,decoder_Adam_performed=0,teacher_Adam_performed=0,
        learning_rate=.0003,batch_size=256,directions=9,components_including_prior=10,best_of=9,
        initialization_seed=34210,minibatch_seed=34211,
        evaluation_data_used_for_learning=False,snapshots=[],evaluations={},
        reused_controls={k:parent['evaluations'][k]for k in ('exact_winner_fast_decoder','prior_fast_decoder')},
        scope='Private fixed-teacher Q-guided selector ablation with identical frozen decoder and cache; NLL anchors selection, actor-value gradient passes only through one-step decoder inputs. No native joint9task/fresh/final-speed claim.')
    write(a.output/'status.json',result)
    @jax.jit
    def fit1000(state,key):
        def update(carry,_):
            s,k=carry;k,ik,_=jax.random.split(k,3)
            idx=jax.random.randint(ik,(256,),0,train)
            def objective(params):
                x=observations[idx];pars=router_def.apply({'params':params},x)
                nll=-order.log_probability(pars,targets[idx]).mean()/d
                actions=jnp.moveaxis(proposal_actions[idx],1,0)
                ob=jnp.broadcast_to(x,(9,*x.shape))
                heads=agent.network.select('critic')(ob,actions=actions)
                gaussian=jax.random.normal(jax.random.fold_in(k,49385),(256,d))
                qloss,qinfo=guidance.expected_Q_loss(pars,gaussian,x,decoder_def,decoder_params,
                    agent.network.select('critic'),heads,9)
                value=alpha*nll+qloss
                return value,dict(loss=value,NLL=nll,Q_loss=qloss)
            s,_=s.apply_loss_fn(objective)
            return (s,k),None
        return jax.lax.scan(update,(state,key),None,length=1000)[0]
    @jax.jit
    def diagnose(params,x,z):
        pars=router_def.apply({'params':params},x)
        nll=-order.log_probability(pars,z).mean()/d
        base=-.5*jnp.sum(z*z,axis=-1)-.5*d*jnp.log(2*jnp.pi)
        prior_weight=jax.nn.softmax(pars['logits'],axis=-1)[...,0].mean()
        ratio=order.log_probability(pars,z)-base
        return jnp.array([nll,prior_weight,jnp.max(ratio)])
    key=jax.random.PRNGKey(34211)
    for count in range(0,100001,1000):
        if count:state,key=fit1000(state,key)
        assert int(state.opt_state[0].count)==count
        if count%5000==0:
            assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
            values=[]
            for offset in range(train,train+4096,256):
                values.append(np.asarray(diagnose(state.params,observations[offset:offset+256],targets[offset:offset+256])))
            v=np.stack(values)
            row=dict(actual_Adam=count,heldout_NLL_per_dim=float(v[:,0].mean()),
                mean_prior_weight=float(v[:,1].mean()),max_log_density_ratio=float(v[:,2].max()),
                elapsed_seconds=time.time()-result['time'])
            assert row['max_log_density_ratio']<=math.log(9)+1e-5
            result['snapshots'].append(row);write(a.output/'status.json',result);print(json.dumps(row),flush=True)
    params=dict(router=flax.serialization.to_state_dict(jax.device_get(state.params)),
                decoder=old_actor['actor_params']['decoder'])
    metadata=dict(parent['metadata'],schema='private_guided_orderstat_selected_latent_actor_v1',
        architecture='Q_guided_maximum_law_selector_then_frozen_one_step_decoder',
        selector_alpha=alpha,guidance_updates_decoder=False,
        directions=9,best_of=9,components=10,scalar_inverse_CDF_calls=1,noise_dim=d+1,
        sampler_projection_tail_guard=10.,ideal_Gaussian_tail_guard_mass=math.erfc(10/math.sqrt(2)))
    metadata.pop('standard_deviation_floor',None)
    artifact=a.output/'guided_orderstat_latent_actor_100K_private_v1.pkl'
    with artifact.open('xb')as f:pickle.dump(dict(schema=metadata['schema'],metadata=metadata,
        actor_params=params,actual_router_Adam=100000,decoder_Adam_performed=0,
        decoder_parent_Adam=100000,frozen_decoder_parent_sha256=sha(parent['private_artifact']),
        source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    opt=a.output/'guided_orderstat_router_optimizer_100K_private_v1.pkl'
    with opt.open('xb')as f:pickle.dump(flax.serialization.to_state_dict(jax.device_get(state)),f,pickle.HIGHEST_PROTOCOL)
    policy=order.make_policy(metadata)
    fixtures=observations[train:train+16];fixture_key=jax.random.PRNGKey(411)
    noise=jax.random.normal(jax.random.split(fixture_key)[0],(16,d+1))
    latent=order.sample_latent(router_def.apply({'params':state.params},fixtures),noise)
    decoder_def=order._Actor(hidden,d,False);decoder_params=params['decoder']
    manual=jnp.clip(decoder_def.apply({'params':decoder_params},fixtures,latent),-1,1)
    difference=float(np.max(np.abs(np.asarray(policy(params,fixtures,fixture_key))-np.asarray(manual))))
    assert difference<=1e-6
    result['composite_manual_max_abs_difference']=difference
    def exact_flow(_,x,key):
        action_key,_=jax.random.split(key)
        z=order.sample_latent(router_def.apply({'params':state.params},x),jax.random.normal(action_key,(*x.shape[:-1],d+1)))
        return flat.lean.flow(agent,x,z,10)
    policies=dict(composite=policy,learned_router_exact_Flow=jax.jit(exact_flow,backend='cpu'))
    for mode,fn in policies.items():
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(result['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42);evaluated=evaluate(namespace,env,fn,100000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_guided_orderstat_latent_router_CPU50_v1',mode=mode,
            reset_metadata=reset,actor_artifact_sha256=sha(artifact),no_optimizer_updates=True,
            evaluation_data_used_for_learning=False,counts_toward_native_goal=False,
            deployable_actor_only=mode=='composite',hybrid_uses_teacher=mode!='composite')
        ep=a.output/(mode+'_CPU50_v1.json');write(ep,evaluated)
        result['evaluations'][mode]=dict(success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(a.output/'status.json',result);print(json.dumps(dict(mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(cache)==result['cache_sha256']and sha(checkpoint)==result['teacher_checkpoint_sha256']
    result.update(status='complete',completed=time.time(),actual_router_Adam=100000,
        private_artifact=str(artifact),private_artifact_sha256=sha(artifact),
        optimizer_artifact=str(opt),optimizer_sha256=sha(opt),new_CPU_evaluation_episodes=100)
    write(a.output/'result.json',result);write(a.output/'status.json',result)


if __name__=='__main__':main()
