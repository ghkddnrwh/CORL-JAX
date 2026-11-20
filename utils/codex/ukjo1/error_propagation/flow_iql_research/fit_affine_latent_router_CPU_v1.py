"""Proper likelihood fitting of an expressive, bank-free latent transport.

Keep the same frozen successful action decoder, winning-latent labels, data
split, batch order and optimizer settings as previous router experiments.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import affine_latent_actor_v1 as affine
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
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    proposal=json.loads(a.proposal.read_text());case=proposal['cases'][a.case]
    for path,digest in proposal['source_sha256'].items():assert sha(path)==digest,path
    pp=Path(case['parent_result']);assert sha(pp)==case['parent_result_sha256'];parent=json.loads(pp.read_text())
    assert parent['status']=='complete'and sha(parent['private_artifact'])==parent['private_artifact_sha256']
    with Path(parent['private_artifact']).open('rb')as f:old=pickle.load(f)
    cache=Path(case['cache']);assert sha(cache)==parent['label_cache_sha256']
    with np.load(cache)as f:
        obs=jnp.asarray(f['observations'].copy());targets=jnp.asarray(f['selected_latents'].copy());train=int(f['train_size'])
    assert train==524288 and len(obs)==train+4096
    d=targets.shape[-1];width=case['router_width'];hidden=(width,width)
    decoder_hidden=tuple(parent['metadata']['hidden_dims'])
    definition=affine.ConditionalAffineRouter(hidden,d,4,1.)
    initial=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,d)))['params']
    check=definition.apply({'params':initial},obs[:4],targets[:4])
    np.testing.assert_array_equal(check['latent'],targets[:4]);np.testing.assert_array_equal(check['logdet'],0)
    actor=TrainState.create(definition,initial,optax.adam(.0003));a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],
        proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        parent_result_sha256=sha(pp),cache_sha256=sha(cache),teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
        frozen_decoder_parent_sha256=sha(parent['private_artifact']),training_states=train,heldout_states=4096,
        actual_router_target_Adam=100000,decoder_Adam_performed=0,teacher_Adam_performed=0,
        learning_rate=.0003,batch_size=256,initialization_seed=34210,minibatch_seed=34211,
        router_hidden_dims=list(hidden),coupling_layers=4,log_scale_bound=1.,identity_initialization_exact=True,
        objective='Exact conditional latent likelihood, divided by latent dimension; noQloss.',
        evaluation_data_used_for_learning=False,snapshots=[],evaluations={},
        reused_controls={k:parent['evaluations'][k]for k in ('exact_winner_fast_decoder','prior_fast_decoder')},
        scope='Private frozen-teacher CPU100K transport-fit feasibility. Bank-free actor, one Gaussian, four finite coupling blocks and one decoder. Not native1M/9task/fresh/isolatedspeed.')
    write(a.output/'status.json',result)
    @jax.jit
    def fit1000(state,key):
        def update(carry,_):
            s,k=carry;k,ik,_=jax.random.split(k,3);idx=jax.random.randint(ik,(256,),0,train)
            def loss(params):
                value=-affine.log_probability(definition,params,obs[idx],targets[idx]).mean()/d
                return value,dict(loss=value)
            s,_=s.apply_loss_fn(loss);return(s,k),None
        return jax.lax.scan(update,(state,key),None,length=1000)[0]
    @jax.jit
    def diagnose(params,x,z,key):
        nll=-affine.log_probability(definition,params,x,z).mean()/d
        prior_CE=jnp.mean(.5*jnp.sum(z*z,-1)+.5*d*jnp.log(2*jnp.pi))
        epsilon=jax.random.normal(key,z.shape);forward=definition.apply({'params':params},x,epsilon)
        sample=forward['latent']
        logq=-.5*jnp.sum(epsilon*epsilon,-1)-.5*d*jnp.log(2*jnp.pi)-forward['logdet']
        logprior=-.5*jnp.sum(sample*sample,-1)-.5*d*jnp.log(2*jnp.pi)
        ratio=logq-logprior
        return jnp.array([nll,math.log(9)-8/9+nll*d-prior_CE,ratio.max(),
            jnp.mean((ratio>jnp.log(9.)).astype(jnp.float32)),jnp.mean(jnp.sum(sample*sample,-1))])
    key=jax.random.PRNGKey(34211)
    for count in range(0,100001,1000):
        if count:actor,key=fit1000(actor,key)
        assert int(actor.opt_state[0].count)==count
        if count%5000==0:
            assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor))
            values=[]
            for start in range(train,train+4096,256):
                values.append(np.asarray(diagnose(actor.params,obs[start:start+256],targets[start:start+256],jax.random.PRNGKey(412+start))))
            v=np.stack(values);row=dict(actual_Adam=count,heldout_NLL_per_dim=float(v[:,0].mean()),
                conditional_KL_proxy_assuming_continuous_scores=float(v[:,1].mean()),
                sampled_max_log_density_ratio=float(v[:,2].max()),sampled_fraction_density_ratio_above9=float(v[:,3].mean()),
                sampled_mean_latent_norm_squared=float(v[:,4].mean()),elapsed_seconds=time.time()-result['time'])
            assert all(math.isfinite(v)for v in row.values())
            result['snapshots'].append(row);write(a.output/'status.json',result);print(json.dumps(row),flush=True)
    params=dict(router=flax.serialization.to_state_dict(jax.device_get(actor.params)),decoder=old['actor_params']['decoder'])
    metadata=dict(schema='private_affine_coupling_latent_actor_v1',architecture='conditional_affine_coupling_latent_then_decoder',
        observation_dim=int(obs.shape[-1]),action_dim=int(d),noise_dim=int(d),input='single_Gaussian',
        router_hidden_dims=list(hidden),decoder_hidden_dims=list(decoder_hidden),coupling_layers=4,log_scale_bound=1.,
        MLP_blocks=5,dense_layers=17,latent_draws=1,realized_latents=1,generated_candidate_actions=0,
        realized_actions=1,realized_action_samples=1,actor_forward_passes=1,flow_integration_steps=0,
        requires_critic=False,requires_BC_flow=False,prior_density_ratio_bound_guaranteed=False)
    artifact=a.output/'affine_latent_actor_100K_private_v1.pkl'
    with artifact.open('xb')as f:pickle.dump(dict(schema=metadata['schema'],metadata=metadata,actor_params=params,
        actual_router_Adam=100000,decoder_Adam_performed=0,decoder_parent_Adam=100000,
        frozen_decoder_parent_sha256=sha(parent['private_artifact']),source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    opt=a.output/'affine_router_optimizer_100K_private_v1.pkl'
    with opt.open('xb')as f:pickle.dump(flax.serialization.to_state_dict(jax.device_get(actor)),f,pickle.HIGHEST_PROTOCOL)
    policy=affine.make_policy(metadata);fixtures=obs[train:train+16];fk=jax.random.PRNGKey(411)
    epsilon=jax.random.normal(jax.random.split(fk)[0],(16,d));z=definition.apply({'params':actor.params},fixtures,epsilon)['latent']
    decoder=affine._Actor(decoder_hidden,d,False)
    manual=jnp.clip(decoder.apply({'params':params['decoder']},fixtures,z),-1,1)
    delta=float(jnp.max(jnp.abs(policy(params,fixtures,fk)-manual)));assert delta<1e-6
    result['composite_manual_max_abs_difference']=delta
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    conf=json.loads((checkpoint.parent/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
    template=legacy.create(conf['identity']['seed'],obs[:1],jnp.zeros((1,d)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==1000000 and int(teacher.base.v_updates)==0
    def exact_flow(_,x,key):
        ak,_=jax.random.split(key);epsilon=jax.random.normal(ak,(*x.shape[:-1],d))
        z=definition.apply({'params':actor.params},x,epsilon)['latent']
        return legacy.lean.flow(teacher.base.agent,x,z,10)
    policies=dict(composite=policy,learned_router_exact_Flow=jax.jit(exact_flow,backend='cpu'))
    for mode,fn in policies.items():
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,fn,100000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_affine_latent_router_CPU50_v1',mode=mode,
            reset_metadata=reset,actor_artifact_sha256=sha(artifact),no_optimizer_updates=True,
            counts_toward_native_goal=False,evaluation_data_used_for_learning=False,
            deployable_actor_only=mode=='composite',hybrid_uses_teacher=mode!='composite')
        ep=a.output/(mode+'_CPU50_v1.json');write(ep,evaluated)
        result['evaluations'][mode]=dict(success=evaluated['success_rate'],evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(a.output/'status.json',result);print(json.dumps(dict(mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(cache)==result['cache_sha256']and sha(checkpoint)==result['teacher_checkpoint_sha256']
    result.update(status='complete',completed=time.time(),actual_router_Adam=100000,private_artifact=str(artifact),
        private_artifact_sha256=sha(artifact),optimizer_artifact=str(opt),optimizer_sha256=sha(opt))
    write(a.output/'result.json',result);write(a.output/'status.json',result)


if __name__=='__main__':main()
