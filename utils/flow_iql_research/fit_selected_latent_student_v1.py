"""Frozen-teacher CPU feasibility of latent selection and one-step decoding.

Offline replay only. Two independent Adam optimizers; no critic/Flow update.
Full optimizer states are saved to permit actual update-count verification.
Native joint learners and their goal selection remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

# Legacy environment shim must precede optax import.
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import latent_selected_actor_v1 as latent
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
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    proposal=json.loads(a.proposal.read_text());case=proposal['cases'][a.case]
    for path,digest in proposal['source_sha256'].items():assert sha(path)==digest,path
    parent_path=Path(case['parent_result']);assert sha(parent_path)==case['parent_result_sha256']
    parent=json.loads(parent_path.read_text())
    assert parent['status']=='complete'and parent['training_states']==524288
    broad_path=Path(parent['parent_result']);assert sha(broad_path)==parent['parent_result_sha256']
    broad=json.loads(broad_path.read_text())
    cache=Path(case['observation_cache']);assert sha(cache)==parent['labeled_cache_sha256']
    with np.load(cache)as f:observations=f['observations'].copy();train=int(f['train_size'])
    assert train==524288 and len(observations)==train+4096
    held_hashes={hashlib.sha256(x.tobytes()).digest()for x in observations[train:]}
    assert not any(hashlib.sha256(x.tobytes()).digest()in held_hashes for x in observations[:train])
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==broad['teacher_checkpoint_sha256']
    conf=json.loads((checkpoint.parent/'config.json').read_text())
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    dim=int(json.loads((checkpoint.parent/'dataset.json').read_text())['action_dim'])
    template=flat.create(conf['identity']['seed'],jnp.asarray(observations[:1]),
        jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==int(teacher.actor_updates)==1000000
    assert int(teacher.base.v_updates)==0
    agent=teacher.base.agent
    a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='labeling',case=a.case,environment=broad['environment'],
        proposal=str(a.proposal),proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        teacher_checkpoint=str(checkpoint),teacher_checkpoint_sha256=sha(checkpoint),
        observation_cache_sha256=sha(cache),training_states=train,heldout_states=4096,
        byte_exact_train_heldout_overlap=False,trajectory_disjoint_split_claim=False,
        label_seed=49511,router_init=34210,decoder_init=34220,batch_seed=34211,
        batch_size=256,Adam_per_network=100000,learning_rate=.0003,
        hidden_dims=[256]*4,components=9,latent_scale_floor=.1,
        router_loss='mean(-ordinaryGaussianMixture_logp(selectedZ))/D',
        decoder_loss='mean MSE(rawdecoder(s,Z_uniform_candidate),clippedFlow10(s,Z))',
        teacher_optimizer_updates=0,evaluation_data_used_for_learning=False,
        stochastic_best9_teacher_CPU50=parent['stochastic_best9_teacher_CPU50'],
        scope='Private fixed1Mteacher CPU feasibility only; not joint1M/native nine-task/fresh/speed evidence.',
        snapshots=[],evaluations={})
    write(a.output/'status.json',result)

    @jax.jit
    def label(obs,key):
        z=jax.random.normal(key,(9,len(obs),dim))
        ob=jnp.broadcast_to(obs,(9,*obs.shape))
        actions=flat.lean.flow(agent,ob,z,10)
        q=agent.network.select('critic')(ob,actions=actions).min(0)
        return jnp.moveaxis(z,0,1),jnp.moveaxis(actions,0,1),jnp.moveaxis(q,0,1)

    zs=[];actions=[];qs=[]
    for chunk,offset in enumerate(range(0,len(observations),256)):
        z,b,q=label(jnp.asarray(observations[offset:offset+256]),
            jax.random.fold_in(jax.random.PRNGKey(49511),chunk))
        zs.append(np.asarray(z));actions.append(np.asarray(b));qs.append(np.asarray(q))
        if chunk%128==0:print(json.dumps(dict(stage='labels',states=offset+len(z))),flush=True)
    zs=np.concatenate(zs);actions=np.concatenate(actions);qs=np.concatenate(qs)
    selected=qs.argmax(-1);selected_z=zs[np.arange(len(zs)),selected]
    assert zs.shape==actions.shape==(train+4096,9,dim)
    assert all(np.isfinite(x).all()for x in (zs,actions,qs))
    labels=a.output/'selected_latent_replay_cache_v1.npz'
    np.savez_compressed(labels,observations=observations,noises=zs,actions=actions,
        scores=qs,best_index=selected,selected_latents=selected_z,train_size=train)
    result.update(status='fitting',label_cache_sha256=sha(labels))
    write(a.output/'status.json',result)
    obs=jnp.asarray(observations);z=jnp.asarray(zs);targets=jnp.asarray(actions)
    winner_z=jnp.asarray(selected_z)
    router_def=latent.MixtureActor((256,)*4,dim,9,.1)
    decoder_def=latent._Actor((256,)*4,dim,False)
    rparams=router_def.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,dim+1)),return_parameters=True)['params']
    dparams=decoder_def.init(jax.random.PRNGKey(34220),obs[:1],jnp.zeros((1,dim)))['params']
    router=TrainState.create(router_def,rparams,optax.adam(.0003))
    decoder=TrainState.create(decoder_def,dparams,optax.adam(.0003))

    def router_parameters(params,x):
        return router_def.apply({'params':params},x,jnp.zeros((*x.shape[:-1],dim+1)),return_parameters=True)

    @jax.jit
    def fit1000(router,decoder,key):
        def update(carry,_):
            r,d,k=carry;k,ik,zk=jax.random.split(k,3)
            idx=jax.random.randint(ik,(256,),0,train)
            zi=jax.random.randint(zk,(256,),0,9)
            def rloss(params):
                v=-latent.latent_log_probability(router_parameters(params,obs[idx]),winner_z[idx]).mean()/dim
                return v,dict(loss=v)
            def dloss(params):
                raw=decoder_def.apply({'params':params},obs[idx],z[idx,zi])
                v=jnp.mean((raw-targets[idx,zi])**2)
                return v,dict(loss=v)
            r,_=r.apply_loss_fn(rloss);d,_=d.apply_loss_fn(dloss)
            return (r,d,k),None
        return jax.lax.scan(update,(router,decoder,key),None,length=1000)[0]

    @jax.jit
    def diagnostics(r,d,x,zz,aa,selected_zz):
        pars=router_parameters(r,x)
        logp=latent.latent_log_probability(pars,selected_zz)
        raw=decoder_def.apply({'params':d},jnp.broadcast_to(x[:,None],(len(x),9,x.shape[-1])),zz)
        selected_raw=decoder_def.apply({'params':d},x,selected_zz)
        # Reconstruct selected targets by matching the exact saved winner latent.
        index=jnp.argmin(jnp.sum((zz-selected_zz[:,None])**2,axis=-1),axis=-1)
        selected_aa=aa[jnp.arange(len(aa)),index]
        prior=-.5*jnp.sum(selected_zz**2,axis=-1)-.5*dim*jnp.log(2*jnp.pi)
        return jnp.array([-logp.mean()/dim,-prior.mean()/dim,
            jnp.mean((jnp.clip(raw,-1,1)-aa)**2),jnp.mean((jnp.clip(selected_raw,-1,1)-selected_aa)**2)])

    key=jax.random.PRNGKey(34211)
    for count in range(0,100001,1000):
        if count:router,decoder,key=fit1000(router,decoder,key)
        assert int(router.opt_state[0].count)==int(decoder.opt_state[0].count)==count
        if count%5000==0:
            for state in (router,decoder):
                assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
            rows=[]
            for off in range(train,train+4096,256):
                rows.append(np.asarray(diagnostics(router.params,decoder.params,
                    obs[off:off+256],z[off:off+256],targets[off:off+256],winner_z[off:off+256])))
            nll,prior,all_mse,selected_mse=np.stack(rows).mean(0).tolist()
            snapshot=dict(actual_router_Adam=count,actual_decoder_Adam=count,
                heldout_NLL_per_dim=nll,prior_NLL_per_dim=prior,
                heldout_uniform_decoder_MSE=all_mse,heldout_selected_decoder_MSE=selected_mse,
                elapsed_seconds=time.time()-result['time'])
            result['snapshots'].append(snapshot);write(a.output/'status.json',result)
            print(json.dumps(snapshot),flush=True)
    params=dict(router=flax.serialization.to_state_dict(jax.device_get(router.params)),
                decoder=flax.serialization.to_state_dict(jax.device_get(decoder.params)))
    metadata=dict(schema='private_selected_latent_actor_v1',observation_dim=obs.shape[-1],
        action_dim=dim,hidden_dims=[256]*4,components=9,standard_deviation_floor=.1,
        MLP_blocks=2,dense_layers=10,actor_forward_passes=1,
        generated_candidate_actions=0,realized_latents=1,realized_actions=1,
        flow_integration_steps=0,requires_critic=False,requires_BC_flow=False)
    artifact=a.output/'selected_latent_actor_100K_private_v1.pkl'
    optimizer_artifact=a.output/'selected_latent_fit_state_100K_private_v1.pkl'
    with optimizer_artifact.open('xb')as f:
        pickle.dump(dict(router_state=flax.serialization.to_state_dict(jax.device_get(router)),
            decoder_state=flax.serialization.to_state_dict(jax.device_get(decoder)),
            router_Adam_updates=100000,decoder_Adam_updates=100000,
            source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    with artifact.open('xb')as f:
        pickle.dump(dict(schema=metadata['schema'],metadata=metadata,actor_params=params,
            router_Adam_updates=100000,decoder_Adam_updates=100000,
            teacher_checkpoint_sha256=sha(checkpoint),source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    composite=latent.make_policy(metadata)
    router_params=router.params;decoder_params=decoder.params

    def router_exactflow(_,x,key):
        action_key,_=jax.random.split(key)
        zz=latent.sample_latent(router_parameters(router_params,x),
            jax.random.normal(action_key,(*x.shape[:-1],dim+1)))
        return flat.lean.flow(agent,x,zz,10)

    def winner_fastdecoder(_,x,key):
        ob,zz=flat.lean.bank(agent,x,key,9)
        aa=flat.lean.flow(agent,ob,zz,10)
        q=agent.network.select('critic')(ob,actions=aa).min(0)
        best=flat.gather(zz,q.argmax(0))
        return jnp.clip(decoder_def.apply({'params':decoder_params},x,best),-1,1)

    def prior_fastdecoder(_,x,key):
        action_key,_=jax.random.split(key)
        zz=jax.random.normal(action_key,(*x.shape[:-1],dim))
        return jnp.clip(decoder_def.apply({'params':decoder_params},x,zz),-1,1)

    policies=dict(composite=composite,
        learned_router_exact_Flow=jax.jit(router_exactflow,backend='cpu'),
        exact_winner_fast_decoder=jax.jit(winner_fastdecoder,backend='cpu'),
        prior_fast_decoder=jax.jit(prior_fastdecoder,backend='cpu'))
    fixtures=obs[train:train+16];fixture_key=jax.random.PRNGKey(411)
    fixture_noise=jax.random.normal(jax.random.split(fixture_key)[0],(16,dim+1))
    manual_z=latent.sample_latent(router_parameters(router_params,fixtures),fixture_noise)
    manual_action=jnp.clip(decoder_def.apply({'params':decoder_params},fixtures,manual_z),-1,1)
    fixture_difference=float(np.max(np.abs(np.asarray(composite(params,fixtures,fixture_key))-np.asarray(manual_action))))
    assert fixture_difference<=1e-6
    result['composite_vs_manual_fixture_max_abs_difference']=fixture_difference
    for mode,policy in policies.items():
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(result['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,100000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_selected_latent_CPU50_v1',
            mode=mode,no_optimizer_updates=True,evaluation_data_used_for_learning=False,
            counts_toward_native_goal=False,actor_artifact_sha256=sha(artifact),
            reset_metadata=reset,deployable_actor_only=mode in ('composite','prior_fast_decoder'),
            hybrid_uses_teacher=mode in ('learned_router_exact_Flow','exact_winner_fast_decoder'))
        ep=a.output/(mode+'_CPU50_v1.json');write(ep,evaluated)
        result['evaluations'][mode]=dict(success=evaluated['success_rate'],
            evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(a.output/'status.json',result)
        print(json.dumps(dict(mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(checkpoint)==result['teacher_checkpoint_sha256']
    assert sha(cache)==result['observation_cache_sha256']
    result.update(status='complete',completed=time.time(),private_artifact=str(artifact),
        private_artifact_sha256=sha(artifact),metadata=metadata,
        optimizer_state_artifact=str(optimizer_artifact),optimizer_state_sha256=sha(optimizer_artifact),
        parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)))
    write(a.output/'result.json',result);write(a.output/'status.json',result)


if __name__=='__main__':main()
