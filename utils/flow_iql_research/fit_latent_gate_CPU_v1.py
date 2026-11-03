"""Freeze the successful decoder; distill best9 indices from stochastic banks.

Shared logits see (state,latent), unlike historical state-only fixed32 routing.
Only the gate is fitted. Deployment never decodes nine actions or queries Q.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import latent_gate_actor_v1 as gate
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
    parentpath=Path(case['parent_result']);assert sha(parentpath)==case['parent_result_sha256']
    parent=json.loads(parentpath.read_text());assert parent['status']=='complete'
    assert sha(parent['private_artifact'])==parent['private_artifact_sha256']
    with Path(parent['private_artifact']).open('rb')as f:old=pickle.load(f)
    cache=Path(case['cache']);assert sha(cache)==parent['label_cache_sha256']
    with np.load(cache)as f:
        obs=jnp.asarray(f['observations'].copy());z=jnp.asarray(f['noises'].copy())
        scores=jnp.asarray(f['scores'].copy());labels=jnp.asarray(f['best_index'].copy());train=int(f['train_size'])
    assert train==524288 and len(obs)==train+4096 and z.shape[1]==9
    np.testing.assert_array_equal(labels,scores.argmax(-1))
    dimension=z.shape[-1];hidden=tuple(parent['metadata']['hidden_dims'])
    definition=gate.SharedLatentGate(hidden)
    initial=definition.init(jax.random.PRNGKey(34210),obs[:1],z[:1])['params']
    actor=TrainState.create(definition,initial,optax.adam(.0003))
    a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='fitting',environment=parent['environment'],
        proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        parent_result_sha256=sha(parentpath),cache_sha256=sha(cache),
        teacher_checkpoint_sha256=parent['teacher_checkpoint_sha256'],
        frozen_decoder_parent_sha256=sha(parent['private_artifact']),
        training_states=train,heldout_states=4096,learning_rate=.0003,batch_size=256,
        actual_router_target_Adam=100000,decoder_Adam_performed=0,teacher_Adam_performed=0,
        initialization_seed=34210,minibatch_seed=34211,objective='CE(best9index,shared_latent_logits)',
        latent_proposals=9,action_proposals_generated_at_deployment=0,
        evaluation_data_used_for_learning=False,snapshots=[],evaluations={},
        reused_controls={k:parent['evaluations'][k]for k in ('exact_winner_fast_decoder','prior_fast_decoder')},
        scope='Fixed-teacher CPU100K classification-only study with exact same frozen successful decoder. No native1M/9task/fresh/speed claim. Nine latent proposals remain internal to actor; only one decoded action.')
    write(a.output/'status.json',result)
    @jax.jit
    def fit1000(state,key):
        def update(carry,_):
            s,k=carry;k,ik,_=jax.random.split(k,3);idx=jax.random.randint(ik,(256,),0,train)
            def objective(params):
                value=optax.softmax_cross_entropy_with_integer_labels(definition.apply({'params':params},obs[idx],z[idx]),labels[idx]).mean()
                return value,dict(loss=value)
            s,_=s.apply_loss_fn(objective);return(s,k),None
        return jax.lax.scan(update,(state,key),None,length=1000)[0]
    @jax.jit
    def diagnose(params,x,latents,values,best):
        logits=definition.apply({'params':params},x,latents);index=logits.argmax(-1)
        selected=jnp.take_along_axis(values,index[:,None],axis=-1)[:,0]
        return jnp.array([optax.softmax_cross_entropy_with_integer_labels(logits,best).mean(),
            jnp.mean((index==best).astype(jnp.float32)),jnp.mean(values.max(-1)-selected)])
    key=jax.random.PRNGKey(34211)
    for count in range(0,100001,1000):
        if count:actor,key=fit1000(actor,key)
        assert int(actor.opt_state[0].count)==count
        if count%5000==0:
            assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor))
            diagnostics=[]
            for start in range(train,train+4096,256):
                diagnostics.append(np.asarray(diagnose(actor.params,obs[start:start+256],z[start:start+256],scores[start:start+256],labels[start:start+256])))
            values=np.stack(diagnostics).mean(0)
            row=dict(actual_Adam=count,heldout_CE=float(values[0]),heldout_index_accuracy=float(values[1]),
                within_teacher_bank_predicted_Q_regret=float(values[2]),elapsed_seconds=time.time()-result['time'])
            result['snapshots'].append(row);write(a.output/'status.json',result);print(json.dumps(row),flush=True)
    params=dict(router=flax.serialization.to_state_dict(jax.device_get(actor.params)),decoder=old['actor_params']['decoder'])
    metadata=dict(schema='private_latent_gate_actor_v1',architecture='shared_latent_classification_then_single_decoder',
        observation_dim=int(obs.shape[-1]),action_dim=int(dimension),hidden_dims=list(hidden),
        input='state_and_nine_Gaussian_latents',noise_dim=9*dimension,latent_draws=9,gating_logits=9,
        MLP_blocks=2,dense_layers=10,generated_candidate_actions=0,realized_action_samples=1,
        actor_forward_passes=1,realized_latents=9,selected_latents_decoded=1,
        realized_actions=1,flow_integration_steps=0,requires_critic=False,requires_BC_flow=False)
    artifact=a.output/'latent_gate_actor_100K_private_v1.pkl'
    with artifact.open('xb')as f:pickle.dump(dict(schema=metadata['schema'],metadata=metadata,actor_params=params,
        actual_router_Adam=100000,decoder_Adam_performed=0,decoder_parent_Adam=100000,
        frozen_decoder_parent_sha256=sha(parent['private_artifact']),source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
    opt=a.output/'latent_gate_router_optimizer_100K_private_v1.pkl'
    with opt.open('xb')as f:pickle.dump(flax.serialization.to_state_dict(jax.device_get(actor)),f,pickle.HIGHEST_PROTOCOL)
    checkpoint=Path(parent['teacher_checkpoint']);assert sha(checkpoint)==parent['teacher_checkpoint_sha256']
    conf=json.loads((checkpoint.parent/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
    template=legacy.create(conf['identity']['seed'],obs[:1],jnp.zeros((1,dimension)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==1000000 and int(teacher.base.v_updates)==0
    def exact_flow(_,x,key):
        ak,_=jax.random.split(key)
        latents=jax.random.normal(ak,(*x.shape[:-1],9*dimension)).reshape((*x.shape[:-1],9,dimension))
        logits=definition.apply({'params':actor.params},x,latents)
        return legacy.lean.flow(teacher.base.agent,x,gate.select_latent(latents,logits),10)
    policies=dict(composite=gate.make_policy(metadata),learned_router_exact_Flow=jax.jit(exact_flow,backend='cpu'))
    for mode,policy in policies.items():
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42);evaluated=evaluate(namespace,env,policy,100000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_shared_latent_gate_CPU50_v1',mode=mode,
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
