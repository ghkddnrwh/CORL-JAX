"""Fixed-teacher CPU screen: same bank actor, critic-gradient imitation metric.

Three prespecified global coefficients share initialization, data and batch
keys. The frozen critic is queried during label construction and diagnostics,
never differentiated at a student action in the training objective.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import teacher_gradient_metric_v1 as metric
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
    cache=Path(case['selected_latent_cache']);assert sha(cache)==case['cache_sha256']
    with np.load(cache)as f:
        obs=f['observations'].copy();noises=f['noises'].copy();actions=f['actions'].copy()
        scores=f['scores'].copy();index=f['best_index'].copy();train=int(f['train_size'])
    assert train==524288 and len(obs)==train+4096
    assert np.array_equal(index,scores.argmax(-1))
    dim=actions.shape[-1];target=actions[np.arange(len(obs)),index]
    selected_score=scores[np.arange(len(obs)),index]
    del actions,scores
    conf=json.loads((Path(case['teacher_run'])/'config.json').read_text())
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=Path(case['teacher_checkpoint']);assert sha(checkpoint)==case['teacher_checkpoint_sha256']
    cfg=dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
    template=flat.create(0,jnp.asarray(obs[:1]),jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==1000000 and int(teacher.base.v_updates)==0
    agent=teacher.base.agent
    a.output.mkdir(parents=True)
    result=dict(time=time.time(),status='teacher_gradients',environment=conf['identity']['environment'],
        proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        teacher_checkpoint_sha256=sha(checkpoint),cache_sha256=sha(cache),
        training_states=train,heldout_states=4096,coefficients=[0.,1.,10.],
        initialization_seed=34210,minibatch_seed=34211,Adam_per_method=100000,
        learning_rate=.0003,batch_size=256,teacher_optimizer_updates=0,
        evaluation_states_used_for_learning=False,actors={},
        scope='Private frozen1Mteacher CPU diagnosis; not joint/native last3/9task/fresh or final speed evidence.')
    write(a.output/'status.json',result)
    def q_single(x,act):return agent.network.select('critic')(x,actions=act)
    gradient=jax.jit(jax.vmap(jax.jacrev(q_single,argnums=1)))
    q_batch=jax.jit(lambda x,act:agent.network.select('critic')(x,actions=act))
    gradients=[];max_difference=0.
    for chunk,offset in enumerate(range(0,len(obs),256)):
        x=jnp.asarray(obs[offset:offset+256]);act=jnp.asarray(target[offset:offset+256])
        g=np.asarray(gradient(x,act));gradients.append(g)
        if chunk<2:
            actual=np.asarray(q_batch(x,act)).min(0)
            max_difference=max(max_difference,float(np.max(np.abs(actual-selected_score[offset:offset+256]))))
            assert max_difference<=.01
        if chunk%256==0:print(json.dumps(dict(stage='teacher_gradients',states=offset+len(g))),flush=True)
    g=np.concatenate(gradients);assert g.shape==(train+4096,2,dim)and np.isfinite(g).all()
    scale=float(np.mean(g[:train].astype(np.float64)**2));assert scale>0 and np.isfinite(scale)
    gradient_cache=a.output/'teacher_action_gradients_v1.npz'
    np.savez_compressed(gradient_cache,gradients=g,training_mean_square=scale,train_size=train)
    result.update(status='fitting',gradient_cache_sha256=sha(gradient_cache),
        training_gradient_mean_square=scale,cached_Q_check_max_abs_difference=max_difference,
        raw_training_gradient_norm_quantiles=np.quantile(np.linalg.norm(g[:train],axis=-1),[0,.5,.9,.99,1]).tolist(),
        metric_norm_cap_fraction=float(np.mean(np.sum(g[:train]**2,axis=-1)>10*dim*scale)))
    observations=jnp.asarray(obs);packed=jnp.asarray(noises.reshape((len(obs),9*dim)))
    targets=jnp.asarray(target);gg=jnp.asarray(g)
    features=metric.normalized_gradients(gg,jnp.asarray(scale,jnp.float32))
    metadata=flat.actor_metadata(flat.config_for(result['environment'],'bank_bc'),obs.shape[-1],dim)
    definition=flat.actor_definition(metadata)
    initial=definition.init(jax.random.PRNGKey(34210),observations[:1],packed[:1])['params']

    @jax.jit
    def diagnostics(params,x,z,t,geometry,raw_gradient):
        raw=definition.apply({'params':params},x,z);act=jnp.clip(raw,-1,1)
        _,info=metric.loss(raw,t,geometry,1.)
        qt=q_batch(x,t);qp=q_batch(x,act)
        linear=jnp.sum(raw_gradient*(act-t)[:,None,:],axis=-1).T
        return jnp.array([info['raw_MSE'],info['clipped_MSE'],info['teacher_linearized_penalty'],
            jnp.mean((qp-qt)**2),jnp.mean((qp-qt-linear)**2),jnp.mean(qt.min(0)-qp.min(0))])

    for coefficient in result['coefficients']:
        name='metric_lambda'+str(int(coefficient))
        actor=TrainState.create(definition,initial,optax.adam(.0003))
        def fit1000(actor,key):
            def update(carry,_):
                state,k=carry;k,ik,_=jax.random.split(k,3)
                idx=jax.random.randint(ik,(256,),0,train)
                def objective(params):
                    raw=definition.apply({'params':params},observations[idx],packed[idx])
                    return metric.loss(raw,targets[idx],features[idx],coefficient)
                state,_=state.apply_loss_fn(objective)
                return (state,k),None
            return jax.lax.scan(update,(actor,key),None,length=1000)[0]
        fit1000=jax.jit(fit1000);key=jax.random.PRNGKey(34211);snapshots=[]
        for count in range(0,100001,1000):
            if count:actor,key=fit1000(actor,key)
            assert int(actor.opt_state[0].count)==count
            if count%5000==0:
                assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(actor))
                diagnostic=[]
                for off in range(train,train+4096,256):
                    diagnostic.append(np.asarray(diagnostics(actor.params,observations[off:off+256],packed[off:off+256],
                        targets[off:off+256],features[off:off+256],gg[off:off+256])))
                values=np.stack(diagnostic).mean(0).tolist()
                row=dict(actual_Adam=count,raw_MSE=values[0],clipped_MSE=values[1],
                    linearized_metric_penalty=values[2],actual_critic_head_value_error_MSE=values[3],
                    critic_Taylor_remainder_MSE=values[4],predicted_Q_regret=values[5],
                    elapsed_seconds=time.time()-result['time'])
                snapshots.append(row);result['actors'][name]=dict(status='fitting',coefficient=coefficient,snapshots=snapshots)
                write(a.output/'status.json',result);print(json.dumps(dict(method=name,**row)),flush=True)
        params=flax.serialization.to_state_dict(jax.device_get(actor.params))
        optimizer_path=a.output/(name+'_optimizer_100K_private_v1.pkl')
        with optimizer_path.open('xb')as f:pickle.dump(flax.serialization.to_state_dict(jax.device_get(actor)),f,pickle.HIGHEST_PROTOCOL)
        artifact=a.output/(name+'_actor_100K_private_v1.pkl')
        with artifact.open('xb')as f:pickle.dump(dict(schema='private_teacher_gradient_metric_actor_v1',
            metadata=metadata,actor_params=params,actual_Adam=100000,coefficient=coefficient,
            source_sha256=result['source_sha256'],teacher_checkpoint_sha256=sha(checkpoint),
            jointly_trained=False),f,pickle.HIGHEST_PROTOCOL)
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(result['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,flat.make_actor_policy(metadata),100000,50);env.close()
        evaluated.update(time=time.time(),protocol='private_teacher_gradient_metric_CPU50_v1',
            reset_metadata=reset,actor_artifact_sha256=sha(artifact),no_optimizer_updates=True,
            coefficient=coefficient,evaluation_states_used_for_learning=False,counts_toward_native_goal=False,
            learned_model_trees_loaded=['actor_params'],requires_critic=False,requires_BC_flow=False,
            actor_forward_passes=1,generated_candidate_actions=0)
        ep=a.output/(name+'_CPU50_v1.json');write(ep,evaluated)
        result['actors'][name].update(status='complete',success=evaluated['success_rate'],
            actor_path=str(artifact),actor_sha256=sha(artifact),optimizer_path=str(optimizer_path),
            optimizer_sha256=sha(optimizer_path),evaluation_path=str(ep),evaluation_sha256=sha(ep))
        write(a.output/'status.json',result);print(json.dumps(dict(method=name,success=evaluated['success_rate'])),flush=True)
    assert sha(checkpoint)==case['teacher_checkpoint_sha256']and sha(cache)==case['cache_sha256']
    result.update(status='complete',completed=time.time(),total_actual_Adam=300000,total_evaluation_episodes=150)
    write(a.output/'result.json',result);write(a.output/'status.json',result)


if __name__=='__main__':main()
