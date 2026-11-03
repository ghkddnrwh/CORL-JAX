"""Read-only actor-loss gradients from immutable joint-training checkpoints."""
from utils.flow_iql_research.onestep_implementation_registry_v2 import implementation
import argparse
import json
import math
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def norm(tree):return math.sqrt(sum(float(np.square(np.asarray(v,np.float64)).sum())for v in jax.tree_util.tree_leaves(tree)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    conf=json.loads((args.run/'config.json').read_text());identity=conf['identity']
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.run/f'checkpoint_{args.step}.pkl';digest=sha(checkpoint)
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    assert saved['step']==args.step and saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
    selected_core=implementation(identity['recipe']).core
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    env,data,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    indices=np.random.RandomState(21931).randint(len(data['actions']),size=256)
    batch={k:jnp.asarray(data[k][indices],jnp.float32)for k in ('observations','actions','next_observations','rewards','masks')}
    batch['actions']=jnp.clip(batch['actions'],-1+1e-5,1-1e-5)
    template=selected_core.create(identity['seed'],batch['observations'][:1],batch['actions'][:1],cfg,identity['recipe'])
    state=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),jax.devices('cpu')[0])
    assert int(state.base.q_updates)==int(state.actor_updates)==args.step and int(state.base.v_updates)==0
    spec=state.base.agent.config['onestep_specification'];_,rng=jax.random.split(state.base.agent.rng)
    teacher_batch=getattr(selected_core,'teacher_batch',core.teacher_batch)
    actor_loss=getattr(selected_core,'actor_loss',core.actor_loss)
    observations,noises,target,heads,actions,index=teacher_batch(state,batch,rng,spec)
    def info(params):return actor_loss(state,observations,noises,target,heads,spec,params)[1]
    params=state.actor.params
    mse_grad=jax.jit(jax.grad(lambda p:info(p)['distill_MSE']))(params)
    q_grad=jax.jit(jax.grad(lambda p:info(p)['Q_loss']))(params)
    total_grad=jax.jit(jax.grad(lambda p:actor_loss(state,observations,noises,target,heads,spec,p)[0]))(params)
    values=jax.device_get(jax.jit(info)(params))
    mse_norm,q_norm,total_norm=norm(mse_grad),norm(q_grad),norm(total_grad)
    dot=sum(float((np.asarray(a,np.float64)*np.asarray(b,np.float64)).sum())for a,b in zip(jax.tree_util.tree_leaves(mse_grad),jax.tree_util.tree_leaves(q_grad)))
    reconstructed=jax.tree_util.tree_map(lambda a,b:spec['effective_alpha']*a+b,mse_grad,q_grad)
    residual=jax.tree_util.tree_map(lambda a,b:a-b,total_grad,reconstructed)
    metadata=selected_core.actor_metadata(cfg,batch['observations'].shape[-1],batch['actions'].shape[-1])
    small=batch['observations'][:16];ob,eps=core.lean.bank(state.base.agent,jnp.broadcast_to(small,(8,*small.shape)),jax.random.PRNGKey(21932),9)
    teacher_actions=core.lean.flow(state.base.agent,ob,eps,10)
    teacher_q=state.base.agent.network.select('critic')(ob,actions=teacher_actions).min(0)
    selected=core.gather(teacher_actions,teacher_q.argmax(0))
    actor_actions=jnp.clip(state.actor(jnp.broadcast_to(small,(8,*small.shape)),core.packed_input(eps,spec)),-1,1)
    teacher_var=float(jnp.var(selected,axis=0).mean());actor_var=float(jnp.var(actor_actions,axis=0).mean())
    tie={}
    if actions.shape[0]>1:
        q=heads.mean(0)if spec.get('teacher_q_aggregation')=='mean'else heads.min(0);order=jnp.argsort(q,axis=0);top=order[-1];second=order[-2]
        gaps=core.gather(q[...,None],top)[...,0]-core.gather(q[...,None],second)[...,0]
        distance=jnp.mean((core.gather(actions,top)-core.gather(actions,second))**2,axis=-1)
        relative=gaps/jnp.maximum(q.std(0),1.)
        tie=dict(teacher_top2_Q_gap_mean=float(gaps.mean()),teacher_top2_Q_gap_median=float(jnp.median(gaps)),
            teacher_top2_action_MSE_mean=float(distance.mean()),near_tie_fraction=float(jnp.mean((relative<.05).astype(jnp.float32))))
    result=dict(time=time.time(),run=args.run.name,environment=identity['environment'],training_seed=identity['seed'],recipe=identity['recipe'],step=args.step,
        checkpoint_sha256=digest,config_sha256=sha(args.run/'config.json'),source_sha256=sha(__file__),core_source_sha256=sha(selected_core.__file__),shared_loss_core_source_sha256=sha(core.__file__),
        actual_Q_updates=int(state.base.q_updates),actual_actor_updates=int(state.actor_updates),optimizer_updates_performed=0,
        fixed_dataset_index_seed=21931,fixed_batch_size=256,device='cpu',alpha=spec['effective_alpha'],loss_info={k:float(v)for k,v in values.items()},
        MSE_gradient_L2=mse_norm,weighted_MSE_gradient_L2=spec['effective_alpha']*mse_norm,Q_gradient_L2=q_norm,total_gradient_L2=total_norm,
        Q_to_weighted_MSE_gradient_ratio=q_norm/max(spec['effective_alpha']*mse_norm,1e-20),
        MSE_Q_gradient_cosine=dot/max(mse_norm*q_norm,1e-20),gradient_additivity_residual_L2=norm(residual),
        actor_same_state_variance=actor_var,teacher_best9_same_state_variance=teacher_var,actor_to_teacher_diversity_ratio=actor_var/max(teacher_var,1e-20),
        teacher_ties=tie,training_label_Q_aggregation=spec.get('teacher_q_aggregation','min'),diversity_teacher_Q_aggregation='min',scope='Read-only fixed-batch gradient/teacher-mode diagnostics, not policy evaluation and not an OOD or true-return error oracle.')
    assert all(math.isfinite(float(v))for v in (mse_norm,q_norm,total_norm,teacher_var,actor_var))
    assert sha(checkpoint)==digest
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps({k:result[k]for k in ('run','step','Q_to_weighted_MSE_gradient_ratio','MSE_Q_gradient_cosine','actor_to_teacher_diversity_ratio')}),flush=True)


if __name__=='__main__':main()
