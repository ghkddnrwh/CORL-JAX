"""Prototype: distill mean-Q best9 actions; keep lean bootstrap unchanged.

The paired reference policy remains the original min-Q best9 policy. The new
flag changes training labels only. A separate existing __meanQ flag controls
the optional actor Q gradient. Exported inference is the same single MLP.
"""
from pathlib import Path
from utils.flow_iql_research import onestep_deploy_core_v1 as base
import jax
import jax.numpy as jnp
import numpy as np


ENVIRONMENTS=base.ENVIRONMENTS
RECIPES=('bank_bc__meanTeacher','bank_q_a03__meanQ__meanTeacher',
         'bank_q_a1__meanQ__meanTeacher','bank_q_a3__meanQ__meanTeacher')
State=base.State
lean=base.lean


def specification(recipe):
    assert recipe.count('__meanTeacher')==1
    underlying=recipe.replace('__meanTeacher','')
    spec=base.specification(underlying)
    assert spec['target']=='best9'and spec['input']=='bank'
    spec.update(name=recipe,teacher_q_aggregation='mean')
    return spec


def config_for(environment,recipe,width=None):
    spec=specification(recipe)
    cfg=base.config_for(environment,recipe.replace('__meanTeacher',''),width)
    cfg['onestep_specification'].update(name=recipe,teacher_q_aggregation=spec['teacher_q_aggregation'])
    return cfg


def choose_teacher(actions,heads):
    index=jnp.argmax(heads.mean(axis=0),axis=0)
    return jax.lax.stop_gradient(base.gather(actions,index)),index


def teacher_batch(state,batch,rng,spec):
    assert spec['teacher_q_aggregation']=='mean'
    observations,noises,_,heads,actions,_=base.teacher_batch(state,batch,rng,spec)
    target,index=choose_teacher(actions,heads)
    return observations,noises,target,heads,actions,index


actor_loss=base.actor_loss
create=base.create
actor_metadata=base.actor_metadata
actor_definition=base.actor_definition
make_actor_policy=base.make_actor_policy
make_teacher_policy=base.make_teacher_policy
export_actor=base.export_actor
make_scan=base.make_scan


def build_update(recipe,bounds):
    lean_update=lean.build_update('lean',bounds)
    def update(state,batch):
        next_base,logs=lean_update(state.base,batch)
        _,rng=jax.random.split(state.base.agent.rng)
        spec=state.base.agent.config['onestep_specification']
        observations,noises,target,heads,_,_=teacher_batch(state,batch,rng,spec)
        actor,info=state.actor.apply_loss_fn(lambda params:actor_loss(state,observations,noises,target,heads,spec,params))
        logs.update({'deployment/'+key:value for key,value in info.items()})
        return state.replace(base=next_base,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


def make_probe(recipe,bounds):
    old_probe=base.make_probe(recipe,bounds)
    @jax.jit
    def probe(state,batch):
        values=old_probe(state,batch);agent=state.base.agent;obs=batch['observations']
        ob,noises=lean.bank(agent,obs,jax.random.PRNGKey(74191),9)
        actions=lean.flow(agent,ob,noises,10)
        heads=agent.network.select('critic')(ob,actions=actions)
        chosen,index=choose_teacher(actions,heads)
        values.update(mean_teacher_action=chosen,mean_teacher_Q=heads.mean(0).max(0),
            actor_Q_mean=agent.network.select('critic')(obs,actions=values['actor']).mean(0),
            mean_min_choice_disagreement=(index!=heads.min(0).argmax(0)).astype(jnp.float32))
        return values
    return probe


def probe_summary(values,bounds):
    result=base.probe_summary(values,bounds)
    target=np.asarray(values['mean_teacher_action'])
    result.update(mean_teacher_raw_actor_MSE=float(np.mean((np.asarray(values['raw'])-target)**2)),
        mean_teacher_clipped_actor_MSE=float(np.mean((np.asarray(values['actor'])-target)**2)),
        mean_teacher_predicted_Q_mean_regret=float(np.mean(np.asarray(values['mean_teacher_Q'])-np.asarray(values['actor_Q_mean']))),
        mean_min_teacher_selection_disagreement=float(np.asarray(values['mean_min_choice_disagreement']).mean()))
    return result


def source_paths():return [Path(__file__).resolve()]+base.source_paths()
