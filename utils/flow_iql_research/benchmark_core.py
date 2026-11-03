"""Official FQL/ReBRAC controls under the shared scoring and replay protocol."""
from utils.flow_iql_research import core as reference
from utils.flow_iql_research.core import ENVIRONMENTS,State,make_scan
from utils.flow_iql_research.vendor import fql,rebrac
import jax
import jax.numpy as jnp
import numpy as np
import optax

PROFILES={'official':{}}


def config_for(environment,discount=None,expectile=None,width=None,alpha=None,kind='fql'):
    c=(fql if kind=='fql' else rebrac).get_config()
    w=width or 512
    g=.995 if any(s in environment for s in ('humanoidmaze','antsoccer')) else .99
    c.update(actor_hidden_dims=(w,)*4,value_hidden_dims=(w,)*4,
             discount=g if discount is None else discount,encoder=None)
    if kind=='fql':
        a=1000. if 'puzzle-' in environment else 300. if any(s in environment for s in ('cube-','scene-')) else 30. if 'humanoidmaze' in environment else 10.
        c.update(alpha=a if alpha is None else alpha,q_agg='min' if 'antmaze-' in environment else 'mean')
    else:
        a1,a2=(.003,.01) if 'antmaze-' in environment else (.01,.01) if any(s in environment for s in ('humanoidmaze','antsoccer')) else (1.,0.) if 'cube-single' in environment else (.1,0.) if 'cube-double' in environment else (.1,.01) if 'scene-' in environment else (.3,.01)
        c.update(alpha_actor=a1,alpha_critic=a2)
    result=c.to_dict()
    # Generic runner metadata slots, unused by either official algorithm.
    result['expectile']=None
    result.setdefault('alpha',None)
    return result


def create(seed,observations,actions,config,kind='fql'):
    cls=fql.FQLAgent if kind=='fql' else rebrac.ReBRACAgent
    agent=cls.create(seed,observations,actions,dict(config))
    return State(agent,optax.adam(config['lr']).init({}),jnp.asarray(0),jnp.asarray(0))


def build_update(profile_name,bounds,kind='fql'):
    assert profile_name=='official'
    if kind=='fql':return reference.build_update('official',bounds,'fql')
    def update(state,batch):
        def branch(full):
            def run(agent):
                new,info=agent.update(batch,full_update=full)
                # Official full/critic-only updates return different info trees.
                logs={k:info[k] for k in ('critic/critic_loss','critic/q_mean','grad/norm')}
                logs['actor/actor_loss']=info.get('actor/actor_loss',jnp.asarray(0.))
                logs['actor/gradient_evaluation']=jnp.asarray(float(full))
                return new,logs
            return run
        full=(state.q_updates+1)%state.agent.config['actor_freq']==0
        agent,logs=jax.lax.cond(full,branch(True),branch(False),state.agent)
        return state.replace(agent=agent,q_updates=state.q_updates+1),logs
    return update


@jax.jit
def probe(state,batch):
    agent=state.agent;net=agent.network;cfg=agent.config
    q=net.select('critic')(batch['observations'],actions=batch['actions'])
    tq=net.select('target_critic')(batch['observations'],actions=batch['actions'])
    if cfg['agent_name']=='fql':
        actions=agent.sample_actions(batch['next_observations'],seed=jax.random.PRNGKey(178))
        nextq=net.select('target_critic')(batch['next_observations'],actions=actions)
        nextq=nextq.min(0) if cfg['q_agg']=='min' else nextq.mean(0)
    else:
        dist=net.select('target_actor')(batch['next_observations'])
        noise=jnp.clip(jax.random.normal(jax.random.PRNGKey(178),dist.mode().shape)*cfg['actor_noise'],
                       -cfg['actor_noise_clip'],cfg['actor_noise_clip'])
        actions=jnp.clip(dist.mode()+noise,-1,1)
        nextq=net.select('target_critic')(batch['next_observations'],actions=actions).min(0)
        nextq-=cfg['alpha_critic']*jnp.sum((actions-batch['next_actions'])**2,axis=-1)
    y=batch['rewards']+cfg['discount']*batch['masks']*nextq
    finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)]))
    return q,tq,y,finite


def probe_summary(values,expectile,bounds,alpha):
    q,tq,y=[np.asarray(x,np.float64) for x in values[:3]]
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    return dict(q_mean=float(q.mean()),q_rms=rms(q),q_centered_rms=rms(q-q.mean()),
                q_abs_p99=float(np.quantile(np.abs(q),.99)),q_disagreement_rms=rms(q[0]-q[1]),
                td_rms=rms(q-y[None]),q_target_lag_rms=rms(q-tq),
                q_rms_over_return_bound=rms(q)/max(abs(bounds[0]),abs(bounds[1]),1.),
                finite=bool(np.asarray(values[3])))
