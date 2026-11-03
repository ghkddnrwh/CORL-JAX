"""Learned-V mean-target-Q control for the finite-expectile mean backup.

Only the fixed V teacher changes from target twin minimum to twin mean versus
boundedV2. Two same-batch V fits, fresh learned-V bootstrap, projection, old
online-Q EMA and unweighted behavior flow stay unchanged. No finite action
samples are used by the backup in this control.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import ENVIRONMENTS, State, create, make_scan, probe
import jax
import jax.numpy as jnp
import numpy as np
import optax

PROFILES={'bounded_v2_mean_value_teacher':reference.PROFILES['bounded_v2']}


def config_for(environment,discount=None,expectile=.9,width=None,alpha=None):
    cfg=reference.config_for(environment,discount,expectile,width,alpha)
    cfg.update(value_teacher_Q_aggregation='mean',backup_kind='learned_V_fitted_to_mean_target_Q',
        diagnostic_score_teacher='samplewise_mean_target_Q_not_population_conditional_residual')
    return cfg


UPSTREAM_UPDATE_FILE_SHA256='c6136b093586ff20590a450f6bea83773210faa315978aafbb7a759bfe5f460e'

def _build_update_mean_teacher(profile_name, bounds, kind='iql'):
    profile = PROFILES[profile_name]
    lower, upper = bounds
    if not profile:
        def update(state, batch):
            agent, info = state.agent.update(batch)
            return state.replace(agent=agent, v_updates=state.v_updates+int(kind != 'fql'),
                                 q_updates=state.q_updates+1), info
        return update

    def update(state, batch):
        agent = state.agent
        net = agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        tx = optax.adam(cfg['lr'])
        # Frozen same-minibatch target Q for all V steps.
        target_q = net.select('target_critic')(batch['observations'], actions=batch['actions']).mean(0)
        def value_loss(params):
            values = net.select('value')(batch['observations'], params={**net.params, 'modules_value': params})
            diff = target_q-values
            return jnp.mean(jnp.where(diff >= 0, cfg['expectile'], 1-cfg['expectile'])*diff**2)
        vp, vo = net.params['modules_value'], state.v_opt
        initial_v_loss = value_loss(vp)
        vsteps = profile.get('v_steps', 1)
        for _ in range(vsteps):
            grads = jax.grad(value_loss)(vp)
            upd, vo = tx.update(grads, vo, vp)
            vp = optax.apply_updates(vp, upd)
        vparams = {**net.params, 'modules_value': vp} if profile.get('fresh') else net.params
        next_v = net.select('value')(batch['next_observations'], params=vparams)
        y = batch['rewards'] + cfg['discount'] * batch['masks'] * next_v
        bounded_y = jnp.clip(y, lower, upper) if profile.get('bound') else y
        # Actor sees original old online Q/V. No extra Q/actor updates are made.
        _, actor_rng = jax.random.split(rng)
        def loss_fn(grad_params):
            qs = net.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
            critic_loss = jnp.mean(jnp.sum((qs-bounded_y)**2, axis=0))
            actor_loss, actor_info = agent.actor_loss(batch, grad_params, actor_rng)
            return critic_loss + actor_loss, {
                'critic/critic_loss': critic_loss,
                'value/value_loss': initial_v_loss,
                'value/post_fit_loss': value_loss(vp),
                'value/steps': jnp.asarray(vsteps, jnp.float32),
                'backup/clip_fraction': jnp.mean((y != bounded_y).astype(jnp.float32)),
                'backup/clip_rms': jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/fresh_v_shift_rms': jnp.sqrt(jnp.mean((next_v-net.select('value')(batch['next_observations']))**2)),
                **{'actor/'+k:v for k,v in actor_info.items()}}
        new_net, info = net.apply_loss_fn(loss_fn)
        # V's entries in the joint optimizer stay zero: their separate Adam owns
        # all V updates. Installing the fitted parameters is the only V mutation.
        new_net.params['modules_value'] = vp
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        return state.replace(agent=agent, v_opt=vo, v_updates=state.v_updates+vsteps,
                             q_updates=state.q_updates+1), info
    return update


def build_update(profile_name,bounds,kind='ifql'):
    assert profile_name=='bounded_v2_mean_value_teacher'and kind=='ifql'
    update=_build_update_mean_teacher(profile_name,bounds,kind)
    def monitored(state,batch):
        new,logs=update(state,batch)
        net=state.agent.network;cfg=state.agent.config
        heads=net.select('target_critic')(batch['observations'],actions=batch['actions'])
        fitted=new.agent.network.select('value')(batch['observations'])
        delta=heads.mean(0)-fitted;score=jnp.where(delta>=0,cfg['expectile'],1-cfg['expectile'])*delta
        vn=new.agent.network.select('value')(batch['next_observations'])
        y=batch['rewards']+cfg['discount']*batch['masks']*vn
        return new,{**logs,
            'backup/lower_clip_fraction':jnp.mean((y<bounds[0]).astype(jnp.float32)),
            'backup/upper_clip_fraction':jnp.mean((y>bounds[1]).astype(jnp.float32)),
            'value/teacher_mean_minus_min_mean':jnp.mean(heads.mean(0)-heads.min(0)),
            'value/post_fit_sample_mean_teacher_score_mean':jnp.mean(score),
            'value/post_fit_sample_mean_teacher_score_rms':jnp.sqrt(jnp.mean(score**2))}
    return monitored


def probe_summary(values,expectile,bounds,alpha):
    result=reference.probe_summary(values,expectile,bounds,alpha)
    result['counterfactual_min_teacher_score_mean']=result.pop('score_mean')
    result['counterfactual_min_teacher_score_rms']=result.pop('score_rms')
    tq=np.asarray(values[1],np.float64);v=np.asarray(values[2],np.float64)
    delta=tq.mean(0)-v;score=np.where(delta>=0,expectile,1-expectile)*delta
    result.update(score_mean=float(np.mean(score)),score_rms=float(np.sqrt(np.mean(score**2))))
    return result
