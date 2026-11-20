"""Bounded unit-interval return heads with MSE or Bernoulli-Bregman fitting.

For known rewards in[-C,0], p=1+(1-gamma)*Q/C preserves expectiles and
action order. Critic/value raw outputs are logits. The KL value loss is a
proper asymmetric Bregman score with exactly the original expectile root.
No new actor, temperature, reward oracle, or per-environment tuned constant.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import ENVIRONMENTS, create, make_scan
from utils.propagation_research.reward_bounds import REWARD_COMPONENTS
import jax
import jax.numpy as jnp
import numpy as np
import optax

PROFILES={
    'bounded_v2_unit_mse':{'geometry':'mse'},
    'bounded_v2_unit_kl':{'geometry':'bernoulli_kl'},
}


def config_for(environment,discount=None,expectile=.9,width=None,alpha=None):
    cfg=reference.config_for(environment,discount,expectile,width,alpha)
    components=REWARD_COMPONENTS[environment]
    cfg.update(return_representation='unit_interval_sigmoid_of_raw_logits',
        return_reward_components=components,return_affine_scale=(1-cfg['discount'])/components,
        return_inverse='original_Q=(sigmoid(raw_logit)-1)/return_affine_scale',
        policy_order='best9_min_raw_online_logits_monotone_equivalent_to_unit_return',
        value_scoring_rule='profile_selects_MSE_or_asymmetric_Bernoulli_KL',
        diagnostic_units='Q_V_TD_and_score_in_original_cost_units;separately_labelled_unit/logit_fields',
        diagnostic_score='samplewise_target_Q_minus_V_not_population_conditional_residual')
    return cfg


def normalized_reward(reward,mask,discount,components):
    # Stable version of a*r+1-gamma*mask; binary nav nonterminal term is exactly0.
    return (1-discount)*(1+reward/components)+discount*(1-mask)


def bernoulli_kl(target,logit):
    negative_entropy=(target*jnp.log(jnp.where(target>0,target,1))+
        (1-target)*jnp.log(jnp.where(target<1,1-target,1)))
    return optax.sigmoid_binary_cross_entropy(logit,target)+negative_entropy


def expectile_loss(logit,target,tau,geometry):
    probability=jax.nn.sigmoid(logit)
    weight=jnp.where(target>=probability,tau,1-tau)
    error=(probability-target)**2 if geometry=='mse' else bernoulli_kl(target,logit)
    return jnp.mean(weight*error)


def build_update(profile_name,bounds,kind='ifql'):
    assert profile_name in PROFILES and kind=='ifql'
    geometry=PROFILES[profile_name]['geometry']
    def update(state,batch):
        agent=state.agent;net=agent.network;cfg=agent.config
        scale=cfg['return_affine_scale'];tau=cfg['expectile']
        assert np.isclose(bounds[0],-1/scale,rtol=1e-12,atol=1e-10) and bounds[1]==0
        new_rng,rng=jax.random.split(agent.rng)
        target_q=jax.nn.sigmoid(net.select('target_critic')(
            batch['observations'],actions=batch['actions']).min(0))
        tx=optax.adam(cfg['lr']);vp,vo=net.params['modules_value'],state.v_opt
        def value_prediction(parameters):
            return net.select('value')(batch['observations'],params={**net.params,'modules_value':parameters})
        def value_loss(parameters):
            return expectile_loss(value_prediction(parameters),target_q,tau,geometry)
        initial=value_loss(vp)
        for _ in range(2):
            gradient=jax.grad(value_loss)(vp)
            increments,vo=tx.update(gradient,vo,vp);vp=optax.apply_updates(vp,increments)
        fitted={**net.params,'modules_value':vp}
        next_v=jax.nn.sigmoid(net.select('value')(batch['next_observations'],params=fitted))
        old_next_v=jax.nn.sigmoid(net.select('value')(batch['next_observations']))
        reward=normalized_reward(batch['rewards'],batch['masks'],cfg['discount'],cfg['return_reward_components'])
        y=reward+cfg['discount']*batch['masks']*next_v
        target=jnp.clip(y,0,1)
        _,actor_rng=jax.random.split(rng)
        def loss_fn(grad_params):
            logits=net.select('critic')(batch['observations'],actions=batch['actions'],params=grad_params)
            probability=jax.nn.sigmoid(logits)
            mse=jnp.mean(jnp.sum((probability-target)**2,axis=0))
            bce=jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits,target),axis=0))
            critic_loss=mse if geometry=='mse' else bce
            actor_loss,actor_info=agent.actor_loss(batch,grad_params,actor_rng)
            fitted_v=jax.nn.sigmoid(value_prediction(vp));diff=target_q-fitted_v
            w=jnp.where(diff>=0,tau,1-tau)
            return critic_loss+actor_loss,{
                'critic/critic_loss':critic_loss,'critic/unit_MSE':mse,'critic/unit_BCE':bce,
                'critic/original_units_target_MSE':mse/scale**2,
                'value/value_loss':initial,'value/post_fit_loss':value_loss(vp),
                'value/post_fit_original_units_expectile_MSE':jnp.mean(w*diff**2)/scale**2,
                'value/post_fit_sample_score_mean_original_units':jnp.mean(w*diff)/scale,
                'value/post_fit_sample_score_rms_original_units':jnp.sqrt(jnp.mean((w*diff)**2))/scale,
                'value/steps':jnp.asarray(2.,jnp.float32),
                'backup/clip_fraction':jnp.mean((y!=target).astype(jnp.float32)),
                'backup/clip_rms':jnp.sqrt(jnp.mean((y-target)**2))/scale,
                'backup/lower_clip_fraction':jnp.mean((y<0).astype(jnp.float32)),
                'backup/upper_clip_fraction':jnp.mean((y>1).astype(jnp.float32)),
                'backup/fresh_v_shift_rms':jnp.sqrt(jnp.mean((next_v-old_next_v)**2))/scale,
                'backup/unit_target_mean':target.mean(),
                **{'actor/'+k:v for k,v in actor_info.items()}}
        new_net,logs=net.apply_loss_fn(loss_fn)
        new_net.params['modules_value']=vp
        agent.target_update(new_net,'critic')
        return state.replace(agent=agent.replace(network=new_net,rng=new_rng),v_opt=vo,
            q_updates=state.q_updates+1,v_updates=state.v_updates+2),logs
    return update


@jax.jit
def probe(state,batch):
    net=state.agent.network;cfg=state.agent.config;a=cfg['return_affine_scale']
    qlog=net.select('critic')(batch['observations'],actions=batch['actions'])
    tqlog=net.select('target_critic')(batch['observations'],actions=batch['actions'])
    vlog=net.select('value')(batch['observations'])
    vnlog=net.select('value')(batch['next_observations'])
    qp,tqp,vp,vnp=map(jax.nn.sigmoid,(qlog,tqlog,vlog,vnlog))
    q,tq,v,vn=((x-1)/a for x in (qp,tqp,vp,vnp))
    y=batch['rewards']+cfg['discount']*batch['masks']*vn
    leaves=jax.tree_util.tree_leaves(state)+[qlog,tqlog,vlog,vnlog]
    finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))for x in leaves]))
    return q,tq,v,y,finite,qlog,vlog,qp,vp


def probe_summary(values,expectile,bounds,alpha):
    result=reference.probe_summary(values[:5],expectile,bounds,alpha)
    qlog,vlog,qp,vp=(np.asarray(x,np.float64)for x in values[5:])
    result.update(raw_Q_logit_abs_max=float(np.abs(qlog).max()),raw_V_logit_abs_max=float(np.abs(vlog).max()),
        unit_Q_mean=float(qp.mean()),unit_V_mean=float(vp.mean()),
        unit_Q_near_endpoint_fraction=float(np.mean((qp<1e-4)|(qp>1-1e-4))),
        unit_V_near_endpoint_fraction=float(np.mean((vp<1e-4)|(vp>1-1e-4))))
    return result
