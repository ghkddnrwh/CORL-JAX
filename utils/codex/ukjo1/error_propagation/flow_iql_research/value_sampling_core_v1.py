"""Two V fits with either a repeated or an independent second minibatch.

Keep Q/BC batches, update counts, frozen target critic, fresh-V projected
bootstrap and old-online-Q EMA identical to the existing bounded_v2 recipe.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, config_for, create, make_scan, probe, probe_summary,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {
    'bounded_v2_repeated_batches': {'independent_second_V_batch': False},
    'bounded_v2_fresh_batches': {'independent_second_V_batch': True},
}


def build_update(profile_name, bounds, kind='ifql'):
    assert kind == 'ifql' and profile_name in PROFILES
    fields = ('observations','actions','next_observations','rewards','masks')
    if not PROFILES[profile_name]['independent_second_V_batch']:
        original = reference.build_update('bounded_v2', bounds, 'ifql')
        return lambda state,batch: original(state,{k:batch[k] for k in fields})
    def update(state,batch):
        agent=state.agent; net=agent.network; cfg=agent.config
        new_rng,rng=jax.random.split(agent.rng)
        tx=optax.adam(cfg['lr'])
        tq=net.select('target_critic')(batch['observations'],actions=batch['actions']).min(0)
        extra_q=net.select('target_critic')(batch['extra_value_observations'],actions=batch['extra_value_actions']).min(0)
        def value_loss(vp,observations,q):
            v=net.select('value')(observations,params={**net.params,'modules_value':vp})
            diff=q-v
            return jnp.mean(jnp.where(diff>=0,cfg['expectile'],1-cfg['expectile'])*diff**2)
        vp,vo=net.params['modules_value'],state.v_opt
        initial=value_loss(vp,batch['observations'],tq)
        for observations,q in ((batch['observations'],tq),(batch['extra_value_observations'],extra_q)):
            gradients=jax.grad(value_loss)(vp,observations,q)
            updates,vo=tx.update(gradients,vo,vp);vp=optax.apply_updates(vp,updates)
        fitted={**net.params,'modules_value':vp}
        next_v=net.select('value')(batch['next_observations'],params=fitted)
        y=batch['rewards']+cfg['discount']*batch['masks']*next_v
        bounded_y=jnp.clip(y,*bounds)
        _,actor_rng=jax.random.split(rng)
        base_batch={k:batch[k] for k in fields}
        def loss_fn(grad_params):
            qs=net.select('critic')(batch['observations'],actions=batch['actions'],params=grad_params)
            critic_loss=jnp.mean(jnp.sum((qs-bounded_y)**2,axis=0))
            actor_loss,actor_info=agent.actor_loss(base_batch,grad_params,actor_rng)
            return critic_loss+actor_loss,{
                'critic/critic_loss':critic_loss,'value/value_loss':initial,
                'value/post_fit_loss':value_loss(vp,batch['observations'],tq),
                'value/steps':jnp.asarray(2.,jnp.float32),
                'backup/clip_fraction':jnp.mean((y!=bounded_y).astype(jnp.float32)),
                'backup/clip_rms':jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/fresh_v_shift_rms':jnp.sqrt(jnp.mean((next_v-net.select('value')(batch['next_observations']))**2)),
                **{'actor/'+k:v for k,v in actor_info.items()}}
        new_net,logs=net.apply_loss_fn(loss_fn)
        new_net.params['modules_value']=vp
        agent.target_update(new_net,'critic')
        agent=agent.replace(network=new_net,rng=new_rng)
        logs={**logs,'backup/lower_clip_fraction':jnp.mean((y<bounds[0]).astype(jnp.float32)),
              'backup/upper_clip_fraction':jnp.mean((y>bounds[1]).astype(jnp.float32))}
        return state.replace(agent=agent,v_opt=vo,v_updates=state.v_updates+2,q_updates=state.q_updates+1),logs
    return update
