"""Two V fits with independent replay batches and a frozen target critic.

Only the second V minibatch differs from bounded_v2. Q and BC flow see the
original replay stream. Both V fits use the old target critic; the critic uses
the new V, a known-return projection, and the original target-EMA convention.
"""
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, config_for, create, make_scan, probe, probe_summary,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {'independent_v2': {'v_steps': 2, 'fresh': True, 'bound': True}}


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'independent_v2'
    lower, upper = bounds

    def update(state, batch):
        agent = state.agent
        net, cfg = agent.network, agent.config
        new_rng, rng = jax.random.split(agent.rng)
        tx = optax.adam(cfg['lr'])
        observations = (batch['observations'], batch['vfit_observations'])
        actions = (batch['actions'], batch['vfit_actions'])
        target_qs = tuple(net.select('target_critic')(o, actions=a).min(0)
                          for o, a in zip(observations, actions))

        def objective(vp, index):
            v = net.select('value')(observations[index],
                                    params={**net.params, 'modules_value': vp})
            diff = target_qs[index] - v
            return jnp.mean(jnp.where(diff >= 0, cfg['expectile'],
                                      1-cfg['expectile']) * diff**2)

        vp, vo = net.params['modules_value'], state.v_opt
        initial = (objective(vp, 0), objective(vp, 1))
        gradients = jax.grad(lambda p: objective(p, 0))(vp)
        updates, vo = tx.update(gradients, vo, vp)
        vp = optax.apply_updates(vp, updates)
        # The second batch was sampled independently, before seeing this fit.
        heldout_after_first = objective(vp, 1)
        gradients = jax.grad(lambda p: objective(p, 1))(vp)
        updates, vo = tx.update(gradients, vo, vp)
        vp = optax.apply_updates(vp, updates)

        next_v = net.select('value')(batch['next_observations'],
                                     params={**net.params, 'modules_value': vp})
        y = batch['rewards'] + cfg['discount'] * batch['masks'] * next_v
        bounded_y = jnp.clip(y, lower, upper)
        _, actor_rng = jax.random.split(rng)

        def loss_fn(params):
            q = net.select('critic')(batch['observations'],
                                     actions=batch['actions'], params=params)
            critic_loss = jnp.mean(jnp.sum((q-bounded_y)**2, axis=0))
            actor_loss, actor_info = agent.actor_loss(batch, params, actor_rng)
            return critic_loss + actor_loss, {
                'critic/critic_loss': critic_loss,
                'value/value_loss': initial[0],
                'value/post_fit_loss': objective(vp, 0),
                'value/independent_pre_loss': initial[1],
                'value/independent_after_first_loss': heldout_after_first,
                'value/independent_post_loss': objective(vp, 1),
                'value/steps': jnp.asarray(2., jnp.float32),
                'backup/clip_fraction': jnp.mean((y != bounded_y).astype(jnp.float32)),
                'backup/clip_rms': jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/lower_clip_fraction': jnp.mean((y < lower).astype(jnp.float32)),
                'backup/upper_clip_fraction': jnp.mean((y > upper).astype(jnp.float32)),
                'backup/fresh_v_shift_rms': jnp.sqrt(jnp.mean(
                    (next_v-net.select('value')(batch['next_observations']))**2)),
                **{'actor/'+k: v for k, v in actor_info.items()},
            }

        new_net, info = net.apply_loss_fn(loss_fn)
        new_net.params['modules_value'] = vp
        agent.target_update(new_net, 'critic')
        return state.replace(agent=agent.replace(network=new_net, rng=new_rng),
                             v_opt=vo, v_updates=state.v_updates+2,
                             q_updates=state.q_updates+1), info

    return update
