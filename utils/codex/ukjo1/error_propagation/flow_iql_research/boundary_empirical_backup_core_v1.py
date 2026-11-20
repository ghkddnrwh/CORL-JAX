"""Finite expectile backup with a known success-state Q boundary condition.

For OGBench state-based reward/mask relabeling, a mask-zero state has Q=0
for every action. Add equal-weight supervised boundary loss on32 success
states with their observed and one independent uniform action each. BC, V
fits and main replay streams retain the original finite-backup recipe.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, create, make_scan,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_boundary_empirical_backup': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(backup_kind='finite_behavior_flow_target_Q_expectile', backup_samples=9,
                  boundary_batch_size=32, boundary_coefficient=1.,
                  boundary_kind='zero_Q_for_all_actions_at_state_based_mask_zero')
    return config


@jax.jit
def probe(state, batch):
    q, tq, v, learned_y, finite = reference.probe(state, batch)
    net, cfg = state.agent.network, state.agent.config
    actions, _ = selection_v2.proposals(state.agent, batch['next_observations'],
        jax.random.PRNGKey(19402), cfg['backup_samples'], 'behavior')
    repeated = jnp.broadcast_to(batch['next_observations'],
        (cfg['backup_samples'], *batch['next_observations'].shape))
    root = selection_v2.empirical_expectile(net.select('target_critic')(repeated, actions=actions).min(0), cfg['expectile'])
    y = batch['rewards']+cfg['discount']*batch['masks']*root
    bq = net.select('critic')(batch['boundary_observations'], actions=batch['boundary_actions'])
    random_actions = jax.random.uniform(jax.random.PRNGKey(19403), batch['boundary_actions'].shape, minval=-1., maxval=1.)
    uq = net.select('critic')(batch['boundary_observations'], actions=random_actions)
    bv = net.select('value')(batch['boundary_observations'])
    return q, tq, v, y, finite, learned_y, bq, uq, bv, batch['boundary_valid']


def probe_summary(values, expectile, bounds, alpha):
    import numpy as np
    result = reference.probe_summary(values[:5], expectile, bounds, alpha)
    q, y, learned_y, bq, uq, bv, valid = [np.asarray(values[index], np.float64) for index in (0, 3, 5, 6, 7, 8, 9)]
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    count = max(valid.sum(), 1.)
    result.update(snapshot_finite_backup_td_RMS=result['td_rms'],
        snapshot_projected_finite_backup_td_RMS=rms(q-np.clip(y, *bounds)[None]),
        snapshot_learned_V_only_counterfactual_td_RMS=rms(q-learned_y[None]),
        snapshot_boundary_active_fraction=float(valid.mean()),
        snapshot_boundary_observed_Q_RMS=float(np.sqrt(np.sum(bq*bq*valid[None])/(2*count))),
        snapshot_boundary_uniform_Q_RMS=float(np.sqrt(np.sum(uq*uq*valid[None])/(2*count))),
        snapshot_boundary_V_RMS=float(np.sqrt(np.sum(bv*bv*valid)/count)))
    return result


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'bounded_v2_boundary_empirical_backup' and kind == 'ifql'

    def update(state, batch):
        agent, net = state.agent, state.agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        tx = optax.adam(cfg['lr'])
        target_q = net.select('target_critic')(
            batch['observations'], actions=batch['actions']).min(0)

        def value_loss(params):
            v = net.select('value')(batch['observations'],
                params={**net.params, 'modules_value': params})
            difference = target_q-v
            return jnp.mean(jnp.where(difference >= 0, cfg['expectile'], 1-cfg['expectile'])*difference**2)

        vp, vo = net.params['modules_value'], state.v_opt
        initial_loss = value_loss(vp)
        for _ in range(2):
            gradients = jax.grad(value_loss)(vp)
            increments, vo = tx.update(gradients, vo, vp)
            vp = optax.apply_updates(vp, increments)

        # Independent folded-in key does not alter official FM or replay RNGs.
        proposal_key = jax.random.fold_in(rng, 61723)
        next_actions, _ = selection_v2.proposals(agent, batch['next_observations'],
                                                proposal_key, cfg['backup_samples'], 'behavior')
        next_observations = jnp.broadcast_to(batch['next_observations'],
                                            (cfg['backup_samples'], *batch['next_observations'].shape))
        target_proposal_q = net.select('target_critic')(
            next_observations, actions=next_actions).min(0)
        root = selection_v2.empirical_expectile(target_proposal_q, cfg['expectile'])
        learned_next_v = net.select('value')(batch['next_observations'],
            params={**net.params, 'modules_value': vp})
        y = batch['rewards']+cfg['discount']*batch['masks']*root
        bounded_y = jnp.clip(y, *bounds)
        difference = target_proposal_q-root
        weights = jnp.where(difference >= 0, cfg['expectile'], 1-cfg['expectile'])
        score = jnp.mean(weights*difference, axis=0)
        _, actor_rng = jax.random.split(rng)
        boundary_obs = batch['boundary_observations']
        boundary_data_actions = batch['boundary_actions']
        random_actions = jax.random.uniform(jax.random.fold_in(rng, 67101),
            boundary_data_actions.shape, minval=-1., maxval=1.)
        boundary_all_obs = jnp.concatenate((boundary_obs, boundary_obs), axis=0)
        boundary_all_actions = jnp.concatenate((boundary_data_actions, random_actions), axis=0)
        boundary_valid = jnp.concatenate((batch['boundary_valid'], batch['boundary_valid']))

        def loss_fn(grad_params):
            qs = net.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
            critic_loss = jnp.mean(jnp.sum((qs-bounded_y)**2, axis=0))
            boundary_q = net.select('critic')(boundary_all_obs, actions=boundary_all_actions, params=grad_params)
            boundary_loss = jnp.mean(jnp.sum(boundary_q**2, axis=0)*boundary_valid)
            actor_loss, actor_info = agent.actor_loss(batch, grad_params, actor_rng)
            return critic_loss+cfg['boundary_coefficient']*boundary_loss+actor_loss, {
                'critic/critic_loss': critic_loss+cfg['boundary_coefficient']*boundary_loss,
                'critic/main_TD_loss': critic_loss,
                'critic/boundary_loss': boundary_loss,
                'critic/boundary_coefficient': jnp.asarray(cfg['boundary_coefficient']),
                'critic/boundary_active_fraction': jnp.mean(boundary_valid),
                'critic/boundary_observed_Q_RMS': jnp.sqrt(jnp.mean(boundary_q[:, :boundary_data_actions.shape[0]]**2)),
                'critic/boundary_uniform_Q_RMS': jnp.sqrt(jnp.mean(boundary_q[:, boundary_data_actions.shape[0]:]**2)),
                'value/value_loss': initial_loss, 'value/post_fit_loss': value_loss(vp),
                'value/steps': jnp.asarray(2., jnp.float32),
                'backup/clip_fraction': jnp.mean((y != bounded_y).astype(jnp.float32)),
                'backup/clip_rms': jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/lower_clip_fraction': jnp.mean((y < bounds[0]).astype(jnp.float32)),
                'backup/upper_clip_fraction': jnp.mean((y > bounds[1]).astype(jnp.float32)),
                'backup/empirical_root_minus_learned_V_mean': jnp.mean(root-learned_next_v),
                'backup/empirical_root_minus_learned_V_rms': jnp.sqrt(jnp.mean((root-learned_next_v)**2)),
                'backup/finite_proposal_score_abs_mean': jnp.mean(jnp.abs(score)),
                'backup/finite_proposal_score_abs_max': jnp.max(jnp.abs(score)),
                'backup/proposal_Q_within_state_std_mean': jnp.std(target_proposal_q, axis=0).mean(),
                'backup/learned_V_above_all_proposals_fraction': jnp.mean(
                    (learned_next_v > target_proposal_q.max(0)).astype(jnp.float32)),
                **{'actor/'+k: v for k, v in actor_info.items()},
            }

        new_net, info = net.apply_loss_fn(loss_fn)
        new_net.params['modules_value'] = vp
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        return state.replace(agent=agent, v_opt=vo, v_updates=state.v_updates+2,
                             q_updates=state.q_updates+1), info

    return update
