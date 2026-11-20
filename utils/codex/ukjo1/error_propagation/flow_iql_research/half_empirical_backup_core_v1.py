"""Damp the learned-value displacement toward a finite proposal expectile.

The production recipe uses a fixed one-half correction, with all other IFQLV2
updates and random streams retained. The finite proposal score and convex loss
are diagnostic objectives under the current generator, not oracle residuals.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, create, make_scan,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_half_empirical_backup': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(backup_kind='half_correction_to_finite_target_Q_expectile', backup_samples=9,
                  backup_correction_fraction=.5)
    return config


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'bounded_v2_half_empirical_backup' and kind == 'ifql'
    return _build_update(bounds, .5)


def corrected_value(learned_value, empirical_root, fraction=.5):
    return (1-fraction)*learned_value+fraction*empirical_root


@jax.jit
def probe(state, batch):
    q, tq, v, learned_y, finite = reference.probe(state, batch)
    net, cfg = state.agent.network, state.agent.config
    actions, _ = selection_v2.proposals(state.agent, batch['next_observations'],
        jax.random.PRNGKey(19402), cfg['backup_samples'], 'behavior')
    observations = jnp.broadcast_to(batch['next_observations'],
        (cfg['backup_samples'], *batch['next_observations'].shape))
    proposal_q = net.select('target_critic')(observations, actions=actions).min(0)
    root = selection_v2.empirical_expectile(proposal_q, cfg['expectile'])
    vn = net.select('value')(batch['next_observations'])
    y = batch['rewards']+cfg['discount']*batch['masks']*corrected_value(vn, root)
    return q, tq, v, y, finite, learned_y, root, vn


def probe_summary(values, expectile, bounds, alpha):
    import numpy as np
    result = reference.probe_summary(values[:5], expectile, bounds, alpha)
    q, y, learned_y, root, vn = [np.asarray(values[i], np.float64) for i in (0, 3, 5, 6, 7)]
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    result.update(snapshot_half_backup_td_rms=result['td_rms'],
                  snapshot_projected_half_backup_td_rms=rms(q-np.clip(y, *bounds)[None]),
                  snapshot_learned_V_only_counterfactual_td_rms=rms(q-learned_y[None]),
                  snapshot_finite_root_minus_V_mean=float(np.mean(root-vn)),
                  snapshot_finite_root_minus_V_rms=rms(root-vn),
                  snapshot_half_backup_projection_fraction=float(np.mean(y != np.clip(y, *bounds))))
    return result


def _build_update(bounds, fraction):
    # Private endpoints are used only for implementation validation, never CLI tuning.
    assert 0 <= fraction <= 1

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
        next_v = corrected_value(learned_next_v, root, fraction)
        y = batch['rewards']+cfg['discount']*batch['masks']*next_v
        bounded_y = jnp.clip(y, *bounds)
        difference = target_proposal_q-root
        weights = jnp.where(difference >= 0, cfg['expectile'], 1-cfg['expectile'])
        score = jnp.mean(weights*difference, axis=0)
        def proposal_objective(value):
            delta = target_proposal_q-value
            weight = jnp.where(delta >= 0, cfg['expectile'], 1-cfg['expectile'])
            return jnp.mean(weight*delta**2, axis=0), jnp.mean(weight*delta, axis=0)
        before_loss, before_score = proposal_objective(learned_next_v)
        after_loss, after_score = proposal_objective(next_v)
        _, actor_rng = jax.random.split(rng)

        def loss_fn(grad_params):
            qs = net.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
            critic_loss = jnp.mean(jnp.sum((qs-bounded_y)**2, axis=0))
            actor_loss, actor_info = agent.actor_loss(batch, grad_params, actor_rng)
            return critic_loss+actor_loss, {
                'critic/critic_loss': critic_loss,
                'value/value_loss': initial_loss, 'value/post_fit_loss': value_loss(vp),
                'value/steps': jnp.asarray(2., jnp.float32),
                'backup/clip_fraction': jnp.mean((y != bounded_y).astype(jnp.float32)),
                'backup/clip_rms': jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/lower_clip_fraction': jnp.mean((y < bounds[0]).astype(jnp.float32)),
                'backup/upper_clip_fraction': jnp.mean((y > bounds[1]).astype(jnp.float32)),
                'backup/empirical_root_minus_learned_V_mean': jnp.mean(root-learned_next_v),
                'backup/empirical_root_minus_learned_V_rms': jnp.sqrt(jnp.mean((root-learned_next_v)**2)),
                'backup/correction_fraction': jnp.asarray(fraction, jnp.float32),
                'backup/corrected_minus_learned_V_rms': jnp.sqrt(jnp.mean((next_v-learned_next_v)**2)),
                'backup/remaining_distance_to_root_rms': jnp.sqrt(jnp.mean((next_v-root)**2)),
                'backup/proposal_expectile_loss_before': jnp.mean(before_loss),
                'backup/proposal_expectile_loss_after': jnp.mean(after_loss),
                'backup/proposal_expectile_loss_increase_fraction': jnp.mean((after_loss > before_loss+1e-5).astype(jnp.float32)),
                'backup/proposal_score_abs_before': jnp.mean(jnp.abs(before_score)),
                'backup/proposal_score_abs_after': jnp.mean(jnp.abs(after_score)),
                'backup/proposal_score_increase_fraction': jnp.mean((jnp.abs(after_score) > jnp.abs(before_score)+1e-5).astype(jnp.float32)),
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
