"""Replace the learned-V backup by a behavior-flow sampled expectile.

Keep official IFQL BC flow and the existing two auxiliary V fits. For each next
state generate N behavior actions, compute their frozen target-Q minimum heads,
solve the finite-proposal expectile exactly, and use it in the bounded backup.
Thus V fitting residual cannot enter this backup. Finite-proposal bias and
behavior-model error remain. For a fixed generator, the ideal sampled operator
is gamma-Lipschitz in Q; this is not a neural-training convergence guarantee.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, create, make_scan, probe, probe_summary,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_empirical_backup': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(backup_kind='finite_behavior_flow_target_Q_expectile', backup_samples=9)
    return config


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'bounded_v2_empirical_backup' and kind == 'ifql'

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
