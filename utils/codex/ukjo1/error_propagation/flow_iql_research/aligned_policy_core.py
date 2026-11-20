"""FQL-style student with Q guidance that stops at the old IQL V.

Relative to the published-coefficient FQL-policy control, only the student Q
objective changes: -mean(Q-V) becomes mean(relu(V-Q)). Q/V, BC flow, distillation,
network sizes, optimizer clocks and one-step deployment are unchanged. The
threshold is learned V, so its residual/model error remains a material risk.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import (
    ENVIRONMENTS, State, make_scan, probe, probe_summary,
)
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_value_hinge': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(policy_guidance='linear_hinge_at_old_IQL_V')
    return config


class AlignedFlowIQLAgent(reference.FlowIQLAgent):
    def actor_loss(self, batch, grad_params, rng=None):
        config, net = self.config, self.network
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, noise_rng = jax.random.split(rng, 4)
        x0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        xt = (1-t)*x0+t*batch['actions']
        velocity = batch['actions']-x0
        prediction = net.select('actor_flow')(batch['observations'], xt, t, params=grad_params)
        per_example = jnp.mean((prediction-velocity)**2, axis=-1)
        bc_loss = per_example.mean()
        data_qs = net.select('critic')(batch['observations'], actions=batch['actions'])
        v = net.select('value')(batch['observations'])
        advantage = data_qs.min(0)-v
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        teacher = self.compute_flow_actions(batch['observations'], noises)
        raw = net.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill = jnp.mean((raw-teacher)**2)
        student = jnp.clip(raw, -1, 1)
        # Keep the official FQL mean-head policy objective for a single-change
        # control. Both Q and the old V are constants with respect to their own
        # parameters here; only the student receives the Q action gradient.
        q = net.select('critic')(batch['observations'], actions=student).mean(0)
        deficit = v-q
        qloss = jnp.maximum(deficit, 0).mean()
        loss = bc_loss+config['distill_coefficient']*distill+qloss
        return loss, dict(
            actor_loss=loss, bc_flow_loss=bc_loss, distill_loss=distill,
            q_loss=qloss, q_loss_normalizer=jnp.asarray(1.),
            q_loss_normalization_active=jnp.asarray(0.),
            guidance_active_fraction=jnp.mean((deficit > 0).astype(jnp.float32)),
            guidance_deficit_mean=deficit.mean(),
            guidance_deficit_positive_mean=jnp.maximum(deficit, 0).mean(),
            guidance_surplus_positive_mean=jnp.maximum(-deficit, 0).mean(),
            data_adv_mean=advantage.mean(), data_adv_abs_mean=jnp.abs(advantage).mean(),
            student_adv_mean=(q-v).mean(), student_minus_data_q=(q-data_qs.mean(0)).mean(),
            teacher_displacement_rms=jnp.sqrt(distill),
            student_saturation_fraction=jnp.mean((jnp.abs(raw) > 1).astype(jnp.float32)),
        )


def create(seed, observations, actions, config, kind='fql_policy'):
    assert kind == 'fql_policy'
    config = dict(config)
    config['policy_type'] = 'fql_policy'
    agent = AlignedFlowIQLAgent.create(seed, observations, actions, config)
    return State(agent, optax.adam(config['lr']).init(agent.network.params['modules_value']),
                 jnp.asarray(0), jnp.asarray(0))


def build_update(profile_name, bounds, kind='fql_policy'):
    assert profile_name == 'bounded_v2_value_hinge' and kind == 'fql_policy'
    return reference.build_update('bounded_v2', bounds, kind)
