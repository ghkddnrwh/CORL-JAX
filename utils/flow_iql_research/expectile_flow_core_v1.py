"""Fit the bounded policy tilt implied by the expectile first-order condition.

For q=min(target-Q), w=tau if q>=V else 1-tau. The conditional distribution
pi(a|s) proportional to mu(a|s)*w(s,a) obeys E_pi[q]-V=G(s)/Z(s), where
G=E_mu[w*(q-V)] is the conditional expectile score residual and Z=E_mu[w].
Weighted flow matching targets this tilt without an exponential temperature.
Finite data/model error can invalidate practical improvement; the identity
alone does not imply that learned-policy rollouts improve.
"""
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import ENVIRONMENTS, State, make_scan, probe, probe_summary
from utils.flow_iql_research.vendor.ifql import IFQLAgent
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_expectile_flow': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(actor_weighting='bounded_expectile_sign_old_target_Q_minus_old_V', num_samples=9)
    return config


class ExpectileFlowAgent(IFQLAgent):
    def actor_loss(self, batch, grad_params, rng=None):
        # Same FM noise split and vector field as official IFQL.
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        xt = (1-t)*x0+t*batch['actions']
        velocity = batch['actions']-x0
        prediction = self.network.select('actor_flow')(
            batch['observations'], xt, t, params=grad_params)
        per_example = jnp.mean((prediction-velocity)**2, axis=-1)
        q = self.network.select('target_critic')(
            batch['observations'], actions=batch['actions']).min(0)
        v = self.network.select('value')(batch['observations'])
        advantage = q-v
        tau = self.config['expectile']
        weights = jax.lax.stop_gradient(jnp.where(advantage >= 0, tau, 1-tau))
        loss = jnp.mean(weights*per_example)/weights.mean()
        return loss, dict(
            actor_loss=loss, unweighted_flow_loss=per_example.mean(),
            positive_advantage_fraction=jnp.mean((advantage >= 0).astype(jnp.float32)),
            target_advantage_mean=advantage.mean(),
            expectile_score_batch_mean=(weights*advantage).mean(),
            weight_mean=weights.mean(),
            weight_ess_fraction=weights.mean()**2/jnp.mean(weights**2),
            weight_min=weights.min(), weight_max=weights.max(),
        )


def create(seed, observations, actions, config, kind='ifql'):
    assert kind == 'ifql'
    agent = ExpectileFlowAgent.create(seed, observations, actions, dict(config))
    return State(agent, optax.adam(config['lr']).init(agent.network.params['modules_value']),
                 jnp.asarray(0), jnp.asarray(0))


def build_update(profile_name, bounds, kind='ifql'):
    assert profile_name == 'bounded_v2_expectile_flow' and kind == 'ifql'
    return reference.build_update('bounded_v2', bounds, kind)
