"""Rank-weighted real-action flow matching with unchanged IQL V2 / BC.

Nine old-BC comparator actions provide a bounded, scale-free importance
weight for each actual dataset action. The policy target is always the real
action. Q/V and the unweighted behavior flow do not receive policy gradients.
"""
from utils.flow_iql_research import rank_flow_core_v1 as reference
from utils.flow_iql_research.rank_flow_core_v1 import ENVIRONMENTS, State, make_scan, probe, probe_summary
import jax
import jax.numpy as jnp


PROFILES = {'bounded_v2_data_rank_flow': reference.PROFILES['bounded_v2_rank_flow']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    config = reference.config_for(environment, discount, expectile, width, alpha)
    config.update(policy_objective='FM_of_real_data_actions_with_bounded_best9_rank_weights',
                  rank_comparators=9, rank_selected_order=9,
                  rank_ties='independent_uniform_lexicographic',
                  rank_loss_normalization='unbiased_batch_mean_without_weight_normalization')
    return config


def data_rank_weights(data_q, comparator_q, data_tie, comparator_tie):
    """Average nine leave-one-comparator-out best-of-nine estimators.

    C is the count of nine comparators below the data action in augmented
    (Q, Uniform) order. The averaged weight is choose(C, 8), hence 0, 1, or 9.
    Under exact conditional behavior sampling, E[w|s,a,u] = 9 F(Q,u)^8.
    """
    assert comparator_q.shape[0] == 9 and comparator_tie.shape == comparator_q.shape
    below = (comparator_q < data_q) | ((comparator_q == data_q) & (comparator_tie < data_tie))
    count = below.astype(jnp.int32).sum(0)
    weights = jnp.where(count == 9, 9., jnp.where(count == 8, 1., 0.))
    return jax.lax.stop_gradient(weights), count


class DataRankFlowAgent(reference.RankFlowAgent):
    def actor_loss(self, batch, grad_params, rng=None):
        net, config = self.network, self.config
        assert config['rank_comparators'] == config['rank_selected_order'] == 9
        shape = batch['actions'].shape
        # Match the official IFQL's unweighted BC stream exactly in real
        # arithmetic. Independent folded keys do not consume the replay stream.
        _, xkey, tkey = jax.random.split(rng, 3)
        x0 = jax.random.normal(xkey, shape)
        t = jax.random.uniform(tkey, (*shape[:-1], 1))
        xt = (1-t)*x0+t*batch['actions']
        prediction = net.select('actor_flow')(batch['observations'], xt, t, params=grad_params)
        bc_loss = jnp.mean((prediction-(batch['actions']-x0))**2)

        observations = jnp.broadcast_to(batch['observations'], (9, *batch['observations'].shape))
        noise = jax.random.normal(jax.random.fold_in(rng, 71927), (9, *shape))
        comparator_actions = self.compute_flow_actions(observations, noise)
        comparator_q = net.select('critic')(observations, actions=comparator_actions).min(0)
        data_q = net.select('critic')(batch['observations'], actions=batch['actions']).min(0)
        ties = jax.random.uniform(jax.random.fold_in(rng, 72931), (10, *shape[:-1]))
        weights, count = data_rank_weights(data_q, comparator_q, ties[0], ties[1:])

        p0 = jax.random.normal(jax.random.fold_in(rng, 71928), shape)
        pt = jax.random.uniform(jax.random.fold_in(rng, 71929), (*shape[:-1], 1))
        pxt = (1-pt)*p0+pt*batch['actions']
        policy_prediction = net.select('actor_policy_flow')(batch['observations'], pxt, pt, params=grad_params)
        per_example = jnp.mean((policy_prediction-(batch['actions']-p0))**2, axis=-1)
        # Do not divide by the realized batch sum of weights: that would change
        # this unbiased estimating objective and hide zero-mass minibatches.
        policy_loss = jnp.mean(weights*per_example)
        mass = weights.mean()
        loss = bc_loss+policy_loss
        return loss, dict(actor_loss=loss, bc_flow_loss=bc_loss, policy_flow_loss=policy_loss,
            real_data_rank_weight_mean=mass,
            real_data_rank_weight_ESS_fraction=mass**2/jnp.maximum((weights**2).mean(), 1e-20),
            real_data_rank_zero_weight_fraction=jnp.mean((weights == 0).astype(jnp.float32)),
            real_data_rank_weight_one_fraction=jnp.mean((weights == 1).astype(jnp.float32)),
            real_data_rank_weight_nine_fraction=jnp.mean((weights == 9).astype(jnp.float32)),
            real_data_rank_zero_batch_mass=(mass == 0).astype(jnp.float32),
            real_data_rank_count_mean=count.astype(jnp.float32).mean(),
            real_data_Q_minus_comparator_mean=jnp.mean(data_q-comparator_q.mean(0)),
            real_data_comparator_numeric_tie_fraction=jnp.mean((data_q == comparator_q).astype(jnp.float32)),
            real_data_weighted_Q_minus_same_state_comparator_mean=jnp.sum(weights*(data_q-comparator_q.mean(0)))/jnp.maximum(weights.sum(), 1.),
            real_data_weighted_action_saturation=jnp.sum(weights*jnp.mean((jnp.abs(batch['actions']) >= .98).astype(jnp.float32), axis=-1))/jnp.maximum(weights.sum(), 1.),
            comparator_candidate_Q_std_mean=jnp.std(comparator_q, axis=0).mean(),
            comparator_candidate_action_std_mean=jnp.std(comparator_actions, axis=0).mean(),
            policy_real_data_unweighted_FM_loss=per_example.mean())


def create(seed, observations, actions, config, kind='rank_flow'):
    assert kind == 'rank_flow'
    # Preserve exactly the architecture and initial common Q/V/BC/policy
    # parameters of the synthetic-label rank-flow control.
    base = reference.create(seed, observations, actions, config, kind)
    agent = DataRankFlowAgent(base.agent.rng, network=base.agent.network, config=base.agent.config)
    return base.replace(agent=agent)


def build_update(profile_name, bounds, kind='rank_flow'):
    assert profile_name == 'bounded_v2_data_rank_flow' and kind == 'rank_flow'
    return reference.build_update('bounded_v2_rank_flow', bounds, kind)
