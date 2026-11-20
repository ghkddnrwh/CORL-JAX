"""Validate correction endpoints, finite-root geometry, isolation and resume."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import half_empirical_backup_core_v1 as half_core
from utils.flow_iql_research import empirical_backup_core_v1 as full_core
from utils.flow_iql_research import policy_core as baseline


def difference(a, b):
    aa, bb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(aa) == len(bb)
    return max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64))))
               for x, y in zip(aa, bb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.RandomState(61627)
    # Independently solve the root by bisection, including nonuniform scales.
    mathematical = []
    for tau in (.1, .5, .9, .99):
        x = rng.normal(size=(9, 300))*rng.lognormal(size=(1, 300))
        v = rng.normal(size=300)*5
        low, high = x.min(0), x.max(0)
        for _ in range(100):
            mid = (low+high)/2
            delta = x-mid
            score = np.mean(np.where(delta >= 0, tau, 1-tau)*delta, axis=0)
            low, high = np.where(score > 0, mid, low), np.where(score > 0, high, mid)
        root = (low+high)/2
        corrected = .5*v+.5*root
        def objective(value):
            delta = x-value
            weights = np.where(delta >= 0, tau, 1-tau)
            return np.mean(weights*delta**2, axis=0), np.mean(weights*delta, axis=0)
        old_loss, old_score = objective(v)
        new_loss, new_score = objective(corrected)
        assert np.max(new_loss-old_loss) < 1e-10
        assert np.max(np.abs(new_score)-np.abs(old_score)) < 1e-10
        np.testing.assert_allclose(corrected-root, .5*(v-root), atol=1e-12)
        mathematical.append(dict(tau=tau, states=300, maximum_loss_increase=float(np.max(new_loss-old_loss)),
                                 maximum_score_magnitude_increase=float(np.max(np.abs(new_score)-np.abs(old_score)))))

    cfg = half_core.config_for('cube-single-play-singletask-v0', .999, .9, 8)
    cfg.update(num_samples=9, policy_type='ifql', agent_name='ifql')
    initial = half_core.create(31, jnp.zeros((1, 3)), jnp.zeros((1, 2)), cfg, 'ifql')
    batches = dict(observations=jnp.asarray(rng.normal(size=(4, 12, 3)), jnp.float32),
                   next_observations=jnp.asarray(rng.normal(size=(4, 12, 3)), jnp.float32),
                   actions=jnp.asarray(rng.uniform(-1, 1, (4, 12, 2)), jnp.float32),
                   rewards=-jnp.ones((4, 12)), masks=jnp.ones((4, 12)))
    bounds = (-1000., 0.)
    update = half_core.build_update('bounded_v2_half_empirical_backup', bounds)
    scan = half_core.make_scan(update)
    full, logs = scan(initial, batches)
    halfway, _ = scan(initial, jax.tree_util.tree_map(lambda value: value[:2], batches))
    restored = flax.serialization.from_state_dict(initial, copy.deepcopy(flax.serialization.to_state_dict(halfway)))
    resumed, _ = scan(restored, jax.tree_util.tree_map(lambda value: value[2:], batches))
    assert difference(full, resumed) == 0
    zero, _ = half_core.make_scan(half_core._build_update(bounds, 0.))(initial, batches)
    base, _ = baseline.make_scan(baseline.build_update('bounded_v2', bounds, 'ifql'))(initial, batches)
    one, _ = half_core.make_scan(half_core._build_update(bounds, 1.))(initial, batches)
    empirical, _ = full_core.make_scan(full_core.build_update('bounded_v2_empirical_backup', bounds))(initial, batches)
    endpoints = dict(zero_vs_V2=difference(zero, base), one_vs_full_empirical=difference(one, empirical))
    assert max(endpoints.values()) < 1e-6, endpoints
    bc_difference = max(difference(full.agent.network.params['modules_actor_flow'], state.agent.network.params['modules_actor_flow'])
                        for state in (base, empirical))
    assert bc_difference < 1e-6
    q_difference = difference(full.agent.network.params['modules_critic'], empirical.agent.network.params['modules_critic'])
    assert q_difference > 1e-8
    np.testing.assert_array_equal(full.agent.rng, base.agent.rng)
    np.testing.assert_array_equal(full.agent.rng, empirical.agent.rng)
    assert int(full.q_updates) == 4 and int(full.v_updates) == 8
    first = jax.tree_util.tree_map(lambda value: value[0], batches)
    actor_gradient = jax.grad(lambda params: initial.agent.actor_loss(first, params, jax.random.PRNGKey(7))[0])(
        initial.agent.network.params)
    actor_QV_gradient = {name:max(float(jnp.max(jnp.abs(value))) for value in jax.tree_util.tree_leaves(actor_gradient[name]))
                         for name in ('modules_critic', 'modules_target_critic', 'modules_value')}
    assert max(actor_QV_gradient.values()) == 0
    for name in ('proposal_expectile_loss_increase_fraction', 'proposal_score_increase_fraction'):
        assert np.max(np.asarray(logs['backup/'+name])) == 0
    np.testing.assert_allclose(logs['backup/remaining_distance_to_root_rms'],
                               .5*logs['backup/empirical_root_minus_learned_V_rms'], atol=1e-6)
    # V now affects the target, unlike full empirical replacement.
    altered_params = copy.deepcopy(initial.agent.network.params)
    altered_params['modules_value'] = jax.tree_util.tree_map(lambda value:value+.25, altered_params['modules_value'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=altered_params)))
    _, original_log = jax.jit(update)(initial, first)
    _, altered_log = jax.jit(update)(altered, first)
    target_loss_delta = abs(float(original_log['critic/critic_loss']-altered_log['critic/critic_loss']))
    assert target_loss_delta > 1e-6
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(full))
    result = dict(time=time.time(), independent_root_geometry=mathematical,
                  serialized_two_plus_two_full_state_max_abs=0., endpoint_full_state_max_abs=endpoints,
                  BC_four_update_vs_both_controls_max_abs=bc_difference,
                  critic_change_vs_full_empirical_max_abs=q_difference, actor_QV_gradient_max_abs=actor_QV_gradient,
                  correction_halves_distance_to_finite_root=True, finite_proposal_loss_and_score_nonincrease=True,
                  value_perturbation_critic_loss_delta=target_loss_delta, q_updates=4, v_updates=8,
                  scope='CPU implementation and finite-proposal algebra only; no oracle residual, neural convergence or return claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
