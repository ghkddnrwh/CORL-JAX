"""Check the real-data rank estimator, gradient isolation and resumability."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import data_rank_flow_core_v1 as core
from utils.flow_iql_research import rank_flow_core_v1 as rank
from utils.flow_iql_research import policy_core as official


def difference(first, second):
    leaves = list(zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)))
    assert len(jax.tree_util.tree_leaves(first)) == len(jax.tree_util.tree_leaves(second))
    return max(float(np.max(np.abs(np.asarray(a, np.float64)-np.asarray(b, np.float64)))) for a,b in leaves)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.RandomState(91321)
    # All possible augmented ranks among the data action and nine comparators.
    qdata = jnp.arange(10, dtype=jnp.float32)-.5
    comparators = jnp.broadcast_to(jnp.arange(9, dtype=jnp.float32)[:,None], (9,10))
    weights, counts = core.data_rank_weights(qdata, comparators, jnp.zeros((10,)), jnp.zeros((9,10)))
    np.testing.assert_array_equal(counts, np.arange(10))
    np.testing.assert_array_equal(weights, [0.]*8+[1.,9.])
    changed, _ = core.data_rank_weights(3*qdata+200, 3*comparators+200, jnp.zeros((10,)), jnp.zeros((9,10)))
    np.testing.assert_array_equal(changed, weights)
    assert np.mean(np.asarray(weights)) == 1.
    assert np.mean(np.asarray(weights, np.float64)**2) == 8.2
    # Monte Carlo integration is a mathematical estimator check, not OGBench.
    n = 65536
    u = rng.uniform(size=(10,n)).astype(np.float32)
    weights, _ = core.data_rank_weights(jnp.asarray(u[0]), jnp.asarray(u[1:]), jnp.zeros((n,)), jnp.zeros((9,n)))
    weights = np.asarray(weights)
    distribution_check = []
    for threshold in (.25, .5, .75, 1.):
        values = weights*(u[0] <= threshold)
        estimate, expected = float(values.mean()), threshold**9
        # A rare tail can have no observed positive samples; empirical SE=0
        # would then be invalid. Integrate the exact second moment instead:
        # E[w^2 I(U<=t)] = t^9 + 7.2*t^10.
        se = float(np.sqrt((threshold**9+7.2*threshold**10-expected**2)/n))
        assert abs(estimate-expected) <= 6*se+1e-6, (estimate,expected,se)
        distribution_check.append(dict(threshold=threshold, weighted_CDF_mass=estimate,
            ideal_best9_CDF=expected, analytic_Monte_Carlo_SE=se))
    # With flat Q, independent tie breaking cannot select by an unrelated action
    # coordinate. The weighted mass on an independent half-space stays one half.
    tied, _ = core.data_rank_weights(jnp.zeros((n,)), jnp.zeros((9,n)), jnp.asarray(u[0]), jnp.asarray(u[1:]))
    independent_action = rng.uniform(size=n)
    values = np.asarray(tied)*(independent_action <= .5)
    se = float(values.astype(np.float64).std(ddof=1)/np.sqrt(n))
    assert abs(float(values.mean())-.5) < 6*se

    cfg = core.config_for('cube-single-play-singletask-v0', .999, .9, 8)
    initial = core.create(31, jnp.zeros((1,3)), jnp.zeros((1,2)), cfg)
    reference = rank.create(31, jnp.zeros((1,3)), jnp.zeros((1,2)), cfg)
    assert difference(initial, reference) == 0
    common = ('modules_critic','modules_target_critic','modules_value','modules_actor_flow')
    base_cfg = dict(cfg, policy_type='ifql',agent_name='ifql',num_samples=9)
    baseline = official.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),base_cfg,'ifql')
    batches = dict(observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,(4,12,2)),jnp.float32),
        rewards=-jnp.ones((4,12)), masks=jnp.ones((4,12)))
    first = jax.tree_util.tree_map(lambda value:value[0],batches)
    key = jax.random.PRNGKey(7)
    gradients = jax.grad(lambda params:initial.agent.actor_loss(first,params,key)[0])(initial.agent.network.params)
    actor_QV_gradient = {name:max(float(jnp.max(jnp.abs(value)))for value in jax.tree_util.tree_leaves(gradients[name]))
        for name in ('modules_critic','modules_target_critic','modules_value')}
    assert max(actor_QV_gradient.values()) == 0
    assert max(float(jnp.max(jnp.abs(value)))for value in jax.tree_util.tree_leaves(gradients['modules_actor_policy_flow'])) > 0
    loss, _ = initial.agent.actor_loss(first,initial.agent.network.params,key)
    params = copy.deepcopy(initial.agent.network.params)
    params['modules_value'] = jax.tree_util.tree_map(lambda value:value+10,params['modules_value'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=params)))
    other_loss, _ = altered.agent.actor_loss(first,altered.agent.network.params,key)
    assert float(loss) == float(other_loss)

    update = core.build_update('bounded_v2_data_rank_flow',(-1000.,0.))
    scan = core.make_scan(update)
    full, logs = scan(initial,batches)
    base, _ = official.make_scan(official.build_update('bounded_v2',(-1000.,0.),'ifql'))(baseline,batches)
    common_deltas = {name:difference(full.agent.network.params[name],base.agent.network.params[name])for name in common}
    assert max(common_deltas.values()) <= 1e-6, common_deltas
    half, _ = scan(initial,jax.tree_util.tree_map(lambda value:value[:2],batches))
    resumed_template = flax.serialization.from_state_dict(initial,copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed, _ = scan(resumed_template,jax.tree_util.tree_map(lambda value:value[2:],batches))
    assert difference(full,resumed) == 0
    # Existing inference/frozen-Q diagnostic templates use RankFlowAgent. It has
    # identical arrays and deployment methods, but a different actor loss.
    reloaded_reference = flax.serialization.from_state_dict(reference,flax.serialization.to_state_dict(full))
    assert difference(full,reloaded_reference) == 0
    actions = full.agent.sample_actions(first['observations'],key,0.)
    np.testing.assert_array_equal(actions,reloaded_reference.agent.sample_actions(first['observations'],key,0.))
    np.testing.assert_array_equal(full.agent.selected_teacher_targets(first['observations'],key)[0],
                                  reloaded_reference.agent.selected_teacher_targets(first['observations'],key)[0])
    params = copy.deepcopy(initial.agent.network.params)
    params['modules_actor_policy_flow'] = jax.tree_util.tree_map(lambda value:value+.1,params['modules_actor_policy_flow'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=params)))
    plain, _ = jax.jit(update)(initial,first)
    changed, _ = jax.jit(update)(altered,first)
    isolation = {name:difference(plain.agent.network.params[name],changed.agent.network.params[name])for name in common}
    assert max(isolation.values()) == 0
    assert int(full.q_updates) == 4 and int(full.v_updates) == 8
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(full))
    assert np.min(np.asarray(logs['actor/real_data_rank_weight_mean'])) >= 0
    result = dict(time=time.time(), all_ten_augmented_ranks_correct=True,
        ideal_weight_mass=1., ideal_weight_second_moment=8.2, ideal_large_batch_ESS_fraction=1/8.2,
        positive_affine_rank_invariance=True, Monte_Carlo_distribution_checks=distribution_check,
        flat_Q_tie_independence_checked=True, common_initial_arrays_exact=True,
        actor_QV_parameter_gradient_max_abs=actor_QV_gradient, actor_loss_V_perturbation_delta=0.,
        four_update_common_parameters_vs_IFQL_V2_max_abs=common_deltas,
        serialized_two_plus_two_full_state_max_abs=0., policy_perturbation_common_update_max_abs=isolation,
        existing_diagnostic_template_state_and_deployment_exact=True,q_updates=4,v_updates=8,
        scope='ShortCPUimplementationandmathematicalestimatorchecks;notGPUtrajectoryidentity,behavior-modelaccuracyorbenchmarkperformance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
