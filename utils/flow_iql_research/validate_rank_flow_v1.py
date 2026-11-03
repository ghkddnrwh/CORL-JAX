"""Validate policy-distribution distillation without conflating Q/V learning."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import rank_flow_core_v1 as rank
from utils.flow_iql_research import policy_core as reference


def difference(a, b):
    first, second = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(first) == len(second)
    return max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64)))) for x,y in zip(first, second))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = rank.config_for('cube-single-play-singletask-v0', .999, .9, 8)
    initial = rank.create(31, jnp.zeros((1, 3)), jnp.zeros((1, 2)), cfg)
    base_cfg = dict(cfg, policy_type='ifql', agent_name='ifql', num_samples=9)
    baseline = reference.create(31, jnp.zeros((1, 3)), jnp.zeros((1, 2)), base_cfg, 'ifql')
    common = ('modules_critic', 'modules_target_critic', 'modules_value', 'modules_actor_flow')
    assert max(difference(initial.agent.network.params[name], baseline.agent.network.params[name]) for name in common) == 0
    rng = np.random.RandomState(987)
    actions = jnp.asarray(rng.normal(size=(9, 6, 2)), jnp.float32)
    scores = jnp.asarray(np.stack([rng.permutation(9) for _ in range(6)], axis=1), jnp.float32)
    selected = np.asarray(rank.select_best(actions, scores))
    expected = np.asarray(actions)[np.asarray(scores).argmax(0), np.arange(6)]
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(rank.select_best(actions, 3*scores+jnp.arange(6)*100), expected)
    batches = dict(observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   next_observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   actions=jnp.asarray(rng.uniform(-1, 1, (4, 6, 2)), jnp.float32),
                   rewards=-jnp.ones((4, 6)), masks=jnp.ones((4, 6)))
    first = jax.tree_util.tree_map(lambda x:x[0], batches)
    gradients = jax.grad(lambda params:initial.agent.actor_loss(first, params, jax.random.PRNGKey(7))[0])(initial.agent.network.params)
    actor_QV_gradient = {name:max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(gradients[name]))
                         for name in ('modules_critic', 'modules_target_critic', 'modules_value')}
    assert max(actor_QV_gradient.values()) == 0
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(gradients['modules_actor_policy_flow'])) > 0
    update = rank.build_update('bounded_v2_rank_flow', (-1000., 0.))
    scan = rank.make_scan(update)
    full, logs = scan(initial, batches)
    old, _ = reference.make_scan(reference.build_update('bounded_v2', (-1000., 0.), 'ifql'))(baseline, batches)
    common_delta = {name:difference(full.agent.network.params[name], old.agent.network.params[name]) for name in common}
    assert max(common_delta.values()) < 1e-6, common_delta
    half, _ = scan(initial, jax.tree_util.tree_map(lambda x:x[:2], batches))
    restored = flax.serialization.from_state_dict(initial, copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed, _ = scan(restored, jax.tree_util.tree_map(lambda x:x[2:], batches))
    resume_delta = difference(full, resumed); assert resume_delta == 0
    altered_params = initial.agent.network.params.copy()
    altered_params['modules_actor_policy_flow'] = jax.tree_util.tree_map(lambda x:x+.1, altered_params['modules_actor_policy_flow'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=altered_params)))
    ordinary_next, _ = jax.jit(update)(initial, first)
    altered_next, _ = jax.jit(update)(altered, first)
    isolation = {name:difference(ordinary_next.agent.network.params[name], altered_next.agent.network.params[name]) for name in common}
    assert max(isolation.values()) == 0
    deployed = initial.agent.sample_actions(first['observations'], jax.random.PRNGKey(9), 0.)
    q_altered = initial.agent.network.params.copy()
    for name in ('modules_critic', 'modules_target_critic', 'modules_value'):
        q_altered[name] = jax.tree_util.tree_map(lambda x:x+10., q_altered[name])
    changed = initial.agent.replace(network=initial.agent.network.replace(params=q_altered))
    np.testing.assert_array_equal(deployed, changed.sample_actions(first['observations'], jax.random.PRNGKey(9), 0.))
    assert deployed.shape == (6, 2) and np.max(np.abs(np.asarray(deployed))) <= 1
    assert initial.agent.sample_actions(first['observations'][0], jax.random.PRNGKey(9), 0.).shape == (2,)
    assert int(full.q_updates) == 4 and int(full.v_updates) == 8
    assert np.min(np.asarray(logs['actor/selected_teacher_Q_gain_over_mean'])) >= 0
    result = dict(time=time.time(), common_initial_parameters_exact=True,
        actor_loss_QV_parameter_gradient_max_abs=actor_QV_gradient,
        four_update_common_parameters_vs_IFQL_V2_max_abs=common_delta,
        four_vs_serialized_two_plus_two_full_state_max_abs=resume_delta,
        policy_parameter_perturbation_common_update_max_abs=isolation,
        selector_matches_manual=True, selector_positive_affine_invariance=True,
        deployment_independent_of_QV=True, single_and_batch_actions_bounded=True,
        q_updates=4, v_updates=8,
        scope='Short CPU implementation checks; no long-GPU identity, policy improvement or distribution-fit guarantee.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
