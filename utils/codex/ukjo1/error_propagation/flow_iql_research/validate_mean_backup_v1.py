"""Check mean-backup actuation, unchanged BC, V isolation, and resume clocks."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import mean_empirical_backup_core_v1 as mean_core
from utils.flow_iql_research import empirical_backup_core_v1 as min_core


def difference(a, b):
    first, second = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(first) == len(second)
    return max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64)))) for x,y in zip(first, second))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = mean_core.config_for('cube-single-play-singletask-v0', .999, .9, 8)
    cfg.update(num_samples=9, policy_type='ifql', agent_name='ifql')
    initial = mean_core.create(31, jnp.zeros((1, 3)), jnp.zeros((1, 2)), cfg, 'ifql')
    rng = np.random.RandomState(987)
    batches = dict(observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   next_observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   actions=jnp.asarray(rng.uniform(-1, 1, (4, 6, 2)), jnp.float32),
                   rewards=-jnp.ones((4, 6)), masks=jnp.ones((4, 6)))
    update = mean_core.build_update('bounded_v2_mean_empirical_backup', (-1000., 0.), 'ifql')
    scan = mean_core.make_scan(update)
    full, logs = scan(initial, batches)
    half, _ = scan(initial, jax.tree_util.tree_map(lambda x:x[:2], batches))
    restored = flax.serialization.from_state_dict(initial, copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed, _ = scan(restored, jax.tree_util.tree_map(lambda x:x[2:], batches))
    resume_difference = difference(full, resumed); assert resume_difference == 0
    min_update = min_core.build_update('bounded_v2_empirical_backup', (-1000., 0.), 'ifql')
    min_full, _ = min_core.make_scan(min_update)(initial, batches)
    bc_difference = difference(full.agent.network.params['modules_actor_flow'], min_full.agent.network.params['modules_actor_flow'])
    assert bc_difference < 1e-6
    q_difference = difference(full.agent.network.params['modules_critic'], min_full.agent.network.params['modules_critic'])
    assert q_difference > 1e-8, 'Aggregation change did not affect Q'
    first = jax.tree_util.tree_map(lambda x:x[0], batches)
    equal_params = initial.agent.network.params.copy()
    def duplicate(array):
        assert array.shape[0] == 2
        return jnp.repeat(array[:1], 2, axis=0)
    equal_params['modules_target_critic'] = jax.tree_util.tree_map(duplicate, equal_params['modules_target_critic'])
    equal = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=equal_params)))
    equal_mean, _ = jax.jit(update)(equal, first)
    equal_min, _ = jax.jit(min_update)(equal, first)
    equal_difference = difference(equal_mean, equal_min); assert equal_difference < 1e-6
    altered_params = initial.agent.network.params.copy()
    altered_params['modules_value'] = jax.tree_util.tree_map(lambda x:x+2., altered_params['modules_value'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=altered_params)))
    standard, _ = jax.jit(update)(initial, first)
    perturbed, _ = jax.jit(update)(altered, first)
    isolated = {name:difference(standard.agent.network.params[name], perturbed.agent.network.params[name])
                for name in ('modules_critic', 'modules_target_critic', 'modules_actor_flow')}
    assert max(isolated.values()) == 0
    assert int(full.q_updates) == 4 and int(full.v_updates) == 8
    assert np.min(np.asarray(logs['backup/mean_minus_min_empirical_root_mean'])) > -1e-6
    result = dict(time=time.time(), four_vs_serialized_two_plus_two_full_state_max_abs=resume_difference,
                  identical_target_heads_mean_vs_min_full_state_max_abs=equal_difference,
                  BC_four_updates_mean_vs_min_max_abs=bc_difference, Q_change_mean_vs_min_max_abs=q_difference,
                  auxiliary_V_perturbation_max_abs=isolated,
                  finite_root_residual_max=float(np.max(np.asarray(logs['backup/finite_proposal_score_abs_mean']))),
                  mean_minus_min_root_positive=True, q_updates=4, v_updates=8,
                  scope='CPU implementation checks only, no policy performance or neural convergence claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
