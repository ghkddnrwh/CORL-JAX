"""Boundary, fallback, residual-isolation and resume checks for observed anchors."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import anchored_empirical_backup_core_v1 as anchored
from utils.flow_iql_research import empirical_backup_core_v1 as reference
from utils.flow_iql_research.next_action_anchors_v1 import augment
from utils.flow_iql_research.selection_v2 import empirical_expectile


def difference(a, b):
    pairs = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64)))) for x,y in pairs)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observations = np.array([[0], [1], [3], [4]], np.float32)
    data = dict(observations=observations, next_observations=np.array([[1], [2], [4], [5]], np.float32),
                actions=np.array([[.1], [.2], [.3], [.4]], np.float32))
    augmented = augment(dict(terminals=np.array([0, 1, 0, 1])), data)
    assert augmented['next_action_valid'].tolist() == [1, 0, 1, 0]
    np.testing.assert_array_equal(augmented['next_actions'], np.array([[.2], [0], [.4], [0]], np.float32))
    wrong = dict(data, next_observations=np.zeros((4, 1), np.float32))
    try:
        augment(dict(terminals=np.array([0, 1, 0, 1])), wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError('Misaligned next states were accepted')
    q = jnp.full((9, 1), -2000.).at[0].set(-100.)
    expected = (.9*(-100.) + 8*.1*(-2000.)) / (.9+8*.1)
    assert abs(float(empirical_expectile(q, .9)[0])-expected) < .001
    cfg = anchored.config_for('cube-single-play-singletask-v0', .999, .9, 8)
    cfg.update(num_samples=9, policy_type='ifql', agent_name='ifql')
    obs = jnp.zeros((1, 3)); action = jnp.zeros((1, 2))
    initial = anchored.create(31, obs, action, cfg, 'ifql')
    rng = np.random.RandomState(987)
    batches = dict(observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   next_observations=jnp.asarray(rng.normal(size=(4, 6, 3)), jnp.float32),
                   actions=jnp.asarray(rng.uniform(-1, 1, (4, 6, 2)), jnp.float32),
                   rewards=-jnp.ones((4, 6)), masks=jnp.ones((4, 6)),
                   next_actions=jnp.asarray(rng.uniform(-1, 1, (4, 6, 2)), jnp.float32),
                   next_action_valid=jnp.ones((4, 6)))
    update = anchored.build_update('bounded_v2_anchored_empirical_backup', (-1000., 0.), 'ifql')
    scan = anchored.make_scan(update)
    full, logs = scan(initial, batches)
    half, _ = scan(initial, jax.tree_util.tree_map(lambda x:x[:2], batches))
    restored = flax.serialization.from_state_dict(initial, copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed, _ = scan(restored, jax.tree_util.tree_map(lambda x:x[2:], batches))
    resume_difference = difference(full, resumed); assert resume_difference == 0
    first = jax.tree_util.tree_map(lambda x:x[0], batches)
    invalid = dict(first, next_action_valid=jnp.zeros((6,)))
    fallback, _ = jax.jit(update)(initial, invalid)
    ordinary, _ = jax.jit(reference.build_update('bounded_v2_empirical_backup', (-1000., 0.), 'ifql'))(initial, invalid)
    fallback_difference = difference(fallback, ordinary); assert fallback_difference < 1e-6
    altered_params = initial.agent.network.params.copy()
    altered_params['modules_value'] = jax.tree_util.tree_map(lambda x:x+2., altered_params['modules_value'])
    altered = initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=altered_params)))
    standard_next, _ = jax.jit(update)(initial, first)
    perturbed_next, _ = jax.jit(update)(altered, first)
    isolated = {name:difference(standard_next.agent.network.params[name], perturbed_next.agent.network.params[name])
                for name in ('modules_critic', 'modules_target_critic', 'modules_actor_flow')}
    assert max(isolated.values()) == 0
    assert int(full.q_updates) == 4 and int(full.v_updates) == 8
    assert np.isfinite(np.asarray(logs['backup/finite_proposal_score_abs_mean'])).all()
    result = dict(time=time.time(), trajectory_boundaries_preserved=True, misalignment_rejected=True,
                  observed_anchor_scalar_root_matches=True, fallback_full_state_max_abs=fallback_difference,
                  four_vs_serialized_two_plus_two_full_state_max_abs=resume_difference,
                  auxiliary_V_perturbation_max_abs=isolated,
                  root_residual_max=float(np.max(np.asarray(logs['backup/finite_proposal_score_abs_mean']))),
                  q_updates=4, v_updates=8, scope='Numerical implementation checks; no performance or neural convergence claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
