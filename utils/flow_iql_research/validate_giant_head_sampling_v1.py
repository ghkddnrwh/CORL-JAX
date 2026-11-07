"""Check head interpolation, actual proposal counts and frozen min9 control."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import giant_head_sampling_core_v1 as core
from utils.flow_iql_research import target_ablation_core_v1 as old
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


def identical(a, b):
    a, b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    heads = jnp.asarray([[[1.], [-6.]], [[3.], [-2.]]])
    np.testing.assert_array_equal(np.asarray(core.aggregate_heads(heads, 1.)), [[1.], [-6.]])
    np.testing.assert_array_equal(np.asarray(core.aggregate_heads(heads, .5)), [[1.5], [-5.]])
    rng = np.random.RandomState(2731)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.ones(8))
    cases = []
    for recipe, (weight, n) in core.SETTINGS.items():
        cfg = core.config_for(core.GIANT, recipe, width=16)
        assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['tau'] == .005
        assert cfg['num_samples'] == 9 and cfg['backup_samples'] == n
        assert cfg['onestep_specification']['actor_label_samples'] == 9
        bounds = core.return_bounds(core.GIANT, .999, -1., 0.)
        state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
        actual, _ = jax.jit(core.build_update(recipe, bounds))(state, batch)
        expected_base, _ = jax.jit(core.base_update(recipe, bounds))(state.base, batch)
        assert identical(actual.base, expected_base)
        assert int(actual.base.q_updates) == int(actual.actor_updates) == 1 and int(actual.base.v_updates) == 0
        if recipe == 'p100_n9':
            original, _ = jax.jit(old.build_update('ant_min_clip', bounds))(state, batch)
            assert identical(actual, original)
        values = core.make_probe(recipe, bounds)(actual, batch)
        assert bool(values['finite'])
        assert tuple(np.asarray(values['diagnostic_proposal_shape'])) == (n, 8)
        agent = actual.base.agent
        obs, noise = core.lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), n)
        actions = core.lean.flow(agent, obs, noise, 10)
        h = agent.network.select('target_critic')(obs, actions=actions)
        q = core.aggregate_heads(h, weight)
        root = selection_v2.empirical_expectile(q, .9)
        expected_raw = batch['rewards']+.999*root
        np.testing.assert_allclose(np.asarray(values['diagnostic_raw_target']), np.asarray(expected_raw), atol=1e-6, rtol=1e-6)
        assert np.min(np.asarray(values['diagnostic_root_above_min'])) >= -1e-5
        assert np.min(np.asarray(values['diagnostic_root_below_mean'])) >= -1e-5
        _, _, _, _, teacher_actions, _ = core.actor_core.teacher_batch(state, batch, jax.random.PRNGKey(77), cfg['onestep_specification'])
        assert teacher_actions.shape[0] == 9
        cases.append(dict(recipe=recipe, pessimism_weight=weight, actual_backup_proposals=n,
                          actor_label_proposals=9, deployment_proposals=9,
                          no_actor_feedback_to_base=True, correct_root_target_and_probe=True))
    result = dict(status='PASS', cases=cases, min9_control_full_state_bitwise_equal=True,
                  scope='Small real optimizer/proposal/root checks, not benchmark performance.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
