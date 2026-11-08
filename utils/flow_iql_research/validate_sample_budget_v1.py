"""Validate real optimizer updates, frozen controls, counts, and actor isolation."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import sample_budget_core_v1 as core
from utils.flow_iql_research import target_ablation_core_v1 as reference
from utils.flow_iql_research import giant_head_sampling_core_v1 as min27_reference
from utils.flow_iql_research import mean_proposal_ablation_core_v1 as mean27_reference
from utils.flow_iql_research import selection_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def identical(a, b):
    aa, bb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(aa) == len(bb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(aa, bb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(d.platform == 'cpu' for d in jax.devices())
    rng = np.random.RandomState(7368)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=jnp.asarray([0, -1, -1, 0, -1, -1, -1, -1], jnp.float32),
                 masks=jnp.asarray([0, 1, 1, 0, 1, 1, 1, 1], jnp.float32))
    for env in core.ENVIRONMENTS:
        for recipe, n in core.SETTINGS.items():
            cfg = core.config_for(env, recipe)
            assert cfg['backup_Q_aggregation'] == ('min' if env.startswith('antmaze-') else 'mean')
            assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['target_clipping']
            assert cfg['backup_samples'] == n and cfg['num_samples'] == 9
            assert cfg['onestep_specification']['actor_label_samples'] == 9
    cases = []
    for env in (reference.GIANT, 'cube-double-play-singletask-v0'):
        for recipe, n in core.SETTINGS.items():
            cfg = core.config_for(env, recipe, width=16)
            bounds = core.return_bounds(env, .999, -1., 0.)
            state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
            update = jax.jit(core.build_update(recipe, bounds))
            actual, _ = update(state, batch)
            if n in (8, 36):
                alone, _ = jax.jit(core.base_update(recipe, bounds))(state.base, batch)
                assert identical(actual.base, alone)
            if n == 9:
                old, _ = jax.jit(reference.build_update(core.reference_recipe(env), bounds))(state, batch)
                assert identical(actual, old)
            if n == 27:
                old_module = min27_reference if env == reference.GIANT else mean27_reference
                old_recipe = 'p100_n27' if env == reference.GIANT else 'mean_n27'
                old, _ = jax.jit(old_module.build_update(old_recipe, bounds))(state, batch)
                assert identical(actual, old)
            values = core.make_probe(recipe, bounds)(actual, batch)
            assert bool(values['finite']) and tuple(np.asarray(values['diagnostic_proposal_shape'])) == (n, 8)
            agent = actual.base.agent
            obs, noise = core.lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), n)
            actions = core.lean.flow(agent, obs, noise, 10)
            heads = agent.network.select('target_critic')(obs, actions=actions)
            q = heads.min(0) if env == reference.GIANT else heads.mean(0)
            expected = batch['rewards'] + .999*batch['masks']*selection_v2.empirical_expectile(q, .9)
            np.testing.assert_allclose(np.asarray(values['diagnostic_raw_target']), np.asarray(expected), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(values['diagnostic_final_target']), np.clip(np.asarray(expected), *bounds), atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(np.asarray(values['diagnostic_final_target'])[[0, 3]], [0., 0.])
            _, _, _, _, labels, _ = core.actor_core.teacher_batch(state, batch, jax.random.PRNGKey(88), cfg['onestep_specification'])
            assert labels.shape[0] == 9
            assert int(actual.base.v_updates) == 0 and int(actual.base.q_updates) == int(actual.actor_updates) == 1
            if n == 36:
                uninterrupted, _ = update(actual, batch)
                restored = flax.serialization.from_state_dict(state, flax.serialization.to_state_dict(actual))
                resumed, _ = update(restored, batch)
                assert identical(uninterrupted, resumed)
            cases.append(dict(environment=env, bootstrap_samples=n, actor_label_samples=9,
                              deployment_samples=9, frozen_control_bitwise_equal=n in (9, 27),
                              causal_actor_isolation_checked=n in (8, 36), serialization_checked=n == 36))
            print('PASS', env, n, flush=True)
    result = dict(status='PASS', configurations_checked=70, real_update_cases=cases,
                  scope='Numerical and optimizer validation, not performance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
