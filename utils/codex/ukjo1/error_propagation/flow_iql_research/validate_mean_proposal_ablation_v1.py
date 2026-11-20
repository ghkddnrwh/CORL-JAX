"""Pure mean N9 reference equivalence and actual N27 bootstrap/label checks."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import mean_proposal_ablation_core_v1 as core
from utils.flow_iql_research import target_ablation_core_v1 as reference
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
    rng = np.random.RandomState(2739)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.ones(8))
    cases = []
    for env in core.ENVIRONMENTS:
        for recipe, n in core.SETTINGS.items():
            cfg = core.config_for(env, recipe, width=16)
            assert cfg['backup_Q_aggregation'] == 'mean' and cfg['backup_samples'] == n and cfg['num_samples'] == 9
            bounds = core.return_bounds(env, .999, -1., 0.)
            state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
            actual, _ = jax.jit(core.build_update(recipe, bounds))(state, batch)
            base, _ = jax.jit(core.base_update(recipe, bounds))(state.base, batch)
            assert identical(actual.base, base)
            if n == 9:
                old, _ = jax.jit(reference.build_update('mean_clip', bounds))(state, batch)
                assert identical(actual, old)
            values = core.make_probe(recipe, bounds)(actual, batch)
            assert bool(values['finite']) and tuple(np.asarray(values['diagnostic_proposal_shape'])) == (n, 8)
            agent = actual.base.agent
            obs, noise = core.lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), n)
            actions = core.lean.flow(agent, obs, noise, 10)
            q = agent.network.select('target_critic')(obs, actions=actions).mean(0)
            expected = batch['rewards']+.999*selection_v2.empirical_expectile(q, .9)
            np.testing.assert_allclose(np.asarray(values['diagnostic_raw_target']), np.asarray(expected), atol=1e-6, rtol=1e-6)
            _, _, _, _, labels, _ = core.actor_core.teacher_batch(state, batch, jax.random.PRNGKey(88), cfg['onestep_specification'])
            assert labels.shape[0] == 9
            assert int(actual.base.v_updates) == 0 and int(actual.base.q_updates) == int(actual.actor_updates) == 1
            cases.append(dict(environment=env, recipe=recipe, mean_target_only=True,
                              actual_bootstrap_samples=n, actor_label_samples=9, deployment_samples=9,
                              independent_base_update_bitwise_equal=True,
                              frozen_mean9_full_state_equal=(n == 9)))
    result = dict(status='PASS', cases=cases, scope='Small real optimizer/proposal checks; not performance.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
