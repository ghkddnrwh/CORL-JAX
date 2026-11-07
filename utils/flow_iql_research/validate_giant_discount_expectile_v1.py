"""Verify dynamic discount/expectile and derived return bounds in Giant probes."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import giant_discount_expectile_core_v1 as core
from utils.flow_iql_research import target_ablation_core_v1 as old
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    rng = np.random.RandomState(1127)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.ones(8))
    cases = []
    for recipe, (gamma, tau) in core.SETTINGS.items():
        cfg = core.config_for(core.GIANT, recipe, width=16)
        bounds = core.return_bounds(core.GIANT, cfg['discount'], -1., 0.)
        assert np.allclose(bounds, (-1/(1-gamma), 0.))
        assert cfg['discount'] == gamma and cfg['expectile'] == tau and cfg['tau'] == .005
        state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
        updated, _ = jax.jit(core.build_update(recipe, bounds))(state, batch)
        expected, _ = jax.jit(old.build_update('ant_min_clip', bounds))(state, batch)
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                   zip(jax.tree_util.tree_leaves(updated), jax.tree_util.tree_leaves(expected)))
        values = core.make_probe(recipe, bounds)(updated, batch)
        agent = updated.base.agent
        obs, noise = core.lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), 9)
        actions = core.lean.flow(agent, obs, noise, 10)
        proposal_q = agent.network.select('target_critic')(obs, actions=actions).min(0)
        root = selection_v2.empirical_expectile(proposal_q, tau)
        raw = batch['rewards']+gamma*root
        np.testing.assert_allclose(np.asarray(values['diagnostic_raw_target']), np.asarray(raw), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(values['base']['y']), np.clip(np.asarray(raw), *bounds), rtol=1e-6, atol=1e-6)
        assert bool(values['finite']) and np.min(np.asarray(values['diagnostic_expectile95_minus90'])) >= -1e-6
        assert int(updated.base.v_updates) == 0 and int(updated.base.q_updates) == int(updated.actor_updates) == 1
        cases.append(dict(recipe=recipe, gamma=gamma, expectile=tau, return_bounds=list(bounds),
                          frozen_base_update_bitwise_equal=True, correct_probe_gamma_and_tau=True))
    result = dict(status='PASS', cases=cases, scope='Small real optimizer and probe verification, not performance.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
