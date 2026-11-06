"""Numerical endpoint and real optimizer-update validation; not performance."""
import argparse
import inspect
import json
from pathlib import Path

from utils.flow_iql_research import max_mean_lean_core_v1 as core
from utils.flow_iql_research import efficient_mean_core_v1 as lean
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    q = jnp.asarray([[1., -3., 4.], [2., -2., 4.], [0., -1., 4.]])
    maximum = np.asarray(q.max(0))
    assert np.array_equal(maximum, np.array([2., -1., 4.]))
    gaps = []
    for tau in (.9, .99, .9999):
        root = np.asarray(selection_v2.empirical_expectile(q, tau))
        assert np.isfinite(root).all() and np.all(root <= maximum+1e-6)
        gaps.append(float(np.max(maximum-root)))
    assert gaps[-1] < gaps[0]
    # Mean of per-head maxima is different from the intended max of means.
    heads = jnp.asarray([[[10.], [0.]], [[0.], [10.]]])
    assert float(heads.mean(0).max(0)[0]) == 5.
    assert float(heads.max(1).mean(0)[0]) == 10.

    environment = core.ENVIRONMENTS[0]
    cfg = core.config_for(environment, 'max_q', width=16)
    rng = np.random.RandomState(93)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0., 1.]*4))
    state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, 'max_q')
    bounds = (-1000., 0.)
    # Independently adapt the preserved lean update at its one root expression.
    # This compares all real optimizer state and parameter arrays, not just loss.
    source = inspect.getsource(lean.build_update)
    before = "root=selection_v2.empirical_expectile(proposal_q,cfg['expectile'])"
    assert source.count(before) == 1
    namespace = dict(vars(lean))
    exec(source.replace(before, 'root=jnp.max(proposal_q,axis=0)'), namespace)
    adapted = jax.jit(namespace['build_update']('lean', bounds))
    actual = jax.jit(core.build_update('max_q', bounds))
    expected, _ = adapted(state, batch)
    updated, logs = actual(state, batch)
    comparisons = [np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(updated))]
    assert all(comparisons)
    assert int(updated.q_updates) == 1 and int(updated.v_updates) == 0
    assert cfg['expectile'] == 1. and cfg['tau'] == .005 and cfg['discount'] == .999
    cfg09 = lean.config_for(environment, 'lean', width=16)
    state09 = lean.create(0, batch['observations'][:1], batch['actions'][:1], cfg09, 'lean')
    next09, _ = jax.jit(lean.build_update('lean', bounds))(state09, batch)
    a = updated.agent.network.params['modules_actor_flow']
    b = next09.agent.network.params['modules_actor_flow']
    assert all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
    policy = core.make_policy(updated, 'max_q')
    action = np.asarray(policy(core.policy_parameters(updated), batch['observations'][0], jax.random.PRNGKey(42)))
    assert action.shape == (2,) and np.isfinite(action).all() and np.abs(action).max() <= 1.
    probe = core.make_probe('max_q', bounds)(updated, batch)
    result = dict(status='PASS', scope='small real optimizer update and endpoint identities, not performance',
                  finite_expectile_max_gaps=gaps, bitwise_equal_state_leaves=len(comparisons),
                  original_lean_flow_update_bitwise_equal=True, expectile=cfg['expectile'],
                  target_EMA=cfg['tau'], discount=cfg['discount'], V_updates=0,
                  finite_probe=bool(probe['finite']), bounded_deployment_action=action.tolist())
    assert result['finite_probe']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
