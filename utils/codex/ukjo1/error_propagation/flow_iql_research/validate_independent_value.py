"""Optimizer, replay-stream and resume checks for independent second V fits."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import independent_value_core as control, policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax


def equal_tree(a, b):
    aa, bb = map(jax.tree_util.tree_leaves, (a, b))
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(x, y)


def main():
    out = Path('logs/diagnosis/flow_iql_research_20261026/analysis')
    cfg = control.config_for(control.ENVIRONMENTS[4], discount=.999, width=8)
    initial = control.create(0, jnp.zeros((1, 4)), jnp.zeros((1, 2)), cfg, 'ifql')
    rng = np.random.RandomState(913)

    def batch():
        return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in dict(
            observations=rng.randn(7, 4), actions=rng.uniform(-1, 1, (7, 2)),
            next_observations=rng.randn(7, 4), rewards=-np.ones(7), masks=np.ones(7)).items()}

    first, second = batch(), batch()
    same = {**first, 'vfit_observations': first['observations'],
            'vfit_actions': first['actions']}
    independent = {**first, 'vfit_observations': second['observations'],
                   'vfit_actions': second['actions']}
    reference = jax.jit(policy_core.build_update('bounded_v2', (-1000, 0), 'ifql'))
    update = jax.jit(control.build_update('independent_v2', (-1000, 0), 'ifql'))
    a = b = initial
    errors = []
    for _ in range(3):
        a, _ = reference(a, first)
        b, _ = update(b, same)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
            errors.append(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64)))))

    # Explicit two-batch optimizer oracle: both targets must use the OLD Q.
    net = initial.agent.network
    vp, vo = initial.agent.network.params['modules_value'], initial.v_opt
    tx = optax.adam(cfg['lr'])
    for data in (first, second):
        q = net.select('target_critic')(data['observations'], actions=data['actions']).min(0)

        def objective(p):
            v = net.select('value')(data['observations'], params={**net.params, 'modules_value': p})
            d = q-v
            return jnp.mean(jnp.where(d >= 0, cfg['expectile'], 1-cfg['expectile'])*d*d)

        gradients = jax.grad(objective)(vp)
        updates, vo = tx.update(gradients, vo, vp)
        vp = optax.apply_updates(vp, updates)
    result, logs = update(initial, independent)
    for x, y in zip(jax.tree_util.tree_leaves(vp),
                    jax.tree_util.tree_leaves(result.agent.network.params['modules_value'])):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
    same_result, _ = update(initial, same)
    equal_tree(same_result.agent.network.params['modules_actor_flow'],
               result.agent.network.params['modules_actor_flow'])
    assert any(not np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(
        same_result.agent.network.params['modules_value']), jax.tree_util.tree_leaves(
        result.agent.network.params['modules_value'])))
    for oldq, oldt, newt in zip(jax.tree_util.tree_leaves(net.params['modules_critic']),
                              jax.tree_util.tree_leaves(net.params['modules_target_critic']),
                              jax.tree_util.tree_leaves(result.agent.network.params['modules_target_critic'])):
        np.testing.assert_allclose(newt, cfg['tau']*oldq+(1-cfg['tau'])*oldt, atol=1e-7, rtol=2e-6)

    # Checkpoint includes both independent replay RNGs as well as all Adam state.
    r1 = np.random.RandomState(0)
    r2 = np.random.RandomState(66341)

    def draw(r1, r2):
        i1, i2 = r1.randint(7, size=7), r2.randint(7, size=7)
        return {**{k: v[i1] for k, v in first.items()},
                'vfit_observations': first['observations'][i2],
                'vfit_actions': first['actions'][i2]}

    state = initial
    for _ in range(2):
        state, _ = update(state, draw(r1, r2))
    saved = pickle.loads(pickle.dumps(dict(state=flax.serialization.to_state_dict(state),
        replay_rng=r1.get_state(), value_replay_rng=r2.get_state())))
    restored = flax.serialization.from_state_dict(initial, saved['state'])
    rr1, rr2 = np.random.RandomState(), np.random.RandomState()
    rr1.set_state(saved['replay_rng']); rr2.set_state(saved['value_replay_rng'])
    for _ in range(2):
        data, recovered = draw(r1, r2), draw(rr1, rr2)
        equal_tree(data, recovered)
        state, _ = update(state, data)
        restored, _ = update(restored, recovered)
    equal_tree(state, restored)
    assert int(state.q_updates) == 4 and int(state.v_updates) == 8
    assert int(state.v_opt[0].count) == 8 and int(state.agent.network.opt_state[0].count) == 4
    assert all(np.isfinite(np.asarray(v)).all() for v in logs.values())
    record = dict(passed=True, time=time.time(),
        same_second_batch_recovers_original_bounded_v2=True, maximum_state_difference=max(errors),
        independent_two_fit_V_matches_explicit_old_targetQ_oracle=True,
        BC_flow_update_bitwise_identical=True, independent_second_batch_changes_V=True,
        old_Q_target_EMA_verified=True, full_state_and_two_replay_RNG_resume_bitwise_equal=True,
        q_updates=4, v_updates=8, scope='Tiny CPU semantics checks; no learning-performance conclusion.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(control.__file__), Path(__file__))})
    (out/'independent_value_validation_v1.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
