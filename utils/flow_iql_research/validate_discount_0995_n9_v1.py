"""Check the discount-only configuration, real updates and reusable Giant runs."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import discount_0995_n9_core_v1 as core
from utils.flow_iql_research import giant_discount_expectile_core_v1 as giant
from utils.flow_iql_research import selection_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def identical(a, b):
    x, y = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(x) == len(y) and all(np.array_equal(np.asarray(i), np.asarray(j)) for i, j in zip(x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(d.platform == 'cpu' for d in jax.devices())
    for env in core.ENVIRONMENTS:
        cfg = core.config_for(env, 'g995_n9')
        reference = core.base.config_for(env, 'n9')
        assert cfg['discount'] == .995 and cfg['expectile'] == .9 and cfg['tau'] == .005
        assert cfg['backup_samples'] == cfg['num_samples'] == 9 and cfg['lr'] == .0003
        assert cfg['target_clipping'] and cfg['onestep_specification']['actor_label_samples'] == 9
        assert cfg['backup_Q_aggregation'] == ('min' if env.startswith('antmaze-') else 'mean')
        reference['discount'] = .995
        reference['onestep_specification'].update(name='g995_n9', discount=.995)
        assert cfg == reference
    rng = np.random.RandomState(99519)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0, 1, 1, 0, 1, 1, 1, 1], jnp.float32))
    cases = []
    envs = (giant.GIANT, 'humanoidmaze-medium-navigate-singletask-v0', 'puzzle-4x4-play-singletask-v0')
    for env in envs:
        for seed in (0, 1):
            cfg = core.config_for(env, 'g995_n9', width=16)
            bounds = core.return_bounds(env, cfg['discount'], -1., 0.)
            reward_components = 16 if env.startswith('puzzle-4x4-') else 1
            np.testing.assert_allclose(bounds, [-200. * reward_components, 0.])
            state = core.create(seed, batch['observations'][:1], batch['actions'][:1], cfg, 'g995_n9')
            update = jax.jit(core.build_update('g995_n9', bounds))
            actual, _ = update(state, batch)
            reference, _ = jax.jit(core.base.build_update('n9', bounds))(state, batch)
            assert identical(actual, reference)
            legacy_equal = None
            if env == giant.GIANT:
                legacy_cfg = giant.config_for(env, 'g995_t90', width=16)
                legacy = giant.create(seed, batch['observations'][:1], batch['actions'][:1], legacy_cfg, 'g995_t90')
                assert identical(state, legacy)
                legacy_update = jax.jit(giant.build_update('g995_t90', bounds))
                current = state
                for _ in range(3):
                    current, _ = update(current, batch)
                    legacy, _ = legacy_update(legacy, batch)
                    assert identical(current, legacy)
                legacy_equal = True
            values = core.make_probe('g995_n9', bounds)(actual, batch)
            assert bool(values['finite']) and tuple(np.asarray(values['diagnostic_proposal_shape'])) == (9, 8)
            agent = actual.base.agent
            obs, noise = core.lean.bank(agent, batch['next_observations'], jax.random.PRNGKey(41019), 9)
            actions = core.lean.flow(agent, obs, noise, 10)
            heads = agent.network.select('target_critic')(obs, actions=actions)
            q = heads.min(0) if env.startswith('antmaze-') else heads.mean(0)
            raw = batch['rewards'] + .995 * batch['masks'] * selection_v2.empirical_expectile(q, .9)
            np.testing.assert_allclose(np.asarray(values['diagnostic_raw_target']), np.asarray(raw), rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(np.asarray(values['diagnostic_final_target'])[[0, 3]], [-1., -1.])
            assert int(actual.base.q_updates) == int(actual.actor_updates) == 1 and int(actual.base.v_updates) == 0
            restored = flax.serialization.from_state_dict(state, flax.serialization.to_state_dict(actual))
            continued, _ = update(actual, batch)
            resumed, _ = update(restored, batch)
            assert identical(continued, resumed)
            cases.append(dict(environment=env, seed=seed, target_gamma=.995, bounds=list(bounds),
                              unchanged_N9_update_bitwise_equal=True, legacy_giant_three_updates_bitwise_equal=legacy_equal,
                              serialization_resume_bitwise_equal=True))
            print('PASS', env, seed, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS', configurations_checked=10, real_update_cases=cases,
        scope='Implementation and update checks; no performance claim.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
