"""Validate all four target switches against independently adapted frozen lean."""
import argparse
import inspect
import json
from pathlib import Path
from utils.flow_iql_research import target_ablation_core_v1 as core
from utils.flow_iql_research import efficient_mean_core_v1 as lean
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as original
import jax
import jax.numpy as jnp
import numpy as np


def identical(a, b):
    a, b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    environment = lean.ENVIRONMENTS[0]
    rng = np.random.RandomState(991)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=jnp.asarray([-5., 5.]*4), masks=jnp.ones(8))
    bounds = (-1., 0.)  # Intentionally force both bounds to exercise clipping.
    source = inspect.getsource(lean.build_update)
    cases = []
    for recipe in core.RECIPES:
        cfg = core.config_for(environment, recipe, width=16)
        state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
        text = source
        if recipe.startswith('ant_min'):
            text = text.replace("net.select('target_critic')(obs,actions=actions).mean(0)",
                                "net.select('target_critic')(obs,actions=actions).min(0)")
        if recipe.endswith('unclipped'):
            text = text.replace('target=jnp.clip(unbounded,*interval)', 'target=unbounded')
        namespace = dict(vars(lean))
        exec(text, namespace)
        expected, _ = jax.jit(namespace['build_update']('lean', bounds))(state.base, batch)
        actual_base, logs = jax.jit(core.base_update(recipe, bounds))(state.base, batch)
        actual_joint, _ = jax.jit(core.build_update(recipe, bounds))(state, batch)
        assert identical(expected, actual_base), recipe+' adapted frozen lean mismatch'
        assert identical(actual_base, actual_joint.base), recipe+' actor changes base learner'
        assert int(actual_joint.actor_updates) == int(actual_joint.base.q_updates) == 1
        assert int(actual_joint.base.v_updates) == 0
        clipping = not recipe.endswith('unclipped')
        assert float(logs['backup/would_clip_fraction']) == 1.
        assert float(logs['backup/clip_fraction']) == (1. if clipping else 0.)
        values = core.make_probe(recipe, bounds)(actual_joint, batch)
        assert bool(values['finite'])
        raw, y = (np.asarray(values[k]) for k in ('diagnostic_raw_target', 'diagnostic_final_target'))
        if clipping:
            assert np.all(y >= -1.) and np.all(y <= 0.)
        else:
            assert np.array_equal(raw, y) and (np.any(y < -1.) or np.any(y > 0.))
        metadata = core.actor_metadata(cfg, 5, 2)
        actor = core.make_actor_policy(metadata)
        action = np.asarray(actor(actual_joint.actor.params, batch['observations'][0], jax.random.PRNGKey(42)))
        teacher = core.make_teacher_policy(actual_joint)
        reference_action = np.asarray(teacher(core.lean.policy_parameters(actual_joint.base), batch['observations'][0], jax.random.PRNGKey(42)))
        assert all(x.shape == (2,) and np.isfinite(x).all() and np.max(np.abs(x)) <= 1 for x in (action, reference_action))
        cases.append(dict(recipe=recipe, frozen_lean_adaptation_bitwise_equal=True,
                          separate_actor_does_not_change_base_state=True,
                          real_optimizer_Q_updates=1, actor_updates=1, V_updates=0,
                          target_switches_exercised=True, finite_bounded_both_deployments=True))
    # The original joint recipe is also preserved at the default mean+clip cell.
    cfg = core.config_for(environment, 'mean_clip', width=16)
    state = core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, 'mean_clip')
    old, _ = jax.jit(original.build_update('mdn9_bc', bounds))(state, batch)
    new, _ = jax.jit(core.build_update('mean_clip', bounds))(state, batch)
    assert identical(old, new)
    giant_cfg = core.config_for(core.GIANT, 'mean_clip', width=16)
    assert giant_cfg == cfg
    assert np.allclose(core.return_bounds(core.GIANT, .999, -1., 0.), (-1000., 0.))
    result = dict(status='PASS', scope='Small numerical/optimizer/deployment checks; not benchmark performance.',
                  cases=cases, original_joint_mean_clip_state_bitwise_equal=True,
                  giant_uses_same_ant_family_config=True, giant_known_reward_range=[-1., 0.])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
