"""Check that legacy unused-V fits and current lean/actor updates agree."""
import argparse
import json
from pathlib import Path
from utils.flow_iql_research import mean_empirical_backup_core_v1 as original
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as mixture
from utils.flow_iql_research import target_ablation_core_v1 as current
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as routed
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy
import jax
import jax.numpy as jnp
import numpy as np


def compare(a, b):
    aa, bb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(aa) == len(bb)
    largest = 0.
    for x, y in zip(aa, bb):
        x, y = np.asarray(x), np.asarray(y)
        np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-6)
        largest = max(largest, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return largest


def active(agent):
    return {k: agent.network.params[k] for k in
            ('modules_critic', 'modules_target_critic', 'modules_actor_flow')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and all(d.platform == 'cpu' for d in jax.devices())
    assert routed.decode is lazy.decode
    env = 'humanoidmaze-medium-navigate-singletask-v0'
    rng = np.random.RandomState(20261120)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0, 1, 1, 1, 0, 1, 1, 1], jnp.float32))
    cfg = original.config_for(env, .999, .9, width=16)
    cfg.update(num_samples=9, policy_type='ifql', agent_name='ifql')
    old = original.create(0, batch['observations'][:1], batch['actions'][:1], cfg, 'ifql')
    cur_cfg = current.config_for(env, 'mean_clip', width=16)
    new = current.create(0, batch['observations'][:1], batch['actions'][:1], cur_cfg, 'mean_clip')
    mix_cfg = mixture.config_for(env, 'mdn9_bc', width=16)
    mix = mixture.create(0, batch['observations'][:1], batch['actions'][:1], mix_cfg, 'mdn9_bc')
    assert compare(active(old.agent), active(new.base.agent)) == 0.
    assert compare(mix.actor.params, new.actor.params) == 0.
    bounds = (-1000., 0.)
    old_step = jax.jit(original.build_update('bounded_v2_mean_empirical_backup', bounds))
    new_step = jax.jit(current.build_update('mean_clip', bounds))
    mix_step = jax.jit(mixture.build_update('mdn9_bc', bounds))
    altered_params = dict(old.agent.network.params)
    altered_params['modules_value'] = jax.tree_util.tree_map(lambda x: x + 17., altered_params['modules_value'])
    altered = old.replace(agent=old.agent.replace(network=old.agent.network.replace(params=altered_params)))
    a, _ = old_step(old, batch)
    b, _ = old_step(altered, batch)
    value_independence = compare(active(a.agent), active(b.agent))
    cases = []
    for step in range(1, 4):
        old, _ = old_step(old, batch)
        new, _ = new_step(new, batch)
        mix, _ = mix_step(mix, batch)
        legacy_difference = compare(active(old.agent), active(new.base.agent))
        actor_base_difference = compare(active(mix.base.agent), active(new.base.agent))
        actor_difference = compare(mix.actor.params, new.actor.params)
        np.testing.assert_array_equal(np.asarray(old.agent.rng), np.asarray(new.base.agent.rng))
        assert int(old.v_updates) == 2*step and int(new.base.v_updates) == 0
        cases.append(dict(step=step, original_vs_lean_active_param_max_abs=legacy_difference,
                          legacy_mdn_vs_current_base_max_abs=actor_base_difference,
                          legacy_mdn_vs_current_actor_max_abs=actor_difference))
        print('PASS', step, cases[-1], flush=True)
    result = dict(status='PASS', real_optimizer_steps=3, cases=cases,
        altered_unused_value_network_active_param_max_abs=value_independence,
        old_and_current_routed_decoder_use_identical_function=True,
        mathematical_scope='N9 mean target, tau=.9 gamma=.999, same bounds/BC Flow10, min-Q best9 extraction; pure MDN NLL and component-location deployment.',
        exclusions=['Mean AntMaze target cannot be pooled with min AntMaze target.',
                    'Native MDN Gaussian rollout cannot be pooled with component-location rollout.',
                    'Different N/tau/clipping/LR interventions are not equivalent.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
