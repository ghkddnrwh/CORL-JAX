"""Check that inherited Gaussian/student settings do not affect primary IFQL.

Uses one already-seen actual CubeS model and in-memory CPU updates only. No
learning state, primary evaluation, registered config or source is rewritten.
"""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research import policy_core
from utils.flow_iql_research import independent_critic_policy_v2 as policy
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.rank_mean_bundle_v1 import sha
import flax
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def equal(first, second):
    a = flax.serialization.to_state_dict(jax.device_get(first))
    b = flax.serialization.to_state_dict(jax.device_get(second))
    x, tx = jax.tree_util.tree_flatten(a)
    y, ty = jax.tree_util.tree_flatten(b)
    assert tx == ty and len(x) == len(y)
    assert all(np.array_equal(np.asarray(i), np.asarray(j)) for i, j in zip(x, y))
    return len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    plan_path = args.root / 'independent_critic_factorial_registration_v1.json'
    plan = json.loads(plan_path.read_text())
    job = next(j for j in plan['jobs'] if j['environment'] == 'cube-single-play-singletask-v0'
        and j['step'] == 1000000 and j['rule'] == 'mean_rank')
    run = Path(job['actor_run'])
    conf, saved, checkpoint = read(run, 1000000)
    cfg = dict(conf['identity']['config'])
    for field in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    changes = dict(alpha=-12345., actor_loss='unused_IFQL_sentinel', const_std=False,
        flow_temperature=.001, distill_coefficient=100000., normalize_advantage=False)
    assert all(k in cfg for k in changes)
    altered = {**cfg, **changes}
    env, data, _ = ogbench.make_env_and_datasets(job['environment'])
    fields = ('observations', 'actions', 'next_observations', 'rewards', 'masks')
    rng = np.random.RandomState(91400)
    indices = rng.randint(len(data['actions']), size=(4, 16))
    batches = {k: jnp.asarray(data[k][indices], dtype=jnp.float32) for k in fields}
    batches['actions'] = jnp.clip(batches['actions'], -1 + 1e-5, 1 - 1e-5)
    obs = jnp.asarray(data['observations'][:1], jnp.float32)
    action = jnp.asarray(data['actions'][:1], jnp.float32)
    initial = policy_core.create(0, obs, action, cfg, 'ifql')
    changed_initial = policy_core.create(0, obs, action, altered, 'ifql')
    initial_arrays = equal(initial, changed_initial)
    state = flax.serialization.from_state_dict(initial, saved['state'])
    changed = flax.serialization.from_state_dict(changed_initial, saved['state'])
    equal(state, changed)
    dataset = json.loads((run / 'dataset.json').read_text())
    bounds = (dataset['return_lower'], dataset['return_upper'])
    scan = policy_core.make_scan(policy_core.build_update('bounded_v2', bounds, 'ifql'))
    new, logs = scan(state, batches)
    other, other_logs = scan(changed, batches)
    updated_arrays = equal(new, other)
    equal(logs, other_logs)
    assert int(new.q_updates) == int(other.q_updates) == 1000004
    assert int(new.v_updates) == int(other.v_updates) == 2000008
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(new))
    members = []
    for name, pin in zip(job['critic_runs'], job['member_pins']):
        c, actual, path = read(Path(name), 1000000)
        assert sha(path) == pin['checkpoint_sha256']
        members.append(jax.device_put(actual['state']['agent']['network']['params']['modules_critic'], jax.devices('cpu')[0]))
    original_selector = policy.make_selector(state.agent, 'mean_rank', 9)
    changed_selector = policy.make_selector(changed.agent, 'mean_rank', 9)
    observation = jnp.asarray(data['observations'][indices[0, 0]], jnp.float32)
    for key in range(8):
        seed = jax.random.PRNGKey(key + 19409)
        first = original_selector(state.agent.network.params, tuple(members), observation, seed)
        second = changed_selector(changed.agent.network.params, tuple(members), observation, seed)
        equal(first, second)
    env.close()
    result = dict(time=time.time(), status='passed', source_sha256=sha(__file__),
        plan_sha256=sha(plan_path), actual_checkpoint_sha256=sha(checkpoint),
        original_inherited_values={k: cfg[k] for k in changes}, contrast_values=changes,
        initialization_parameter_optimizer_RNG_arrays_exact=True, initialization_arrays_compared=initial_arrays,
        actual_restored_state_exact=True, four_CPU_updates_all_state_and_metrics_exact=True,
        updated_arrays_compared=updated_arrays, actual_final_Q_count=1000004, actual_final_V_count=2000008,
        eight_actual_three_critic_N9_policy_action_Q_score_outputs_exact=True,
        no_updated_learning_states_saved=True, primary_evaluations_modified=False, goal_complete=False,
        scope='Existing CubeS architecture/checkpoint, same data/RNG and four CPU-only update batches. No new policy evaluation or training replication.',
        interpretation='For the primary IFQL path these inherited Gaussian/AWR/student fields are not extra tuned algorithm components. Active value/proposal/ensemble settings remain as registered.',
        limits=['This is implementation invariance of specified fields, not a performance or convergence guarantee.',
            'These fields can matter for OTHER kinds such as Gaussian IQL, AWRflow, distilled or FQL-policy agents.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ('status', 'initialization_arrays_compared',
        'updated_arrays_compared', 'four_CPU_updates_all_state_and_metrics_exact',
        'eight_actual_three_critic_N9_policy_action_Q_score_outputs_exact')}), flush=True)


if __name__ == '__main__':
    main()
