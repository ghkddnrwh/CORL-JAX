"""Test nominally identical value updates from a common real Puzzle state.

This CPU check cannot identify the cause of a historical GPU trajectory gap.
It checks for a mathematical implementation dependency on the policy family.
"""
import copy
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, mixture_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def difference(a, b):
    aa, bb = map(jax.tree_util.tree_leaves, (a, b))
    assert len(aa) == len(bb)
    return dict(bitwise_equal=all(np.array_equal(x, y) for x, y in zip(aa, bb)),
                max_absolute_difference=max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64))))
                                            for x, y in zip(aa, bb)))


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    names = ['p1_pz3_ifql_c0_bounded_v2_g999_t90_s0',
             'm1_pz3_gaussian_bc_bounded_v2_g999_t90_s0',
             'm1_pz3_mixture_bc_bounded_v2_g999_t90_s0']
    value_keys = ('modules_value', 'modules_critic', 'modules_target_critic')
    loaded = []
    for name in names:
        run = root/'runs'/name
        config = json.loads((run/'config.json').read_text())
        with (run/'checkpoint_100000.pkl').open('rb') as file:
            saved = pickle.load(file)
        loaded.append((config['identity'], saved))
    env, raw, _ = ogbench.make_env_and_datasets(loaded[0][0]['environment'])
    o = jnp.zeros((1, raw['observations'].shape[-1]))
    a = jnp.zeros((1, raw['actions'].shape[-1]))
    states, initial_states, updates = [], [], []
    for index, (ident, saved) in enumerate(loaded):
        cfg = dict(ident['config'])
        for field in ('value_hidden_dims', 'actor_hidden_dims'):
            cfg[field] = tuple(cfg[field])
        module = policy_core if ident['kind'] == 'ifql' else mixture_core
        initial = module.create(ident['seed'], o, a, cfg, ident['kind'])
        initial_states.append(initial)
        merged = copy.deepcopy(saved['state'])
        source = loaded[0][1]['state']
        for key in value_keys:
            merged['agent']['network']['params'][key] = copy.deepcopy(source['agent']['network']['params'][key])
            for moment in ('mu', 'nu'):
                merged['agent']['network']['opt_state']['0'][moment][key] = copy.deepcopy(
                    source['agent']['network']['opt_state']['0'][moment][key])
        merged['v_opt'] = copy.deepcopy(source['v_opt'])
        states.append(flax.serialization.from_state_dict(initial, merged))
        updates.append(jax.jit(module.build_update('bounded_v2', (-9000, 0), ident['kind'])))
    initial_comparisons = {names[i]: {k: difference(initial_states[0].agent.network.params[k],
                                                   initial_states[i].agent.network.params[k])
                                    for k in value_keys} for i in (1, 2)}
    rng = np.random.RandomState(919123)
    trace = []
    for step in range(1, 6):
        indices = rng.randint(len(raw['actions']), size=256)
        data = {k: jnp.asarray(raw[k][indices], jnp.float32)
                for k in ('observations', 'actions', 'next_observations', 'rewards', 'masks')}
        data['actions'] = jnp.clip(data['actions'], -1+1e-5, 1-1e-5)
        states = [update(state, data)[0] for update, state in zip(updates, states)]
        trace.append(dict(extra_CPU_updates=step, comparisons={names[i]: {
            k: difference(states[0].agent.network.params[k], states[i].agent.network.params[k])
            for k in value_keys} for i in (1, 2)}))
    record = dict(time=time.time(), kind='CPU_value_update_policy_independence_diagnostic',
        environment=loaded[0][0]['environment'], real_checkpoint_step=100000,
        initial_real_dimension_parameter_comparisons=initial_comparisons,
        trace=trace,
        caveats=['Starts from a common actual Q/V/Adam state; actors retain their own actual states.',
                 'CPU compilation/arithmetic can differ from historical GPU executions.',
                 'Matching CPU updates cannot prove a numerical cause for the observed historical GPU gap.',
                 'Primary training checkpoints, optimizers, replay streams and scores are unchanged.'])
    path = root/'analysis/puzzle_policy_value_CPU_equivalence.json'
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2)); tmp.replace(path)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
