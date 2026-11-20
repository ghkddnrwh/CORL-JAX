"""Compare actual control/large-batch updates across an import-only relocation."""
import argparse
import importlib
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['before', 'after'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    core = importlib.import_module(
        'utils.codex.ukjo1.error_propagation.flow_iql_research.stability_study_core_v1')
    import jax
    import jax.numpy as jnp
    import flax.serialization
    outputs = {}
    for environment in (core.ENVIRONMENTS[0], core.ENVIRONMENTS[-1]):
        for recipe in ('control', 'critic_batch1024'):
            rng = np.random.RandomState(761)
            batch = {k: jnp.asarray(v, jnp.float32) for k, v in dict(
                observations=rng.normal(size=(8, 5)),
                actions=rng.uniform(-.9, .9, (8, 2)),
                next_observations=rng.normal(size=(8, 5)),
                rewards=-rng.uniform(0, 1, 8),
                masks=np.array([1, 1, 1, 1, 1, 1, 0, 0])).items()}
            if recipe == 'critic_batch1024':
                batch['critic_extra'] = {k: jnp.stack([v, v, v]) for k, v in batch.items()}
            cfg = core.config_for(environment, recipe, 16)
            state = core.create(2, jnp.zeros((1, 5)), jnp.zeros((1, 2)), cfg, recipe)
            update = jax.jit(core.build_update(recipe, (-1000., 0.)))
            for _ in range(3):
                state, logs = update(state, batch)
            restored = flax.serialization.from_bytes(state, flax.serialization.to_bytes(state))
            next_state, next_logs = update(state, batch)
            resumed, resumed_logs = update(restored, batch)
            for a, b in zip(jax.tree_util.tree_leaves((next_state, next_logs)),
                            jax.tree_util.tree_leaves((resumed, resumed_logs))):
                np.testing.assert_array_equal(a, b)
            outputs[environment + ':' + recipe] = jax.device_get(
                (flax.serialization.to_state_dict(next_state), next_logs))
            print(environment, recipe, 'resume exact', flush=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / 'numerical_before.pkl'
    if args.phase == 'before':
        assert not path.exists()
        with path.open('wb') as stream:
            pickle.dump(outputs, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with path.open('rb') as stream:
            before = pickle.load(stream)
        assert before.keys() == outputs.keys()
        checked = 0
        for key in outputs:
            a, b = jax.tree_util.tree_leaves(before[key]), jax.tree_util.tree_leaves(outputs[key])
            assert len(a) == len(b)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
                checked += 1
        (args.directory / 'numerical_validation.json').write_text(json.dumps(dict(
            status='PASS', cases=len(outputs), arrays_compared=checked,
            maximum_difference=0, serialization_resume_exact=True), indent=2) + '\n')
        print('PASS: all updates and optimizer/RNG states exactly equal', flush=True)


if __name__ == '__main__':
    main()
