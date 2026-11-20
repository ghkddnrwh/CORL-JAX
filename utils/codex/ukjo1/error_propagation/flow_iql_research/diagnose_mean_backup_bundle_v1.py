"""Same-observation full/pruned forward controls, without primary score edits."""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core
from utils.flow_iql_research import train_mean_empirical_backup_v1 as training
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.mean_backup_bundle_v1 import load, sha
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    bundle, compact, compact_policy = load(args.bundle)
    checkpoint = args.reference_run/'checkpoint_1000000.pkl'
    assert sha(checkpoint) == bundle['provenance']['checkpoint_sha256']
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    cfg = dict(bundle['identity']['config'])
    for k in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[k] = tuple(cfg[k])
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        template = core.create(bundle['training_seed'], jnp.zeros((1, bundle['observation_dim'])),
            jnp.zeros((1, bundle['action_dim'])), cfg, 'ifql')
        full = flax.serialization.from_state_dict(template, saved['state'])
    full = jax.device_put(full, cpu)
    scaffold = copy.deepcopy(template.agent.network.params)
    scaffold.update(compact.agent.network.params)
    full_policy = training.make_policy(template.agent)
    scaffold_state = template.replace(agent=template.agent.replace(network=template.agent.network.replace(params=scaffold)))
    noopt = template.agent.network.replace(params=scaffold, tx=None, opt_state=None, step=0)
    variants = {
        'compact': (compact.agent.network.params, compact_policy),
        'initial_unused_scaffolds': (scaffold_state.agent.network.params, training.make_policy(template.agent)),
        'full_tree_no_optimizer': (scaffold, training.make_policy(template.agent.replace(network=noopt))),
        'pruned_tree_original_optimizer_scaffold': (compact.agent.network.params, training.make_policy(template.agent)),
    }
    checks = {k: dict(queries=0, exact=0, max_abs_difference=0., first_difference=None) for k in variants}
    env = ogbench.make_env_and_datasets(bundle['environment'], env_only=True)
    prepare_seeded_action_space(env, 42)
    np.random.seed(42)
    obs, _ = env.reset(seed=42)
    rng = jax.device_put(jax.random.PRNGKey(42), cpu)
    for step in range(300):
        rng, key = jax.random.split(rng)
        query = np.asarray(obs, np.float32)
        reference = np.asarray(full_policy(full.agent.network.params, query, key))
        for name, (params, policy) in variants.items():
            action = np.asarray(policy(params, query, key))
            row = checks[name]
            row['queries'] += 1
            same = np.array_equal(reference, action)
            row['exact'] += int(same)
            gap = float(np.max(np.abs(reference-action)))
            row['max_abs_difference'] = max(row['max_abs_difference'], gap)
            if not same and row['first_difference'] is None:
                row['first_difference'] = dict(step=step, reference=reference.tolist(), action=action.tolist(), absolute_max=gap)
        obs, _, terminated, truncated, _ = env.step(reference.copy())
        if terminated or truncated:
            break
    env.close()
    result = dict(time=time.time(), kind='bundle_same_observation_forward_diagnostic', environment=bundle['environment'],
        reference_checkpoint_sha256=sha(checkpoint), bundle_sha256=sha(args.bundle), variants=checks,
        CPU_affinity=sorted(os.sched_getaffinity(0)), source_sha256=sha(__file__),
        primary_metric_modified=False, new_training_replication=False, goal_complete=False,
        scope='Identical observations/keys on one reference trajectory. Not an environment return comparison.')
    helper.atomic(args.output, result)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
