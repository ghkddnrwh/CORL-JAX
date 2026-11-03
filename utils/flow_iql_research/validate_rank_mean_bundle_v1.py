"""Serialization and CPU50 equivalence of one actual discovery policy.

This is deployment verification, excluded from all benchmark aggregates.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from unittest import mock

from utils.flow_iql_research.rank_mean_bundle_v1 import load, sha
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan_path = root / 'independent_critic_factorial_registration_v1.json'
    plan = json.loads(plan_path.read_text())
    job = next(j for j in plan['jobs'] if j['environment'] == 'cube-single-play-singletask-v0'
        and j['step'] == 1000000 and j['rule'] == 'mean_rank')
    model = out / 'CubeS_1M_rank_mean.pkl'
    progress = dict(pid=os.getpid(), started_at=time.time(), source_sha256=sha(__file__),
        plan_sha256=sha(plan_path), reference_path=job['output'], reference_sha256=sha(job['output']),
        benchmark_scores_modified=False, new_training_replications=0)
    (out / 'progress.json').write_text(json.dumps(progress | dict(stage='exporting'), indent=2))
    subprocess.run([sys.executable, '-m', 'utils.flow_iql_research.export_rank_mean_bundle_v1',
        '--critic-runs', *job['critic_runs'], '--step', '1000000', '--output', str(model)], check=True)
    with model.open('rb') as stream:
        saved = pickle.load(stream)
    assert not {'state', 'replay_rng', 'v_opt', 'opt_state'} & saved.keys()
    assert saved['recipe']['rule'] == 'mean_rank' and len(saved['members']) == 3
    for run, member, pin in zip(job['critic_runs'], saved['members'], job['member_pins']):
        assert member['checkpoint_sha256'] == pin['checkpoint_sha256']
        assert member['config_sha256'] == pin['config_sha256']
    # Refuse any access to original learner directories while loading, proving
    # source checkpoint/config paths are provenance rather than dependencies.
    original_open = Path.open
    refused_paths = []

    def guarded_open(path, *positional, **keywords):
        if '/runs/' in str(path) or '/checkpoint_archive/' in str(path):
            refused_paths.append(str(path))
            raise AssertionError(f'Deployment loader accessed a source model: {path}')
        return original_open(path, *positional, **keywords)

    with mock.patch.object(Path, 'open', guarded_open):
        bundle, state, _ = load(model)
    assert not refused_paths
    restored = state.agent.network.params
    for index, (run, params) in enumerate(zip(job['critic_runs'],
        [restored, *bundle['other_critic_params']])):
        _, original, _ = read(Path(run), 1000000)
        expected = original['state']['agent']['network']['params']
        if index:
            expected = expected['modules_critic']
        a, ta = jax.tree_util.tree_flatten(params)
        b, tb = jax.tree_util.tree_flatten(expected)
        assert ta == tb and len(a) == len(b)
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    (out / 'progress.json').write_text(json.dumps(progress | dict(stage='CPU50_evaluating',
        source_parameter_trees_exact=True, no_source_model_reads_during_load=True), indent=2))
    result_path = out / 'CPU50_reference_equivalence.json'
    subprocess.run([sys.executable, '-m', 'utils.flow_iql_research.evaluate_rank_mean_bundle_v1', str(model),
        '--output', str(result_path), '--reference', job['output']], check=True)
    result = json.loads(result_path.read_text())
    assert all(result['reference_episode_arrays_exact'].values())
    report = progress | dict(completed_at=time.time(), status='complete', bundle=str(model),
        bundle_sha256=sha(model), bundle_bytes=model.stat().st_size,
        source_parameter_trees_exact=True, no_source_model_reads_during_load=True,
        CPU50_episode_success_reward_length_arrays_exact=result['reference_episode_arrays_exact'],
        CPU50_success=result['success_rate'], result_sha256=sha(result_path), goal_complete=False,
        scope='Deployment serialization verification of one already-scored retained CubeS model; excluded from benchmark counts.')
    (out / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('status', 'bundle_bytes', 'CPU50_success',
        'CPU50_episode_success_reward_length_arrays_exact', 'no_source_model_reads_during_load')}), flush=True)


if __name__ == '__main__':
    main()
