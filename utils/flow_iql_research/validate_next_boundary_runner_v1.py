"""Exercise actual dataset runner, source pinning, replay resume and CPU policy."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np


def equal_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'runner.json').exists(), 'Use a fresh validation directory'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    common = ['--env', 'cube-single-play-singletask-v0', '--train-device', 'cpu', '--width', '8',
              '--discount', '.999', '--expectile', '.9', '--samples', '9', '--steps', '4',
              '--chunk', '2', '--probe-every', '2', '--episodes', '1', '--aux-eval-steps', '2,4']
    cpu_ids = sorted(os.sched_getaffinity(0))[:4]
    prefix = ['taskset', '-c', ','.join(map(str, cpu_ids)), sys.executable, '-u', '-m']
    def run(output, extra, label):
        with (root/(label+'.log')).open('w') as log:
            subprocess.run(prefix+['utils.flow_iql_research.train_next_boundary_backup_v1',
                                   '--output', str(output)]+common+extra,
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    run(root/'full', [], 'full')
    run(root/'resumed', ['--stop-at', '2'], 'stage')
    run(root/'resumed', [], 'resume')
    with (root/'full/latest.pkl').open('rb') as stream: full = pickle.load(stream)
    with (root/'resumed/latest.pkl').open('rb') as stream: resumed = pickle.load(stream)
    equal_tree(full, resumed)
    evals = []
    for directory in ('full', 'resumed'):
        evals.append([json.loads(line) for line in (root/directory/'evaluations.jsonl').read_text().splitlines()])
    assert [record['step'] for record in evals[0]] == [2, 4]
    for a, b in zip(*evals):
        for key in ('episode_successes', 'episode_rewards', 'episode_lengths'):
            assert a[key] == b[key]
    frozen = root/'frozen.json'
    with (root/'frozen.log').open('w') as log:
        subprocess.run(prefix+['utils.flow_iql_research.frozen_diagnostics_v3', str(root/'full'),
                              '--step', '4', '--output', str(frozen)], env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    diagnostic = json.loads(frozen.read_text())
    assert diagnostic['initial_v_Adam_count'] == 8 and diagnostic['final_auxiliary_v_Adam_count'] == 16
    result = dict(time=time.time(), actual_cube_single_four_CPU_updates=True,
                  source_pinned_payload_and_replay_RNG_full_vs_resumed_exact=True,
                  two_checkpoint_episode_arrays_exact=True, evaluation_episodes_per_checkpoint=1,
                  frozen_diagnostic_compatible=True, actual_frozen_V_clock=[8, 16],
                  scope='Tiny CPU integration validation only; excluded from every benchmark score.')
    (root/'runner.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
