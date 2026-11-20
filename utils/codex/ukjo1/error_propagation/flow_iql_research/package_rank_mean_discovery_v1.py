"""Package all nine fixed1M discovery policies and verify their CPU50 replay.

The800K/900K/1M research metric is unchanged. This1M-only deployment manifest
is not a benchmark aggregate, new training replicate or checkpoint selection.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from utils.flow_iql_research.rank_mean_bundle_v1 import sha


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    assert sha(plan['factorial_plan']) == plan['factorial_plan_sha256']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    running, complete, failures = {}, {}, []
    jobs = plan['jobs']
    assert len(jobs) == 9 and len({j['environment'] for j in jobs}) == 9
    # The first CubeS proof already checks all episode arrays and denies original
    # checkpoint reads during load. Preserve that record and copy identical bytes.
    cube = next(j for j in jobs if j['environment'] == 'cube-single-play-singletask-v0')
    prior = json.loads(Path(plan['prior_validation']).read_text())
    assert prior['status'] == 'complete' and all(prior['CPU50_episode_success_reward_length_arrays_exact'].values())
    assert sha(prior['bundle']) == prior['bundle_sha256']
    shutil.copy2(prior['bundle'], cube['bundle'])
    assert sha(cube['bundle']) == prior['bundle_sha256']
    complete[cube['id']] = dict(environment=cube['environment'], bundle=cube['bundle'],
        bundle_sha256=prior['bundle_sha256'], CPU50_success=prior['CPU50_success'],
        reference_episode_arrays_exact=prior['CPU50_episode_success_reward_length_arrays_exact'],
        validation_record=plan['prior_validation'], validation_sha256=sha(plan['prior_validation']))

    def launch(job, stage, slot):
        log = (out / (job['id'] + '_' + stage + '.log')).open('a')
        command = ['taskset', '-c', ','.join(map(str, plan['CPUslots'][slot])), sys.executable, '-u', '-m']
        if stage == 'export':
            command += ['utils.flow_iql_research.export_rank_mean_bundle_v1', '--critic-runs',
                *job['critic_runs'], '--step', '1000000', '--output', job['bundle']]
        else:
            command += ['utils.flow_iql_research.evaluate_rank_mean_bundle_v1', job['bundle'],
                '--output', job['evaluation'], '--reference', job['reference']]
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        running[job['id']] = dict(process=process, log=log, job=job, stage=stage, slot=slot,
            process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()[19])

    while True:
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest, path
        for key, row in list(running.items()):
            code = row['process'].poll()
            if code is None:
                continue
            row['log'].close()
            del running[key]
            if code:
                failures.append(dict(job=key, stage=row['stage'], exit_code=code))
                continue
            job = row['job']
            if row['stage'] == 'export':
                with Path(job['bundle']).open('rb') as stream:
                    bundle = pickle.load(stream)
                assert [m['checkpoint_sha256'] for m in bundle['members']] == [p['checkpoint_sha256'] for p in job['member_pins']]
                assert bundle['recipe']['rule'] == 'mean_rank' and bundle['step'] == 1000000
                launch(job, 'evaluate', row['slot'])
            else:
                result = json.loads(Path(job['evaluation']).read_text())
                assert result['n_episodes'] == 50 and all(result['reference_episode_arrays_exact'].values())
                assert result['bundle_sha256'] == sha(job['bundle'])
                complete[key] = dict(environment=job['environment'], bundle=job['bundle'],
                    bundle_sha256=result['bundle_sha256'], CPU50_success=result['success_rate'],
                    reference_episode_arrays_exact=result['reference_episode_arrays_exact'],
                    validation_record=job['evaluation'], validation_sha256=sha(job['evaluation']))
        if failures:
            for row in running.values():
                row['process'].wait()
                row['log'].close()
            atomic(out / 'status.json', dict(time=time.time(), status='failure_requires_inspection',
                failures=failures, goal_complete=False))
            return
        occupied = {row['slot'] for row in running.values()}
        for slot in range(len(plan['CPUslots'])):
            if slot in occupied:
                continue
            job = next((j for j in jobs if j['id'] not in complete and j['id'] not in running), None)
            if job is not None:
                launch(job, 'export', slot)
        finished = len(complete) == 9
        result = dict(time=time.time(), pid=os.getpid(), status='complete' if finished else 'running',
            completed_tasks=len(complete), artifacts=list(complete.values()), failures=failures,
            running={key: dict(pid=row['process'].pid, stage=row['stage'], slot=row['slot'],
                process_identity=row['process_identity']) for key, row in running.items()},
            plan_sha256=sha(args.plan), source_sha256=sha(__file__), goal_complete=False,
            note='Only fixed1M deployment artifacts; not the last3 research metric or independent seed confirmation. No primary score file modified.')
        atomic(out / 'status.json', result)
        if finished:
            atomic(out / 'manifest.json', result)
            print(json.dumps(dict(status='complete', models=9, all_CPU50_arrays_exact=True)), flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
