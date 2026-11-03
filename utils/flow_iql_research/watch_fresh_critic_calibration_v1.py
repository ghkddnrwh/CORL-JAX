"""Frozen-query calibration diagnostics on the two preassigned fresh cohorts.

No GPU work or environment policy episodes are launched. Measurements use the
same pre-existing512train/512validation query recipe as discovery. They cannot
change a locked method or substitute for its late policy evaluations.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job
from utils.flow_iql_research.review_rank_mean_calibration_v1 import describe
import numpy as np


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare(plan, index, verifier):
    root = Path(plan['root'])
    jobs = []
    for env in plan['environments']:
        members = [j for j in plan['training_jobs'] if j['args']['env'] == env and j['args']['seed'] in plan['fresh_groups'][index]]
        assert len(members) == 3
        if not all((root / 'runs' / j['id'] / 'completed.json').exists() for j in members):
            return None
        for step in (800000, 900000, 1000000):
            job = ready_job(root, plan, index, env, step, 'mean_rank', verifier)
            assert job is not None
            job['short'] = job['id'].split('_')[1]
            assert job['short'] in ('ant', 'hmmed', 'hmlarge', 'soccer', 'cube', 'cubed', 'pz3', 'pz4', 'scene')
            jobs.append(job)
    assert len(jobs) == 27 and len({j['short'] for j in jobs}) == 9
    return jobs


def review(directory, inputs, group, verifier):
    path = directory / 'summary.json'
    original = json.loads(path.read_text())
    assert original['status'] == 'complete' and original['models_audited'] == 81
    assert len(original['rows']) == 54 and original['unique_environments'] == 9
    assert original['source_sha256'] == verifier.digest('utils/flow_iql_research/independent_critic_calibration_v2.py')
    assert original['plan_sha256'] == verifier.digest(directory.parent / (directory.name + '_input.json'))
    expected_models = {(run, job['step']): pin for job in inputs['jobs']
        for run, pin in zip(job['critic_runs'], job['member_pins'])}
    assert len(expected_models) == len(original['models']) == 81
    for model in original['models']:
        pin = expected_models[(model['run'], model['step'])]
        assert model['checkpoint_sha256'] == pin['checkpoint_sha256']
        assert model['config_sha256'] == pin['config_sha256']
    rows, evidence = [], {}
    for env in inputs['environments']:
        job = next(j for j in inputs['jobs'] if j['environment'] == env)
        file = directory / (job['short'] + '_values.npz')
        evidence[str(file)] = verifier.digest(file)
        with np.load(file, allow_pickle=False) as saved:
            for split in ('train', 'validation'):
                indices = saved[split + '_indices']
                index_sha = __import__('hashlib').sha256(indices.tobytes()).hexdigest()
                for step in (800000, 900000, 1000000):
                    prior = next(r for r in original['rows'] if r['environment'] == env and r['split'] == split and r['step'] == step)
                    assert prior['index_sha256'] == index_sha
                    q = saved[f'{split}_{step}_Q']
                    assert q.shape == (3, 9, 512) and np.isfinite(q).all()
                    rows.append(dict(environment=env, split=split, step=step, **describe(q)))
    validation = [r for r in rows if r['split'] == 'validation']
    assert len(rows) == 54 and len(validation) == 27
    result = dict(time=time.time(), status='complete', learner_group=group,
        query_rows=rows, raw_table_sha256=evidence, prior_summary_sha256=verifier.digest(path),
        validation_aggregate=dict(mean_rank_vs_raw_mean_choice_difference=statistics.mean(
            r['contrasts']['raw_mean']['choice_difference_fraction'] for r in validation),
            mean_rank_vs_median_rank_choice_difference=statistics.mean(
            r['contrasts']['rank_median']['choice_difference_fraction'] for r in validation),
            mean_rank_top_tie_fraction=statistics.mean(r['rank_mean_top_tie_fraction'] for r in validation),
            mean_rank_overrides_unique_top_majority_fraction=statistics.mean(r['rank_mean_overrides_unique_top_majority_fraction'] for r in validation)),
        source_sha256=verifier.digest(__file__), primary_recipe_unchanged=True, goal_complete=False,
        scope='One fresh ensemble group, fixed dataset queries, not policy visitation or extra training replication.',
        limits=['The same512query indices are reused across3checkpoints and groups; do not treat rows as independent state replications.',
            'No oracle action rank, conditional residual or propagated B_k label.',
            'All rank/readout methods remain locked. Calibration proxies cannot replace benchmark success or choose per-task winners.'])
    atomic(directory / 'rank_mean_review.json', result)
    return dict(directory=str(directory), review_sha256=verifier.digest(directory / 'rank_mean_review.json'),
        member_seeds=group, raw_tables=9, rows=54, actual_member_snapshots=81,
        validation_aggregate=result['validation_aggregate'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration', type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    out = Path(registration['output'])
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verifier = Verifier()
    assert verifier.digest(registration['training_plan']) == registration['training_plan_sha256']
    plan = json.loads(Path(registration['training_plan']).read_text())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    completed, running, failures = {}, None, []
    while True:
        for source, digest in registration['source_sha256'].items():
            assert verifier.digest(source) == digest, source
        if running is not None and running['process'].poll() is not None:
            code = running['process'].returncode
            running['log'].close()
            if code:
                failures.append(dict(group=running['index'] + 1, exit_code=code))
            else:
                completed[str(running['index'] + 1)] = review(running['directory'], running['inputs'],
                    plan['fresh_groups'][running['index']], verifier)
            running = None
        if failures:
            atomic(out / 'status.json', dict(time=time.time(), pid=os.getpid(), status='failure_requires_inspection',
                failures=failures, goal_complete=False))
            return
        if running is None:
            for index in range(2):
                if str(index + 1) in completed:
                    continue
                jobs = prepare(plan, index, verifier)
                if jobs is None:
                    continue
                inputs = dict(environments=plan['environments'], jobs=jobs, source_sha256=registration['source_sha256'])
                input_path = out / f'group{index+1}_input.json'
                assert not input_path.exists()
                atomic(input_path, inputs)
                directory = out / f'group{index+1}'
                log = (out / f'group{index+1}.log').open('a')
                command = ['taskset', '-c', ','.join(map(str, registration['CPUcores'])), sys.executable,
                    '-u', '-m', 'utils.flow_iql_research.independent_critic_calibration_v2', str(input_path),
                    '--output', str(directory)]
                process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                running = dict(process=process, log=log, index=index, directory=directory, inputs=inputs,
                    process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()[19])
                break
        complete = len(completed) == 2
        atomic(out / 'status.json', dict(time=time.time(), pid=os.getpid(), status='complete' if complete else 'waiting_or_running',
            completed_groups=completed, running=None if running is None else dict(pid=running['process'].pid,
                group=running['index'] + 1, process_identity=running['process_identity']), failures=failures,
            registration_sha256=verifier.digest(args.registration), source_sha256=verifier.digest(__file__),
            additional_GPU_jobs=0, additional_environment_episodes=0,
            fixed_query_recipe='PrivateRandomState91349,512train/512validation,BCproposalkeys19405/19406,128batches,N9.',
            primary_recipe_unchanged='mean_rank', goal_complete=False))
        if complete:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
