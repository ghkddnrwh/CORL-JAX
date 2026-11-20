"""Fixed Ant pilot for extracting actions from the existing AWR mixture.

Never writes the GPU queue or changes either locked ensemble confirmation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from utils.flow_iql_research.rank_mean_bundle_v1 import sha


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def verify(path, job, plan):
    result = json.loads(path.read_text())
    assert result['kind'] == 'frozen_AWR_mixture_decoder_diagnostic'
    assert result['step'] == result['actual_Q_Adam_count'] == job['step']
    assert result['actual_V_Adam_count'] == 2 * job['step']
    assert result['rule'] == job['rule']
    assert result['config_sha256'] == plan['config_sha256']
    assert result['checkpoint_sha256'] == sha(Path(plan['run']) / f"checkpoint_{job['step']}.pkl")
    assert result['source_sha256'] == plan['source_sha256']['utils/flow_iql_research/evaluate_awr_mixture_decoder_v1.py']
    assert result['n_episodes'] == 50 and result['first_episode_seed'] == 42
    assert result['eval_device'] == 'cpu' and result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
    assert len(result['episode_successes']) == len(result['episode_rewards']) == len(result['episode_lengths']) == 50
    assert set(result['episode_successes']) <= {0., 1.}
    assert statistics.mean(result['episode_successes']) == result['success_rate']
    if job.get('self_reference'):
        assert all(result['self_reference_episode_arrays_exact'].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = Path(plan['run'])
    assert sha(run / 'config.json') == plan['config_sha256']
    config = json.loads((run / 'config.json').read_text())
    for path, digest in {**config['source_sha256'], **plan['source_sha256']}.items():
        assert sha(path) == digest, path
    running, valid, failures = {}, {}, []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    control = plan['jobs'][0]
    assert control['self_reference'] and control['step'] == 800000
    while True:
        for key, row in list(running.items()):
            code = row['process'].poll()
            if code is None:
                continue
            row['log'].close()
            del running[key]
            if code:
                failures.append(dict(id=key, exit_code=code))
            else:
                valid[key] = verify(Path(row['job']['output']), row['job'], plan)
        if failures:
            for row in running.values():
                row['process'].wait()
                row['log'].close()
            atomic(out / 'status.json', dict(time=time.time(), status='failure_requires_inspection',
                failures=failures, goal_complete=False))
            return
        allowed = plan['jobs'] if control['id'] in valid else [control]
        occupied = {row['slot'] for row in running.values()}
        for slot, cpus in enumerate(plan['CPUslots']):
            if slot in occupied:
                continue
            job = next((j for j in allowed if j['id'] not in valid and j['id'] not in running
                and (run / f"checkpoint_{j['step']}.pkl").exists()), None)
            if job is None:
                continue
            command = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-u', '-m',
                'utils.flow_iql_research.evaluate_awr_mixture_decoder_v1', str(run), '--step', str(job['step']),
                '--rule', job['rule'], '--output', job['output']]
            if job.get('self_reference'):
                command.append('--self-reference')
            log = (out / (job['id'] + '.log')).open('a')
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            running[job['id']] = dict(process=process, log=log, job=job, slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()[19])
        parent_done = (run / 'completed.json').exists()
        summaries = []
        for rule in ('component_Q', 'sample9_Q'):
            jobs = [j for j in plan['jobs'] if j['rule'] == rule]
            ready = len(jobs) == 3 and all(j['id'] in valid for j in jobs) and parent_done
            summaries.append(dict(rule=rule, complete=ready,
                per_step={str(j['step']): valid[j['id']]['success_rate'] for j in jobs if j['id'] in valid},
                last3_mean=statistics.mean(valid[j['id']]['success_rate'] for j in jobs) if ready else None))
        complete = len(valid) == 7 and parent_done
        baseline = None
        if complete:
            records = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
            late = [next(r for r in records if r['step'] == s)['success_rate'] for s in (800000, 900000, 1000000)]
            baseline = dict(rule='highest_probability_mean', per_step=dict(zip((800000, 900000, 1000000), late)), last3_mean=statistics.mean(late))
        result = dict(time=time.time(), pid=os.getpid(), status='complete' if complete else 'running',
            valid_evaluations=len(valid), parent_learner_complete=parent_done, summaries=summaries,
            original_decoder=baseline, failures=failures,
            running={key: dict(pid=row['process'].pid, slot=row['slot'],
                process_identity=row['process_identity']) for key, row in running.items()},
            result_sha256={key: sha(next(j['output'] for j in plan['jobs'] if j['id'] == key)) for key in valid},
            plan_sha256=sha(args.plan), source_sha256=sha(__file__),
            only_one_Ant_discovery_seed=True, nine_task_mean=None, primary_confirmation_unchanged=True,
            goal_complete=False)
        atomic(out / 'status.json', result)
        if complete:
            atomic(out / 'summary.json', result)
            print(json.dumps(dict(status='complete', baseline=baseline, methods=summaries)), flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
