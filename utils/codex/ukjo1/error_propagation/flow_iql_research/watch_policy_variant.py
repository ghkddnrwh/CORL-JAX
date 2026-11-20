"""Evaluate one registered inference rule as actual parent checkpoints arrive.

Variant scores remain separate from the parents' original primary scores.
Missing tasks/steps never contribute a zero or a partial nine-task mean.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('plan', type=Path)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    out = Path(plan['output_directory']); out.mkdir(exist_ok=True, parents=True)
    lock = (out/'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for source, digest in plan['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
    running = {}; failures = {}; slots = [list(range(24, 28)), list(range(28, 32))]
    while True:
        for key, process in list(running.items()):
            code = process['p'].poll()
            if code is None:
                continue
            process['log'].close(); del running[key]
            if code:
                failures[key] = dict(exit_code=code, time=time.time())
        rows = []; pending = []; valid = {}
        for job in plan['jobs']:
            run = Path(job['run']); config_path = run/'config.json'
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == job['config_sha256']
            config = json.loads(config_path.read_text())['identity']
            assert config['kind'] == 'ifql' and config['profile'] == 'bounded_v2'
            assert config['seed'] == 0 and config['config']['discount'] == .999
            assert config['config']['expectile'] == .9
            path = Path(job['output'])
            if path.exists():
                result = json.loads(path.read_text())
                assert result['step'] == job['step'] and result['selection_samples'] == plan['samples']
                assert result['selection_rule'] == plan['rule'] and result['n_episodes'] == 50
                assert result['eval_device'] == 'cpu' and result['first_episode_seed'] == 42
                assert result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert result['config_sha256'] == job['config_sha256']
                successes = result['episode_successes']
                assert len(successes) == 50 and set(successes) <= {0., 1.}
                assert abs(statistics.mean(successes)-result['success_rate']) < 1e-12
                valid[job['id']] = result['success_rate']
            elif not job.get('externally_managed') and job['id'] not in running and job['id'] not in failures:
                if Path(job['checkpoint']).exists():
                    pending.append(job)
        for environment in plan['environments']:
            jobs = [j for j in plan['jobs'] if j['environment'] == environment]
            assert {j['step'] for j in jobs} == {800000, 900000, 1000000}
            complete = all(j['id'] in valid for j in jobs)
            parent_done = Path(jobs[0]['run'])/'completed.json'
            parent_complete = False
            if parent_done.exists():
                counts = json.loads(parent_done.read_text())
                parent_complete = counts['step'] == counts['q_updates'] == 1000000 and counts['v_updates'] == 2000000
            rows.append(dict(environment=environment, parent_complete=parent_complete,
                per_step={str(j['step']): valid.get(j['id']) for j in jobs},
                last3_mean=statistics.mean(valid[j['id']] for j in jobs) if complete and parent_complete else None))
        means = [row['last3_mean'] for row in rows if row['last3_mean'] is not None]
        macro = statistics.mean(means) if len(means) == 9 else None
        atomic(out/'variant_last3_audit.json', dict(time=time.time(), training_seed=0,
            rule=plan['rule'], inference_samples=plan['samples'], rows=rows,
            completed_task_means=len(means), nine_task_mean=macro,
            exceeds_provided_ReBRAC_FQL=macro > .47 if macro is not None else None,
            caveat='One global inference variant, exploratory seed0; no parent primary score is replaced. Nine tasks and all three checkpoints are required. Independent training-seed confirmation is still required.'))
        occupied = {p['slot'] for p in running.values()}
        for slot in range(2):
            if slot in occupied or not pending:
                continue
            job = pending.pop(0); target = Path(job['output']); target.parent.mkdir(exist_ok=True, parents=True)
            log = target.with_suffix('.log').open('ab')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', MUJOCO_GL='egl', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
            command = ['taskset', '-c', ','.join(map(str, slots[slot])), sys.executable, '-u', '-m',
                'utils.flow_iql_research.evaluate_selection_v2', job['run'], '--checkpoint', job['checkpoint'],
                '--output', str(target), '--episodes', '50', '--rule', plan['rule'], '--samples', str(plan['samples'])]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[job['id']] = dict(p=process, log=log, slot=slot)
        atomic(out/'status.json', dict(pid=os.getpid(), time=time.time(),
            running={k: p['p'].pid for k, p in running.items()}, completed_evaluations=len(valid),
            ready_pending=len(pending), failures=failures, completed_task_means=len(means)))
        if len(valid) == len(plan['jobs']) and not running and len(means) == 9:
            break
        if (out/'STOP').exists() and not running:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
