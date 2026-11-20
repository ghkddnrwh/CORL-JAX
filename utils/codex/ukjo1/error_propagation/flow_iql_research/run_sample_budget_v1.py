"""Persistent seed0 sample-count sweep: 16 learners and independent CPU evals."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def birth(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lease = (root/'controller.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root/'registration_v1.json').read_text())
    assert plan['training_seed'] == 0
    assert json.loads((root/'analysis/numerical_validation_v1.json').read_text())['status'] == 'PASS'
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 96
    train_slots = [(gpu, slot) for slot in range(8) for gpu in (0, 1)]
    train_cpus = {slot: cpus[4*i:4*i+4] for i, slot in enumerate(train_slots)}
    eval_cpus = {slot: cpus[64+4*slot:68+4*slot] for slot in range(8)}
    jobs = {job['id']: job for job in plan['jobs']}
    state_path = root/'controller_status.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    assert not birth(old.get('pid', -1)), 'A controller already owns this study'
    finished, eval_finished = old.get('finished', {}), old.get('evaluation_finished', {})
    attempts, eval_attempts = old.get('attempts', {}), old.get('evaluation_attempts', {})
    history = old.get('retry_history', [])
    running, evaluating = {}, {}
    events = (root/'controller_events.jsonl').open('a', buffering=1)
    retry_after = {}
    last_analysis = 0.

    def event(kind, **values):
        events.write(json.dumps(dict(time=time.time(), kind=kind, **values))+'\n')
        print(kind, values, flush=True)

    def adopt(records, destination, training):
        for token, saved in records.items():
            if birth(saved['pid']) != saved['birth']:
                continue
            identity = token if training else token.split(':', 1)[0]
            job = jobs[identity]
            command = [v.decode() for v in Path(f"/proc/{saved['pid']}/cmdline").read_bytes().split(b'\0') if v]
            assert command[command.index('-m')+1] == job['train_module' if training else 'eval_module']
            flag = '--output' if training else '--run'
            assert Path(command[command.index(flag)+1]).resolve() == Path(job['output']).resolve()
            saved.update(process=None, log=None)
            if training:
                saved['slot'] = tuple(saved['slot'])
            destination[token] = saved
            event('ADOPT', identity=token, pid=saved['pid'])

    adopt(old.get('running', {}), running, True)
    adopt(old.get('evaluating', {}), evaluating, False)

    def done(record):
        return record['process'].poll() is not None if record['process'] else birth(record['pid']) != record['birth']

    def close(record):
        if record['log']:
            record['log'].close()
        return record['process'].returncode if record['process'] else None

    def serial(records):
        return {key: {k: v for k, v in rec.items() if k not in ('process', 'log')} for key, rec in records.items()}

    while True:
        stopping = (root/'STOP').exists()
        for token, rec in list(running.items()):
            if not done(rec):
                continue
            code = close(rec)
            output = Path(jobs[token]['output'])
            del running[token]
            complete, nonfinite = (output/'completed.json').exists(), (output/'stopped.json').exists()
            if not complete and not nonfinite and attempts.get(token, 0) < 2 and not stopping:
                entry = dict(kind='training', identity=token, exit_code=code, time=time.time(), reason='worker_exit')
                history.append(entry)
                retry_after[token] = time.time()+30
                event('TRAIN_RETRY_QUEUED', **{k: v for k, v in entry.items() if k != 'kind'})
            else:
                finished[token] = dict(result='complete' if complete else 'stopped' if nonfinite else 'failed', exit_code=code, time=time.time())
                event('TRAIN_FINISH', identity=token, **finished[token])
        for token, rec in list(evaluating.items()):
            if not done(rec):
                continue
            code = close(rec)
            del evaluating[token]
            complete = Path(rec['output']).exists()
            if not complete and eval_attempts.get(token, 0) < 2 and not stopping:
                history.append(dict(kind='evaluation', identity=token, exit_code=code, time=time.time()))
                retry_after[token] = time.time()+10
                event('EVAL_RETRY_QUEUED', identity=token, exit_code=code)
            else:
                eval_finished[token] = dict(result='complete' if complete else 'failed', exit_code=code, time=time.time())
                event('EVAL_FINISH', identity=token, **eval_finished[token])

        if not stopping:
            used = {tuple(r['slot']) for r in running.values()}
            for slot in train_slots:
                if slot in used:
                    continue
                job = next((j for j in plan['jobs'] if j['id'] not in running and j['id'] not in finished and time.time() >= retry_after.get(j['id'], 0)), None)
                if job is None:
                    break
                identity = job['id']
                output = Path(job['output'])
                output.mkdir(parents=True, exist_ok=True)
                if (output/'completed.json').exists():
                    finished[identity] = dict(result='complete', exit_code=None, rediscovered=True, time=time.time())
                    continue
                if (output/'stopped.json').exists():
                    finished[identity] = dict(result='stopped', exit_code=None, rediscovered=True, time=time.time())
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]),
                           XLA_PYTHON_CLIENT_MEM_FRACTION='0.1', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl',
                           PYTHONUNBUFFERED='1', PYTHONHASHSEED='0')
                env.pop('JAX_PLATFORMS', None)
                env.pop('JAX_PLATFORM_NAME', None)
                command = ['taskset', '-c', ','.join(map(str, train_cpus[slot])), sys.executable,
                           '-u', '-m', job['train_module'], '--env', job['environment'],
                           '--recipe', job['recipe'], '--seed', str(job['seed']), '--output', str(output)]
                log = (output/'stdout.log').open('a')
                worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                attempts[identity] = attempts.get(identity, 0)+1
                running[identity] = dict(pid=worker.pid, birth=birth(worker.pid), process=worker, log=log,
                                         gpu=slot[0], slot=slot, started=time.time())
                event('TRAIN_START', identity=identity, pid=worker.pid, gpu=slot[0], CPUs=train_cpus[slot], attempt=attempts[identity])

        tasks = []
        waiting_eval_artifacts = False
        for step in plan['evaluation_steps']:
            for identity, job in jobs.items():
                for mode in plan['deployment_modes']:
                    token = f'{identity}:{step}:{mode}'
                    output = root/'evaluations'/identity/f'{step}_{mode}.json'
                    if token in evaluating or token in eval_finished:
                        continue
                    if output.exists():
                        eval_finished[token] = dict(result='complete', exit_code=None, rediscovered=True, time=time.time())
                    elif (Path(job['output'])/f'checkpoint_{step}.pkl').exists() and (Path(job['output'])/f'actor_{step}.pkl').exists():
                        waiting_eval_artifacts = True
                        if time.time() >= retry_after.get(token, 0):
                            tasks.append((token, identity, step, mode, output))
        if not stopping:
            used = {r['slot'] for r in evaluating.values()}
            for slot in eval_cpus:
                if slot in used or not tasks:
                    continue
                token, identity, step, mode, output = tasks.pop(0)
                output.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
                command = ['taskset', '-c', ','.join(map(str, eval_cpus[slot])), sys.executable,
                           '-u', '-m', jobs[identity]['eval_module'], '--run', jobs[identity]['output'],
                           '--step', str(step), '--mode', mode, '--output', str(output)]
                log = output.with_suffix('.log').open('a')
                worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                eval_attempts[token] = eval_attempts.get(token, 0)+1
                evaluating[token] = dict(pid=worker.pid, birth=birth(worker.pid), process=worker, log=log,
                                         slot=slot, output=str(output), started=time.time())
                event('EVAL_START', identity=token, pid=worker.pid, CPUs=eval_cpus[slot])

        complete_train = sum(r['result'] == 'complete' for r in finished.values())
        complete_eval = sum(r['result'] == 'complete' for r in eval_finished.values())
        expected_eval = len(jobs)*len(plan['evaluation_steps'])*len(plan['deployment_modes'])
        retry_pending = any(k not in finished and k not in running for k in jobs)
        all_done = len(finished) == len(jobs) and not running and not evaluating and not waiting_eval_artifacts and not retry_pending
        complete = all_done and complete_train == len(jobs) and complete_eval == expected_eval
        status = dict(pid=os.getpid(), birth=birth(os.getpid()), time=time.time(),
                      phase='complete' if complete else 'incomplete' if all_done else 'draining' if stopping else 'running',
                      running=serial(running), evaluating=serial(evaluating),
                      pending=[j for j in jobs if j not in running and j not in finished],
                      finished=finished, evaluation_finished=eval_finished, attempts=attempts,
                      evaluation_attempts=eval_attempts, retry_history=history,
                      training_complete=complete_train, expected_training=len(jobs),
                      evaluations_complete=complete_eval, expected_evaluations=expected_eval,
                      reused_training=len(plan['reused_jobs']), total_requested_cells=plan['total_cells'])
        atomic(state_path, status)
        if all_done or time.time()-last_analysis >= 600:
            command = [sys.executable, '-u', '-m', 'utils.flow_iql_research.analyze_sample_budget_v1', '--root', str(root)]
            if complete:
                command.append('--final')
            with (root/'analysis_stdout.log').open('a') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            last_analysis = time.time()
            event('ANALYSIS', final=complete, exit_code=result.returncode)
            if all_done:
                status['analysis_exit_code'] = result.returncode
                if result.returncode:
                    status['phase'] = 'verification_failed'
                atomic(state_path, status)
        if all_done or (stopping and not running and not evaluating):
            event('CONTROLLER_EXIT', phase=status['phase'])
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
