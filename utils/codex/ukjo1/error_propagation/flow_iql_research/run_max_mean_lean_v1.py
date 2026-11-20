"""Finite, persistent max-Q ablation queue: 18 learners, 2x8 GPU slots.

CPU50 late evaluations run independently on eight CPU slots. Every learner and
evaluation has a frozen identity. Existing experiment controllers are untouched.
"""
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
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def birth(pid):
    try:
        parts = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if parts[0] == 'Z' else parts[19]
    except (OSError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root/'registration_v1.json').read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 80, len(cpus)
    train_slots = [(gpu, slot) for slot in range(8) for gpu in range(2)]
    train_cpus = {slot: cpus[i*4:(i+1)*4] for i, slot in enumerate(train_slots)}
    spare = cpus[64:]
    eval_cpus = {i: spare[i*4:(i+1)*4] for i in range(min(8, len(spare)//4))}
    state_path = root/'controller_status.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    assert not birth(old.get('pid', -1)), 'Previous controller is alive'
    finished = old.get('finished', {})
    eval_finished = old.get('evaluation_finished', {})
    running, evaluating = {}, {}
    events = (root/'controller_events.jsonl').open('a', buffering=1)
    last_analysis = 0.

    def event(kind, **data):
        events.write(json.dumps(dict(time=time.time(), kind=kind) | data)+'\n')
        print(kind, data, flush=True)

    def adopt(records, destination):
        for key, rec in records.items():
            if birth(rec['pid']) != rec['birth']:
                continue
            cmd = Path(f"/proc/{rec['pid']}/cmdline").read_bytes().split(b'\0')
            assert str(root).encode() in b' '.join(cmd)
            rec.update(process=None, log=None)
            if 'gpu' in rec:
                rec['slot'] = tuple(rec['slot'])
            destination[key] = rec
            event('ADOPT', identity=key, pid=rec['pid'])

    adopt(old.get('running', {}), running)
    adopt(old.get('evaluating', {}), evaluating)

    def done(record):
        p = record['process']
        return p.poll() is not None if p else birth(record['pid']) != record['birth']

    def cleaned(record):
        if record.get('log'):
            record['log'].close()
        return record['process'].returncode if record['process'] else None

    while True:
        for key, rec in list(running.items()):
            if not done(rec):
                continue
            code = cleaned(rec)
            out = root/'runs'/key
            result = 'complete' if (out/'completed.json').exists() else 'stopped' if (out/'stopped.json').exists() else 'failed'
            finished[key] = dict(result=result, exit_code=code, time=time.time())
            del running[key]
            event('TRAIN_FINISH', identity=key, **finished[key])
        for key, rec in list(evaluating.items()):
            if not done(rec):
                continue
            code = cleaned(rec)
            output = Path(rec['output'])
            result = 'complete' if output.exists() else 'failed'
            eval_finished[key] = dict(result=result, exit_code=code, time=time.time())
            del evaluating[key]
            event('EVAL_FINISH', identity=key, **eval_finished[key])

        stopping = (root/'STOP').exists()
        if not stopping:
            used = {tuple(rec['slot']) for rec in running.values()}
            for slot in train_slots:
                if slot in used:
                    continue
                job = next((j for j in plan['jobs'] if j['id'] not in running and j['id'] not in finished), None)
                if job is None:
                    break
                key = job['id']
                out = root/'runs'/key
                out.mkdir(parents=True, exist_ok=True)
                if (out/'completed.json').exists():
                    finished[key] = dict(result='complete', exit_code=None, time=time.time(), rediscovered=True)
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]),
                           XLA_PYTHON_CLIENT_MEM_FRACTION='0.1', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl',
                           PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(job['seed']))
                env.pop('JAX_PLATFORMS', None)
                env.pop('JAX_PLATFORM_NAME', None)
                command = ['taskset', '-c', ','.join(map(str, train_cpus[slot])), sys.executable,
                           '-u', '-m', 'utils.flow_iql_research.train_max_mean_lean_v1',
                           '--env', job['environment'], '--variant', 'max_q', '--seed', str(job['seed']),
                           '--output', str(out)]
                log = (out/'stdout.log').open('a')
                p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[key] = dict(pid=p.pid, birth=birth(p.pid), process=p, log=log,
                                    slot=slot, gpu=slot[0], started=time.time())
                event('TRAIN_START', identity=key, pid=p.pid, gpu=slot[0], CPUs=train_cpus[slot])

        # Evaluate fixed late models as soon as they appear; no optimizer or GPU.
        tasks = []
        for step in plan['evaluation_steps']:
            for job in plan['jobs']:
                key = job['id']
                token = f'{key}:{step}'
                out = root/'evaluations'/key/f'{step}.json'
                if token in evaluating or token in eval_finished:
                    continue
                if out.exists():
                    eval_finished[token] = dict(result='complete', exit_code=None, time=time.time(), rediscovered=True)
                    continue
                if (root/'runs'/key/f'checkpoint_{step}.pkl').exists():
                    tasks.append((token, key, step, out))
        if not stopping:
            used = {rec['slot'] for rec in evaluating.values()}
            for slot in eval_cpus:
                if slot in used or not tasks:
                    continue
                token, key, step, output = tasks.pop(0)
                output.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                           MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
                command = ['taskset', '-c', ','.join(map(str, eval_cpus[slot])), sys.executable,
                           '-u', '-m', 'utils.flow_iql_research.evaluate_max_mean_lean_v1',
                           '--run', str(root/'runs'/key), '--step', str(step), '--output', str(output)]
                log = output.with_suffix('.log').open('a')
                p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                evaluating[token] = dict(pid=p.pid, birth=birth(p.pid), process=p, log=log,
                                         slot=slot, output=str(output), started=time.time())
                event('EVAL_START', identity=token, pid=p.pid, CPUs=eval_cpus[slot])

        def serial(records):
            return {key: {k: v for k, v in rec.items() if k not in ('process', 'log')}
                    for key, rec in records.items()}

        pending = [j['id'] for j in plan['jobs'] if j['id'] not in running and j['id'] not in finished]
        complete_train = sum(x['result'] == 'complete' for x in finished.values())
        complete_eval = sum(x['result'] == 'complete' for x in eval_finished.values())
        expected_eval = len(plan['jobs'])*len(plan['evaluation_steps'])
        all_done = len(finished) == len(plan['jobs']) and not running and not evaluating and (
            len(eval_finished) == expected_eval or not tasks)
        status = dict(pid=os.getpid(), birth=birth(os.getpid()), time=time.time(),
                      phase='complete' if all_done and complete_train == len(plan['jobs']) and complete_eval == expected_eval
                      else 'incomplete' if all_done else 'draining' if stopping else 'running',
                      running=serial(running), evaluating=serial(evaluating), pending=pending,
                      finished=finished, evaluation_finished=eval_finished,
                      training_complete=complete_train, evaluations_complete=complete_eval,
                      expected_training=len(plan['jobs']), expected_evaluations=expected_eval)
        atomic(state_path, status)
        if all_done or time.time()-last_analysis >= 600:
            command = [sys.executable, '-u', '-m', 'utils.flow_iql_research.analyze_max_mean_lean_v1',
                       '--root', str(root)]
            if all_done and status['phase'] == 'complete':
                command.append('--final')
            with (root/'analysis_stdout.log').open('a') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            last_analysis = time.time()
            event('ANALYSIS', final='--final' in command, exit_code=result.returncode)
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
