"""Collect rank-flow distribution probes and same-model teacher CPU50 scores."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    output = root/'analysis/rank_policy_diagnostics_v1'; output.mkdir(exist_ok=True)
    lock = (output/'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    slots = [list(range(80,84)), list(range(84,88))]
    running = {}; failed = json.loads((output/'failures.json').read_text()) if (output/'failures.json').exists() else {}
    while True:
        for name, record in list(running.items()):
            code = record['process'].poll()
            if code is None:
                continue
            record['log'].close(); del running[name]
            if code != 0:
                failed[name] = dict(exit_code=code, time=time.time()); atomic(output/'failures.json', failed)
            print('FINISH', name, code, flush=True)
        jobs = []
        for run in sorted((root/'runs').iterdir()):
            if not (run/'config.json').exists():
                continue
            identity = json.loads((run/'config.json').read_text())['identity']
            if identity['kind'] != 'rank_flow':
                continue
            for step in (100000,300000,800000,900000,1000000):
                if not (run/f'checkpoint_{step}.pkl').exists():
                    continue
                modes = ['distribution']+(['teacher_cpu50'] if step >= 300000 else [])
                for mode in modes:
                    name = f'{run.name}_{step}_{mode}'
                    if name in running or name in failed or (output/(name+'.json')).exists():
                        continue
                    jobs.append((name, run, step, mode))
        stopping = (root/'STOP_RANK_POLICY_DIAGNOSTICS').exists()
        occupied = {record['slot'] for record in running.values()}
        if not stopping:
            for slot, cpus in enumerate(slots):
                if slot in occupied or not jobs:
                    continue
                name, run, step, mode = jobs.pop(0)
                module = 'utils.flow_iql_research.rank_flow_policy_diagnostics_v1' if mode == 'distribution' else 'utils.flow_iql_research.evaluate_rank_flow_teacher_v1'
                command = ['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',module,
                           str(run),'--step',str(step),'--output',str(output/(name+'.json'))]
                if mode != 'distribution':
                    command += ['--mode','teacher','--episodes','50']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', MUJOCO_GL='egl', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
                log = (output/(name+'.log')).open('a')
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                running[name] = dict(process=process, log=log, slot=slot, started_at=time.time())
                print('START', name, process.pid, flush=True)
        atomic(output/'status.json', dict(pid=os.getpid(), time=time.time(),
            running={name:dict(pid=record['process'].pid, slot=record['slot'], started_at=record['started_at']) for name,record in running.items()},
            pending=len(jobs), failed=failed,
            completed_outputs=len([path for path in output.glob('*.json') if path.name not in ('status.json','failures.json','launch.json')]),
            scope='Auxiliary distributions and teacher returns; primary student scores remain untouched.'))
        if stopping and not running:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
