"""Collect registered goal-boundary diagnostics while GPU learning continues."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();plan=json.loads((root/'terminal_boundary_snapshot_registration_v1.json').read_text())
    out=root/'analysis/terminal_boundary_snapshots_v1';out.mkdir(exist_ok=True)
    lock=(out/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[8:12],cpus[-12:-8]]
    running={};failure_path=out/'failures.json';failures=json.loads(failure_path.read_text())if failure_path.exists()else{}
    while True:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del running[key]
            if code:failures[key]=dict(exit_code=code,time=time.time());atomic(failure_path,failures)
        pending=[job for job in plan['jobs']if job['id']not in running and job['id']not in failures and not Path(job['output']).exists()]
        ready=[job for job in pending if (Path(job['run'])/f"checkpoint_{job['step']}.pkl").exists()]
        occupied={job['slot']for job in running.values()};stopping=(out/'STOP').exists()
        if not stopping:
            for slot in range(2):
                if slot in occupied or not ready:continue
                job=ready.pop(0);log=(out/(job['id']+'.log')).open('a')
                command=['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                         'utils.flow_iql_research.terminal_boundary_snapshot_v1',job['run'],'--step',str(job['step']),
                         '--output',job['output']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
                process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                running[job['id']]=dict(process=process,log=log,slot=slot,started_at=time.time())
                print('START',job['id'],process.pid,flush=True)
        atomic(out/'status.json',dict(pid=os.getpid(),updated_at=time.time(),completed=sum(Path(job['output']).exists()for job in plan['jobs']),
            pending=len(pending),failures=len(failures),running={key:dict(pid=job['process'].pid,slot=job['slot'],started_at=job['started_at'])for key,job in running.items()}))
        if stopping and not running:break
        time.sleep(10)


if __name__=='__main__':main()
