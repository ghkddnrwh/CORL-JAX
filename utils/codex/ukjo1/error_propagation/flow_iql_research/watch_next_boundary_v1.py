"""Collect fixed entry-edge diagnostics for registered pilots and their controls."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=Path(plan['diagnostic_output_directory']);out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not(out/'status.json').exists(),'Inspect old process and children before explicit recovery'
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[40:44],cpus[44:48]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    running={};failures=[]
    while True:
        for path,digest in plan['diagnostic_source_sha256'].items():assert sha(path)==digest,path
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is None:continue
            row['log'].close();del running[key]
            if code:failures.append(dict(id=key,code=code,time=time.time()))
        rejected={row['id']for row in failures}
        pending=[job for job in plan['diagnostic_jobs']if job['id']not in running and job['id']not in rejected and not Path(job['output']).exists()]
        ready=[job for job in pending if (Path(job['run'])/f"checkpoint_{job['step']}.pkl").exists()]
        occupied={row['slot']for row in running.values()}
        for slot in range(2):
            if slot in occupied or not ready:continue
            job=ready.pop(0);log=(out/(job['id']+'.log')).open('a')
            cmd=['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                'utils.flow_iql_research.next_boundary_snapshot_v1',job['run'],'--step',str(job['step']),'--output',job['output']]
            process=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,log=log,slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        count=sum(Path(job['output']).exists()for job in plan['diagnostic_jobs'])
        atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),completed=count,total=len(plan['diagnostic_jobs']),
            failures=failures,running={key:dict(pid=row['process'].pid,process_identity=row['process_identity'],slot=row['slot'])for key,row in running.items()},
            goal_complete=False))
        if count+len(failures)==len(plan['diagnostic_jobs'])and not running:return
        time.sleep(10)


if __name__=='__main__':main()
