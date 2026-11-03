"""Collect one CPU rank-calibration probe per registered policy checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();output=root/'analysis/rank_calibration_v1';output.mkdir(exist_ok=True)
    lock=(output/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures=json.loads((output/'failures.json').read_text())if(output/'failures.json').exists()else{}
    running=None
    while True:
        if running is not None and running['process'].poll()is not None:
            code=running['process'].returncode;running['log'].close()
            if code!=0:
                failures[running['name']]=dict(exit_code=code,time=time.time());atomic(output/'failures.json',failures)
            print('FINISH',running['name'],code,flush=True);running=None
        jobs=[]
        for run in sorted((root/'runs').iterdir()):
            if not(run/'config.json').exists():continue
            identity=json.loads((run/'config.json').read_text())['identity']
            if identity['kind']!='rank_flow':continue
            for step in (100000,300000,800000,900000,1000000):
                name=f'{run.name}_{step}'
                if not(run/f'checkpoint_{step}.pkl').exists()or(output/(name+'.json')).exists()or name in failures:continue
                if running is not None and running['name']==name:continue
                jobs.append((name,run,step))
        stopping=(root/'STOP_RANK_CALIBRATION').exists()
        if running is None and jobs and not stopping:
            name,run,step=jobs.pop(0)
            command=['taskset','-c','68-71',sys.executable,'-u','-m','utils.flow_iql_research.rank_data_calibration_v1',str(run),'--step',str(step),'--output',str(output/(name+'.json'))]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
            log=(output/(name+'.log')).open('a');process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running=dict(name=name,process=process,log=log,started_at=time.time());print('START',name,process.pid,flush=True)
        atomic(output/'status.json',dict(pid=os.getpid(),time=time.time(),running=None if running is None else dict(name=running['name'],pid=running['process'].pid,started_at=running['started_at']),pending=len(jobs),failures=failures,
            completed_outputs=len([path for path in output.glob('*.json')if path.name not in ('status.json','launch.json','failures.json')]),scope='IndependentCPUdiagnostics,notGPUtrainingreplicatesorbenchmarkevaluations.'))
        if stopping and running is None:return
        time.sleep(15)


if __name__=='__main__':main()
