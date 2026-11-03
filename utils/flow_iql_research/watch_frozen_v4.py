"""Two CPU workers diagnose retained snapshots while GPU learning continues."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis/frozen_Q_V_diagnostics_v1';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[:4],cpus[-4:]]
    running={};failures_path=out/'failures.json'
    failures=json.loads(failures_path.read_text()) if failures_path.exists() else {}
    while True:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del running[key]
            if code!=0:
                failures[key]=dict(exit_code=code,time=time.time());atomic(failures_path,failures)
            print('FINISH',key,code,flush=True)
        jobs=[]
        for run in sorted((root/'runs').glob('*')):
            if not (run/'config.json').exists():continue
            conf=json.loads((run/'config.json').read_text())
            if conf['identity']['kind'] in ('fql','rebrac'):continue
            for step in (100000,300000,800000,900000,1000000):
                key=f'{run.name}_{step}'
                if key in running or key in failures or (out/f'{key}.json').exists():continue
                if not (run/f'checkpoint_{step}.pkl').exists():continue
                jobs.append((key,run,step))
        stopping=(root/'STOP_FROZEN_DIAGNOSTICS').exists()
        occupied={j['slot'] for j in running.values()}
        if not stopping:
            for slot in range(2):
                if slot in occupied or not jobs:continue
                key,run,step=jobs.pop(0);log=(out/f'{key}.log').open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
                identity = json.loads((run/'config.json').read_text())['identity']
                diagnostic = ('utils.flow_iql_research.frozen_diagnostics_v5' if identity['kind'] in ('awr_mixture','awr_gaussian')
                    else 'utils.flow_iql_research.frozen_diagnostics_v4' if identity['kind']=='rank_flow'
                    else 'utils.flow_iql_research.frozen_diagnostics_v3')
                cmd=['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                     diagnostic,str(run),'--step',str(step),'--output',str(out/f'{key}.json')]
                p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[key]=dict(process=p,log=log,slot=slot,started_at=time.time())
                print('START',key,'PID',p.pid,flush=True)
        atomic(out/'status.json',dict(pid=os.getpid(),updated_at=time.time(),running={k:dict(pid=v['process'].pid,slot=v['slot'],started_at=v['started_at']) for k,v in running.items()},
                                    pending=len(jobs),completed=len(list(out.glob('*_[0-9]*.json'))),failures=len(failures)))
        if stopping and not running:break
        time.sleep(10)


if __name__=='__main__':main()
