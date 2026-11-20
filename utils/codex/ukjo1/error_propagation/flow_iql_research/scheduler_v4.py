"""Persistent 2x8 queue with ownership-checked adoption after manager restarts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MODULES = {'utils.flow_iql_research.train', 'utils.flow_iql_research.train_policy',
           'utils.flow_iql_research.train_fql', 'utils.flow_iql_research.train_mixture'}


def atomic_json(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)


def process_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z': return None
        return stat[19]
    except (OSError,IndexError): return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--per-gpu',type=int,default=8);args=parser.parse_args()
    root=args.directory.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu_ids=sorted(os.sched_getaffinity(0))
    slots=[(gpu,slot) for slot in range(args.per_gpu) for gpu in range(2)]
    per_process=max(1,len(cpu_ids)//len(slots))
    slot_cpus={slot:cpu_ids[i*per_process:(i+1)*per_process] for i,slot in enumerate(slots)}
    finished_path=root/'finished.json';status_path=root/'scheduler_status.json'
    finished=json.loads(finished_path.read_text()) if finished_path.exists() else {}
    running={}
    initial_queue={j['id']:j for j in json.loads((root/'queue.json').read_text())}
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        old_pid=previous.get('pid')
        if old_pid!=os.getpid() and process_identity(old_pid):
            raise RuntimeError(f'Previous scheduler {old_pid} is still alive')
        for key,record in previous.get('running',{}).items():
            pid=record['pid'];identity=process_identity(pid)
            if not identity: continue
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert '-m' in cmd and cmd[cmd.index('-m')+1] in MODULES
            assert Path(cmd[cmd.index('--output')+1]).resolve()==root/'runs'/initial_queue.get(key,{}).get('output_run',key)
            matches=[s for s in slots if s[0]==record['gpu'] and set(slot_cpus[s])==os.sched_getaffinity(pid)]
            assert len(matches)==1, (pid,matches)
            running[key]=dict(record,process=None,log=None,slot=matches[0],process_identity=identity)
            print('ADOPT',key,pid,flush=True)
    while True:
        for key,record in list(running.items()):
            process=record['process']
            if process is not None:
                code=process.poll()
                if code is None: continue
            else:
                if process_identity(record['pid'])==record['process_identity']: continue
                code=None  # An adopted orphan's exit code is not observable.
            if record['log'] is not None: record['log'].close()
            output=Path(record['output'])
            result='complete' if (output/'completed.json').exists() else 'screened' if (output/'screened.json').exists() else 'stopped' if (output/'stopped.json').exists() else 'staged' if (output/'staging_request.json').exists() else 'failed'
            finished[key]=dict(exit_code=code,result=result,finished_at=time.time(),output=str(output))
            atomic_json(finished_path,finished);del running[key]
            print('FINISH',key,code,result,flush=True)
        queue=json.loads((root/'queue.json').read_text())
        stopping=(root/'STOP').exists()
        used={v['slot'] for v in running.values()}
        if not stopping:
            for slot in slots:
                if slot in used: continue
                job=next((j for j in queue if j['id'] not in running and j['id'] not in finished
                          and str(root/'runs'/j.get('output_run',j['id'])) not in {v['output'] for v in running.values()}),None)
                if job is None: break
                module=job.get('module','utils.flow_iql_research.train');assert module in MODULES
                output=root/'runs'/job.get('output_run',job['id']);output.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot[0]),XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',
                         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
                         PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
                cmd=['taskset','-c',','.join(map(str,slot_cpus[slot])),sys.executable,'-u','-m',module,'--output',str(output)]
                for k,v in job['args'].items(): cmd.extend(['--'+k,str(v)])
                log=(output/'stdout.log').open('a')
                p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[job['id']]=dict(process=p,log=log,pid=p.pid,slot=slot,output=str(output),
                                        started_at=time.time(),process_identity=process_identity(p.pid))
                print('START',job['id'],'GPU',slot[0],'PID',p.pid,flush=True)
        atomic_json(status_path,dict(pid=os.getpid(),updated_at=time.time(),phase='draining' if stopping else 'running',
            running={k:{name:v[name] for name in ('pid','output','started_at','process_identity')}|{'gpu':v['slot'][0]} for k,v in running.items()},
            finished=len(finished),pending=sum(j['id'] not in running and j['id'] not in finished for j in queue)))
        if stopping and not running: break
        time.sleep(5)


if __name__=='__main__': main()
