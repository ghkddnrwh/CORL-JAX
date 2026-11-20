"""One low-priority CPU diagnostic worker; never uses GPU learner slots."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return fields[19]if fields[0]!='Z'else None
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('contract',type=Path);args=parser.parse_args()
    contract=json.loads(args.contract.read_text());root=Path(contract['root']);out=root/'analysis/actor_gradient_diagnosis_v1';out.mkdir(exist_ok=True)
    lock=(out/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    running=None;failed={};done={}
    if (out/'status.json').exists():
        previous=json.loads((out/'status.json').read_text());failed=previous['failed'];done=previous['completed']
        saved=previous.get('running')
        if saved and identity(saved['pid'])==saved['process_identity']:
            command=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().split('\0')
            assert any(module in command for module in ('utils.flow_iql_research.diagnose_onestep_gradients_v1','utils.flow_iql_research.diagnose_onestep_gradients_v2','utils.flow_iql_research.diagnose_onestep_gradients_v3','utils.flow_iql_research.diagnose_onestep_gradients_v4')) and saved['output']in command
            running=dict(process=None,log=None,**saved)
    while True:
        for path,digest in contract['source_sha256'].items():assert sha(path)==digest,path
        alive=(running['process'].poll()is None if running['process']is not None else identity(running['pid'])==running['process_identity'])if running else False
        if running and not alive:
            if running['log']:running['log'].close()
            key=running['key'];path=Path(running['output'])
            if path.exists():
                result=json.loads(path.read_text());assert result['optimizer_updates_performed']==0;done[key]=dict(path=str(path),sha256=sha(path))
            else:failed[key]=dict(exit_code=running['process'].returncode if running['process']is not None else None,log=running['log_path'])
            running=None
        pending=[]
        for run in sorted((root/'runs').iterdir()):
            for step in (100000,300000,1000000):
                if not(run/f'checkpoint_{step}.pkl').exists():continue
                key=f'{run.name}:{step}';path=out/f'{run.name}_{step}.json'
                if path.exists():
                    if key not in done:done[key]=dict(path=str(path),sha256=sha(path))
                elif key not in failed and (running is None or running['key']!=key):pending.append((run,step,key,path))
        if running is None and pending:
            run,step,key,path=pending[0];log_path=out/f'{run.name}_{step}.log';log=log_path.open('a')
            command=['nice','-n','15','taskset','-c','36-41',sys.executable,'-u','-m','utils.flow_iql_research.diagnose_onestep_gradients_v4','--run',str(run),'--step',str(step),'--output',str(path)]
            p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running=dict(process=p,log=log,pid=p.pid,process_identity=identity(p.pid),key=key,output=str(path),log_path=str(log_path))
        atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),completed=done,failed=failed,pending=len(pending),
            running={key:running[key]for key in ('key','pid','process_identity','output','log_path')}if running else None,scope='Read-onlyCPUactorgradientdiagnostics;nogpu/learner/modelupdates.'))
        if (root/'STOP').exists()and not running and not pending:return
        time.sleep(20)


if __name__=='__main__':main()
