"""Execute four registered archived500K student/reference CPU50 diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=Path(plan['output_directory']);out.mkdir(exist_ok=True)
    assert not(out/'status.json').exists(),'Do not restart a diagnostic manager from an observation timeout'
    queue=list(plan['jobs']);running={};completed=[];failures=[];cpus=sorted(os.sched_getaffinity(0));slots=[cpus[20:24],cpus[24:28]]
    while queue or running:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del running[key]
            (failures if code else completed).append(dict(id=key,exit_code=code,time=time.time()))
        occupied={job['slot']for job in running.values()}
        for slot in range(2):
            if slot in occupied or not queue:continue
            job=queue.pop(0);command=['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                'utils.flow_iql_research.evaluate_rank_flow_teacher_v2',job['run'],'--step','500000',
                '--checkpoint',job['checkpoint'],'--mode',job['mode'],'--episodes','50','--output',job['output']]
            log=(out/(job['id']+'.log')).open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,log=log,slot=slot,process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        atomic(out/'status.json',dict(pid=os.getpid(),updated_at=time.time(),phase='running'if queue or running else'complete',
            completed=completed,failures=failures,pending=[job['id']for job in queue],
            running={key:dict(pid=job['process'].pid,process_identity=job['process_identity'],slot=job['slot'])for key,job in running.items()}))
        if queue or running:time.sleep(5)
    if failures:raise RuntimeError(failures)


if __name__=='__main__':main()
