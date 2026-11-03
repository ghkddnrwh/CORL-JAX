"""Run registered global ensemble rules after an exact single-critic CPU check."""
import argparse
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


def command(job,cpus):
    result=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
        'utils.flow_iql_research.evaluate_independent_critic_policy_v2','--actor-run',job['actor_run'],
        '--critic-runs',*job['critic_runs'],'--step',str(job['step']),'--rule',job['rule'],'--output',job['output']]
    if 'reference' in job:result+=['--reference',job['reference']]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=Path(plan['output_directory']);out.mkdir(exist_ok=True)
    assert not(out/'status.json').exists(),'Inspect existing process handles; do not restart from an observation timeout'
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[16+4*i:20+4*i]for i in range(4)]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    check=plan['self_reference']
    with(out/'self_reference.log').open('a')as log:
        process=subprocess.Popen(command(check,slots[0]),env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic(out/'status.json',dict(pid=os.getpid(),phase='self_reference',updated_at=time.time(),running={check['id']:dict(pid=process.pid,
            process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])}))
        code=process.wait()
    assert code==0,'Single-member self-reference failed; discovery jobs were not started'
    reference=json.loads(Path(check['output']).read_text());assert all(reference['self_reference_episode_arrays_exact'].values())
    queue=list(plan['jobs']);running={};completed=[];failures=[]
    while queue or running:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del running[key]
            (failures if code else completed).append(dict(id=key,exit_code=code,time=time.time()))
        occupied={job['slot']for job in running.values()}
        for slot in range(4):
            if slot in occupied or not queue:continue
            job=queue.pop(0)
            for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
            log=(out/(job['id']+'.log')).open('a');process=subprocess.Popen(command(job,slots[slot]),env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,log=log,slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        atomic(out/'status.json',dict(pid=os.getpid(),phase='running'if queue or running else'complete',updated_at=time.time(),
            completed=completed,failures=failures,pending=len(queue),running={key:dict(pid=job['process'].pid,
                process_identity=job['process_identity'],slot=job['slot'])for key,job in running.items()}))
        if queue or running:time.sleep(5)
    assert not failures,failures
    methods=[]
    for rule in('median_value','mean_rank'):
        tasks=[]
        for environment in plan['environments']:
            jobs=[job for job in plan['jobs']if job['rule']==rule and job['environment']==environment]
            assert sorted(job['step']for job in jobs)==[800000,900000,1000000]
            records=[json.loads(Path(job['output']).read_text())for job in jobs]
            assert all(record['n_episodes']==50 and record['num_independent_learners']==3 for record in records)
            tasks.append(dict(environment=environment,last3_mean=sum(record['success_rate']for record in records)/3,
                per_step={str(record['step']):record['success_rate']for record in records}))
        assert len(tasks)==9
        methods.append(dict(rule=rule,tasks=tasks,nine_task_mean=sum(task['last3_mean']for task in tasks)/9))
    atomic(out/'summary.json',dict(time=time.time(),methods=methods,primary_rule_unchanged='median_rank',
        mechanism_ablation_only=True,one_ensemble_group_only=True,
        training_cost='Sameexistingthreelearners;noadditionaltraining',goal_complete=False))


if __name__=='__main__':main()
