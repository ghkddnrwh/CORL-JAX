"""Audit one global mean-head inference rule as parent models finish."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());root=Path(plan['root']);out=Path(plan['output_directory']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verifier=Verifier();running={};failures=json.loads((out/'failures.json').read_text())if(out/'failures.json').exists()else{}
    slots=['52-55','56-59']
    while True:
        for source,digest in plan['source_sha256'].items():assert verifier.digest(source)==digest,source
        for name,item in list(running.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();del running[name]
            if code!=0:
                failures[name]=dict(exit_code=code,time=time.time());atomic(out/'failures.json',failures)
            print('FINISH',name,code,flush=True)
        valid={};pending=[]
        for job in plan['jobs']:
            run=Path(job['run']);config=json.loads((run/'config.json').read_text())
            assert verifier.digest(run/'config.json')==job['config_sha256']
            assert config['identity']==job['identity']
            for source,digest in config['source_sha256'].items():assert verifier.digest(source)==digest,source
            path=Path(job['output'])
            if path.exists():
                result=json.loads(path.read_text());assert result['head_aggregation']=='mean'
                assert result['step']==job['step']and result['training_seed']==0
                assert result['selection_samples']==9 and result['critic_members']==1
                assert result['environment']==job['identity']['environment']
                assert result['n_episodes']==50 and result['first_episode_seed']==42 and result['eval_device']=='cpu'
                assert result['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
                for field in ('episode_successes','episode_rewards','episode_lengths'):
                    assert len(result[field])==50 and np.isfinite(result[field]).all()
                assert set(result['episode_successes'])<={0.,1.}
                assert abs(statistics.mean(result['episode_successes'])-result['success_rate'])<1e-12
                assert result['config_sha256']==job['config_sha256']
                assert result['checkpoint_sha256']==verifier.checkpoint(run/f"checkpoint_{job['step']}.pkl",config,job['step'])
                assert result['selector_source_sha256']==plan['source_sha256']['utils/flow_iql_research/head_aggregation_policy_v1.py']
                assert result['evaluator_source_sha256']==plan['source_sha256']['utils/flow_iql_research/evaluate_head_aggregation_v1.py']
                valid[job['id']]=result
            elif job['id']not in running and job['id']not in failures and(run/f"checkpoint_{job['step']}.pkl").exists():
                pending.append(job)
        groups=[]
        for group in plan['groups']:
            rows=[]
            for environment in group['environments']:
                jobs=[job for job in plan['jobs']if job['group']==group['id']and job['primary']and job['identity']['environment']==environment]
                assert len(jobs)==3 and{job['step']for job in jobs}=={800000,900000,1000000}
                run=Path(jobs[0]['run']);parent=False
                if(run/'completed.json').exists():
                    done=json.loads((run/'completed.json').read_text());parent=done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
                complete=parent and all(job['id']in valid for job in jobs)
                rows.append(dict(environment=environment,run=run.name,parent_complete=parent,
                    per_step={str(job['step']):valid[job['id']]['success_rate']if job['id']in valid else None for job in jobs},
                    last3_mean=statistics.mean(valid[job['id']]['success_rate']for job in jobs)if complete else None))
            count=sum(row['last3_mean']is not None for row in rows)
            macro=statistics.mean(row['last3_mean']for row in rows)if len(rows)==count==9 else None
            groups.append(dict(id=group['id'],rows=rows,completed_tasks=count,registered_tasks=len(rows),nine_task_mean=macro,
                               exceeds_provided_plain_FQL=macro>.47 if macro is not None else None))
        early=[dict(job=job['id'],step=job['step'],success=valid[job['id']]['success_rate'])for job in plan['jobs']if not job['primary']and job['id']in valid]
        atomic(out/'last3_audit.json',dict(time=time.time(),groups=groups,early_exploratory_only=early,
            inference_rule='mean_twin_online_Q_best9_current_BC',training_seed=0,
            plan_sha256=verifier.digest(args.plan),goal_complete=False,
            limits=['No parent primary score is replaced; mean-head variants remain separate methods.',
                    'Three mean-backup pilot tasks cannot form a nine-task score.',
                    'All seed0 inference searches are exploratory; a qualified recipe needs fresh independent training seeds.']))
        occupied={item['slot']for item in running.values()};stopping=(out/'STOP').exists()
        if not stopping:
            for slot,cpus in enumerate(slots):
                if slot in occupied or not pending:continue
                job=pending.pop(0);path=Path(job['output']);path.parent.mkdir(parents=True,exist_ok=True)
                command=['taskset','-c',cpus,sys.executable,'-u','-m','utils.flow_iql_research.evaluate_head_aggregation_v1',job['run'],'--step',str(job['step']),'--aggregation','mean','--episodes','50','--output',str(path)]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
                log=path.with_suffix('.log').open('a');process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                running[job['id']]=dict(process=process,log=log,slot=slot,started_at=time.time());print('START',job['id'],process.pid,flush=True)
        atomic(out/'status.json',dict(pid=os.getpid(),time=time.time(),running={name:dict(pid=item['process'].pid,slot=item['slot'],started_at=item['started_at'])for name,item in running.items()},
            completed=len(valid),pending=len(pending),failures=failures,registered=len(plan['jobs'])))
        if not running and(stopping or len(valid)==len(plan['jobs'])):return
        time.sleep(15)


if __name__=='__main__':main()
