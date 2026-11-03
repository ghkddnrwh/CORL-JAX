"""Confirm the discovery-selected rank-mean policy on existing blind cohorts.

This controller never writes the GPU queue or alters any learner. The older
median-rank confirmation stays registered as a paired comparator. Both methods
must be reported; there is no per-environment method or seed selection.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.flow_iql_research.audit_independent_critic_policy_v2 import audit, audit_job
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier, provided_thresholds
from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job, summarize
from utils.flow_iql_research.run_independent_critic_factorial_v1 import command


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    registration=json.loads(args.plan.read_text());root=Path(registration['root']);out=Path(registration['output_directory'])
    out.mkdir(parents=True,exist_ok=True);lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not(out/'status.json').exists(),'Inspect actual original controller/child handles before any recovery'
    assert file_sha(registration['training_plan'])==registration['training_plan_sha256']
    training=json.loads(Path(registration['training_plan']).read_text());plan={**training,'output_directory':str(out)}
    parent=Path(registration['factorial_plan']);assert file_sha(parent)==registration['factorial_plan_sha256']
    factorial=json.loads(parent.read_text());verifier=Verifier()
    for source,expected_sha in registration['source_sha256'].items():assert verifier.digest(source)==expected_sha,source
    discovery=audit(factorial,verifier);assert discovery['status']=='complete'and discovery['valid_evaluations']==54 and not discovery['errors']
    selected=next(row for row in discovery['methods']if row['rule']=='mean_rank')
    assert selected['nine_task_mean']==registration['discovery_nine_task_mean']
    assert selected['nine_task_mean']==max(row['nine_task_mean']for row in discovery['methods'])
    assert selected['nine_task_mean']>registration['previous_median_rank_discovery_mean']>.47
    assert registration['registered_before_fresh_policy_outcomes']
    atomic(out/'discovery_audit.json',discovery)
    atomic(out/'preference_lock.json',dict(time=time.time(),selected_rule='mean_rank',selected_nine_task_mean=selected['nine_task_mean'],
        registration_sha256=file_sha(args.plan),previous_median_rank_method_preserved=True,
        no_GPU_training_added=True,selection_inputs='Onlyalready-seendiscoverymodels0/1/2;fourcommonaggregationrulesreported.'))
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[16+4*i:20+4*i]for i in range(4)]
    assert all(len(slot)==4 for slot in slots)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    known={};validated={};running={};failures=[];evidence_plan=dict(source_sha256=factorial['source_sha256'])
    while True:
        for source,expected_sha in registration['source_sha256'].items():assert verifier.digest(source)==expected_sha,source
        thresholds,criterion_hash=provided_thresholds(Path(training['criterion_file']))
        assert criterion_hash==training['criterion_sha256']and max(thresholds.values())==plan['threshold']==.47
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is None:continue
            row['log'].close();del running[key]
            if code:failures.append(dict(id=key,exit_code=code,time=time.time()))
        if failures:
            for row in running.values():row['process'].wait();row['log'].close()
            atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),status='evaluation_failed_requires_inspection',failures=failures,goal_complete=False));return
        for group in range(2):
            for env in plan['environments']:
                for step in(800000,900000,1000000):
                    job=ready_job(root,plan,group,env,step,'mean_rank',verifier)
                    if job is not None:
                        if job['id']in known:assert known[job['id']]==job
                        known[job['id']]=job
        atomic(out/'prepared_evaluation_jobs.json',list(known.values()))
        for key,job in known.items():
            if Path(job['output']).exists():validated[key]=audit_job(evidence_plan,job,verifier)
        occupied={row['slot']for row in running.values()}
        for slot in range(4):
            if slot in occupied:continue
            job=next((job for key,job in known.items()if key not in running and key not in validated),None)
            if job is None:break
            log=(out/(job['id']+'.log')).open('a')
            process=subprocess.Popen(command(job,slots[slot]),env=environment,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,log=log,slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        result=summarize(plan,'mean_rank',known,validated,dict(selected_nine_task_mean=selected['nine_task_mean']))
        atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='fresh_confirmation_complete'if result['all_fresh_complete']else'fresh_confirmation_running',
            **result,ready_evaluations=len(known),valid_evaluations=len(validated),failures=failures,cpu_evaluation_slots=4,
            running={key:dict(pid=row['process'].pid,slot=row['slot'],process_identity=row['process_identity'])for key,row in running.items()},
            registration_sha256=file_sha(args.plan),actual_controller_source_sha256=file_sha(__file__),
            additional_GPU_training=0,original_median_rank_comparator_preserved=True))
        if result['all_fresh_complete']:
            for row in running.values():row['process'].wait();row['log'].close()
            return
        time.sleep(30)


if __name__=='__main__':main()
