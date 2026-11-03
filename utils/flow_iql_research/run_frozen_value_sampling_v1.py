"""Execute a fixed18-snapshot value-sampling diagnostic, two CPU workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def audit(plan,job,verifier):
    result=json.loads(Path(job['output']).read_text());run=Path(job['run'])
    config=json.loads((run/'config.json').read_text());identity=config['identity']
    assert identity==job['identity'] and identity['kind']=='ifql' and identity['profile']=='bounded_v2'
    assert result['run']==run.name and result['environment']==identity['environment'] and result['learner_seed']==identity['seed']
    assert result['step']==1000000 and result['initial_V_Adam_count']==2000000
    assert result['kind']=='frozen_Q_value_sampling_equal_optimizer_budget_CPU_diagnostic'
    assert result['source_sha256']==plan['source_sha256']['utils/flow_iql_research/frozen_value_sampling_v1.py']
    assert result['checkpoint_sha256']==job['checkpoint_sha256']==verifier.checkpoint(run/'checkpoint_1000000.pkl',config,1000000)
    assert result['config_sha256']==job['config_sha256']==verifier.digest(run/'config.json')
    assert result['prior_repeated_fit_agreement'] is not None
    assert result['prior_repeated_fit_agreement']['max_objective_abs_difference']<=1e-3
    assert not result['primary_training_modified'] and result['benchmark_evaluations_added']==result['GPU_training_added']==0
    for path,digest in config['source_sha256'].items():assert verifier.digest(path)==digest,path
    assert set(result['modes'])=={'repeated_minibatch','fresh_minibatch'}
    for trace in result['modes'].values():
        assert [t['extra_V_Adam_steps']for t in trace]==[0,1,2,4,8]
        assert all(t['actual_V_Adam_count']==2000000+t['extra_V_Adam_steps']for t in trace)
    assert result['modes']['repeated_minibatch'][:2]==result['modes']['fresh_minibatch'][:2]
    return dict(result_sha256=verifier.digest(job['output']),task=job['task'],seed=identity['seed'],
                modes=result['modes'],query_index_sha256=result['query_index_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path);args=parser.parse_args()
    plan=json.loads(args.registration.read_text());out=Path(plan['output']);out.mkdir(exist_ok=False)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verifier=Verifier();finished={};running={};failures=[]
    for path,digest in plan['source_sha256'].items():assert verifier.digest(path)==digest,path
    queue=[]
    assert len(plan['jobs'])==18
    for job in plan['jobs']:
        if job['reuse_validation']:finished[job['id']]=audit(plan,job,verifier)
        else:queue.append(job)
    slots=plan['CPUslots'];environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    while queue or running:
        for key in list(running):
            worker=running[key]
            if worker['process'].poll() is None:continue
            worker['log'].close()
            if worker['process'].returncode:failures.append(dict(job=key,exit_code=worker['process'].returncode))
            else:finished[key]=audit(plan,worker['job'],verifier)
            del running[key]
        for slot in range(len(slots)):
            if not queue or failures or any(w['slot']==slot for w in running.values()):continue
            job=queue.pop(0)
            for path,digest in plan['source_sha256'].items():assert verifier.digest(path)==digest,path
            command=['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                'utils.flow_iql_research.frozen_value_sampling_v1',job['run'],'--step','1000000','--output',job['output']]
            log=(out/(job['id']+'.log')).open('w');process=subprocess.Popen(command,env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,job=job,log=log,slot=slot,
                identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='failure_requires_inspection'if failures else'running'if queue or running else'complete',
            completed=finished,pending=len(queue),failures=failures,
            running={k:dict(pid=w['process'].pid,process_identity=w['identity'],slot=w['slot'])for k,w in running.items()},
            new_GPU_jobs=0,new_benchmark_evaluations=0,goal_complete=False))
        if failures and not running:return
        if queue or running:time.sleep(5)
    aggregates=[]
    for task in plan['tasks']:
        rows=[row for row in finished.values()if row['task']==task];assert len(rows)==6
        for n in (2,4,8):
            paired=[]
            for row in rows:
                repeated=next(t for t in row['modes']['repeated_minibatch']if t['extra_V_Adam_steps']==n)['sets']
                fresh=next(t for t in row['modes']['fresh_minibatch']if t['extra_V_Adam_steps']==n)['sets']
                paired.append(dict(seed=row['seed'],fresh_minus_repeated={name:fresh[name]['expectile_objective']-repeated[name]['expectile_objective']for name in fresh}))
            aggregates.append(dict(task=task,extra_V_Adam_steps=n,paired_rows=paired,
                mean_fresh_minus_repeated={name:statistics.mean(p['fresh_minus_repeated'][name]for p in paired)for name in ('same_minibatch','independent_train','validation')}))
    atomic(out/'summary.json',dict(time=time.time(),status='complete',rows=finished,aggregates=aggregates,
        registration_sha256=verifier.digest(args.registration),source_sha256=verifier.digest(__file__),
        actual_unique_1M_models_verified=len(verifier.checkpoints),new_GPU_jobs=0,new_benchmark_evaluations=0,
        limits=plan['limits'],goal_complete=False))


if __name__=='__main__':main()
