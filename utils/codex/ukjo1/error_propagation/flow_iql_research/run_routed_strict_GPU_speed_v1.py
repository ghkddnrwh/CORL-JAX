"""Preregister/run a supplemental matched FP32 benchmark after primary success.

Both teacher and actor receive NVIDIA_TF32_OVERRIDE=0 in fresh processes.
The original default-runtime speed results remain separate and unchanged.
No learner/evaluator is interrupted; execution requires the final GPU drain.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def birth(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    assert fields[0]!='Z';return fields[19]


def readiness(root):
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    attempt=state['attempts'][state['active_attempt']]
    checks=dict(expected_attempt=attempt['number']==4 and attempt['recipe']=='mdn9_bc__component_location',
        performance_confirmed=bool(attempt.get('confirmation',{}).get('performance_passed')),
        default_runtime_speed_passed=bool(attempt.get('speed',{}).get('passed')),
        root_review_stage=state['phase']=='success_pending_root_review',
        STOP_present=(root/'STOP').exists(),launch_pause_present=(root/'PAUSE_NEW_LAUNCHES.json').exists(),
        CPU_evaluators_drained=not state['evaluation_workers'],GPU_benchmark_workers_drained=not state['benchmark_workers'],
        learners_drained=not json.loads((root/'scheduler_status.json').read_text())['running'])
    if all(checks.values()):
        checks['GPU_process_query_empty']=not subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--register',action='store_true')
    mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true');a=p.parse_args()
    root=a.root.resolve();area=root/'analysis/routed_strict_GPU_speed_v1';area.mkdir(exist_ok=True)
    proposal=area/'prospective_matched_precision_plan_v1.json'
    benchmark=Path(__file__).resolve().with_name('benchmark_routed_mixture_isolated_v1.py')
    if a.register:
        assert not proposal.exists();registration=root/'registration_v19.json';plan=json.loads(registration.read_text())
        assert plan['routed_deployment_extension']['policy_key']=='mdn9_bc__component_location'
        atomic(proposal,dict(time=time.time(),source_sha256={str(Path(__file__).resolve()):sha(__file__),str(benchmark):sha(benchmark)},
            registration_sha256=sha(registration),thresholds=plan['thresholds'],expected_attempt=4,
            prerequisite='Full9seed0+3freshseed confirmation and original default-runtime speed pass, all GPU/CPU producers drained.',
            NVIDIA_TF32_OVERRIDE='0',common_environment_for_both_policies=True,
            original_default_runtime_results_preserved=True,never_choose_precision_by_environment=True,
            input_paths=[str(root/f'analysis/onestep_pipeline_v1/benchmark_attempt4/GPU{gpu}_input.json')for gpu in [0,1]],
            output_protocol='Same frozen benchmark source and same native/checkpoint/actor hashes, batch1,residentweights,10warm+3x100sync.',
            scope='Supplemental matched full-FP32 deployment timing, preregistered before final nine-task speed measurement. Cannot replace failed default-runtime speed results.'))
        print(json.dumps(dict(status='registered_no_workers_started',proposal=str(proposal))));return
    plan=json.loads(proposal.read_text())
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest
    checks=readiness(root)
    if a.check:
        print(json.dumps(dict(ready=all(checks.values()),checks=checks)));return
    assert all(checks.values()),checks
    invocation=area/'invocation_before_workers_v1.json';assert not invocation.exists()
    jobs=[]
    for gpu in [0,1]:
        input_path=Path(plan['input_paths'][gpu]);assert input_path.exists()
        output=area/f'GPU{gpu}_strict_timings_v1.json';assert not output.exists()
        argv=['taskset','-c',f'{6*gpu}-{6*gpu+5}',sys.executable,'-u','-m',
            'utils.flow_iql_research.benchmark_routed_mixture_isolated_v1','--input',str(input_path),'--output',str(output)]
        jobs.append(dict(gpu=gpu,input=str(input_path),input_sha256=sha(input_path),output=str(output),command=argv))
    atomic(invocation,dict(time=time.time(),proposal_sha256=sha(proposal),admission_checks=checks,
        NVIDIA_TF32_OVERRIDE='0',jobs=jobs,no_training_or_result_changes=True))
    children=[];handles=[]
    for job in jobs:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),NVIDIA_TF32_OVERRIDE='0',
            XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env.pop('JAX_PLATFORM_NAME',None);env.pop('JAX_PLATFORMS',None)
        with (area/f'GPU{job["gpu"]}_stdout_v1.log').open('xb')as log:
            child=subprocess.Popen(job['command'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        handle=dict(pid=child.pid,birth=birth(child.pid),**job);handles.append(handle);children.append(child)
        atomic(area/'worker_handles_v1.json',dict(time=time.time(),workers=handles))
    done=set()
    while len(done)<2:
        for i,child in enumerate(children):
            if i in done or child.poll()is None:continue
            assert child.returncode==0,'Inspect saved worker handle/log; do not restart on observation timeout.'
            done.add(i);print(json.dumps(dict(GPU=jobs[i]['gpu'],status='complete')),flush=True)
        if len(done)<2:time.sleep(.2)
    rows=[]
    for job in jobs:
        data=json.loads(Path(job['output']).read_text());assert data['status']=='complete'and data['source_sha256']==sha(benchmark)
        assert data['input_sha256']==job['input_sha256'];rows+=data['rows']
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    default=[row for data in state['benchmark_results'].values()for row in data['rows']]
    assert len(rows)==len(default)==9 and {r['environment']for r in rows}=={r['environment']for r in default}
    by_run={row['run']:row for row in default}
    for row in rows:
        ref=by_run[row['run']]
        assert row['actor_artifact_sha256']==ref['actor_artifact_sha256']and row['checkpoint_sha256']==ref['checkpoint_sha256']
    ratios={b:math.exp(statistics.mean(math.log(row[f'{b}_latency_ratio'])for row in rows))for b in ['cpu','gpu']}
    limits=plan['thresholds'];passed=ratios['cpu']<=limits['CPU_latency_ratio_max']and ratios['gpu']<=limits['GPU_latency_ratio_max']
    result=dict(time=time.time(),status='complete',rows=rows,geometric_latency_ratios=ratios,
        practical_speed_gates_passed=passed,NVIDIA_TF32_OVERRIDE='0',same_flag_for_teacher_and_actor=True,
        original_default_runtime_results_unchanged=True,invocation_sha256=sha(invocation),proposal_sha256=sha(proposal))
    atomic(area/'root_precision_speed_summary_v1.json',result);print(json.dumps(dict(ratios=ratios,passed=passed)))


if __name__=='__main__':main()
