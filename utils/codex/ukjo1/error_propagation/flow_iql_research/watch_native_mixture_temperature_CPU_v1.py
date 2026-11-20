"""Private global variance calibration; never mutates production training.

All nine native seed0 MDN learners, all800K/900K/1M CPU50 points, and the
same three temperatures. Four low-priority CPU evaluators refill naturally.
The formal zero-temperature candidate and its fresh-seed gates continue.
"""
import argparse
import fcntl
from fractions import Fraction
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
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19]if fields[0]!='Z'else None
    except(FileNotFoundError,ProcessLookupError):return None


def verify(path,job,plan):
    row=json.loads(path.read_text())
    assert row['protocol']=='posthoc_native_mixture_global_temperature_CPU50_v1'
    assert row['run']==job['run_id']and row['step']==job['step']and row['temperature']==job['temperature']
    assert row['environment']==job['environment']and row['training_seed']==0
    assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
    assert row['optimizer_updates_performed']==0 and row['learned_model_trees_loaded']==['actor_params']
    assert row['computed_action_heads']==row['realized_actions']==1
    assert not row['candidate_action_bank']and not row['critic_loaded']and not row['Flow_loaded']
    assert not row['counts_toward_production_primary']and not row['production_state_or_scores_modified']
    assert row['actor_artifact_sha256']==sha(job['actor'])
    assert row['config_sha256']==sha(Path(job['run'])/'config.json')
    assert row['source_sha256']==plan['evaluation_source_sha256']
    for key in ['episode_successes','episode_rewards','episode_lengths']:
        assert len(row[key])==50 and all(math.isfinite(float(x))for x in row[key])
    assert all(x in [0,1]for x in row['episode_successes'])
    successes=int(sum(row['episode_successes']));assert successes/50==row['success_rate']
    assert abs(statistics.mean(row['episode_rewards'])-row['reward_mean'])<1e-8
    return dict(successes=successes,episodes=50,rate=successes/50,path=str(path),sha256=sha(path))


def create(root):
    area=root/'analysis/native_mixture_temperature_v1';area.mkdir(exist_ok=True)
    proposal=area/'prospective_global_temperature_plan_v1.json';assert not proposal.exists()
    proof=area/'implementation_high_precision_preflight_v1.json';validation=json.loads(proof.read_text())
    assert validation['status']=='pass'and validation['production_remains_FP32']
    assert validation['zero_temperature_exact_to_frozen_center']
    for path,digest in validation['source_sha256'].items():assert sha(path)==digest
    acceptance=area/'ROOT_PREFLIGHT_ACCEPTANCE_v1.json';accepted=json.loads(acceptance.read_text())
    assert accepted['status']=='pass'and accepted['preflight_sha256']==sha(proof)
    assert accepted['max_new_temperature_FP32_error']<1e-5
    registration=root/'registration_v19.json';registered=json.loads(registration.read_text())
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert not any(j['args']['seed']in[24,25,26]for j in state['jobs'])
    jobs=[j for j in state['jobs']if j['args']['recipe']=='mdn9_bc'and j['args']['seed']==0]
    assert len(jobs)==len({j['args']['env']for j in jobs})==9
    priority=['hmmed','cubed','pz3','cube','ant','hmlarge','soccer','pz4','scene']
    jobs.sort(key=lambda j:priority.index(j['id'].split('_')[1]))
    evaluations=[]
    for job in jobs:
        run=root/'runs'/job['id']
        for step in [800000,900000,1000000]:
            for temperature,tag in [(.25,'t025'),(.5,'t050'),(1.,'t100')]:
                evaluations.append(dict(run=str(run),run_id=job['id'],environment=job['args']['env'],
                    step=step,temperature=temperature,tag=tag,actor=str(run/f'actor_{step}.pkl'),
                    output=str(area/f'{job["id"]}_{step}_{tag}_CPU50_v1.json')))
    folder=Path(__file__).resolve().parent
    names=['watch_native_mixture_temperature_CPU_v1.py','lazy_tempered_mixture_actor_v1.py',
        'evaluate_tempered_mixture_CPU_v1.py','validate_tempered_mixture_actor_v1.py',
        'validate_tempered_mixture_actor_v2.py','validate_tempered_mixture_actor_v3.py',
        'lazy_mixture_location_actor_v1.py','train_mean_empirical_backup_v1.py']
    sources={str(folder/name):sha(folder/name)for name in names}
    evaluation={str(folder/name):sha(folder/name)for name in ['evaluate_tempered_mixture_CPU_v1.py',
        'lazy_tempered_mixture_actor_v1.py','train_mean_empirical_backup_v1.py']}
    plan=dict(time=time.time(),root=str(root),schema='private_global_temperature_CPU50_v1',
        source_sha256=sources,evaluation_source_sha256=evaluation,
        registration_sha256=sha(registration),thresholds=registered['thresholds'],
        expected_native_identities={j['id']:state['identities'][j['id']]for j in jobs},
        native_training_source_sha256=registered['implementation_training_source_sha256']['mixture'],
        implementation_validation=str(proof),implementation_validation_sha256=sha(proof),
        root_implementation_acceptance=str(acceptance),root_implementation_acceptance_sha256=sha(acceptance),
        earlier_FP32_reference_failures_preserved=True,production_inference_dtype='float32',
        temperatures=[.25,.5,1.],evaluations=evaluations,expected_CPU50_points=81,
        max_CPU_workers=4,CPU_affinity_slots=[list(range(48+6*i,54+6*i))for i in range(4)],
        no_training_or_primary_result_changes=True,no_GPU_queue_mutations=True,
        existing_formal_center_candidate_continues=True,
        no_per_environment_or_checkpoint_temperature_selection=True,
        zero_temperature_control='Existing formal component_location role, reused without overwrite.',
        temperature_one_control='Selected mean+scale head implementation, compared to original full-output MDN sampler to expose numerical-trajectory differences.',
        fresh_training_jobs_at_registration=0,
        heldout_scope='Seed0 discovery only. Any newly selected deployment rule requires a versioned confirmation before using fresh outcomes; observer never allocates or changes training.',
        rationale='Keep mixture-component diversity; calibrate within-component noise with two common intermediate strengths. HMM800K center.68 versus sampler.72 is motivation, not a causal or complete-last3 conclusion.',
        priority_scope='HMM available points first for diagnosis; all9tasks receive identical options and all3late checkpoints.')
    with proposal.open('x')as f:json.dump(plan,f,indent=2,allow_nan=False)
    argv=['nice','-n','19','taskset','-c','48-71',sys.executable,'-u','-m',
        'utils.flow_iql_research.watch_native_mixture_temperature_CPU_v1','--proposal',str(proposal)]
    with(area/'observer_stdout_v1.log').open('xb')as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    handle=dict(time=time.time(),pid=child.pid,birth=birth(child.pid),command=argv,proposal_sha256=sha(proposal))
    atomic(area/'observer_handle_v1.json',handle);print(json.dumps(handle))


def observe(proposal):
    plan=json.loads(proposal.read_text());root=Path(plan['root']);area=proposal.parent
    lock=(area/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=area/'observer_state_v1.json'
    state=json.loads(target.read_text())if target.exists()else dict(running={},failures={})
    children={}
    while True:
        for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
        complete={}
        for job in plan['evaluations']:
            output=Path(job['output'])
            if output.exists():complete[str(output)]=verify(output,job,plan)
        for key,info in list(state['running'].items()):
            child=children.get(key)
            live=child.poll()is None if child else birth(info['pid'])==info['birth']
            if not live:
                if key not in complete:state['failures'][key]=dict(time=time.time(),**info)
                del state['running'][key];children.pop(key,None)
        if not state['failures']:
            for job in plan['evaluations']:
                key=job['output']
                if key in complete or key in state['running']:continue
                if len(state['running'])>=4:break
                run=Path(job['run'])
                if not Path(job['actor']).exists()or not(run/f'checkpoint_{job["step"]}.pkl').exists():continue
                conf=json.loads((run/'config.json').read_text())
                assert conf['identity']==plan['expected_native_identities'][job['run_id']]
                assert conf['source_sha256']==plan['native_training_source_sha256']
                slot=next(i for i in range(4)if i not in {r['slot']for r in state['running'].values()})
                argv=['nice','-n','19','taskset','-c',','.join(map(str,plan['CPU_affinity_slots'][slot])),
                    sys.executable,'-u','-m','utils.flow_iql_research.evaluate_tempered_mixture_CPU_v1',
                    '--run',job['run'],'--step',str(job['step']),'--temperature',str(job['temperature']),'--output',key]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
                logpath=area/f'{job["run_id"]}_{job["step"]}_{job["tag"]}_stdout_v1.log'
                with logpath.open('xb')as log:
                    child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,close_fds=True)
                children[key]=child
                state['running'][key]=dict(pid=child.pid,birth=birth(child.pid),slot=slot,output=key,
                    command=argv,actor_sha256=sha(job['actor']))
                atomic(target,state)
        rows=[]
        for rid in plan['expected_native_identities']:
            for temperature in plan['temperatures']:
                group=[j for j in plan['evaluations']if j['run_id']==rid and j['temperature']==temperature]
                points=[complete[j['output']]for j in group if j['output']in complete]
                row=dict(run=rid,environment=group[0]['environment'],temperature=temperature,
                    observed_points=len(points),complete=len(points)==3,points=points)
                if row['complete']:row['last3_mean']=float(Fraction(sum(p['successes']for p in points),150))
                rows.append(row)
        state.update(time=time.time(),pid=os.getpid(),birth=birth(os.getpid()),proposal_sha256=sha(proposal),
            complete_points=len(complete),expected_points=81,rows=rows,status='waiting_or_evaluating',
            optimizer_updates_performed=0,production_GPU_jobs_or_state_modified=False)
        if state['failures']:state['status']='failed_requires_root_review'
        if len(complete)==81 and not state['running']:
            state['temperature_nine_task_means']={str(t):float(Fraction(sum(p['successes']for j in plan['evaluations']
                if j['temperature']==t for p in [complete[j['output']]]),1350))for t in plan['temperatures']}
            state['status']='all_temperature_points_complete_pending_root_review'
        atomic(target,state)
        if state['status']in ['failed_requires_root_review','all_temperature_points_complete_pending_root_review']:return
        time.sleep(15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--proposal',type=Path);a=p.parse_args()
    if a.proposal:observe(a.proposal)
    else:create(a.root.resolve())


if __name__=='__main__':main()
