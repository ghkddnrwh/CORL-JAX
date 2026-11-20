"""Prospective held-out deployment evaluation; never allocates GPU training.

Observes only MDN training seeds21/22/23 if production later allocates them.
The common decoder is fixed before those runs exist. Nine-task discovery
must pass first. Every held-out checkpoint is evaluated, with no peak or
task-specific decoder choice. Root model-integrity and speed review remain.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from utils.flow_iql_research.watch_native_mixture_decoder_full9_v1 import sha,birth,atomic


def create(root):
    area=root/'analysis/mixture_location_fresh_confirmation_v1';area.mkdir(parents=True,exist_ok=True)
    proposal=area/'prospective_heldout_decoder_plan_v1.json';assert not proposal.exists()
    state_path=root/'analysis/onestep_pipeline_v1/research_state.json';state=json.loads(state_path.read_text())
    attempt=state['attempts'][3]
    assert attempt['recipe']=='mdn9_bc' and attempt['fresh_seeds']==[21,22,23]
    assert not any(j['args']['seed'] in [21,22,23] for j in state['jobs'])
    reg=root/'registration_v17.json';registration=json.loads(reg.read_text())
    audit=root/'analysis/native_mixture_locations_v1/COMPLETE_FOUR_TASK_COMMON_DECODER_ROOT_AUDIT_v1.json'
    assert json.loads(audit.read_text())['status']=='pass'
    replay=root/'analysis/mixture_location_deployment_v1/PZ3_1000K_PORTABLE_CPU50_REPLAY_v1.json'
    assert all(json.loads(replay.read_text())['reference_episode_arrays_exact'].values())
    folder=Path(__file__).resolve().parent
    sources=[Path(__file__),folder/'watch_native_mixture_decoder_full9_v1.py',
        folder/'evaluate_portable_mixture_locations_v1.py',folder/'export_mixture_location_actor_v1.py',
        folder/'standalone_mixture_location_actor_v1.py',folder/'lazy_mixture_location_actor_v1.py',
        folder/'train_mean_empirical_backup_v1.py']
    plan=dict(time=time.time(),root=str(root),schema='prospective_common_decoder_heldout_CPU_v1',
        native_training_recipe='mdn9_bc',deployment_candidate='mdn9_component_location_v1',
        fixed_global_decoder='sampled_component_location',training_seeds=[21,22,23],
        environments=registration['scheduling_amendment']['all_nine_environments'],
        steps=[800000,900000,1000000],episodes=50,first_episode_seed=42,
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        unchanged_thresholds=registration['thresholds'],registration_sha256=sha(reg),
        decoder_selection_evidence=str(audit),decoder_selection_evidence_sha256=sha(audit),
        actual_portable_rollout_replay=str(replay),actual_portable_rollout_replay_sha256=sha(replay),
        original_production_state_sha256=sha(state_path),fresh_runs_allocated_at_freeze=0,
        full_nine_seed0_discovery_required=True,updates_performed_by_this_observer=0,
        no_training_identity_relabelling=True,no_GPU_queue_or_production_state_mutations=True,
        no_task_or_peak_mixing=True,no_formal_noninferiority_claim=True,
        scope='Prospective extra deployment policy over the same future jointly trained MDN models. Allocation remains exclusively with production. Goal requires independent full model and CPU/GPU speed review.')
    with proposal.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False)
    command=['nice','-n','19','taskset','-c','48-71',sys.executable,'-u','-m',
        'utils.flow_iql_research.watch_mixture_location_fresh_CPU_v1','--proposal',str(proposal)]
    with (area/'observer_stdout_v1.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
    atomic(area/'observer_handle_v1.json',dict(time=time.time(),pid=child.pid,birth=birth(child.pid),
        command=command,proposal_sha256=sha(proposal)))
    print(json.dumps(dict(proposal=str(proposal),pid=child.pid,birth=birth(child.pid))))


def observe(proposal):
    plan=json.loads(proposal.read_text());root=Path(plan['root']);area=proposal.parent
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    target=area/'observer_state_v1.json'
    state=json.loads(target.read_text()) if target.exists() else dict(running={},failures={})
    children={}
    while True:
        production=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
        stage='waiting_for_nine_task_discovery'
        discovery=root/'analysis/native_mixture_decoder_full9_v1/observer_state_v1.json'
        d=json.loads(discovery.read_text())
        statuses=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())['rows']
        teacher_rows=[r for r in statuses if r['recipe']=='mdn9_bc' and r['seed']==0 and r['complete']]
        limits=plan['unchanged_thresholds'];can_evaluate=False
        if d['status']=='decoder_evaluations_complete' and len(teacher_rows)==9:
            primary=d['nine_task_means'][plan['fixed_global_decoder']]
            teacher=statistics.mean(r['teacher_mean'] for r in teacher_rows)
            passed=primary>=teacher-limits['mean_loss_tolerance']-1e-12 and primary>=limits['historical_lean_seed0_nine_task_reference']-limits['mean_loss_tolerance']-1e-12
            state['seed0_discovery']=dict(primary_mean=primary,teacher_mean=teacher,passed=passed,
                decoder_evidence_sha256=sha(discovery))
            if not passed:
                state.update(status='location_discovery_failed_no_fresh_evaluation',time=time.time())
                atomic(target,state);return
            can_evaluate=True;stage='waiting_for_production_fresh_allocation'
        jobs=[j for j in production['jobs'] if j['args']['recipe']=='mdn9_bc' and j['args']['seed'] in plan['training_seeds']]
        assert len({(j['args']['seed'],j['args']['env']) for j in jobs})==len(jobs)
        assert all(j['args']['env'] in plan['environments'] for j in jobs)
        complete={};definitions={}
        for job in jobs:
            run=root/'runs'/job['id']
            for step in plan['steps']:
                key=f'{job["id"]}_{step}';output=area/f'{key}_CPU50_v1.json'
                actor=area/f'{key}_component_location_v1.pkl'
                definitions[key]=dict(run=str(run),job=job,step=step,output=str(output),actor=str(actor))
                if output.exists():
                    row=json.loads(output.read_text());seed=job['args']['seed']
                    assert row['run']==job['id'] and row['training_seed']==seed and row['step']==step
                    assert row['environment']==job['args']['env'] and row['n_episodes']==50 and row['first_episode_seed']==42
                    assert row['deployment_rule']==plan['fixed_global_decoder'] and row['optimizer_updates_performed']==0
                    assert row['actor_artifact_sha256']==sha(actor)
                    assert row['native_actor_artifact_sha256']==sha(run/f'actor_{step}.pkl')
                    assert row['config_sha256']==sha(run/'config.json')
                    assert row['learned_model_trees_loaded']==['actor_params'] and row['computed_action_heads']==1
                    assert not row['candidate_action_bank'] and not row['requires_critic'] and not row['requires_BC_flow']
                    assert all(v in (0,1) for v in row['episode_successes'])
                    assert len(row['episode_successes'])==50 and sum(row['episode_successes'])/50==row['success_rate']
                    assert all(len(row[k])==50 and all(math.isfinite(float(x)) for x in row[k]) for k in ['episode_rewards','episode_lengths'])
                    assert all(plan['source_sha256'][p]==digest for p,digest in row['source_sha256'].items())
                    complete[key]=dict(path=str(output),sha256=sha(output),seed=seed,environment=row['environment'],
                                       successes=int(sum(row['episode_successes'])),step=step)
        for key,info in list(state['running'].items()):
            child=children.get(key);live=child.poll() is None if child is not None else birth(info['pid'])==info['birth']
            if not live:
                if key not in complete:state['failures'][key]=info
                del state['running'][key];children.pop(key,None)
        if can_evaluate and not state['failures']:
            for key,definition in definitions.items():
                if key in complete or key in state['running']:continue
                if len(state['running'])==4:break
                native=Path(definition['run'])/f'actor_{definition["step"]}.pkl'
                if not native.exists():continue
                slot=next(x for x in range(4) if x not in {r['slot'] for r in state['running'].values()})
                argv=['nice','-n','19','taskset','-c',f'{48+slot*6}-{53+slot*6}',sys.executable,'-u','-m',
                    'utils.flow_iql_research.evaluate_portable_mixture_locations_v1','--run',definition['run'],
                    '--step',str(definition['step']),'--actor-output',definition['actor'],'--output',definition['output']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
                with (area/f'{key}_stdout_v1.log').open('xb') as log:
                    child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True,close_fds=True)
                children[key]=child;state['running'][key]=dict(pid=child.pid,birth=birth(child.pid),slot=slot,
                    output=definition['output'],command=argv,native_actor_sha256=sha(native))
                atomic(target,state)
            if jobs:stage='waiting_for_or_evaluating_actual_fresh_checkpoints'
        state.update(time=time.time(),pid=os.getpid(),birth=birth(os.getpid()),proposal_sha256=sha(proposal),
            status=stage,actual_GPU_training_jobs_observed=len(jobs),completed_CPU50_points=len(complete),
            expected_CPU50_points=81,completed_points=complete,no_GPU_jobs_allocated_by_observer=True)
        if state['failures']:state['status']='failed_requires_root_review'
        if len(complete)==81 and not state['running']:
            paired=[]
            for seed in plan['training_seeds']:
                rows=[x for x in complete.values() if x['seed']==seed];assert len(rows)==27
                teachers=[r for r in statuses if r['recipe']=='mdn9_bc' and r['seed']==seed and r['complete']]
                if len(teachers)!=9:break
                paired.append(dict(seed=seed,primary_mean=sum(x['successes'] for x in rows)/1350,
                                   teacher_mean=statistics.mean(r['teacher_mean'] for r in teachers)))
            if len(paired)==3:
                primary=statistics.mean(x['primary_mean'] for x in paired);teacher=statistics.mean(x['teacher_mean'] for x in paired)
                gates=dict(fresh_paired_mean=primary>=teacher-limits['mean_loss_tolerance']-1e-12,
                    fresh_historical_mean=primary>=limits['historical_original_fresh_mean_reference']-limits['mean_loss_tolerance']-1e-12)
                state.update(status='performance_passed_pending_root_model_and_speed_review' if all(gates.values()) else 'fresh_performance_failed',
                    seed_means=paired,fresh_mean=primary,fresh_teacher_mean=teacher,
                    fresh_seed_sample_SD=statistics.stdev(x['primary_mean'] for x in paired),unchanged_gates=gates)
        atomic(target,state)
        if state['status'] in ('failed_requires_root_review','performance_passed_pending_root_model_and_speed_review','fresh_performance_failed'):return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--proposal',type=Path)
    a=p.parse_args();observe(a.proposal) if a.proposal else create(a.root.resolve())


if __name__=='__main__':
    main()
