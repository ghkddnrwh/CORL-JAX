"""Report audited full first fresh group, without promoting discovery scores."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
import time

ENVIRONMENTS=['antmaze-large-navigate-singletask-v0','humanoidmaze-medium-navigate-singletask-v0',
    'humanoidmaze-large-navigate-singletask-v0','antsoccer-arena-navigate-singletask-v0',
    'cube-single-play-singletask-v0','cube-double-play-singletask-v0','puzzle-3x3-play-singletask-v0',
    'puzzle-4x4-play-singletask-v0','scene-play-singletask-v0']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis'
    review=copy.deepcopy(json.loads((out/'method_review_v26.json').read_text()))
    complete=json.loads((out/'fresh_group1_confirmation_summary_v1.json').read_text())
    audit=out/'fresh_group1_root_audit_v1/audit.json';assert sha(audit)==complete['root_audit_sha256']
    evidence=json.loads(audit.read_text());assert not evidence['errors']
    assert complete['actual_first_group_models_verified']==81
    states={}
    for name,path in [('mean','independent_critic_rank_mean_confirmation_v1/status.json'),
                      ('median','independent_critic_fresh_confirmation_v1/status.json'),
                      ('raw','independent_critic_raw_mean_control_v1/status.json')]:
        states[name]=json.loads((out/path).read_text())
    for i,group in enumerate(evidence['groups'],1):
        for method,labelpart in [('mean_rank','mean_rank'),('median_rank','median_rank'),('mean','raw_mean_control')]:
            row=next(m for m in review['methods'] if m['label'].startswith(f'Independent3 {labelpart} fresh group{i} '))
            mapped={t['environment']:t['ensemble_policies'][method] for t in group['tasks']}
            row['values']=[mapped[e]['last3_mean'] if mapped[e]['complete'] else None for e in ENVIRONMENTS]
            row['completed_tasks']=sum(v is not None for v in row['values'])
            row['nine_task_mean']=statistics.mean(row['values']) if row['completed_tasks']==9 else None
            row['evidence']=[dict(environment=e,**mapped[e]) for e in ENVIRONMENTS]
            row['actual_model_audit']='analysis/fresh_group1_root_audit_v1/audit.json'
        for method,label in [('first_BC_own_single_critic','First BC own critic'),
                             ('uniform_member_expected_return','Uniform member expected return')]:
            mapped={t['environment']:t['baselines'][method]for t in group['tasks']}
            values=[mapped[e]['last3_mean']if mapped[e]['complete']else None for e in ENVIRONMENTS]
            review['methods'].append(dict(label=f'{label} fresh group{i} (paired baseline)',
                training_seeds=group['learner_seeds'],values=values,completed_tasks=sum(v is not None for v in values),
                nine_task_mean=statistics.mean(values)if all(v is not None for v in values)else None,
                evidence=[dict(environment=e,**mapped[e])for e in ENVIRONMENTS],
                source_role='Fixed original BCfirst or uniform learner draw; no best-member selection.'))
    status=json.loads((root/'scheduler_status.json').read_text());workers=[]
    for run,handle in status['running'].items():
        pid=handle['pid'];proc=Path(f'/proc/{pid}')
        identity=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
        command=proc.joinpath('cmdline').read_bytes().decode().split('\0')
        env=dict(x.split('=',1)for x in proc.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
        assert identity==handle['process_identity'] and '-m'in command
        assert env['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1' and env['CUDA_VISIBLE_DEVICES']==str(handle['gpu'])
        runpath=Path(handle['output']);conf=json.loads((runpath/'config.json').read_text())
        for p,d in conf['source_sha256'].items():assert sha(p)==d,p
        probes=[json.loads(x)for x in (runpath/'diagnostics.jsonl').read_text().splitlines()] if(runpath/'diagnostics.jsonl').exists()else []
        native=conf['identity']['profile']=='official'
        step=None;s=None
        if(runpath/'latest.pkl').exists():
            with(runpath/'latest.pkl').open('rb')as stream:saved=pickle.load(stream)
            s=saved['state'];step=saved['step'];assert int(s['q_updates'])==int(s['agent']['network']['opt_state']['0']['count'])==step
            assert int(s['v_updates'])==(1 if native else 2)*step
            assert int(s['v_opt']['0']['count'])==(0 if native else 2)*step
        workers.append(dict(run=run,**handle,actual_process_identity=identity,module=command[command.index('-m')+1],
            last_probe_step=probes[-1]['step']if probes else None,last_probe_finite=probes[-1]['finite']if probes else None,
            source_pins_exact=True,actual_saved_step=step,Q_Adam=step,V_total=None if s is None else int(s['v_updates']),
            separate_V_Adam=None if s is None else int(s['v_opt']['0']['count']),native_official=native,
            saved_clock_audited=s is not None,memory_fraction=env['XLA_PYTHON_CLIENT_MEM_FRACTION']))
    per_gpu={str(g):sum(w['gpu']==g for w in workers)for g in (0,1)}
    assert len(workers)==16 and per_gpu=={'0':8,'1':8}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    assert {int(line.split(',')[0])for line in apps.splitlines()}=={w['pid']for w in workers}
    runtime=dict(time=time.time(),scheduler_pid=status['pid'],scheduler_finished=status['finished'],
        scheduler_pending=status['pending'],actual_GPU_learners=16,per_GPU=per_gpu,workers=workers,
        source_sha256=sha(__file__),nvidia_compute_apps=apps,free_NVMe_bytes=shutil.disk_usage(root).free,
        queue_size=len(json.loads((root/'queue.json').read_text())),goal_complete=False)
    write_new(out/'runtime_audit_v28.json',runtime)
    snapshot=out/'verification_status_snapshot_v27.json';write_new(snapshot,dict(time=time.time(),**states))
    review.update(time=time.time(),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='first_fresh_full9_below_FQL_second_fresh_pending_continue',goal_complete=False,runtime=runtime,
        point_in_time_confirmation_status_snapshot=dict(path=str(snapshot),sha256=sha(snapshot)),
        new_findings=[
            'Root audit verified all27first-group parents,81distinct late checkpoints and81ensemble CPU50 records; no errors.',
            'Discovery mean-rank.5459259 became first fresh full9.3481481. The benchmark/replication goal is not achieved.',
            'First fresh ownBC=.2977778,uniformmember=.3202469,rawmean=.3407407,medianrank=.3392593; meanrank gains only.0074074 over rawmean.',
            'Second group Pz3 ownBC6=.4733333,meanrank=.0533333. A critic ensemble can suppress a useful individual policy.',
            'Calibration/query choice mechanics persist across discovery/fresh but do not certify true action ranking or returns.',
            'Tau9/99 x repeated/fresh three-task pilots plus native tau99 controls remain exploratory; newlyselected methods need seeds>=9.',
            'Registered secondary nestedN99 before any high-tau late outcome; primary N9 unchanged. Ideal expectile odds motivate it, not a practical guarantee.',
            'Existing expectile-sign flow pilots already have frozen300K checkpoints; complete those exact recipes rather than add a duplicate variant.',
            'A frozen historical file-name collision was immediately restored SHA-exact; all214run source manifests pass. No active learner imported that module during interval.'])
    review['preferred_candidate'].update(first_fresh_nine_task_mean=complete['nine_task_means']['mean_rank'],
        confirmation_status='first_full9_not_replicated;second_group_pending',fresh_pooled_mean=None,
        required_second_group_mean_for_pooled_fresh_pass=complete['required_second_group_mean_for_pooled_fresh_pass'])
    review['confirmations']['first_fresh_complete']=complete
    review['confirmations']['live_status_snapshot']=states
    review['high_tau']=dict(registration='high_tau_value_sampling_registration_v1.json',new_jobs=9,
        proposed=6,native=3,expectile=.99,discount=.999,primary_samples=9,
        secondary_registration='tau_matched_extraction_registration_v1.json',secondary_samples=99,
        analysis_registration='tau_sampling_factorial_analysis_registration_v1.json',late_outcomes_available=False)
    review['existing_expectile_flow_completion']=dict(registration='existing_expectile_flow_completion_registration_v1.json',
        tasks=3,new_GPU_jobs=0,seen_early_outcomes=True,unseen_late_outcomes=True,
        scheduling_amendment='scheduling_amendment_20261028_v2.json')
    for key,path in {
        'first_fresh_full9':'analysis/fresh_group1_confirmation_summary_v1.json',
        'first_fresh_root_audit':'analysis/fresh_group1_root_audit_v1/audit.json',
        'rank_calibration_cohort_comparison':'analysis/rank_calibration_discovery_vs_first_fresh_v1.json',
        'tau_matched_registration':'tau_matched_extraction_registration_v1.json',
        'tau_matched_actual_validation':'analysis/tau_matched_actual_model_validation_v1.json',
        'tau_matched_oracle_validation':'analysis/tau_matched_oracle_theory_validation_v1.json',
        'tau_sampling_factorial_registration':'tau_sampling_factorial_analysis_registration_v1.json',
        'existing_expectile_flow_registration':'existing_expectile_flow_completion_registration_v1.json',
        'scheduling_amendment_v2':'scheduling_amendment_20261028_v2.json',
        'all_source_manifests_restored':'analysis/all_run_source_manifests_after_restoration_20261028_v1.json',
        'runtime_v28':'analysis/runtime_audit_v28.json'}.items():
        review['evidence'][key]=dict(path=path,sha256=sha(root/path))
    review['historical_review']='analysis/method_review_v26.json'
    target=out/'method_review_v27.json';write_new(target,review)
    with(out/'common9_comparison_v15.csv').open('x',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*ENVIRONMENTS,'completed_tasks','nine_task_mean'])
        for row in review['methods']:writer.writerow([row['label'],*row['values'],row['completed_tasks'],row['nine_task_mean']])
    continuation=json.loads((root/'CONTINUATION.json').read_text())
    continuation.update(updated_at=time.time(),status=review['status'],latest_review='analysis/method_review_v27.json',
        latest_comparison='analysis/common9_comparison_v15.csv',goal_complete=False,
        last_goal_turn_classification='progress',last_goal_turn_evidence='analysis/tau_matched_extraction_launch_v1.json')
    continuation['preferred_candidate'].update(current_ready_evals=states['mean']['ready_evaluations'],
        current_valid_evals=states['mean']['valid_evaluations'],first_fresh_nine_task_mean=.34814814814814815)
    continuation['GPU_queue'].update(counts={k:status[k]for k in ('finished','pending')},running_handles=status['running'],
        queue_size=runtime['queue_size'],priority='Roundrobin sampling_tau9,tau99,existing_expectile_flow,confirmation,mean_backup under ORDERONLYv2')
    continuation['GPU_queue']['counts']['running']=16
    for name,file in [('tau_matched_extraction','tau_matched_extraction_launch_v1.json'),
                      ('existing_expectile_flow_completion','existing_expectile_flow_completion_launch_v1.json'),
                      ('tau_sampling_factorial_audit','tau_sampling_factorial_audit_launch_v1.json')]:
        launch=json.loads((out/file).read_text());continuation[name]=launch
        continuation['controllers'][name]=launch['pid']
    continuation['scheduling_amendment'].update(registration='scheduling_amendment_20261028_v2.json',
        buckets=['sampling_tau9','tau99','existing_expectile_flow','confirmation','mean_backup'])
    continuation['source_restoration']='analysis/all_run_source_manifests_after_restoration_20261028_v1.json'
    continuation['unused_draft']='expectile_sign_flow_core_v2.py: unregistered/unlaunched; existing v1 is being completed'
    temporary=root/'CONTINUATION.tmp';temporary.write_text(json.dumps(continuation,indent=2,allow_nan=False));temporary.replace(root/'CONTINUATION.json')
    print(json.dumps(dict(report=str(target),first_fresh=.34814814814814815,workers=16,per_GPU=per_gpu)),flush=True)


if __name__=='__main__':main()
