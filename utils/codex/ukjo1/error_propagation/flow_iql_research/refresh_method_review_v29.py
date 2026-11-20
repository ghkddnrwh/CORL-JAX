"""Refresh audited last-three results without turning partial tasks into means."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
import time
from utils.flow_iql_research.refresh_method_review_v27 import ENVIRONMENTS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis';review=json.loads((out/'method_review_v28.json').read_text())
    evidence=json.loads((out/'fresh_confirmation_root_audit_v3/audit.json').read_text());assert not evidence['errors']
    for i,group in enumerate(evidence['groups'],1):
        for method,label in [('mean_rank','mean_rank'),('median_rank','median_rank'),('mean','raw_mean_control')]:
            row=next(m for m in review['methods']if m['label'].startswith(f'Independent3 {label} fresh group{i} '))
            data={t['environment']:t['ensemble_policies'][method]for t in group['tasks']}
            row.update(values=[data[e]['last3_mean']if data[e]['complete']else None for e in ENVIRONMENTS],
                evidence=[dict(environment=e,**data[e])for e in ENVIRONMENTS],actual_model_audit='analysis/fresh_confirmation_root_audit_v3/audit.json')
            row['completed_tasks']=sum(v is not None for v in row['values'])
            row['nine_task_mean']=statistics.mean(row['values'])if row['completed_tasks']==9 else None
        for method,label in [('first_BC_own_single_critic','First BC own critic'),('uniform_member_expected_return','Uniform member expected return')]:
            row=next(m for m in review['methods']if m['label']==f'{label} fresh group{i} (paired baseline)')
            data={t['environment']:t['baselines'][method]for t in group['tasks']}
            row.update(values=[data[e]['last3_mean']if data[e]['complete']else None for e in ENVIRONMENTS],
                evidence=[dict(environment=e,**data[e])for e in ENVIRONMENTS])
            row['completed_tasks']=sum(v is not None for v in row['values'])
            row['nine_task_mean']=statistics.mean(row['values'])if row['completed_tasks']==9 else None
    native=json.loads((out/'native_IFQL_N9_reference_root_audit_v1.json').read_text())
    assert native['status']=='complete'and native['actual_unique_models_verified']==54
    data={t['environment']:t['native_N9_last3']for t in native['rows']}
    review['methods'].append(dict(label='Native IFQL, best9 (s0)',values=[data[e]for e in ENVIRONMENTS],
        completed_tasks=9,nine_task_mean=native['native_N9_nine_task_mean'],evidence=native['rows'],
        actual_model_audit='analysis/native_IFQL_N9_reference_root_audit_v1.json',
        limits='Same proposal count as V2; actual own BC weights differ, so this is not a pure value-only contrast.'))
    mean=json.loads((out/'mean_backup_full9_progress_audit_v2.json').read_text());data={r['environment']:r for r in mean['rows']}
    review['methods'].append(dict(label='Mean-head finite expectile backup, best9 (s0)',
        values=[data[e]['last3_mean']if data[e]['complete']else None for e in ENVIRONMENTS],
        completed_tasks=mean['completed_tasks'],nine_task_mean=mean['nine_task_mean'],evidence=mean['rows'],
        actual_model_audit='analysis/mean_backup_full9_progress_audit_v2.json'))
    status=json.loads((root/'scheduler_status.json').read_text());workers=[]
    for rid,h in status['running'].items():
        proc=Path(f"/proc/{h['pid']}");identity=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
        assert identity==h['process_identity']
        cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0');assert cmd[cmd.index('--output')+1]==h['output']
        environment=dict(x.split('=',1)for x in proc.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
        assert environment['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'and environment['CUDA_VISIBLE_DEVICES']==str(h['gpu'])
        run=Path(h['output']);conf=json.loads((run/'config.json').read_text())
        for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
        probes=[json.loads(x)for x in(run/'diagnostics.jsonl').read_text().splitlines()]if(run/'diagnostics.jsonl').exists()else[]
        actual=None
        if(run/'latest.pkl').exists():
            with(run/'latest.pkl').open('rb')as stream:saved=pickle.load(stream)
            state=saved['state'];step=saved['step'];official=conf['identity']['profile']=='official'
            assert int(state['q_updates'])==int(state['agent']['network']['opt_state']['0']['count'])==step
            assert int(state['v_updates'])==(1 if official else 2)*step
            assert int(state['v_opt']['0']['count'])==(0 if official else 2)*step
            actual=dict(step=step,Q_Adam=step,V_total=int(state['v_updates']),separate_V_Adam=int(state['v_opt']['0']['count']))
        workers.append(dict(run=rid,**h,module=cmd[cmd.index('-m')+1],source_pins_exact=True,
            last_probe_step=probes[-1]['step']if probes else None,last_probe_finite=probes[-1]['finite']if probes else None,
            actual_saved_clocks=actual))
    assert len(workers)==16 and collections.Counter(w['gpu']for w in workers)=={0:8,1:8}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    assert {int(line.split(',')[0])for line in apps.splitlines()}<={w['pid']for w in workers}
    runtime=dict(time=time.time(),scheduler_pid=status['pid'],scheduler_module='utils.flow_iql_research.scheduler_v20',
        scheduler_finished=status['finished'],scheduler_pending=status['pending'],actual_learner_processes=16,
        per_GPU={'0':8,'1':8},workers=workers,nvidia_compute_apps=apps,
        transition_note='Fresh startup or CPU evaluation can temporarily precede/leave CUDA context ownership.',
        free_NVMe_bytes=shutil.disk_usage(root).free,queue_size=len(json.loads((root/'queue.json').read_text())),goal_complete=False)
    create(out/'runtime_audit_v30.json',runtime)
    states={}
    for key,path in [('rank_mean','independent_critic_rank_mean_confirmation_v1'),('native_N9','native_ifql_N9_reference_v1'),
        ('common_BC','common_BC_value_ablation_v1'),('unit_return','unit_return_scoring_pilots_v1'),
        ('unit_fitting','unit_value_fitting_diagnostic_v1'),('sampling','value_minibatch_reuse_audit_v1'),
        ('tau99','tau_sampling_factorial_audit_v1'),('mean_confirmation','mean_backup_independent_confirmation_v1')]:
        p=out/path/'status.json'
        if p.exists():states[key]=json.loads(p.read_text())
    create(out/'verification_status_snapshot_v29.json',dict(time=time.time(),states=states))
    review.update(time=time.time(),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='matched_N9_reference_complete_common_BC_ablation_running_fresh_confirmation_below_target',
        runtime=runtime,goal_complete=False,historical_review='analysis/method_review_v28.json',
        point_in_time_confirmation_status_snapshot=dict(path=str(out/'verification_status_snapshot_v29.json'),sha256=sha(out/'verification_status_snapshot_v29.json')))
    review['confirmations']['latest_root_audit']=dict(completed_members=evidence['completed_members'],
        actual_unique_late_models_verified=evidence['actual_unique_late_models_verified'],errors=evidence['errors'])
    review['confirmations']['live_status_snapshot']=states
    review['native_N9_reference']=dict(actual_model_audit='analysis/native_IFQL_N9_reference_root_audit_v1.json',
        native_nine_task_mean=native['native_N9_nine_task_mean'],V2_nine_task_mean=native['V2_N9_nine_task_mean'],
        all_actual_BC_parameters_equal=native['all_BC_parameters_exact'],common_BC_registration='common_BC_value_ablation_registration_v1.json',
        cause_of_BC_weight_difference='Not established. Short arithmetic probes show graph-dependent rounding; they do not prove the historical cause.')
    review['unit_return_scoring_pilots'].update(current_status_snapshot='analysis/verification_status_snapshot_v29.json',
        new_diagnostic='unit_value_fitting_diagnostic_registration_v1.json',
        independently_recomputed_metrics=252,observed_diagnostic_jobs=10,
        correction='Two legacy raw-logit auxiliary diagnostics were invalid for sigmoid heads; quarantined SHA-exact. No training/evaluation used them.',
        correction_receipt='analysis/legacy_unit_dispatch_migration_v1.json')
    review['diagnostic_semantics']=dict(gradient_log='grad/norm is sum of per-tensor L2 norms, not global L2 and no gradient clipping.',
        missing_field='Older plans asking for grad_norm omit this metric; omitted does not mean zero.',
        terminal_labels='Exact zero only at relabelled absorbing success boundary; not arbitrary nonterminal Q/V ground truth.',
        population_residual='Sample fitting loss and score RMS are not conditional residual or B_k estimates.',
        normalized_return='p=1+(1-gamma)Q/C preserves shaped rewards; not generally a binary success probability.')
    review['new_findings']=[
        'MatchedN9 native=.31259259, V2=.40888889 over allnine/last3; both below providedFQL.47. OwnBCweights differ.',
        'CommonBC ablation uses three exact self-composition rollout checks, then27 fixed-BC nativeQ evaluations; no GPU training.',
        'Freshgroup3/4/5 allnine=.348148; group6/7/8 six tasks complete, no pooled fresh macro yet.',
        'Latest independent fresh root audit:45 completed parents,136 actual unique late models, zero errors.',
        'Tau99/gamma999 native Pz3 seed3 late scores are0/0/0 with positive cost Q/V in thousands. Bounded repeated800K=0; finiteness is not a performance win.',
        'UnitMSE/KL pilots train unchanged; diagnostic arrays independently reproduce252 original-cost metrics across10 observed points.',
        'Pz3 unitKL100K success-boundary targetQ RMSE~.0255 versus unitMSE~2.526; distinct teachers and one early checkpoint, not policy improvement or V-only causality.',
        'Fullnine mean-head finite-backup discovery still incomplete; no fresh promotion before strict nine-task>.47.',
        'Legacy sigmoid-logit diagnostic mismatch fixed by restricting dispatcher scope and preserving invalid artifacts in quarantine; source model/config SHAs unchanged.']
    for key,path in {
        'fresh_root_audit_v3':'analysis/fresh_confirmation_root_audit_v3/audit.json',
        'native_N9_root_audit':'analysis/native_IFQL_N9_reference_root_audit_v1.json',
        'common_BC_registration':'common_BC_value_ablation_registration_v1.json',
        'unit_fitting_observed_audit':'analysis/unit_value_fitting_observed_audit_v1.json',
        'unit_dispatch_correction':'analysis/legacy_unit_dispatch_migration_v1.json',
        'mean_full9_progress_audit':'analysis/mean_backup_full9_progress_audit_v2.json',
        'scheduling_amendment_v4':'scheduling_amendment_20261028_v4.json',
        'runtime_v30':'analysis/runtime_audit_v30.json'}.items():review['evidence'][key]=dict(path=path,sha256=sha(root/path))
    create(out/'method_review_v29.json',review)
    with(out/'common9_comparison_v17.csv').open('x',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*ENVIRONMENTS,'completed_tasks','nine_task_mean'])
        for row in review['methods']:writer.writerow([row['label'],*row['values'],row['completed_tasks'],row['nine_task_mean']])
    continuation=json.loads((root/'CONTINUATION.json').read_text())
    continuation.update(updated_at=time.time(),latest_review='analysis/method_review_v29.json',
        latest_comparison='analysis/common9_comparison_v17.csv',latest_runtime='analysis/runtime_audit_v30.json',
        status=review['status'],last_goal_turn_classification='progress',last_goal_turn_evidence='analysis/native_IFQL_N9_reference_root_audit_v1.json')
    continuation['latest_reports'].update(review=continuation['latest_review'],comparison=continuation['latest_comparison'],runtime=continuation['latest_runtime'],
        point_in_time_note='MatchedN9 native .312593 versusV2 .408889; actualBCdiffers. Firstfresh .348148, second6/9complete. No confirmedwin.')
    continuation['GPU_queue'].update(counts={k:status[k]for k in ('finished','pending')},running_handles=status['running']);continuation['GPU_queue']['counts']['running']=16
    continuation['next_actions']=[
        'Revalidate16learnerPID/startidentities and memoryfraction.1; schedulerV20PID1584686. No timeout restarts.',
        'All registered sources remain frozen. NewcommonBCcontroller1620504 is registered; never edit it or evaluator/helper imports.',
        'NativeN9reference1607969completed27CPU50; do not restart. Rootaudit checked54actual models, .312593 native versus.408889 V2; ownBCnotexact.',
        'WatchcommonBC30-evaluation ablation: three self-composition sanities must match raw episode arrays, then27crosses; no new GPU jobs.',
        'Fresh firstgroup .348148, secondgroup6/9complete; source/modelrootauditv3 136models/45done/noerrors. Full9/pooledfresh not ready.',
        'Unit6pilots remain exploratory seed3. CPUgeometry/affine/update/resume validated; only actual100KMSE BC equality established so far, not all future points.',
        'Dedicatedunitfitting1604651 decodescosts correctly. LegacyV5 dispatcher1609437 skips customreturnmodels. Never use quarantined raw-logit metrics.',
        'Tau99 nativePz3 complete0/0/0 and positive cost values inthousands. Boundedtau99finite does not imply return improvement; wait late3/N99secondary.',
        'Sampling fresh-vs-repeat three-pilotgate must finish and rootreview. Fresh8e5Pz3=.84 is one checkpoint, not last3/full9.',
        'Meanfinite full9still incomplete; conditional1547729 may enqueue fresh9/10/11 only after strict allnine>.47.',
        'If a unit proper-score profile succeeds, separate Q-score versus V-score factorial before attributing gain specifically to residual optimization.',
        'Queue ORDERv4 preservesall308 recipes and allactive handles; keep GPU8+8. Monitorfree disk25GiB and protect pinned models when offloading.',
        'Goal remains active. Require uniform9 mean>.47/.44 and independent fresh training confirmation; no taskwise winners or missing-as-zero.']
    temporary=root/'CONTINUATION.tmp';temporary.write_text(json.dumps(continuation,indent=2,allow_nan=False));temporary.replace(root/'CONTINUATION.json')
    print(json.dumps(dict(review='method_review_v29.json',workers=16,queue=runtime['queue_size'],native_N9=native['native_N9_nine_task_mean'])),flush=True)


if __name__=='__main__':main()
