"""Record the audited fresh progress and prospective bounded-return pilots."""
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


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(p,value):
    with p.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis';review=json.loads((out/'method_review_v27.json').read_text())
    evidence=json.loads((out/'fresh_confirmation_root_audit_v2/audit.json').read_text());assert not evidence['errors']
    for i,group in enumerate(evidence['groups'],1):
        for method,label in [('mean_rank','mean_rank'),('median_rank','median_rank'),('mean','raw_mean_control')]:
            row=next(m for m in review['methods']if m['label'].startswith(f'Independent3 {label} fresh group{i} '))
            data={t['environment']:t['ensemble_policies'][method]for t in group['tasks']}
            row.update(values=[data[e]['last3_mean']if data[e]['complete']else None for e in ENVIRONMENTS],
                evidence=[dict(environment=e,**data[e])for e in ENVIRONMENTS],
                actual_model_audit='analysis/fresh_confirmation_root_audit_v2/audit.json')
            row['completed_tasks']=sum(v is not None for v in row['values'])
            row['nine_task_mean']=statistics.mean(row['values'])if row['completed_tasks']==9 else None
        for method,label in [('first_BC_own_single_critic','First BC own critic'),('uniform_member_expected_return','Uniform member expected return')]:
            row=next(m for m in review['methods']if m['label']==f'{label} fresh group{i} (paired baseline)')
            data={t['environment']:t['baselines'][method]for t in group['tasks']}
            row.update(values=[data[e]['last3_mean']if data[e]['complete']else None for e in ENVIRONMENTS],
                evidence=[dict(environment=e,**data[e])for e in ENVIRONMENTS])
            row['completed_tasks']=sum(v is not None for v in row['values'])
            row['nine_task_mean']=statistics.mean(row['values'])if row['completed_tasks']==9 else None
    status=json.loads((root/'scheduler_status.json').read_text());workers=[]
    for rid,h in status['running'].items():
        proc=Path(f"/proc/{h['pid']}");identity=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]
        assert identity==h['process_identity']
        cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0');assert cmd[cmd.index('--output')+1]==h['output']
        environment=dict(x.split('=',1)for x in proc.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
        assert environment['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1' and environment['CUDA_VISIBLE_DEVICES']==str(h['gpu'])
        run=Path(h['output']);conf=json.loads((run/'config.json').read_text())
        for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
        probes=[json.loads(x)for x in(run/'diagnostics.jsonl').read_text().splitlines()]if(run/'diagnostics.jsonl').exists()else[]
        actual=None
        if(run/'latest.pkl').exists():
            with(run/'latest.pkl').open('rb')as stream:saved=pickle.load(stream)
            state=saved['state'];step=saved['step'];native=conf['identity']['profile']=='official'
            assert int(state['q_updates'])==int(state['agent']['network']['opt_state']['0']['count'])==step
            assert int(state['v_updates'])==(1 if native else 2)*step
            assert int(state['v_opt']['0']['count'])==(0 if native else 2)*step
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
    create(out/'runtime_audit_v29.json',runtime)
    states={}
    for name,path in [('mean','independent_critic_rank_mean_confirmation_v1/status.json'),
        ('median','independent_critic_fresh_confirmation_v1/status.json'),('raw','independent_critic_raw_mean_control_v1/status.json')]:
        states[name]=json.loads((out/path).read_text())
    create(out/'verification_status_snapshot_v28.json',dict(time=time.time(),**states))
    review.update(time=time.time(),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='first_fresh_full9_below_threshold_second_group_partial_new_scoring_pilots_queued',
        runtime=runtime,goal_complete=False,historical_review='analysis/method_review_v27.json',
        point_in_time_confirmation_status_snapshot=dict(path=str(out/'verification_status_snapshot_v28.json'),sha256=sha(out/'verification_status_snapshot_v28.json')))
    review['confirmations']['live_status_snapshot']=states
    review['confirmations']['latest_root_audit']=dict(completed_members=evidence['completed_members'],
        actual_unique_late_models_verified=evidence['actual_unique_late_models_verified'],errors=evidence['errors'])
    review['unit_return_scoring_pilots']=dict(registration='unit_return_scoring_pilots_registration_v1.json',
        jobs=6,tasks=['Ant','HMM','Pz3'],seed=3,expectile=.9,discount=.999,proposals=9,
        variants=['sigmoid_unit_MSE','sigmoid_unit_Bernoulli_KL'],new_late_policy_outcomes=0,
        exact_Bellman_affine_checks=18,scalar_expectile_distributions=300,
        actual_CPU_update_comparison=12,CPU_resume_modes=2,checkpoint_SHA_exact=True,
        hypothesis='Boundedreturnheadsandconsistentexpectilescoringimproveconditioning;finitevaluesalonearenotperformanceproof.',
        representation='p=1+(1-gamma)*Q/C,C=declaredrewardcomponents;shapedrewardsretained.',
        attribution_limit='MSEvsKLsharesrepresentation/init. Rawvsunitalsochangesdecodedinitialization/outputparameterization.',
        actor='UnweightedofficialBCflowunchanged;actualBC/moments/replay/RNGlatecomparisonsregistered.',
        independent_confirmation_required='Uniform9andfreshunusedtrainingseeds>=9')
    review['trace_fixedpoint_analysis']=dict(oracle_models=8,trace_runs=24,OGBench_performance_claim=False,
        result='Equalvalue-residual,conditional-scoreandpropagated-Qsupnormscancoexistwithdifferentgreedypolicyreturn.',
        decision='Do not queue trace-only as a systematicbiasfix: invertibleTDpreconditioningpreservesthebiasedroot.',
        caveat='Thisdoesnotruleoutoptimizationbenefitswithstochastic/neuraltrainingorproveOGBenchcausality.',
        figure='analysis/trace_residual_fixedpoint_oracle_v1.png')
    review['new_findings']=[
        'Firstfreshmeanrank9task=.348148,below.47/.44;noindependentbenchmarkwin.',
        f"Latestrootauditchecked{evidence['actual_unique_late_models_verified']}actualuniqueCPs,{evidence['completed_members']}completedparents;errors=[].",
        'Secondgroupnow4taskscomplete;Pz3=.053333,Ant=.026667,CubeS=.886667,CubeD=.493333. No partial9mean.',
        'ExistingBC/criticcross: Pz3freshBC+seencritics=.92,seenBC+freshcritics=0. Criticqualitylimitscannotbeassumedfixedbybetteractoralone.',
        'Sixprospectiveboundedunit-returnMSE/KLpilotsregistered/queuedafterscalarandCPUupdate/resumevalidation;noGPUperformanceclaim.',
        'SchedulerV20adoptedall16livehandlesandpreservedall302oldjobdicts;queue308. No learnerrestart.',
        'Meanbackupconditionalfresh9/10/11confirmationstillwaitsforall9seed0strictaudit>.47.',
        'Synthetictraceanalysisrejectedanunsupportedgamma**Hsystematic-biasargumentwithoutspendingnewGPUruns.']
    for key,path in {
        'fresh_root_audit_v2':'analysis/fresh_confirmation_root_audit_v2/audit.json',
        'trace_fixedpoint_oracle':'analysis/trace_residual_fixedpoint_oracle_v1.json',
        'trace_fixedpoint_figure':'analysis/trace_residual_fixedpoint_oracle_v1.png',
        'unit_return_pilot_registration':'unit_return_scoring_pilots_registration_v1.json',
        'unit_return_update_validation':'analysis/probability_value_update_validation_v1.json',
        'unit_return_resume_validation':'analysis/probability_value_resume_validation_v1/validation.json',
        'scheduler_v20_adoption':'analysis/scheduler_v20_adoption_v1.json',
        'scheduling_amendment_v3':'scheduling_amendment_20261028_v3.json',
        'mean_backup_fresh_registration':'mean_backup_independent_confirmation_registration_v1.json',
        'runtime_v29':'analysis/runtime_audit_v29.json'}.items():review['evidence'][key]=dict(path=path,sha256=sha(root/path))
    create(out/'method_review_v28.json',review)
    with(out/'common9_comparison_v16.csv').open('x',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*ENVIRONMENTS,'completed_tasks','nine_task_mean'])
        for row in review['methods']:writer.writerow([row['label'],*row['values'],row['completed_tasks'],row['nine_task_mean']])
    continuation=json.loads((root/'CONTINUATION.json').read_text())
    continuation.update(updated_at=time.time(),latest_review='analysis/method_review_v28.json',
        latest_comparison='analysis/common9_comparison_v16.csv',latest_runtime='analysis/runtime_audit_v29.json',
        status=review['status'],last_goal_turn_classification='progress',
        last_goal_turn_evidence='analysis/scheduler_v20_adoption_v1.json')
    continuation['GPU_queue'].update(counts={k:status[k]for k in ('finished','pending')},running_handles=status['running']);continuation['GPU_queue']['counts']['running']=16
    continuation['next_actions']=[
        'Revalidateactual16PID/startidentities;currentmanagerV20PID1584686. Neverrestartfortimeout.',
        'Allregisteredtrain/helper/manager/note sources frozen. Use newversions iffixneeded;do noteditunitcore afterregistration.',
        'Unitreturn6pilotsnowqueuedwithnewanalyzer1584794. Bothupdateandresumevalidationspassed. CompareKL/MSE/rawinoriginalcostunits;no newlatepolicyoutcomesyet.',
        'ContinueN9original54parentconfirmation;firstfresh.348148,secondgroup4/9complete. Rootactualauditv2checks133CPs/43parentswithnoerrors.',
        'Tau9sampling,tau99native/repeated/fresh pilots still running/queued. N99secondary1538829registeredbeforelateoutcomes;primaryN9unchanged.',
        'Meanbackup2Humanoidsnear800K;4remaining9tasksqueued/starting. Conditional1547729canadd27fresh9/10/11jobsonlyafterstrictall9>.47;currentlynotpromoted.',
        'Qualifiedmeanpromotionretainsitsfrozenprioritypredicate;itdoesnotincludeunit1prefix,whichisacceptableasfreshconfirmationpriority.',
        'Existingexpectile-flowpilotsresumeexact300Ksourceswithauditor1542731. No duplicateunregisteredsignv2draftused.',
        'TraceoracleisnotOGBenchcausality:equalb/rho/Qbiasnormsbutdifferentpolicyregret;tracepreconditioningdoesnotremovefixedbrootbias. Do notqueueunmotivatedtrace-onlyGPUruns.',
        'NVMe~33GiBfree;monitor25GiBthresholdandprotectactive/fresh/latestandstrictstat-pinnedCPswhenoffloading.',
        'Goalnotmet;newlyselectedmethodneedsuniform9average>.47/.44andfreshunusedtrainingseeds>=9;do notmixper-envwinners.']
    temporary=root/'CONTINUATION.tmp';temporary.write_text(json.dumps(continuation,indent=2,allow_nan=False));temporary.replace(root/'CONTINUATION.json')
    print(json.dumps(dict(review='method_review_v28.json',workers=16,queue=runtime['queue_size'],fresh_group2_complete_tasks=4,new_GPU_pilots=6)),flush=True)


if __name__=='__main__':main()
