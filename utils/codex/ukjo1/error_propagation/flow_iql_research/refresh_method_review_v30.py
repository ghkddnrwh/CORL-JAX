"""Full9 mean-backup discovery, completed controls and fresh validation status."""
import argparse
import collections
import csv
import json
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
import time
from utils.flow_iql_research.refresh_method_review_v27 import ENVIRONMENTS
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def create(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False)


def assign(row,data,evidence):
    row.update(values=[data.get(e)for e in ENVIRONMENTS],evidence=evidence)
    row['completed_tasks']=sum(v is not None for v in row['values'])
    row['nine_task_mean']=statistics.mean(row['values'])if row['completed_tasks']==9 else None


def runtime(root):
    status=json.loads((root/'scheduler_status.json').read_text());workers=[]
    for rid,h in status['running'].items():
        proc=Path(f'/proc/{h["pid"]}');assert proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]==h['process_identity']
        cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0');assert cmd[cmd.index('--output')+1]==h['output']
        env=dict(x.split('=',1)for x in proc.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
        assert env['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'and env['CUDA_VISIBLE_DEVICES']==str(h['gpu'])
        run=Path(h['output']);conf=json.loads((run/'config.json').read_text());helper.check_sources(conf['source_sha256'])
        clocks=None
        if(run/'latest.pkl').exists():
            with(run/'latest.pkl').open('rb')as stream:saved=pickle.load(stream)
            state=saved['state'];step=saved['step'];assert helper.finite_tree(state)
            assert saved['source_sha256']==conf['source_sha256']
            assert int(state['q_updates'])==int(state['agent']['network']['opt_state']['0']['count'])==step
            assert int(state['v_updates'])==int(state['v_opt']['0']['count'])==2*step
            clocks=dict(step=step,Q_Adam=step,V_Adam=2*step,actual_full_saved_state_finite=True)
        assert conf['identity']['profile']=='bounded_v2_mean_empirical_backup'
        probes=[json.loads(x)for x in(run/'diagnostics.jsonl').read_text().splitlines()]if(run/'diagnostics.jsonl').exists()else[]
        assert not probes or probes[-1]['finite']
        workers.append(dict(run=rid,**h,module=cmd[cmd.index('-m')+1],actual_saved_clocks=clocks,
            latest_checkpoint_status='verified'if clocks else'not_yet_saved',
            last_probe_step=probes[-1]['step']if probes else None,source_pins_exact=True))
    assert len(workers)==16 and collections.Counter(w['gpu']for w in workers)=={0:8,1:8}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    assert {int(line.split(',')[0])for line in apps.splitlines()}<={w['pid']for w in workers}
    return dict(time=time.time(),scheduler_pid=status['pid'],scheduler_module='utils.flow_iql_research.scheduler_v20',
        scheduler_finished=status['finished'],scheduler_pending=status['pending'],actual_learner_processes=16,
        per_GPU={'0':8,'1':8},workers=workers,nvidia_compute_apps=apps,
        free_NVMe_bytes=shutil.disk_usage(root).free,queue_size=len(json.loads((root/'queue.json').read_text())),goal_complete=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis';review=json.loads((out/'method_review_v29.json').read_text())
    fresh=json.loads((out/'fresh_confirmation_root_audit_v4/audit.json').read_text());assert fresh['status']=='complete'and not fresh['errors']
    assert fresh['completed_members']==54 and fresh['actual_unique_late_models_verified']==162
    for i,group in enumerate(fresh['groups'],1):
        for method,label in [('mean_rank','mean_rank'),('median_rank','median_rank'),('mean','raw_mean_control')]:
            row=next(m for m in review['methods']if m['label'].startswith(f'Independent3 {label} fresh group{i} '))
            assign(row,{t['environment']:t['ensemble_policies'][method]['last3_mean']for t in group['tasks']},group['tasks'])
        for method,label in [('first_BC_own_single_critic','First BC own critic'),('uniform_member_expected_return','Uniform member expected return')]:
            row=next(m for m in review['methods']if m['label']==f'{label} fresh group{i} (paired baseline)')
            assign(row,{t['environment']:t['baselines'][method]['last3_mean']for t in group['tasks']},group['tasks'])
    groups=[next(m for m in review['methods']if m['label']==f'Independent3 mean_rank fresh group{i} (preferred)')for i in (1,2)]
    pooled={e:statistics.mean(g['values'][k]for g in groups)for k,e in enumerate(ENVIRONMENTS)}
    row=dict(label='Independent3 mean_rank fresh two-group average (failed confirmation)');assign(row,pooled,'analysis/fresh_confirmation_root_audit_v4/audit.json');review['methods'].append(row)
    mean=json.loads((out/'mean_backup_discovery_root_audit_v2.json').read_text());assert mean['completed_tasks']==9
    row=next(m for m in review['methods']if m['label']=='Mean-head finite expectile backup, best9 (s0)')
    assign(row,{r['environment']:r['last3_mean']for r in mean['rows']},mean['rows'])
    component=json.loads((out/'mean_min_backup_full9_components_v1.json').read_text());assert component['all_actual_BC_moments_and_RNG_exact']
    common=json.loads((out/'common_BC_value_ablation_root_audit_v1.json').read_text());assert common['evaluations_verified']==30
    row=dict(label='Native IFQL Q with common V2 BC, best9 (s0)');assign(row,{r['environment']:r['native_Q_common_V2_BC_last3']for r in common['rows']},common['rows']);review['methods'].append(row)
    confirmation=json.loads((out/'mean_backup_independent_confirmation_v1/status.json').read_text())
    for seed in (9,10,11):
        rows=[r for r in confirmation['rows']if r['seed']==seed];assert len(rows)==9
        row=dict(label=f'Mean-head finite expectile backup fresh seed{seed}')
        assign(row,{r['environment']:r['last3_mean']if r['complete']else None for r in rows},rows);review['methods'].append(row)
    states={}
    for label,path in [('mean_confirmation','mean_backup_independent_confirmation_v1'),('rank_mean','independent_critic_rank_mean_confirmation_v1'),
        ('unit','unit_return_scoring_pilots_v1'),('sampling','value_minibatch_reuse_audit_v1'),('tau99','tau_sampling_factorial_audit_v1'),
        ('tau99_N99','tau_matched_extraction_v1'),('unit_fitting','unit_value_fitting_diagnostic_v1'),('fresh_boundary','fresh_value_boundary_diagnostic_v1')]:
        p=out/path/'status.json'
        if p.exists():states[label]=json.loads(p.read_text())
    create(out/'verification_status_snapshot_v30b.json',dict(time=time.time(),states=states))
    current=runtime(root);create(out/'runtime_audit_v31.json',current)
    review.update(time=time.time(),status='mean_backup_discovery_0617_independent_confirmation_running',goal_complete=False,
        historical_review='analysis/method_review_v29.json',runtime=current,
        point_in_time_confirmation_status_snapshot=dict(path='analysis/verification_status_snapshot_v30b.json',sha256=helper.digest(out/'verification_status_snapshot_v30b.json')))
    review['preferred_candidate']=dict(recipe='mean target twin Q finite flow expectile; projected Bellman target; min online twin Q deployment',
        tau=.9,gamma=.999,candidates=9,learners=1,discovery_seed0_mean=mean['nine_task_mean'],fresh_seeds=[9,10,11],
        fresh_confirmation_passed=confirmation['independent_confirmation_criteria_passed'],
        reproducibility='utils/flow_iql_research/REPRODUCE_MEAN_FINITE_BACKUP_V1.txt',
        unused_V_fits='Retained2Adamfits are clock-control overhead; they do not enter finite backup or deployment.')
    review['component_control']=dict(actual_late_models=54,BC_and_moments_and_RNG_all_exact=True,
        finite_min_mean=component['min_backup_nine_task_mean'],finite_mean_mean=component['mean_backup_nine_task_mean'],
        limits='Finite-min did not reach .47. Learned-V/mean-head ablation is missing; no claim that every component is independently necessary or that true-Q bias was directly measured.')
    review['confirmations']['live_status_snapshot']=states
    review['new_findings']=[
        'Uniform mean-head finite backup full9 seed0=.617037; strictroot27actualCP/episode audit passed. Fresh9/10/11 running, no finalwin yet.',
        'Finite minimum versus mean targetQ: .4 versus .617037;54actualmodels; allBCweights/moments/replay/agentRNG EXACT. This is a controlled aggregation contrast.',
        'Fixed commonactualBC nativeQ=.31185185 versus V2Q=.40888889.54modelSHA rechecks and30rawCPU50/sanity evaluations audited.',
        'Old independent3rankmean failed: pooledfresh=.37888889; all3groups incldiscovery=.43456790.162actuallateCPs/54parents rootaudited; zeroerrors.',
        'FreshsecondVbatch three-taskmean=.44888889 versusrepeat=.12222222,2of3improve; actualBCexact. Not a full9 result.',
        'UnitKL three-taskmean=.231111 versusraw=.122222;unitMSE=.008889. Ant improves, Pz3doesnot; small boundaryerror alone is insufficient for policyperformance.',
        'Tau99 three-task N9means near0 for native/repeat/fresh; increasing toN99 also0. Not a successful high-tau method.',
        'Losslessoffload542oldcompletedinodegroups/751paths recovered32GiBfree; allactive/latest/fresh/mean/efmodels untouched. No learnerrestart.',
        'No environmentwise hyperparameters or method mixing; current priority is confirming the one uniform mean-backup rule.']
    for key,path in {
        'mean_root_discovery_v2':'analysis/mean_backup_discovery_root_audit_v2.json',
        'mean_min_full9_control':'analysis/mean_min_backup_full9_components_v1.json',
        'common_BC_root_audit':'analysis/common_BC_value_ablation_root_audit_v1.json',
        'fresh_root_audit_v4':'analysis/fresh_confirmation_root_audit_v4/audit.json',
        'offload_validation_20261029':'analysis/offload_validation_20261029_v1.json',
        'mean_confirmation_registration':'mean_backup_independent_confirmation_registration_v1.json',
        'runtime_v31':'analysis/runtime_audit_v31.json'}.items():review['evidence'][key]=dict(path=path,sha256=helper.digest(root/path))
    create(out/'method_review_v30.json',review)
    with(out/'common9_comparison_v18.csv').open('x',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*ENVIRONMENTS,'completed_tasks','nine_task_mean'])
        for row in review['methods']:writer.writerow([row['label'],*row['values'],row['completed_tasks'],row['nine_task_mean']])
    continuation=json.loads((root/'CONTINUATION.json').read_text())
    continuation.update(updated_at=time.time(),latest_review='analysis/method_review_v30.json',latest_comparison='analysis/common9_comparison_v18.csv',
        latest_runtime='analysis/runtime_audit_v31.json',last_goal_turn_classification='progress',
        last_goal_turn_evidence='analysis/mean_backup_discovery_root_audit_v2.json')
    continuation['latest_reports'].update(review=continuation['latest_review'],comparison=continuation['latest_comparison'],runtime=continuation['latest_runtime'],
        point_in_time_note='Meanbackupdiscovery.617037; fresh9/10/11 running. Oldrankmeanfresh.378889 failed. No final independentwin yet.')
    continuation['next_actions']=[
        'Revalidate16workers/PIDstartIDs and fraction.1, GPU8+8. SchedulerV20PID1584686;335jobs incl27meanconfirmation; no timeout restarts.',
        'Preferredcandidate is mean-headfiniteexpectile, not the failed independent3rankmean. Frozen source/registration exact; do not edit current training or helpers.',
        'Meanfull9discovery.617037 audited27actualCPs. Fresh9/10/11 gate all27parents/81lateCPs;freshmean>.47,all4mean>.47,atleast2freshseedmeans>.47. Goal stays active until finalrootreview.',
        'Full9min-vs-mean actual54model auditpassed;allBCparams/moments/replay/agentRNGexact. Min.4 versusmean.617 is controlled. No learned-V/mean-head ablation yet; do not claim finite-root necessity or directly measured true-Q bias.',
        'RootcommonBC auditpassed30evals/54modelSHAchecks; nativeQ.311852 versusV2Q.408889. Its controller is complete; do not restart.',
        'Oldrankmean full54freshlearners complete,pooledfresh.378889/all3.434568 FAILED. Rootauditv4 verified162lateCPs,zeroerrors. No reuse as preferred winner.',
        'Sampling3pilotgate passed with actualBCexact and2of3improve. Full9expansion remains an eligible fallback, not completed or silently mixed into current method.',
        'Unit3pilotKL improvesAnt butnotPz3;overall3mean.2311 versusfreshbatch.4489. Do not expand/tune from smaller boundaryerror alone. UnitMSE .00889 fails.',
        'High-tau pilots completed almost0 atN9 and0 atN99; keep primaryN9 and report failure, no successfultau99claim.',
        'Allnewdiagnosticcontrollers are complete except ongoing meanconfirmation. Never restart terminal controllers for observation timeout.',
        'Storage recovered32GiB withnew V3offload journalOct29v2.542groups/751pathslosslesslySHAverified; preservealltargets/symlinks;allfresh/mean/ef/active/latestunmoved.',
        'Current validated meanrunner keeps2V fits as unusedclock-control overhead. Do not silently remove or claim compute equality. GuideREPRODUCE_MEAN_FINITE_BACKUP_V1.txt.',
        'Before goalcomplete: audit all rawlateepisodes/configs/models/source/clocks/finite, reportseedSD/tasktable/failedbranches and reviewfairness. No missing-as-zero or per-envwinners.']
    temp=root/'CONTINUATION.tmp';temp.write_text(json.dumps(continuation,indent=2,allow_nan=False));temp.replace(root/'CONTINUATION.json')
    print(json.dumps(dict(review='method_review_v30.json',workers=16,discovery=mean['nine_task_mean'],goal_complete=False)),flush=True)


if __name__=='__main__':main()
