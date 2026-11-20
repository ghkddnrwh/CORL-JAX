"""Read allnine controlled training traces; distinguish floor saturation from fit."""
import argparse
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper

UPDATE_METRICS=('critic/critic_loss','grad/norm','backup/lower_clip_fraction','backup/upper_clip_fraction',
    'backup/clip_fraction','backup/clip_rms','backup/finite_proposal_score_abs_mean','backup/finite_proposal_score_abs_max',
    'backup/mean_minus_min_empirical_root_mean','backup/mean_minus_min_empirical_root_rms',
    'backup/empirical_root_minus_learned_V_mean','backup/empirical_root_minus_learned_V_rms',
    'backup/proposal_Q_within_state_std_mean','backup/learned_V_above_all_proposals_fraction')
PROBE_METRICS=('q_mean','q_rms','q_centered_rms','q_disagreement_rms','v_mean','v_rms',
    'q_above_upper_rms','q_below_lower_rms','q_rms_over_return_bound')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();root=args.root.resolve()
    audit_path=root/'analysis/mean_min_backup_full9_components_v1.json';audit=json.loads(audit_path.read_text())
    assert audit['status']=='complete'and audit['actual_unique_late_models_verified']==54 and audit['all_actual_BC_moments_and_RNG_exact']
    mean_plan_path=root/'mean_backup_full9_expansion_v1.json';plan=json.loads(mean_plan_path.read_text());rows=[]
    for job in plan['all9jobs']:
        short=job['id'].split('_')[1];traces={};families={}
        for mode,rid in [('min',f'eb1_{short}_bounded_v2_empirical_backup_g999_t90_s0'),('mean',job['id'])]:
            run=root/'runs'/rid;conf=json.loads((run/'config.json').read_text());helper.check_sources(conf['source_sha256'])
            assert conf['identity']['seed']==0 and conf['identity']['config']['discount']==.999 and conf['identity']['config']['expectile']==.9
            dataset=json.loads((run/'dataset.json').read_text());bound=abs(dataset['return_lower']);assert dataset['return_upper']==0
            file=run/'diagnostics.jsonl';logs=[json.loads(x)for x in file.read_text().splitlines()]
            assert [d['step']for d in logs]==list(range(1000,1000001,1000))
            assert all(d['finite']and d['q_updates']==d['step']and d['v_updates']==2*d['step']for d in logs)
            assert all(d['metric_window_updates']==1000 for d in logs)
            phases={}
            for phase,low,high in [('early',0,100000),('middle',100000,800000),('late',800000,1000000)]:
                chosen=[d for d in logs if low<d['step']<=high];updates=sum(d['metric_window_updates']for d in chosen)
                metrics={k:sum(d[k]*d['metric_window_updates']for d in chosen)/updates for k in UPDATE_METRICS if all(k in d for d in chosen)}
                probes={k:statistics.mean(d[k]for d in chosen)for k in PROBE_METRICS}
                probes['negative_mean_Q_over_abs_return_lower']=-probes['q_mean']/bound
                phases[phase]=dict(start_exclusive=low,end_inclusive=high,recorded_updates=updates,
                    online_update_weighted_metrics=metrics,fixed_query_probe_time_means=probes)
                if mode=='mean':
                    metrics['derived_mean_root_min_minus_unused_learned_V']=metrics['backup/empirical_root_minus_learned_V_mean']-metrics['backup/mean_minus_min_empirical_root_mean']
            traces[mode]={d['step']:d for d in logs}
            families[mode]=dict(run=rid,log_sha256=helper.digest(file),config_sha256=helper.digest(run/'config.json'),
                known_return_lower=dataset['return_lower'],known_return_upper=0,all1000logged_probe_states_finite=True,phases=phases)
        assert families['min']['known_return_lower']==families['mean']['known_return_lower']
        common_actor=sorted(set.intersection(*[set(d)for trace in traces.values()for d in trace.values()]))
        common_actor=[k for k in common_actor if k.startswith('actor/')];assert common_actor
        actor_differences={k:dict(max_abs=max(abs(traces['min'][step][k]-traces['mean'][step][k])for step in traces['min']),
            differing_windows=sum(traces['min'][step][k]!=traces['mean'][step][k]for step in traces['min']))for k in common_actor}
        control=next(r for r in audit['rows']if r['environment']==job['args']['env'])
        rows.append(dict(short=short,environment=job['args']['env'],min_last3=control['min_backup_last3'],
            mean_last3=control['mean_backup_last3'],families=families,all_shared_actor_window_metrics_exact=all(x['differing_windows']==0 for x in actor_differences.values()),
            actor_window_comparisons=actor_differences))
    assert len(rows)==9
    result=dict(time=time.time(),status='complete',rows=rows,nine_task_min_mean=audit['min_backup_nine_task_mean'],
        nine_task_mean_mean=audit['mean_backup_nine_task_mean'],total_logged_updates_per_method=9000000,
        all_shared_actor_window_metrics_exact=all(r['all_shared_actor_window_metrics_exact']for r in rows),
        actual_model_control_audit_sha256=helper.digest(audit_path),mean_registration_sha256=helper.digest(mean_plan_path),source_sha256=helper.digest(__file__),
        new_GPU_jobs=0,new_policy_evaluations=0,goal_complete=False,
        metric_semantics=dict(online_metrics='1000-update windows are weighted by their recorded update counts; late is800K<step<=1M.',
            fixed_queries='Probe values are instantaneous on a per-run fixed1024-state sample. Their time means are descriptive, not averages over every replay transition.',
            finite_score='The residual is only on nine generated actions for each observed sampled teacher. Tiny finite score is not a population expectile or true-Q guarantee.',
            gradient_norm='grad/norm is the SUM of per-tensor L2 norms; no global gradient clipping.',
            unused_V='Both finite algorithms fit a min-Q learnedV as an unused clock control. Its root discrepancy is not the actual bootstrap fitting residual.',
            mean_difference='Subtracting the logged mean root gaps gives a mean discrepancy only. RMS values cannot be subtracted to obtain another RMS.',
            head_identity='Pointwise arithmetic mean minus minimum of two heads equals half their absolute disagreement. A head gap is not an oracle critic bias.',
            propagation='A constant scalar bias can accumulate geometrically in an ideal fixed operator. Multiplying these observed root gaps by1/(1-gamma) is NOT a measurement of B_k.'),
        limits=['Posthoc descriptive mechanism study on one discovery seed; independent seed9/10/11 confirmation remains required.',
            'Many floor-clipped targets and nearly constant bound-scale Q values are saturation indicators, not exact nonterminal true-Q error labels.',
            'The aggregation intervention is controlled by actual BC/moment/RNG equality; the full training dynamics still couple changing critics and targets.',
            'Actor-log equality is checked at every logged window, not asserted as a separate proof of every hidden intermediate parameter array.',
            'Changing learned V to a mean-Q teacher remains a separate registered3task mechanism control; its outcomes are not assumed.'])
    helper.atomic(args.output,result)
    print(json.dumps(dict(status='complete',tasks=9,all_actor_windows_exact=result['all_shared_actor_window_metrics_exact'])),flush=True)


if __name__=='__main__':main()
