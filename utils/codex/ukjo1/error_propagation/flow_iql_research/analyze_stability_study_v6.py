# Scope v6: batch-size comparison restricted to seeds 0 and 1.
"""Paired seed-level stability analysis, including failed/incomplete cells."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import time
from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_discount_0995_n9_v2 import sha, atomic, csv_file


def summarize(values):
    return (st.mean(values), st.stdev(values), min(values)) if len(values)>1 else (None, None, None)


def committed_lines(path):
    if not path.exists(): return []
    text=path.read_text()
    return [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith('\n') and line.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    plan=json.loads((root/'registration_v6.json').read_text())
    rows, evidence, problems=[], {}, []
    trained=evaluated=0
    for job in plan['jobs']+plan['reused_jobs']:
        run=Path(job['output'])
        conf=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else None
        if conf:
            ident=conf['identity']; cfg=ident['config']
            assert ident['environment']==job['environment'] and ident['seed']==job['seed']
            assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']==9
            assert cfg.get('backup_samples',9)==9 and cfg['target_clipping']
            assert cfg['backup_Q_aggregation']==job['head_aggregation']
            if not job['reused']:
                assert ident['recipe']==job['recipe']
                assert cfg['tau']==(.001 if job['recipe']=='slow_target' else .005)
                assert cfg['critic_batch_size']==(1024 if job['recipe']=='critic_batch1024' else 256)
        complete=(run/'completed.json').exists()
        if complete:
            c=json.loads((run/'completed.json').read_text())
            assert c['step']==c['q_updates']==c['actor_updates']==1000000 and c['v_updates']==0
            trained+=1
        if (run/'stopped.json').exists():
            problems.append(dict(run=job['id'], reason='nonfinite_state'))
        diagnostics=committed_lines(run/'diagnostics.jsonl')
        representations=committed_lines(run/'representation_diagnostics.jsonl')
        tail=[r for r in diagnostics if r['step']>=800000]
        rep_tail=[r for r in representations if r['step']>=800000]
        diagnostic_summary={}
        for key in ('q_rms','td_rms','q_disagreement_rms','q_target_lag_rms','backup/clip_fraction'):
            vals=[r[key] for r in tail if key in r]
            diagnostic_summary['tail_'+key.replace('/','_')]=st.mean(vals) if vals else None
        for key in ('feature_cosine_mean','feature_dot_mean','feature_norm_mean',
                    'rank_change_fraction','centered_Q_drift_rms','candidate_margin_median',
                    'centered_feature_effective_rank_mean'):
            vals=[r[key] for r in rep_tail if key in r]
            diagnostic_summary['tail_'+key]=st.mean(vals) if vals else None
        diagnostic_summary['representation_points']=len(rep_tail)
        if diagnostics and not all(r['finite'] for r in diagnostics):
            problems.append(dict(run=job['id'],reason='nonfinite_recorded_diagnostics'))
        directory=Path(job['evaluation_directory'])
        for mode in plan['deployment_modes']:
            scores={}
            for step in plan['primary_evaluation_steps']:
                path=directory/f'{step}_{mode}.json'
                if not path.exists(): continue
                v=json.loads(path.read_text())
                assert conf and v['training_seed']==job['seed'] and v['environment']==job['environment']
                assert v['step']==step and v['policy_mode']==mode
                assert v['n_episodes']==len(v['episode_successes'])==50 and v['first_episode_seed']==42
                assert v['protocol']=='last3_cpu50_independent_seeded_episodes_v1' and v['no_optimizer_updates']
                assert abs(v['success_rate']-st.mean(v['episode_successes']))<1e-12
                assert v['config_sha256']==sha(run/'config.json')
                if args.final:
                    for p, expected in ((run/f'checkpoint_{step}.pkl',v['checkpoint_sha256']),
                                        (run/f'actor_{step}.pkl',v['actor_artifact_sha256'])):
                        if str(p) not in evidence: assert sha(p)==expected
                        evidence[str(p)]=expected
                evidence[str(path)]=sha(path)
                scores[step]=100*v['success_rate']; evaluated+=1
            available=len(scores)==3
            row=dict(environment=job['environment'],method=job['method'],seed=job['seed'],deployment=mode,
                     reused=job['reused'],training_complete=complete,evaluation_points=len(scores),
                     last3_mean_pct=st.mean(scores.values()) if available else None,
                     final_pct=scores.get(1000000),success_800K=scores.get(800000),success_900K=scores.get(900000),
                     late_range_pp=max(scores.values())-min(scores.values()) if available else None,
                     late_drop_pp=max(0,scores[800000]-scores[1000000]) if available else None,
                     run=job['id'],run_directory=str(run),**diagnostic_summary)
            rows.append(row)
    groups=[]; paired=[]
    for env in plan['environments']:
        for mode in plan['deployment_modes']:
            baseline={r['seed']:r for r in rows if r['environment']==env and r['deployment']==mode and r['method']=='control'}
            for method in plan['methods']:
                rs=[r for r in rows if r['environment']==env and r['deployment']==mode and r['method']==method]
                available=[r for r in rs if r['last3_mean_pct'] is not None]
                full=len(available)==len(plan['training_seeds'])
                values=[r['last3_mean_pct'] for r in available]
                mean,sd,worst=summarize(values) if full else (None,None,None)
                final=[r['final_pct'] for r in available]
                groups.append(dict(environment=env,deployment=mode,method=method,complete_seeds=len(available),
                    expected_seeds=len(plan['training_seeds']),mean_pct=mean,sample_sd_pp=sd,worst_seed_pct=worst,
                    final_mean_pct=st.mean(final) if full else None,
                    final_sd_pp=st.stdev(final) if full else None,
                    mean_late_range_pp=st.mean(r['late_range_pp'] for r in available) if full else None,
                    mean_late_drop_pp=st.mean(r['late_drop_pp'] for r in available) if full else None,
                    collapse_20pp_count=sum(r['late_drop_pp']>=20 for r in available) if full else None))
                if method!='control':
                    for r in available:
                        b=baseline[r['seed']]
                        if b['last3_mean_pct'] is None: continue
                        paired.append(dict(environment=env,deployment=mode,method=method,seed=r['seed'],
                            mean_change_pp=r['last3_mean_pct']-b['last3_mean_pct'],
                            final_change_pp=r['final_pct']-b['final_pct'],
                            late_range_change_pp=r['late_range_pp']-b['late_range_pp']))
    # Stability flags are descriptive, not significance tests or automatic model selection.
    index={(r['environment'],r['deployment'],r['method']):r for r in groups}
    for row in groups:
        base=index[row['environment'],row['deployment'],'control']
        ready=row['mean_pct'] is not None and base['mean_pct'] is not None
        row['mean_change_pp']=row['mean_pct']-base['mean_pct'] if ready else None
        row['sd_change_pp']=row['sample_sd_pp']-base['sample_sd_pp'] if ready else None
        row['worst_seed_change_pp']=row['worst_seed_pct']-base['worst_seed_pct'] if ready else None
        row['descriptive_stability_candidate']=bool(ready and row['method']!='control' and
            row['mean_change_pp']>=0 and row['sd_change_pp']<0 and row['worst_seed_change_pp']>=0 and
            row['mean_late_range_pp']<=base['mean_late_range_pp']) if ready else None
    complete=trained==plan['total_cells'] and evaluated==plan['total_cells']*6 and not problems
    if args.final:
        assert complete
        for field in ('source_sha256','reused_evidence_sha256'):
            for p,d in plan[field].items(): assert sha(p)==d,p
    suffix='final_v6' if args.final else 'progress_v6'
    out=root/'analysis'
    csv_file(out/f'per_seed_{suffix}.csv',rows)
    csv_file(out/f'task_summary_{suffix}.csv',groups)
    if paired: csv_file(out/f'paired_differences_{suffix}.csv',paired)
    if args.final:
        from utils.codex.ukjo1.error_propagation.flow_iql_research.report_stability_study_v6 import render
        render(root,plan,rows,groups,paired)
    result=dict(time=time.time(),complete=complete,training_complete=trained,total_cells=plan['total_cells'],
        primary_evaluations=evaluated,expected_primary_evaluations=plan['total_cells']*6,
        failures=problems,evidence_sha256=evidence,
        interpretation='Two paired training seeds. Full two-seed means only; SD is descriptive. Descriptive stability flags; no multiplicity-adjusted claims.')
    atomic(out/f'summary_{suffix}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'}),flush=True)


if __name__=='__main__': main()
