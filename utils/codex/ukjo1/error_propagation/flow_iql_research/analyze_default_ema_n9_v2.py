# Scope v2: retain only AM-L, HM-M, Cube-Double and Puzzle4x4, as requested.
"""Paired seed0/1 EMA-rate comparison with fixed checkpoint aggregation."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import sha


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/'registration_v2.json').read_text())
    rows=[];failures=[];trained=evaluated=0
    for job in plan['jobs']+plan['reused_jobs']:
        run=Path(job['output'])
        conf=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else None
        if conf:
            identity=conf['identity'];cfg=identity['config']
            assert identity['environment']==job['environment'] and identity['seed']==job['seed']
            assert cfg['tau']==job['target_EMA'] and cfg['discount']==.999
            assert cfg.get('backup_samples',9)==cfg['num_samples']==9 and cfg['batch_size']==256
        complete=(run/'completed.json').exists()
        if complete:
            done=json.loads((run/'completed.json').read_text())
            assert done['step']==done['q_updates']==done['actor_updates']==1000000 and done['v_updates']==0
            trained+=1
        if (run/'stopped.json').exists():failures.append(dict(id=job['id'],reason='nonfinite_state'))
        for mode in plan['deployment_modes']:
            scores={}
            for step in plan['evaluation_steps']:
                path=Path(job['evaluation_directory'])/f'{step}_{mode}.json'
                if not path.exists():continue
                result=json.loads(path.read_text())
                assert conf and result['training_seed']==job['seed'] and result['environment']==job['environment']
                assert result['step']==step and result['policy_mode']==mode
                assert result['n_episodes']==len(result['episode_successes'])==50 and result['first_episode_seed']==42
                assert result['protocol']=='last3_cpu50_independent_seeded_episodes_v1' and result['no_optimizer_updates']
                assert result['config_sha256']==sha(run/'config.json')
                assert abs(result['success_rate']-st.mean(result['episode_successes']))<1e-12
                if args.final:
                    assert sha(run/f'checkpoint_{step}.pkl')==result['checkpoint_sha256']
                    assert sha(run/f'actor_{step}.pkl')==result['actor_artifact_sha256']
                scores[step]=100*result['success_rate'];evaluated+=1
            ready=len(scores)==3
            rows.append(dict(environment=job['environment'],code=job['code'],method=job['method'],
                target_EMA=job['target_EMA'],seed=job['seed'],deployment=mode,reused=job['reused'],
                training_complete=complete,evaluation_points=len(scores),
                last3_mean_pct=st.mean(scores.values()) if ready else None,
                success_800K=scores.get(800000),success_900K=scores.get(900000),success_1M=scores.get(1000000),
                late_range_pp=max(scores.values())-min(scores.values()) if ready else None,
                late_drop_pp=max(0,scores[800000]-scores[1000000]) if ready else None,run=job['id']))
    groups=[];paired=[]
    for env in plan['environments']:
        for mode in plan['deployment_modes']:
            baseline={r['seed']:r for r in rows if r['environment']==env and r['deployment']==mode and r['method']=='control'}
            for method in plan['methods']:
                selected=sorted([r for r in rows if r['environment']==env and r['deployment']==mode and r['method']==method],key=lambda r:r['seed'])
                assert len(selected)==2
                available=[r for r in selected if r['last3_mean_pct'] is not None]
                ready=len(available)==2;values=[r['last3_mean_pct'] for r in available]
                groups.append(dict(environment=env,code=selected[0]['code'],deployment=mode,method=method,
                    target_EMA=plan['rates'][method],complete_seeds=len(available),expected_seeds=2,
                    mean_pct=st.mean(values) if ready else None,sample_sd_pp=st.stdev(values) if ready else None,
                    worst_seed_pct=min(values) if ready else None,
                    seed0_pct=selected[0]['last3_mean_pct'],seed1_pct=selected[1]['last3_mean_pct'],
                    mean_late_range_pp=st.mean(r['late_range_pp'] for r in available) if ready else None,
                    mean_late_drop_pp=st.mean(r['late_drop_pp'] for r in available) if ready else None))
                if method=='control':continue
                for r in available:
                    b=baseline[r['seed']]
                    if b['last3_mean_pct'] is not None:
                        paired.append(dict(environment=env,deployment=mode,method=method,seed=r['seed'],
                            mean_change_pp=r['last3_mean_pct']-b['last3_mean_pct'],
                            late_range_change_pp=r['late_range_pp']-b['late_range_pp'],
                            late_drop_change_pp=r['late_drop_pp']-b['late_drop_pp']))
    complete=trained==plan['total_cells'] and evaluated==plan['total_cells']*6 and not failures
    if args.final:
        assert complete
        for field in ('source_sha256','reused_evidence_sha256'):
            for path,digest in plan[field].items():assert sha(path)==digest,path
    out=root/'analysis';out.mkdir(exist_ok=True)
    suffix='final_v2' if args.final else 'progress_v2'
    csv_file(out/f'per_seed_{suffix}.csv',rows)
    csv_file(out/f'task_summary_{suffix}.csv',groups)
    csv_file(out/f'paired_differences_{suffix}.csv',paired)
    result=dict(time=time.time(),complete=complete,training_complete=trained,total_cells=plan['total_cells'],
                primary_evaluations=evaluated,expected_primary_evaluations=plan['total_cells']*6,failures=failures,
                interpretation='Two paired training seeds; sample SD is descriptive and has high uncertainty.')
    temporary=out/f'summary_{suffix}.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(out/f'summary_{suffix}.json')
    lines=['Default four-task target-Q EMA sweep','',
        'gamma=.999, N=9, seed0/1. EMA=.005 existing control vs .001 vs .0005.',
        '각 시드의 800K/900K/1M CPU50 성공률 평균; ±는 두 학습 시드의 sample SD.',
        '시드 두 개이므로 안정성 개선의 확정적 통계 판단은 하지 않는다.','',
        f"완료 학습 {trained}/{plan['total_cells']} (기존 대조군 포함), 평가 {evaluated}/{plan['total_cells']*6}.",
        '환경 | EMA | Lean mean±SD | actor mean±SD']
    def format_group(group):
        return '대기' if group['mean_pct'] is None else f"{group['mean_pct']:.2f}±{group['sample_sd_pp']:.2f}"
    for code,env,_ in plan['families']:
        for method in plan['methods']:
            lookup={g['deployment']:g for g in groups if g['environment']==env and g['method']==method}
            lines.append(f"{code} | {plan['rates'][method]:g} | {format_group(lookup['lean'])} | {format_group(lookup['actor'])}")
    (out/f'RESULTS_KO_{suffix}.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
