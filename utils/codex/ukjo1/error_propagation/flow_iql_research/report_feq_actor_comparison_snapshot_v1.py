"""Read-only audit of runs; write a dated partial/final comparison snapshot."""
import argparse
import json
import pickle
import statistics as st
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from utils.codex.ukjo1.error_propagation.flow_iql_research import manage_feq_actor_comparison_v1 as study
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import atomic, sha, write_csv


def flatten(tree, prefix=''):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from flatten(tree[key], prefix+'/'+key)
    elif isinstance(tree, (tuple, list)):
        for i, value in enumerate(tree):
            yield from flatten(value, prefix+'/'+str(i))
    else:
        yield prefix, np.asarray(tree)


def base_comparison(new, old):
    with new.open('rb') as f:
        new_base = pickle.load(f)['state']['base']
    with old.open('rb') as f:
        old_base = pickle.load(f)['state']['base']
    results = {}
    for module in ('modules_actor_flow', 'modules_critic', 'modules_target_critic'):
        a = dict(flatten(new_base['agent']['network']['params'][module]))
        b = dict(flatten(old_base['agent']['network']['params'][module]))
        assert a.keys() == b.keys()
        squared = total = maximum = 0.
        exact = True
        for key in a:
            assert a[key].shape == b[key].shape
            diff = a[key].astype(float)-b[key].astype(float)
            squared += float((diff*diff).sum()); total += diff.size
            maximum = max(maximum, float(np.abs(diff).max()))
            exact &= np.array_equal(a[key], b[key])
        results[module] = dict(exact=bool(exact), rms_difference=(squared/total)**.5, max_abs_difference=maximum)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=study.ROOT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root=args.root.resolve(); out=args.output.resolve()
    assert not out.exists(), 'Use a new snapshot directory'
    out.mkdir(parents=True)
    plan=json.loads((root/'registration.json').read_text())
    status=json.loads((root/'controller_status.json').read_text())
    for path, expected in json.loads((root/'study_sources.json').read_text()).items():
        assert sha(path)==expected, path
    jobs=[json.loads(p.read_text()) for p in (root/'jobs').glob('*.json')]
    rows=[]; evidence={}; trained=0; evaluated=0; running=[]
    for job in jobs:
        run=Path(job['output'])
        conf=json.loads((run/'config.json').read_text())
        assert conf['identity']['job']==job
        cfg=conf['identity']['config']
        assert cfg['discount']==.995 and cfg['backup_samples']==9 and cfg['tau']==.005
        for path, expected in conf['source_sha256'].items():
            if path not in evidence: evidence[path]=sha(path)
            assert evidence[path]==expected
        if (run/'completed.json').exists():
            done=json.loads((run/'completed.json').read_text())
            assert done['q_updates']==done['actor_updates']==done['step']==1000000
            trained+=1
        elif (run/'runtime.json').exists():
            running.append(dict(id=job['id'], **json.loads((run/'runtime.json').read_text())))
        for name, spec in job['actors'].items():
            values=[]
            for step in study.STEPS:
                path=study.eval_path(root,job,name,step)
                if not path.exists(): continue
                result=study.read_eval(path,job,name,step)
                assert result['config_sha256']==sha(run/'config.json')
                artifact=run/f'actor_{name}_{step}.pkl'
                assert result['actor_artifact_sha256']==sha(artifact), artifact
                evidence[str(path)]=sha(path)
                values.append(result['success_rate']*100); evaluated+=1
            if job['stage']=='final' and len(values)==3:
                rows.append(dict(code=job['code'], method=spec['method'], seed=job['seed'], alpha=spec['alpha'],
                    success_800K=values[0], success_900K=values[1], success_1M=values[2], success_pct=st.mean(values)))
    for ref in plan['reused_mdn']:
        values=[]
        for e in ref['evaluations']:
            assert sha(e['path'])==e['sha256']
            result=json.loads(Path(e['path']).read_text())
            assert result['training_seed']==ref['seed'] and result['environment']==ref['environment']
            assert result['n_episodes']==50 and result['first_episode_seed']==42
            assert len(result['episode_successes'])==50
            assert abs(st.mean(result['episode_successes'])-result['success_rate'])<1e-12
            values.append(result['success_rate']*100)
        rows.append(dict(code=ref['code'], method='mdn', seed=ref['seed'], alpha='',
            success_800K=values[0], success_900K=values[1], success_1M=values[2], success_pct=st.mean(values)))
    complete_codes=[c for c,_ in study.FAMILIES if all(sum(r['code']==c and r['method']==m for r in rows)==4 for m in ('mdn','fql','cvae'))]
    summary=[]; paired=[]
    for code,_ in study.FAMILIES:
        for method in ('mdn','fql','cvae'):
            selected=[r for r in rows if r['code']==code and r['method']==method]
            values=[r['success_pct'] for r in selected]
            summary.append(dict(code=code,method=method,complete_seeds=len(values),
                mean_pct=st.mean(values) if len(values)==4 else '', sample_sd=st.stdev(values) if len(values)==4 else ''))
            if method!='mdn' and len(values)==4:
                d=[next(r['success_pct'] for r in selected if r['seed']==s)-next(r['success_pct'] for r in rows if r['code']==code and r['method']=='mdn' and r['seed']==s) for s in range(4)]
                paired.append(dict(code=code,method=method,mean_delta_pp=st.mean(d),sample_sd_delta=st.stdev(d),
                    **{f'seed{s}_delta_pp':v for s,v in enumerate(d)}))
    macro=[]
    for method in ('mdn','fql','cvae'):
        values=[st.mean(r['success_pct'] for r in rows if r['code'] in complete_codes and r['method']==method and r['seed']==s) for s in range(4)]
        macro.append(dict(method=method,tasks=len(complete_codes),mean_pct=st.mean(values),sample_sd=st.stdev(values),
            **{f'seed{s}_pct':v for s,v in enumerate(values)}))
    tuning=[]
    for path in sorted((root/'selections').glob('*.json')):
        selection=json.loads(path.read_text())
        assert selection['tuning_seed']==4 and selection['no_final_seed_results_used']
        for method, item in selection['selection'].items():
            for candidate in item['candidates']:
                for e in candidate['evidence']: assert sha(e['path'])==e['sha256']
                tuning.append(dict(code=selection['code'],method=method,alpha=candidate['alpha'],
                    tuning_success_pct=100*candidate['mean_success'],selected=candidate==item['selected'],
                    selected_at_grid_boundary=item['boundary_selected']))
    base_audit=[]
    for ref in plan['reused_mdn']:
        if ref['code'] not in complete_codes: continue
        new=root/'runs'/f"final_{ref['code']}_s{ref['seed']}"/'checkpoint_1000000.pkl'
        old=Path(ref['run'])/'checkpoint_1000000.pkl'
        base_audit.append(dict(code=ref['code'],seed=ref['seed'],modules=base_comparison(new,old)))
    exact=all(v['exact'] for r in base_audit for v in r['modules'].values())
    moment=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    for name,data in [('per_seed',rows),('task_mean_sd',summary),('paired_delta_vs_mdn',paired),('macro_complete_tasks',macro),('alpha_tuning',tuning)]:
        write_csv(out/f'{name}.csv',data)
    atomic(out/'audit.json',dict(time_kst=moment,training_complete=trained,expected_training=60,
        evaluation_complete=evaluated,expected_evaluations=420,completed_codes=complete_codes,running=running,
        controller_failed=status['failed'],same_1M_Q_Flow_parameters_as_MDN=exact,
        matched_final_checkpoints=len(base_audit),base_audit=base_audit,result_source_hashes=evidence))
    lines=[f'FEQ actor 비교 결과 ({moment})',f'완료 학습 {trained}/60, 평가 {evaluated}/420, 완료 환경 {len(complete_codes)}/10.',
        'gamma=.995, N=9. Alpha는 별도 training seed4로 선택; 본 비교는 seed0..3.',
        '800K/900K/1M에서 각각 50회 평가한 성공률을 시드 안에서 평균. 아래는 네 학습 시드의 평균 +/- 표본 SD.',
        f'완료 {len(base_audit)}개 final run의 1M Q/target-Q/Flow가 재사용 MDN과 정확히 동일: {exact}.','',
        '환경 | MDN | FQL-loss noise actor | CVAE TD3+BC-style actor']
    for code,_ in study.FAMILIES:
        cells=[]
        for method in ('mdn','fql','cvae'):
            r=next(x for x in summary if x['code']==code and x['method']==method)
            cells.append(f"{r['mean_pct']:.2f} +/- {r['sample_sd']:.2f}" if r['complete_seeds']==4 else '평가 대기')
        lines.append(code+' | '+' | '.join(cells))
    lines+=['',f'세 방법 모두 완료된 {len(complete_codes)}환경의 동일 가중 평균:']
    lines += [f"{r['method']}: {r['mean_pct']:.2f} +/- {r['sample_sd']:.2f}" for r in macro]
    lines+=['','해석: FQL-loss actor는 Cube-Single/Puzzle3x3에서 네 시드 모두 MDN보다 높다.',
        'AM-L/HM-L/Puzzle4x4에서는 MDN이 네 시드 모두 FQL-loss actor보다 높다.',
        'CVAE는 이 loss/architecture/alpha 탐색 범위에서 navigation과 Scene의 성능이 낮다.',
        'CVAE의 여러 환경에서 alpha 탐색 하한 .25가 선택되어, 더 약한 Q guidance 가능성은 미검증이다.',
        'FQL/TD3+BC 전체 알고리즘 결과가 아니라 FEQ critic/Flow를 유지한 actor 학습 방식의 비교다.',
        '비교의 한계: 새 FQL-loss/CVAE actor는 동일한 Q/Flow를 공유하지만, 재사용 MDN과는 최종 Q 파라미터가 다르다.',
        'MDN 대비 값은 동일 설정의 별도 재학습 결과 비교이며, 모든 차이를 actor만의 효과로 분리할 수 없다.',
        '원인을 분리하려면 MDN도 새 두 actor와 동일한 Q/Flow 학습 안에서 함께 학습하는 대조가 필요하다.',
        '단일 tuning seed와 세 alpha 후보만 사용했으므로 actor 종류의 보편적 우열이나 전역 최적 alpha를 뜻하지 않는다.',
        ('AntSoccer를 포함한 모든 학습/평가가 완료됐다.' if trained==60 and evaluated==420
         else '미완료 학습/평가는 기존 controller 상태를 확인한다.')]
    (out/'RESULTS_KO.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(complete_codes)); width=.25
    for i,(method,label,color) in enumerate((('mdn','MDN','#0072b2'),('fql','Noise MLP / FQL loss','#d55e00'),('cvae','CVAE / TD3+BC-style loss','#009e73'))):
        rs=[next(r for r in summary if r['code']==c and r['method']==method) for c in complete_codes]
        ax.bar(x+(i-1)*width,[r['mean_pct'] for r in rs],width,yerr=[r['sample_sd'] for r in rs],capsize=2,color=color,alpha=.8,label=label)
        for j,code in enumerate(complete_codes):
            vals=[r['success_pct'] for r in sorted(rows,key=lambda r:r['seed']) if r['code']==code and r['method']==method]
            ax.scatter(j+(i-1)*width+np.linspace(-.035,.035,4),vals,color='black',s=9,zorder=3)
    ax.set_xticks(x);ax.set_xticklabels(complete_codes);ax.set_ylabel('Success (%)');ax.set_ylim(0,106)
    ax.grid(axis='y',alpha=.15);ax.set_title(f'FEQ gamma=0.995, N=9: {len(complete_codes)}/10 tasks complete; seeds 0-3')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=3)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');fig.savefig(out/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(output=str(out),training_complete=trained,evaluations_complete=evaluated,
        complete_tasks=len(complete_codes),macro=macro,base_parameters_exact=exact),indent=2))


if __name__=='__main__': main()
