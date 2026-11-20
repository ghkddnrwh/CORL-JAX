"""Standalone completed-study figures and an outcome-neutral Korean report."""
from pathlib import Path
import numpy as np
from utils.flow_iql_research.analyze_discount_0995_n9_v2 import csv_file


def render(root,plan,rows,groups,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'analysis'; figures=out/'figures'; figures.mkdir(exist_ok=True)
    labels={'control':'Baseline','slow_target':'EMA .001','critic_batch1024':'Q batch1024',
            'dr3_0p001':'DR3 .001','dr3_0p01':'DR3 .01'}
    tasks=dict(zip(plan['environments'],['AM-L','HM-M','Cube-Double','Puzzle4x4']))
    effects=[]; rng=np.random.default_rng(20261120)
    for env in plan['environments']:
        for mode in plan['deployment_modes']:
            fig,ax=plt.subplots(figsize=(8,4))
            for i,method in enumerate(plan['methods']):
                rs=sorted([r for r in rows if r['environment']==env and r['deployment']==mode and r['method']==method],key=lambda r:r['seed'])
                scores=np.array([r['last3_mean_pct'] for r in rs]); assert len(scores)==5
                ax.errorbar(i,float(scores.mean()),yerr=float(scores.std(ddof=1)),fmt='o',color='black',capsize=5)
                ax.scatter(i+np.linspace(-.12,.12,5),scores,s=25,alpha=.8)
                if method!='control':
                    ds=np.array([r['mean_change_pp'] for r in paired if r['environment']==env and r['deployment']==mode and r['method']==method])
                    assert len(ds)==5
                    boot=ds[rng.integers(0,5,size=(20000,5))].mean(1)
                    low,high=np.quantile(boot,[.025,.975])
                    effects.append(dict(environment=env,deployment=mode,method=method,paired_mean_change_pp=float(ds.mean()),
                        exploratory_bootstrap_95_low=float(low),exploratory_bootstrap_95_high=float(high),
                        improved_seed_count=int((ds>0).sum()),seed_count=5,
                        caveat='Five paired seeds; exploratory percentile interval, not adjusted for multiple comparisons.'))
            ax.set_xticks(range(len(plan['methods'])));ax.set_xticklabels([labels[m] for m in plan['methods']],rotation=15)
            ax.set_ylim(0,100);ax.set_ylabel('Last-3 success (%)');ax.set_title(f'{tasks[env]} / {mode}: five seeds, mean ± sample SD')
            ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figures/f'{tasks[env]}_{mode}_seeds.png',dpi=180);plt.close(fig)
    csv_file(out/'paired_effects_exploratory_final_v1.csv',effects)
    lines=['네 환경 critic 안정화 실험: 최종 결과','',
        'N=9, gamma=.999, expectile=.9, 학습 seed0/1/2/3/4. AM-L min, 나머지 mean; TD target clipping 유지.',
        '각 시드의 800K/900K/1M CPU50 성공률을 평균하고 다섯 학습 시드의 sample SD를 표시한다.',
        '보조 CPU10 곡선은 주 통계에 포함하지 않는다. 전체 100셀, 기존기준선11개 재사용, 신규89개.',
        '같은 체크포인트의 Lean best9와 단일 actor를 별도로 집계한다.', '',
        '환경 | 배포 | 방법 | 평균±SD(%) | 최악 시드 | 1M 평균 | 평균 후반 범위(pp)']
    for g in groups:
        lines.append(f"{tasks[g['environment']]} | {g['deployment']} | {labels[g['method']]} | {g['mean_pct']:.2f}±{g['sample_sd_pp']:.2f} | {g['worst_seed_pct']:.2f} | {g['final_mean_pct']:.2f} | {g['mean_late_range_pp']:.2f}")
    flags=[g for g in groups if g['descriptive_stability_candidate']]
    lines+=['','사전 정의한 기술적 기준을 만족한 셀:',
        '평균 및 최악 시드가 기준선 이상, SD 감소, 후반 범위가 기준선 이하인 경우다. 통계적 유의성 판정이 아니다.']
    lines += [f"{tasks[g['environment']]} / {g['deployment']} / {labels[g['method']]}" for g in flags] or ['없음']
    lines+=['','해석 제한:',
        '큰 critic 배치는 같은 갱신 횟수에서 critic 표본·계산량이 증가한다. 동일 계산량 우월성을 뜻하지 않는다.',
        'DR3는 유한 expectile 가중 다음 표현에 적용한 변형이다. 표현 통계 개선만으로 성공률 개선이나 원인 증명을 주장하지 않는다.',
        '기존 재사용 기준선에는 새 표현 진단이 없으므로 paired feature 비교는 진단이 함께 존재하는 동일 시드에 한정한다.',
        '고정 후보는 replay 상태에서 순위 변화를 측정하는 전용 도구이며 학습·배포에 사용하지 않았다.',
        '다섯 시드의 탐색적 비교이고 여러 조건을 시험했다. 환경별 최고 결과만 모아 일반적인 안정화로 주장하지 않는다.']
    (out/'FINAL_REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
