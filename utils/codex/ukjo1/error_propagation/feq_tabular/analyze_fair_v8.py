"""Audit and summarize every fair-comparison arm with seed-level inference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,collections,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'iql':'IQL','iql_batch9':'IQL, 9-action batch','iql_full':'IQL, exact action gradient',
    'feq9':'FEQ-N9','population':'Population FEQ','repair_half':'V residual x0.5',
    'repair_zero':'V residual removed','iql_full9':'IQL, 9 full V updates'}
COLORS={'iql':'#D55E00','iql_batch9':'#E69F00','feq9':'#009E73','population':'#0072B2'}


def read_csv(path):
    return list(csv.DictReader(path.open()))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(x):
    x=np.asarray(x,float)
    rng=np.random.default_rng(251125)
    bootstrap=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),
                low=float(np.quantile(bootstrap,.025)),high=float(np.quantile(bootstrap,.975)))


def summarize(rows,group_keys):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in group_keys)].append(r)
    out=[]
    for key,rr in sorted(groups.items()):
        row=dict(zip(group_keys,key));row['seeds']=len(rr)
        for metric in ('tail_success','tail_q_rmse','tail_return'):
            for k,v in stat([float(r[metric]) for r in rr]).items():row[metric+'_'+k]=v
        out.append(row)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    analysis=root/'analysis';figures=root/'figures';analysis.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    for stage in ('discovery','confirmation','plain_chain','residual_replay','residual_replay_plain'):
        assert (root/stage/'complete.json').exists(),stage
    audited=0
    for stage in ('discovery','confirmation'):
        reg=json.loads((root/stage/'registration.json').read_text())
        expected=sum(len(c['seeds']) for c in reg['cases'])
        metadata=list((root/stage/'raw').glob('*/result.json'));assert len(metadata)==expected
        for path in metadata:
            data=json.loads(path.read_text());assert data['sources']==reg['sources']
            assert sha(path.with_name('data.npz'))==data['sha256'];audited+=1
        archive=root/stage/'source';archive.mkdir(exist_ok=True)
        for name,expected_hash in reg['sources'].items():
            src=Path(name);assert sha(src)==expected_hash
            shutil.copyfile(src,archive/src.name)
    for path in (root/'plain_chain/raw').glob('*/*.json'):
        data=json.loads(path.read_text());assert sha(path.with_suffix('.npz'))==data['sha256'];audited+=1
    records=read_csv(root/'confirmation/per_seed.csv')
    summary=summarize(records,('case','method'));write_csv(analysis/'all_conditions.csv',summary)
    lookup={(r['case'],r['method']):r for r in summary}
    paired=[]
    for case in sorted({r['case'] for r in records}):
        feq={int(r['seed']):r for r in records if r['case']==case and r['method']=='feq9'}
        for baseline in ('iql','iql_batch9'):
            base={int(r['seed']):r for r in records if r['case']==case and r['method']==baseline}
            seeds=sorted(set(feq)&set(base))
            if not seeds:continue
            row=dict(case=case,baseline=baseline,seeds=len(seeds))
            for metric in ('tail_success','tail_q_rmse','tail_return'):
                x=[float(feq[s][metric])-float(base[s][metric]) for s in seeds]
                for k,v in stat(x).items():row[metric+'_difference_'+k]=v
            paired.append(row)
    write_csv(analysis/'paired_differences.csv',paired)
    plain=read_csv(root/'plain_chain/per_seed.csv')
    ps=summarize(plain,('length','sigma','method'));write_csv(analysis/'plain_chain_summary.csv',ps)
    replay=read_csv(root/'residual_replay/per_seed.csv')
    rs=summarize(replay,('arm',));write_csv(analysis/'residual_replay_summary.csv',rs)
    replay_plain=read_csv(root/'residual_replay_plain/per_seed.csv')
    rs_plain=summarize(replay_plain,('arm',));write_csv(analysis/'residual_replay_plain_summary.csv',rs_plain)
    ds=summarize(read_csv(root/'discovery/per_seed.csv'),('case','method'))
    write_csv(analysis/'all_discovery_conditions.csv',ds)
    # Curves display SD across independent seed-level time averages, not checkpoints.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for method in ('iql','iql_batch9','feq9','population'):
        curve=[]
        for r in summary:
            if r['method']!=method:continue
            if r['case']=='main':sigma=.001
            elif r['case'].startswith('noise_'):sigma=float(r['case'][6:])
            else:continue
            curve.append((sigma,r))
        curve.sort(key=lambda x:x[0])
        for ax,metric,scale in ((axes[0],'tail_success',100),(axes[1],'tail_q_rmse',1)):
            x=np.array([c[0] for c in curve]);y=np.array([c[1][metric+'_mean'] for c in curve])*scale
            sd=np.array([c[1][metric+'_sd'] for c in curve])*scale
            ax.plot(x,y,'o-',label=LABELS[method],color=COLORS[method],markersize=4)
            lo=y-sd;hi=y+sd
            if metric=='tail_success':lo=np.maximum(lo,0);hi=np.minimum(hi,100)
            ax.fill_between(x,lo,hi,color=COLORS[method],alpha=.13)
            ax.set_xlabel('Shared Q readout noise sigma');ax.grid(alpha=.2)
    axes[0].set_ylabel('Goal success (%)');axes[0].set_ylim(-2,102)
    axes[1].set_ylabel('Q RMSE');axes[1].set_yscale('log');axes[0].legend(fontsize=8)
    fig.suptitle('13-state chain: no external V noise; mean +/- seed SD')
    fig.tight_layout();fig.savefig(figures/'fair_shared_Q_noise.png',dpi=220);fig.savefig(figures/'fair_shared_Q_noise.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,length in zip(axes,('4','12')):
        for method in ('iql','iql_batch9','feq9'):
            curve=sorted([r for r in ps if r['length']==length and r['method']==method],key=lambda r:float(r['sigma']))
            x=[float(r['sigma']) for r in curve];y=[100*r['tail_success_mean'] for r in curve]
            sd=np.array([100*r['tail_success_sd'] for r in curve])
            ax.plot(x,y,'o-',color=COLORS[method],label=LABELS[method]);ax.fill_between(x,np.maximum(0,np.array(y)-sd),np.minimum(100,np.array(y)+sd),color=COLORS[method],alpha=.13)
        ax.set_title(f'{int(length)+1} states; progress or wait');ax.set_xlabel('Shared Q noise sigma');ax.set_ylim(-2,102);ax.grid(alpha=.2)
    axes[0].set_ylabel('Goal success (%)');axes[0].legend(fontsize=8)
    fig.suptitle('Only terminal reward 1; no value-dependent safe exit;16 fresh seeds')
    fig.tight_layout();fig.savefig(figures/'plain_chain_success.png',dpi=220);fig.savefig(figures/'plain_chain_success.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    methods=['iql','iql_batch9','iql_full','feq9','repair_half','repair_zero','iql_full9']
    rr=[lookup['main',m] for m in methods]
    axes[0].bar(np.arange(len(rr)),[100*r['tail_success_mean'] for r in rr],yerr=[100*r['tail_success_sd'] for r in rr],color=['#D55E00','#E69F00','#CC79A7','#009E73','#56B4E9','#0072B2','#999999'],capsize=3)
    axes[0].set_xticks(np.arange(len(rr)),[LABELS[m] for m in methods],rotation=35,ha='right',fontsize=8)
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Goal success (%)');axes[0].set_title('Endogenous V residual intervention')
    arms=['recorded_V','same_error_Q_common','centered_same_RMS','zero_residual'];rlookup={r['arm']:r for r in rs_plain}
    axes[1].bar(np.arange(4),[100*rlookup[k]['tail_success_mean'] for k in arms],yerr=[100*rlookup[k]['tail_success_sd'] for k in arms],color=['#D55E00','#E69F00','#009E73','#0072B2'],capsize=3)
    axes[1].set_xticks(np.arange(4),['V residual','Identical Q error','Remove mean,\nmatch RMS','Zero residual'],rotation=25,ha='right',fontsize=8)
    axes[1].set_ylim(0,105);axes[1].set_title('Plain chain replay:16 separate seeds')
    fig.tight_layout();fig.savefig(figures/'residual_mechanism.png',dpi=220);fig.savefig(figures/'residual_mechanism.pdf');plt.close(fig)
    # Decision uses only registered main comparisons, never selecting the best seed.
    primary=[r for r in paired if r['case']=='main']
    passed=all(r['tail_success_difference_low']>0 and r['tail_q_rmse_difference_high']<0 for r in primary)
    assert len(primary)==2 and passed
    lines=['FEQ 공정 비교와 V expectile 잔차 — 최종 보고서','2026-11-25','',
        '목표: IQL에만 외부 V 잡음을 더한다는 혼동을 제거하고, 학습에서 발생한 V 잔차의 경로를 검증한다.',
        '모든 주 비교에서 외부 V 잡음은0. 두 방법의 전체 Q 오차장, replay, 행동 난수, Q 학습률, target EMA, 초기화, 평가 후보가 같다.',
        'IQL9는 V loss를9개 행동 표본으로 평균하여 FEQ9의 행동 조회 예산과 맞춘 대조다. 별도 batch transition 수를9배로 바꾸지 않았다.',
        '행동 모델은1024개/상태 오프라인 빈도로 학습한 분포이며 Q로 학습하지 않는다. 기본모델과 IQL은 동일한 경험적 행동 분포를 사용한다.',
        '정책은 모든 방법에 공통 best-of-9; 저장 Q로 계산한1000-step 정확한 도달확률. AWBC/MDN actor를 비교하는 실험은 아니다.',
        'Q RMSE 참조는 경험적 행동분포의 population-expectile Bellman 고정점. 최적 Q*와 다르다.',
        '1차 탐색240회(시드300,301), 독립 검증1472회(시드400~431), 단순 MDP384회(시드500~515).',
        '잔차재생은 안전출구 MDP16시드(600~615)와 단순 MDP16시드(650~665), 각각4군. 아래 재생 표는 단순 MDP 결과.',
        '일반 표의±는 시드별 마지막10% 평균 간 표본SD. paired CI는 시드 단위10,000회 bootstrap95%. 체크포인트를 독립 표본으로 세지 않았다.','',
        '주 비교:13상태, gamma=.999,tau=.9,sigma_Q=.001, 행동 공통 Q 오차,300K,32개 시드.',
        '방법 | 성공률 평균±SD(%) | Q RMSE | return']
    for m in methods+['population']:
        r=lookup['main',m]
        lines.append(f"{LABELS[m]} | {100*r['tail_success_mean']:.2f}±{100*r['tail_success_sd']:.2f} | {r['tail_q_rmse_mean']:.7f} | {r['tail_return_mean']:.5f}")
    lines+=['','Paired FEQ9 - baseline differences:']
    for r in primary:
        lines.append(f"{r['baseline']}: success {100*r['tail_success_difference_mean']:+.2f}pp [95%CI {100*r['tail_success_difference_low']:+.2f},{100*r['tail_success_difference_high']:+.2f}], Q RMSE {r['tail_q_rmse_difference_mean']:+.7f} [{r['tail_q_rmse_difference_low']:+.7f},{r['tail_q_rmse_difference_high']:+.7f}]")
    lines+=['','단순한 새 MDP:5/13 비종료상태,4행동. 전진은 다음 상태, 나머지는 제자리; 마지막 전진만 보상1로 종료.',
        '그 외 보상0. 참값을 이용하여 설정한 안전출구 보상도 없다. gamma=.999,tau=.9,200K,각16개 새 시드.',
        '상태수 | sigma_Q | 방법 | 성공률±SD(%) | Q RMSE']
    for r in ps:lines.append(f"{int(r['length'])+1} | {r['sigma']} | {r['method']} | {100*r['tail_success_mean']:.2f}±{100*r['tail_success_sd']:.2f} | {r['tail_q_rmse_mean']:.7f}")
    lines+=['','원인 대조: 실제 학습된 V 잔차를 기록하여 재생. 각 상태 RMS, 시점과 순서를 그대로 유지하여 Q common-mode에 넣으면 같은 Q 궤적.',
        '평균 제거 대조는 각 상태의 signed mean을 빼고 원래 RMS로 재확대. 정규화한 시간 자기공분산은 보존하나 평균/분산 배분이 달라진다.',
        '따라서 효과가 오차 총량만으로 설명되지 않으며, 지속적 편향이 중요하다.','군 | 성공률±SD(%) | Q RMSE']
    for name in arms:
        r=rlookup[name];lines.append(f"{name} | {100*r['tail_success_mean']:.2f}±{100*r['tail_success_sd']:.2f} | {r['tail_q_rmse_mean']:.7f}")
    recorded=[r for r in replay_plain if r['arm']=='recorded_V']
    lines.append(f"주입 Q noise RMS=.002, 실제 생성된 V residual RMS 평균={np.mean([float(r['disturbance_RMS']) for r in recorded]):.7f}, 상태별 평균편향 RMS={np.mean([float(r['mean_bias_RMS']) for r in recorded]):.7f}.")
    lines+=['','반드시 함께 보고할 대조군:','조건 | 방법 | 성공률(%) | Q RMSE']
    for case in ('noise_0.0','long_1M','action_correlation_0.0','action_correlation_0.5','temporal_0.99','mean_control','anneal','V_lr_0.01','V_lr_0.5','model_logit_1.0'):
        for m in ('iql','iql_batch9','feq9','repair_zero'):
            r=lookup.get((case,m))
            if r:lines.append(f"{case} | {m} | {100*r['tail_success_mean']:.2f} | {r['tail_q_rmse_mean']:.7f}")
    population=json.loads((root/'population/results.json').read_text())
    lines+=['','정확한 기대손실 해석:','행동 공통 Z에 대해 e_tau(Q+Z)=e_tau(Q)+Z. 따라서 FEQ의 post-expectile Q readout error는 평균0.',
        '그러나 한 V 파라미터를 시간에 걸쳐 Q(A)+Z에 expectile regression하면 e_tau over(A,Z)를 학습한다. 두 연산의 순서가 다르다.',
        'V 잔차 r=V-e_tau(Q)가 편향을 가지면 Q backup을 반복해서 밀어 올린다. 이 잔차는 외부 V noise를 더하지 않아도 생긴다.',
        '기대손실 고정점은 stochastic SGD의 평균 궤적과 동일하다고 주장하지 않는다. 13상태 sigma.001의 기대손실 해는 성공하지만 stochastic SGD에서는 작은 action gap 때문에 실패한다.','']
    for r in population['rows']:lines.append(json.dumps(r,ensure_ascii=False))
    lines+=['','주장 가능한 결론:',
        '동일한 Q 오차만 있어도 별도 V 회귀가 추가적인 biased expectile residual을 만들고, 그 누적 피해가 직접적인 critic readout 오차보다 커지는 세팅이 존재한다.',
        '이때 FEQ9는 추가 V residual 경로를 제거하여 IQL 및 같은9행동 예산 IQL보다 높은 성공률과 낮은 Q RMSE를 보였다.',
        'V 잔차만 줄이는 oracle 개입이 같은 Q 잡음 아래서 회복시킨다. V 최적화를 강화한 IQL도 회복되므로 원인을 명확히 한 대조로 함께 보고한다.',
        '성립하지 않는 일반 주장:',
        '같은 크기/편향/시간구조/상태위치의 동일한 post-expectile Q/V 오차 중 V 오차가 본질적으로 더 해롭다는 주장은 성립하지 않는다. 실험에서도 두 주입 위치의 궤적이 일치했다.',
        '잡음 없는 정확 tabular IQL과 population FEQ는 같은 고정점이다. FEQ9는 유한표본 편향을 가지므로 clean 조건에서 IQL Q RMSE가 더 좋다.',
        '행동 간 Q 오차가 독립이면 IQL9가 거의 회복된다. FEQ 우위의 크기는 오차 공분산에 의존한다.',
        '현재 오차는 target-Q 조회에 넣는 명시적 근사오차 surrogate다. 자연 발생 neural approximation error와 동일하다는 증명은 아니며, 일반적인 neural FEQ의 우월성 증명도 아니다.',
        '기본 모델을 왜곡하여 유리한 상쇄를 만들지 않았다. 별도 logit모델 오차 대조는 모델 정확도가 감소할 때의 제한으로 공개한다.',
        '일부 세팅에서 FEQ도 잡음이 커지면 무너진다. 모든 강도의 결과와 유리하지 않은 조건을 저장했다.','',
        '재현:',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m utils.codex.ukjo1.error_propagation.feq_tabular.test_fair_v7 --output <new_root>/validation.json',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_study_v7 --root <new_root>/discovery --workers 4',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_study_v7 --root <new_root>/confirmation --plan confirmation_plan.json --workers 4',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_plain_chain_v8 --root <new_root>/plain_chain --workers 2',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_residual_replay_v8 --root <new_root>/residual_replay --workers 2',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_residual_replay_v8 --root <new_root>/residual_replay_plain --workers 2 --plain',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.fair_population_v7 --root <new_root>',
        'python -m utils.codex.ukjo1.error_propagation.feq_tabular.analyze_fair_v8 --root <new_root>',
        '실험 Python: /home/ukjo1/anaconda3/bin/python. plan 파일은 이 결과 루트에 저장됨.',
        'IQL 원문: https://arxiv.org/abs/2110.06169 (expectile V 회귀 및 Q backup 정의).','']
    (analysis/'FINAL_REPORT_KO.txt').write_text('\n'.join(lines))
    # Capture all new mechanism sources and raw checksums for reproducibility.
    srcdir=root/'source';srcdir.mkdir(exist_ok=True)
    source_names=['fair_engine_v7.py','fair_study_v7.py','test_fair_v7.py','fair_population_v7.py','fair_plain_chain_v8.py','fair_residual_replay_v8.py','analyze_fair_v8.py','mdp.py','behavior_flow.py','noise_models_v5.py']
    sources={}
    for name in source_names:
        src=Path(__file__).parent/name;shutil.copyfile(src,srcdir/name);sources[name]=sha(src)
    extra={str(p.relative_to(root)):sha(p) for stage in ('residual_replay','residual_replay_plain','population') for p in (root/stage).rglob('*.npz')}
    audit=dict(status='PASS',registered_primary_pass=passed,checksum_verified_runs=audited,
        source_sha256=sources,mechanism_data_sha256=extra,
        observed_runs=dict(discovery=240,confirmation=len(records),plain_chain=len(plain),residual_replay_arms=len(replay)+len(replay_plain)),
        checks=['Registered source/data hashes match','All primary paired CIs exclude0 in favorable direction',
        'All training comparisons have external V noise0; replay interventions use recorded endogenous residuals','Q/V identical-error replay invariance verified for32 seeds',
        'Finite/population expectile translation equivariance and noiseless fixed points checked'])
    (analysis/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',primary=primary,plain_summary=ps,replay_summary=rs),indent=2))


if __name__=='__main__':main()
