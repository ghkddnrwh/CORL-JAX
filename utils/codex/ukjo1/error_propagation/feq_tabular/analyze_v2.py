"""Audit and summarize independent FEQ confirmation; all outputs go under logs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json
from pathlib import Path
import statistics as st
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def boot(values,seed=7171):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    samples=x[rng.integers(len(x),size=(20000,len(x)))].mean(-1)
    return float(x.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args();r=a.root.resolve()
    plan=json.loads((r/'confirmation_registration_v2.json').read_text());rows=[];files={};groups={}
    for case in plan['conditions']:
        for seed in case['seeds']:
            for meta in (r/'raw'/case['name']/f'seed{seed}').glob('*.json'):
                obj=json.loads(meta.read_text());assert obj['identity']['case']==case and obj['identity']['seed']==seed
                assert obj['source_sha256']==plan['source_sha256']
                assert obj['metrics']['identity_error']<1e-10
                if a.final:assert sha(meta.with_suffix('.npz'))==obj['data_sha256']
                rows.append(obj['metrics']);files[(case['name'],seed,obj['identity']['label'])]=meta.with_suffix('.npz')
    for row in rows:groups.setdefault((row['case'],row['method']),[]).append(row)
    summary=[]
    metrics=('tail_q_rmse','early_100k_q_rmse','tail_return','tail_success','final_q_rmse','q_rmse_optimal',
             'behavior_TV','extra_flow_TV','target_prior_Q_rms','target_dataset_rms','target_V_fit_rms',
             'target_flow_fit_rms','target_sampling_rms','target_Q_readout_rms','target_clipping_rms')
    for (case,method),values in groups.items():
        expected=next(c for c in plan['conditions'] if c['name']==case)
        record=dict(case=case,method=method,seeds=len(values),expected_seeds=len(expected['seeds']),complete=len(values)==len(expected['seeds']))
        for key in metrics:
            x=[v[key] for v in values];record[key+'_mean']=st.mean(x);record[key+'_sd']=st.stdev(x) if len(x)>1 else None
        summary.append(record)
    pairs=[]
    for case in plan['conditions']:
        iid={v['seed']:v for v in groups.get((case['name'],'iql'),[])}
        feq={v['seed']:v for v in groups.get((case['name'],'feq_n9'),[])}
        keys=sorted(iid.keys()&feq.keys())
        if not keys:continue
        record=dict(case=case['name'],paired_seeds=len(keys),complete=len(keys)==len(case['seeds']))
        for metric,sign in (('tail_q_rmse',1),('early_100k_q_rmse',1),('tail_success',-1),('tail_return',-1)):
            delta=[sign*(iid[s][metric]-feq[s][metric]) for s in keys]
            mean,low,high=boot(delta);record[metric+'_benefit_mean']=mean;record[metric+'_benefit_ci_low']=low;record[metric+'_benefit_ci_high']=high
        pairs.append(record)
    out=r/'analysis';out.mkdir(exist_ok=True);suffix='final_v2' if a.final else 'progress_v2'
    write_csv(out/f'per_seed_{suffix}.csv',rows);write_csv(out/f'mean_sd_{suffix}.csv',summary);write_csv(out/f'paired_benefits_{suffix}.csv',pairs)
    primary={c:next((v for v in pairs if v['case']==c),None) for c in ('main_iid','main_field','main_clean_long')}
    passed=all(primary[c] and primary[c]['complete'] and primary[c]['tail_q_rmse_benefit_ci_low']>0 and primary[c]['tail_success_benefit_ci_low']>0 for c in ('main_iid','main_field'))
    status=dict(status='PASS' if passed else 'PENDING_OR_FAIL',confirmation_trajectories=len(rows),expected=plan['tasks'],
                full_confirmation_complete=len(rows)==plan['tasks'],primary_criterion_passed=passed,primary_paired_statistics=primary)
    if a.final:
        assert len(rows)==plan['tasks'],status
        assert (r/'confirmation_completed_v2.json').exists()
        assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    (out/f'audit_{suffix}.json').write_text(json.dumps(status,indent=2)+'\n')
    lines=['FEQ Known-MDP 독립 검증', '',f"학습 {len(rows)}/{plan['tasks']}, 주 검증 통과={passed}",
           '학습 시드100~131(32개). 발견 시드0~5와 겹치지 않는다.',
           'Q RMSE의 참값은 행동분포에 대한 population-expectile Bellman 고정점 Q_tau다. Q*는 별도로 저장했다.',
           '성공률은 공통 best9 정책이1000스텝 안에 goal에 도달할 확률을 전이행렬로 정확히 계산한다.',
           '최종 수치는900K~1M 관찰 시점 평균을 각 시드에서 계산하고32시드 평균±sample SD로 집계한다.', '',
           '조건 | 방법 | Q RMSE | goal 성공률(%) | 할인 return']
    for case in ('main_iid','main_field','main_clean_long'):
        for method in ('iql','feq_n9','feq_population'):
            v=next((x for x in summary if x['case']==case and x['method']==method),None)
            if not v or not v['complete']:continue
            lines.append(f"{case} | {method} | {v['tail_q_rmse_mean']:.6f} ± {v['tail_q_rmse_sd']:.6f} | {100*v['tail_success_mean']:.3f} ± {100*v['tail_success_sd']:.3f} | {v['tail_return_mean']:.6f}")
    lines+=['','같은 업데이트 예산 비교이며, Q조회/계산량/모델수까지 같다는 주장은 아니다.',
            '오차 있는 조건은 Q 및 V에 centered readout approximation error를 명시적으로 주입한 모형이다.',
            '행동모델은 동일 offline 상태별1024개 행동의 MLE를 조건부 단조1D flow로 구현한다.',
            'Flow matching 신경망과 MDN actor 전체를 검증한 것이 아니라 FEQ 가치 업데이트 메커니즘의 tabular 검증이다.',
            'V를 완전히 최적화하면 IQL과 population FEQ는 같은 고정점이며, FEQ-N9에는 유한 표본 오차가 남는다.',
            'Q->expectile->Q 재귀는 FEQ에도 있다. 별도 V-회귀 오차 경로가 제거되는 것으로 해석한다.',
            'Q와 무관한 behavior fitting은 IQL에도 부착할 수 있어 그 독립성만으로 FEQ의 우월성을 증명하지 않는다.',
            '발견/확인/모든 대조군 원본은 유지하며 불리한 조건도 mean_sd 및 discovery CSV에 포함한다.',
            'IQL 원논문: https://arxiv.org/abs/2110.06169']
    (out/f'RESULTS_KO_{suffix}.txt').write_text('\n'.join(lines)+'\n')
    if a.final:
        plot(r,files,summary)
    print(json.dumps(status),flush=True)


def plot(root,files,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    colors={'iql':'#d55e00','feq_n9':'#0072b2','feq_population':'#009e73'}
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex='col')
    for col,case in enumerate(('main_clean_long','main_iid','main_field')):
        for method in colors:
            series=[];success=[]
            for seed in range(100,132):
                with np.load(files[case,seed,method]) as z:
                    steps=z['step'];series.append(z['metrics'][:,0]);success.append(z['goal_success'])
            for row,values in ((0,series),(1,success)):
                x=np.array(values);mean=x.mean(0);sd=x.std(0,ddof=1)
                axes[row,col].plot(steps/1000,mean,label=method,color=colors[method])
                axes[row,col].fill_between(steps/1000,np.maximum(1e-14,mean-sd) if row==0 else np.maximum(0,mean-sd),mean+sd if row==0 else np.minimum(1,mean+sd),color=colors[method],alpha=.15)
        axes[0,col].set_title(case);axes[0,col].set_yscale('log');axes[1,col].set_xlabel('Replay updates (thousands)')
        axes[0,col].grid(alpha=.15);axes[1,col].grid(alpha=.15);axes[1,col].set_ylim(-.02,1.02)
    axes[0,0].set_ylabel('Q RMSE to exact Q_tau');axes[1,0].set_ylabel('Exact goal probability within1000steps');axes[0,2].legend(fontsize=9)
    fig.tight_layout();fig.savefig(folder/'learning_curves_32seeds.png',dpi=200);fig.savefig(folder/'learning_curves_32seeds.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,case in zip(axes,('main_iid','main_field')):
        labels=[];values=[]
        for method in colors:
            row=next(v for v in summary if v['case']==case and v['method']==method)
            for name in ('V_fit','flow_fit','sampling','Q_readout'):
                labels.append(method+'\n'+name);values.append(row['target_'+name+'_rms_mean'])
        ax.bar(np.arange(len(values)),values,color=[colors[m] for m in colors for _ in range(4)])
        ax.set_xticks(np.arange(len(values)));ax.set_xticklabels(labels,rotation=65,ha='right',fontsize=7);ax.set_title(case);ax.set_ylabel('RMS target-error component')
    fig.tight_layout();fig.savefig(folder/'target_components.png',dpi=200);fig.savefig(folder/'target_components.pdf');plt.close(fig)


if __name__=='__main__':main()
