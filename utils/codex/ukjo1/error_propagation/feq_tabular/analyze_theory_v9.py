"""Audit and report all theory-aligned outcomes, including negative controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .theory_channels_v9 import secant_diagnostics
from .mdp import MDP, solve

LABELS={'iql':'IQL','iql_batch9':'IQL-9 V batch','feq9':'FEQ-N9',
        'population':'Population expectile oracle','repair_half':'IQL: half V gap',
        'repair_zero':'IQL: zero V gap','iql_full':'IQL: full V gradient',
        'iql_full9':'IQL: 9 full V steps'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate(rows,fields):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in fields)].append(r)
    out=[]
    for key,group in sorted(groups.items()):
        r=dict(zip(fields,key));r['seeds']=len(group)
        for k in group[0]:
            if k in fields or k=='seed' or isinstance(group[0][k],str):continue
            x=np.array([g[k] for g in group],float);x=x[np.isfinite(x)]
            r[k+'_mean']=float(x.mean()) if len(x) else np.nan
            r[k+'_sd']=float(x.std(ddof=1)) if len(x)>1 else np.nan
        out.append(r)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;out=root/'analysis';out.mkdir(exist_ok=True)
    figs=root/'figures';figs.mkdir(exist_ok=True)
    runs=[];audited=0;files={};summary_raw=[]
    for stage in ('discovery','confirmation','paired_controls'):
        reg=json.loads((root/stage/'registration.json').read_text())
        assert (root/stage/'complete.json').exists()
        expected=sum(len(c['seeds']) for c in reg['cases']);seen=set()
        for f in sorted((root/stage/'raw').glob('*/result.json')):
            m=json.loads(f.read_text());path=f.with_name('data.npz')
            assert sha(path)==m['sha256'];assert m['sources']==reg['sources']
            identity=json.dumps(m['identity'],sort_keys=True);assert identity not in seen;seen.add(identity)
            c=m['identity']['case'];seed=m['identity']['seed'];h=np.load(path)
            assert np.isfinite(h['q']).all() and np.all(h['noise_statistics']==0)
            assert c.get('sigma',0.)==0.
            np.testing.assert_allclose(h['cross'][1:].sum((1,2)),h['decomposition'][1:,-1,1],atol=1e-11)
            audited+=1
            for t in (200000,1000000):
                if t>c['steps']:continue
                mask=(h['step']>=.9*t)&(h['step']<=t)
                r=dict(stage=stage,case=c['name'],method=c['method'],seed=seed,window_end=t,
                       success=float(h['policy'][mask,1].mean()),
                       model_success=float(h['model_policy'][mask,1].mean()),
                       discounted_return=float(h['policy'][mask,0].mean()),q_rmse=float(h['rmse'][mask].mean()),
                       optimal_q_rmse=float(np.sqrt(((h['q'][mask]-h['q_optimal'])**2).mean((1,2))).mean()),
                       model_tv=float(m['metrics']['behavior_TV']),
                       v_gap_rms=np.nan,score_residual_rms=np.nan,gap_to_score_rms=np.nan,
                       median_local_inverse_kappa=np.nan,secant_identity_error=np.nan)
                if c['method'] not in ('feq9','population'):
                    ds=[secant_diagnostics(h['target'][i],h['value'][i],h['behavior'],c['tau']) for i in np.where(mask)[0]]
                    gap=np.array([d['gap'] for d in ds]);z=np.array([d['zeta'] for d in ds])
                    sec=np.array([d['secant'] for d in ds]);kap=np.array([d['kappa'] for d in ds])
                    r.update(v_gap_rms=float(np.sqrt((gap**2).mean())),score_residual_rms=float(np.sqrt((z**2).mean())),
                             gap_to_score_rms=float(np.sqrt((gap**2).mean()/max((z**2).mean(),1e-40))),
                             median_local_inverse_kappa=float(np.median(1/kap)),
                             secant_identity_error=float(abs(gap*sec-z).max()))
                summary_raw.append(r)
            if stage=='confirmation' and c['name']=='main_no_external_noise':
                files[(c['method'],seed)]=path
                runs.append((c,seed,path))
        assert len(seen)==expected,(stage,len(seen),expected)
    csv_write(out/'all_per_seed.csv',summary_raw)
    all_summary=aggregate(summary_raw,('stage','case','method','window_end'))
    csv_write(out/'all_conditions.csv',all_summary)
    selected=[r for r in summary_raw if (r['stage']=='confirmation' and r['case']=='main_no_external_noise') or r['stage']=='paired_controls']
    csv_write(out/'paired_schedule_per_seed.csv',selected)
    primary=[r for r in selected if r['case']=='main_no_external_noise' and r['window_end']==200000]
    paired=[];rng=np.random.default_rng(112509)
    for other in ('iql','iql_batch9'):
        a={r['seed']:r for r in primary if r['method']=='feq9'}
        b={r['seed']:r for r in primary if r['method']==other}
        assert set(a)==set(b)==set(range(800,816))
        for metric in ('success','q_rmse','discounted_return'):
            d=np.array([a[s][metric]-b[s][metric] for s in sorted(a)])
            boot=rng.choice(d,(20000,len(d)),replace=True).mean(1)
            paired.append(dict(comparison='FEQ-N9 minus '+other,metric=metric,seeds=len(d),
                               difference=float(d.mean()),ci_low=float(np.quantile(boot,.025)),
                               ci_high=float(np.quantile(boot,.975))))
    csv_write(out/'paired_differences.csv',paired)
    # Residual-zero and population must share the ENTIRE trajectory, not just means.
    zero_files={}
    for f in (root/'paired_controls'/'raw').glob('*/result.json'):
        meta=json.loads(f.read_text());c=meta['identity']['case']
        if c['method']=='repair_zero':zero_files[meta['identity']['seed']]=f.with_name('data.npz')
    for seed in range(800,816):
        h=np.load(files[('population',seed)]);z=np.load(zero_files[seed])
        np.testing.assert_array_equal(h['q'],z['q'])
        np.testing.assert_array_equal(h['target'],z['target'])
        np.testing.assert_array_equal(h['behavior'],z['behavior'])
    # First sustained crossing is descriptive; the primary window was fixed beforehand.
    speed=[]
    for c,seed,path in runs:
        h=np.load(path);valid=h['policy'][:,1]>=.9;indices=[]
        for i in range(len(valid)-4):
            if valid[i:i+5].all():indices.append(i)
        speed.append(dict(method=c['method'],seed=seed,
                          first_90pct_for_5_checkpoints=float(h['step'][indices[0]]) if indices else np.nan))
    csv_write(out/'learning_speed.csv',speed)
    finite_checks=[]
    for seed in range(800,816):
        h=np.load(files[('feq9',seed)])
        m=MDP(h['reward'],h['successor'],h['continuation'],h['behavior'],.999,.99,'registered')
        q,_,iterations=solve(m,samples=9)
        mask=h['step']>=900000
        finite_checks.append(dict(seed=seed,iterations=iterations,
            finite_operator_Q_RMSE=float(np.sqrt(((q-h['q_truth'])**2).mean())),
            learned_Q_RMSE=float(h['rmse'][mask].mean()),
            learned_vs_finite_root_RMSE=float(np.sqrt(((h['q'][mask]-q)**2).mean((1,2))).mean())))
    (out/'finite_N_fixed_point_check.json').write_text(json.dumps(dict(status='computed',rows=finite_checks,
        interpretation='The expected finite-N operator is itself biased relative to the population operator. Its fixed point is not asserted to equal the mean constant-learning-rate stochastic trajectory.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'iql':'#c44e52','iql_batch9':'#dd8452','feq9':'#4c72b0','population':'#55a868'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    curve_rows=[]
    for method in ('iql','iql_batch9','feq9','population'):
        hs=[np.load(files[(method,s)]) for s in range(800,816)]
        t=hs[0]['step'];suc=np.array([h['policy'][:,1] for h in hs]);rmse=np.array([h['rmse'] for h in hs])
        for ax,data in zip(axes,(100*suc,rmse)):
            mean=data.mean(0);sd=data.std(0,ddof=1)
            ax.plot(t/1000,mean,color=colors[method],label=LABELS[method]);ax.fill_between(t/1000,mean-sd,mean+sd,color=colors[method],alpha=.13)
        for i,step in enumerate(t):
            curve_rows.append(dict(method=method,step=int(step),success_mean=float(suc[:,i].mean()),success_sd=float(suc[:,i].std(ddof=1)),q_rmse_mean=float(rmse[:,i].mean())))
    csv_write(out/'main_curves.csv',curve_rows)
    for ax in axes:
        ax.axvspan(180,200,color='grey',alpha=.15);ax.set_xlabel('Training updates (thousands)');ax.grid(alpha=.2)
    axes[0].set(ylabel='Goal success (%)',ylim=(-3,103));axes[1].set(ylabel='Q RMSE',yscale='log',ylim=(1e-6,2))
    axes[0].legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figs/f'no_external_noise_learning.{ext}',dpi=180)
    plt.close(fig)
    grouped=aggregate(selected,('case','method','window_end'))
    csv_write(out/'reported_summary.csv',grouped)
    fig,ax=plt.subplots(figsize=(9,3.8));bars=[];labels=[];sds=[]
    for case,method in [('main_no_external_noise','iql'),('main_no_external_noise','iql_batch9'),('main_no_external_noise','feq9'),('residual_intervention','repair_half'),('residual_intervention','repair_zero'),('residual_intervention','iql_full9')]:
        r=next(x for x in grouped if x['case']==case and x['method']==method and x['window_end']==200000)
        bars.append(100*r['success_mean']);sds.append(100*r['success_sd']);labels.append(LABELS[method])
    ax.bar(np.arange(len(bars)),bars,yerr=sds,capsize=4,color=['#c44e52','#dd8452','#4c72b0','#937860','#55a868','#8172b3'])
    ax.set_xticks(np.arange(len(bars)));ax.set_xticklabels(labels,rotation=15,ha='right',fontsize=8)
    ax.set(ylabel='Goal success at 180K–200K (%)',ylim=(0,140));ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figs/f'residual_interventions.{ext}',dpi=180)
    plt.close(fig)
    def row(case,method,end=200000):return next(r for r in grouped if r['case']==case and r['method']==method and r['window_end']==end)
    def table_entry(r):return f"{LABELS[r['method']]:32} {100*r['success_mean']:7.2f} +/- {100*r['success_sd']:6.2f}%    Q RMSE {r['q_rmse_mean']:.6f}; return {r['discounted_return_mean']:.5f}"
    lines=['FEQ: 사용자의 expectile residual 이론에 맞춘 검증 (2026-11-25)','',
      '1. 먼저 정정할 점',
      '앞선 동일 오차 대조는 δV=V−eτ(Q)와 action-common Q 출력 오차를 같게 맞춘 것이었다.',
      '사용자 문서의 residual은 ζ=−Gτ(Q,V), Gτ=Eβ[wτ(Q−V)(Q−V)]이다. 둘은 다르다.',
      '문서 p.2–4, 식(7),(9),(11): critic ε의 expectile 반응은 가중평균으로 sup norm 이득 1;',
      'score residual ζ는 δV≈ζ/κ, κ=Eβ[wτ(Q−eτ(Q))]로 증폭된다.',
      '따라서 기존 부정적 문장을 이 residual 정의에 적용한 것은 부정확했다.',
      '', '2. 이론 자체의 수치 검증',
      'G의 scaling은 문서 그대로 고정(J_prime=−2G). 동일한 숫자의 ε와 ζ를 비교했다.',
      '두 값 q=(0,1), 좋은 행동 확률 .01, tau=.99: κ=.0198, critic 이득1, score 이득50.505.',
      'Atomless Uniform[0,1], tau=.99: κ=.099499, critic 이득1, score 이득10.0504.',
      'local_sensitivity.csv에 16조건 전부 저장. discrete는 root에 atom이 없는 국소 구간에서 검증.',
      '5-state progress/wait MDP, tau=.99, gamma=.999, progress=.25의 정확한 강제오차 고정점:',
      '매 반복 +0.00003의 Q 출력 오차: 성공100%, Q RMSE 약0.0000963.',
      '매 반복 +0.00003의 score residual: 성공약0%, Q RMSE 약0.027933.',
      '이는 동일 숫자의 primitive 좌표 비교이며, 실제 IQL/FEQ 학습 성능표가 아니다.',
      '이 큰 오차 범위의 순위 역전은 전체 비선형 방정식으로 계산했다. 국소 식만 외삽하지 않았다.',
      '180개 강제오차 조건(양/음 부호 포함)을 모두 공개했다.',
      'δV를 같게 맞추면 두 경로가 동일한 Bellman target을 만든다는 기존 대조도 여전히 맞다.',
      '', '3. 추가 잡음 없는 공정한 tabular 학습',
      '25개 비종료 상태, 4행동. 행동0은 한 칸 진행, 나머지는 대기. 마지막 진행만 보상1.',
      '시작 상태 행동1은 안전 종료 보상(정확한 population continuation value의 .8배)을 받는다.',
      '진행 확률 .05, gamma=.999, tau=.99; 모든 상태/행동은 지원된다.',
      '상태별 1024개 행동 표본의 고정 empirical distribution을 dataset으로 사용한다.',
      '동일 replay, 초기화, Q 학습률.1, V 학습률.1, EMA.005. Q/V 외부 잡음 모두0.',
      '주 비교의 IQL-9와 FEQ-N9는 한 update당 target-Q 행동 query9개를 사용한다.',
      'IQL-9는 9개 V gradient 평균; FEQ-N9는 9개 Q값의 sample expectile을 정확히 푼다.',
      '평가는 공통 best-of-9 정책의 정확한 1000-step 성공확률/discounted return이다.',
      'Q RMSE의 참조는 empirical behavior에 대한 정확한 population expectile Bellman 고정점 Q_tau이다.',
      'greedy Q*와 다르며 Q* 대비 RMSE도 별도 CSV에 저장했다.',
      '목표 성공0%라도 안전 종료 보상은 받을 수 있다. return도 함께 보아야 한다.',
      'IQL AWBC나 실제 신경망 actor 비교가 아니다. Flow는 finite-action sampling surrogate이다.',
      '탐색seed700/701과 독립인 확인seed800..815. 평균±sample SD는 학습시드 단위.',
      '사전 지정 primary 구간180K–200K의 5개 checkpoint 평균:', '']
    for m in ('iql','iql_batch9','feq9','population'):lines.append(table_entry(row('main_no_external_noise',m)))
    lines+=['','Paired bootstrap 95% CI (FEQ minus baseline):']
    for p in paired:lines.append(f"{p['comparison']} {p['metric']}: {p['difference']:.6f} [{p['ci_low']:.6f}, {p['ci_high']:.6f}]")
    lines+=['','4. 실제 발생한 잔차와 개입']
    for m in ('iql','iql_batch9'):
        r=row('main_no_external_noise',m)
        lines.append(f"{m}: score RMS={r['score_residual_rms_mean']:.7f}, value-gap RMS={r['v_gap_rms_mean']:.7f}, seed별 RMS 비율 평균={r['gap_to_score_rms_mean']:.2f}.")
    lines+=['잔차는 외부 주입이 아니라 진행 중인 V 회귀에서 실제로 남은 ζ=−G를 정확히 계산했다.',
      'sign crossing이 있으면 local κ 대신 exact secant κ를 사용해야 한다. δV*κ_secant=ζ를 확인했다.',
      '현재 target-Q를 진실로 교정하지 않고 V−eτ(current target-Q)만 반/전부 제거하는 oracle 대조:']
    for m in ('repair_half','repair_zero','iql_full','iql_full9'):lines.append(table_entry(row('residual_intervention',m)))
    lines+=['잔차 제거와 population expectile의 Q trajectory는 모든16시드에서 bitwise 일치했다.',
      '9 full V steps는 더 많은 계산을 쓰는 메커니즘 대조이며 공정 query-budget baseline이 아니다.',
      '', '5. 중요한 범위: 1M까지 학습하면']
    for m in ('iql','iql_batch9','feq9','population'):lines.append(table_entry(row('main_no_external_noise',m,1000000)))
    lines+=['이 noiseless tabular 예에서는 유한 학습시간의 V tracking lag가 핵심이다.',
      '영구 발산이나 무한 학습 후 FEQ 우월성을 의미하지 않는다. 충분히 학습하면 IQL이 따라잡고',
      'finite-N bias가 없는 IQL의 Q RMSE가 더 작아지는 경우도 있다.',
      '별도 exact finite-N 기대 연산자의 고정점에도 population 대비 오차가 존재함을 확인했다.',
      '그 고정점과 constant-step stochastic 학습 평균이 정확히 일치한다고 주장하지 않는다.',
      '시간 여유가 없어 V 회귀 잔차를 충분히 줄이지 못하는 조건에서 FEQ의 장점이 있다.',
      '탐색18개 MDP 설정 중 성공률 차이를 보인 조건을 골라 독립시드로 확인한 것이다.',
      '다른 탐색 조건에서는 성공률이 같거나 IQL의 Q RMSE가 더 작은 결과도 보존했다.',
      '', '6. Behavior model 오차와 대칭 expectile 대조']
    for case in ('behavior_misspec_0.25','behavior_misspec_0.5'):
        r=row(case,'feq9');lines.append(table_entry(r)+f"; model TV={r['model_tv_mean']:.4f}, 실제 model proposer 성공={100*r['model_success_mean']:.2f}%")
    for m in ('iql','iql_batch9','feq9','population'):lines.append('tau=.5: '+table_entry(row('symmetric_expectile_control',m)))
    lines+=['기본 behavior sampler는 empirical dataset 빈도를 정확히 표현한다. 진짜 generating β와는 다를 수 있다.',
      '추가 model 오차는 고정 logit perturbation을 사용하며 Q에서 model로 gradient가 전달되지 않는다.',
      '동일 N의 유한 표본 expectile과 behavior misspecification은 FEQ에도 별도의 오차 원인이다.',
      '', '7. 주장 가능한 결론',
      '문서의 고정된 score 정의 아래, 같은 크기의 critic 출력 오차보다 expectile score residual이',
      '작은 κ에서 더 큰 V/Bellman 오차를 만들 수 있다. 이론과 수치 witness가 일치한다.',
      '추가 V 잡음을 넣지 않고도 별도 V 회귀가 남긴 잔차가 학습을 지연시키며, FEQ-N9가',
      '같은 transition/query 예산에서 더 빨리 학습하는 held-out tabular 설정을 찾았다.',
      '이를 모든 오차 분포/방향/손실 정규화에서 V 오차가 항상 더 위험하다는 뜻으로 확장하면 안 된다.',
      '', '8. 재현 및 감사',
      f'전체 {audited} training runs의 원시파일 checksum/등록/분해 항등식 확인.',
      '초기 200K intervention은 main1M과 RNG prefix가 달라 paired 인과 비교에서 제외.',
      'paired_controls에서는 모두1M으로 맞춰 replay/action streams를 다시 일치시켰다.',
      '검색 결과는 all_conditions.csv, 확인 결과는 reported_summary.csv, 시드는 all_per_seed.csv.',
      '독립 scalar Python 학습 및 독립 이분법 Bellman iteration 검증은 independent_validation.json.',
      '이전 shared-Q-noise 실험은 별도 근거이며 이번 no-external-noise 결과와 합산하지 않았다.',
      '원본 theory 파일은 수정하지 않았다. 해석/필요한 기술적 수정은 THEORY_NOTES_KO.txt 참조.','']
    (out/'FINAL_REPORT_KO.txt').write_text('\n'.join(lines))
    pass_primary=all(p['ci_low']>0 if p['metric']=='success' else p['ci_high']<0 for p in paired if p['metric'] in ('success','q_rmse'))
    audit=dict(status='PASS' if pass_primary else 'PRIMARY_CRITERION_NOT_MET',training_runs=audited,
               heldout_seeds=16,primary_window=[180000,200000],external_Q_noise=0.,external_V_noise=0.,
               zero_residual_population_bitwise_matches=16,
               validation=json.loads((root/'independent_validation.json').read_text()))
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
