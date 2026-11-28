"""Concise combined report after both robustness studies pass their audits."""
import argparse
import csv
import json
from pathlib import Path
from .analyze_v2 import boot, write_csv


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=root/'analysis'
    audit=json.loads((out/'audit.json').read_text())
    assert audit['complete'] and audit['state_profile_supplement']['complete']
    summary=read(out/'mean_sd.csv');state=read(out/'state_profile_mean_sd.csv')
    comparisons=read(out/'state_profile_comparison.csv');raw=read(out/'state_profile_per_seed.csv')
    points=json.loads((root/'registration.json').read_text())['points']
    original=read(out/'per_seed.csv');lookup={(r['case_id'],int(r['seed'])):r for r in original}
    pair_rows=[]
    for mode in ('iid','field'):
        for epsilon in (0.,.1,.25,.5,.75,.9,.95,.98,.995,.999):
            p=next(r for r in points if r['family']=='private' and r['panel_mode']==mode and r['label']=='FEQ-N9' and r['x']==epsilon)
            vals=[r for r in raw if r['mode']==mode and r['clip']=='False' and float(r['epsilon'])==epsilon]
            delta=[float(lookup[p['case_id'],int(r['seed'])]['tail_success'])-float(r['tail_success']) for r in vals]
            mean,lo,hi=boot(delta)
            pair_rows.append(dict(mode=mode,epsilon=epsilon,paired_seeds=len(vals),
                                 FEQ_minus_statewise_IQL_success=mean,ci_low=lo,ci_high=hi))
    write_csv(out/'statewise_paired_benefits.csv',pair_rows)
    def pick(family,mode,label,x):
        return next(r for r in summary if r['family']==family and r['panel_mode']==mode and r['label']==label and float(r['x'])==x)
    def pct(r,key='tail_success_mean'):
        return f"{100*float(r[key]):.2f}"
    lines=['FEQ/IQL 노이즈 강도 실험 — 최종 요약','2026-11-25 KST','',
        '새 시드200~215의16개 paired seed,주 실험2736회+상태별 대조640회=3376회 완료.',
        '13상태 chain,gamma=.999,tau=.9,N9,200K 업데이트. 마지막180K~200K의 시드별 평균.',
        '성공률은 공통 best9 행동 후보를 사용하는 정책의1000step 정확한 목표 도달확률이다.',
        'Q 조회 sigma0은 참 Q를 유지한다는 뜻이 아니며 학습으로 Q오차가 생길 수 있다.',
        '모델에서 직접 후보를 생성하는 성능 및 oracle-Q의 후보 coverage도 따로 저장했다.','',
        '같은 Q 조회잡음 비교: IQL에는 추가 V 잡음이 없다.',
        'mode | sigma_Q | IQL Q-only % | FEQ MLE % | FEQ progress-50% % | FEQ random-logits0.5 %']
    for mode in ('iid','field'):
        for sigma in (0.,.0005,.00075,.001,.0015,.002,.003):
            labels=['IQL Q-only','FEQ data MLE','FEQ progress -50%','FEQ random logits0.5']
            lines.append(f"{mode} | {sigma:g} | "+' | '.join(pct(pick('shared_Q',mode,label,sigma)) for label in labels))
    lines += ['', 'V-only와 Flow-only: 한 단계 source target RMS를 reference Q에서 맞춘 대조.',
        '전체 RMS만 맞추면 상태별 오차 위치가 다르다. 상태별 MSE를 추가로 맞춘 결과도 같이 본다.',
        'mode | depletion epsilon | matched RMS | IQL uniform V % | IQL statewise V % | FEQ % | IQL statewise clipped %']
    for mode in ('iid','field'):
        for epsilon in (.25,.5,.75,.9,.95,.98,.995,.999):
            r=next(v for v in comparisons if v['mode']==mode and v['clip']=='False' and float(v['epsilon'])==epsilon)
            c=next(v for v in state if v['mode']==mode and v['clip']=='True' and float(v['epsilon'])==epsilon)
            lines.append(f"{mode} | {epsilon:g} | {float(r['reference_target_rms']):.7f} | "
                f"{100*float(r['IQL_uniform_success']):.2f} | {100*float(r['IQL_statewise_success']):.2f} | "
                f"{100*float(r['FEQ_success']):.2f} | {100*float(c['tail_success_mean']):.2f}")
    lines += ['', '모델 오차 고유 강도 비교: 전진확률 손실과 random-logit model error.',
        'kind | strength | sigma_Q | mean TV(data) | common-proposal success % | model-proposal success % | oracle model coverage %']
    for r in summary:
        if r['family']!='flow':continue
        if r['error_kind']=='deplete' and float(r['x']) not in (0.,.5,.95,.98,.995,.999):continue
        lines.append(f"{r['error_kind']} | {r['x']} | {r['common_q_sigma']} | {float(r['flow_TV_data_mean']):.4f} | "
            f"{pct(r)} | {pct(r,'tail_model_success_mean')} | {pct(r,'oracle_model_goal_coverage_mean')}")
    lines += ['', '해석 범위',
        'V-only 오차만으로 IQL이 실패할 수 있다. FEQ의 부정확한 행동모델도 큰 오류에서는 실패한다.',
        '같은 critic readout sigma에서 상대 성능을 비교하는 결과와 V/flow의 private error 크기를 맞춘 결과는 서로 다른 대조다.',
        'Global RMS 비교의 격차에는 오차가 어느 상태에 발생하는지의 영향도 포함된다.',
        'depletion error의 참조 에너지는 출발 상태에 집중된다. state_profile_comparison.csv에 비중을 기록했다.',
        '상태별 MSE까지 맞추면 IQL이 유지되는 영역이 크게 넓어지며, 극단적인 모델 오차에서는 FEQ가 먼저 실패하는 경우도 있다.',
        '따라서 V error라는 이유만으로 같은 크기의 모든 flow/Q error보다 더 치명적이라는 일반 결론은 내리지 않는다.',
        'Statewise 대조에서도 V 오차는 centered IID/AR이고 Flow bias는 고정된 하향 편향이므로 시간구조/부호까지 같지는 않다.',
        '일부 행동모델 편향은 Q readout 상향 편향과 상쇄될 수 있어 오류 강도에 대한 성공률이 항상 단조롭지는 않다.',
        '관측 grid에서50% 아래로 처음 내려간 지점과 직전 지점을 기록했으며 정확한 임계값으로 보간하지 않았다.',
        '여기서 붕괴는 정책 성공률 저하를 뜻한다. 학습한 최종 Q의 NaN/Inf는 모든 조건에서 검사했으며 수치적 발산의 발생률을 비교한 것이 아니다.',
        'Mean/SD 및95% bootstrap CI는 독립 학습 시드16개 단위로 계산했다. 각 곡선의 실패 조건도 모두 포함한다.',
        '단일-Q tabular 가치 업데이트와 분포 오차의 controlled surrogate다. 신경망 flow/MDN actor 전체의 성능 증명이 아니다.',
        'Q*,V*,expectile Bellman 참값과 모든 원본 궤적/오차분해를 보존했다. 기존GPU 학습은 수정하거나 종료하지 않았다.','',
        '파일',
        'figures/shared_q_noise.png: 공통 Q 조회잡음에 대한 성공률/Q RMSE',
        'figures/matched_private_errors.png: 전체 RMS 매칭',
        'figures/state_profile_control.png: 상태별 RMS 매칭 대조',
        'figures/behavior_model_errors.png: 실제 Flow 모델 오차/후보 coverage',
        'analysis/mean_sd.csv: 모든 조건 mean/SD/CI',
        'analysis/noise_collapse.csv: 최초50%/80% crossing grid와 AUC',
        'analysis/statewise_paired_benefits.csv: statewise-matched FEQ-IQL paired CI',
        'analysis/RESULTS_KO.txt,STATE_PROFILE_KO.txt: 상세 정의와 전체 표',
        'analysis/audit.json: 전 실험/소스 해시 검증','',
        '재현은 기존 noise_sweep_v5 후 state_profile_v6,analyze_noise_v5 --final,finalize_noise_v6 순서다.',
        '사용한 실행기는 /home/ukjo1/anaconda3/bin/python이며 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1을 설정했다.',
        '새 output에서 CPU workers4/2를 사용한다. 코드와로그의 source hash가 다르면 기존 결과를 덮어쓰지 않는다.']
    (out/'FINAL_REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',trajectories=3376,report=str(out/'FINAL_REPORT_KO.txt'))),flush=True)


if __name__=='__main__':main()
