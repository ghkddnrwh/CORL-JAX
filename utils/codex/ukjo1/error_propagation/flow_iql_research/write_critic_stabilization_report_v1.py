"""Final evidence report; requires all registered training and evaluations."""
import argparse
import csv
import json
from pathlib import Path
import time

from utils.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,sha


def run(path):
    plan=load_plan(path)
    root=path.parent
    result=json.loads((root/'analysis/results_v2.json').read_text())
    post=json.loads((root/'analysis/postprocess_complete_v2.json').read_text())
    controller=json.loads((root/'controller_status_v1.json').read_text())
    selectors=json.loads((root/'analysis/selector_comparison_v1.json').read_text())
    early=json.loads((root/'analysis/early_curve_summary_v1.json').read_text())
    assert result['status']==post['status']==controller['phase']=='complete'
    assert controller['training_complete']==12 and controller['evaluation_complete']==48 and not controller['failures']
    assert len(result['continuation_audits'])==12
    assert all(a['all_probes_finite'] and a['replay_rng_exact'] and a['agent_rng_exact'] and
               a['BC_flow_max_abs_difference']==0. for a in result['continuation_audits'])
    assert sum(a['diagnostic_points'] for a in result['continuation_audits'])==480
    for case in plan['cases']:
        for step,digest in case['parent_checkpoints'].items():
            assert sha(Path(case['parent_run'])/f'checkpoint_{step}.pkl')==digest
    for job in plan['jobs']:
        records=[json.loads(x) for x in (Path(job['output'])/'diagnostics.jsonl').read_text().splitlines()]
        assert [x['step'] for x in records]==list(range(805000,1000001,5000))
        assert json.loads((Path(job['output'])/'completed.json').read_text())['session_updates']==200000
    with (root/'analysis/per_seed_results_v2.csv').open() as stream:raw=list(csv.DictReader(stream))
    assert len(raw)==42
    names={'baseline':'기존','q_lr_0p1':'Q 학습률 1/10','q_cosine':'Q cosine 감쇠',
           'temporal_Q_average':'800K/현재 Q 점수 평균','online_mean_selector':'현재 두 head 평균'}
    environments=sorted({r['environment'] for r in result['summary']})
    task=lambda env:'HM-M (N=18)' if env.startswith('humanoid') else 'Puzzle 4x4 (N=27)'
    lookup=lambda env,m,d:next(r for r in result['summary'] if (r['environment'],r['method'],r['deployment'])==(env,m,d))
    pm=lambda r,key:f"{r[key+'_mean']:.2f} ± {r[key+'_sd']:.2f}"
    lines=[
        'Finite-mean critic 안정화 실험 결과',
        '',
        'Q 학습률 감소와 후반 cosine 감쇠만으로 두 환경의 시드 민감성이 해결되지는 않았다. '
        'Q 순위 변동 감소와 정책 성능 안정화는 서로 다른 결과였다. '
        '과거/현재 Q 점수 평균은 HM-M의 1M Lean 배포 정책에는 도움이 됐지만, 학습 자체 또는 두 환경 전체의 안정화로 해석할 수 없다.',
        '',
        'HM-M N=18, Puzzle 4x4 N=27에서 학습 seed 0/1/2를 사용했다. N은 bootstrap next-action 후보 수다. '
        'critic head 수는 2, Lean 실행 후보 수와 actor 교사 후보 수는 각각 9다. gamma=0.999, expectile=0.9다. '
        '각 원래 800K checkpoint의 파라미터·Adam 상태·replay RNG·agent RNG를 복원하고 1M까지 학습했다. '
        '두 개입 × 두 환경 × 세 시드 = 12개 분기, 총 2.4M 추가 갱신이다. 처음부터 재학습한 12개 독립 시드는 아니다.',
        '',
        'Q 학습률 1/10은 3e-4에서 3e-5로 내린다. Adam gradient를 줄이는 대신 최종 Q 갱신량만 줄인다. '
        'Cosine은 800K 이후 Q 학습률을 3e-4에서 1M의 0까지 감쇠한다. '
        'BC Flow와 likelihood actor 학습률은 계속 3e-4, target EMA는 0.005와 원래의 old-online 갱신 순서를 유지한다.',
        '',
        '평가는 checkpoint마다 독립 reset seed 42..91의 50 episode, 정확히 4 CPU affinity cores를 사용했다. '
        '아래 수치는 성공률(%)이며 ±는 세 학습 시드 간 sample SD다. '
        '주 지표는 각 시드의 800K/900K/1M 성공률 평균이다. 1M의 성능과 최악 시드도 함께 판단한다.',
        '',
        '환경\t방법\tLean 3시점 평균\tLean 1M\tActor 3시점 평균\tActor 1M']
    for env in environments:
        for method in ('baseline','q_lr_0p1','q_cosine'):
            a,b=lookup(env,method,'lean'),lookup(env,method,'actor')
            lines.append('\t'.join([task(env),names[method],pm(a,'late3_mean'),pm(a,'success_1M'),
                                    pm(b,'late3_mean'),pm(b,'success_1M')]))
    hm=environments[0]
    assert hm.startswith('humanoid')
    lines += ['', 'HM-M의 학습률 1/10은 평균이 올랐지만 시드 안정화로 볼 수 없었다. '
        'Lean 1M의 seed 0/1/2는 기존 82/6/10%에서 72/90/2%로 바뀌었다. '
        '평균 상승은 주로 seed 1의 큰 회복에서 나왔고 나머지 두 시드는 낮아졌다. '
        'Actor는 기존 72/10/50%에서 34/80/18%가 됐다. 평균은 같아도 성공한 시드가 바뀌었다.',
        '', 'Cosine 역시 HM-M seed 1의 900K Lean 86%를 1M 0%로 떨어뜨렸다. '
        '학습률이 마지막에 0이 되는 것만으로 그 이전의 정책 붕괴를 막지는 못했다.',
        '', '고정 상태·행동 후보에서의 Q 순위 변화 (900K→1M):',
        '환경\t방법\targmax 변경률(%)\t행동 공통 offset 제거 Q 변화 RMS']
    for env in environments:
        for method in ('baseline','q_lr_0p1','q_cosine'):
            value=lookup(env,method,'lean')
            lines.append(f"{task(env)}\t{names[method]}\t{100*value['rank_flip_900K_1M_mean']:.2f}\t{value['action_centered_Q_drift_rms_mean']:.4f}")
    lines += ['', '이 진단은 각 시드의 고정 replay 상태 512개와 고정 800K Flow 후보 9개를 사용했다. '
        '모든 방법과 기준선을 동일한 CPU backend에서 다시 채점했다. '
        '순위 변경률 감소는 올바른 순위 유지 또는 성공률 향상을 보장하지 않았다. Oracle Q 오류를 측정한 실험은 아니다.',
        '', '원래 1M 모델에 대한 배포 대조 (새 학습 없음):',
        '환경\t선택 방법\tLean 1M 평균 ± SD\t최악 시드\t최고 시드']
    for row in selectors['summary']:
        lines.append(f"{task(row['environment'])}\t{names[row['method']]}\t{pm(row,'success_1M')}\t{row['worst_seed']:.2f}\t{row['best_seed']:.2f}")
    lines += ['', '시점 Q 평균은 각 후보에서 min(twin Q)를 계산한 뒤 800K/현재 값을 50:50 평균한다. '
        '현재 Flow가 만드는 동일한 9개 후보를 사용하며, 두 Q forward가 필요하다. '
        '이 평균 Q를 교사로 actor를 다시 학습하지 않았으므로 actor 안정화 효과로 해석하지 않는다.',
        '현재 두 head 평균은 같은 현재 checkpoint의 두 online Q를 평균하여 선택한다. '
        '이 대조는 학습률 실험의 실패를 확인한 뒤 등록한 탐색적 추가 비교이며, 1M에서만 평가했다.',
        '', '시점 Q 평균의 HM-M 1M 성공률은 32.67±42.77%에서 62.67±10.07%로 개선됐다. '
        '그러나 원래 3시점 평균 기준의 SD는 14.24에서 24.43으로 증가했다. '
        '평가 시점 하나의 배포 개선과 학습 전체의 안정화를 구분해야 한다. '
        'Puzzle에서는 최종 평균이 같고 SD가 증가했다.',
        '', '원래 학습의 보조 곡선 (유지된 다섯 checkpoint만 측정):',
        '환경\tseed\t배포\t100K\t300K\t800K\t900K\t1M']
    for row in early['rows']:
        lines.append('\t'.join([task(row['environment']),str(row['seed']),row['deployment']]+
            [f"{row['success_'+str(s)]:.1f}" for s in (100000,300000,800000,900000,1000000)]))
    lines += ['', 'HM-M의 300K Lean은 세 시드 모두 76~82%였지만 이후 큰 하락이 있었다. '
        'Puzzle은 이미 300K에서 seed 0/1/2가 68/6/36%로 갈라졌다. '
        '단순히 학습 시간을 늘리면 모든 사례가 해결된다고 볼 수 없으며, 800K 이후 개입은 더 이른 학습 경로 차이를 검증하지 않는다.',
        '', '검증과 보존:',
        '12개 분기 모두 replay RNG와 agent RNG가 원래 실행과 정확히 같고, BC Flow 파라미터 차이는 0이었다. '
        'Q 갱신과 그에 따라 달라지는 actor 교사 경로가 개입 대상이었다. '
        '480개 등록 진단 지점에서 모든 학습 상태가 유한했다. V fit 수는 0이다.',
        '개입을 끈 구현은 두 실제 checkpoint에서 8회 갱신 뒤 전체 상태가 기존 구현과 bitwise 동일했다. '
        'Q 갱신량 1/10 검증은 fp32 두 weight ULP 이내였으며 첫 갱신의 Flow·actor·Adam 모멘트·target Q는 동일했다. '
        'v1 검증의 고정 절대 허용오차만 v2의 ULP 기준으로 수정했으며 학습 코드는 바꾸지 않았다.',
        '배포 대조의 현재-Q-only 제어는 여섯 모델 모두 원래 50 episode의 성공·보상·길이 배열과 '
        '64개 고정 입력의 행동을 정확히 재현했다. 기존 모델·소스·평가 파일은 SHA로 재확인했다.',
        '', '범위와 결론:',
        '이번 학습률 개입으로 Q 변동은 줄일 수 있었지만 두 환경 전반의 시드 민감성이 해결되지는 않았다. '
        '시점 Q 평균은 HM-M의 최종 Lean 정책에 유망한 배포 대조였다. '
        '수치적 유한성, 순위 변동 감소, 성공률 향상, 시드 편차 감소를 같은 결과로 취급해서는 안 된다.',
        '이 결과는 관찰 후 고른 두 환경/N와 기존 세 시드의 후반 분기 연구다. '
        '추가 독립 시드·처음부터 학습·N 전체 sweep 또는 모든 checkpoint 사이의 연속적 안정화는 검증하지 않았다. '
        '평가 episode를 학습 시드의 독립 반복으로 세지 않는다. '
        'gamma, 유한 표본 target 실현, 생성 행동에 대한 Q 일반화의 근본적 역할은 이번에 분리하지 않았다.',
        '', '재집계:',
        'python -m utils.flow_iql_research.rescore_critic_stabilization_rankings_v1 --plan REGISTRATION --require-complete',
        'python -m utils.flow_iql_research.analyze_critic_stabilization_v2 --plan REGISTRATION --require-complete',
        'python -m utils.flow_iql_research.summarize_critic_stabilization_selectors_v1 --plan REGISTRATION',
        'REGISTRATION = '+str(path), '']
    report=root/'analysis/FINDINGS_KO_v1.txt'
    report.write_text('\n'.join(lines))
    code_files=[p for p in Path(__file__).parent.glob('*critic_stabilization*.py')]
    code_files += [Path(__file__).with_name('evaluate_critic_temporal_average_v1.py'),
                   Path(__file__).with_name('validate_temporal_average_policy_v1.py')]
    code_hashes={str(p.resolve()):sha(p) for p in code_files}
    for p in code_files:(root/'source'/p.name).write_bytes(p.read_bytes())
    files=[report,root/'analysis/results_v2.json',root/'analysis/per_seed_results_v2.csv',
        root/'analysis/task_summary_v2.csv',root/'analysis/ranking_stability_v2.csv',
        root/'analysis/continuation_audit_v2.csv',root/'analysis/selector_comparison_v1.json',
        root/'analysis/early_curve_summary_v1.json',root/'analysis/implementation_validation_v1.json',
        root/'analysis/temporal_average_validation_summary_v1.json']+list((root/'figures').glob('*'))
    atomic(root/'analysis/DELIVERY_MANIFEST_v1.json',dict(status='complete',timestamp_unix=time.time(),
        new_training_branches=12,additional_Q_updates=2400000,additional_actor_updates=2400000,
        main_evaluation_cells=48,temporal_average_cells=12,reproduction_control_cells=6,
        early_curve_cells=24,exploratory_mean_head_cells=6,new_CPU_rollout_episodes=4800,
        reused_primary_baseline_cells=36,reused_primary_baseline_episodes=1800,
        training_seeds=[0,1,2],all_12_Flow_and_RNG_audits_exact=True,all_480_probes_finite=True,
        original_parent_models_unchanged=True,core_source_pins_verified=True,
        source_sha256=code_hashes,artifact_sha256={str(p):sha(p) for p in files},
        limits=plan['limits']))
    print(json.dumps(dict(status='complete',report=str(report),new_rollout_episodes=4800)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();run(args.plan.resolve())
