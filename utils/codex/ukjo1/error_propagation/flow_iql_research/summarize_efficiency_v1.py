"""Wait for immutable efficiency audit and write a conservative Korean report."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(root,status,output,terminal_reason):
    assert not output.exists()
    output.mkdir(parents=True)
    plan=json.loads((root/'registration_v1.json').read_text())
    rules=plan['promotion_rule']
    timings={(r['variant'],r['environment']):r for r in status.get('isolated_timing_rows',[])}
    lines=['FINITE-MEAN 효율 개선 실험 — 최종 감사 기반 보고서',
        f'생성 시각 Unix: {time.time()}',f'상태: {terminal_reason}',
        '','판정 기준',
        '주지표는 CPU에서 독립 seed 42..91의 50개 에피소드로 평가한 800K, 900K, 1M 성공률의 평균이다.',
        '환경과 학습 시드에도 같은 가중치를 준다. 누락값을 0으로 채우거나 최고 checkpoint를 선택하지 않는다.',
        'teacher 재평가는 같은 critic에서 원래 10-step BC flow best9를 실행한 보조 대조이며 주지표와 섞지 않는다.',
        '파일럿은 Ant-large, Cube-double, Humanoid-medium의 seed0 실험이다. 3개 환경 평균을 9개 환경 평균으로 부르지 않는다.',
        '','검증 규모',f'계획 GPU 학습 수: {status.get("planned_GPU_jobs")}',
        f'주평가까지 완료된 학습 수: {status.get("completed_primary_runs")}',
        f'감사된 실제 저장 모델 수: {status.get("actual_finite_models")}',
        f'감사된 실제 CPU50 평가 수: {status.get("valid_CPU_evaluations")}',
        f'오류: {json.dumps(status.get("failures",{}),ensure_ascii=False)}',
        '','방법의 논리',
        'reference: 기존 finite-mean. BC flow 행동9개와 twin-Q 평균으로 finite expectile을 계산하고 알려진 반환 경계로 target을 제한한다.',
        'lean: bootstrap/배포에서 사용하지 않는 learned-V fitting을 제거한다. finite expectile 계산과 BC flow 경로를 유지한다.',
        'euler5/euler2: flow 적분 횟수를 줄인다. target 분포 자체의 변화가 있으므로 속도만으로 성능 유지를 주장할 수 없다.',
        'bc_student: BC flow의 noise-to-action 대응을 Q 손실 없이 one-step 모델에 증류한다. 첫50K는 teacher backup, 이후 student 행동9개로 finite expectile을 계산한다.',
        'anchor_student: student 행동9개 중 한 개를 정확한 teacher 행동으로 유지한다. 배포는 student best9이다.',
        'guided_actor: 원래 finite-mean backup을 유지하고 배포에만 FQL 형태의 Q-guided one-step actor를 사용한다.',
        'student_guided: BC student로 backup 계산을 줄이고 별도의 Q-guided actor로 배포 계산을 줄인다.',
        'Q-guided actor를 BC expectile의 행동 분포로 바로 대체하지 않는다. 두 증류 모델의 역할을 구분한다.',
        '','초기 격리 속도 비교 — 짧게 warmup한 모델, 학습 성능 결과 아님']
    initial=json.loads((root/'analysis/initial_isolated_speed_v1.json').read_text())
    lines.append('task / method / width / 학습 ms/update / CPU 행동 ms / GPU 행동 ms')
    for row in initial['rows']:
        lines.append(f'{row["task"]} / {row["method"]} / {row["width"]} / {row["training_ms_per_update_median"]:.4f} / {row["cpu_action_ms_median"]:.4f} / {row["gpu_action_ms_median"]:.4f}')
    lines+=['Ant는 두 방법 모두512너비이다. Cube-double은 원래 finite-mean256, 공개 FQL512이며 별도의 FQL256 속도 대조도 제시했다.',
        '논문 A5000 수치를 이 장비의 실측값으로 사용하지 않았다.',
        '','실제1M 저장 모델에서 측정한 비용과 파일럿 성능',
        '학습 비용은 resident batch256, scan64, warm64 후3회 반복 중앙값이다. 배포는 단일 관측100회 동기 호출이다.',
        '컴파일, dataset IO, 진단, 저장, CPU50 환경 평가를 제외한다. 각 run/runtime.json에 학습 중 해당 비용을 별도 기록한다.',
        '동일 환경의 모든 방법과 reference는 같은 GPU/CPU affinity에서 측정한다. 16개 동시 학습의 공유 GPU ms를 격리 latency로 해석하지 않는다.',
        'variant / 파일럿 성공률 / 학습 reference 비율 / CPU 배포 reference 비율 / GPU 배포 reference 비율']
    cost_rows=[]
    for group in status.get('pilot_groups',[]):
        variant=group['variant'];ratios={}
        for metric in ('training_ms_per_update_median','cpu_action_ms_median','gpu_action_ms_median'):
            values=[]
            for env in plan['pilot_environments']:
                current=timings.get((variant,env));base=timings.get(('reference',env))
                if current is not None and base is not None:values.append(current[metric]/base[metric])
            ratios[metric]=math.exp(statistics.mean(math.log(v) for v in values)) if len(values)==3 else None
        def display(value):return '미완료' if value is None else f'{value:.6f}'
        lines.append(' / '.join([variant,display(group['pilot_three_task_mean']),*[display(v) for v in ratios.values()]]))
        cost_rows.append(dict(variant=variant,pilot_mean=group['pilot_three_task_mean'],**ratios))
    selected=status.get('selected_candidate')
    lines+=['','선정과 9개 환경 확인',f'선정 후보: {json.dumps(selected,ensure_ascii=False)}',
        f'선정 규칙: {json.dumps(rules,ensure_ascii=False)}',
        '학습/CPU배포 비율은 각각 최대0.85/0.5, 파일럿 평균 성능 하락은 최대0.03, 개별 환경 하락은 최대0.10이다.',
        '이 조건 안에서 두 속도 향상 배수의 곱이 큰 단일 공통 설정을 선정한다. 환경별 최우수 기법을 조합하지 않는다.']
    for row in status.get('full9',[]):lines.append(json.dumps(row,ensure_ascii=False))
    for key in ('discovery_practical_performance_threshold','discovery_performance_preserved_descriptively',
                'fresh_three_seed_nine_task_mean','fresh_seed_sample_SD','fresh_practical_performance_threshold',
                'fresh_performance_preserved_descriptively'):
        if key in status:lines.append(f'{key}: {status[key]}')
    if terminal_reason=='complete':
        if selected is None:lines.append('등록한 후보 중 파일럿 성능과 두 가지 비용 개선 기준을 동시에 충족한 방법이 없었다. 성공한 효율 개선으로 주장하지 않는다.')
        elif status.get('fresh_performance_preserved_descriptively') is True and status.get('discovery_performance_preserved_descriptively') is True:
            lines.append('공통 후보가 등록한 파일럿 비용 기준과 9개 환경 discovery/새3시드의 실용적 성능 허용 범위를 통과했다.')
        else:lines.append('속도 기준을 통과한 후보도 9개 환경 성능 유지까지 확립한 것으로 해석하지 않는다. 위의 실제 full9/새시드 수치와 판정을 확인한다.')
    else:lines.append('실험이 전체 완료되지 않았다. 아래 완료된 부분 결과만 설명하며 미완료를 성능 실패나 성공으로 바꾸어 해석하지 않는다.')
    lines+=['','진단과 인과 해석의 범위',
        '각 run의 paired_action_MSE, paired_root_error_RMS는 같은 critic과 같은 noise에서 현재 제안 분포와 정확한10-step teacher의 차이를 측정한다.',
        'finite_proposal_score_abs_mean은 현재 제안 행동9개의 finite expectile 방정식 오차이다. true behavior expectile 또는 전체 propagation error의 oracle 진단이 아니다.',
        '작은 증류 MSE만으로 bootstrap 오차가 작다고 결론 내리지 않는다. root 오차, critic 크기/TD/clip, 같은-Q teacher 배포와 주정책 성공률을 함께 확인한다.',
        'teacher 배포가 주정책보다 낫다면 배포 근사 문제를 시사한다. 두 배포가 모두 악화되면 critic/backup 변화도 함께 의심해야 한다.',
        '이 비교는 완전한 인과 증명이 아니며 환경/학습 시드 변동을 포함한다.',
        '새 확인 시드는12/13/14, 과거 reference3시드는9/10/11이다. 독립 집단의 기술적 비교이며 동일 시드 paired 통계 검정이나 정식 비열등성 증명이 아니다.',
        '공개 FQL의 너비/설정은 환경에 따라 제안 방법과 다르다. 실제 설정과 비용을 함께 제시하며 학습 속도만으로 더 좋은 전체 알고리즘이라고 주장하지 않는다.',
        '','학습별 주평가와 같은-Q teacher 대조','run / environment / seed / 주지표 last3 / teacher last3']
    diagnostics=[]
    keys=('q_rms','q_rms_over_return_bound','q_disagreement_rms','td_rms','q_target_lag_rms',
          'paired_action_MSE','paired_root_error_RMS','paired_proposal_Q_error_RMS',
          'finite_proposal_score_abs_mean','backup/clip_fraction','student/BC_distill_MSE','policy/distill_MSE')
    for row in status.get('rows',[]):
        lines.append(f'{row["run"]} / {row["environment"]} / {row["seed"]} / {row.get("last3_mean")} / {row.get("auxiliary_teacher_last3")}')
        path=root/'runs'/row['run']/'diagnostics.jsonl'
        probes=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        late=[p for p in probes if 800000<=p['step']<=1000000]
        item=dict(run=row['run'],variant=row['variant'],environment=row['environment'],seed=row['seed'],late_probe_count=len(late))
        item.update({key:statistics.mean(float(p[key]) for p in late) if late and all(key in p for p in late) else None for key in keys})
        diagnostics.append(item)
    lines+=['','재현 및 실제 소스 경로',
        'utils/flow_iql_research/efficient_mean_core_v1.py',
        'utils/flow_iql_research/train_efficient_v1.py',
        'utils/flow_iql_research/evaluate_efficient_v1.py',
        'utils/flow_iql_research/benchmark_efficient_models_v1.py',
        '원본 성공 알고리즘: utils/flow_iql_research/mean_empirical_backup_core_v1.py',
        '원본 학습 runner: utils/flow_iql_research/train_mean_empirical_backup_v1.py',
        '등록/실행 계약: registration_v1.json, analysis/execution_contract_v1.json',
        '최종 실제 모델/평가/설정 SHA: analysis/efficiency_pipeline_v1/final_audit.json',
        '세부 비용 및 진단: speed_performance.csv, late_training_diagnostics.csv',
        'Giant 실험은 사용자의 지시대로 보류했다. 기존 성공 결과와 등록 소스는 변경하지 않았다.']
    (output/'RESEARCH_REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
    for name,rows in [('speed_performance.csv',cost_rows),('late_training_diagnostics.csv',diagnostics)]:
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader();writer.writerows(rows);(output/name).write_text(buffer.getvalue())
    (output/'provenance.json').write_text(json.dumps(dict(time=time.time(),terminal_reason=terminal_reason,
        source_sha256=sha(__file__),registration_sha256=sha(root/'registration_v1.json'),
        status_sha256=sha(root/'analysis/efficiency_pipeline_v1/status.json'),
        no_training_or_policy_updates=True),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();pipeline=root/'analysis/efficiency_pipeline_v1'
    receipt=json.loads((root/'analysis/controller_launch_v1.json').read_text())
    contract=json.loads((root/'analysis/report_execution_v1.json').read_text())
    while True:
        assert sha(__file__)==contract['source_sha256']
        assert sha(root/'analysis/execution_contract_v1.json')==contract['analysis_execution_contract_sha256']
        path=pipeline/'status.json'
        if path.exists():
            status=json.loads(path.read_text())
            if status['status'] in ('complete','failure_requires_inspection'):
                write_report(root,status,root/'analysis/efficiency_final_review_v1',status['status']);return
            if (pipeline/'controller_failure.json').exists():
                write_report(root,status,root/'analysis/efficiency_final_review_v1','controller_failure');return
        process=Path(f'/proc/{receipt["pid"]}/stat')
        try:fields=process.read_text().split(') ',1)[1].split()
        except FileNotFoundError:fields=None
        identity=fields[19] if fields else None
        if (identity!=receipt['process_identity'] or fields[0]=='Z') and path.exists():
            write_report(root,status,root/'analysis/efficiency_final_review_v1','controller_exited_before_completion');return
        time.sleep(20)


if __name__=='__main__':main()
