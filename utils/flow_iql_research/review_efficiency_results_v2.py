"""Independent read-only audit and explanatory review of completed speed pilots."""
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import statistics
import time
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def leaves(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):yield from leaves(tree[key],prefix+'/'+str(key))
    elif isinstance(tree,(tuple,list)):
        for key,item in enumerate(tree):yield from leaves(item,prefix+'/'+str(key))
    elif tree is not None:yield prefix,np.asarray(tree)


def compare(left,right):
    a=dict(leaves(left));b=dict(leaves(right));assert a.keys()==b.keys()
    return dict(bitwise_equal=all(np.array_equal(a[k],b[k]) for k in a),
        max_absolute_difference=max(float(np.max(np.abs(a[k].astype(np.float64)-b[k].astype(np.float64)))) for k in a))


def save_csv(path,rows):
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.write_text(buffer.getvalue())


def main():
    root=Path('logs/diagnosis/finite_mean_efficiency_20261030').resolve()
    output=root/'analysis/efficiency_results_review_v2';output.mkdir(exist_ok=False)
    audit_path=root/'analysis/efficiency_pipeline_v1/final_audit.json'
    audit=json.loads(audit_path.read_text());assert audit['status']=='complete' and not audit['failures']
    plan=json.loads((root/'registration_v1.json').read_text())
    contract=json.loads((root/'analysis/execution_contract_v1.json').read_text())
    assert sha(root/'registration_v1.json')==audit['registration_sha256']
    assert sha(root/'analysis/execution_contract_v1.json')==audit['execution_contract_sha256']
    for path,digest in {**plan['all_source_sha256'],**contract['source_sha256']}.items():assert sha(path)==digest,path
    models={};model_receipts=[]
    for key,pin in audit['actual_model_pins'].items():
        assert sha(pin['path'])==pin['sha256'],key
        with Path(pin['path']).open('rb') as stream:saved=pickle.load(stream)
        state=saved['state'];step=int(key.rsplit(':',1)[1]);variant=saved['identity']['variant']
        assert int(state['q_updates'])==saved['step']==step
        assert int(state['agent']['network']['opt_state']['0']['count'])==step
        assert int(state['v_updates'])==int(state['v_opt']['0']['count'])==(2*step if variant=='reference' else 0)
        assert all(np.isfinite(array).all() for _,array in leaves(state))
        model_receipts.append(dict(key=key,sha256=pin['sha256'],actual_Q_Adam=step,finite_full_state=True))
        if step==1000000 and variant in ('reference','lean','guided_actor') and saved['identity']['environment'] in plan['pilot_environments']:
            models[(saved['identity']['environment'],variant)]=state['agent']['network']['params']
    verified={}
    for key,receipt in audit['evaluation_receipts'].items():
        rid,step,mode=key.split(':');path=root/'analysis/evaluations_v1'/f'{rid}_{step}_{mode}.json'
        assert sha(path)==receipt['sha256']
        row=json.loads(path.read_text());assert row['step']==int(step) and row['policy_mode']==mode
        assert row['first_episode_seed']==42 and row['n_episodes']==50 and row['eval_device']=='cpu'
        assert row['checkpoint_sha256']==audit['actual_model_pins'][f'{rid}:{step}']['sha256']
        assert sha(root/'runs'/rid/'config.json')==row['config_sha256']
        assert row['source_sha256']==sha(Path('utils/flow_iql_research/evaluate_efficient_v1.py'))
        assert row['core_source_sha256']==sha(Path('utils/flow_iql_research/efficient_mean_core_v1.py'))
        assert all(len(row[k])==50 for k in ('episode_successes','episode_rewards','episode_lengths'))
        assert all(x in (0,1) for x in row['episode_successes'])
        score=Fraction(sum(int(x)for x in row['episode_successes']),50)
        assert abs(float(score)-row['success_rate'])<1e-12 and abs(float(score)-receipt['success_rate'])<1e-12
        verified[key]=score
    by_run=[]
    for row in audit['rows']:
        primary=sum((verified[f'{row["run"]}:{s}:primary'] for s in (800000,900000,1000000)),Fraction(0))/3
        assert abs(float(primary)-row['last3_mean'])<1e-12
        teacher=[verified[f'{row["run"]}:{s}:teacher'] for s in (800000,900000,1000000) if f'{row["run"]}:{s}:teacher' in verified]
        by_run.append(dict(run=row['run'],environment=row['environment'],variant=row['variant'],seed=row['seed'],
            primary_last3=float(primary),same_checkpoint_teacher_last3=float(sum(teacher,Fraction(0))/3) if len(teacher)==3 else None))
    assert len(by_run)==33 and len(verified)==153 and len(model_receipts)==99
    timings={(r['variant'],r['environment']):r for r in audit['isolated_timing_rows']}
    assert len(timings)==27
    summaries=[]
    for group in audit['pilot_groups']:
        variant=group['variant'];runs=[r for r in by_run if r['variant']==variant and r['environment'] in plan['pilot_environments']]
        assert len(runs)==3
        score=statistics.mean(r['primary_last3'] for r in runs);assert abs(score-group['pilot_three_task_mean'])<1e-12
        teacher=statistics.mean(r['same_checkpoint_teacher_last3'] for r in runs) if all(r['same_checkpoint_teacher_last3'] is not None for r in runs) else None
        row=dict(variant=variant,pilot_success=score,same_checkpoint_teacher_success=teacher)
        for label,metric in [('training','training_ms_per_update_median'),('CPU_action','cpu_action_ms_median'),('GPU_action','gpu_action_ms_median')]:
            ratios=[]
            for env in plan['pilot_environments']:
                a=timings[(variant,env)];b=timings[('reference',env)]
                assert a['measurement_GPU']==b['measurement_GPU'] and a['measurement_CPUs']==b['measurement_CPUs']
                ratios.append(a[metric]/b[metric])
            row[label+'_latency_ratio']=math.exp(statistics.mean(math.log(x)for x in ratios))
        summaries.append(row)
    comparisons=[]
    for env in plan['pilot_environments']:
        for variant in ('lean','guided_actor'):
            left=models[(env,'reference')];right=models[(env,variant)]
            comparisons.append(dict(environment=env,variant=variant,
                Q=compare(left['modules_critic'],right['modules_critic']),
                target_Q=compare(left['modules_target_critic'],right['modules_target_critic']),
                BC=compare(left['modules_actor_flow'],right['modules_actor_flow'])))
    old=Path('logs/diagnosis/flow_iql_research_20261026/analysis/mean_backup_final_review_v1/task_results.csv')
    baseline={r['environment']:float(r['discovery_s0'])for r in csv.DictReader(old.open())}
    lean=[r for r in by_run if r['variant']=='lean'];assert len(lean)==9
    nine_comparison=[dict(environment=r['environment'],old_reference_seed0=baseline[r['environment']],
        lean_seed0=r['primary_last3'],difference=r['primary_last3']-baseline[r['environment']])for r in lean]
    result=dict(time=time.time(),status='independently_verified',source_sha256=sha(__file__),
        original_audit_sha256=sha(audit_path),verified_full_models=model_receipts,verified_CPU50_evaluations=len(verified),
        total_CPU_episodes=50*len(verified),model_comparisons_1M=comparisons,
        pilot_summaries=summaries,lean_nine_task_mean=statistics.mean(r['primary_last3']for r in lean),
        old_reference_nine_task_seed0_mean=statistics.mean(baseline.values()),
        conditional_confirmation_launched=False,selected_candidate=audit['selected_candidate'],
        no_model_policy_or_training_source_mutations=True,
        limits=['Pilot=3tasks/seed0. Onlylean has9tasks/seed0. No new3seedconfirmation.',
            'Microbenchmark latency differs from end-to-end16learner walltime.',
            'Same-checkpointteacher evaluation isolates deployment effects; teacher-vs-reference compares different learned critics.',
            'Small differences between mathematically equivalent models can reflect floating-point compilation and long-training sensitivity.'])
    (output/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    save_csv(output/'pilot_speed_performance.csv',summaries)
    save_csv(output/'per_environment_results.csv',by_run)
    save_csv(output/'lean_nine_task_comparison.csv',nine_comparison)
    lines=['효율 개선 실험 재검토 — 2026-10-31',
        '','결론',
        '33개1M학습/99개 late checkpoint/153개 CPU50평가(7,650에피소드)가 모두 완료됐다. 원시 성공 배열과 모델 SHA/실제 Adam 횟수를 독립 재검증했다.',
        '미사용 learned-V fitting 제거(lean)는 파일럿 성공률을 유지하면서 격리 학습 업데이트 시간을15.46% 줄였다. CPU/GPU 배포 시간은 거의 그대로다.',
        '학습과 배포를 모두 가속한 후보는 이번 파일럿에서 성능 유지 조건까지 통과하지 못했다. 성공한 one-step 대체 알고리즘을 찾았다고 주장할 수 없다.',
        '','평가 프로토콜',
        'tau0.9/gamma0.999. 각800K/900K/1M에서 CPU50독립reset seed42..91 성공률을 동일 가중 평균했다.',
        '파일럿=Ant-large,Cube-double,Humanoid-medium,학습seed0. 아래3환경 평균을9환경 평균으로 해석하면 안 된다.',
        '격리 시간은 실제1M모델, 같은 환경별 같은GPU/CPU affinity, residentbatch256/scan64/warm64/3회64update, 배포100동기호출 중앙값이다.',
        '환경별 시간 비율의 기하평균이며 전체1M학습/환경평가 wall-clock이 아니다. 실제 run의 전체 학습 프로세스 시간은 별도 completed.json에 있다.',
        '','파일럿 결과 — 시간 비율1은 원본과 동일, 작을수록 빠름',
        'variant / 평균 성공률 / 학습 시간 비율 / CPU배포 시간 비율 / GPU배포 시간 비율 / 같은checkpoint teacher성공률']
    for r in summaries:
        teacher='없음' if r['same_checkpoint_teacher_success'] is None else f'{r["same_checkpoint_teacher_success"]:.6f}'
        lines.append(f'{r["variant"]} / {r["pilot_success"]:.6f} / {r["training_latency_ratio"]:.6f} / {r["CPU_action_latency_ratio"]:.6f} / {r["GPU_action_latency_ratio"]:.6f} / {teacher}')
    lines+=['','학습된 원본과 FQL의 비용 비교',
        '3환경에서 원본/FQL 격리 학습 latency는 기하평균 약1.81배, CPU배포 약5.16배, GPU배포 약2.31배였다.',
        'Ant와Humanoid의 너비는 둘 다512이다. Cube-double은 원본256/FQL512이므로 공개설정 비용 비교와 동일구조 비교를 구분해야 한다.',
        '','9환경 lean 결과',
        f'lean seed0 평균={result["lean_nine_task_mean"]:.9f}; 과거 원본 seed0={result["old_reference_nine_task_seed0_mean"]:.9f}.',
        '새로운 성능 향상이 입증됐다는 의미는 아니다. 파일럿3환경에서는800K/900K/1M 점수가 각각 원본과 동일했다.',
        '나머지6환경은 과거 원본 결과와 비교했다. 새3시드 검증은 수행하지 않았다. 수학적으로 미사용 계산을 제거한 최적화로 해석한다.',
        '','실패 원인에 대한 근거',
        '1) guided_actor의 같은checkpoint teacher배포 평균은0.524444, one-step배포는0.171111이다.',
        '   Ant의 같은checkpoint에서 원래 best9 정책은0.46, guided정책은0.00이었다. 이 checkpoint의 Q/targetQ/BC는 원본과 정확히 같은지 independent_audit.json에 기록했다.',
        '   critic이 발산해서 실패했다고만 설명할 수 없다. 빠른 배포 정책이 원래의 best-of-9 선택을 보존하지 못한 문제가 강하게 나타났다.',
        '   guided_actor는 BC flow단일행동 모방+Q최대화를 학습한다. 원래 정책인9개BC후보의critic최솟값argmax를 직접 모방하지 않는다.',
        '   Ant late policy/distill_MSE는약0.599이며 Q손실과 모방손실의 균형이 충분하지 않은 정황이다. 정확한 OOD Q오차 oracle을 측정한 것은 아니다.',
        '2) BC-only student도 행동MSE가 작다고 backup을 보존하지는 못했다.',
        '   Cube-double late 행동MSE~0.00137, teacher대비root오차RMS~1.01; 주정책0.16/같은Qteacher0.3933/원본0.32.',
        '   Ant에서는 student주정책0.18/같은Qteacher0.2333/원본0.46으로, 배포를 teacher로 바꿔도 원본까지 회복하지 못했다. backup/critic 변화도 관여했을 가능성이 있다.',
        '   현재 student제안에 대한 finite-expectile 방정식 오차는약1e-6이다. root솔버가 정확해도 제안행동분포의 근사 오차가 추가될 수 있다.',
        '3) Euler2의 Cube-double은0.00, 같은Qteacher도0.02였다. late Q_RMS는원본315대비703, TD_RMS는3.96대비14.02였다.',
        '   이 경우에는 배포 적분만 바꾸면 되는 문제가 아니며 학습 target/critic까지 악화된 정황이 강하다.',
        '4) 1개teacher anchor는 BC-only student보다평균0.384444→0.477778로 개선했지만 원본0.533333까지 유지하지 못했다.',
        '수치 NaN 없이 끝나는 안정성과 정책 성공률 유지는 별개다. 모든 설명은 이번3환경/1학습시드 실험의 범위에 한정한다.',
        '','왜 자동으로 다음27개 새시드를 실행하지 않았는가?',
        '결과 전에 등록한 성능하락<=0.03/개별<=0.10, 학습latency<=0.85, CPU배포latency<=0.50을 모두 충족한 후보가 없었다.',
        'lean은 배포조건을 통과하지 못했고, 빠른 student는 성능조건을 통과하지 못했다. 확장 조건에 따라 후속 학습은 시작되지 않았다.',
        '현재 두GPU가 쉬는 것은33개 학습/평가/격리측정이 모두 종료되고 해당 대기열을 닫았기 때문이다. 현재 실행 중이라고 보고하지 않는다.',
        '','다음으로 설득력 있는 실험 방향 — 아직 실행/성공을 주장하지 않음',
        '학습에는 lean을 기본으로 삼고, 우선 critic/backup을 고정한 배포 정책 증류부터 분리 검증하는 것이 타당하다.',
        'BC단일행동+Q최대화 대신 실제 teacher의 best-of-9 선택 행동을 직접 증류하면 이번에 드러난 목표 불일치를 줄일 수 있다.',
        'teacher9개noise전체를 조건으로 쓰거나 선택 분포를 보존하도록 설계해야 한다. 단순한 stochastic label MSE는 다시 평균화 문제를 만들 수 있다.',
        '그 단계가 성공한 뒤 Q/finite-root 오차도 제약하는 BC student를 검토한다. 이는 제안일 뿐 아직 검증된 결과가 아니다.',
        '','현재 권장 구현',
        'utils/flow_iql_research/efficient_mean_core_v1.py, variant=lean',
        'utils/flow_iql_research/train_efficient_v1.py --variant lean',
        '성능이 검증된 원본은 mean_empirical_backup_core_v1.py와 train_mean_empirical_backup_v1.py에 보존돼 있다.',
        '원시 결과: ../efficiency_pipeline_v1/results.csv 및 final_audit.json',
        '독립 재검증/비교 표: independent_audit.json, pilot_speed_performance.csv, per_environment_results.csv, lean_nine_task_comparison.csv']
    (output/'RESEARCH_REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
    colors={'reference':'#111827','lean':'#2563eb','fql':'#0f766e'}
    for ax,key,label,limit in [(axes[0],'training_latency_ratio','Training latency / reference',.85),(axes[1],'CPU_action_latency_ratio','CPU action latency / reference',.5)]:
        ax.axhline(.5333333333333333,color='#64748b',linestyle='--',linewidth=1)
        ax.axhline(.5033333333333333,color='#64748b',linestyle=':',linewidth=1)
        ax.axvline(limit,color='#94a3b8',linestyle=':',linewidth=1)
        for r in summaries:
            variant=r['variant'];x=r[key];y=r['pilot_success'];ax.scatter(x,y,s=65,color=colors.get(variant,'#c2410c'),zorder=3)
            offset=(5,6)
            if variant=='lean':offset=(-28,-17)
            if variant=='reference':offset=(-45,8)
            if variant=='euler2':offset=(-35,-16)
            if variant=='bc_student':offset=(5,-15)
            ax.annotate(variant,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
        ax.set(xlim=(0,1.12),ylim=(0,.62),xlabel=label);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean success rate (three pilot tasks)')
    fig.suptitle('Speedups mostly traded away performance; lean preserved returns',fontsize=13)
    fig.text(.5,.015,'Training seed 0; CPU50 at 800K / 900K / 1M. Lower latency and higher success are preferred. No multi-seed claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(output/'speed_performance.png',dpi=180);fig.savefig(output/'speed_performance.pdf');plt.close(fig)
    assert sha(audit_path)==result['original_audit_sha256']
    print(json.dumps({k:result[k] for k in ('status','verified_CPU50_evaluations','total_CPU_episodes','lean_nine_task_mean','old_reference_nine_task_seed0_mean')},indent=2))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
