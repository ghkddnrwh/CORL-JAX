"""Compare measured gamma=.999/.99 curves, keeping reward protocols explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .matched_error_v11 import save_csv, save_json
from .mdp import evaluate_policy
from .gamma_penalty_v14 import environment
from .penalty_sweep_v12 import MODES
from .plot_penalty_sweep_v12 import ALGORITHMS, PENALTY_LABELS, METRICS, mean_sd, xaxis, save_figure
from .plot_refined_penalty_v13 import render, summarize, crossing_brackets

PROTOCOLS = {
    'original': ('.999: original rewards', '#0072B2', '-'),
    'fixed_reward': ('.99: SAME rewards', '#D55E00', '--'),
    'matched_gap': ('.99: RECALIBRATED safe reward', '#009E73', '-.'),
}


def select(rows, algorithm, mode):
    return sorted((r for r in rows if r['algorithm'] == algorithm and r['penalty_mode'] == mode),
                  key=lambda r: r['strength'])


def draw(ax, rows, metric, scale, protocol):
    label, color, style = PROTOCOLS[protocol]
    x = [r['strength'] for r in rows]
    y = np.array([r[metric+'_mean'] for r in rows])*scale
    sd = np.array([r[metric+'_sd'] for r in rows])*scale
    line, = ax.plot(x, y, color=color, linestyle=style, marker='o', markersize=2,
                    linewidth=1.6, label=label)
    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.10)
    return line


def comparisons(summaries, figures):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for metric, ylabel, scale in METRICS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True)
        handles = []
        for i, algorithm in enumerate(ALGORITHMS):
            for j, mode in enumerate(('q', 'v', 'qv')):
                ax = axes[i, j]
                if mode not in MODES[algorithm]:
                    ax.set_axis_off()
                    ax.text(.5, .5, 'FEQ has no V regression', ha='center', transform=ax.transAxes)
                    continue
                for protocol, rows in summaries.items():
                    line = draw(ax, select(rows, algorithm, mode), metric, scale, protocol)
                    if i == j == 0:
                        handles.append(line)
                ax.set_title(f'{ALGORITHMS[algorithm]}: {PENALTY_LABELS[mode]}')
                xaxis(ax)
                if i == 0 or (i == 1 and j == 0):
                    ax.set_xlabel('')
                if i == 1 and j > 0:
                    ax.tick_params(axis='x', labelbottom=True)
                ax.set_ylabel(ylabel)
                if metric == 'success':
                    ax.set_ylim(-3, 103)
                elif metric == 'q_rmse':
                    ax.set_ylim(bottom=-.003)
        fig.suptitle(f'Discount and penalty: {ylabel}', fontsize=15, y=.995)
        fig.text(.5, .955, '16 paired seeds; 2M updates; last 10%; +/- 1 seed SD; actual trained points', ha='center')
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .055, 1, .935))
        save_figure(fig, figures, 'gamma_comparison_'+metric)

    # Compact overview; the complete seven-condition comparison is saved above.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True)
    handles = []
    for i, (algorithm, mode) in enumerate((('iql', 'qv'), ('iql9', 'qv'), ('feq9', 'q'))):
        for protocol, rows in summaries.items():
            line = draw(axes[i], select(rows, algorithm, mode), 'success', 100., protocol)
            if i == 0:
                handles.append(line)
        axes[i].set_title(f'{ALGORITHMS[algorithm]}: {PENALTY_LABELS[mode]}')
        axes[i].set_ylim(-3, 103)
        xaxis(axes[i])
    axes[0].set_ylabel('Goal-reaching probability (%)')
    fig.suptitle('Lower discount changes both error propagation and the preferred policy', fontsize=14, y=.99)
    fig.text(.5, .91, '16 paired seeds; 2M updates; last 10% average; bands: +/- 1 seed SD', ha='center')
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .87))
    save_figure(fig, figures, 'gamma_comparison_compact')


def baseline_diagnostics(records):
    rows = []
    for protocol in ('fixed_reward', 'matched_gap'):
        for algorithm in ALGORITHMS:
            samples = []
            for obj in records[protocol]:
                c = obj['config']
                if c['algorithm'] != algorithm or c['strength'] != 0:
                    continue
                m = environment(protocol)
                with np.load(Path(obj['artifact'])/'data.npz') as h:
                    m.behavior = h['beta']
                    reference_return, reference_success = evaluate_policy(m, h['q_truth'])
                    samples.append(dict(clean_population_success=reference_success,
                        clean_population_return=reference_return,
                        population_root_gap=float(h['q_truth'][0, 0]-h['q_truth'][0, 1]),
                        exact_root_gap=float(h['q_fixed'][0, 0]-h['q_fixed'][0, 1]),
                        exact_success=obj['metrics']['exact_success'],
                        learned_success=obj['metrics']['success']))
            assert len(samples) == 16
            row = dict(protocol=protocol, algorithm=algorithm)
            for key in samples[0]:
                row[key+'_mean'], row[key+'_sd'] = mean_sd([s[key] for s in samples])
            rows.append(row)
    return rows


def remaining_intervals(summaries):
    result = []
    for protocol in ('fixed_reward', 'matched_gap'):
        factor = .1 if protocol == 'fixed_reward' else 1.
        floors = dict(q=.000025*factor, v=.0000125*factor, qv=.00000625*factor)
        for algorithm in ALGORITHMS:
            for mode in MODES[algorithm]:
                curve = select(summaries[protocol], algorithm, mode)
                for a, b in zip(curve, curve[1:]):
                    change = abs(a['success_mean']-b['success_mean'])
                    if change > .08:
                        result.append(dict(protocol=protocol, algorithm=algorithm, penalty_mode=mode,
                            lower=a['strength'], upper=b['strength'], change_pp=change*100,
                            reason='minimum spacing reached' if b['strength']-a['strength'] < 2*floors[mode]-1e-13
                                   else 'six refinement rounds reached'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((args.root/'artifact_validation.json').read_text())['status'] == 'passed'
    assert json.loads((args.reference/'artifact_validation.json').read_text())['status'] == 'passed'
    out = args.root/'analysis'
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    records = {'original': json.loads((args.reference/'artifact_index.json').read_text())}
    summaries = {'original': summarize(records['original'])}
    for protocol in ('fixed_reward', 'matched_gap'):
        records[protocol] = json.loads((args.root/protocol/'artifact_index.json').read_text())
        summaries[protocol], _ = render(records[protocol], out/protocol, gamma=.99,
            description='Fixed original rewards' if protocol == 'fixed_reward' else 'Recalibrated safe reward (matched relative gap)',
            prefix='gamma099_'+protocol)
    comparisons(summaries, figures)
    # Restrict combined CSV to shared metric fields (v12 stores additional diagnostics).
    joint = []
    for protocol, rows in summaries.items():
        for r in rows:
            joint.append(dict(protocol=protocol, gamma=.999 if protocol == 'original' else .99,
                **{key: r[key] for key in ('algorithm', 'penalty_mode', 'strength', 'n', 'seeds')},
                **{key+'_'+stat: r[key+'_'+stat] for key in ('success', 'return', 'q_rmse', 'exact_success', 'exact_q_rmse')
                   for stat in ('mean', 'sd')}))
    save_csv(out/'comparison_summary.csv', joint)
    crossings = [dict(protocol=protocol, **r) for protocol, rows in summaries.items() for r in crossing_brackets(rows)]
    save_csv(out/'comparison_crossing_brackets.csv', crossings)
    baselines = baseline_diagnostics(records)
    save_csv(out/'zero_penalty_diagnostics.csv', baselines)
    remaining = remaining_intervals(summaries)
    save_json(out/'remaining_sharp_intervals.json', remaining)
    selected = [r for r in joint if r['strength'] in (0., .0001, .001, .003, .01)]
    save_csv(out/'selected_strengths.csv', selected)
    audit = json.loads((args.root/'artifact_validation.json').read_text())
    lines = [
        'Gamma=.99 penalty sweep and comparison to gamma=.999', '',
        f'추가 학습 {audit["runs"]}개 완료. 9상태, 4행동, tau=.9, 완전 Q/V table.',
        '모든 조건은 같은 16시드(2600~2615), 각 2M 업데이트, 마지막 10% 평균.',
        '표준편차는 seed별 시간평균 사이의 sample SD이며 평가 episode SD가 아니다.',
        'IQL / IQL-9 각각 Q-only, V-only, Q+V. FEQ-N9은 Q-only.',
        'IQL의 V와 Q에 같은 형태의 출력 ridge penalty .5*lambda*x^2를 사용.',
        'FEQ는 V 회귀가 없으며 next-state 행동 9개의 sample expectile을 직접 계산.',
        '행동 모델은 공통 offline counts로 적합한 categorical model. 외부 Q/V 잡음 없음.',
        '비교 알고리즘의 Q 업데이트 수, 학습률, EMA, replay, 배포 best-of-9를 맞춤.', '',
        '보상 프로토콜을 구분해야 한다:',
        'original: gamma=.999, 안전 종료 보상 .8904478499333031.',
        'fixed_reward (주 비교): 보상/전이는 그대로, gamma만 .99로 변경.',
        'matched_gap (보조 대조): gamma=.99이며 안전 종료 보상을 .8086528829953168로 재설정.',
        '  이는 uniform behavior에서 안전 보상이 clean population 경로 가치의 90%가 되게 한 것이다.',
        '  seed별 empirical behavior에서는 격차가 조금 달라진다. 이 보조 대조는 보상이 바뀐 MDP이다.', '',
        '핵심 결과:',
        '보상을 고정하면 gamma=.99의 FEQ는 penalty=0부터 목표 도달률이 낮다.',
        '이는 Q_tau의 작은 진행-안전 행동 gap과 finite-N expectile 고정점의 차이로도 재현된다.',
        '따라서 이 경우 성공률 저하를 전부 penalty 또는 미수렴 탓으로 해석할 수 없다.',
        '상대 보상 격차를 맞춘 보조 대조에서는 penalty=0의 세 방법 모두 약 100%이다.',
        '그러나 IQL/IQL-9 Q+V의 50% 전환 lambda는 .999/.99 모두 약 .0023으로 비슷하다.',
        'FEQ Q-only의 전환은 .999 약 .0089, .99 보조 대조 약 .0081이다.',
        '이 MDP에서는 gamma를 .99로 낮추는 것만으로 작은 lambda 민감도가 해소되지 않았다.', '',
        '선택 lambda에서의 목표 도달률 (%) 및 Q RMSE:',
    ]
    for protocol, rows in summaries.items():
        lines += ['', PROTOCOLS[protocol][0]]
        for algorithm in ALGORITHMS:
            for mode in MODES[algorithm]:
                for r in select(rows, algorithm, mode):
                    if r['strength'] in (0., .001, .003, .01):
                        lines.append(f'{ALGORITHMS[algorithm]} {mode:2s} lambda={r["strength"]:.6g}: '
                            f'goal {100*r["success_mean"]:.3f} +/- {100*r["success_sd"]:.3f}%, '
                            f'return {r["return_mean"]:.6f}, QRMSE {r["q_rmse_mean"]:.6f}')
    lines += ['', '50%를 처음 하회하는 관측 구간 (보간한 임계값이 아님):']
    for r in crossings:
        if r['cutoff'] == .5:
            lines.append(f'{r["protocol"]}: {r["algorithm"]} {r["penalty_mode"]}: '
                         f'[{r["lower"]:.9g}, {r["upper"]:.9g}]')
    lines += ['', 'Penalty=0의 exact fixed-point/실제 학습 대조:']
    for r in baselines:
        lines.append(f'{r["protocol"]} {r["algorithm"]}: clean population goal '
            f'{100*r["clean_population_success_mean"]:.3f}%, exact method fixed-point goal '
            f'{100*r["exact_success_mean"]:.3f}%, learned goal {100*r["learned_success_mean"]:.3f}%; '
            f'root progress-safe gap, population {r["population_root_gap_mean"]:.6f}, '
            f'method fixed point {r["exact_root_gap_mean"]:.6f}')
    lines += ['', '해석 범위:',
        'gamma를 낮추면 Bellman contraction의 최악 오차 전파 상한 1/(1-gamma)는 줄어든다.',
        '동시에 먼 목표의 현재 가치와 정책 순위도 바뀐다. 따라서 성공률이 반드시 좋아지는 것은 아니다.',
        '이 MDP는 terminal에 도달하는 짧은 경로다. 1000->100의 무한-horizon 상한 변화를 그대로 실제 증폭 배수로 해석하면 안 된다.',
        'tau와 behavior가 동일하므로 같은 Q-V 부호 패턴에서 expectile의 local curvature kappa는 gamma만으로 증가하지 않는다.',
        '작은 penalty 민감도에는 V 회귀의 곡률과 정책 action gap도 관여한다.',
        'FEQ의 finite-N sample expectile도 population expectile과 다르며, 안전 보상 격차가 작으면 이 차이만으로 선택이 바뀔 수 있다.',
        '이는 penalty=0의 exact method fixed point와 clean population reference도 함께 확인해야 하는 이유다.',
        '같은 lambda는 같은 크기의 최종 Q/V 출력 오차를 뜻하지 않는다.',
        '성공률은 목표 terminal 도달 확률이다. 안전 종료 선택은 성공 0이지만 양의 return을 준다.',
        'Q RMSE는 각 gamma/보상/empirical behavior의 무정규화 population fixed point Q_tau 기준이다.',
        '알고리즘의 보편적 우열이나 기본 IQL의 실제 학습 문제 전체에 대한 결론은 아니다.', '',
        '그래프: 실제 학습점 연결, spline/이동평균/단조화 없음. adaptive x 선택은 별도 가설검정이 아니다.',
        f'최소 격자 또는 6회 세분 제한에서도 8%p를 넘는 인접 구간 {len(remaining)}개를 그대로 보존했다.',
        'comparison_summary.csv: 전체 평균/SD; comparison_crossing_brackets.csv: 전환 구간.',
        'selected_strengths.csv: 공통 lambda 비교; zero_penalty_diagnostics.csv: 무정규화 원인 대조.',
        'figures/gamma_comparison_*: 세 프로토콜 비교; 각 프로토콜 폴더: 전체/확대/알고리즘별 그림.',
    ]
    (out/'REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete', gamma099_runs=audit['runs'], summary_rows=len(joint),
                         remaining_sharp_intervals=len(remaining))))


if __name__ == '__main__':
    main()
