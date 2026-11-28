"""Seed-level summaries and exportable figures for the registered v12 sweep."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .matched_error_v11 import save_csv, save_json
from .penalty_sweep_v12 import MODES, config_key, make_jobs


ALGORITHMS = {'iql': 'IQL', 'iql9': 'IQL-9', 'feq9': 'FEQ-N9'}
PENALTY_LABELS = {'q': 'Q only', 'v': 'V only', 'qv': 'Q + V'}
ALGORITHM_COLORS = {'iql': '#0072B2', 'iql9': '#D55E00', 'feq9': '#009E73'}
PENALTY_COLORS = {'q': '#0072B2', 'v': '#D55E00', 'qv': '#CC79A7'}
STYLES = {'q': ('-', 'o'), 'v': ('--', '^'), 'qv': (':', 's')}
METRICS = (
    ('success', 'Goal-reaching probability (%)', 100.),
    ('return', 'Actual discounted return', 1.),
    ('q_rmse', r'Q RMSE relative to $Q_\tau$', 1.),
)
XTICKS = [0., .0001, .001, .003, .01, .03]
XLABELS = ['0', '0.0001', '0.001', '0.003', '0.01', '0.03']


def mean_sd(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.


def xaxis(ax):
    ax.set_xscale('symlog', linthresh=.0001, linscale=.6)
    ax.set_xticks(XTICKS, XLABELS, rotation=20)
    ax.set_xlim(-.00001, .035)
    ax.set_xlabel(r'Penalty coefficient $\lambda$ (symlog; includes zero)')
    ax.grid(alpha=.18)


def curve(ax, rows, metric, scale, color, style='-', marker='o', label=None):
    x = np.array([r['strength'] for r in rows])
    y = np.array([r[metric+'_mean'] for r in rows])*scale
    sd = np.array([r[metric+'_sd'] for r in rows])*scale
    line, = ax.plot(x, y, linestyle=style, marker=marker, color=color,
                    markersize=4, linewidth=1.8, label=label)
    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.10)
    return line


def save_figure(fig, directory, name):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(directory/f'{name}.{extension}', dpi=210, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    out = root/'analysis'
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    audit = json.loads((root/'artifact_validation.json').read_text())
    assert audit['status'] == 'passed'
    jobs = make_jobs()
    records = []
    groups = defaultdict(list)
    for job in jobs:
        path = root/'raw'/config_key(job)
        r = json.loads((path/'result.json').read_text())['metrics']
        records.append(r)
        for mode in MODES[r['algorithm']] if r['strength'] == 0 else (r['penalty_mode'],):
            expanded = {**r, 'penalty_mode': mode}
            groups[r['stage'], r['algorithm'], mode, r['strength']].append(expanded)
    metric_names = [key for key in records[0] if key not in jobs[0]]
    summaries = []
    for (stage, algorithm, mode, strength), rs in sorted(groups.items()):
        expected_n = 16 if stage == 'main' else 4
        assert len(rs) == expected_n and len({r['seed'] for r in rs}) == expected_n
        row = dict(stage=stage, algorithm=algorithm, penalty_mode=mode, strength=strength,
                   seeds=' '.join(str(s) for s in sorted(r['seed'] for r in rs)),
                   n=expected_n, steps=rs[0]['steps'], lq=rs[0]['lq'], lv=rs[0]['lv'])
        for metric in metric_names:
            row[metric+'_mean'], row[metric+'_sd'] = mean_sd([r[metric] for r in rs])
        summaries.append(row)
    save_csv(out/'summary.csv', summaries)
    save_json(out/'summary.json', summaries)

    def select(algorithm, mode, stage='main'):
        return sorted((r for r in summaries if r['stage'] == stage and
                       r['algorithm'] == algorithm and r['penalty_mode'] == mode),
                      key=lambda r: r['strength'])

    thresholds = []
    for algorithm in ALGORITHMS:
        for mode in MODES[algorithm]:
            rs = select(algorithm, mode)
            row = dict(algorithm=algorithm, penalty_mode=mode)
            for cutoff in (.9, .5, .1):
                hits = [r['strength'] for r in rs if r['success_mean'] < cutoff]
                row[f'first_grid_lambda_below_{int(cutoff*100)}pct'] = min(hits) if hits else np.nan
            thresholds.append(row)
    save_csv(out/'grid_thresholds.csv', thresholds)
    # Paired contrasts only within one dataset/training seed, not between time points.
    contrasts = []
    for strength in sorted({r['strength'] for r in records if r['stage'] == 'main'}):
        rng = np.random.default_rng(1212)
        bootstrap = rng.integers(16, size=(20000, 16))
        for left_algorithm, left_mode, right_algorithm, right_mode in (
            ('iql', 'q', 'iql', 'v'), ('iql9', 'q', 'iql9', 'v'),
            ('iql', 'qv', 'feq9', 'q'), ('iql9', 'qv', 'feq9', 'q'),
            ('iql', 'q', 'feq9', 'q'), ('iql9', 'q', 'feq9', 'q'),
            ('iql', 'v', 'iql9', 'v'),
        ):
            left = sorted(groups['main', left_algorithm, left_mode, strength], key=lambda r:r['seed'])
            right = sorted(groups['main', right_algorithm, right_mode, strength], key=lambda r:r['seed'])
            assert [r['seed'] for r in left] == [r['seed'] for r in right]
            for metric, _, _ in METRICS:
                delta = np.array([r[metric]-l[metric] for l, r in zip(left, right)])
                lower, upper = np.quantile(delta[bootstrap].mean(-1), [.025, .975])
                contrasts.append(dict(strength=strength, left=f'{left_algorithm}_{left_mode}',
                    right=f'{right_algorithm}_{right_mode}', metric=metric,
                    right_minus_left_mean=float(delta.mean()), paired_sd=float(delta.std(ddof=1)),
                    ci95_low=float(lower), ci95_high=float(upper)))
    save_csv(out/'paired_contrasts.csv', contrasts)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    # All seven requested conditions in one figure.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    handles = []
    for algorithm, label in ALGORITHMS.items():
        for mode in MODES[algorithm]:
            rs = select(algorithm, mode)
            style, marker = STYLES[mode]
            for i, (metric, ylabel, scale) in enumerate(METRICS):
                line = curve(axes[i], rs, metric, scale, ALGORITHM_COLORS[algorithm],
                    style, 'D' if algorithm == 'feq9' else marker,
                    f'{label}: {PENALTY_LABELS[mode]}')
                if i == 0:
                    handles.append(line)
                axes[i].set_ylabel(ylabel)
                xaxis(axes[i])
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylim(.88, 1.002)
    axes[2].set_ylim(bottom=-.008)
    fig.suptitle('Penalty-strength sweep: IQL / IQL-9 / FEQ-N9', fontsize=15, y=.99)
    fig.text(.5, .92, '9 states, 4 actions; gamma=0.999, expectile=0.9; 16 seeds; last 10% of 2M updates; bands: +/- 1 seed SD', ha='center', fontsize=10)
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .12, 1, .89))
    save_figure(fig, figures, 'penalty_sweep_overview')

    # Separate rows make overlapped IQL/IQL-9 curves readable.
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey='col')
    for i, (algorithm, label) in enumerate(ALGORITHMS.items()):
        for j, (metric, ylabel, scale) in enumerate(METRICS):
            ax = axes[i, j]
            for mode in MODES[algorithm]:
                style, marker = STYLES[mode]
                curve(ax, select(algorithm, mode), metric, scale, PENALTY_COLORS[mode],
                      style, marker, PENALTY_LABELS[mode])
            ax.set_title(f'{label} — {ylabel}', fontsize=11)
            xaxis(ax)
            if i < 2:
                ax.set_xlabel('')
            if j == 0:
                ax.set_ylim(-3, 103)
            elif j == 1:
                ax.set_ylim(.88, 1.002)
            else:
                ax.set_ylim(bottom=-.008)
    axes[0, 0].legend(loc='lower left')
    axes[2, 0].text(.04, .12, 'FEQ has no V regression.\nOnly Q penalty is applicable.',
                   transform=axes[2, 0].transAxes, fontsize=10)
    fig.suptitle('Penalty location within each algorithm — mean +/- 1 seed SD', fontsize=15)
    fig.tight_layout(rect=(0, .01, 1, .96))
    save_figure(fig, figures, 'penalty_sweep_by_algorithm')

    # Mechanism quantities from independently solved expected-update fixed points.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    both = select('iql9', 'qv')
    for metric, label, color in (
        ('q_local_fit_rms', r'Local Q fitting error $\|\lambda Q\|_{RMS}$', '#0072B2'),
        ('v_score_residual_rms', r'V score residual $\|\lambda V\|_{RMS}$', '#D55E00'),
    ):
        curve(axes[0], both, metric, 1., color, label=label)
    for metric, label, color in (
        ('q_local_fit_rms', 'Local Q fitting error', '#0072B2'),
        ('v_output_fit_rms', 'V output fitting error', '#D55E00'),
    ):
        curve(axes[1], both, metric, 1., color, label=label)
    gains = []
    for row in both:
        if row['strength'] == 0:
            continue
        records_here = groups['main', 'iql9', 'qv', row['strength']]
        gain = [r['v_output_fit_rms']/r['v_score_residual_rms'] for r in records_here]
        average, sd = mean_sd(gain)
        gains.append((row['strength'], average, sd))
    xs, ys, sds = np.array(gains).T
    axes[2].plot(xs, ys, 'o-', color='#CC79A7', label='V output RMS / V score RMS')
    axes[2].fill_between(xs, ys-sds, ys+sds, color='#CC79A7', alpha=.15)
    for ax, label in zip(axes, ('Local source magnitude (RMS)', 'Output fitting error (RMS)', 'Amplification ratio')):
        xaxis(ax)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle('Both-penalty condition: local errors at the expected-update fixed point', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, figures, 'penalty_sweep_mechanism')

    # Full seed matrix exposes heterogeneity around policy-switch thresholds.
    fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    index = 0
    for algorithm, label in ALGORITHMS.items():
        for mode in MODES[algorithm]:
            rs = select(algorithm, mode)
            values = np.array([[r['success']*100 for r in sorted(
                groups['main', algorithm, mode, row['strength']], key=lambda r:r['seed'])] for row in rs]).T
            im = axes[index].imshow(values, vmin=0, vmax=100, cmap='viridis', aspect='auto', origin='lower')
            axes[index].set_title(f'{label}: {PENALTY_LABELS[mode]}', loc='left', fontsize=11)
            axes[index].set_yticks([0, 5, 10, 15], ['2600', '2605', '2610', '2615'])
            axes[index].set_ylabel('Seed')
            index += 1
    axes[-1].set_xticks(range(len(rs)), [f'{r["strength"]:g}' for r in rs], rotation=35)
    axes[-1].set_xlabel('Penalty coefficient lambda (discrete sampled settings)')
    fig.suptitle('Goal-reaching probability per training seed (last 10% of 2M updates)', fontsize=14)
    fig.tight_layout(rect=(0, .02, .91, .97))
    cax = fig.add_axes([.93, .15, .016, .7])
    fig.colorbar(im, cax=cax, label='Goal-reaching probability (%)')
    save_figure(fig, figures, 'penalty_sweep_seed_heatmap')

    lines = [
        'FEQ / IQL penalty sweep v12 — 2026-11-25', '',
        '설정: 기존 3번 실험의 9상태, 4행동 chain. gamma=.999, expectile tau=.9.',
        '기본 IQL은 replay 행동 하나로 V 회귀. IQL-9는 같은 상태 행동 9개의 gradient 평균.',
        'FEQ-N9는 next-state 행동 9개의 sample expectile. 세 방법 모두 단일 Q table.',
        '평가는 공통 best-of-9 정책. goal은 1000 step 내 목표 도달 확률을 정확히 계산.',
        '안전한 보상을 받고 종료하면 goal=0이어도 return은 약 .8904다.',
        '신경망 actor / Flow 학습, 외부 Gaussian 잡음, 추가 행동 모델 교란은 없다.',
        'Offline action counts 1024/state에서 추정한 같은 empirical behavior를 쓴다.',
        '',
        'Penalty: half-squared 회귀 손실에 lambda/2 * output^2 추가.',
        'Q only=(lambda_Q=lambda, lambda_V=0), V only=(0,lambda), Q+V=(lambda,lambda).',
        'FEQ에는 별도 V가 없어 Q only만 정의한다. Q+V는 두 회귀 각각에 lambda를 적용.',
        '같은 lambda가 같은 최종 출력 오차나 같은 상대적 regularization 강도를 의미하지 않는다.',
        '이는 명시적인 개입 실험이며, 기본 알고리즘이 원래 이 penalty를 쓴다는 뜻이 아니다.',
        '',
        'Main: seed 2600~2615, 각 2M, 마지막 10% (21 checkpoint)의 seed별 평균.',
        'lambda = '+', '.join(f'{r["strength"]:g}' for r in select('feq9', 'q')),
        'Long: 새로운 seed 2650~2653, 각 5M, lambda=0/.0015/.003/.01.',
        '전체 1376개 독립 학습 실행 (main 1280 + long 96). lambda=0 중복곡선은 같은 baseline 재사용.',
        '오차막대/음영: 학습 seed 간 sample SD. checkpoint를 독립 sample로 세지 않음.',
        'Q RMSE 기준: 동일 empirical behavior의 무정규화 population-expectile 고정점 Q_tau.',
        '정규화된 각 방법의 고정점으로부터의 거리도 별도로 기록.',
        '',
        'Main 결과: 평균 +/- seed SD',
        'algorithm | penalty | lambda | goal (%) | return | Q RMSE',
    ]
    for strength in (0., .001, .002, .003, .005, .007, .01, .03):
        for algorithm in ALGORITHMS:
            for mode in MODES[algorithm]:
                row = next(r for r in select(algorithm, mode) if r['strength'] == strength)
                lines.append(f'{ALGORITHMS[algorithm]:7} | {mode:2} | {strength:g} | '
                    f'{100*row["success_mean"]:.3f} +/- {100*row["success_sd"]:.3f} | '
                    f'{row["return_mean"]:.6f} +/- {row["return_sd"]:.6f} | '
                    f'{row["q_rmse_mean"]:.6f} +/- {row["q_rmse_sd"]:.6f}')
    lines += ['', 'Long-run 확인: 새로운 4시드, 5M (main과 다른 시드이므로 paired 시간 비교는 아님)',
              'algorithm | penalty | lambda | goal (%) | return | Q RMSE']
    for row in summaries:
        if row['stage'] == 'long':
            lines.append(f'{ALGORITHMS[row["algorithm"]]} | {row["penalty_mode"]} | '
                f'{row["strength"]:g} | {row["success_mean"]*100:.3f} +/- {row["success_sd"]*100:.3f} | '
                f'{row["return_mean"]:.6f} | {row["q_rmse_mean"]:.6f}')
    lines += ['', '읽는 방법 및 한계:',
        '1. V-only와 Q+V는 Q-only보다 작은 penalty에서 목표 선택을 잃는다.',
        '2. Q-only에서는 IQL/IQL-9가 FEQ보다 Q_tau RMSE가 더 작다. FEQ의 보편적 우월성은 아니다.',
        '3. IQL-9는 V gradient 표본 수를 늘리지만 penalty가 만드는 목적함수 편향은 제거하지 못한다.',
        '4. 경계에서는 작은 action-gap 변화가 큰 goal 차이를 만든다. return과 RMSE를 함께 본다.',
        '5. IQL과 IQL-9의 기대 업데이트 고정점은 같지만, 고정 학습률의 확률적 궤적 평균이',
        '   그 고정점과 정확히 같을 필요는 없다. 단일표본과 9표본의 차이가 항상 단조롭지 않다.',
        '6. main/long의 seed 집합은 다르다. 2M과 5M의 작은 차이를 학습 연장 효과로 단정하지 않는다.',
        '7. 유한 empirical dataset과 학습 난수를 함께 바꾸는 training seed SD이며, rollout 평가 잡음은 없다.',
        '', '파일:',
        'summary.csv/json: 모든 조건, 실제 return, Q RMSE, 정확한 고정점, local residual, seed SD.',
        'paired_contrasts.csv: 같은 seed를 짝지은 20,000회 bootstrap CI.',
        'grid_thresholds.csv: 실험한 lambda 격자에서 처음 목표 확률이 90/50/10%를 하회하는 값.',
        'figures/penalty_sweep_overview: 7곡선 전체 비교.',
        'figures/penalty_sweep_by_algorithm: 알고리즘별 Q/V/Q+V 분리.',
        'figures/penalty_sweep_mechanism: 기대 고정점에서 잔차와 출력 오차.',
        'figures/penalty_sweep_seed_heatmap: 모든 main seed의 목표 확률.',
        '그림은 PNG, PDF, SVG로 저장. x축 symlog는 0을 포함하며, heatmap은 이산 열이다.',
        '', '독립 검증: ../preflight_validation.json, ../artifact_validation.json.',
    ]
    (out/'REPORT_KO.txt').write_text('\n'.join(lines)+'\n')
    save_json(out/'analysis_manifest.json', dict(status='complete', unique_trajectories=len(records),
        plotted_condition_groups=len(summaries), bootstrap_resamples=20000,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=sorted(p.name for p in figures.iterdir())))
    print(json.dumps(dict(status='complete', trajectories=len(records), summary_groups=len(summaries))))


if __name__ == '__main__':
    main()
