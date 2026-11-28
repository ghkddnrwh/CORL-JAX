"""Draw measured penalty curves and linear-scale transition zoom panels."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .matched_error_v11 import save_csv, save_json
from .penalty_sweep_v12 import MODES
from .refine_penalty_v13 import grouped
from .plot_penalty_sweep_v12 import (
    ALGORITHMS, ALGORITHM_COLORS, PENALTY_COLORS, PENALTY_LABELS,
    STYLES, METRICS, mean_sd, xaxis, save_figure,
)


def summarize(records):
    groups = grouped(records)
    fields = [k for k in records[0]['metrics'] if k not in records[0]['config']]
    rows = []
    for (algorithm, mode, strength), rs in sorted(groups.items()):
        row = dict(algorithm=algorithm, penalty_mode=mode, strength=strength, n=len(rs),
                   seeds=' '.join(str(s) for s in sorted(r['seed'] for r in rs)))
        for field in fields:
            row[field+'_mean'], row[field+'_sd'] = mean_sd([r[field] for r in rs])
        rows.append(row)
    return rows


def draw(ax, rs, metric, scale, algorithm, mode, by_mode=False, label=None):
    x = [r['strength'] for r in rs]
    y = np.array([r[metric+'_mean'] for r in rs])*scale
    sd = np.array([r[metric+'_sd'] for r in rs])*scale
    color = PENALTY_COLORS[mode] if by_mode else ALGORITHM_COLORS[algorithm]
    style, marker = STYLES[mode]
    line, = ax.plot(x, y, color=color, linestyle=style, marker=marker,
                    markersize=2.4, linewidth=1.6, label=label)
    ax.fill_between(x, y-sd, y+sd, color=color, alpha=.10)
    return line


def crossing_brackets(rows):
    result = []
    for algorithm in ALGORITHMS:
        for mode in MODES[algorithm]:
            rs = sorted((r for r in rows if r['algorithm']==algorithm and r['penalty_mode']==mode),
                        key=lambda r:r['strength'])
            for level in (.9, .5, .1):
                hits = [i for i, r in enumerate(rs) if r['success_mean'] < level]
                if not hits:
                    result.append(dict(algorithm=algorithm, penalty_mode=mode, cutoff=level,
                        lower=np.nan, upper=np.nan, lower_success=np.nan, upper_success=np.nan))
                    continue
                i = hits[0]
                result.append(dict(algorithm=algorithm, penalty_mode=mode, cutoff=level,
                    lower=rs[max(0,i-1)]['strength'], upper=rs[i]['strength'],
                    lower_success=rs[max(0,i-1)]['success_mean'], upper_success=rs[i]['success_mean']))
    return result


def render(records, output, gamma=.999, description='Original MDP rewards', prefix='penalty_refined'):
    output = Path(output)
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    rows = summarize(records)
    save_csv(output/'summary.csv', rows)
    save_json(output/'summary.json', rows)
    crossings = crossing_brackets(rows)
    save_csv(output/'crossing_brackets.csv', crossings)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    select = lambda algorithm, mode: sorted((r for r in rows
        if r['algorithm']==algorithm and r['penalty_mode']==mode), key=lambda r:r['strength'])
    returns = [r['return_mean'] for r in rows]
    low, high = min(returns), max(returns)
    margin = max(.01, .08*(high-low))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    handles = []
    for algorithm in ALGORITHMS:
        for mode in MODES[algorithm]:
            rs = select(algorithm, mode)
            for index, (metric, ylabel, scale) in enumerate(METRICS):
                line = draw(axes[index], rs, metric, scale, algorithm, mode,
                    label=f'{ALGORITHMS[algorithm]}: {PENALTY_LABELS[mode]}')
                if index == 0:
                    handles.append(line)
                xaxis(axes[index])
                axes[index].set_ylabel(ylabel)
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylim(low-margin, high+margin)
    axes[2].set_ylim(bottom=-.005)
    fig.suptitle(f'Measured penalty sweep: gamma={gamma:g} — {description}', fontsize=14, y=.99)
    fig.text(.5, .925, '16 paired seeds; 2M updates; last 10% average; bands: +/- 1 seed SD; dots are actual trained conditions', ha='center')
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.12,1,.89))
    save_figure(fig, figures, prefix+'_overview')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for mode, ax in zip(('qv','v','q'), axes):
        locations = []
        for algorithm in ALGORITHMS:
            if mode not in MODES[algorithm]:
                continue
            rs = select(algorithm, mode)
            draw(ax, rs, 'success', 100., algorithm, mode, label=ALGORITHMS[algorithm])
            active = [i for i,r in enumerate(rs) if .01 < r['success_mean'] < .99]
            if active:
                locations.extend((rs[max(0,active[0]-1)]['strength'], rs[min(len(rs)-1,active[-1]+1)]['strength']))
        if locations:
            lo, hi = min(locations), max(locations)
            pad = .04*max(hi-lo, 1e-6)
            ax.set_xlim(max(0,lo-pad), hi+pad)
        ax.set_xlabel(r'Penalty coefficient $\lambda$ (linear scale)')
        ax.set_title(PENALTY_LABELS[mode], fontsize=13)
        ax.set_ylim(-3,103)
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useMathText=True)
        ax.legend()
    axes[0].set_ylabel('Goal-reaching probability (%)')
    fig.suptitle(f'Transition detail — gamma={gamma:g}; {description}', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93))
    save_figure(fig, figures, prefix+'_transition_zoom')

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey='col')
    for i, algorithm in enumerate(ALGORITHMS):
        for j, (metric, ylabel, scale) in enumerate(METRICS):
            ax = axes[i,j]
            for mode in MODES[algorithm]:
                draw(ax, select(algorithm,mode), metric, scale, algorithm, mode,
                     by_mode=True, label=PENALTY_LABELS[mode])
            xaxis(ax)
            if i<2:
                ax.set_xlabel('')
            ax.set_title(f'{ALGORITHMS[algorithm]} — {ylabel}',fontsize=10)
            if j==0:
                ax.set_ylim(-3,103)
            elif j==1:
                ax.set_ylim(low-margin,high+margin)
            else:
                ax.set_ylim(bottom=-.005)
    axes[0,0].legend()
    axes[2,0].text(.03,.12,'FEQ has no V regression.\nOnly Q penalty is applicable.',transform=axes[2,0].transAxes)
    fig.suptitle(f'Fine penalty grid — gamma={gamma:g}; {description}',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    save_figure(fig,figures,prefix+'_by_algorithm')
    return rows, crossings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    audit = json.loads((root/'artifact_validation.json').read_text())
    assert audit['status']=='passed'
    records = json.loads((root/'artifact_index.json').read_text())
    rows, crossings = render(records, root/'analysis')
    remaining = json.loads((root/'remaining_sharp_intervals.json').read_text())
    text = [
        'Penalty transition refinement v13', '',
        '기존 v12의 gamma=.999, 9상태 MDP, 같은 seed 2600~2615, 각 2M 업데이트.',
        '학습/평가 코드는 그대로이며, 성공률이 변하는 penalty 구간의 실제 학습점을 추가.',
        '처음에는 기존 5%p 초과 변화 구간을 10분할, 이후 남은 5%p 초과 구간의 중점을 학습.',
        '같은 penalty 위치에 대해서는 비교 알고리즘들이 같은 lambda 격자를 공유한다.',
        '예: Q-only는 IQL/IQL9/FEQ, V-only와 Q+V는 IQL/IQL9를 함께 추가.',
        f'추가 학습 {audit["new_runs"]}개, 기존 재사용 {audit["reused_runs"]}개.',
        '모든 추가 lambda는 동일한 16개 시드, 같은 offline counts 및 replay 난수를 사용.',
        '집계는 각 seed의 마지막 10% 평균 후 seed 간 평균 +/- sample SD.',
        '그래프는 실제 측정점을 연결했다. 이동평균, spline, 인위적인 단조화는 사용하지 않음.',
        '인접 점 사이에 여전히 큰 차이가 있으면 최소격자/실제 정책 전환의 결과를 그대로 보존.',
        f'최소격자 또는 라운드 제한에서 남은 5%p 초과 구간: {len(remaining)}개.',
        '', '50% 성공률을 처음 하회하는 측정 구간:',
    ]
    for r in crossings:
        if r['cutoff']==.5:
            text.append(f'{ALGORITHMS[r["algorithm"]]} {PENALTY_LABELS[r["penalty_mode"]]}: '
                f'{r["lower"]:.8g} ({100*r["lower_success"]:.2f}%) -> '
                f'{r["upper"]:.8g} ({100*r["upper_success"]:.2f}%)')
    text += ['', '주의: goal 도달 실패는 작은 안전 보상을 선택한 경우를 포함한다.',
        'Q RMSE 기준은 무정규화 population-expectile 고정점 Q_tau이다.',
        '기본 알고리즘에 이 penalty가 원래 존재한다는 뜻은 아니며, 같은 lambda는 같은 출력 오차를 의미하지 않는다.',
        'summary.csv / crossing_brackets.csv / ../per_seed.csv에 전체 수치가 있다.',
        'figures/에 전체 비교, 전환 구간 확대, 알고리즘별 분리를 PNG/PDF/SVG로 저장했다.',
    ]
    (root/'analysis'/'REPORT_KO.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='complete', new_runs=audit['new_runs'], groups=len(rows))))


if __name__=='__main__':
    main()
