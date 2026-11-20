"""Draw the complete primary metric and visibly incomplete confirmation rows."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    assert not review['goal_complete']
    choices = [
        ('Gaussian IQL, original gamma (s0)', 'Official Gaussian IQL, original gamma'),
        ('Gaussian IQL V2, original gamma (s0)', 'V2 Gaussian IQL, original gamma'),
        ('Gaussian IQL, gamma .999 (s0)', 'Official Gaussian IQL, gamma.999'),
        ('Gaussian IQL V2, gamma .999 (s0)', 'V2 Gaussian IQL, gamma.999'),
        ('Native IFQL, best32 (s0)', 'Native IFQL, N32'),
        ('Native IFQL, best9 (s0)', 'Native IFQL, matched N9'),
        ('V2 IFQL, best32 (s0)', 'V2 IFQL, N32'),
        ('V2 IFQL, best9 (s0)', 'V2 IFQL, N9'),
        ('V2 IFQL, best9 + K3 (3-seed mean)', 'V2 + temporal K3, mean of3policy seeds'),
        ('V2 IFQL independent3 mean (discovery group0)', 'V2 + independent3 raw Q mean, discovery'),
        ('V2 IFQL independent3 median_rank (discovery group0)', 'V2 + independent3 median ranks, discovery'),
        ('V2 IFQL independent3 mean_rank (discovery group0)', 'V2 + independent3 mean ranks, discovery'),
        ('FQL, fresh reference (s0)', 'FQL reference, one training seed'),
        ('ReBRAC, fresh reference (s0)', 'ReBRAC reference, one training seed'),
        ('Finite expectile backup, best9 (s0)', 'Finite backup, twin minimum, N9'),
        ('Mean-head finite expectile backup, best9 (s0)', 'Finite backup, twin mean, N9 (incomplete)'),
        ('First BC own critic fresh group1 (paired baseline)', 'First BC3, own critic (paired baseline)'),
        ('Independent3 mean_rank fresh group1 (preferred)', 'Locked mean ranks, fresh group3/4/5'),
        ('First BC own critic fresh group2 (paired baseline)', 'First BC6, own critic (paired baseline)'),
        ('Independent3 mean_rank fresh group2 (preferred)', 'Locked mean ranks, fresh group6/7/8'),
    ]
    selected = []
    for key, label in choices:
        row = next(m for m in review['methods'] if m['label'] == key)
        selected.append((row, label))
    array = np.asarray([[np.nan if v is None else v for v in row['values'] + [row['nine_task_mean']]]
        for row, _ in selected])
    fig, axis = plt.subplots(figsize=(14.6, 10.3))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#e7e9eb')
    shown = axis.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axis.set_xticks(range(10), ['AM-L', 'HM-M', 'HM-L', 'AS-A', 'Cube-S', 'Cube-D', 'Pz-3', 'Pz-4', 'Scene', 'Avg'])
    axis.set_yticks(range(len(selected)), [label for _, label in selected])
    axis.xaxis.tick_top()
    axis.tick_params(length=0, pad=7, labelsize=9)
    for i in range(len(selected)):
        for j, value in enumerate(array[i]):
            text = 'pending' if np.isnan(value) else f'{value:.4f}' if j == 9 else f'{value:.2f}'
            axis.text(j, i, text, ha='center', va='center', fontsize=8,
                color='#61666b' if np.isnan(value) else 'black' if value > .55 else 'white',
                fontweight='bold' if j == 9 else 'normal')
    axis.axvline(8.5, color='white', lw=2)
    highlight = next(i for i,(row,_) in enumerate(selected) if row['label'] == 'V2 IFQL independent3 mean_rank (discovery group0)')
    axis.add_patch(plt.Rectangle((-.5, highlight-.5), 10, 1, fill=False, edgecolor='#ce7f24', lw=2.4))
    axis.axhline(next(i for i,(row,_) in enumerate(selected) if row['label'].startswith('First BC own critic fresh group1'))-.5, color='white', lw=2)
    fig.colorbar(shown, ax=axis, fraction=.022, pad=.02, label='Success rate')
    fresh = next(i for i,(row,_) in enumerate(selected) if row['label'] == 'Independent3 mean_rank fresh group1 (preferred)')
    axis.add_patch(plt.Rectangle((-.5, fresh-.5), 10, 1, fill=False, edgecolor='#cc3c43', lw=2.4))
    fig.suptitle('Matched N9: native=.3126, V2=.4089; first fresh mean ranks=.3481 (target .47)', fontsize=12)
    fig.text(.5, .025, 'Equal means over800K /900K /1M CPU50 and nine tasks. Except the first two rows, gamma=.999; tau=.9 where applicable.\nIndependent3 costs3base learners: discovery0/1/2 is ONE ensemble group. Fresh3/4/5 and6/7/8 were locked before outcomes.\nTemporalK3 uses correlated snapshots, not independent critics. Provided plain FQL=.47, ReBRAC=.44; DD-ReBRAC=.63.\nNative/V2 own BC weights differ; common-BC ablation is running. Pending is never zero-filled. These rows do not establish equal-compute superiority, independent confirmation, SOTA, or universally lower seed variability.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .14, .99, .95))
    for ext in ('png', 'pdf'):
        path = args.review.parent / f'common9_overview_v25.{ext}'
        assert not path.exists()
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
