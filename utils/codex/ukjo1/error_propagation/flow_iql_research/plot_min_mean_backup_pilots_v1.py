"""Cross-check the complete pilot comparisons and plot the three late epochs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--min-audit', type=Path, required=True)
    parser.add_argument('--mean-audit', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    compare = json.loads(args.comparison.read_text())
    minimum = json.loads(args.min_audit.read_text())
    mean = json.loads(args.mean_audit.read_text())
    assert compare['status'] == 'complete' and compare['completed_pilots'] == 3
    assert compare['actual_unique_late_checkpoints_verified'] == 18
    assert minimum['completed_tasks'] == 9 and mean['completed_pilots'] == 3
    assert compare['source_sha256'] == sha('utils/flow_iql_research/audit_min_mean_backup_pilots_v1.py')
    assert mean['source_sha256'] == sha('utils/flow_iql_research/audit_mean_backup_pilots_v1.py')
    primary_files = {}
    for row in compare['rows']:
        baseline = next(r for r in minimum['rows'] if r['environment'] == row['environment'])
        variant = next(r for r in mean['rows'] if r['environment'] == row['environment'])
        assert abs(baseline['last3_mean'] - row['min_target_last3']) < 1e-12
        assert abs(variant['last3_mean'] - row['mean_target_last3']) < 1e-12
        for cell in row['cells']:
            a = next(c for c in baseline['cells'] if c['step'] == cell['step'])
            b = next(c for c in variant['cells'] if c['step'] == cell['step'])
            assert a['checkpoint_sha256'] == cell['checkpoint_sha256']['min']
            assert b['checkpoint_sha256'] == cell['checkpoint_sha256']['mean']
            assert abs(a['CPU50_success'] - cell['min_target_CPU50_success']) < 1e-12
            assert abs(b['success'] - cell['mean_target_CPU50_success']) < 1e-12
        root = args.comparison.resolve().parent.parent
        for audited in (baseline, variant):
            path = root / 'runs' / audited['run'] / 'evaluations.jsonl'
            assert sha(path) == audited.get('evaluation_file_sha256', audited.get('evaluation_sha256'))
            primary_files[str(path)] = sha(path)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), sharey=True)
    labels = dict(ant='Antmaze large', cubed='Cube double', pz3='Puzzle3x3')
    for axis, row in zip(axes, compare['rows']):
        cells = row['cells']
        x = [cell['step'] / 1000 for cell in cells]
        axis.plot(x, [cell['min_target_CPU50_success'] for cell in cells], 'o-', color='#69747d', label='Min target-Q')
        axis.plot(x, [cell['mean_target_CPU50_success'] for cell in cells], 'o-', color='#447e9c', label='Mean target-Q')
        axis.set(title=labels[row['short']], xlabel='Updates (thousands)', xticks=[800, 900, 1000], ylim=(-.06, 1.08))
        text = f"Last3: {row['min_target_last3']:.3f} / {row['mean_target_last3']:.3f}\nBC exact: {sum(c['BC']['exact'] for c in cells)}/3 checkpoints"
        axis.text(.03, .43, text, transform=axis.transAxes, fontsize=9)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('CPU50 success')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .19), ncol=2, frameon=False)
    fig.suptitle('Finite-expectile target min/mean: same BC parameters, different critic effects', fontsize=12)
    fig.text(.5, .025, 'All18 actual late checkpoints and raw CPU50 were checked. Same replay/agent RNG, tau=.9, gamma=.999, N9 and min-Q deployment.\nThis is a separate finite-backup pilot branch:3tasks and one seed, not the nine-task comparison or independent critic-rank policy.\nExact BC supports a critic-source explanation, not true Q/residual/B_k labels. No per-environment winner mixture or new training.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .28, 1, .93))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = args.output_prefix.with_suffix('.' + ext)
        assert not path.exists()
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    review = dict(time=time.time(), status='complete', source_sha256=sha(__file__),
        inputs_sha256={str(path): sha(path) for path in (args.comparison, args.min_audit, args.mean_audit)},
        raw_primary_evaluation_files_sha256=primary_files,
        actual_models=18, pilot_tasks=3, all_available_BC_parameters_exact=all(
            r['all_available_BC_parameters_exact'] for r in compare['rows']),
        method_means=dict(min=statistics.mean(r['min_target_last3'] for r in compare['rows']),
            mean=statistics.mean(r['mean_target_last3'] for r in compare['rows'])),
        nine_task_mean=None, goal_complete=False,
        scope='Complete fixed pilot comparison; no9task claim, confidence interval, or fresh training replication.')
    path = args.output_prefix.with_suffix('.json')
    assert not path.exists()
    path.write_text(json.dumps(review, indent=2, allow_nan=False))
    print(json.dumps(dict(status='complete', method_means=review['method_means'],
        all_BC_exact=review['all_available_BC_parameters_exact'])), flush=True)


if __name__ == '__main__':
    main()
