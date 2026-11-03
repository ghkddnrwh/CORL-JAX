"""Scientific figure for controlled policy-path comparisons; no nine-task claim."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    identity = json.loads((root/'analysis/late_policy_family_value_identity.json').read_text())
    density = json.loads((root/'analysis/factorized_density_controls/summary.json').read_text())
    policies = [
        ('w1_scene_bounded_v2_g999_t90_s0', 'Gaussian AWR\n1 action'),
        ('m1_scene_gaussian_bc_bounded_v2_g999_t90_s0', 'Gaussian BC\nbest32'),
        ('m1_scene_mixture_bc_bounded_v2_g999_t90_s0', 'Mixture BC, K5\nbest32'),
        ('p2_scene_fql_policy_bounded_v2_g999_t90_s0', 'Flow Q-distill\n1 action'),
        ('p1_scene_ifql_c0_bounded_v2_g999_t90_s0', 'BC flow\nbest32'),
    ]
    scene = []
    for name, label in policies:
        scores = []
        for step in (800000, 900000, 1000000):
            group = next(g for g in identity['groups'] if g['task'] == 'scene' and g['step'] == step)
            record = next(r for r in group['records'] if r['run'] == name)
            assert record['same_replay_rng']
            assert all(v['bitwise_equal'] for v in record['modules'].values())
            scores.append(record['primary_success_at_this_step'])
        scene.append(dict(run=name, label=label, scores=scores, mean=statistics.mean(scores)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.65, 1]})
    x = np.arange(len(scene))
    colors = ['#929ca3', '#658a9e', '#497c9b', '#a5a46f', '#277e8e']
    ax = axes[0]
    means = [r['mean'] for r in scene]
    ax.bar(x, means, color=colors, width=.7)
    for i, row in enumerate(scene):
        ax.scatter(x[i]+np.array([-.12, 0, .12]), row['scores'], c='black', s=20, zorder=4)
        ax.text(x[i], max(row['scores'])+.035, f"{row['mean']:.3f}", ha='center', fontsize=10)
    ax.set_xticks(x, [r['label'] for r in scene], fontsize=9)
    ax.set_title('Scene: last-three primary evaluations\nQ, target-Q and V are bitwise identical at each step', fontsize=11)
    ax.set_ylabel('Success rate')
    ax = axes[1]
    values = density['success_with_common_Q']
    labels = ['Gaussian BC\nbest32', 'Mixture BC, K5\nbest32', 'BC flow\nbest32']
    scores = [values['GaussianBC_K1_best32'], values['GaussianMixture_K5_best32'], values['BC_flow_best32']]
    ax.bar(np.arange(3), scores, color=[colors[1], colors[2], colors[4]], width=.65)
    for i, score in enumerate(scores):
        ax.text(i, score+.035, f'{score:.2f}', ha='center', fontsize=10)
    ax.set_xticks(np.arange(3), labels, fontsize=9)
    ax.set_title('Puzzle-3x3: frozen 600K auxiliary evaluation\nThree proposal models use the same IFQL critic', fontsize=11)
    for ax in axes:
        ax.set_ylim(0, 1.075)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, .035, 'One training seed. Scene dots: 800K, 900K and 1M, each50episodes; these are not independent training seeds.\n'
             'Different policies have different training objectives and inference costs. Flow uses10Eulersteps. Auxiliary Puzzle scores never replace primary evaluations.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'analysis/policy_extraction_controls.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (root/'analysis/policy_extraction_controls.json').write_text(json.dumps(dict(
        scene_last3=scene, puzzle_common_critic_600k=values,
        caveat='Controlled policy-path evidence on two pilot tasks, not a nine-task macro or a statistical significance result.'), indent=2))


if __name__ == '__main__':
    main()
