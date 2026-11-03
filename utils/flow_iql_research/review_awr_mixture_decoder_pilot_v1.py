"""Strictly review the completed negative Ant mixture-extraction intervention."""
import argparse
import json
from pathlib import Path
import statistics
import time

from utils.flow_iql_research import core  # Runtime compatibility before pickle.
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def episodes(record):
    assert record['n_episodes'] == 50 and record['first_episode_seed'] == 42
    assert record['eval_device'] == 'cpu' and record['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
    for name in ('episode_successes', 'episode_rewards', 'episode_lengths'):
        assert len(record[name]) == 50 and np.isfinite(record[name]).all()
    assert set(record['episode_successes']) <= {0., 1.}
    assert record['success_rate'] == statistics.mean(record['episode_successes'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'analysis/Ant_AWR_mixture_decoder_v1'
    assert not (output / 'strict_review_v1.json').exists()
    verifier = Verifier()
    inputs = {}

    def read(path):
        inputs[str(path)] = verifier.digest(path)
        return json.loads(path.read_text())

    plan = read(root / 'Ant_AWR_mixture_decoder_registration_v1.json')
    run = Path(plan['run'])
    conf = read(run / 'config.json')
    assert inputs[str(run / 'config.json')] == plan['config_sha256']
    for path, digest in {**conf['source_sha256'], **plan['source_sha256']}.items():
        assert verifier.digest(path) == digest
    done = read(run / 'completed.json')
    assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
    assert not (run / 'stopped.json').exists()
    summary = read(output / 'summary.json')
    assert summary['status'] == 'complete' and summary['valid_evaluations'] == 7 and not summary['failures']
    raw_path = run / 'evaluations.jsonl'
    inputs[str(raw_path)] = verifier.digest(raw_path)
    primary = [json.loads(line) for line in raw_path.read_text().splitlines()]
    baseline, checkpoints = {}, {}
    for step in (800000, 900000, 1000000):
        found = [r for r in primary if r['step'] == step]
        assert len(found) == 1
        episodes(found[0])
        baseline[step] = found[0]
        checkpoints[step] = verifier.checkpoint(run / f'checkpoint_{step}.pkl', conf, step)
    records = []
    for job in plan['jobs']:
        result = read(Path(job['output']))
        episodes(result)
        assert result['rule'] == job['rule'] and result['step'] == job['step']
        assert result['environment'] == conf['identity']['environment']
        assert result['config_sha256'] == plan['config_sha256']
        assert result['checkpoint_sha256'] == checkpoints[job['step']]
        assert result['actual_Q_Adam_count'] == job['step'] and result['actual_V_Adam_count'] == 2 * job['step']
        assert result['source_sha256'] == plan['source_sha256']['utils/flow_iql_research/evaluate_awr_mixture_decoder_v1.py']
        if job.get('self_reference'):
            for field in ('episode_successes', 'episode_rewards', 'episode_lengths'):
                assert result[field] == baseline[job['step']][field]
            assert all(result['self_reference_episode_arrays_exact'].values())
        records.append(result)
    rows = [dict(rule='highest_probability_mean', per_step={str(s): baseline[s]['success_rate'] for s in baseline},
        last3_mean=statistics.mean(r['success_rate'] for r in baseline.values()))]
    for rule in ('component_Q', 'sample9_Q'):
        available = [r for r in records if r['rule'] == rule]
        assert len(available) == 3
        rows.append(dict(rule=rule, per_step={str(r['step']): r['success_rate'] for r in available},
            last3_mean=statistics.mean(r['success_rate'] for r in available)))
    fixed = read(root / 'analysis/Ant_AWR_mixture_decoder_fixed_queries_v1/review.json')
    assert fixed['status'] == 'complete' and len(fixed['models']) == 3 and len(fixed['rows']) == 6
    assert verifier.digest('utils/flow_iql_research/inspect_awr_mixture_decoder_v1.py') == fixed['source_sha256']
    assert fixed['selector_source_sha256'] == plan['source_sha256']['utils/flow_iql_research/evaluate_awr_mixture_decoder_v1.py']
    assert fixed['config_sha256'] == plan['config_sha256']
    assert all(m['checkpoint_sha256'] == checkpoints[m['step']] for m in fixed['models'])
    validation = [r for r in fixed['rows'] if r['split'] == 'validation']
    support = dict(Q_component_changed_fraction=statistics.mean(r['Q_component_decoder_differs_from_most_probable_fraction'] for r in validation),
        Q_selected_component_weight_below_point1_fraction=statistics.mean(r['Q_selected_component_weight_below_point1_fraction'] for r in validation),
        selected_component_median_probability_per_step=[r['Q_selected_component_probability_quantiles'][1] for r in validation],
        methods=[])
    for rule in ('component_Q', 'sample9_Q'):
        found = [next(m for m in r['methods'] if m['rule'] == rule) for r in validation]
        support['methods'].append(dict(rule=rule, **{key: statistics.mean(r[key] for r in found)
            for key in found[0] if key != 'rule'}))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    colors = ['#626b74', '#447e9c', '#b77548']
    labels = ['Most-probable mean', 'Q over5means', 'Q over9samples']
    for row, label, color in zip(rows, labels, colors):
        axes[0].plot([800, 900, 1000], list(row['per_step'].values()), 'o-', label=label, color=color)
    axes[0].set(xlabel='Updates (thousands)', ylabel='CPU50 success', ylim=(-.02, .3), title='All three late evaluations')
    axes[0].legend(frameon=False, fontsize=7, loc='upper right')
    for axis, key, title in zip(axes[1:], ('own_Q_minus_greedy_mean', 'fitted_preclip_log_density_minus_greedy_mean'),
        ('Higher model Q did not improve return', 'Lower fitted density is a proxy')):
        axis.bar([0, 1], [r[key] for r in support['methods']], color=colors[1:])
        axis.set_xticks([0, 1], labels[1:], rotation=15)
        axis.set_title(title, fontsize=9)
        axis.axhline(0, color='#333333', lw=.7)
    axes[1].set_ylabel('Own Q change on validation queries')
    axes[2].set_ylabel('Fitted pre-clipping log density change')
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Ant AWR mixture: same actor and critic, action extraction only', fontsize=11)
    fig.text(.5, .025, 'Same actual800K/900K/1M models, CPU50 seeds42..91. Baseline last3=.1867; both Q decoders=0.\nThe fixed512 validation observations are reused across checkpoints. Mixture density and Q are model proxies, not true support/advantage.\nThis is one discovery seed. No new training or changes to either heldout BC-flow ensemble confirmation.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'mechanism_v1.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    review = dict(time=time.time(), status='complete', actual_checkpoints_verified=len(verifier.checkpoints),
        raw_CPU50_evaluations_verified=7, baseline_raw_evaluations_verified=3, rows=rows,
        support_proxy=support, inputs_sha256=inputs, source_sha256=verifier.digest(__file__),
        interpretation='Frozen decoder changes raise own-model Q while reducing observed Ant return. Low fitted-density component selection is consistent with exploiting weakly constrained candidate values, but does not identify true support/rank errors or propagation causality.',
        scope='One Ant seed. No nine-task mean, fresh replicate, universal rejection of GMM policies or modification of the locked candidate.', goal_complete=False)
    (output / 'strict_review_v1.json').write_text(json.dumps(review, indent=2, allow_nan=False))
    print(json.dumps(dict(rows=rows, support_proxy=support)), flush=True)


if __name__ == '__main__':
    main()
