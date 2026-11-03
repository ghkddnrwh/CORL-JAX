"""Describe the locked rank-mean rule on archived common-proposal queries.

This postprocessing neither queries new checkpoints nor evaluates/tunes a policy.
The input contains already-seen discovery models, not fresh confirmations.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(q):
    return (q[:, None, :] < q[:, :, None]).sum(2) + .5 * (q[:, None, :] == q[:, :, None]).sum(2)


def describe(q):
    assert q.shape[:2] == (3, 9) and np.isfinite(q).all()
    rr = ranks(q)
    normalized = (rr - .5) / 8
    # Explicit pairwise probabilities against each of the eight other proposals.
    pairwise = ((q[:, :, None] > q[:, None, :]).sum(2)
                + .5 * ((q[:, :, None] == q[:, None, :]).sum(2) - 1)) / 8
    assert np.array_equal(normalized, pairwise)
    assert np.min(normalized) >= 0 and np.max(normalized) <= 1
    score = rr.mean(0)
    choices = dict(raw_mean=q.mean(0).argmax(0), rank_mean=score.argmax(0),
                   rank_median=np.median(rr, 0).argmax(0))
    own = q.argmax(1)
    n = q.shape[-1]
    idx = np.arange(n)
    contrasts = {}
    for other in ('raw_mean', 'rank_median'):
        different = choices[other] != choices['rank_mean']
        delta = q[:, choices['rank_mean'], idx] - q[:, choices[other], idx]
        support = (delta > 0).sum(0)
        oppose = (delta < 0).sum(0)
        contrasts[other] = dict(
            choice_difference_fraction=float(different.mean()),
            rank_mean_preferred_by_two_among_differences=(
                float(np.mean((support[different] == 2) & (oppose[different] == 1))) if different.any() else None),
            rank_mean_preferred_by_one_among_differences=(
                float(np.mean((support[different] == 1) & (oppose[different] == 2))) if different.any() else None))
    scales = np.asarray([.1, 1., 10.])[:, None, None]
    offsets = np.asarray([-1000., 0., 1000.])[:, None, None]
    assert np.array_equal(ranks(q * scales + offsets), rr)
    # Positive monotone cubic transforms on centered values preserve each ordering.
    centered = q - q.mean(1, keepdims=True)
    assert np.array_equal(ranks(centered ** 3), rr)
    # Pareto domination cannot win a rank-mean election: >= for every member,
    # and > for at least one member, implies a strictly larger aggregate rank.
    dominates = (q[:, :, None] >= q[:, None, :]).all(0) & (q[:, :, None] > q[:, None, :]).any(0)
    winner_dominated = dominates[:, choices['rank_mean'], idx].any(0)
    assert not winner_dominated.any()
    sorted_scores = np.sort(score, axis=0)
    unique = (q == q.max(1, keepdims=True)).sum(1) == 1
    counts = (own[:, None] == np.arange(9)[None, :, None]).sum(0)
    majority_choice = counts.argmax(0)
    majority_exists = (((own == majority_choice[None]) & unique).sum(0) >= 2)
    return dict(query_states=n, contrasts=contrasts,
        rank_mean_top_tie_fraction=float(((score == score.max(0)).sum(0) > 1).mean()),
        rank_mean_score_margin_quantiles=dict(zip(('p10', 'p50', 'p90'),
            np.quantile(sorted_scores[-1] - sorted_scores[-2], [.1, .5, .9]).tolist())),
        rank_mean_winner_is_no_critic_top_fraction=float((counts[choices['rank_mean'], idx] == 0).mean()),
        unique_top_majority_exists_fraction=float(majority_exists.mean()),
        rank_mean_overrides_unique_top_majority_fraction=float((majority_exists & (choices['rank_mean'] != majority_choice)).mean()),
        rank_mean_overrides_unique_top_majority_when_exists=(
            float(np.mean(choices['rank_mean'][majority_exists] != majority_choice[majority_exists])) if majority_exists.any() else None),
        pairwise_win_probability_identity_exact=True,
        positive_affine_and_centered_cubic_rank_invariance_exact=True,
        no_selected_Pareto_dominated_proposal=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'analysis/independent_critic_calibration_v1'
    summary_path = source / 'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete' and summary['models_audited'] == 81
    factorial_path = root / 'analysis/independent_critic_factorial_v1/review_v1.json'
    factorial = json.loads(factorial_path.read_text())
    registration = root / 'independent_critic_rank_mean_confirmation_registration_v1.json'
    plan = json.loads(registration.read_text())
    assert plan['registered_before_fresh_policy_outcomes'] and plan['selected_rule'] == 'mean_rank'
    inputs = {str(p): sha(p) for p in (summary_path, factorial_path, registration)}
    rows = []
    files = sorted(source.glob('*_values.npz'))
    assert len(files) == 9
    # Named-file mapping uses the original pinned actor paths, never score-based ordering.
    discovery = json.loads((root / 'independent_critic_policy_discovery_registration_v1.json').read_text())
    for file in files:
        short = file.stem.removesuffix('_values')
        jobs = [j for j in discovery['jobs'] if Path(j['actor_run']).name.split('_')[1] == short]
        envs = {j['environment'] for j in jobs}
        assert len(envs) == 1
        environment = envs.pop()
        inputs[str(file)] = sha(file)
        with np.load(file, allow_pickle=False) as saved:
            for split in ('train', 'validation'):
                indices = saved[f'{split}_indices']
                for step in (800000, 900000, 1000000):
                    original = next(r for r in summary['rows'] if r['environment'] == environment and r['step'] == step and r['split'] == split)
                    assert hashlib.sha256(indices.tobytes()).hexdigest() == original['index_sha256']
                    q = saved[f'{split}_{step}_Q']
                    assert q.shape == (3, 9, 512)
                    rows.append(dict(environment=environment, short=short, split=split, step=step, **describe(q)))
    assert len(rows) == 54
    raw = next(r for r in factorial['rows'] if r['rule'] == 'mean')
    ranked = next(r for r in factorial['rows'] if r['rule'] == 'mean_rank')
    # Same fixed order used by the complete factorial report, not sorted NPZ names.
    shorts = ['ant', 'hmmed', 'hmlarge', 'soccer', 'cube', 'cubed', 'pz3', 'pz4', 'scene']
    assert set(shorts) == {r['short'] for r in rows}
    per_task = []
    for i, short in enumerate(shorts):
        selected = [r for r in rows if r['short'] == short and r['split'] == 'validation']
        assert len(selected) == 3
        per_task.append(dict(short=short, raw_mean_last3=raw['values'][i], rank_mean_last3=ranked['values'][i],
            rank_mean_minus_raw=ranked['values'][i] - raw['values'][i],
            dataset_choice_difference_fraction=statistics.mean(r['contrasts']['raw_mean']['choice_difference_fraction'] for r in selected),
            rank_mean_top_tie_fraction=statistics.mean(r['rank_mean_top_tie_fraction'] for r in selected),
            rank_mean_unique_majority_override_fraction=statistics.mean(r['rank_mean_overrides_unique_top_majority_fraction'] for r in selected)))
    validation = [r for r in rows if r['split'] == 'validation']
    aggregate = dict(equal_task_step_choice_difference_fraction=statistics.mean(r['contrasts']['raw_mean']['choice_difference_fraction'] for r in validation),
        equal_task_step_rank_mean_vs_rank_median_difference_fraction=statistics.mean(r['contrasts']['rank_median']['choice_difference_fraction'] for r in validation),
        equal_task_step_rank_mean_top_tie_fraction=statistics.mean(r['rank_mean_top_tie_fraction'] for r in validation),
        equal_task_step_majority_override_fraction=statistics.mean(r['rank_mean_overrides_unique_top_majority_fraction'] for r in validation),
        tasks_higher=sum(r['rank_mean_minus_raw'] > 0 for r in per_task),
        tasks_lower=sum(r['rank_mean_minus_raw'] < 0 for r in per_task),
        tasks_equal=sum(r['rank_mean_minus_raw'] == 0 for r in per_task),
        nine_task_gain=ranked['nine_task_mean'] - raw['nine_task_mean'])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(9)
    gains = [r['rank_mean_minus_raw'] for r in per_task]
    axes[0].bar(x, gains, color=['#477f9f' if g >= 0 else '#ae694f' for g in gains])
    axes[0].axhline(0, color='#333333', lw=.7)
    axes[0].set(ylabel='Rank mean minus raw mean success', title='Same frozen critics, BC and N9 proposals')
    axes[1].bar(x, [r['dataset_choice_difference_fraction'] for r in per_task], color='#477f9f', label='Rank mean vs raw mean')
    axes[1].plot(x, [r['rank_mean_top_tie_fraction'] for r in per_task], 'o-', color='#ae694f', label='Rank-mean top tie')
    axes[1].set(ylabel='Fraction of validation queries', ylim=(0, .65), title='Fixed dataset queries; no oracle action ranks')
    for axis in axes:
        axis.set_xticks(x, ['AM-L', 'HM-M', 'HM-L', 'AS-A', 'Cube-S', 'Cube-D', 'Pz-3', 'Pz-4', 'Scene'], rotation=30, ha='right')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Locked rank-mean discovery mechanism: one ensemble group, learners0/1/2', fontsize=11)
    fig.text(.5, .025, 'Left: equal mean of800K /900K /1M CPU50. Right: the same512 validation observations reused at each checkpoint.\nRank transforms control critic scale influence; they do not certify correct action rankings or establish stability across training groups.\nNo fresh model outcomes, per-task tuning, additional training, or new policy evaluation enters this analysis.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, .94))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'mechanism_v1.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    with (out / 'per_task_v1.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader()
        writer.writerows(per_task)
    result = dict(time=time.time(), status='complete', preferred_rule_unchanged='mean_rank',
        source_sha256=sha(__file__), inputs_sha256=inputs, rows=rows, per_task=per_task, validation_aggregate=aggregate,
        unit_scope='Actual archived Q tables: exact pairwise identity, positive affine/cubic invariance, Pareto exclusion.',
        limits=['One already-seen discovery ensemble group; not fresh policy or training replication.',
            'Dataset queries, not deployment visitation or independently sampled state replications across checkpoints.',
            'No correct within-state action ranks or true residual/propagation errors are available.',
            'Rank mean can override a two-critic unique-top majority; no median-majority guarantee.',
            'The relation between fixed-query disagreement and return changes is descriptive, not causal correctness.'],
        goal_complete=False)
    (out / 'review_v1.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
