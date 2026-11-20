"""Audit and summarize frozen component interventions, not training replicates."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, sha


def interval(a, b, indices):
    difference = np.asarray(a, float) - np.asarray(b, float)
    distribution = difference[indices].mean(1) * 100
    return dict(difference_pp=float(100 * difference.mean()),
        paired_reset_bootstrap_95_interval_pp=np.quantile(distribution, [.025, .975]).tolist(),
        A_only_success=int(np.sum(difference > 0)), B_only_success=int(np.sum(difference < 0)))


def main(args):
    root = args.plan.parent
    plan = json.loads(args.plan.read_text())
    source_pins = {}
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
        source_pins[path] = digest
    for source in plan['sources'].values():
        assert sha(source['checkpoint']) == source['checkpoint_sha256']
        source_pins[source['checkpoint']] = source['checkpoint_sha256']
    results, evidence, rows = {}, {}, []
    for job in plan['jobs']:
        path = root / 'evaluations' / f'{job["id"]}.json'
        data = json.loads(path.read_text())
        assert data['job'] == job and data['optimizer_updates'] == 0
        assert data['n_episodes'] == len(data['episode_successes']) == 50
        assert abs(np.mean(data['episode_successes']) - data['success_rate']) < 1e-12
        assert not data['mixed_training_state_saved']
        assert data['source_sha256'] == plan['source_sha256']
        assert sha(path.with_suffix('.npz')) == data['trajectory_arrays_sha256']
        if job['self_control']:
            assert all(data['self_episode_arrays_exact'].values())
        evidence[str(path)] = sha(path)
        evidence[str(path.with_suffix('.npz'))] = sha(path.with_suffix('.npz'))
        results[job['q'], job['flow']] = data
        rows.append(dict(job=job['id'], Q_source=job['q'], Flow_source=job['flow'],
            self_control=job['self_control'], success_percent=100 * data['success_rate'],
            mean_episode_return=data['reward_mean'], episodes=50))
    assert len(results) == 19
    with (root / 'crossover_cells_v1.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranking = {}
    for scope in ('replay', 'visitation'):
        path = root / f'rankings_{scope}_v1/summary_v1.json'
        data = json.loads(path.read_text())
        assert len(data['results']) == 5
        evidence[str(path)] = sha(path)
        ranking[scope] = {r['pair']['id']: r for r in data['results']}
        for record in data['results']:
            path = root / f'rankings_{scope}_v1/{record["pair"]["id"]}.npz'
            assert sha(path) == record['arrays_sha256']
            evidence[str(path)] = sha(path)
    indices = np.random.default_rng(20261118).integers(0, 50, (20000, 50))
    pairs, metric_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    titles = ('HM-M N18 seed1: 800K / 1M', 'HM-M N18 seed2: 800K / 1M',
              'HM-M N36: seed1 / seed2', 'Puzzle4 N27: seed0 / seed1',
              'Puzzle4 N27: seed0 / seed2')
    for i, pair in enumerate(plan['pairs']):
        a, b = pair['A'], pair['B']
        cells = {q + f: results[qn, fn] for q, qn in (('A', a), ('B', b))
                 for f, fn in (('A', a), ('B', b))}
        successes = {k: v['episode_successes'] for k, v in cells.items()}
        row = dict(pair=pair, success_percent={k: 100 * v['success_rate'] for k, v in cells.items()},
            Q_A_minus_B_with_Flow_A=interval(successes['AA'], successes['BA'], indices),
            Q_A_minus_B_with_Flow_B=interval(successes['AB'], successes['BB'], indices),
            Flow_A_minus_B_with_Q_A=interval(successes['AA'], successes['AB'], indices),
            Flow_A_minus_B_with_Q_B=interval(successes['BA'], successes['BB'], indices))
        row['interaction_pp'] = (row['success_percent']['AA'] - row['success_percent']['AB']
                                 - row['success_percent']['BA'] + row['success_percent']['BB'])
        row['rankings'] = {scope: ranking[scope][pair['id']] for scope in ranking}
        pairs.append(row)
        metric_rows.append(dict(pair=pair['id'], **row['success_percent'],
            **{k: row[k]['difference_pp'] for k in ('Q_A_minus_B_with_Flow_A',
                'Q_A_minus_B_with_Flow_B', 'Flow_A_minus_B_with_Q_A', 'Flow_A_minus_B_with_Q_B')},
            interaction_pp=row['interaction_pp']))
        ax = axes.flat[i]
        values = np.array([[row['success_percent'][q + f] for f in ('A', 'B')] for q in ('A', 'B')])
        artist = ax.imshow(values, vmin=0, vmax=100, cmap='viridis')
        ax.set_xticks([0, 1], ['Flow A', 'Flow B'])
        ax.set_yticks([0, 1], ['Q A', 'Q B'])
        ax.set_title(titles[i], fontsize=10)
        for y in range(2):
            for x in range(2):
                ax.text(x, y, f'{values[y, x]:.0f}%', ha='center', va='center',
                        fontsize=19, color='white' if values[y, x] < 55 else 'black')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .8, 'A: originally stronger checkpoint\nB: originally weaker checkpoint\n\n'
        '50 fixed reset seeds per cell\n9 self controls exactly reproduce\noriginal episode arrays\n\n'
        'No training or optimizer updates\nPost-hoc component diagnostics', va='top', fontsize=10)
    fig.suptitle('Frozen Q / BC-Flow component crossover: success rate', fontsize=13)
    fig.tight_layout(rect=(0, .02, 1, .95))
    for extension in ('png', 'pdf'):
        fig.savefig(root / ('crossover_success_matrices_v1.' + extension), dpi=190)
    plt.close(fig)
    with (root / 'crossover_effects_v1.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    source_pins[str(Path(__file__))] = sha(__file__)
    excluded = root.with_name('component_crossover_20261118_v1')
    excluded_status = json.loads((excluded / 'controller_status_v1.json').read_text())
    assert excluded_status['phase'] == 'failed' and not excluded_status['running']
    excluded_finished = len(excluded_status['completed']) + len(excluded_status['failures'])
    atomic(root / 'analysis_final_v1.json', dict(status='complete', pairs=pairs,
        evaluations=19, episodes=950, exact_self_controls=9, new_training_runs=0,
        excluded_six_core_execution_attempt=dict(path=str(excluded),
            reason='Self controls did not reproduce original four-core evaluation episodes.',
            finished_rollout_cells=excluded_finished, rollout_episodes=50 * excluded_finished,
            included_in_component_effects=False),
        auxiliary_one_episode_reproduction_checks=4,
        bootstrap_scope='Paired resampling of the 50 fixed reset episodes only; '
            'not a confidence interval across unseen training seeds. Cases selected post hoc.',
        preserved_source_and_model_sha256=source_pins, raw_evidence_sha256=evidence))
    print(json.dumps(dict(status='complete', cells=rows, effects=metric_rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args())
