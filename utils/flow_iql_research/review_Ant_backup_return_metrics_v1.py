"""Check whether the paired Ant decline is only a success/reward tradeoff.

Reconstructs horizon-limited discounted return from the verified navigation
reward rule. It is not the infinite-horizon IQL implicit-policy Q function.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.root.resolve()
    comparison_path = root / 'analysis/min_mean_backup_pilot_parameters_complete_v1.json'
    comparison = json.loads(comparison_path.read_text())
    ant = next(r for r in comparison['rows'] if r['short'] == 'ant')
    assert ant['complete'] and ant['all_available_BC_parameters_exact']
    gamma = .999
    rows, inputs = [], {str(comparison_path): sha(comparison_path)}
    for label, prefix, profile in (
            ('min', 'eb1', 'bounded_v2_empirical_backup'),
            ('mean', 'meb1', 'bounded_v2_mean_empirical_backup')):
        run = root / 'runs' / f'{prefix}_ant_{profile}_g999_t90_s0'
        conf = json.loads((run / 'config.json').read_text())
        assert conf['identity']['config']['discount'] == gamma
        path = run / 'evaluations.jsonl'
        inputs[str(path)] = sha(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for step in (800000, 900000, 1000000):
            record = next(r for r in records if r['step'] == step)
            assert record['n_episodes'] == 50 and record['first_episode_seed'] == 42
            assert record['eval_device'] == 'cpu'
            success = np.asarray(record['episode_successes'], np.float64)
            length = np.asarray(record['episode_lengths'], np.float64)
            reward = np.asarray(record['episode_rewards'], np.float64)
            assert len(success) == len(length) == len(reward) == 50
            assert np.array_equal(reward, -(length - success))
            # Reward is -1 before the first goal, then one terminal0 on success.
            discounted = -(1 - np.power(gamma, length - success)) / (1 - gamma)
            unknown_tail = np.where(success == 1, 0., np.power(gamma, length) / (1 - gamma))
            rows.append(dict(controller=label, step=step, success=float(success.mean()),
                mean_undiscounted_episode_reward=float(reward.mean()),
                mean_episode_length=float(length.mean()),
                conditional_success_length_mean=float(length[success == 1].mean()) if success.any() else None,
                horizon_limited_discounted_return_mean=float(discounted.mean()),
                conservative_continuation_lower_mean=float((discounted - unknown_tail).mean()),
                conservative_continuation_upper_mean=float(discounted.mean()),
                all50_observed_reward_length_success_identities_exact=True))
    summaries = {}
    for label in ('min', 'mean'):
        group = [r for r in rows if r['controller'] == label]
        summaries[label] = {key: statistics.mean(r[key] for r in group) for key in (
            'success', 'mean_undiscounted_episode_reward', 'horizon_limited_discounted_return_mean',
            'conservative_continuation_lower_mean', 'conservative_continuation_upper_mean')}
    difference = {key: summaries['mean'][key] - summaries['min'][key] for key in summaries['min']}
    assert difference['success'] < 0 and difference['horizon_limited_discounted_return_mean'] < 0
    scope_source = Path('/home/ukjo1/anaconda3/envs/ogbench/lib/python3.10/site-packages/ogbench/locomaze/maze.py')
    result = dict(time=time.time(), status='complete', gamma=gamma, rows=rows, last3=summaries,
        mean_minus_min_last3=difference, source_sha256=sha(__file__),
        observed_navigation_reward_source=str(scope_source), reward_source_sha256=sha(scope_source),
        inputs_sha256=inputs, primary_metric_unchanged='800K/900K/1Msuccessmean',
        interpretation='The observed Ant last3 decline is present in both success and horizon-limited reward/discounted return; these records do not support a simple observed-metric tradeoff explanation.',
        limits=['One environment/seed and a paired controller intervention; no new training or policy evaluation.',
            'Reconstruction relies on the verified -1-until-goal/terminal0 reward rule; it is not valid for dense multi-object manipulation rewards.',
            'Unobserved continuation after unsuccessful episode end is only bounded. Intervals are deterministic reward-range bounds, not statistical confidence intervals.',
            'The implicit expectile target policy differs from deployed best9. These reconstructed returns do not identify true IQL Q errors, conditional residuals, or B_k.'],
        nine_task_mean=None, goal_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(last3=summaries, mean_minus_min=difference)), flush=True)


if __name__ == '__main__':
    main()
