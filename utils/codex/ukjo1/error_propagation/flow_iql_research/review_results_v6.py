"""Refresh the research record from complete raw last-three evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'analysis'
    inputs = {}

    def read(path):
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def episodes(record):
        assert record['n_episodes'] == 50 and record['first_episode_seed'] == 42
        assert record['eval_device'] == 'cpu'
        assert record['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
        for key in ('episode_successes', 'episode_rewards', 'episode_lengths'):
            assert len(record[key]) == 50 and np.isfinite(record[key]).all()
        assert abs(np.mean(record['episode_successes']) - record['success_rate']) < 1e-12

    audit = read(out / 'live_audit.json')
    cohort = read(out / 'confirmation_three_seed_audit_v1.json')
    envs = [row['environment'] for row in cohort['seeds'][0]['rows']]
    steps = [800000, 900000, 1000000]
    short = ['AM-L', 'HM-M', 'HM-L', 'AS-A', 'Cube-S', 'Cube-D', 'Pz-3', 'Pz-4', 'Scene']
    methods = []

    def primary(label, kind, profile, discount, samples=None):
        rows = [row for row in audit['rows'] if row['seed'] == 0 and
                row['group']['kind'] == kind and row['group']['profile'] == profile and
                row['group']['discount'] == discount and
                (kind in ('fql', 'rebrac') or row['group']['expectile'] == .9) and
                (samples is None or row['group'].get('num_samples') == samples)]
        values, evidence = {}, {}
        for row in rows:
            if not row['valid_primary']:
                continue
            path = root / 'runs' / row['run'] / 'evaluations.jsonl'
            raw = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines()]
            scores = []
            for step in steps:
                found = [record for record in records if record['step'] == step]
                assert len(found) == 1
                episodes(found[0])
                scores.append(found[0]['success_rate'])
            assert row['environment'] not in values
            values[row['environment']] = float(np.mean(scores))
            evidence[row['environment']] = dict(run=row['run'], per_step=dict(zip(steps, scores)))
        result = dict(label=label, training_seeds=[0],
                      values=[values.get(env) for env in envs], evidence=evidence)
        methods.append(result)
        return result

    primary('Gaussian IQL, original gamma (s0)', 'iql', 'official', 'original')
    primary('Gaussian IQL V2, gamma .999 (s0)', 'iql', 'bounded_v2', .999)
    primary('Native IFQL, best32 (s0)', 'ifql', 'official', .999, 32)
    primary('V2 IFQL, best32 (s0)', 'ifql', 'bounded_v2', .999, 32)
    n9 = read(out / 'derived_budget_full9_v1/variant_last3_audit.json')
    methods.append(dict(label='V2 IFQL, best9 (s0)', training_seeds=[0],
                        values=[next(row['last3_mean'] for row in n9['rows'] if row['environment'] == env) for env in envs]))
    for seed in cohort['seeds']:
        assert seed['completed_tasks'] == 9 and not seed['errors']
        methods.append(dict(label=f'V2 IFQL, best9 + K3 (s{seed["training_seed"]})',
                            training_seeds=[seed['training_seed']],
                            values=[next(row['last3_mean'] for row in seed['rows'] if row['environment'] == env) for env in envs]))
    methods.append(dict(label='V2 IFQL, best9 + K3 (3-seed mean)', training_seeds=[0, 1, 2],
                        values=[float(np.mean([next(row['last3_mean'] for row in seed['rows'] if row['environment'] == env) for seed in cohort['seeds']])) for env in envs]))
    primary('V2 IQL critic + FQL policy (s0)', 'fql_policy', 'bounded_v2', .999)
    primary('FQL, fresh reference (s0)', 'fql', 'official', .999)
    primary('ReBRAC, fresh reference (s0)', 'rebrac', 'official', .999)
    empirical = primary('Finite expectile backup, best9 (s0)', 'ifql', 'bounded_v2_empirical_backup', .999, 9)
    hinge = primary('V2 + value-hinge policy (s0)', 'fql_policy', 'bounded_v2_value_hinge', .999)
    for method in methods:
        complete = all(value is not None for value in method['values'])
        method.update(completed_tasks=sum(value is not None for value in method['values']),
                      nine_task_mean=float(np.mean(method['values'])) if complete else None)

    common = []
    base_n9 = dict(zip(envs, methods[4]['values']))
    base_k3 = {row['environment']:row['last3_mean'] for row in cohort['seeds'][0]['rows']}
    folder = out / 'native_value_common_BC_late_v1'
    for name, env in [('ant', envs[0]), ('hmmed', envs[1]), ('pz3', envs[6])]:
        for members in [1, 3]:
            records = [read(folder / f'{name}_{step}_nativeQ_sameBC_K{members}.json') for step in steps]
            for record in records:
                episodes(record)
                assert record['actor_training_step'] == record['step']
                expected = [record['step']] if members == 1 else [record['step']-200000, record['step']-100000, record['step']]
                assert [member['step'] for member in record['critic_members']] == expected
            native = float(np.mean([record['success_rate'] for record in records]))
            corrected = (base_n9 if members == 1 else base_k3)[env]
            common.append(dict(environment=env, members=members, native_Q_same_actual_BC_last3=native,
                               V2_Q_same_actual_BC_last3=corrected, V2_minus_native=corrected-native,
                               native_per_step={record['step']:record['success_rate'] for record in records}))
    for members in [1, 3]:
        reference = read(folder / f'self_pz3_1000000_K{members}.json')
        assert all(reference['self_reference_episode_arrays_exact'].values())
    common_record = dict(time=time.time(), rows=common, self_references_all50_arrays_exact=True,
                         interpretation='The critic-controller effect with actual BC fixed differs across tasks. It strongly helps Pz3, harms HMM here, and is not a nine-task causal average. K3 can interact with the value controller.',
                         limits=['Critic-source interventions isolate deployed parameters, not true Q errors or individual V2 components.',
                                 'No mixed optimizer state was trained or saved; these are not extra training seeds.'])
    (out / 'common_BC_value_controller_review_v2.json').write_text(json.dumps(common_record, indent=2))

    backup_diagnostics = []
    for env, evidence in empirical['evidence'].items():
        path = root / 'runs' / evidence['run'] / 'diagnostics.jsonl'
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        records = [json.loads(line) for line in raw.splitlines()]
        assert all(record['finite'] for record in records)
        last = records[-1]
        backup_diagnostics.append(dict(environment=env, step=last['step'], q_rms=last['q_rms'],
            q_rms_over_return_bound=last['q_rms_over_return_bound'],
            finite_proposal_score_abs_mean=last['backup/finite_proposal_score_abs_mean'],
            empirical_root_minus_learned_V_rms=last['backup/empirical_root_minus_learned_V_rms'],
            last_backup_clip_fraction=last['backup/clip_fraction'], all_observed_probes_finite=True))

    with (out / 'common9_comparison_v2.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', *short, 'nine_task_mean', 'complete_tasks', 'training_seeds'])
        for method in methods:
            writer.writerow([method['label'], *[value if value is not None else 'pending' for value in method['values']],
                             method['nine_task_mean'] if method['nine_task_mean'] is not None else 'pending',
                             method['completed_tasks'], ','.join(map(str, method['training_seeds']))])
    array = np.asarray([[np.nan if value is None else value for value in method['values']+[method['nine_task_mean']]] for method in methods])
    fig, ax = plt.subplots(figsize=(15, 8.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#e5e7eb')
    im = ax.imshow(np.ma.masked_invalid(array), vmin=0, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(range(10)); ax.set_xticklabels(short+['Avg'])
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([method['label'] for method in methods])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=8)
    for i, method in enumerate(methods):
        for j, value in enumerate(array[i]):
            label = 'pending' if np.isnan(value) else f'{value:.4f}' if j == 9 else f'{value:.2f}'
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='#555555' if np.isnan(value) else 'black' if value > .55 else 'white')
    ax.axvline(8.5, color='white', lw=2)
    ax.add_patch(plt.Rectangle((-.5, 7.5), 10, 1, fill=False, edgecolor='#ef9c32', lw=2))
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label='CPU50 success')
    fig.suptitle('Exact 800K / 900K / 1M mean: independent confirmation missed the 0.47 goal', fontsize=13, y=.995)
    fig.text(.5, .035, 'All rows except original IQL use gamma=0.999; expectile=0.9 where applicable. Pending tasks are never zero-filled.\nSeed0 selected K3; seeds1/2 were locked before their results. The three-seed mean is 0.4523 (seed-level sample SD 0.0438).\nFresh competitors have one training seed; these rows do not establish statistical superiority. The supplied plain FQL threshold is 0.47.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .105, .985, .965))
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'common9_comparison_v2.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    review = dict(time=time.time(), status='active_goal_not_achieved', metric=dict(steps=steps, episodes=50,
                  first_episode_seed=42, task_mean='equal mean over nine complete task means', missing_policy='pending; never fill zero'),
                  methods=methods, independent_confirmation=cohort,
                  fixed_actual_BC_controller_effect=common_record, finite_backup_endpoint_diagnostics=backup_diagnostics,
                  logical_assessment=dict(value='Two fits, fresh backup and known-return projection form a coherent residual intervention, but heterogeneous causal effects do not support a universal gain claim.',
                  policy='Plain behavioral flow plus finite-Q action selection is a common global recipe; K3 addresses critic-ranking drift. Its independent confirmation missed the target.',
                  next='Finish the one common finite-expectile-backup recipe on all nine tasks; paired high-tau pilots are exploratory, never per-task winners.'),
                  limitations=['Temporal snapshots are correlated, not training replicates.',
                               'Finite-proposal residuals are not oracle conditional residuals.',
                               'Removing learned V from the backup leaves finite-sampling, generator and Q approximation errors.',
                               'The provided table includes DD-ReBRAC .63; exceeding plain FQL .47 would not establish SOTA.'],
                  inputs_sha256=inputs, goal_complete=False)
    for key, relative in [('cross_seed_critic_BC', 'Pz3_cross_seed_critic_BC_v1/summary.json'),
                          ('finite_backup_failure', 'finite_backup_failure_review_v1.json')]:
        path = out / relative
        if path.exists():
            review[key] = read(path)
    anchored_plan = root / 'anchored_finite_backup_pilot_plan_v1.json'
    if anchored_plan.exists():
        review['new_anchored_backup'] = read(anchored_plan)
    old_correction = out / 'previous_results_last3_correction.json'
    if old_correction.exists():
        old = read(old_correction)
        review['historical_metric_correction'] = dict(macros=old['macros'], caveats=old['caveats'],
            architecture_limit='Historical code and evaluation protocol differ; do not pool with the official-style campaign.')
    (out / 'method_review_v6.json').write_text(json.dumps(review, indent=2))
    print(json.dumps([dict(method=method['label'], tasks=method['completed_tasks'], mean=method['nine_task_mean']) for method in methods]))


if __name__ == '__main__':
    main()
