"""Recompute the final tables from raw episodes after the independent gate.

This read-only scientific review does not change training, evaluation records,
the registered algorithm, queue, or thread goal state.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


STEPS = (800000, 900000, 1000000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def raw_counts(run):
    assert (run/'completed.json').exists() and not (run/'stopped.json').exists()
    rows = [json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
    counts = []
    for step in STEPS:
        chosen = [row for row in rows if row['step'] == step]
        assert len(chosen) == 1
        row = chosen[0]
        assert row['n_episodes'] == 50 and row['first_episode_seed'] == 42 and row['eval_device'] == 'cpu'
        assert row['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
        assert all(len(row[field]) == 50 for field in ('episode_successes', 'episode_rewards', 'episode_lengths'))
        assert all(s in (0, 1) for s in row['episode_successes'])
        assert all(math.isfinite(float(x)) for field in ('episode_rewards', 'episode_lengths') for x in row[field])
        assert all(float(n).is_integer() and n > 0 for n in row['episode_lengths'])
        count = sum(int(s) for s in row['episode_successes'])
        assert abs(row['success_rate'] - count/50) < 1e-12
        counts.append(count)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root/'analysis/mean_backup_final_review_v1'
    out.mkdir(exist_ok=False)
    audit_path = root/'analysis/mean_backup_confirmation_final_reread_v1/audit.json'
    audit = read(audit_path)
    assert audit['status'] == 'complete' and audit['passes_original_independent_confirmation_gate'] is True
    assert audit['actual_finite_late_models_verified'] == 81 and audit['source_and_frozen_copy_contracts_verified'] == 27
    assert audit['discovery_model_hashes_rechecked'] == 27 and len(audit['seed_diversity']) == 9
    contract = root/'mean_backup_confirmation_root_audit_contract_v1.json'
    assert sha(contract) == audit['execution_contract_sha256']
    assert sha('utils/flow_iql_research/audit_mean_backup_confirmation_v1.py') == audit['audit_source_sha256']
    plan_path = root/'mean_backup_independent_confirmation_registration_v1.json'
    plan = read(plan_path)
    assert sha(plan_path) == read(contract)['primary_registration_sha256']
    discovery = read(root/'analysis/mean_backup_discovery_root_audit_v2.json')
    environments = [row['environment'] for row in discovery['rows']]
    assert len(environments) == len(set(environments)) == 9
    counts = {}
    run_pins = {}
    for row in discovery['rows'] + audit['rows']:
        run = root/'runs'/row['run']
        cfg = read(run/'config.json')
        seed = cfg['identity']['seed']
        key = (row['environment'], seed)
        assert key not in counts and seed in (0, 9, 10, 11)
        counts[key] = raw_counts(run)
        assert Fraction(sum(counts[key]), 150) == Fraction(str(row['last3_mean'])).limit_denominator(150)
        assert sha(run/'config.json') == row['config_sha256']
        eval_sha = row.get('evaluations_sha256', row.get('evaluation_sha256'))
        assert sha(run/'evaluations.jsonl') == eval_sha
        run_pins[run.name] = dict(config_sha256=sha(run/'config.json'), evaluations_sha256=eval_sha)
    assert len(counts) == 36
    seed_means = {seed: Fraction(sum(sum(counts[(env, seed)]) for env in environments), 1350) for seed in (0, 9, 10, 11)}
    fresh = sum((seed_means[s] for s in (9, 10, 11)), Fraction(0))/3
    all_four = sum(seed_means.values(), Fraction(0))/4
    assert fresh == Fraction(audit['fresh_exact_mean']) == Fraction(53, 90)
    assert all_four == Fraction(audit['all_four_exact_mean'])
    assert all(seed_means[s] > Fraction(47, 100) for s in (9, 10, 11))
    sd = statistics.stdev(float(seed_means[s]) for s in (9, 10, 11))
    tasks = []
    for env in environments:
        values = {str(seed): float(Fraction(sum(counts[(env, seed)]), 150)) for seed in (0, 9, 10, 11)}
        tasks.append(dict(environment=env, per_seed_last3=values,
            fresh_mean=statistics.mean(values[str(s)] for s in (9, 10, 11)),
            fresh_sample_SD=statistics.stdev(values[str(s)] for s in (9, 10, 11)),
            fresh_successes=sum(sum(counts[(env, s)]) for s in (9, 10, 11)),
            per_seed_per_step_success_counts={str(s): dict(zip(map(str, STEPS), counts[(env, s)])) for s in (0, 9, 10, 11)}))
    prior = read(root/'analysis/method_review_v30.json')
    labels = ['Gaussian IQL, original gamma (s0)', 'Gaussian IQL V2, original gamma (s0)',
        'Gaussian IQL, gamma .999 (s0)', 'Gaussian IQL V2, gamma .999 (s0)',
        'FQL, fresh reference (s0)', 'ReBRAC, fresh reference (s0)']
    baselines = []
    for label in labels:
        method = next(row for row in prior['methods'] if row['label'] == label)
        evidence = method['evidence']
        if isinstance(evidence, list):
            evidence = {row['environment']: row for row in evidence}
        entries = []
        for env in environments:
            rid = evidence[env]['run']
            raw = raw_counts(root/'runs'/rid)
            entries.append(dict(environment=env, run=rid, last3_mean=float(Fraction(sum(raw), 150)), success_counts=raw,
                evaluations_sha256=sha(root/'runs'/rid/'evaluations.jsonl')))
        macro = statistics.mean(e['last3_mean'] for e in entries)
        assert abs(macro-method['nine_task_mean']) < 1e-12
        baselines.append(dict(label=label, training_seeds=[0], nine_task_mean=macro, rows=entries))
    support = [audit_path, contract, plan_path, root/'analysis/mean_backup_discovery_root_audit_v2.json',
        root/'analysis/mean_min_backup_full9_components_v1.json', root/'analysis/common_BC_value_ablation_root_audit_v1.json',
        root/'analysis/native_IFQL_N9_reference_root_audit_v1.json', root/'analysis/mean_backup_online_mechanism_v1.json',
        root/'analysis/mean_backup_floor_timing_v1.json', root/'analysis/mean_value_teacher_pilots_v1/status.json',
        root/'analysis/tau_sampling_factorial_audit_v1/status.json', root/'analysis/tau_matched_extraction_v1/status.json',
        root/'analysis/fresh_confirmation_root_audit_v4/audit.json',
        root/'analysis/reproduction_environment_v1/official_vendor_translation.json',
        Path('algorithms/uk_offline/basic/propagation/criterion.txt').resolve()]
    summary = dict(time=time.time(), status='independent_confirmation_scientific_review_ready',
        metric='Equal50-episode800K/900K/1M means, equalnine tasks, equalthree fresh training seeds. Not peak/1M-only.',
        fresh_training_seeds=[9, 10, 11], seed_nine_task_means={str(s): float(v) for s, v in seed_means.items()},
        fresh_exact_mean=str(fresh), fresh_three_seed_nine_task_mean=float(fresh), fresh_seed_sample_SD=sd,
        discovery_nine_task_mean=float(seed_means[0]), all_four_seed_nine_task_mean=float(all_four),
        original_confirmation_gate_passed=True, all_three_fresh_seed_macros_above_plain_FQL=True,
        provided_plain_FQL=.47, provided_plain_ReBRAC=.44, provided_DD_ReBRAC=.63,
        tasks=tasks, independently_recomputed_raw_baselines=baselines, run_metadata_sha256=run_pins,
        evidence_sha256={str(p): sha(p) for p in support}, source_sha256=sha(__file__),
        actual_finite_fresh_late_models=81, prior_finite_discovery_models_rehashed=27,
        fresh_logged_probe_windows=27000, fresh_episode_outcomes=4050, fresh_training_replicates=3,
        equal_compute_claim=False, SOTA_claim=False, universal_stability_claim=False, goal_complete=False,
        limits=['Fresh seeds were registered before their outcomes; discovery seed0 is reported separately.',
            'Three training replicates, not4050independent replicates. SD is descriptive; no paired multi-seed FQL superiority test.',
            'Exact finite expectile concerns9flow samples, not the population behavior distribution or trueQ.',
            'Target projection bounds training targets; it does not guarantee globally bounded neural outputs.',
            'Unused2VAdam fits remain in the frozen implementation as research controls; no necessity/equal-compute claim.',
            'Mean/min finite comparison controls actualBC/moments/RNG. Learned/finite control BC differs, so its value-only attribution requires separate commonBC evidence.',
            'High tau=.99 success is not established for the selected method. Ant remains below providedFQL/ReBRAC.',
            'Optional compact serialization attempts do not replace or add primary observations.'])
    (out/'review.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (out/'task_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['environment', 'discovery_s0', 'fresh_s9', 'fresh_s10', 'fresh_s11', 'fresh_mean', 'fresh_sample_SD'])
        for row in tasks:
            writer.writerow([row['environment'], *[row['per_seed_last3'][str(s)] for s in (0, 9, 10, 11)], row['fresh_mean'], row['fresh_sample_SD']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names = ['Ant-L', 'Hum-M', 'Hum-L', 'Soccer', 'Cube-S', 'Cube-D', 'Pz-3', 'Pz-4', 'Scene']
    fig, (ax, macro) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[2.6, 1]))
    x = np.arange(9)
    ax.bar(x, [t['fresh_mean'] for t in tasks], yerr=[t['fresh_sample_SD'] for t in tasks], capsize=3, color='#247d8b', alpha=.8)
    for seed, offset, color in [(9, -.13, '#132f54'), (10, 0, '#b05027'), (11, .13, '#75528d')]:
        ax.scatter(x+offset, [t['per_seed_last3'][str(seed)] for t in tasks], label=f'Seed {seed}', color=color, s=20, zorder=3)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.08); ax.set_ylabel('Success, mean of 800K / 900K / 1M')
    ax.set_title('One shared recipe; three fresh training seeds'); ax.legend(ncol=3, loc='lower right', fontsize=8)
    macro.bar([0, 1, 2], [.44, .47, float(fresh)], color=['#aaaaaa', '#7f7f7f', '#247d8b'])
    macro.errorbar(2, float(fresh), yerr=sd, fmt='none', color='black', capsize=4)
    macro.scatter([1.9, 2., 2.1], [float(seed_means[s]) for s in (9, 10, 11)], color=['#132f54', '#b05027', '#75528d'], s=24, zorder=3)
    for i, value in enumerate([.44, .47, float(fresh)]):
        macro.text(i, value+.075, f'{value:.3f}', ha='center')
    macro.set_xticks([0, 1, 2], ['ReBRAC\nprovided', 'FQL\nprovided', 'Finite mean\nfresh3'])
    macro.set_ylim(0, .76); macro.set_title('Nine-task averages')
    fig.suptitle('Mean-head finite expectile backup with a BC flow', fontsize=13)
    fig.text(.5, .01, 'Error bars: sample SD across 3 training seeds. Provided reference bars have no reconstructed uncertainty. No equal-compute claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(out/'results.png', dpi=180); fig.savefig(out/'results.pdf'); plt.close(fig)
    print(json.dumps(dict(status=summary['status'], fresh_mean=float(fresh), SD=sd, output=str(out))), flush=True)


if __name__ == '__main__':
    main()
