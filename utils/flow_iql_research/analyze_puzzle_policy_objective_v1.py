"""Audit fixed-policy Puzzle traces and separate dense return from completion.

This reads three sealed, exactly reproduced CPU50 results. Bootstrap ranges
describe variation across the 50 fixed reset seeds, not training-seed evidence.
No learned parameters, rewards, registrations, or primary scores are changed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    area = root / 'analysis/puzzle_policy_objective_diagnosis_v1'
    target = area / 'THREE_POLICY_RETURN_TRACE_ROOT_AUDIT_v1.json'
    assert not target.exists()
    plan_path = area / 'prospective_return_trace_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    base_path = Path(plan['base_array_equality'])
    assert sha(base_path) == plan['base_array_equality_sha256']
    base = json.loads(base_path.read_text())
    assert base['status'] == 'pass' and len(base['rows']) == 6
    pinned_models = {}
    for row in base['rows']:
        assert row['all_base_arrays_bitwise_equal'] and row['base_array_leaves'] == 236
        for side in ['left', 'right']:
            pinned_models[row[side]] = row[side + '_sha256']
    for path, digest in pinned_models.items():
        assert sha(path) == digest
    cases = ['teacher', 'q_actor', 'center']
    rows = {}
    traces = {}
    inputs = {}
    for case in cases:
        path = area / (case + '_return_trace_v1.json')
        d = json.loads(path.read_text())
        assert d['case'] == case and d['step'] == 1000000 and d['training_seed'] == 0
        assert d['gamma'] == .999 and d['n_episodes'] == 50 and d['first_episode_seed'] == 42
        assert all(d['exact_reproduction_of_sealed_episode_arrays'].values())
        assert d['optimizer_updates_performed'] == 0
        assert not d['training_rewards_or_original_results_changed']
        assert sha(d['reference']) == d['reference_sha256']
        assert sha(d['canonical_instrumented_point']) == d['canonical_point_sha256']
        for source, digest in d['source_sha256'].items():
            assert sha(source) == digest
        reference = json.loads(Path(d['reference']).read_text())
        assert reference['n_episodes'] == 50 and reference['eval_device'] == 'cpu'
        assert reference['first_episode_seed'] == 42
        episodes = d['episodes']
        assert len(episodes) == 50
        for index, ep in enumerate(episodes):
            assert ep['seed'] == 42 + index
            assert ep['success'] == reference['episode_successes'][index]
            assert ep['length'] == reference['episode_lengths'][index]
            assert sum(ep['rewards']) == reference['episode_rewards'][index]
            assert ep['length'] <= 500 and len(ep['rewards']) == ep['length']
            assert np.isfinite(ep['rewards']).all()
            weights = .999 ** np.arange(ep['length'])
            assert np.isclose(weights @ ep['rewards'], ep['discounted_dense_return'])
            # OGBench default success_timing='pre': the reward precedes the
            # action, whereas our remaining-buttons trace is after the action.
            assert ep['rewards'][0] == -5
            assert np.array_equal(-np.asarray(ep['incorrect_buttons'][:-1]), ep['rewards'][1:])
            if ep['success']:
                assert ep['last_reward'] == ep['last_incorrect_buttons'] == 0
            else:
                assert ep['length'] == 500 and ep['truncated']
        if case != 'teacher':
            assert [e['initial_observation'] for e in episodes] == [
                e['initial_observation'] for e in traces['teacher']['episodes']]
        failed = [e for e in episodes if not e['success']]
        rows[case] = dict(successes=int(sum(e['success'] for e in episodes)), episodes=50,
            success_rate=d['success_rate'], dense_return=d['undiscounted_dense_return_mean'],
            discounted_dense_return=d['discounted_dense_return_mean'],
            discounted_completion_time_surrogate=d['discounted_completion_time_surrogate_mean'],
            length_mean=float(np.mean([e['length'] for e in episodes])),
            successful_first_success_step_mean=d['success_first_step_mean'],
            failed_final_incorrect_button_histogram=dict(Counter(str(e['last_incorrect_buttons']) for e in failed)),
            pre_action_reward_matches_previous_post_action_buttons_exactly=True,
            reset_observations_equal_to_teacher=True)
        traces[case] = d
        inputs[str(path)] = sha(path)
    rng = np.random.default_rng(90731)
    indices = rng.integers(0, 50, (10000, 50))
    comparisons = {}
    for case in ['q_actor', 'center']:
        comparison = {}
        for key in ['success', 'discounted_dense_return', 'discounted_completion_time_surrogate']:
            delta = np.asarray([e[key] for e in traces[case]['episodes']]) - np.asarray([
                e[key] for e in traces['teacher']['episodes']])
            boot = delta[indices].mean(axis=1)
            comparison[key] = dict(mean_difference=float(delta.mean()),
                paired_reset_bootstrap_95_interval=np.quantile(boot, [.025, .975]).tolist(),
                positive_pairs=int((delta > 0).sum()), negative_pairs=int((delta < 0).sum()))
        comparisons[case + '_minus_teacher'] = comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = dict(teacher='#353535', q_actor='#c44e52', center='#2878b5')
    labels = dict(teacher='Finite-mean teacher (best9)', q_actor='Q-guided one-step (alpha=1)',
        center='Mixture-location one-step')
    x = np.arange(1, 501)
    for case in cases:
        episodes = traces[case]['episodes']
        solved = np.asarray([[e['first_success_step'] is not None and t >= e['first_success_step']
            for t in x] for e in episodes], float)
        remaining = np.zeros((50, 500))
        for index, ep in enumerate(episodes):
            remaining[index, :ep['length']] = ep['incorrect_buttons']
        axes[0, 0].plot(x, solved.mean(axis=0), color=colors[case], label=labels[case])
        axes[0, 1].plot(x, remaining.mean(axis=0), color=colors[case])
        axes[1, 0].scatter([cases.index(case)] * 50,
            [e['discounted_dense_return'] for e in episodes], color=colors[case], alpha=.45, s=12)
        axes[1, 0].scatter([cases.index(case)], [rows[case]['discounted_dense_return']],
            color=colors[case], edgecolor='black', s=70, marker='D', zorder=3)
    axes[0, 0].set(title='Task completion by elapsed step', xlabel='Environment step',
        ylabel='Fraction of 50 episodes completed', ylim=(0, 1.03))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Incomplete buttons (absorbing zero after success)',
        xlabel='Environment step', ylabel='Mean number of incorrect buttons', ylim=(0, 6))
    axes[1, 0].set(title='Realized discounted dense return (higher is better)', ylabel='Return')
    axes[1, 0].set_xticks(range(3), ['Teacher', 'Q-guided', 'Mixture location'])
    q = traces['q_actor']['episodes']
    teacher = traces['teacher']['episodes']
    axes[1, 1].scatter([e['discounted_dense_return'] for e in teacher],
        [e['discounted_dense_return'] for e in q],
        c=[colors['q_actor'] if not e['success'] else '#77aa77' for e in q], alpha=.75)
    lim = [-2200, 0]
    axes[1, 1].plot(lim, lim, '--', color='gray', lw=1)
    axes[1, 1].set(title='Paired reset seeds: Q-guided vs teacher',
        xlabel='Teacher discounted dense return', ylabel='Q-guided discounted dense return')
    fig.suptitle('Puzzle 3x3, fixed 1M models, seed0, CPU50 (resets42–91)\n'
        'Red points: Q-guided failure. No independent training-seed confirmation.', fontsize=11)
    for suffix in ['png', 'pdf']:
        path = area / ('dense_return_vs_completion_v1.' + suffix)
        assert not path.exists()
        fig.savefig(path, dpi=170)
    plt.close(fig)
    reward_sources = [Path('/home/ukjo1/anaconda3/envs/ogbench/lib/python3.10/site-packages/ogbench') / name
        for name in ['manipspace/envs/puzzle_env.py', 'manipspace/envs/manipspace_env.py', 'relabel_utils.py']]
    result = dict(time=time.time(), status='pass', source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        input_sha256=inputs, prospective_plan_sha256=sha(plan_path), actual_models_rehashed=pinned_models,
        base_equality_audit_sha256=sha(base_path), task='puzzle-3x3 default', training_seed=0,
        step=1000000, instrumented_episodes=150, new_independent_training_replicates=0,
        all_three_canonical_episode_arrays_exact=True, same50_initial_observations=True,
        environment_source_sha256={str(p): sha(p) for p in reward_sources},
        reward_timing='pre: r_t = -incorrect(s_t), traced button count is at s_(t+1)',
        all150_previous_post_state_counts_exact_to_next_reward=True,
        caution_on_raw_trace_flag='reward_shape_verified_all_episodes compares reward to post-action buttons; false is the expected one-step timing difference, not reward corruption.',
        rows=rows, comparisons=comparisons,
        bootstrap_scope='10,000 paired resamples of fixed50 reset seeds at one checkpoint; not training-seed uncertainty.',
        interpretation='This fixed sample ranks Q-guided higher in mean discounted dense return but lower in all-buttons completion. 20/21 Q-guided failures end with exactly one incorrect button. This supports objective/criterion mismatch as a hypothesis, alongside policy approximation/coverage error; it does not prove critic overestimation, root cause, or expected-return dominance.',
        limitations=['Only one task, training seed and checkpoint.',
            '500-step truncated return is not infinite-horizon return; no continuation after failed episodes.',
            'Implicit expectile Q is not Q of the deployed policy. Replay-state predicted Q cannot be subtracted from these initial-state returns as a ground-truth bias estimate.',
            'Q-guided and mixture-location actors differ in objective AND architecture; this is not an isolated Q-gradient ablation.',
            'No production recipe, evaluation score, reward, learned model, or queue mutation.'])
    write_new(target, result)
    print(json.dumps(dict(status='pass', rows=rows, comparisons=comparisons)))


if __name__ == '__main__':
    main()
