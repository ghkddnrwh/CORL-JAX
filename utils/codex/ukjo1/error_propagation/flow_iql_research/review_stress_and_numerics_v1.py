"""Review actual Gaussian stress runs and the separate BC arithmetic probe.

No training files, model parameters, or benchmark evaluations are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


STEPS = (800000, 900000, 1000000)
TASKS = ('ant', 'hmmed', 'hmlarge', 'soccer', 'cube', 'cubed', 'pz3', 'pz4', 'scene')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = {}

    def raw(path):
        content = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(content).hexdigest()
        return content

    def read(path):
        return json.loads(raw(path))

    def records(path):
        return [json.loads(line) for line in raw(path).splitlines()]

    def finite(value):
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        return value is None or bool(np.isfinite(np.asarray(value)).all())

    def audit(name, checkpoint_audit=False):
        run = root/'runs'/name
        config = read(run/'config.json')
        identity = config['identity']
        assert identity['kind'] == 'iql' and identity['seed'] == 0
        assert identity['config']['discount'] == .999 and identity['config']['expectile'] == .9
        assert identity['episodes'] == 50 and identity['steps'] == 1000000
        for source, digest in config['source_sha256'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
        evaluations = records(run/'evaluations.jsonl') if (run/'evaluations.jsonl').exists() else []
        scores = {}
        checkpoint_hashes = {}
        for step in STEPS:
            found = [item for item in evaluations if item['step'] == step]
            if not found:
                continue
            assert len(found) == 1
            item = found[0]
            assert item['eval_device'] == 'cpu' and item['n_episodes'] == 50
            assert item['first_episode_seed'] == 42
            assert item['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
            for field in ('episode_successes', 'episode_rewards', 'episode_lengths'):
                assert len(item[field]) == 50 and np.isfinite(item[field]).all()
            assert set(item['episode_successes']) <= {0., 1.}
            assert abs(statistics.mean(item['episode_successes'])-item['success_rate']) < 1e-12
            scores[step] = item['success_rate']
            if checkpoint_audit:
                path = run/f'checkpoint_{step}.pkl'
                content = raw(path)
                saved = pickle.loads(content)
                assert json.loads(json.dumps(saved['identity'])) == identity
                assert saved['source_sha256'] == config['source_sha256']
                assert saved['step'] == int(saved['state']['q_updates']) == step
                multiplier = 2 if 'v2' in identity['profile'] else 1
                assert int(saved['state']['v_updates']) == multiplier*step
                joint = int(saved['state']['agent']['network']['opt_state']['0']['count'])
                own_v = int(saved['state']['v_opt']['0']['count'])
                assert joint == step
                assert own_v == (0 if identity['profile'] == 'official' else multiplier*step)
                assert finite(saved['state'])
                checkpoint_hashes[step] = inputs[str(path)]
        complete = (run/'completed.json').exists()
        if complete:
            done = read(run/'completed.json')
            assert done['step'] == done['q_updates'] == 1000000
            assert len(scores) == 3
        probes = records(run/'diagnostics.jsonl')
        assert all(item['finite'] for item in probes)
        last = probes[-1]
        return dict(run=name, environment=identity['environment'], profile=identity['profile'],
                    complete=complete, per_step=scores,
                    last3_mean=statistics.mean(scores.values()) if complete else None,
                    checkpoints_sha256=checkpoint_hashes,
                    last_probe_step=last['step'], q_mean=last['q_mean'], q_RMS=last['q_rms'],
                    known_bound_Q_error_RMS_lower_bound=float(np.hypot(last['q_above_upper_rms'], last['q_below_lower_rms'])),
                    all_observed_probes_finite=True), probes

    gaussian = {}
    for profile in ('official', 'bounded_v2'):
        rows = []
        for task in TASKS:
            if profile == 'official':
                name = f'w1_{task}_official_g999_t90_s0' if task in ('cube', 'scene') else f'w3_{task}_official_g999_t90_s0'
            else:
                name = f'w1_{task}_bounded_v2_g999_t90_s0'
            row, _ = audit(name, checkpoint_audit=True)
            rows.append(row)
        complete = sum(row['complete'] for row in rows)
        gaussian[profile] = dict(rows=rows, completed_tasks=complete,
            nine_task_mean=statistics.mean(row['last3_mean'] for row in rows) if complete == 9 else None)

    paired = []
    for original, corrected in zip(gaussian['official']['rows'], gaussian['bounded_v2']['rows']):
        assert original['environment'] == corrected['environment']
        paired.append(dict(environment=original['environment'],
            V2_minus_official=corrected['last3_mean']-original['last3_mean']
            if original['complete'] and corrected['complete'] else None))
    paired_summary = dict(rows=paired,
        nine_task_delta=statistics.mean(row['V2_minus_official'] for row in paired)
            if all(row['V2_minus_official'] is not None for row in paired) else None,
        eight_task_delta_excluding_Ant=statistics.mean(row['V2_minus_official'] for row in paired[1:])
            if all(row['V2_minus_official'] is not None for row in paired[1:]) else None,
        scope='Posthoc leave-Ant-out descriptive sensitivity, not a new primary metric or training-seed significance test.')

    ant = []
    ant_probes = []
    frozen = []
    for name in ('w3_ant_official_g999_t90_s0', 'w1_ant_bounded_v2_g999_t90_s0'):
        row, probes = audit(name)
        ant.append(row)
        ant_probes.append(probes)
        for step in (300000, 800000, 1000000):
            path = root/'analysis/frozen_Q_V_diagnostics_v1'/f'{name}_{step}.json'
            fit = read(path)
            frozen.append(dict(run=name, step=step, initial_v_Adam_count=fit['initial_v_Adam_count'],
                               trace=[item for item in fit['trace'] if item['extra_frozen_v_steps'] in (0, 2, 8)]))

    numerical = read(root/'runs/aux_GPU_BC_precision_probe_v1/probe_result.json')
    assert numerical['kind'] == 'numerical_BC_arithmetic_probe_not_a_training_run'
    assert numerical['device'].startswith('gpu:') and numerical['memory_fraction'] == '0.1'
    numerical_summary = {precision: value['comparisons'] for precision, value in numerical['results'].items()}
    repeated = {precision: {name: data['repeat_parameter_max_abs'] for name, data in value['per_graph'].items()}
                for precision, value in numerical['results'].items()}
    assert all(value == 0 for group in repeated.values() for value in group.values())

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    colors = ('#d1495b', '#00798c')
    labels = ('Official IQL', 'IQL + two V fits / fresh backup / bounds')
    for probes, row, color, label in zip(ant_probes, ant, colors, labels):
        x = np.array([item['step'] for item in probes])/1000
        axes[0, 0].plot(x, [item['q_rms'] for item in probes], color=color, label=label)
        axes[0, 1].plot(x, [item['score_rms'] for item in probes], color=color)
        axes[1, 0].plot(x, [item['actor/mse'] for item in probes], color=color)
        axes[1, 1].plot(np.array(STEPS)/1000, [row['per_step'][step] for step in STEPS], '-o', color=color)
    axes[0, 0].axhline(1000, color='#777777', linestyle='--', linewidth=.8)
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    for axis, title, ylabel in zip(axes.flat,
            ('Value scale', 'Logged expectile score scale', 'Actor action reconstruction', 'Actual late CPU50 return'),
            ('Both-head Q RMS', 'Fixed-data score RMS (proxy)', 'Training-window actor MSE', 'Episode success fraction')):
        axis.set(title=title, xlabel='Critic updates (thousands)', ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[1, 1].set_ylim(-.03, 1.03)
    fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(.5, .055), frameon=False)
    fig.suptitle('Antmaze-large: Gaussian actors, expectile 0.9, discount 0.999, training seed 0')
    fig.text(.5, .018, '800K / 900K / 1M success: official 0.0000; corrected 0.7933. MSE improvement does not establish policy improvement.\nScore RMS is not an oracle conditional residual. This comparison tests the complete value intervention, not an individual component.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(root/'analysis'/f'Ant_Gaussian_stress_mechanism_v1.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    result = dict(time=time.time(), metric='Equal means of CPU50 at800K/900K/1M; no missing-task averaging.',
        Gaussian_gamma999=gaussian, paired_Gaussian_comparison=paired_summary,
        Ant_Gaussian_value_stress=ant, Ant_frozen_V_fits=frozen,
        GPU_BC_arithmetic=dict(comparisons=numerical_summary, identical_graph_repeats=repeated,
            interpretation='Same inputs produce reproducible cross-graph BC differences; requested highest matmul precision did not remove them in this single-process probe.',
            scope='32updatesfromonedonorwithfreshAdam;notanepisodeexperimentorhistoricalcausalproof.'),
        observations=[
            'Official Ant Q/V grows far outside the known nonpositive-return range while all state arrays remain finite.',
            'The nine-task Gaussian improvement is driven by Ant: the descriptive eight-task mean excluding Ant declines. This is not a uniform task-level gain.',
            'Actor loss and positive actor-weight mass persist; complete actor starvation is contradicted by these logs.',
            'Official Ant actor MSE improves while late success is zero; reconstruction loss alone misses policy failure.',
            'Same-minibatch frozen V loss can improve while independent-data loss rises. Extra fits do not guarantee conditional residual reduction.',
            'Separate native/V2 GPU runs cannot be assumed to have bitwise-identical BC parameters; fixed-actual-BC deployment interventions remain necessary.'],
        limits=[
            'The known-return interval gives a lower bound on true Q error only; a near-zero lower bound does not imply an accurate Q.',
            'Both critic heads on the fixed1024 dataset probe are not all deployed states or min-Q-only estimates.',
            'A single donor, fresh Adam, shared process and autotuning cache cannot establish the cause of historical million-step BC divergence.',
            'No actual CUDA kernel, reduced-precision fault or hardware nondeterminism is identified.',
            'This artifact contains one training seed and does not establish generalized superiority over FQL/ReBRAC.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs_sha256=inputs,
        goal_complete=False)
    target = root/'analysis/stress_and_numerics_review_v1.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)); temporary.replace(target)
    print(json.dumps(dict(Gaussian={key:dict(tasks=value['completed_tasks'],mean=value['nine_task_mean']) for key,value in gaussian.items()},
                         GPU_cross_graph_BC_parameter_max_abs_32_updates=numerical_summary['default']['native_vs_V2']['final_BC_parameter_max_abs'])))


if __name__ == '__main__':
    main()
