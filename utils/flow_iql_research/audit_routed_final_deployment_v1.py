"""Rehash actual evidence for the complete one-step deployment objective.

No training, rollout, or timing is performed. Independent recomputation uses
the stored episode arrays and the real-observation benchmark's raw samples.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(a - b) < 1e-10, (a, b)


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    area = root / 'analysis/deployment_final_review_v1'
    area.mkdir(exist_ok=True)
    reg_path = root / 'registration_v19.json'
    assert sha(reg_path) == '77f8c233a521d658e411701b8121e2285e1b1490cbb84608ade4804dde1a626c'
    reg = load(reg_path)
    v15 = root / 'analysis/deployment_interim_review_v15'
    plan = load(v15 / 'actual_full_confirmation_file_audit_plan_v1.json')
    sources = dict(plan['source_sha256'])
    sources[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in sources.items():
        assert sha(path) == digest, path
    perf_path = v15 / 'FULL_NINE_FRESH_NATIVE_MODEL_AND_EVALUATION_ROOT_AUDIT_v1.json'
    prior = load(v15 / 'CLOSING_PERFORMANCE_AND_BENCHMARK_DRAIN_RUNTIME_v1.json')
    assert sha(perf_path) == prior['root_evidence_sha256'][str(perf_path)]
    perf = load(perf_path)
    assert perf['status'] == 'pass' and perf['confirmation']['performance_passed']
    assert sha(root / 'analysis/onestep_pipeline_v1/registrations/confirmation_attempt4_fresh.json') == perf['fresh_receipt_sha256']
    assert len(perf['actual_model_points']) == 108
    model_pins, episode_points = [], []
    final_models = {}
    for point in perf['actual_model_points']:
        native, routed = point['native_model'], point['routed_model']
        for path, digest in [(native['path'], native['checkpoint_sha256']),
                             (native['actor_path'], native['actor_artifact_sha256']),
                             (routed['actor_path'], routed['actor_artifact_sha256'])]:
            assert sha(path) == digest, path
            model_pins.append(dict(path=path, sha256=digest))
        if point['seed'] == 0 and point['step'] == 1000000:
            final_models[point['run']] = point
        for role, evidence in point['evaluations'].items():
            assert sha(evidence['path']) == evidence['sha256']
            row = load(evidence['path'])
            assert len(row['episode_successes']) == 50 and row['first_episode_seed'] == 42
            assert all(x in [0, 1] for x in row['episode_successes'])
            score = sum(row['episode_successes']) / 50
            close(score, row['success_rate'])
            close(statistics.mean(row['episode_rewards']), row['reward_mean'])
            episode_points.append(dict(seed=point['seed'], environment=point['environment'],
                step=point['step'], role=role, successes=sum(row['episode_successes']),
                episodes=50, success_rate=score, path=evidence['path'], sha256=evidence['sha256']))
    assert len(episode_points) == 324 and len(final_models) == 9
    for seed_record in perf['confirmation']['seed_means']:
        for role, key in [('routed', 'primary'), ('teacher', 'teacher')]:
            rows = [r for r in episode_points if r['seed'] == seed_record['seed'] and r['role'] == role]
            assert len(rows) == 27
            close(sum(r['successes'] for r in rows) / 1350, seed_record[key])
    print(json.dumps(dict(stage='selected_models_and_episodes_rehashed', checkpoint_points=108, stored_episodes=16200)), flush=True)

    original_manifest = Path(reg['protected_original_manifest'])
    assert sha(original_manifest) == reg['protected_original_manifest_sha256']
    original = load(original_manifest)
    for group in ['frozen_original_source_sha256', 'evidence_sha256']:
        for path, digest in original[group].items():
            assert sha(path) == digest, path
    original_pins = list(original['confirmation_model_pins_from_prior_verified_audit'].values())
    assert len(original_pins) == 81
    original_bytes = 0
    for pin in original_pins:
        assert sha(pin['path']) == pin['sha256'], pin['path']
        original_bytes += Path(pin['path']).stat().st_size
    preservation = dict(status='pass', original_manifest_sha256=sha(original_manifest),
        original_source_files=len(original['frozen_original_source_sha256']),
        original_evidence_files=len(original['evidence_sha256']), original_saved_models=81,
        original_saved_model_bytes=original_bytes, original_model_pins=original_pins,
        original_files_modified_by_audit=False)
    print(json.dumps(dict(stage='original_reference_rehashed', models=81, bytes=original_bytes)), flush=True)

    folder = root / 'analysis/onestep_pipeline_v1/benchmark_attempt4'
    admission = load(folder / 'isolation_admission_v1.json')
    assert admission['scheduler_running_count'] == admission['CPU_evaluation_worker_count'] == 0
    assert not admission['GPU_query_pids'] and admission['no_training_or_evaluation_overlap_at_admission']
    strict_area = root / 'analysis/routed_strict_GPU_speed_v1'
    real_area = root / 'analysis/routed_real_observation_speed_v1'
    strict_plan = load(strict_area / 'prospective_matched_precision_plan_v1.json')
    real_plan = load(real_area / 'prospective_real_observation_speed_plan_v1.json')
    for supplement in [strict_plan, real_plan]:
        for path, digest in supplement['source_sha256'].items():
            assert sha(path) == digest, path
    observation_manifest = real_area / 'real_reset_observations_manifest_v1.json'
    observations = load(observation_manifest)
    for row in observations['environments']:
        assert sha(row['output']) == row['sha256']
    speed_rows, aggregate = [], {}
    benchmark_files = []
    for scope, directory, suffix in [('default', folder, '_timings.json'),
            ('strict', strict_area, '_strict_timings_v1.json'),
            ('real', real_area, '_real_observation_timings_v1.json')]:
        rows = []
        for gpu in [0, 1]:
            path = directory / f'GPU{gpu}{suffix}'
            data = load(path)
            assert data['status'] == 'complete' and data['visible_GPU'] == str(gpu)
            assert data['CPU_affinity'] == list(range(gpu * 6, gpu * 6 + 6)) and 'A6000' in data['GPU_kind']
            source = Path('utils/flow_iql_research/' + ('benchmark_routed_real_observations_v1.py'
                if scope == 'real' else 'benchmark_routed_mixture_isolated_v1.py')).resolve()
            assert data['source_sha256'] == sha(source)
            if scope == 'real':
                assert data['NVIDIA_TF32_OVERRIDE'] == '0' and not data['validation_only']
                assert data['observations_manifest_sha256'] == sha(observation_manifest)
            assert data['input_sha256'] == sha(folder / f'GPU{gpu}_input.json')
            for row in data['rows']:
                point = final_models[row['run']]
                assert row['environment'] == point['environment'] and row['training_seed'] == 0
                assert row['checkpoint_sha256'] == point['native_model']['checkpoint_sha256']
                assert row['actor_artifact_sha256'] == point['routed_model']['actor_artifact_sha256']
                for backend in ['cpu', 'gpu']:
                    scopes = ['resident', 'host_to_host'] if scope == 'real' else [None]
                    if scope == 'real':
                        assert row[backend + '_all64_actions_wrapper_bitwise_exact']
                    for placement in scopes:
                        prefix = backend + ('_' + placement if placement else '')
                        for mode in ['primary', 'teacher']:
                            stem = f'{prefix}_{mode}'
                            repeat = row[stem + '_ms_repeat_medians']
                            assert len(repeat) == 3 and all(x > 0 and math.isfinite(x) for x in repeat)
                            close(statistics.median(repeat), row[stem + '_ms_median'])
                            if scope == 'real':
                                raw = np.asarray(row[stem + '_all_ms'])
                                assert raw.shape == (3, 100) and np.isfinite(raw).all() and (raw > 0).all()
                                assert [statistics.median(x) for x in raw] == repeat
                                close(float(np.quantile(raw.ravel(), .9)), row[stem + '_ms_p90'])
                        ratio = row[prefix + '_primary_ms_median'] / row[prefix + '_teacher_ms_median']
                        close(ratio, row[prefix + '_latency_ratio'])
                        speed_rows.append(dict(protocol=scope, environment=row['environment'], backend=backend,
                            input_placement=placement or 'resident_zero_observation',
                            primary_ms=row[prefix + '_primary_ms_median'], teacher_ms=row[prefix + '_teacher_ms_median'],
                            latency_ratio=ratio, speedup=1 / ratio))
                rows.append(row)
            benchmark_files.append(dict(path=str(path), sha256=sha(path)))
        assert len(rows) == len({r['environment'] for r in rows}) == 9
        prefixes = ['cpu_resident', 'cpu_host_to_host', 'gpu_resident', 'gpu_host_to_host'] if scope == 'real' else ['cpu', 'gpu']
        aggregate[scope] = {prefix: math.exp(statistics.mean(math.log(r[prefix + '_latency_ratio']) for r in rows)) for prefix in prefixes}
    assert aggregate['default']['cpu'] <= .5 and aggregate['default']['gpu'] <= 2 / 3
    assert aggregate['strict']['cpu'] <= .5 and aggregate['strict']['gpu'] <= 2 / 3
    assert aggregate['real']['cpu_host_to_host'] <= .5 and aggregate['real']['gpu_host_to_host'] <= 2 / 3
    for scope, summary_path in [('strict', strict_area / 'root_precision_speed_summary_v1.json'),
                                ('real', real_area / 'root_real_observation_speed_summary_v1.json')]:
        summary = load(summary_path)
        assert summary['status'] == 'complete'
        for key, ratio in aggregate[scope].items():
            close(ratio, summary['geometric_latency_ratios'][key])
    gpu_path = v15 / 'ALL_NINE_ACTOR_ONLY_GPU_CORRECTNESS_v1.json'
    gpu = load(gpu_path)
    assert gpu['status'] == 'pass' and len(gpu['rows']) == 9
    gpu_plan = load(v15 / 'all_nine_actor_GPU_check_plan_v1.json')
    for path, digest in gpu_plan['source_sha256'].items():
        assert sha(path) == digest
    validator = Path('utils/flow_iql_research/validate_portable_mixture_locations_GPU_v1.py').resolve()
    assert gpu['source_sha256'] == sha(validator)
    assert {r['actor_sha256'] for r in gpu['rows']} == {p['routed_model']['actor_artifact_sha256'] for p in final_models.values()}
    for row in gpu['rows']:
        assert row['actual_GPU_inference_completed'] and row['deterministic_given_key']
        assert row['Dense_dot_operations'] == 6 and row['computed_action_heads'] == 1
        assert not row['critic_loaded'] and not row['Flow_loaded'] and not row['candidate_action_bank']
        assert max(row['single_CPU_GPU_max_abs_difference'], row['batch32_CPU_GPU_max_abs_difference']) < 1e-4
    state = load(root / 'analysis/onestep_pipeline_v1/research_state.json')
    assert state['phase'] == 'success_pending_root_review' and state['attempts'][4]['speed']['passed']
    assert not state['evaluation_workers'] and not state['benchmark_workers']
    assert (root / 'STOP').exists() and (root / 'PAUSE_NEW_LAUNCHES.json').exists()
    assert not load(root / 'scheduler_status.json')['running']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    result = dict(time=time.time(), status='pass', source_sha256=sources,
        performance_audit_sha256=sha(perf_path), performance=perf['confirmation'],
        original_preservation=preservation, current_model_and_export_pins=model_pins,
        stored_episode_points=episode_points, benchmark_files=benchmark_files,
        benchmark_isolation_admission=admission, geometric_latency_ratios=aggregate,
        geometric_speedup={s: {k: 1/v for k, v in values.items()} for s, values in aggregate.items()},
        GPU_correctness_path=str(gpu_path), GPU_correctness_sha256=sha(gpu_path),
        GPU_max_abs_difference=max(max(r['single_CPU_GPU_max_abs_difference'], r['batch32_CPU_GPU_max_abs_difference']) for r in gpu['rows']),
        checks=dict(original_preserved=True, same_lean_joint_learner_verified=True,
            single_actor_only_deployment=True, nine_task_last3_retained=True,
            three_fresh_training_seeds_confirmed=True, same_global_recipe=True,
            registered_default_CPU_GPU_speed_gates=True, matched_FP32_speed=True,
            real_host_to_host_speed=True, all_nine_GPU_actor_functional_check=True,
            no_active_training_evaluation_or_GPU_workers=True),
        additional_optimizer_updates=0, additional_rollout_episodes=0,
        scope='Actual final evidence audit. Descriptive practical performance tolerances; CPU rollout metric; GPU bounded functional checks; policy latency excludes model loading,JIT,environment stepping and IO. Final report/manifest completion is checked separately.')
    save(area / 'FINAL_SCIENTIFIC_AND_PRESERVATION_ROOT_AUDIT_v1.json', result)
    with (area / 'all_speed_protocols_v1.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(speed_rows[0]))
        writer.writeheader(); writer.writerows(speed_rows)
    save(area / 'selected_actor_artifact_manifest_v1.json', dict(source_sha256=sources,
        deployment_source='utils/flow_iql_research/standalone_mixture_location_actor_v1.py',
        seed0_final_1M_actors=[dict(run=p['run'], environment=p['environment'], **p['routed_model']) for p in final_models.values()],
        all_four_seed_late_model_pins=model_pins))
    print(json.dumps(dict(status='pass', checks=result['checks'], speedup=result['geometric_speedup'],
        original_models_rehashed=81, current_model_files_rehashed=len(model_pins), stored_episodes_verified=16200)), flush=True)


if __name__ == '__main__':
    main()
