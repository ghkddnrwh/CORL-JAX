"""Audit paired seed0/1/2 results on environments selected from the seed0 sweep."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def paired_stats(values):
    if any(value is None for value in values):
        return None, None
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'registration_v1.json').read_text())
    assert plan['training_seeds'] == [0, 1, 2]
    envs, counts, seeds, modes = (plan[k] for k in ('environments', 'sample_counts', 'training_seeds', 'deployment_modes'))
    jobs = plan['jobs'] + plan['reused_jobs']
    rows, evidence, checked_models, problems = [], {}, set(), []
    new_trained = reused_trained = new_points = reused_points = 0
    for job in jobs:
        folder = Path(job['output'])
        config_path = folder / 'config.json'
        conf = json.loads(config_path.read_text()) if config_path.exists() else None
        if conf:
            identity, cfg = conf['identity'], conf['identity']['config']
            assert identity['environment'] == job['environment'] and identity['seed'] == job['seed']
            assert identity['recipe'] == job['recipe']
            assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['target_clipping']
            assert cfg['backup_samples'] == job['bootstrap_samples'] and cfg['num_samples'] == 9
            assert cfg['backup_Q_aggregation'] == ('min' if job['environment'].startswith('antmaze-') else 'mean')
            assert all(plan['source_sha256'][p] == digest for p, digest in conf['source_sha256'].items())
        completion = folder / 'completed.json'
        if completion.exists():
            value = json.loads(completion.read_text())
            assert value['step'] == value['q_updates'] == value['actor_updates'] == 1000000 and value['v_updates'] == 0
            if job.get('reused'):
                reused_trained += 1
            else:
                new_trained += 1
        if (folder / 'stopped.json').exists():
            problems.append(dict(run=job['id'], reason='stopped.json present'))
        diagnostic_path = folder / 'diagnostics.jsonl'
        probes = [json.loads(line) for line in diagnostic_path.read_text().splitlines() if line] if diagnostic_path.exists() else []
        assert all(p['finite'] and p['q_updates'] == p['actor_updates'] == p['step'] and p['v_updates'] == 0 for p in probes)
        if args.final:
            assert len(probes) == 1000 and probes[-1]['step'] == 1000000
        last = probes[-1] if probes else {}
        assert last.get('diagnostic_actual_proposal_count', job['bootstrap_samples']) == job['bootstrap_samples']
        directory = Path(job['evaluation_directory']) if job.get('reused') else root / 'evaluations' / job['id']
        for mode in modes:
            scores = {}
            for step in plan['evaluation_steps']:
                path = directory / f'{step}_{mode}.json'
                if not path.exists():
                    continue
                value = json.loads(path.read_text())
                assert conf and value['training_seed'] == job['seed'] and value['environment'] == job['environment']
                assert value['recipe'] == job['recipe'] and value['run'] == folder.name
                assert value['step'] == step and value['policy_mode'] == mode
                assert value['n_episodes'] == len(value['episode_successes']) == 50 and value['first_episode_seed'] == 42
                assert value['no_optimizer_updates'] and value['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert abs(value['success_rate'] - statistics.mean(value['episode_successes'])) < 1e-12
                assert value['config_sha256'] == sha(config_path)
                if args.final:
                    for model, expected in ((folder / f'checkpoint_{step}.pkl', value['checkpoint_sha256']),
                                            (folder / f'actor_{step}.pkl', value['actor_artifact_sha256'])):
                        if model not in checked_models:
                            assert sha(model) == expected, str(model)
                            evidence[str(model)] = expected
                            checked_models.add(model)
                evidence[str(path)] = sha(path)
                scores[step] = value['success_rate']
                if job.get('reused'):
                    reused_points += 1
                else:
                    new_points += 1
            rows.append(dict(cell_id=job['id'], actual_run=folder.name, environment=job['environment'],
                             training_seed=job['seed'], bootstrap_samples=job['bootstrap_samples'],
                             target_Q_aggregation=job['head_aggregation'], deployment=mode, reused=bool(job.get('reused')),
                             training_complete=completion.exists(), evaluated_points=len(scores),
                             last3_mean=statistics.mean(scores.values()) if len(scores) == 3 else None,
                             success_800K=scores.get(800000), success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             last_step=last.get('step'), q_rms=last.get('q_rms'), td_rms=last.get('td_rms'),
                             clip_fraction=last.get('backup/clip_fraction'), actual_bootstrap_count=last.get('diagnostic_actual_proposal_count', job['bootstrap_samples'])))
    index = {(r['environment'], r['bootstrap_samples'], r['training_seed'], r['deployment']): r for r in rows}
    assert len(rows) == len(index) == len(envs) * len(counts) * len(seeds) * len(modes)
    for row in rows:
        control = index[row['environment'], 9, row['training_seed'], row['deployment']]['last3_mean']
        row['paired_change_from_N9'] = row['last3_mean'] - control if row['last3_mean'] is not None and control is not None else None

    task_aggregates = []
    for env in envs:
        for n in counts:
            for mode in modes:
                selected = [index[env, n, seed, mode] for seed in seeds]
                values = [r['last3_mean'] for r in selected]
                differences = [r['paired_change_from_N9'] for r in selected]
                mean, sd = paired_stats(values)
                effect, effect_sd = paired_stats(differences)
                task_aggregates.append(dict(environment=env, bootstrap_samples=n, deployment=mode,
                    available_seeds=sum(v is not None for v in values), seed0=values[0], seed1=values[1], seed2=values[2],
                    three_seed_mean=mean, three_seed_sample_sd=sd,
                    seed1_2_mean=statistics.mean(values[1:]) if all(v is not None for v in values[1:]) else None,
                    paired_change_from_N9_mean=effect, paired_change_from_N9_sample_sd=effect_sd,
                    seed1_2_paired_change_from_N9_mean=statistics.mean(differences[1:]) if all(v is not None for v in differences[1:]) else None))
    macro_seed_rows = []
    for n in counts:
        for mode in modes:
            for seed in seeds:
                selected = [index[env, n, seed, mode] for env in envs]
                values = [r['last3_mean'] for r in selected]
                differences = [r['paired_change_from_N9'] for r in selected]
                macro_seed_rows.append(dict(bootstrap_samples=n, deployment=mode, training_seed=seed,
                    available_environments=sum(v is not None for v in values), expected_environments=len(envs),
                    selected_environment_mean=statistics.mean(values) if all(v is not None for v in values) else None,
                    paired_change_from_N9=statistics.mean(differences) if all(v is not None for v in differences) else None))
    macros = []
    for n in counts:
        for mode in modes:
            selected = [r for r in macro_seed_rows if r['bootstrap_samples'] == n and r['deployment'] == mode]
            values, differences = ([r[k] for r in selected] for k in ('selected_environment_mean', 'paired_change_from_N9'))
            mean, sd = paired_stats(values)
            effect, effect_sd = paired_stats(differences)
            macros.append(dict(bootstrap_samples=n, deployment=mode, available_complete_seeds=sum(v is not None for v in values),
                selected_environment_count=len(envs), seed0=values[0], seed1=values[1], seed2=values[2],
                three_seed_mean=mean, three_seed_sample_sd=sd,
                seed1_2_mean=statistics.mean(values[1:]) if all(v is not None for v in values[1:]) else None,
                paired_change_from_N9_mean=effect, paired_change_from_N9_sample_sd=effect_sd))
    complete = (new_trained == len(plan['jobs']) and reused_trained == len(plan['reused_jobs'])
                and new_points == plan['expected_new_evaluations'] and reused_points == plan['expected_reused_evaluations'])
    if args.final:
        assert complete and not problems
        for field in ('source_sha256', 'preserved_reference_sha256', 'reused_evidence_sha256'):
            for path, expected in plan[field].items():
                assert sha(path) == expected, path
    suffix = 'final_v1' if args.final else 'progress_v1'
    output = root / 'analysis'
    write_csv(output / f'task_seed_results_{suffix}.csv', rows)
    write_csv(output / f'task_results_mean_sd_{suffix}.csv', task_aggregates)
    write_csv(output / f'macro_by_seed_{suffix}.csv', macro_seed_rows)
    write_csv(output / f'macro_results_mean_sd_{suffix}.csv', macros)
    summary = dict(time=time.time(), complete=complete, training_seeds=seeds,
        new_training_complete=new_trained, expected_new_training=len(plan['jobs']),
        reused_training_complete=reused_trained, expected_reused_training=len(plan['reused_jobs']),
        new_evaluations=new_points, expected_new_evaluations=plan['expected_new_evaluations'],
        reused_evaluations=reused_points, expected_reused_evaluations=plan['expected_reused_evaluations'],
        late_evaluation_episode_records=50 * (new_points + reused_points), failures=problems, macros=macros,
        source_sha256=plan['source_sha256'], evidence_sha256=evidence,
        scope='Four environments selected using seed0 N-sensitivity. Report paired seed1/2 replication and seed0/1/2 descriptive means. No ten-environment or universal-optimal-N claim.')
    atomic(output / f'summary_{suffix}.json', summary)
    lines = ['표본 수 민감도가 큰 4환경: seed1·2 추가 검증',
             f'새 학습 {new_trained}/{len(plan["jobs"])}, 재사용 {reused_trained}/{len(plan["reused_jobs"])}.',
             f'새 평가 {new_points}/{plan["expected_new_evaluations"]}, 재사용 평가 {reused_points}/{plan["expected_reused_evaluations"]}.',
             '기존 seed0과 동일한 학습·평가 구현. N=8/9/16/18/27/32/36, seed=0/1/2.',
             'Actor label 및 lean 배포 후보9개 유지. Bootstrap 표본 수만 변경.',
             '800K/900K/1M 각 CPU50 reset42..91 전체 평균. SD는 학습 seed별 평균의 sample SD.',
             'Seed0의 N별 성공률 범위20pp 이상인 환경을 선택했다. 선택된 4환경의 기술통계이며 전체10환경 성능으로 해석하지 않는다.',
             '공통 후보 prefix를 강제하는 변경은 도입하지 않았다. 기존 seed0과 같은 난수 생성 구현으로 재현성을 확인한다.',
             'task_seed_results에 개별 시드와 같은 시드의 N9 대비 차이, task_results_mean_sd에 환경별 평균·SD와 seed1/2 평균을 저장한다.',
             '', 'N / 배포 / 완료 시드 수 / 선택 4환경 평균 ± SD']
    for row in macros:
        value = '-' if row['three_seed_mean'] is None else f"{row['three_seed_mean']:.6f} ± {row['three_seed_sample_sd']:.6f}"
        lines.append(f"{row['bootstrap_samples']} / {row['deployment']} / {row['available_complete_seeds']}/3 / {value}")
    (output / f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
