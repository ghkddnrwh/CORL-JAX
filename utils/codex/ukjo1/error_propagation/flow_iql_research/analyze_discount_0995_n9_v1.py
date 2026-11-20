"""Audit and summarize all four training seeds without selecting checkpoints."""
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
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def csv_file(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def stats(values):
    if any(v is None for v in values):
        return None, None
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'registration_v1.json').read_text())
    assert plan['discount'] == .995 and plan['training_seeds'] == [0, 1, 2, 3]
    rows, evidence, checked, problems = [], {}, set(), []
    trained = dict(new=0, reused=0)
    evaluated = dict(new=0, reused=0)
    for job in plan['jobs'] + plan['reused_jobs']:
        folder = Path(job['output'])
        kind = 'reused' if job['reused'] else 'new'
        config_path = folder / 'config.json'
        conf = json.loads(config_path.read_text()) if config_path.exists() else None
        if conf:
            ident, cfg = conf['identity'], conf['identity']['config']
            assert ident['environment'] == job['environment'] and ident['seed'] == job['seed'] and ident['recipe'] == job['recipe']
            assert cfg['discount'] == .995 and cfg['expectile'] == .9 and cfg['tau'] == .005
            assert cfg['backup_samples'] == cfg['num_samples'] == 9 and cfg['target_clipping']
            assert cfg['backup_Q_aggregation'] == job['head_aggregation']
            assert all(plan['source_sha256'][p] == v for p, v in conf['source_sha256'].items())
        complete_path = folder / 'completed.json'
        if complete_path.exists():
            c = json.loads(complete_path.read_text())
            assert c['step'] == c['q_updates'] == c['actor_updates'] == 1000000 and c['v_updates'] == 0
            trained[kind] += 1
        if (folder / 'stopped.json').exists():
            problems.append(dict(run=job['id'], reason='nonfinite_state'))
        diagnostics_path = folder / 'diagnostics.jsonl'
        probes = [json.loads(s) for s in diagnostics_path.read_text().splitlines() if s] if diagnostics_path.exists() else []
        assert all(p['finite'] and p['q_updates'] == p['actor_updates'] == p['step'] and p['v_updates'] == 0 for p in probes)
        if args.final:
            assert len(probes) == 1000 and probes[-1]['step'] == 1000000
        directory = Path(job['evaluation_directory']) if job['reused'] else root / 'evaluations' / job['id']
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = directory / f'{step}_{mode}.json'
                if not path.exists():
                    continue
                v = json.loads(path.read_text())
                assert conf and v['training_seed'] == job['seed'] and v['environment'] == job['environment']
                assert v['recipe'] == job['recipe'] and v['run'] == folder.name
                assert v['step'] == step and v['policy_mode'] == mode
                assert v['n_episodes'] == len(v['episode_successes']) == 50 and v['first_episode_seed'] == 42
                assert v['no_optimizer_updates'] and v['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert abs(v['success_rate'] - statistics.mean(v['episode_successes'])) < 1e-12
                assert v['config_sha256'] == sha(config_path)
                if args.final:
                    for model, expected in ((folder / f'checkpoint_{step}.pkl', v['checkpoint_sha256']),
                                            (folder / f'actor_{step}.pkl', v['actor_artifact_sha256'])):
                        if model not in checked:
                            assert sha(model) == expected
                            evidence[str(model)] = expected
                            checked.add(model)
                evidence[str(path)] = sha(path)
                scores[step] = v['success_rate']
                evaluated[kind] += 1
            last = probes[-1] if probes else {}
            rows.append(dict(environment=job['environment'], training_seed=job['seed'], deployment=mode,
                gamma=.995, bootstrap_samples=9, cell_id=job['id'], reused=job['reused'],
                training_complete=complete_path.exists(), evaluated_points=len(scores),
                last3_mean=statistics.mean(scores.values()) if len(scores) == 3 else None,
                success_800K=scores.get(800000), success_900K=scores.get(900000), success_1M=scores.get(1000000),
                last_step=last.get('step'), q_rms=last.get('q_rms'), td_rms=last.get('td_rms')))
    index = {(r['environment'], r['training_seed'], r['deployment']): r for r in rows}
    assert len(index) == len(rows) == 80
    tasks = []
    for env in plan['environments']:
        for mode in plan['deployment_modes']:
            values = [index[env, seed, mode]['last3_mean'] for seed in plan['training_seeds']]
            final_values = [index[env, seed, mode]['success_1M'] for seed in plan['training_seeds']]
            mean, sd = stats(values)
            final_mean, final_sd = stats(final_values)
            tasks.append(dict(environment=env, deployment=mode, available_seeds=sum(v is not None for v in values),
                **{f'seed{s}': v for s, v in zip(plan['training_seeds'], values)},
                four_seed_mean=mean, four_seed_sample_sd=sd,
                worst_seed=min(values) if all(v is not None for v in values) else None,
                success_1M_mean=final_mean, success_1M_sample_sd=final_sd))
    macro_seed = []
    for mode in plan['deployment_modes']:
        for seed in plan['training_seeds']:
            values = [index[env, seed, mode]['last3_mean'] for env in plan['environments']]
            macro_seed.append(dict(deployment=mode, training_seed=seed,
                available_environments=sum(v is not None for v in values),
                ten_environment_mean=statistics.mean(values) if all(v is not None for v in values) else None))
    macros = []
    for mode in plan['deployment_modes']:
        values = [r['ten_environment_mean'] for r in macro_seed if r['deployment'] == mode]
        mean, sd = stats(values)
        macros.append(dict(deployment=mode, available_complete_seeds=sum(v is not None for v in values),
                          ten_environment_mean=mean, training_seed_sample_sd=sd))
    complete = (trained['new'] == len(plan['jobs']) and trained['reused'] == len(plan['reused_jobs'])
        and evaluated['new'] == plan['expected_new_evaluations'] and evaluated['reused'] == plan['expected_reused_evaluations'])
    if args.final:
        assert complete and not problems
        for field in ('source_sha256', 'preserved_reference_sha256', 'reused_evidence_sha256'):
            for path, expected in plan[field].items():
                assert sha(path) == expected, path
    suffix = 'final_v1' if args.final else 'progress_v1'
    out = root / 'analysis'
    csv_file(out / f'task_seed_results_{suffix}.csv', rows)
    csv_file(out / f'task_results_mean_sd_{suffix}.csv', tasks)
    csv_file(out / f'macro_by_seed_{suffix}.csv', macro_seed)
    csv_file(out / f'macro_results_mean_sd_{suffix}.csv', macros)
    summary = dict(time=time.time(), complete=complete, training_seeds=plan['training_seeds'], discount=.995,
        bootstrap_samples=9, new_training_complete=trained['new'], expected_new_training=len(plan['jobs']),
        reused_training_complete=trained['reused'], expected_reused_training=len(plan['reused_jobs']),
        new_evaluations=evaluated['new'], reused_evaluations=evaluated['reused'],
        expected_new_evaluations=plan['expected_new_evaluations'], expected_reused_evaluations=plan['expected_reused_evaluations'],
        episode_records=50 * sum(evaluated.values()), failures=problems, macros=macros, evidence_sha256=evidence)
    atomic(out / f'summary_{suffix}.json', summary)
    lines = ['Gamma 0.995, N=9, 10환경, 학습 seed 0/1/2/3',
        f"새 학습 {trained['new']}/{len(plan['jobs'])}, 재사용 {trained['reused']}/{len(plan['reused_jobs'])}.",
        f"새 평가 {evaluated['new']}/{plan['expected_new_evaluations']}, 재사용 {evaluated['reused']}/{plan['expected_reused_evaluations']}.",
        '각 시드의 800K/900K/1M CPU50 성공률을 평균한 뒤, 학습 시드 4개의 평균과 sample SD를 계산한다.',
        'Lean 배포 후보와 actor label 후보는 9개. 단일 actor는 기존 likelihood 학습 그대로다.',
        'Q 시간 평균, Q 학습률 변경, checkpoint 선택을 적용하지 않는다.', '', '환경 / 배포 / 완료 시드 / 성공률 평균 ± SD']
    for row in tasks:
        score = 'pending' if row['four_seed_mean'] is None else f"{100*row['four_seed_mean']:.2f} ± {100*row['four_seed_sample_sd']:.2f}%"
        lines.append(f"{row['environment']} / {row['deployment']} / {row['available_seeds']}/4 / {score}")
    (out / f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'evidence_sha256'}), flush=True)


if __name__ == '__main__':
    main()
