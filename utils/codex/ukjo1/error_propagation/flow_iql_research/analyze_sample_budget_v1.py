"""Seed0 seven-count x ten-environment audit; complete-cell macro averages only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def csv_write(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    assert plan['training_seed'] == 0 and plan['total_cells'] == 70
    rows, evidence, problems, seen = [], {}, [], set()
    new_trained = reused_trained = new_points = reused_points = 0
    for job in plan['jobs']+plan['reused_jobs']:
        reused = job.get('reused', False)
        folder = Path(job['output'])
        config_file = folder/'config.json'
        conf = json.loads(config_file.read_text()) if config_file.exists() else None
        if conf:
            identity, cfg = conf['identity'], conf['identity']['config']
            assert identity['environment'] == job['environment'] and identity['seed'] == 0
            assert identity['recipe'] == job['recipe']
            assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['target_clipping']
            assert cfg['backup_samples'] == job['bootstrap_samples'] and cfg['num_samples'] == 9
            assert cfg['backup_Q_aggregation'] == ('min' if job['environment'].startswith('antmaze-') else 'mean')
            assert all(plan['source_sha256'][p] == h for p, h in conf['source_sha256'].items())
        completion = folder/'completed.json'
        if completion.exists():
            value = json.loads(completion.read_text())
            assert value['step'] == value['q_updates'] == value['actor_updates'] == 1000000 and value['v_updates'] == 0
            if reused:
                reused_trained += 1
            else:
                new_trained += 1
        stopped = folder/'stopped.json'
        if stopped.exists():
            problems.append(dict(run=job['id'], **json.loads(stopped.read_text())))
        diagnostic_file = folder/'diagnostics.jsonl'
        probes = [json.loads(line) for line in diagnostic_file.read_text().splitlines() if line] if diagnostic_file.exists() else []
        assert all(v['finite'] and v['q_updates'] == v['actor_updates'] == v['step'] and v['v_updates'] == 0 for v in probes)
        if args.final:
            assert len(probes) == 1000 and probes[-1]['step'] == 1000000
        last = probes[-1] if probes else {}
        if probes:
            assert last.get('diagnostic_actual_proposal_count', job['bootstrap_samples']) == job['bootstrap_samples']
        directory = Path(job['evaluation_directory']) if reused else root/'evaluations'/job['id']
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = directory/f'{step}_{mode}.json'
                if not path.exists():
                    continue
                value = json.loads(path.read_text())
                assert conf and value['training_seed'] == 0 and value['environment'] == job['environment']
                assert value['recipe'] == job['recipe'] and value['run'] == folder.name
                assert value['step'] == step and value['policy_mode'] == mode
                assert value['n_episodes'] == len(value['episode_successes']) == 50 and value['first_episode_seed'] == 42
                assert value['no_optimizer_updates'] and value['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert abs(value['success_rate']-statistics.mean(value['episode_successes'])) < 1e-12
                assert value['config_sha256'] == sha(config_file)
                if args.final:
                    for model, expected in ((folder/f'checkpoint_{step}.pkl', value['checkpoint_sha256']),
                                            (folder/f'actor_{step}.pkl', value['actor_artifact_sha256'])):
                        if model not in seen:
                            assert sha(model) == expected
                            evidence[str(model)] = expected
                            seen.add(model)
                evidence[str(path)] = sha(path)
                scores[step] = value['success_rate']
                if reused:
                    reused_points += 1
                else:
                    new_points += 1
            rows.append(dict(cell_id=job['id'], actual_run=folder.name, environment=job['environment'],
                             training_seed=0, bootstrap_samples=job['bootstrap_samples'],
                             target_Q_aggregation=job['head_aggregation'], deployment=mode, reused=reused,
                             training_complete=completion.exists(), evaluated_points=len(scores),
                             last3_mean=statistics.mean(scores.values()) if len(scores) == 3 else None,
                             success_800K=scores.get(800000), success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             last_step=last.get('step'), q_rms=last.get('q_rms'), td_rms=last.get('td_rms'),
                             clip_fraction=last.get('backup/clip_fraction'),
                             finite_proposal_score=last.get('backup/finite_proposal_score_abs_mean'),
                             actual_bootstrap_count=last.get('diagnostic_actual_proposal_count', job['bootstrap_samples']),
                             sampling_expectile_ESS=last.get('sampling_expectile_ESS_mean'),
                             shared_GPU_scan_ms_per_update=last.get('shared_GPU_scan_ms_per_update')))
    assert len(rows) == 140
    index = {(row['environment'], row['bootstrap_samples'], row['deployment']): row for row in rows}
    assert len(index) == 140
    macros = []
    for n in plan['sample_counts']:
        for mode in plan['deployment_modes']:
            all_rows = [index[env, n, mode] for env in plan['environments']]
            available = [r for r in all_rows if r['last3_mean'] is not None]
            nine = [r for r in available if 'giant' not in r['environment']]
            macros.append(dict(bootstrap_samples=n, deployment=mode, available_environments=len(available),
                               expected_environments=10,
                               ten_environment_mean=statistics.mean(r['last3_mean'] for r in available) if len(available) == 10 else None,
                               original_nine_mean=statistics.mean(r['last3_mean'] for r in nine) if len(nine) == 9 else None))
    effects = []
    for env in plan['environments']:
        for n in plan['sample_counts']:
            if n == 9:
                continue
            for mode in plan['deployment_modes']:
                current, control = index[env, n, mode]['last3_mean'], index[env, 9, mode]['last3_mean']
                effects.append(dict(environment=env, bootstrap_samples=n, deployment=mode, N9_control=control,
                                    candidate=current, difference_from_N9=current-control if current is not None and control is not None else None))
    complete = (new_trained == len(plan['jobs']) and reused_trained == len(plan['reused_jobs'])
                and new_points == plan['expected_new_evaluations'] and reused_points == plan['expected_reused_evaluations'])
    if args.final:
        assert complete and not problems
        for field in ('source_sha256', 'preserved_reference_sha256', 'reused_evidence_sha256'):
            for path, digest in plan[field].items():
                assert sha(path) == digest, path
    summary = dict(time=time.time(), complete=complete, training_seed=0, new_training_complete=new_trained,
                   expected_new_training=len(plan['jobs']), reused_training_complete=reused_trained,
                   expected_reused_training=len(plan['reused_jobs']), new_evaluations=new_points,
                   expected_new_evaluations=plan['expected_new_evaluations'], reused_evaluations=reused_points,
                   expected_reused_evaluations=plan['expected_reused_evaluations'],
                   late_evaluation_episode_records=50*(new_points+reused_points), macros=macros, failures=problems,
                   source_sha256=plan['source_sha256'], evidence_sha256=evidence,
                   scope='Seed0 bootstrap-count sensitivity. No multi-seed superiority or per-task best-N macro claim.')
    output = root/'analysis'
    suffix = 'final_v1' if args.final else 'progress_v1'
    csv_write(output/f'task_results_{suffix}.csv', rows)
    csv_write(output/f'macro_results_{suffix}.csv', macros)
    csv_write(output/f'differences_from_N9_{suffix}.csv', effects)
    atomic(output/f'summary_{suffix}.json', summary)
    lines = [
        'Finite-mean lean + actor: seed0 bootstrap 표본 수 비교',
        f"새 학습 {new_trained}/{len(plan['jobs'])}, 재사용 학습 {reused_trained}/{len(plan['reused_jobs'])}.",
        f"새 평가 {new_points}/{plan['expected_new_evaluations']}, 재사용 평가 {reused_points}/{plan['expected_reused_evaluations']}.",
        'N_bootstrap=8/9/16/18/27/32/36, 10개 환경, training seed0.',
        'Antmaze-large/Giant는 target min, 나머지는 mean. gamma0.999, expectile0.9, clipping 유지.',
        'Actor label과 lean 배포는 best9로 고정. 단일 actor는 forward1회. Learned V fitting0회.',
        '각 learner에서 lean과 actor 배포를 평가하며 두 독립 학습으로 세지 않는다.',
        '800K/900K/1M 각 CPU50 reset42..91 전체 평균. 10환경이 모두 있을 때만 10환경 평균을 계산한다.',
        '재사용 행은 원래 run identity/경로를 유지하며 새 학습으로 표기하지 않는다.',
        '재사용 N9는 완료된 target-ablation 연구, 재사용 N27은 Giant(min)/Cube-double(mean) 연구다.',
        '새 표본 수를 환경별로 골라 합친 점수를 공통 알고리즘 성능으로 제시하지 않는다.',
        '이 실험은 단일 training seed 민감도 조사이며 독립 시드 검증이 아니다.', '',
        'N / 배포 / 완료 환경 수 / 10환경 평균',
    ]
    for row in macros:
        score = '-' if row['ten_environment_mean'] is None else f"{row['ten_environment_mean']:.6f}"
        lines.append(f"{row['bootstrap_samples']} / {row['deployment']} / {row['available_environments']}/10 / {score}")
    lines += ['', '환경별 값은 task_results, 기준 N9 대비 차이는 differences_from_N9에 기록했다.',
              '공유 GPU 업데이트 시간은 진단값이며 단독 하드웨어 속도 비교로 해석하지 않는다.']
    (output/f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
