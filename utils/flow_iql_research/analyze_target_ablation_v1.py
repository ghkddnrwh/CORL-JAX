"""Matched seed/factor/role comparisons for target-Q and projection ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_file(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    rows, pins, trained, failed, episode_records, model_checks = [], {}, 0, [], 0, set()
    diagnostics = {}
    for job in plan['jobs']:
        folder = root/'runs'/job['id']
        config = folder/'config.json'
        if config.exists():
            conf = json.loads(config.read_text())
            identity = conf['identity']
            assert identity['seed'] == job['seed'] and identity['environment'] == job['environment'] and identity['recipe'] == job['recipe']
            cfg = identity['config']
            assert cfg['expectile'] == .9 and cfg['discount'] == .999 and cfg['tau'] == .005
            assert cfg['backup_Q_aggregation'] == job['target_Q_aggregation'] and cfg['target_clipping'] == job['target_clipping']
            assert cfg['num_samples'] == 9 and cfg['flow_steps'] == 10
            assert all(plan['source_sha256'][p] == h for p, h in conf['source_sha256'].items())
        done = folder/'completed.json'
        if done.exists():
            d = json.loads(done.read_text())
            assert d['step'] == d['q_updates'] == d['actor_updates'] == 1000000 and d['v_updates'] == 0
            trained += 1
        if (folder/'stopped.json').exists():
            failed.append(dict(run=job['id'], **json.loads((folder/'stopped.json').read_text())))
        p = folder/'diagnostics.jsonl'
        probes = [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
        if probes:
            assert all(x['finite'] and x['q_updates'] == x['actor_updates'] == x['step'] and x['v_updates'] == 0 for x in probes)
            last = probes[-1]
            diagnostics[job['id']] = last
            if args.final:
                assert len(probes) == 1000 and last['step'] == 1000000
        for mode in plan['deployment_modes']:
            by_step = {}
            for step in plan['evaluation_steps']:
                path = root/'evaluations'/job['id']/f'{step}_{mode}.json'
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                assert d['step'] == step and d['training_seed'] == job['seed'] and d['environment'] == job['environment']
                assert d['recipe'] == job['recipe'] and d['policy_mode'] == mode
                assert d['n_episodes'] == len(d['episode_successes']) == 50 and d['first_episode_seed'] == 42
                assert abs(d['success_rate']-statistics.mean(d['episode_successes'])) < 1e-12
                assert d['config_sha256'] == sha(config)
                if args.final:
                    for p, digest in ((folder/f'checkpoint_{step}.pkl', d['checkpoint_sha256']),
                                      (folder/f'actor_{step}.pkl', d['actor_artifact_sha256'])):
                        if p not in model_checks:
                            assert sha(p) == digest
                            pins[str(p)] = digest
                            model_checks.add(p)
                pins[str(path)] = sha(path)
                episode_records += 50
                by_step[step] = d['success_rate']
            last = diagnostics.get(job['id'], {})
            rows.append(dict(run=job['id'], environment=job['environment'], seed=job['seed'], recipe=job['recipe'],
                             target_Q_aggregation=job['target_Q_aggregation'], target_clipping=job['target_clipping'],
                             deployment=mode, complete_training=done.exists(), evaluated_points=len(by_step),
                             success_800K=by_step.get(800000), success_900K=by_step.get(900000), success_1M=by_step.get(1000000),
                             last3_mean=statistics.mean(by_step.values()) if len(by_step) == 3 else None,
                             last_step=last.get('step'), q_rms=last.get('q_rms'),
                             last_applied_clip_fraction=last.get('backup/clip_fraction'),
                             last_would_clip_fraction=last.get('backup/would_clip_fraction'),
                             last_lower_outside_fraction=last.get('backup/lower_out_of_bounds_fraction'),
                             last_upper_outside_fraction=last.get('backup/upper_out_of_bounds_fraction'),
                             last_actor_NLL=last.get('mixture_NLL_per_dimension')))

    cells = []
    for env in plan['environments']:
        recipes = [j['recipe'] for j in plan['jobs'] if j['environment'] == env and j['seed'] == 0]
        for recipe in recipes:
            for mode in plan['deployment_modes']:
                selected = [r for r in rows if r['environment'] == env and r['recipe'] == recipe and r['deployment'] == mode]
                complete = len(selected) == 2 and all(r['last3_mean'] is not None for r in selected)
                cells.append(dict(environment=env, recipe=recipe, deployment=mode,
                                  seed0=next((r['last3_mean'] for r in selected if r['seed'] == 0), None),
                                  seed1=next((r['last3_mean'] for r in selected if r['seed'] == 1), None),
                                  mean_two_seeds=statistics.mean(r['last3_mean'] for r in selected) if complete else None))
    index = {(r['environment'], r['seed'], r['recipe'], r['deployment']): r['last3_mean'] for r in rows}
    differences = []
    for env in plan['environments']:
        comparisons = [('mean_unclipped', 'mean_clip', 'clipping_off_minus_on_mean')]
        if env.startswith('antmaze-'):
            comparisons += [('ant_min_clip', 'mean_clip', 'min_minus_mean_clipped'),
                            ('ant_min_unclipped', 'mean_unclipped', 'min_minus_mean_unclipped'),
                            ('ant_min_unclipped', 'ant_min_clip', 'clipping_off_minus_on_min')]
        for new, control, comparison in comparisons:
            for mode in plan['deployment_modes']:
                deltas = {}
                for seed in plan['training_seeds']:
                    a, b = index.get((env, seed, new, mode)), index.get((env, seed, control, mode))
                    deltas[seed] = a-b if a is not None and b is not None else None
                differences.append(dict(environment=env, deployment=mode, comparison=comparison,
                                        seed0_delta=deltas[0], seed1_delta=deltas[1],
                                        two_seed_mean_delta=statistics.mean(deltas.values()) if all(x is not None for x in deltas.values()) else None))

    macros = []
    for scope, envs in [('original_nine', plan['original_environments']), ('all_ten_including_giant', plan['environments'])]:
        for deployment in plan['deployment_modes']:
            for clipping in (True, False):
                for rule in ('all_mean', 'antmaze_min_others_mean'):
                    values = []
                    for seed in plan['training_seeds']:
                        task_values = []
                        for env in envs:
                            recipe = ('ant_min' if env.startswith('antmaze-') and rule == 'antmaze_min_others_mean' else 'mean') + ('_clip' if clipping else '_unclipped')
                            value = index.get((env, seed, recipe, deployment))
                            if value is not None:
                                task_values.append(value)
                        values.append(statistics.mean(task_values) if len(task_values) == len(envs) else None)
                    macros.append(dict(scope=scope, deployment=deployment, target_rule=rule, target_clipping=clipping,
                                       seed0=values[0], seed1=values[1],
                                       two_seed_mean=statistics.mean(values) if all(x is not None for x in values) else None))
    expected = len(plan['jobs'])*len(plan['evaluation_steps'])*len(plan['deployment_modes'])
    complete = trained == len(plan['jobs']) and episode_records == expected*50
    if args.final:
        assert complete and not failed
        for p, digest in plan['source_sha256'].items():
            assert sha(p) == digest
        for p, digest in plan['preserved_reference_sha256'].items():
            assert sha(p) == digest
    summary = dict(time=time.time(), complete=complete, training_complete=trained, expected_training=len(plan['jobs']),
                   completed_evaluations=episode_records//50, expected_evaluations=expected,
                   episode_records=episode_records, stopped_runs=failed, macro_results=macros,
                   source_sha256=plan['source_sha256'], evidence_sha256=pins,
                   limits=['48 joint learners provide two same-base deployments, not96 independently trained learners.',
                           'Seed0/1 exploratory comparisons, no statistical significance claim.',
                           'Original nine-task and expanded ten-task means are reported separately.',
                           'Known target return clipping is independent of required action clipping.'])
    out = root/'analysis'
    suffix = 'final_v1' if args.final else 'progress_v1'
    for stem, data in [('run_results', rows), ('task_results', cells), ('paired_effects', differences), ('macro_results', macros)]:
        csv_file(out/f'{stem}_{suffix}.csv', data)
    (out/f'summary_{suffix}.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    text = ['Finite-mean lean 및 actor: target-Q 집계, clipping, Antmaze Giant 실험',
            f"상태: {'완료' if complete else '진행 중'}, 학습{trained}/{len(plan['jobs'])}, CPU50평가{episode_records//50}/{expected}",
            '공통: tau0.9/gamma0.999, target EMA0.005, N9, Flow10, V fitting0, 기존 공통MDN9 NLL actor.',
            'Antmaze Large/Giant: mean/min x target clipping on/off x seed0/1.',
            '나머지8환경: mean x target clipping on/off x seed0/1.',
            '48개 공동학습에서 각기 lean Flow-best9와 actor component-location을 평가한다. 96개 독립 학습이라는 뜻은 아니다.',
            'Actor optimizer는 critic/Flow에 feedback하지 않는다. 별도 lean과 동일한 base 업데이트인지는 기능 검증에서 확인했다.',
            '각 환경/시드/배포의800K/900K/1M을 CPU50(reset42..91)으로 평가한 마지막3회 평균.',
            '모든 cell을 새 소스로 학습해 matched controls를 둔다. 과거 seed0 값은 이 matched 평균에 섞지 않는다.',
            'Clipping off는 TD target projection만 제거하며 행동clip[-1,1]이나 mixture likelihood 정의를 바꾸지 않는다.',
            'Giant도 gamma0.999로 고정했다. FQL 공개 Giant 예시의gamma0.995와 구분한다.',
            '원래9환경 평균과 Giant 포함10환경 평균을 분리하며, 최고checkpoint/환경별최고설정을 골라 합치지 않는다.', '',
            'scope / deployment / target rule / clipping / seed0 / seed1 / 두seed평균']
    fmt = lambda x: '-' if x is None else f'{x:.6f}'
    for r in macros:
        text.append(' / '.join([r['scope'], r['deployment'], r['target_rule'], str(r['target_clipping']), fmt(r['seed0']), fmt(r['seed1']), fmt(r['two_seed_mean'])]))
    text += ['', '환경별 상세는 task_results 및 paired_effects CSV 참조.',
             '미완료/수치실패 run을 조용히 제외하거나0으로 대체하지 않는다. 두seed만으로 유의성을 주장하지 않는다.']
    if failed:
        text += ['', '수치 실패 기록: '+json.dumps(failed, ensure_ascii=False)]
    (out/f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
