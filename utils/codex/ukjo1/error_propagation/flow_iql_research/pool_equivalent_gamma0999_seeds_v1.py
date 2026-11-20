"""Pool all distinct seed IDs with audited equivalent N9 training/deployment.

Keep canonical seed0/1 from the existing comparison. Additional seed IDs
come from the N sweep, historical MDN runs and unused-V reference runs.
Alternative AntMaze mean targets are reported separately from min targets.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

STEPS = (800000, 900000, 1000000)
CONFIG_FIELDS = ('actor_hidden_dims', 'value_hidden_dims', 'actor_layer_norm', 'layer_norm',
                 'batch_size', 'discount', 'expectile', 'lr', 'tau', 'num_samples',
                 'flow_steps', 'policy_type', 'backup_samples', 'alpha')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'analysis/all_equivalent_seeds_gamma0999_20261120_v1'
    assert json.loads((out / 'numerical_equivalence_v1.json').read_text())['status'] == 'PASS'
    workspace = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
    diag = workspace / 'logs/diagnosis'
    canonical_root = (diag / 'finite_mean_target_ablation_20261105').resolve()
    canonical = json.loads((canonical_root / 'registration_v1.json').read_text())
    envs = canonical['environments']
    old_root = (diag / 'finite_mean_onestep_20261031').resolve()
    original_root = (diag / 'flow_iql_research_20261026').resolve()
    recent_root = (diag / 'finite_mean_sample_budget_multiseed_20261115').resolve()
    evidence, candidates, config_cache = {}, [], {}
    standards = {}
    for env in envs:
        recipe = 'ant_min_clip' if env.startswith('antmaze-') else 'mean_clip'
        job = next(j for j in canonical['jobs'] if j['environment'] == env and j['seed'] == 0 and j['recipe'] == recipe)
        run = canonical_root / 'runs' / job['id']
        standards[env] = (json.loads((run / 'config.json').read_text())['identity']['config'],
                          json.loads((run / 'dataset.json').read_text()))

    def pin(path):
        path = Path(path).resolve()
        if str(path) not in evidence:
            evidence[str(path)] = sha(path)
        return evidence[str(path)]

    def config(run):
        if run in config_cache:
            return config_cache[run]
        path = run / 'config.json'
        pin(path)
        conf = json.loads(path.read_text())
        identity, cfg = conf['identity'], conf['identity']['config']
        expected, data = standards[identity['environment']]
        for field in CONFIG_FIELDS:
            assert cfg[field] == expected[field], (run.name, field)
        for source, digest in conf['source_sha256'].items():
            assert pin(source) == digest, source
        dataset = run / 'dataset.json'
        pin(dataset)
        actual = json.loads(dataset.read_text())
        for field in ('size', 'reward_min', 'reward_max', 'return_lower', 'return_upper', 'normalize_states', 'action_clip_eps'):
            assert actual[field] == data[field], (run.name, field)
        completed_path = run / 'completed.json'
        pin(completed_path)
        done = json.loads(completed_path.read_text())
        assert done['step'] == done['q_updates'] == 1000000
        if identity.get('recipe') == 'mdn9_bc' or cfg.get('onestep_specification'):
            assert done['actor_updates'] == 1000000
        config_cache[run] = conf
        return conf

    def add(run, mode, raw_points, cohort, priority):
        conf = config(run)
        identity, cfg = conf['identity'], conf['identity']['config']
        scores = {}
        paths = []
        for path, v, expected_sha in raw_points:
            digest = pin(path)
            if expected_sha:
                assert digest == expected_sha
            assert v['step'] in STEPS and v['step'] not in scores
            assert v['n_episodes'] == len(v['episode_successes']) == 50 and v['first_episode_seed'] == 42
            assert v['protocol'] in ('last3_cpu50_independent_seeded_episodes_v1', 'common_routed_mixture_location_last3_CPU50_v1')
            if 'training_seed' in v:
                assert v['training_seed'] == identity['seed'] and v['environment'] == identity['environment']
            if 'config_sha256' in v:
                assert v['config_sha256'] == pin(run / 'config.json')
            score = statistics.mean(v['episode_successes'])
            assert abs(score - v['success_rate']) < 1e-12
            if mode == 'actor':
                spec = cfg['onestep_specification']
                assert spec['objective'] == 'none' and spec['mixture_components'] == 9
                assert spec['teacher_q_aggregation'] == 'min'
                if 'deployment_rule' in v:
                    assert v['deployment_rule'] == 'sampled_component_location'
                else:
                    assert spec['deployment_decoder'] == 'sampled_component_location'
            scores[v['step']] = 100 * score
            paths.append(str(path))
        assert set(scores) == set(STEPS)
        candidates.append(dict(environment=identity['environment'], deployment=mode,
            target_head_aggregation=cfg['backup_Q_aggregation'], training_seed=identity['seed'],
            cohort=cohort, priority=priority, run=run.name, run_directory=str(run),
            success_800K=scores[800000], success_900K=scores[900000], success_1M=scores[1000000],
            last3_success_pct=statistics.mean(scores.values()), raw_files=';'.join(paths)))

    # Original canonical seed0/1, including separate mean-target AntMaze controls.
    for job in canonical['jobs']:
        if job['recipe'] not in ('mean_clip', 'ant_min_clip'):
            continue
        run = canonical_root / 'runs' / job['id']
        for mode in ('lean', 'actor'):
            points = []
            for step in STEPS:
                p = canonical_root / 'evaluations' / job['id'] / f'{step}_{mode}.json'
                points.append((p, json.loads(p.read_text()), None))
            add(run, mode, points, 'canonical_target_ablation_seed01', 0)
    plan = json.loads((recent_root / 'registration_v1.json').read_text())
    for job in plan['jobs']:
        if job['bootstrap_samples'] != 9 or job['seed'] != 2:
            continue
        for mode in ('lean', 'actor'):
            points = []
            for step in STEPS:
                p = recent_root / 'evaluations' / job['id'] / f'{step}_{mode}.json'
                points.append((p, json.loads(p.read_text()), None))
            add(Path(job['output']), mode, points, 'N9_sweep_seed2', 1)
    old_audit = old_root / 'analysis/deployment_final_review_v1/FINAL_SCIENTIFIC_AND_PRESERVATION_ROOT_AUDIT_v1.json'
    pin(old_audit)
    groups = {}
    for point in json.loads(old_audit.read_text())['stored_episode_points']:
        if point['role'] not in ('teacher', 'routed'):
            continue
        path = Path(point['path'])
        value = json.loads(path.read_text())
        mode = 'lean' if point['role'] == 'teacher' else 'actor'
        groups.setdefault((value['run'], mode), []).append((path, value, point['sha256']))
    assert len(groups) == 72
    for (run, mode), points in groups.items():
        add(old_root / 'runs' / run, mode, points, 'historical_MDN_seed0_24_25_26', 2)
    original_review = original_root / 'analysis/mean_backup_final_review_v1/review.json'
    pin(original_review)
    review = json.loads(original_review.read_text())
    count = 0
    for path in (original_root / 'runs').glob('*/config.json'):
        c = json.loads(path.read_text())
        identity = c.get('identity', {})
        if identity.get('profile') != 'bounded_v2_mean_empirical_backup' or identity.get('seed') not in (0, 9, 10, 11):
            continue
        cfg = identity['config']
        if cfg.get('discount') != .999 or cfg.get('expectile') != .9 or cfg.get('backup_samples') != 9:
            continue
        run = path.parent
        evaluation = run / 'evaluations.jsonl'
        expected = review['run_metadata_sha256'][run.name]
        assert pin(path) == expected['config_sha256'] and pin(evaluation) == expected['evaluations_sha256']
        points = [(evaluation, json.loads(s), None) for s in evaluation.read_text().splitlines()
                  if json.loads(s)['step'] in STEPS]
        add(run, 'lean', points, 'original_unused_V_seed0_9_10_11', 3)
        count += 1
    assert count == 36
    # One contribution per numerical training seed ID, with a fixed outcome-free priority.
    selected, duplicate = {}, []
    for row in sorted(candidates, key=lambda r: (r['priority'], r['run'])):
        key = tuple(row[k] for k in ('environment', 'deployment', 'target_head_aggregation', 'training_seed'))
        if key in selected:
            duplicate.append(dict(**row, selected_run=selected[key]['run'], reason='Repeated numerical seed ID; retain canonical comparison run before historical runs.'))
        else:
            selected[key] = row
    main_rows, alternate_rows = [], []
    for row in selected.values():
        expected_head = 'min' if row['environment'].startswith('antmaze-') else 'mean'
        (main_rows if row['target_head_aggregation'] == expected_head else alternate_rows).append(row)
    assert len(main_rows) == 120 and len(alternate_rows) == 17
    summaries = []
    for env in envs:
        for mode in ('lean', 'actor'):
            rows = sorted([r for r in main_rows if r['environment'] == env and r['deployment'] == mode], key=lambda r: r['training_seed'])
            values = [r['last3_success_pct'] for r in rows]
            summaries.append(dict(environment=env, deployment=mode, num_seeds=len(rows),
                seeds='/'.join(str(r['training_seed']) for r in rows), mean_pct=statistics.mean(values),
                sample_sd_pp=statistics.stdev(values), min_pct=min(values), max_pct=max(values)))
    alternate_summary = []
    for env in sorted({r['environment'] for r in alternate_rows}):
        for mode in ('lean', 'actor'):
            rows = sorted([r for r in alternate_rows if r['environment'] == env and r['deployment'] == mode], key=lambda r: r['training_seed'])
            values = [r['last3_success_pct'] for r in rows]
            alternate_summary.append(dict(environment=env, deployment=mode, target_head_aggregation='mean',
                num_seeds=len(rows), seeds='/'.join(str(r['training_seed']) for r in rows),
                mean_pct=statistics.mean(values), sample_sd_pp=statistics.stdev(values)))
    macro = [dict(deployment=mode, environments=10, mean_pct=statistics.mean(r['mean_pct'] for r in summaries if r['deployment'] == mode),
        rule='Equal weight each environment after averaging its available seeds; unequal seed sets, no common-cohort macro SD.') for mode in ('lean', 'actor')]
    write_csv(out / 'included_unique_seed_results.csv', main_rows)
    write_csv(out / 'task_results_all_available_seeds.csv', summaries)
    write_csv(out / 'duplicate_seed_candidates.csv', duplicate)
    write_csv(out / 'different_AntMaze_mean_target_results.csv', alternate_rows)
    write_csv(out / 'different_AntMaze_mean_target_summary.csv', alternate_summary)
    write_csv(out / 'macro_results.csv', macro)
    write_csv(out / 'all_candidates.csv', candidates)
    result = dict(status='PASS', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gamma=.999, bootstrap_samples=9, expectile=.9, included_environment_seed_deployment_records=len(main_rows),
        unique_Lean_seed_cells=sum(r['deployment'] == 'lean' for r in main_rows),
        unique_actor_seed_cells=sum(r['deployment'] == 'actor' for r in main_rows),
        duplicate_seed_records=len(duplicate), separate_non_equivalent_AntMaze_mean_records=len(alternate_rows),
        original_V_and_current_active_update_equivalence=json.loads((out / 'numerical_equivalence_v1.json').read_text()),
        deduplication_rule='One numerical seed ID per environment/head rule/deployment; canonical seed0/1 first, then N9 sweep, historical MDN, original unused-V.',
        target_rule='AntMaze min; other tasks mean. AntMaze mean-target results remain separate.',
        metric='Per seed arithmetic mean of three independent-reset CPU50 evaluations at800K/900K/1M; then seed mean and sample SD.',
        limitations=['Retrospective pooling; seed sets differ by environment and deployment.',
                     'Same-number historical seed0 runs are listed as duplicates rather than overweighting seed0.',
                     'The all-available gamma=.999 summary is not a paired same-seed comparison with gamma=.995.'],
        macros=macro, evidence_sha256=evidence, source_sha256=sha(__file__))
    (out / 'pooling_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['Gamma=.999 N9 수학적으로 동일한 구현의 가용 시드 통합',
        '동일 시드 번호는 한 번만 집계: 앞서 비교한 canonical seed0/1을 우선 유지한다.',
        '추가 시드2,9/10/11,24/25/26을 조건에 맞는 환경·배포에 합친다.',
        '기존 V 학습은 Q/Flow/배포에 영향을 주지 않는다. 실제3회 업데이트에서 활성 파라미터·actor 차이0을 확인했다.',
        'Native Gaussian actor 평가를 섞지 않는다. 기존 actor는 현재와 같은 component-location 평가만 쓴다.',
        'AntMaze mean target과min target은 수학적으로 다르므로 별도 표로 보존한다.',
        '10환경 평균은 환경별 가용 시드 평균의 단순 평균이며, 모든 환경이 같은 시드를 갖는 결과가 아니다.']
    (out / 'POOLING_RULES_KO.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(tasks=summaries, alternative_AntMaze_mean=alternate_summary, macro=macro,
                          selected_records=len(main_rows), duplicate_records=len(duplicate)), indent=2), flush=True)


if __name__ == '__main__':
    main()
