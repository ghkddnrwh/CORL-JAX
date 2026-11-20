"""Register the selected-environment seed1/2 extension of the frozen N sweep."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/sda2/ukjo1/finite_mean_sample_budget_multiseed_20261115'))
    args = parser.parse_args()
    workspace = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
    old_root = (workspace / 'logs/diagnosis/finite_mean_sample_budget_20261108').resolve()
    root = args.root.resolve()
    assert not (root / 'registration_v1.json').exists(), 'Study already registered'
    old_plan = json.loads((old_root / 'registration_v1.json').read_text())
    old_summary = json.loads((old_root / 'analysis/summary_final_v1.json').read_text())
    assert old_summary['complete'] and not old_summary['failures']
    for field in ('source_sha256', 'preserved_reference_sha256'):
        for path, digest in old_plan[field].items():
            assert sha(path) == digest, path
    with (old_root / 'analysis/task_results_final_v1.csv').open() as stream:
        seed0_rows = list(csv.DictReader(stream))
    selection = []
    for env in old_plan['environments']:
        row = dict(environment=env)
        for mode in old_plan['deployment_modes']:
            values = [float(r['last3_mean']) for r in seed0_rows if r['environment'] == env and r['deployment'] == mode]
            row[f'{mode}_range_percentage_points'] = 100 * (max(values) - min(values))
        row['selected'] = max(row['lean_range_percentage_points'], row['actor_range_percentage_points']) >= 20
        selection.append(row)
    selected = sorted([r for r in selection if r['selected']],
                      key=lambda r: max(r['lean_range_percentage_points'], r['actor_range_percentage_points']), reverse=True)
    envs = [r['environment'] for r in selected]
    assert set(envs) == {'humanoidmaze-medium-navigate-singletask-v0', 'puzzle-4x4-play-singletask-v0',
                         'antmaze-large-navigate-singletask-v0', 'antsoccer-arena-navigate-singletask-v0'}
    tags = {'humanoidmaze-medium-navigate-singletask-v0': 'hmmed', 'puzzle-4x4-play-singletask-v0': 'pz4',
            'antmaze-large-navigate-singletask-v0': 'ant', 'antsoccer-arena-navigate-singletask-v0': 'soccer'}
    counts = old_plan['sample_counts']
    old_index = {(j['environment'], j['bootstrap_samples']): j for j in old_plan['jobs'] + old_plan['reused_jobs']}
    reused = []
    for env in envs:
        for n in counts:
            job = dict(old_index[env, n])
            job.update(reused=True, original_study=job.get('original_study', str(old_root)))
            if 'evaluation_directory' not in job:
                job['evaluation_directory'] = str(old_root / 'evaluations' / job['id'])
            reused.append(job)
    target_root = Path('/data/sda2/ukjo1/finite_mean_target_ablation_20261105')
    for env in envs:
        tag = tags[env]
        recipe = 'ant_min_clip' if env.startswith('antmaze-') else 'mean_clip'
        identity = f'tab1_{tag}_{recipe}_s1'
        reused.append(dict(id=identity, environment=env, recipe=recipe, seed=1, bootstrap_samples=9,
                           head_aggregation='min' if env.startswith('antmaze-') else 'mean',
                           output=str(target_root / 'runs' / identity),
                           evaluation_directory=str(target_root / 'evaluations' / identity),
                           reused=True, original_study=str(target_root)))
    existing = {(j['environment'], j['bootstrap_samples'], j['seed']) for j in reused}
    jobs = []
    priority = [8, 9, 27, 32, 36, 16, 18]
    for n in priority:
        for env in envs:
            for seed in (1, 2):
                if (env, n, seed) in existing:
                    continue
                identity = f'ms1_{tags[env]}_N{n}_s{seed}'
                jobs.append(dict(id=identity, environment=env, recipe=f'n{n}', seed=seed,
                    head_aggregation='min' if env.startswith('antmaze-') else 'mean', bootstrap_samples=n,
                    output=str(root / 'runs' / identity), train_module='utils.codex.ukjo1.error_propagation.flow_iql_research.train_sample_budget_v1',
                    eval_module='utils.codex.ukjo1.error_propagation.flow_iql_research.evaluate_sample_budget_v1', reused=False))
    assert len(jobs) == 52 and len(reused) == 32
    assert len({(j['environment'], j['bootstrap_samples'], j['seed']) for j in jobs + reused}) == 84
    evidence = {}

    def preserve(path):
        path = Path(path)
        if str(path) not in evidence:
            evidence[str(path)] = sha(path)
        return evidence[str(path)]

    for job in reused:
        folder = Path(job['output'])
        config_path = folder / 'config.json'
        conf = json.loads(config_path.read_text())
        cfg, identity = conf['identity']['config'], conf['identity']
        assert identity['seed'] == job['seed'] and identity['environment'] == job['environment'] and identity['recipe'] == job['recipe']
        assert cfg['backup_samples'] == job['bootstrap_samples'] and cfg['num_samples'] == 9
        assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['target_clipping']
        assert cfg['backup_Q_aggregation'] == job['head_aggregation']
        assert all(old_plan['source_sha256'][p] == digest for p, digest in conf['source_sha256'].items())
        completion = json.loads((folder / 'completed.json').read_text())
        assert completion['step'] == completion['q_updates'] == completion['actor_updates'] == 1000000 and completion['v_updates'] == 0
        for p in (config_path, folder / 'completed.json', folder / 'diagnostics.jsonl'):
            preserve(p)
        for step in old_plan['evaluation_steps']:
            checkpoint, actor = folder / f'checkpoint_{step}.pkl', folder / f'actor_{step}.pkl'
            ch, ah = preserve(checkpoint), preserve(actor)
            for mode in old_plan['deployment_modes']:
                path = Path(job['evaluation_directory']) / f'{step}_{mode}.json'
                value = json.loads(path.read_text())
                assert value['training_seed'] == job['seed'] and value['environment'] == job['environment']
                assert value['recipe'] == job['recipe'] and value['step'] == step and value['policy_mode'] == mode
                assert value['n_episodes'] == len(value['episode_successes']) == 50 and value['first_episode_seed'] == 42
                assert value['no_optimizer_updates'] and value['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert value['config_sha256'] == evidence[str(config_path)]
                assert value['checkpoint_sha256'] == ch and value['actor_artifact_sha256'] == ah
                preserve(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'analysis').mkdir(exist_ok=True)
    old_validation = old_root / 'analysis/numerical_validation_v1.json'
    assert json.loads(old_validation.read_text())['status'] == 'PASS'
    check = subprocess.run([sys.executable, '-m', 'utils.codex.ukjo1.error_propagation.flow_iql_research.validate_sample_budget_multiseed_controller_v1'],
                           cwd=workspace, capture_output=True, text=True, check=True)
    regression = json.loads(check.stdout)
    assert regression['status'] == 'PASS'
    dump(root / 'analysis/controller_aggregation_validation_v1.json', regression)
    dump(root / 'analysis/numerical_validation_v1.json', dict(status='PASS',
         scope='Unchanged learner/evaluator validated in the completed seed0 study; new controller and multi-seed aggregation regression tested.',
         unchanged_learner_validation_path=str(old_validation), unchanged_learner_validation_sha256=sha(old_validation),
         inherited_real_update_cases=len(json.loads(old_validation.read_text())['real_update_cases']),
         new_controller_and_aggregation_validation=regression))
    sources = dict(old_plan['source_sha256'])
    for name in ('run_sample_budget_multiseed_v1.py', 'analyze_sample_budget_multiseed_v1.py',
                 'validate_sample_budget_multiseed_controller_v1.py', 'prepare_sample_budget_multiseed_v1.py'):
        p = Path(__file__).resolve().parent / name
        sources[str(p)] = sha(p)
    references = dict(old_plan['preserved_reference_sha256'])
    for p in (old_root / 'registration_v1.json', old_root / 'analysis/summary_final_v1.json',
              old_root / 'analysis/task_results_final_v1.csv', old_root / 'analysis/macro_results_final_v1.csv', old_validation):
        references[str(p)] = sha(p)
    plan = dict(client_date_KST='2026-11-15', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Replicate the seed0-sensitive environments at training seeds1/2 without changing proposal generation.',
        training_seeds=[0, 1, 2], new_training_seeds=[1, 2], environments=envs, sample_counts=counts,
        selection_rule='Across N=8/9/16/18/27/32/36, seed0 lean OR actor success range >=20 percentage points.',
        selection=selection, execution_priority=priority, jobs=jobs, reused_jobs=reused, total_cells=84,
        requested_additional_cells=56, new_training_runs=52, reused_seed1_cells=4, reused_seed0_cells=28,
        steps=1000000, discount=.999, expectile=.9, target_EMA=.005, target_clipping=True,
        head_rule='Antmaze min, other tasks mean', actor_label_samples=9, lean_deployment_samples=9,
        actor_forward_passes=1, flow_steps=10, learned_V_fits=0, shared_base_for_two_deployments=True,
        evaluation_steps=old_plan['evaluation_steps'], deployment_modes=old_plan['deployment_modes'],
        episodes_per_point=50, reset_seeds=list(range(42, 92)),
        expected_new_evaluations=312, expected_reused_evaluations=192, expected_total_evaluations=504,
        expected_episode_records=25200, concurrency=old_plan['concurrency'], source_sha256=sources,
        preserved_reference_sha256=references, reused_evidence_sha256=evidence,
        limitations=['Environments were selected using the seed0 sweep; four-task means do not establish ten-task performance.',
                     'Keep original random tensor shapes and per-N sampling. Same seed does not guarantee common proposal prefixes.',
                     'Report each seed and paired same-seed differences from N9. Evaluation reset seeds/checkpoints are not training seeds.',
                     'Previously completed N9 seed1 controls are reused, not newly trained or prospectively blinded.'])
    dump(root / 'registration_v1.json', plan)
    write_csv(root / 'analysis/environment_selection_v1.csv', selection)
    write_csv(root / 'analysis/experiment_matrix_v1.csv', [dict(environment=j['environment'], seed=j['seed'],
        bootstrap_samples=j['bootstrap_samples'], reused=j['reused'], run=j['id'], output=j['output']) for j in jobs + reused])
    dump(root / 'analysis/reuse_audit_v1.json', dict(status='PASS', completed_cells=32, evaluation_points=192,
        model_artifacts=192, evidence_files=len(evidence), training_seed0_cells=28, training_seed1_cells=4))
    for path in sources:
        destination = root / 'source' / Path(path).relative_to(workspace)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    text = ['표본 수 민감도가 큰 4환경: seed1·2 추가 실험',
            '대상: HumanoidMaze Medium, Puzzle4x4, AntMaze Large, AntSoccer.',
            '선택 기준: 기존 seed0에서 N별 성공률 범위가 lean 또는 actor에서20pp 이상.',
            'N=8/9/16/18/27/32/36 전부 유지. Seed1·2 추가56설정 중 N9·seed1 4개 재사용, 신규52개 학습.',
            '기존 seed0 28설정도 함께 보존해 총84설정의 seed0/1/2 결과를 분석한다.',
            '학습·평가 구현과 모든 하이퍼파라미터는 기존 sweep 그대로다. 공통 후보 prefix 강제 변경은 없다.',
            'Bootstrap 표본 수만 다르며 actor label/lean 배포 후보9개, V fit0, gamma0.999, expectile0.9, target EMA0.005 유지.',
            '각 학습에서 lean/단일 actor 평가. 800K/900K/1M CPU50, reset42..91 전체 평균.',
            'GPU0/1 각8개, 총16개 동시 학습. 각 memory fraction0.1, CPU4개. CPU평가8개 별도.',
            'python -m utils.codex.ukjo1.error_propagation.flow_iql_research.run_sample_budget_multiseed_v1 --root ' + str(root),
            '완료 분석: analysis/task_seed_results_final_v1.csv, task_results_mean_sd_final_v1.csv, macro_results_mean_sd_final_v1.csv.',
            '선택된4환경의 기술통계다. 전체10환경 평균이나 모든환경의 최적 N으로 확대 해석하지 않는다.']
    (root / 'REPRODUCE_AND_SCOPE_KO_v1.txt').write_text('\n'.join(text) + '\n')
    link = workspace / 'logs/diagnosis/finite_mean_sample_budget_multiseed_20261115'
    if not link.exists():
        link.symlink_to(root, target_is_directory=True)
    assert link.resolve() == root
    print(json.dumps(dict(root=str(root), selected_environments=envs, new_training_runs=len(jobs),
                         reused_training_runs=len(reused), total_cells=84, regression='PASS'), indent=2))


if __name__ == '__main__':
    main()
