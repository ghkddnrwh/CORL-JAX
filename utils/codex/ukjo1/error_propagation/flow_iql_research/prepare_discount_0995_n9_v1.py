"""Register the requested 40 gamma=.995 N9 cells and audit exact reuse."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
from utils.codex.ukjo1.error_propagation.flow_iql_research import discount_0995_n9_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_discount_0995_n9_v1 import sha, atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    workspace = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
    folder = Path(__file__).resolve().parent
    assert not (root / 'registration_v1.json').exists(), 'Study already registered'
    numerical = json.loads((root / 'analysis/numerical_validation_v1.json').read_text())
    assert numerical['status'] == 'PASS'
    controller_validation = json.loads((root / 'analysis/controller_validation_v1.json').read_text())
    assert controller_validation['status'] == 'PASS'
    old_root = (workspace / 'logs/diagnosis/finite_mean_sample_budget_20261108').resolve()
    old_plan = json.loads((old_root / 'registration_v1.json').read_text())
    envs = old_plan['environments']
    assert set(envs) == set(core.ENVIRONMENTS) and len(envs) == 10
    declared_bounds = {}
    dataset_references = {}
    for env in envs:
        old_job = next(j for j in old_plan['jobs'] + old_plan['reused_jobs']
                       if j['environment'] == env and j['bootstrap_samples'] == 9)
        dataset_path = Path(old_job['output']) / 'dataset.json'
        dataset = json.loads(dataset_path.read_text())
        bound = core.return_bounds(env, .995, dataset['reward_min'], dataset['reward_max'], 'ogbench')
        assert abs(bound[0] - dataset['return_lower'] / 5.) < 1e-8
        assert bound[1] == dataset['return_upper'] == 0.
        declared_bounds[env] = list(bound)
        dataset_references[str(dataset_path)] = sha(dataset_path)
    giant_root = (workspace / 'logs/diagnosis/finite_mean_giant_gamma_tau_20261107').resolve()
    assert json.loads((giant_root / 'analysis/summary_final_v1.json').read_text())['complete']
    sources = {str(p): sha(p) for p in core.source_paths()}
    for prefix in ('train', 'evaluate', 'run', 'analyze', 'prepare', 'validate'):
        p = folder / f'{prefix}_discount_0995_n9_v1.py'
        sources[str(p)] = sha(p)
    p = folder / 'validate_discount_0995_controller_v1.py'
    sources[str(p)] = sha(p)
    evidence, reused = {}, []
    for seed in (0, 1):
        identity = f'gt1_g995_t90_s{seed}'
        run = giant_root / 'runs' / identity
        config_path = run / 'config.json'
        conf = json.loads(config_path.read_text())
        cfg = conf['identity']['config']
        new_cfg = json.loads(json.dumps(core.config_for(envs[-1], 'g995_n9')))
        assert envs[-1] == 'antmaze-giant-navigate-singletask-v0'
        assert {k: v for k, v in cfg.items() if k != 'onestep_specification'} == {k: v for k, v in new_cfg.items() if k != 'onestep_specification'}
        old_spec = {k: v for k, v in cfg['onestep_specification'].items() if k not in ('name', 'giant_search')}
        new_spec = {k: v for k, v in new_cfg['onestep_specification'].items() if k not in ('name', 'actor_label_samples', 'deployment_flow_samples', 'backup_samples')}
        assert old_spec == new_spec
        for path, digest in conf['source_sha256'].items():
            assert sha(path) == digest
            assert path not in sources or sources[path] == digest
            sources[path] = digest
        completion = json.loads((run / 'completed.json').read_text())
        assert completion['step'] == completion['q_updates'] == completion['actor_updates'] == 1000000 and completion['v_updates'] == 0
        for p in (config_path, run / 'completed.json', run / 'diagnostics.jsonl', run / 'dataset.json'):
            evidence[str(p)] = sha(p)
        dataset = json.loads((run / 'dataset.json').read_text())
        assert abs(dataset['return_lower'] + 200.) < 1e-9 and dataset['return_upper'] == 0.
        for step in old_plan['evaluation_steps']:
            checkpoint, actor = run / f'checkpoint_{step}.pkl', run / f'actor_{step}.pkl'
            evidence[str(checkpoint)], evidence[str(actor)] = sha(checkpoint), sha(actor)
            for mode in ('lean', 'actor'):
                path = giant_root / 'evaluations' / identity / f'{step}_{mode}.json'
                v = json.loads(path.read_text())
                assert v['training_seed'] == seed and v['environment'] == envs[-1]
                assert v['recipe'] == 'g995_t90' and v['step'] == step and v['policy_mode'] == mode
                assert v['n_episodes'] == len(v['episode_successes']) == 50 and v['first_episode_seed'] == 42
                assert v['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1' and v['no_optimizer_updates']
                assert v['checkpoint_sha256'] == evidence[str(checkpoint)] and v['actor_artifact_sha256'] == evidence[str(actor)]
                assert v['config_sha256'] == evidence[str(config_path)]
                evidence[str(path)] = sha(path)
        reused.append(dict(id=identity, environment=envs[-1], recipe='g995_t90', seed=seed,
            bootstrap_samples=9, head_aggregation='min', output=str(run), reused=True,
            original_study=str(giant_root), evaluation_directory=str(giant_root / 'evaluations' / identity)))
    tags = dict(zip(envs, ('ant', 'hmmed', 'hmlarge', 'soccer', 'cube1', 'cube2', 'pz3', 'pz4', 'scene', 'giant')))
    priority = [envs[1], envs[7]] + [e for e in envs if e not in (envs[1], envs[7])]
    existing = {(j['environment'], j['seed']) for j in reused}
    jobs = []
    for seed in range(4):
        for env in priority:
            if (env, seed) in existing:
                continue
            identity = f'g995n9_{tags[env]}_s{seed}'
            jobs.append(dict(id=identity, environment=env, recipe='g995_n9', seed=seed,
                bootstrap_samples=9, head_aggregation='min' if env.startswith('antmaze-') else 'mean',
                output=str(root / 'runs' / identity), train_module='utils.codex.ukjo1.error_propagation.flow_iql_research.train_discount_0995_n9_v1',
                eval_module='utils.codex.ukjo1.error_propagation.flow_iql_research.evaluate_discount_0995_n9_v1', reused=False))
    assert len(jobs) == 38 and len(reused) == 2
    assert len({(j['environment'], j['seed']) for j in jobs + reused}) == 40
    references = dict(dataset_references)
    for p in (old_root / 'registration_v1.json', giant_root / 'registration_v1.json',
              giant_root / 'analysis/summary_final_v1.json', root / 'analysis/numerical_validation_v1.json',
              root / 'analysis/controller_validation_v1.json'):
        references[str(p)] = sha(p)
    plan = dict(client_date_KST='2026-11-19', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='User requested gamma=.995, bootstrap N=9, original ten tasks, training seeds0/1/2/3.',
        environments=envs, training_seeds=[0, 1, 2, 3], new_training_seeds=[0, 1, 2, 3], sample_counts=[9],
        jobs=jobs, reused_jobs=reused, total_cells=40, new_training_runs=38, reused_training_runs=2,
        steps=1000000, discount=.995, expectile=.9, target_EMA=.005, target_clipping=True,
        return_bounds_by_environment=declared_bounds,
        head_rule='AntMaze min target heads; other tasks mean target heads.',
        actor_label_samples=9, lean_deployment_samples=9, critic_heads=2, actor_forward_passes=1,
        flow_steps=10, learned_V_fits=0, optimizer_lr=.0003,
        evaluation_steps=old_plan['evaluation_steps'], deployment_modes=['lean', 'actor'],
        episodes_per_point=50, reset_seeds=list(range(42, 92)), evaluation_affinity_cores=4,
        expected_new_evaluations=228, expected_reused_evaluations=12, expected_total_evaluations=240,
        expected_episode_records=12000, source_sha256=sources, preserved_reference_sha256=references,
        reused_evidence_sha256=evidence, concurrency=dict(training_per_GPU=8, total_training=16,
            memory_fraction=.1, training_CPU_cores=4, CPU_evaluation_workers=8, evaluation_CPU_cores=4),
        aggregation='For each seed average success at800K/900K/1M; report mean and sample SD across4 training seeds.',
        excluded_interventions=['temporal Q-score average', 'Q-only LR reduction', 'Q LR cosine decay', 'modified deployment selector'],
        reuse_scope='Giant gamma=.995 tau=.9 N9 seed0/1 only; config/source/hash audit and bitwise three-update equivalence.',
        limitations=['Four training seeds are distinct from evaluation episodes/checkpoints.',
                    'Existing gamma=.999 comparisons have unequal seed coverage; do not silently pool unmatched seeds.'])
    atomic(root / 'registration_v1.json', plan)
    with (root / 'analysis/experiment_matrix_v1.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['environment', 'seed', 'bootstrap_samples', 'reused', 'id', 'output'])
        writer.writeheader()
        writer.writerows({k: j[k] for k in writer.fieldnames} for j in jobs + reused)
    atomic(root / 'analysis/reuse_audit_v1.json', dict(status='PASS', completed_training_cells=2,
        completed_evaluation_cells=12, evidence_files=len(evidence), numerical_equivalence='three real optimizer updates per seed'))
    for path in sources:
        destination = root / 'source' / Path(path).relative_to(workspace)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    text = ['Gamma=.995, bootstrap N=9: 기존 10환경 x seed0/1/2/3 =40설정.',
        '기존 Giant seed0/1 2설정 재사용, 나머지38개는 처음부터1M까지 새로 학습한다.',
        'Gamma 및 gamma에 따른 return clipping 범위만 변경한다. Q 시간 평균, LR 개입을 적용하지 않는다.',
        'AntMaze target head min, 나머지 mean; expectile=.9, targetEMA=.005, LR=.0003 유지.',
        'Lean 배포 및 actor label 후보9개 유지. 각 학습에서 Lean과 단일 actor를 모두 평가한다.',
        '800K/900K/1M 각CPU50, reset seed42..91. 각 evaluator의 CPU affinity는4개로 고정한다.',
        '학습 시드별 세 checkpoint 평균을 구한 뒤 4시드 평균±sample SD를 집계한다.',
        'python -m utils.codex.ukjo1.error_propagation.flow_iql_research.run_discount_0995_n9_v1 --root ' + str(root),
        '완료 후 analysis/task_seed_results_final_v1.csv와 task_results_mean_sd_final_v1.csv를 확인한다.']
    (root / 'REPRODUCE_AND_SCOPE_KO_v1.txt').write_text('\n'.join(text) + '\n')
    link = workspace / 'logs/diagnosis' / root.name
    if not link.exists():
        link.symlink_to(root, target_is_directory=True)
    assert link.resolve() == root
    print(json.dumps(dict(root=str(root), new_training_runs=38, reused_training_runs=2, total_cells=40,
                          evaluation_points=240, episodes=12000), indent=2), flush=True)


if __name__ == '__main__':
    main()
