"""Apply the user's seed0/1-only correction without changing live learners."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_discount_0995_n9_v2 import atomic, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert not (root / 'registration_v2.json').exists()
    old = json.loads((root / 'registration_v1.json').read_text())
    audit_path = root / 'analysis/prior_completed_environment_audit_v2.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'PASS'
    assert {r['environment'] for r in audit['matched_completed']} == {'antmaze-giant-navigate-singletask-v0'}
    assert {r['seed'] for r in audit['matched_completed']} == {0, 1}
    validation_path = root / 'analysis/controller_validation_v2.json'
    assert json.loads(validation_path.read_text())['status'] == 'PASS'
    selected = [j for j in old['jobs'] if j['seed'] in (0, 1)]
    cancelled = [j for j in old['jobs'] if j['seed'] not in (0, 1)]
    assert len(selected) == 18 and len(cancelled) == 20
    envs = [e for e in old['environments'] if e != 'antmaze-giant-navigate-singletask-v0']
    assert len(envs) == 9
    assert {(j['environment'], j['seed']) for j in selected} == {(e, s) for e in envs for s in (0, 1)}
    for path, expected in old['source_sha256'].items():
        assert sha(path) == expected, path
    plan = dict(old)
    sources = dict(old['source_sha256'])
    directory = Path(__file__).resolve().parent
    for name in ('run_discount_0995_n9_v2.py', 'analyze_discount_0995_n9_v2.py',
                 'validate_discount_0995_controller_v2.py', 'amend_discount_0995_n9_scope_v2.py'):
        path = directory / name
        sources[str(path)] = sha(path)
    references = dict(old['preserved_reference_sha256'])
    for path in (root / 'registration_v1.json', audit_path, validation_path):
        references[str(path)] = sha(path)
    plan.update(amended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        amendment_reason='User corrected requested seeds to0/1 and excluded previously completed matched environments.',
        purpose='Gamma=.995 N9 seed0/1 on9 previously untested environments; Giant already completed and excluded.',
        registration_version=2, environments=envs, training_seeds=[0, 1], new_training_seeds=[0, 1],
        jobs=selected, reused_jobs=[], total_cells=18, new_training_runs=18, reused_training_runs=0,
        excluded_completed_environments=['antmaze-giant-navigate-singletask-v0'],
        excluded_completed_reference_jobs=old['reused_jobs'], cancelled_seed_2_3_jobs=cancelled,
        expected_new_evaluations=108, expected_reused_evaluations=0, expected_total_evaluations=108,
        expected_episode_records=5400, source_sha256=sources, preserved_reference_sha256=references,
        reused_evidence_sha256={}, excluded_completed_evidence_sha256=old['reused_evidence_sha256'],
        return_bounds_by_environment={e: old['return_bounds_by_environment'][e] for e in envs},
        aggregation='For each seed average success at800K/900K/1M; report seed0/1 values, mean and sample SD.',
        reuse_scope='Giant is excluded from this new-study queue and aggregate; previous results remain available separately.',
        limitations=['Only two training seeds; report both seed-level values alongside sample SD.',
                     'Nine-environment aggregate excludes the already completed Giant.'])
    atomic(root / 'registration_v2.json', plan)
    with (root / 'analysis/experiment_matrix_v2.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['environment', 'seed', 'bootstrap_samples', 'id', 'output'])
        writer.writeheader()
        writer.writerows({k: j[k] for k in writer.fieldnames} for j in selected)
    workspace = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
    for path in sources:
        destination = root / 'source' / Path(path).relative_to(workspace)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        assert sha(destination) == sources[path]
    alias = workspace / 'logs/diagnosis/finite_mean_gamma0995_n9_9task_2seed_20261119_v2'
    if not alias.exists():
        alias.symlink_to(root, target_is_directory=True)
    assert alias.resolve() == root
    text = ['사용자 수정 반영: gamma=.995 N=9, 신규9환경 x seed0/1 =18설정.',
        '현재 설정은 registration_v2.json. v1의40설정 계획은 변경 전 이력으로만 보존한다.',
        'AntMaze Giant는 동일 gamma/N/expectile의 seed0/1이 이미 완료되어 신규 실행·집계에서 제외한다.',
        'Seed2/3의20개 예약을 취소한다. 현재 seed0/1 학습 프로세스와 데이터 경로는 유지한다.',
        '학습·actor·평가 구현 및 하이퍼파라미터는 변경하지 않는다.',
        '800K/900K/1M 각CPU50, Lean과actor 평가; seed0/1 각각과 평균±sample SD를 기록한다.',
        'python -m utils.codex.ukjo1.error_propagation.flow_iql_research.run_discount_0995_n9_v2 --root ' + str(root),
        '완료 결과: analysis/task_seed_results_final_v2.csv, task_results_mean_sd_final_v2.csv.',
        '진행 상태: controller_status.json. 물리 경로의4seed 명칭은 기존 학습 경로를 유지하기 위한 역사적 명칭이다.']
    (root / 'CURRENT_SCOPE_KO_v2.txt').write_text('\n'.join(text) + '\n')
    print(json.dumps(dict(training_seeds=[0, 1], environments=9, total_training=18,
        cancelled_seed_2_3=20, excluded_completed_environment='AntMaze Giant', alias=str(alias)), indent=2))


if __name__ == '__main__':
    main()
