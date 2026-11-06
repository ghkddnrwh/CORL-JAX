"""Aggregate fixed last-three CPU50 evaluations; retain seed-level comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    rows, pins, complete_jobs, episode_records = [], {}, 0, 0
    baseline = {}
    for environment, records in plan['baseline_seed0_evaluations'].items():
        values = []
        for p, digest in records.items():
            assert sha(p) == digest
            d = json.loads(Path(p).read_text())
            assert d['step'] in plan['evaluation_steps'] and d['n_episodes'] == 50
            values.append(d['success_rate'])
        assert len(values) == 3
        baseline[environment] = statistics.mean(values)

    for job in plan['jobs']:
        folder = root/'runs'/job['id']
        config = folder/'config.json'
        if config.exists():
            conf = json.loads(config.read_text())
            identity = conf['identity']
            assert identity['seed'] == job['seed'] and identity['environment'] == job['environment']
            assert identity['variant'] == 'max_q' and identity['steps'] == 1000000
            cfg = identity['config']
            assert cfg['expectile'] == 1. and cfg['discount'] == .999 and cfg['tau'] == .005
            assert cfg['num_samples'] == 9 and cfg['flow_steps'] == 10
            assert conf['source_sha256'] == {p: plan['source_sha256'][p] for p in conf['source_sha256']}
        completed = folder/'completed.json'
        trained = completed.exists()
        if trained:
            done = json.loads(completed.read_text())
            assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 0
            complete_jobs += 1
        evaluations = []
        for step in plan['evaluation_steps']:
            p = root/'evaluations'/job['id']/f'{step}.json'
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            assert d['training_seed'] == job['seed'] and d['environment'] == job['environment']
            assert d['step'] == step and d['policy_mode'] == 'primary'
            assert d['Q_updates'] == step and d['V_updates'] == 0
            assert d['BC_student_updates'] == d['guided_policy_updates'] == 0
            assert d['n_episodes'] == 50 and len(d['episode_successes']) == 50 and d['first_episode_seed'] == 42
            assert d['success_rate'] == statistics.mean(d['episode_successes'])
            assert d['config_sha256'] == sha(config)
            if args.final:
                model = folder/f'checkpoint_{step}.pkl'
                assert sha(model) == d['checkpoint_sha256']
                pins[str(model)] = d['checkpoint_sha256']
            pins[str(p)] = sha(p)
            episode_records += len(d['episode_successes'])
            evaluations.append(d)
        by_step = {x['step']: x['success_rate'] for x in evaluations}
        last = {}
        diagnostic = folder/'diagnostics.jsonl'
        if diagnostic.exists():
            probes = [json.loads(line) for line in diagnostic.read_text().splitlines() if line.strip()]
            last = probes[-1] if probes else {}
            assert all(p['finite'] and p['v_updates'] == 0 and p['q_updates'] == p['step'] for p in probes)
            if args.final:
                assert len(probes) == 1000 and last['step'] == 1000000
        late = statistics.mean(by_step.values()) if len(by_step) == 3 else None
        rows.append(dict(run=job['id'], environment=job['environment'], seed=job['seed'],
                         trained=trained, evaluated_points=len(by_step),
                         success_800K=by_step.get(800000), success_900K=by_step.get(900000),
                         success_1M=by_step.get(1000000), last3_mean=late,
                         lean_tau09_seed0=baseline[job['environment']],
                         matched_seed0_difference=late-baseline[job['environment']] if late is not None and job['seed'] == 0 else None,
                         last_step=last.get('step'), q_rms=last.get('q_rms'),
                         q_rms_over_return_bound=last.get('q_rms_over_return_bound'),
                         last_clip_fraction=last.get('backup/clip_fraction'),
                         last_upper_clip_fraction=last.get('backup/upper_clip_fraction'),
                         last_max_minus_expectile09=last.get('max_minus_expectile09_mean'),
                         last_top_two_gap=last.get('top_two_gap_mean')))

    seed_means = {str(seed): statistics.mean([r['last3_mean'] for r in rows if r['seed'] == seed])
                  for seed in plan['training_seeds']
                  if len([r for r in rows if r['seed'] == seed and r['last3_mean'] is not None]) == 9}
    complete = complete_jobs == len(plan['jobs']) and len(seed_means) == len(plan['training_seeds'])
    if args.final:
        assert complete and episode_records == 2700
        for p, digest in plan['source_sha256'].items():
            assert sha(p) == digest
        for p, digest in plan['preserved_reference_sha256'].items():
            assert sha(p) == digest
    tasks = []
    for env in plan['environments']:
        r0 = next(r for r in rows if r['environment'] == env and r['seed'] == 0)
        r1 = next(r for r in rows if r['environment'] == env and r['seed'] == 1)
        both = r0['last3_mean'] is not None and r1['last3_mean'] is not None
        tasks.append(dict(environment=env, tau1_seed0=r0['last3_mean'], tau1_seed1=r1['last3_mean'],
                          tau1_two_seed_mean=statistics.mean([r0['last3_mean'], r1['last3_mean']]) if both else None,
                          historical_tau09_seed0=baseline[env],
                          matched_seed0_difference=r0['matched_seed0_difference']))
    summary = dict(time=time.time(), complete=complete, completed_training=complete_jobs,
                   expected_training=18, completed_evaluations=episode_records//50, expected_evaluations=54,
                   episode_records=episode_records, expectile_endpoint=1., gamma=.999,
                   training_seed_means=seed_means,
                   two_seed_mean=statistics.mean(seed_means.values()) if len(seed_means) == 2 else None,
                   two_seed_sample_SD=statistics.stdev(seed_means.values()) if len(seed_means) == 2 else None,
                   historical_lean_tau09_seed0_mean=statistics.mean(baseline.values()),
                   seed0_mean_difference=float(seed_means['0'])-statistics.mean(baseline.values()) if '0' in seed_means else None,
                   evaluation='all 800K/900K/1M CPU50 independent resets42..91, equal task/seed weighting',
                   scope='Tau1 direct max over9 behavior-flow proposals of mean twin targetQ. Deployment remains online minQ best9.',
                   limits=['No new tau0.9 seed1 baseline; compare tau1 versus tau0.9 directly at seed0 only.',
                           'Two exploratory training seeds do not establish significance or general stability.',
                           'Maximum is over9 sampled actions, not the full action space.',
                           'Tau1 score zero is algebraic and is not evidence of zero population or critic error.'],
                   source_sha256=plan['source_sha256'], evidence_sha256=pins)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    suffix = 'final_v1' if args.final else 'progress_v1'
    write_csv(out/f'run_results_{suffix}.csv', rows)
    write_csv(out/f'task_results_{suffix}.csv', tasks)
    (out/f'summary_{suffix}.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    text = ['Finite-mean lean: tau=1 max-Q 탐색 실험',
            f"상태: {'완료' if complete else '진행 중'}; 학습 {complete_jobs}/18, 평가 {episode_records//50}/54.",
            '9환경, training seed0/1, gamma0.999. Bootstrap만 mean-target-Q9개 최댓값으로 변경했다.',
            'V fitting0, 기존 BC Flow10step 및 online min-Q best9 배포를 유지했다.',
            'tau는 expectile endpoint1을 뜻하며 target EMA tau0.005는 유지했다.',
            '800K/900K/1M 각 CPU50, reset seed42..91 전부 평균한 후 환경과 학습seed를 동일 가중했다.',
            '표본9개의 최대값이며 전 행동공간 max나 모집단 expectile의 정확한 계산이라는 뜻은 아니다.', '']
    for seed, value in seed_means.items():
        text.append(f'tau1 seed{seed} 9환경 평균: {value:.9f}')
    if complete:
        text += [f"tau1 두 seed 평균: {summary['two_seed_mean']:.9f} (sample SD {summary['two_seed_sample_SD']:.9f})",
                 f"기존 lean tau0.9 seed0 평균: {summary['historical_lean_tau09_seed0_mean']:.9f}",
                 f"같은 seed0끼리 차이(tau1 - tau0.9): {summary['seed0_mean_difference']:+.9f}",
                 'tau0.9 seed1을 새로 학습하지 않았으므로 두 seed 차이의 paired 개선이라고 주장하지 않는다.', '']
    text.append('환경 / tau1 seed0 / tau1 seed1 / 두 seed 평균 / 기존tau0.9 seed0 / seed0차이')
    fmt = lambda x: '-' if x is None else f'{x:.6f}'
    for row in tasks:
        text.append(' / '.join([row['environment']] + [fmt(row[k]) for k in (
            'tau1_seed0', 'tau1_seed1', 'tau1_two_seed_mean', 'historical_tau09_seed0', 'matched_seed0_difference')]))
    text += ['', '실험은 가벼운 탐색이며 통계적 유의성, 모든 환경 개선, 발산 원인의 완전한 식별을 주장하지 않는다.',
             '원본 소스/결과를 수정하지 않고 별도 max_mean_lean_core_v1 및 runner/evaluator를 사용했다.']
    (out/f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
