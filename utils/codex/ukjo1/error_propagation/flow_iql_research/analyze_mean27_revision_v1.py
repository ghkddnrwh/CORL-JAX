"""Pure mean N27 on Giant/Cube-double, preserving ongoing min9/min27 controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--final', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    rows, pins, trained, active_points, baseline_points = [], {}, 0, 0, 0
    failures, seen = [], set()
    for job in plan['jobs']+plan['baseline_jobs']:
        reference = job.get('baseline_reference', False)
        folder = Path(job['output'])
        if (folder/'config.json').exists():
            conf = json.loads((folder/'config.json').read_text())
            identity = conf['identity']
            cfg = identity['config']
            assert identity['environment'] == job['environment'] and identity['seed'] == job['seed'] and identity['recipe'] == job['recipe']
            assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['target_clipping']
            assert cfg['backup_Q_aggregation'] == job['head_aggregation'] and cfg['backup_samples'] == job['bootstrap_samples']
            assert cfg['num_samples'] == 9
            assert all(plan['source_sha256'][path] == digest for path, digest in conf['source_sha256'].items())
        done = folder/'completed.json'
        if done.exists():
            d = json.loads(done.read_text())
            assert d['step'] == d['q_updates'] == d['actor_updates'] == 1000000 and d['v_updates'] == 0
            if not reference:
                trained += 1
        if (folder/'stopped.json').exists():
            failures.append(dict(run=job['id'], **json.loads((folder/'stopped.json').read_text())))
        path = folder/'diagnostics.jsonl'
        probes = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        assert all(d['finite'] and d['q_updates'] == d['actor_updates'] == d['step'] and d['v_updates'] == 0 for d in probes)
        if args.final:
            assert len(probes) == 1000 and probes[-1]['step'] == 1000000
        last = probes[-1] if probes else {}
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = (Path(job['evaluation_directory'])/f'{step}_{mode}.json') if reference else root/'evaluations'/job['id']/f'{step}_{mode}.json'
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                assert d['training_seed'] == job['seed'] and d['environment'] == job['environment'] and d['recipe'] == job['recipe']
                assert d['step'] == step and d['policy_mode'] == mode and d['n_episodes'] == len(d['episode_successes']) == 50 and d['first_episode_seed'] == 42
                assert abs(d['success_rate']-statistics.mean(d['episode_successes'])) < 1e-12
                assert d['config_sha256'] == sha(folder/'config.json')
                if args.final:
                    for model, digest in ((folder/f'checkpoint_{step}.pkl', d['checkpoint_sha256']), (folder/f'actor_{step}.pkl', d['actor_artifact_sha256'])):
                        if model not in seen:
                            assert sha(model) == digest
                            pins[str(model)] = digest
                            seen.add(model)
                pins[str(path)] = sha(path)
                scores[step] = d['success_rate']
                if reference:
                    baseline_points += 1
                else:
                    active_points += 1
            rows.append(dict(run=job['id'], environment=job['environment'], head_aggregation=job['head_aggregation'],
                             bootstrap_samples=job['bootstrap_samples'], seed=job['seed'], deployment=mode,
                             historical_mean9_reference=reference, complete_training=done.exists(),
                             evaluated_points=len(scores), last3=statistics.mean(scores.values()) if len(scores) == 3 else None,
                             success_800K=scores.get(800000), success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             last_step=last.get('step'), q_rms=last.get('q_rms'),
                             would_clip_fraction=last.get('backup/would_clip_fraction'),
                             actual_bootstrap_count=last.get('diagnostic_actual_proposal_count', job['bootstrap_samples'])))
    cells = []
    keys = list(dict.fromkeys((r['environment'], r['head_aggregation'], r['bootstrap_samples'], r['deployment']) for r in rows))
    for env, agg, n, mode in keys:
        selected = [r for r in rows if (r['environment'], r['head_aggregation'], r['bootstrap_samples'], r['deployment']) == (env, agg, n, mode)]
        assert len(selected) == 2
        values = [next(r['last3'] for r in selected if r['seed'] == seed) for seed in (0, 1)]
        cells.append(dict(environment=env, head_aggregation=agg, bootstrap_samples=n, deployment=mode,
                          seed0=values[0], seed1=values[1],
                          mean_two_seeds=statistics.mean(values) if all(v is not None for v in values) else None))
    index = {(r['environment'], r['head_aggregation'], r['bootstrap_samples'], r['deployment'], r['seed']): r['last3'] for r in rows}
    effects = []
    for env in plan['environments']:
        comparisons = [('mean', 27, 'mean', 9, 'mean_N27_minus_N9')]
        if env.startswith('antmaze-'):
            comparisons += [('min', 27, 'min', 9, 'min_N27_minus_N9'),
                            ('mean', 27, 'min', 27, 'mean_minus_min_at_N27'),
                            ('mean', 9, 'min', 9, 'mean_minus_min_at_N9')]
        for a, an, b, bn, name in comparisons:
            for mode in plan['deployment_modes']:
                values = []
                for seed in (0, 1):
                    new = index[(env, a, an, mode, seed)]
                    old = index[(env, b, bn, mode, seed)]
                    values.append(new-old if new is not None and old is not None else None)
                effects.append(dict(environment=env, deployment=mode, comparison=name, seed0=values[0], seed1=values[1],
                                    mean_delta=statistics.mean(values) if all(v is not None for v in values) else None))
    complete = trained == 8 and active_points == 48 and baseline_points == 24
    if args.final:
        assert complete and not failures
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest
        for path, digest in plan['preserved_reference_sha256'].items():
            assert sha(path) == digest
    summary = dict(time=time.time(), complete=complete, active_study_training_complete=trained,
                   expected_active_study_training=8, active_study_evaluations=active_points,
                   expected_active_study_evaluations=48, historical_reference_evaluations=baseline_points,
                   expected_historical_reference_evaluations=24, cells=cells, effects=effects, failures=failures,
                   excluded_midpoint_runs=plan['cancelled_midpoint_runs'], source_sha256=plan['source_sha256'], evidence_sha256=pins,
                   limits=['Four ongoing pure-min learners are adopted without restart; four new pure-mean N27 learners replace midpoint candidates.',
                           'MeanN9 controls on Giant and Cube-double come from the completed target-ablation study, same training seeds and configuration.',
                           'Joint learners provide lean-best9 and actor-only deployments; these are not independent retrainings.',
                           'Bootstrap N27 only, both deployments and actor labels retain nine candidates.',
                           'Midpoint jobs were cancelled by explicit user preference before late evaluations; their partial results are excluded.'])
    out = root/'analysis'
    suffix = 'final_v1' if args.final else 'progress_v1'
    for stem, data in [('run_results', rows), ('cell_results', cells), ('paired_effects', effects)]:
        temp = out/f'{stem}_{suffix}.tmp'
        with temp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        temp.replace(out/f'{stem}_{suffix}.csv')
    (out/f'summary_{suffix}.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fmt = lambda x: '-' if x is None else f'{x:.6f}'
    text = ['사용자 요청 반영: pure mean bootstrap27, Antmaze Giant/Cube-double',
            f"학습{trained}/8, 새평가{active_points}/48, 기존mean9참고평가{baseline_points}/24.",
            '중간집계(mean+min)/2의4개run은사용자요청으로중단했고평가/집계에서제외했다.',
            '기존Giant min N9/N27 4개학습은같은PID로인계해중단/재시작없이진행했다.',
            '새Giant meanN27 2개및Cube-double meanN27 2개를seed0/1로학습했다.',
            'Cube-double선정근거:기존meanN9두seed평균lean0.25/actor0.343333으로낮았다.',
            'gamma0.999,tau0.9,known target clipping,BC Flow/MDN NLL actor및네트워크/학습률유지.',
            '행동27개는critic bootstrap에만사용하며배포와actor label은기존best9로고정했다.',
            '800K/900K/1M각CPU50 reset42..91전부평균후seed동일가중.완료전0대체나최고checkpoint선택없음.',
            'MeanN9기준선은이미완료한target-ablation의같은seed를재사용하며별도새N9학습으로표기하지않는다.', '',
            '환경 / head집계 / bootstrapN / 배포 / seed0 / seed1 / 두seed평균']
    for r in cells:
        text.append(' / '.join([r['environment'], r['head_aggregation'], str(r['bootstrap_samples']), r['deployment'], fmt(r['seed0']), fmt(r['seed1']), fmt(r['mean_two_seeds'])]))
    text += ['', '두seed의탐색이며유의성/9환경전체성능개선을주장하지않는다. 원래소스와중단전기록은보존했다.']
    (out/f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
