"""Giant-only matched discount/expectile pilot, fixed last-three CPU50 metric."""
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
    rows, trained, failures, pins, evaluated, seen = [], 0, [], {}, 0, set()
    for job in plan['jobs']:
        run = root/'runs'/job['id']
        if (run/'config.json').exists():
            conf = json.loads((run/'config.json').read_text())
            identity = conf['identity']
            cfg = identity['config']
            assert identity['recipe'] == job['recipe'] and identity['seed'] == job['seed']
            assert identity['environment'] == plan['environment']
            assert cfg['discount'] == job['discount'] and cfg['expectile'] == job['expectile']
            assert cfg['backup_Q_aggregation'] == 'min' and cfg['target_clipping']
            assert all(plan['source_sha256'][p] == h for p, h in conf['source_sha256'].items())
        done = run/'completed.json'
        if done.exists():
            d = json.loads(done.read_text())
            assert d['q_updates'] == d['actor_updates'] == 1000000 and d['v_updates'] == 0
            trained += 1
        if (run/'stopped.json').exists():
            failures.append(dict(run=job['id'], **json.loads((run/'stopped.json').read_text())))
        probe = run/'diagnostics.jsonl'
        diagnostics = [json.loads(s) for s in probe.read_text().splitlines() if s.strip()] if probe.exists() else []
        assert all(d['finite'] and d['q_updates'] == d['actor_updates'] == d['step'] and d['v_updates'] == 0 for d in diagnostics)
        if args.final:
            assert len(diagnostics) == 1000 and diagnostics[-1]['step'] == 1000000
        last = diagnostics[-1] if diagnostics else {}
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = root/'evaluations'/job['id']/f'{step}_{mode}.json'
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                assert d['step'] == step and d['policy_mode'] == mode and d['recipe'] == job['recipe'] and d['training_seed'] == job['seed']
                assert d['n_episodes'] == len(d['episode_successes']) == 50 and d['first_episode_seed'] == 42
                assert abs(d['success_rate']-statistics.mean(d['episode_successes'])) < 1e-12
                assert d['config_sha256'] == sha(run/'config.json')
                if args.final:
                    for m, digest in ((run/f'checkpoint_{step}.pkl', d['checkpoint_sha256']),
                                      (run/f'actor_{step}.pkl', d['actor_artifact_sha256'])):
                        if m not in seen:
                            assert sha(m) == digest
                            pins[str(m)] = digest
                            seen.add(m)
                pins[str(path)] = sha(path)
                scores[step] = d['success_rate']
                evaluated += 1
            rows.append(dict(recipe=job['recipe'], gamma=job['discount'], expectile=job['expectile'], seed=job['seed'],
                             deployment=mode, evaluated_points=len(scores), last3=statistics.mean(scores.values()) if len(scores) == 3 else None,
                             success_800K=scores.get(800000), success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             training_complete=done.exists(), last_step=last.get('step'), q_rms=last.get('q_rms'),
                             q_rms_over_known_bound=last.get('q_rms_over_return_bound'),
                             applied_clip_fraction=last.get('backup/clip_fraction')))
    cells = []
    for recipe in plan['settings']:
        for mode in plan['deployment_modes']:
            selected = [r for r in rows if r['recipe'] == recipe and r['deployment'] == mode]
            values = [next(r['last3'] for r in selected if r['seed'] == seed) for seed in (0, 1)]
            cells.append(dict(recipe=recipe, gamma=plan['settings'][recipe][0], expectile=plan['settings'][recipe][1],
                              deployment=mode, seed0=values[0], seed1=values[1],
                              mean_two_seeds=statistics.mean(values) if all(x is not None for x in values) else None))
    idx = {(r['recipe'], r['deployment'], r['seed']): r['last3'] for r in rows}
    effects = []
    for mode in plan['deployment_modes']:
        for new, control, name in [('g995_t90', 'g999_t90', 'gamma995_minus999_at_tau90'),
                                    ('g995_t95', 'g999_t95', 'gamma995_minus999_at_tau95'),
                                    ('g999_t95', 'g999_t90', 'tau95_minus90_at_gamma999'),
                                    ('g995_t95', 'g995_t90', 'tau95_minus90_at_gamma995')]:
            values = []
            for seed in (0, 1):
                a, b = idx[(new, mode, seed)], idx[(control, mode, seed)]
                values.append(a-b if a is not None and b is not None else None)
            effects.append(dict(deployment=mode, effect=name, seed0=values[0], seed1=values[1],
                                mean_delta=statistics.mean(values) if all(x is not None for x in values) else None))
    complete = trained == 8 and evaluated == 48
    if args.final:
        assert complete and not failures
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest
        for path, digest in plan['preserved_reference_sha256'].items():
            assert sha(path) == digest
    summary = dict(time=time.time(), complete=complete, training_complete=trained, expected_training=8,
                   evaluations_complete=evaluated, expected_evaluations=48, episode_records=evaluated*50,
                   cells=cells, effects=effects, stopped=failures, source_sha256=plan['source_sha256'], evidence_sha256=pins,
                   limits=['Eight joint learners, each evaluated through lean-best9 and a single actor.',
                           'Two exploratory training seeds; no significance/generalization claim.',
                           'Known return projection changes consistently with gamma: lower bound -1000 or -200.',
                           'No nine/ten-task performance claim follows from this Giant-only search.'])
    suffix = 'final_v1' if args.final else 'progress_v1'
    out = root/'analysis'
    for stem, data in [('run_results', rows), ('cell_results', cells), ('factor_effects', effects)]:
        with (out/f'{stem}_{suffix}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out/f'summary_{suffix}.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fmt = lambda x: '-' if x is None else f'{x:.6f}'
    text = ['Antmaze Giant: gamma/expectile 2x2 탐색',
            f"상태: {'완료' if complete else '진행 중'}; 학습{trained}/8, 평가{evaluated}/48.",
            'gamma0.999/0.995 x expectile0.9/0.95 x training seed0/1. Min target-Q, target projection 유지.',
            'Gamma0.995는FQL공식Giant기본값이며 tau0.95는finite expectile의상위행동가중을강하게하는탐색이다.',
            'Gamma에맞춰known return범위도[-1000,0]또는[-200,0]으로계산한다.',
            'BC Flow10,N9,MDN9 likelihood actor및두배포규칙,lr/구조는유지했다.',
            '각learner를lean-best9와단일actor component-location으로평가한다.16개독립학습은아니다.',
            '800K/900K/1M각CPU50 reset42..91을평균한뒤seed를동일가중했다.', '',
            'gamma / tau / 배포 / seed0 / seed1 / 평균']
    for row in cells:
        text.append(' / '.join([str(row['gamma']), str(row['expectile']), row['deployment'], fmt(row['seed0']), fmt(row['seed1']), fmt(row['mean_two_seeds'])]))
    text += ['', '새gamma999/tau90 control을함께학습한다.과거pilot결과를새matched평균에섞지않는다.',
             'Giant두seed의개선만으로원래9환경또는전체10환경개선을주장하지않는다.']
    (out/f'RESEARCH_REPORT_KO_{suffix}.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'evidence_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
