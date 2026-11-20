"""Collect fixed-checkpoint scores without selecting tasks or checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import sha


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    rows, problems, trained, evaluated = [], [], 0, 0
    for job in plan['jobs']:
        run = Path(job['output'])
        config = json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else None
        complete = (run/'completed.json').exists()
        if complete:
            done = json.loads((run/'completed.json').read_text())
            assert done['step'] == done['q_updates'] == done['actor_updates'] == plan['steps']
            assert done['v_updates'] == 0
            trained += 1
        if (run/'stopped.json').exists():
            problems.append(dict(run=job['id'], reason='nonfinite_state'))
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = root/'evaluations'/job['id']/f'{step}_{mode}.json'
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                assert config and result['config_sha256'] == sha(run/'config.json')
                assert result['environment'] == job['environment'] and result['training_seed'] == 0
                assert result['discount'] == plan['discount'] and result['task']['task_id'] == job['task_id']
                assert result['step'] == step and result['policy_mode'] == mode
                assert result['first_episode_seed'] == 42 and result['n_episodes'] == len(result['episode_successes']) == 50
                assert result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert result['no_optimizer_updates']
                assert abs(result['success_rate']-statistics.mean(result['episode_successes'])) < 1e-12
                if args.final:
                    assert sha(run/f'checkpoint_{step}.pkl') == result['checkpoint_sha256']
                    assert sha(run/f'actor_{step}.pkl') == result['actor_artifact_sha256']
                scores[step] = 100*result['success_rate']
                evaluated += 1
            rows.append(dict(family=job['family'], task=job['task_id'], environment=job['environment'],
                             gamma=plan['discount'], seed=0, deployment=mode, training_complete=complete,
                             evaluation_points=len(scores), success_800K=scores.get(800000),
                             success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             last3_success_pct=statistics.mean(scores.values()) if len(scores)==3 else None))
    groups = []
    for _, family, default in plan['families']:
        for mode in plan['deployment_modes']:
            selected = [r for r in rows if r['family']==family and r['deployment']==mode]
            ready = [r for r in selected if r['last3_success_pct'] is not None]
            groups.append(dict(family=family, deployment=mode, excluded_default_task=default,
                               complete_tasks=len(ready), expected_tasks=4,
                               nondefault_mean_pct=statistics.mean(r['last3_success_pct'] for r in ready) if len(ready)==4 else None))
    complete = trained==40 and evaluated==240 and not problems
    if args.final:
        assert complete
        for path, expected in plan['source_sha256'].items():
            assert sha(path)==expected, path
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    suffix = 'final_v1' if args.final else 'progress_v1'
    write_csv(out/f'per_task_{suffix}.csv', rows)
    write_csv(out/f'family_summary_{suffix}.csv', groups)
    status = dict(time=time.time(), complete=complete, training_complete=trained, total_cells=40,
                  evaluations_complete=evaluated, expected_evaluations=240, failures=problems,
                  discount=plan['discount'], training_seed=0,
                  uncertainty='One training seed; seed variance cannot be estimated.')
    temporary = out/f'summary_{suffix}.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(out/f'summary_{suffix}.json')
    lines = [f"OGBench non-default 40 task / N9 / gamma={plan['discount']} / seed0", '',
             f'완료 학습 {trained}/40, 완료 평가 {evaluated}/240.',
             '각 값은 800K/900K/1M, 각 50 episode 성공률의 평균(%). 학습 시드는 하나다.', '',
             '환경 | task | Lean | 단일 actor']
    for job in plan['jobs']:
        selected = [r for r in rows if r['environment']==job['environment']]
        values = {r['deployment']: r['last3_success_pct'] for r in selected}
        fmt = lambda value: '대기' if value is None else f'{value:.2f}'
        lines.append(f"{job['code']} | {job['task_id']} | {fmt(values['lean'])} | {fmt(values['actor'])}")
    (out/f'RESULTS_KO_{suffix}.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
