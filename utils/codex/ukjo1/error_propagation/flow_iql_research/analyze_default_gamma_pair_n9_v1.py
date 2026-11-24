"""Complete gamma x four-seed default-task results, without partial-seed pooling."""
import argparse
import json
from pathlib import Path
import statistics as st
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import (
    FAMILIES, sha, atomic, write_csv, assert_matrix, audit_config, audit_evaluation)


def aggregate(rows, plan):
    results = []
    for code, env in FAMILIES:
        for gamma in plan['discounts']:
            for mode in plan['deployment_modes']:
                selected = sorted([r for r in rows if (r['environment'], r['gamma'], r['deployment']) == (env, gamma, mode)], key=lambda r: r['seed'])
                assert [r['seed'] for r in selected] == plan['training_seeds']
                ready = [r['last3_success_pct'] for r in selected if r['last3_success_pct'] is not None]
                full = len(ready) == 4
                results.append(dict(code=code, environment=env, gamma=gamma, deployment=mode,
                    complete_seeds=len(ready), mean_pct=st.mean(ready) if full else None,
                    sample_sd_pp=st.stdev(ready) if full else None,
                    **{f'seed{r["seed"]}_pct': r['last3_success_pct'] for r in selected}))
    paired = []
    for code, env in FAMILIES:
        for mode in plan['deployment_modes']:
            low = next(r for r in results if (r['environment'], r['gamma'], r['deployment']) == (env, .995, mode))
            high = next(r for r in results if (r['environment'], r['gamma'], r['deployment']) == (env, .999, mode))
            deltas = [low[f'seed{s}_pct'] - high[f'seed{s}_pct'] for s in range(4)
                      if low[f'seed{s}_pct'] is not None and high[f'seed{s}_pct'] is not None]
            paired.append(dict(code=code, environment=env, deployment=mode, complete_pairs=len(deltas),
                               gamma0995_minus0999_pp=st.mean(deltas) if len(deltas) == 4 else None,
                               paired_sample_sd_pp=st.stdev(deltas) if len(deltas) == 4 else None))
    return results, paired


def collect(plan, final=False):
    rows, problems, evidence = [], [], {}
    trained = evaluated = 0
    for job in plan['reused_jobs'] + plan['jobs']:
        run = Path(job['output'])
        conf = json.loads((run / 'config.json').read_text()) if (run / 'config.json').exists() else None
        if conf:
            audit_config(job, conf)
        complete = (run / 'completed.json').exists()
        if complete:
            done = json.loads((run / 'completed.json').read_text())
            assert done['step'] == done['q_updates'] == done['actor_updates'] == plan['steps'] and done['v_updates'] == 0
            trained += 1
        if (run / 'stopped.json').exists():
            problems.append(dict(id=job['id'], reason='nonfinite_state'))
        for mode in plan['deployment_modes']:
            scores = {}
            for step in plan['evaluation_steps']:
                path = Path(job['evaluation_directory']) / f'{step}_{mode}.json'
                if path.exists():
                    assert conf is not None
                    scores[step] = audit_evaluation(job, conf, step, mode, path, evidence if final else None)
                    evaluated += 1
            rows.append(dict(code=job['code'], environment=job['environment'], gamma=job['gamma'], seed=job['seed'],
                deployment=mode, action=job['action'], training_complete=complete, evaluation_points=len(scores),
                success_800K_pct=scores.get(800000), success_900K_pct=scores.get(900000), success_1M_pct=scores.get(1000000),
                last3_success_pct=st.mean(scores.values()) if len(scores) == 3 else None,
                run_directory=str(run), evaluation_directory=job['evaluation_directory']))
    status = dict(time=time.time(), complete=trained == 80 and evaluated == 480 and not problems,
                  training_complete=trained, total_cells=80, evaluations_complete=evaluated,
                  expected_evaluations=480, failures=problems, training_seeds=plan['training_seeds'],
                  discounts=plan['discounts'])
    if final:
        assert status['complete'], status
        for name, digest in plan['source_sha256'].items():
            assert sha(name) == digest, name
    return rows, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'registration_v1.json').read_text())
    assert_matrix(plan)
    rows, status = collect(plan, args.final)
    groups, paired = aggregate(rows, plan)
    suffix = 'final_v1' if args.final else 'progress_v1'
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    for name, data in (('per_seed_results', rows), ('task_mean_sd', groups), ('paired_gamma_difference', paired)):
        write_csv(output / f'{name}_{suffix}.csv', data)
    atomic(output / f'summary_{suffix}.json', status)
    lines = ['Default 10 task / N=9 / gamma=.995,.999 / seed0,1,2,3', '',
             f"완료 학습 {status['training_complete']}/80, 평가 {status['evaluations_complete']}/480.",
             '시드별 800K·900K·1M CPU50 성공률을 평균한 뒤 네 학습 시드 평균±sample SD(%).',
             '네 시드가 모두 있어야 평균을 표시한다. 진행 중 결과는 per_seed_results CSV를 확인한다.', '',
             '환경 | gamma | Lean 평균±SD | 단일 actor 평균±SD']
    for code, env in FAMILIES:
        for gamma in plan['discounts']:
            formatted = []
            for mode in plan['deployment_modes']:
                r = next(r for r in groups if (r['environment'], r['gamma'], r['deployment']) == (env, gamma, mode))
                formatted.append(f"대기({r['complete_seeds']}/4)" if r['mean_pct'] is None else f"{r['mean_pct']:.2f}±{r['sample_sd_pp']:.2f}")
            lines.append(f'{code} | {gamma} | ' + ' | '.join(formatted))
    (output / f'RESULTS_KO_{suffix}.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
