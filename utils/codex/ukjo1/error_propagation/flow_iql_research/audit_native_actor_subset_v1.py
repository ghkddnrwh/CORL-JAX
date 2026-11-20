"""Verify explicitly named native checkpoints; no subset success claim."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research.watch_onestep_deploy_v9 import Verifier
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--steps', nargs='+', type=int, required=True)
    p.add_argument('--with-last3-evaluations', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    plan = json.loads(a.registration.read_text())
    state = json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    verifier = Verifier(plan)
    rows = []
    for rid in a.runs:
        job, = [j for j in state['jobs'] if j['id'] == rid]
        assert state['identities'][rid] == plan['expected_identities'][rid]
        run = a.root/'runs'/rid
        conf = verifier.config(run, state['identities'][rid])
        assert conf
        snapshots, points = [], []
        for step in a.steps:
            pin = verifier.model(run, conf, step)
            snapshots.append(pin)
            if a.with_last3_evaluations:
                assert a.steps == [800000, 900000, 1000000]
                for mode in ('primary', 'teacher'):
                    path = a.root/'analysis/evaluations_v1'/f'{rid}_{step}_{mode}_n50.json'
                    verified = verifier.evaluation(path, job, step, mode, 50, pin)
                    points.append(dict(step=step, mode=mode, path=str(path), **verified))
        row = dict(run=rid, environment=job['args']['env'], recipe=job['args']['recipe'],
                   training_seed=job['args']['seed'], models=snapshots, evaluations=points)
        if points:
            row.update({f'{mode}_last3': sum(x['successes'] for x in points if x['mode']==mode)/150
                        for mode in ('primary','teacher')})
        rows.append(row)
    result = dict(time=time.time(), status='pass', source_sha256=sha(__file__),
        registration_sha256=sha(a.registration), rows=rows,
        actual_models=sum(len(x['models']) for x in rows),
        actual_evaluated_episodes=sum(x['episodes'] for r in rows for x in r['evaluations']),
        no_new_optimizer_updates=True, no_models_or_primary_results_modified=True,
        scope='Explicit actual checkpoint subset integrity. No full-nine, fresh-seed, isolated-latency, or final-goal claim.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status='pass', runs=len(rows), actual_models=result['actual_models'],
                         actual_evaluated_episodes=result['actual_evaluated_episodes'])))


if __name__ == '__main__':
    main()
