"""Actual-model and CPU50 last-three audit for completed two-V-fit pilots."""
import argparse
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    verifier = Verifier(); rows = []
    for run in args.runs:
        run = run.resolve(); config = json.loads((run/'config.json').read_text())
        identity = config['identity']; cfg = identity['config']
        assert identity['steps'] == 1000000 and identity['episodes'] == 50
        assert identity['profile'].startswith('bounded_v2')
        assert not (run/'stopped.json').exists()
        for path, digest in config['source_sha256'].items():
            assert verifier.digest(path) == digest, path
        done = json.loads((run/'completed.json').read_text())
        assert done['step'] == done['q_updates'] == 1000000
        assert done['v_updates'] == 2000000
        records = [json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
        scores, pins = {}, {}
        for step in (800000,900000,1000000):
            found = [r for r in records if r['step'] == step]; assert len(found) == 1
            row = found[0]
            assert row['n_episodes'] == 50 and row['first_episode_seed'] == 42
            assert row['eval_device'] == 'cpu'
            assert row['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
            for field in ('episode_successes','episode_rewards','episode_lengths'):
                assert len(row[field]) == 50 and np.isfinite(row[field]).all()
            assert set(row['episode_successes']) <= {0.,1.}
            assert abs(statistics.mean(row['episode_successes'])-row['success_rate']) < 1e-12
            pins[str(step)] = verifier.checkpoint(run/f'checkpoint_{step}.pkl', config, step)
            scores[str(step)] = row['success_rate']
        probes = [json.loads(line) for line in (run/'diagnostics.jsonl').read_text().splitlines()]
        assert probes and all(p['finite'] for p in probes) and probes[-1]['step'] == 1000000
        rows.append(dict(run=run.name, environment=identity['environment'], kind=identity['kind'],
            profile=identity['profile'], learner_seed=identity['seed'], discount=cfg['discount'],
            expectile=cfg['expectile'], config=cfg, per_step=scores,
            last3_mean=statistics.mean(scores.values()), all_observed_probes_finite=True,
            last_probe={k:probes[-1].get(k) for k in ('q_rms','v_rms','q_centered_rms','td_rms','score_rms','q_rms_over_return_bound')},
            checkpoint_sha256=pins, config_sha256=verifier.digest(run/'config.json'),
            evaluation_sha256=verifier.digest(run/'evaluations.jsonl'),
            diagnostics_sha256=verifier.digest(run/'diagnostics.jsonl')))
    result = dict(time=time.time(), status='complete', rows=rows,
        actual_unique_late_checkpoints_verified=len(verifier.checkpoints),
        source_sha256=verifier.digest(__file__), goal_complete=False,
        limits=['Pilot rows are not a nine-task or independent-confirmation average.',
            'No peak evaluation or environment-wise winner selection.',
            'Legacytd_rms omits target projection; for finite-backup agents it also uses an unused learned V.',
            'Samplewise score_rms does not measure the conditional expectile residual or B_k.',
            'Finite probes and finite saved states do not guarantee correct action ranking or policy improvement.'])
    tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2,allow_nan=False)); tmp.replace(args.output)
    print(json.dumps(dict(rows=[dict(run=r['run'],last3=r['last3_mean']) for r in rows],actual_models=len(verifier.checkpoints))),flush=True)


if __name__ == '__main__':
    main()
