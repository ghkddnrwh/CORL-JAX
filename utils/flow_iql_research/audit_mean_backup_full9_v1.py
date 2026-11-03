"""Require all nine uniform mean-backup runs before reporting their macro."""
import argparse
import json
from pathlib import Path
import statistics
import time

from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier, provided_thresholds
from utils.flow_iql_research.audit_fresh_ensemble_baselines_v1 import last3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    plan = json.loads(args.registration.read_text())
    root = Path(plan['root'])
    verifier = Verifier()
    primary = root / 'independent_critic_fresh_confirmation_registration_v1.json'
    assert verifier.digest(primary) == plan['primary_confirmation_plan_sha256']
    criterion, criterion_hash = provided_thresholds(Path(plan['criterion_file']))
    assert criterion_hash == json.loads(primary.read_text())['criterion_sha256']
    assert max(criterion.values()) == plan['threshold'] == .47
    assert len(plan['all9jobs']) == len({j['args']['env'] for j in plan['all9jobs']}) == 9
    for source, digest in plan['source_sha256'].items():
        assert verifier.digest(source) == digest, source
    rows = []
    for job in plan['all9jobs']:
        run = root / 'runs' / job['id']
        expected = job['args']
        config_path = run / 'config.json'
        if not config_path.exists():
            rows.append(dict(environment=expected['env'], run=run.name,
                status='not_started', complete=False, last3_mean=None))
            continue
        config = json.loads(config_path.read_text())
        identity = config['identity']
        cfg = identity['config']
        assert identity['kind'] == expected['kind'] == 'ifql'
        assert identity['profile'] == expected['profile'] == 'bounded_v2_mean_empirical_backup'
        assert identity['seed'] == expected['seed'] == 0
        assert identity['steps'] == expected['steps'] == 1000000
        assert identity['episodes'] == expected['episodes'] == 50
        assert identity['environment'] == expected['env']
        assert cfg['discount'] == expected['discount'] == .999
        assert cfg['expectile'] == expected['expectile'] == .9
        assert cfg['num_samples'] == cfg['backup_samples'] == expected['samples'] == 9
        assert cfg['backup_Q_aggregation'] == 'mean'
        assert cfg['backup_kind'] == 'finite_behavior_flow_mean_target_Q_expectile'
        width = 256 if any(s in expected['env'] for s in ('cube-', 'puzzle-', 'scene-')) else 512
        assert cfg['actor_hidden_dims'] == cfg['value_hidden_dims'] == [width] * 4
        assert cfg['layer_norm'] and not cfg['actor_layer_norm']
        assert cfg['lr'] == .0003 and cfg['tau'] == .005 and cfg['batch_size'] == 256 and cfg['flow_steps'] == 10
        assert config['source_sha256'] == plan['source_sha256']
        assert not (run / 'stopped.json').exists(), f'Incomplete stopped run cannot get zero: {run}'
        if not (run / 'completed.json').exists():
            rows.append(dict(environment=expected['env'], run=run.name,
                status='incomplete', complete=False, last3_mean=None,
                config_sha256=verifier.digest(config_path)))
            continue
        done = json.loads((run / 'completed.json').read_text())
        assert done['kind'] == 'complete'
        assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
        evaluation = run / 'evaluations.jsonl'
        scores = last3([json.loads(line) for line in evaluation.read_text().splitlines()])
        hashes = {str(step): verifier.checkpoint(run / f'checkpoint_{step}.pkl', config, step)
            for step in (800000, 900000, 1000000)}
        rows.append(dict(environment=expected['env'], run=run.name, status='complete', complete=True,
            per_step=scores, last3_mean=statistics.mean(scores.values()),
            checkpoint_sha256=hashes, config_sha256=verifier.digest(config_path),
            evaluation_sha256=verifier.digest(evaluation), actual_full_states_finite=True,
            actual_Q_Adam_counts=[800000, 900000, 1000000], actual_V_Adam_counts=[1600000, 1800000, 2000000]))
    count = sum(row['complete'] for row in rows)
    macro = statistics.mean(row['last3_mean'] for row in rows) if count == 9 else None
    result = dict(time=time.time(), status='complete' if count == 9 else 'incomplete', completed_tasks=count,
        rows=rows, nine_task_mean=macro, threshold=plan['threshold'],
        qualifies_for_independent_confirmation=macro > plan['threshold'] if macro is not None else None,
        actual_unique_late_checkpoints_verified=len(verifier.checkpoints),
        registration_sha256=verifier.digest(args.registration), criterion_sha256=criterion_hash,
        source_sha256=verifier.digest(__file__), goal_complete=False,
        interpretation='One uniform mean-head finite-backup rule, no per-environment winner mixture. Missing/stopped cells are not filled with zero.',
        limits=['Seed0 full9 is discovery only; independent training confirmation is required before a final performance claim.',
            'This is separate from the locked learned-V/independent3rank policy and does not alter its recipe or gates.',
            'Auxiliary learnedV fits do not enter the finite backup. Legacy learnedV td_rms is not its true target fitting residual.'])
    if count == 9:
        assert len(verifier.checkpoints) == 27
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(args.output)
    print(json.dumps({k: result[k] for k in ('status', 'completed_tasks', 'nine_task_mean',
        'qualifies_for_independent_confirmation', 'actual_unique_late_checkpoints_verified')}), flush=True)


if __name__ == '__main__':
    main()
