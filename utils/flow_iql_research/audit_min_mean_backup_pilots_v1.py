"""Compare the fixed min/mean target-Q pilot controllers and actual BC identity.

This is a separate finite-backup branch. It never changes the locked learned-V
independent-critic policy, trains a model, or combines per-environment winners.
"""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import time

from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.audit_fresh_ensemble_baselines_v1 import last3
import jax
import numpy as np


def compare_arrays(first, second):
    a, ta = jax.tree_util.tree_flatten(first)
    b, tb = jax.tree_util.tree_flatten(second)
    assert ta == tb and len(a) == len(b)
    assert all(np.asarray(x).shape == np.asarray(y).shape for x, y in zip(a, b))
    return dict(exact=all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)),
        max_absolute_difference=max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
            for x, y in zip(a, b)), arrays=len(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.root.resolve()
    verifier = Verifier()
    rows, evidence = [], []
    for short in ('ant', 'cubed', 'pz3'):
        runs = [root / 'runs' / f'{prefix}_{short}_{profile}_g999_t90_s0' for prefix, profile in (
            ('eb1', 'bounded_v2_empirical_backup'), ('meb1', 'bounded_v2_mean_empirical_backup'))]
        configs = [json.loads((run / 'config.json').read_text()) for run in runs]
        for run, conf in zip(runs, configs):
            identity = conf['identity']
            assert identity['kind'] == 'ifql' and identity['seed'] == 0
            assert identity['steps'] == 1000000 and identity['episodes'] == 50
            assert not (run / 'stopped.json').exists()
            assert identity['config']['num_samples'] == identity['config']['backup_samples'] == 9
            assert identity['config']['discount'] == .999 and identity['config']['expectile'] == .9
            for source, digest in conf['source_sha256'].items():
                assert verifier.digest(source) == digest, source
            evidence.append(dict(run=str(run), config_sha256=verifier.digest(run / 'config.json')))
        assert configs[0]['identity']['environment'] == configs[1]['identity']['environment']
        common = [{k: v for k, v in c['identity']['config'].items()
            if k not in ('backup_kind', 'backup_Q_aggregation')} for c in configs]
        assert common[0] == common[1]
        assert configs[0]['identity']['profile'] == 'bounded_v2_empirical_backup'
        assert configs[1]['identity']['profile'] == 'bounded_v2_mean_empirical_backup'
        assert configs[0]['identity']['config']['backup_kind'] == 'finite_behavior_flow_target_Q_expectile'
        assert configs[1]['identity']['config']['backup_kind'] == 'finite_behavior_flow_mean_target_Q_expectile'
        assert configs[1]['identity']['config']['backup_Q_aggregation'] == 'mean'
        complete = all((run / 'completed.json').exists() for run in runs)
        cells = []
        for step in (800000, 900000, 1000000):
            paths = [run / f'checkpoint_{step}.pkl' for run in runs]
            if not all(path.exists() for path in paths):
                continue
            hashes = [verifier.checkpoint(path, conf, step) for path, conf in zip(paths, configs)]
            saved = []
            for path in paths:
                with path.open('rb') as stream:
                    saved.append(pickle.load(stream))
            params = [s['state']['agent']['network']['params'] for s in saved]
            behavior = compare_arrays(params[0]['modules_actor_flow'], params[1]['modules_actor_flow'])
            critic = compare_arrays(params[0]['modules_critic'], params[1]['modules_critic'])
            assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(saved[0]['replay_rng'], saved[1]['replay_rng']))
            assert np.array_equal(saved[0]['state']['agent']['rng'], saved[1]['state']['agent']['rng'])
            cells.append(dict(step=step, checkpoint_sha256=dict(min=hashes[0], mean=hashes[1]),
                BC=behavior, critic=critic, replay_RNG_exact=True, agent_RNG_exact=True,
                actual_Q_Adam_count_per_model=step, actual_V_Adam_count_per_model=2 * step,
                both_actual_full_states_finite=True))
            del saved, params
        scores = None
        if complete:
            assert len(cells) == 3
            for run in runs:
                done = json.loads((run / 'completed.json').read_text())
                assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
            scores = [last3([json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]) for run in runs]
            for cell in cells:
                step = str(cell['step'])
                cell.update(min_target_CPU50_success=scores[0][step], mean_target_CPU50_success=scores[1][step],
                    mean_minus_min_success=scores[1][step] - scores[0][step])
        row = dict(short=short, environment=configs[0]['identity']['environment'], complete=complete, cells=cells,
            min_target_last3=statistics.mean(scores[0].values()) if complete else None,
            mean_target_last3=statistics.mean(scores[1].values()) if complete else None,
            mean_minus_min_last3=statistics.mean(scores[1].values()) - statistics.mean(scores[0].values()) if complete else None,
            all_available_BC_parameters_exact=all(cell['BC']['exact'] for cell in cells),
            same_hyperparameters_except_target_aggregation=True)
        row['critic_only_deployment_effect_supported'] = complete and row['all_available_BC_parameters_exact']
        rows.append(row)
    count = sum(row['complete'] for row in rows)
    result = dict(time=time.time(), status='complete' if count == 3 else 'incomplete',
        rows=rows, completed_pilots=count, actual_unique_late_checkpoints_verified=len(verifier.checkpoints),
        pilot_three_task_min_mean=statistics.mean(row['min_target_last3'] for row in rows) if count == 3 else None,
        pilot_three_task_mean_mean=statistics.mean(row['mean_target_last3'] for row in rows) if count == 3 else None,
        nine_task_mean=None, evidence=evidence, source_sha256=verifier.digest(__file__),
        primary_candidate_unchanged='learned-V IQLV2 plus BCflow and independent3meanranks', goal_complete=False,
        limits=['These are three fixed pilot environments and one training seed, not a nine-task competitor average.',
            'Exact BC parameters plus unchanged inference recipe support a critic-source explanation. Non-exact BC requires further fixed-actor interventions.',
            'Even when BC matches, different Q parameters and observed returns do not identify true Q/residual/B_k errors.',
            'Only target proposal-head min versus mean and accompanying diagnostics differ in the controlled learning branches.',
            'Learned V is an unused auxiliary fitting control in both finite-backup branches; generic learned-V td_rms is not their actual target fitting residual.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], completed_pilots=count,
        actual_models=len(verifier.checkpoints), rows=[{k: row[k] for k in (
            'short', 'complete', 'min_target_last3', 'mean_target_last3',
            'all_available_BC_parameters_exact', 'critic_only_deployment_effect_supported')} for row in rows])), flush=True)


if __name__ == '__main__':
    main()
