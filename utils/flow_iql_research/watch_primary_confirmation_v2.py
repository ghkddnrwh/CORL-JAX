"""Conditionally promote one registered global recipe, then audit fresh seeds.

No peak checkpoint, per-task recipe, zero filling, or holdout-based adaptation.
Queue writers must hold queue_mutation.lock while this controller is enabled.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier, provided_thresholds


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)


def episode_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    selected = []
    for step in (800000, 900000, 1000000):
        found = [record for record in records if record['step'] == step]
        assert len(found) == 1, (path, step)
        record = found[0]
        assert record['n_episodes'] == 50 and record['first_episode_seed'] == 42
        assert record['eval_device'] == 'cpu'
        assert record['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
        for key in ('episode_successes', 'episode_rewards', 'episode_lengths'):
            assert len(record[key]) == 50 and np.isfinite(record[key]).all()
        assert set(record['episode_successes']) <= {0., 1.}
        assert abs(statistics.mean(record['episode_successes'])-record['success_rate']) < 1e-12
        selected.append(record)
    return selected


def audit_seed(root, plan, seed, jobs, verifier):
    rows = []
    for job in jobs:
        run = root/'runs'/job['id']
        if (run/'stopped.json').exists():
            return dict(seed=seed, state='ineligible_stopped', run=job['id'], nine_task_mean=None)
        if not (run/'completed.json').exists():
            continue
        config = json.loads((run/'config.json').read_text())
        identity = config['identity']; environment = job['args']['env']
        assert identity['seed'] == seed and identity['environment'] == environment
        assert identity['kind'] == 'ifql' and identity['profile'] == plan['profile']
        assert identity['steps'] == 1000000 and identity['episodes'] == 50
        assert identity['training_device'] == 'gpu'
        assert identity['config'] == plan['environment_configs'][environment]
        assert config['source_sha256'] == plan['training_source_sha256']
        done = json.loads((run/'completed.json').read_text())
        assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
        records = episode_records(run/'evaluations.jsonl')
        checkpoints = []
        for record in records:
            path = run/f"checkpoint_{record['step']}.pkl"
            checkpoints.append(dict(step=record['step'], sha256=verifier.checkpoint(path, config, record['step'])))
        rows.append(dict(environment=environment, run=job['id'],
            last3_mean=statistics.mean(record['success_rate'] for record in records),
            per_step={record['step']:record['success_rate'] for record in records},
            evaluation_sha256=verifier.digest(run/'evaluations.jsonl'), checkpoints=checkpoints))
    assert len({row['environment'] for row in rows}) == len(rows)
    return dict(seed=seed, state='complete' if len(rows) == 9 else 'incomplete',
                complete_tasks=len(rows), rows=rows,
                nine_task_mean=statistics.mean(row['last3_mean'] for row in rows) if len(rows) == 9 else None)


def enqueue(root, jobs):
    with (root/'queue_mutation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'queue.json'; original = path.read_bytes()
        queue = json.loads(original); existing = {job['id']:job for job in queue}
        added = []; added_jobs = []
        for job in jobs:
            if job['id'] in existing:
                assert existing[job['id']] == job
            else:
                assert not (root/'runs'/job['id']/'config.json').exists()
                added_jobs.append(job); added.append(job['id'])
        assert path.read_bytes() == original, 'Uncooperative concurrent queue writer; retry after inspection'
        if added:
            atomic(path, added_jobs+queue)
        return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--watch-promote', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    plan = json.loads(args.plan.read_text()); out = root/'analysis'/plan['output_directory']
    out.mkdir(exist_ok=True)
    lock = (out/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verifier = Verifier()
    while True:
        for source, digest in {**plan['training_source_sha256'], **plan['audit_source_sha256']}.items():
            assert verifier.digest(source) == digest, source
        thresholds, criterion_sha = provided_thresholds(Path(plan['criterion_file']))
        assert criterion_sha == plan['criterion_sha256']
        references = {}
        for kind, runs in plan['fresh_reference_runs'].items():
            scores = []
            for name, expected_hash in runs.items():
                path = root/'runs'/name/'evaluations.jsonl'
                assert verifier.digest(path) == expected_hash
                scores.append(statistics.mean(record['success_rate'] for record in episode_records(path)))
            assert len(scores) == 9
            references[kind] = statistics.mean(scores)
        threshold = max(*thresholds.values(), *references.values())
        exploratory = audit_seed(root, plan, 0, plan['exploratory_jobs'], verifier)
        record = dict(time=time.time(), status='waiting_complete_seed0', exploratory=exploratory,
            threshold=threshold, provided_thresholds=thresholds, fresh_reference_means=references,
            independent_seeds=plan['independent_seeds'], promotion_enabled=args.watch_promote,
            goal_complete=False, scope='One registered common nine-task recipe; no missing-task mean.')
        if exploratory['state'] == 'ineligible_stopped':
            record['status'] = 'candidate_ineligible_stopped'
        elif exploratory['state'] == 'complete':
            if exploratory['nine_task_mean'] <= threshold:
                record['status'] = 'candidate_did_not_qualify'
            elif not args.watch_promote:
                record['status'] = 'qualified_audit_only_no_queue_mutation'
            else:
                jobs = [job for cohort in plan['holdout_jobs'].values() for job in cohort]
                record['newly_enqueued_jobs'] = enqueue(root, jobs)
                receipt = out/'promotion_receipt.json'
                if not receipt.exists():
                    atomic(receipt, dict(time=time.time(), exploratory=exploratory,
                        threshold=threshold, jobs=jobs, plan_sha256=verifier.digest(args.plan)))
                record['promotion_receipt'] = str(receipt)
                independent = [audit_seed(root, plan, seed, plan['holdout_jobs'][str(seed)], verifier)
                               for seed in plan['independent_seeds']]
                record.update(status='independent_confirmation_running', independent=independent)
                if any(seed['state'] == 'ineligible_stopped' for seed in independent):
                    record['status'] = 'independent_confirmation_ineligible_stopped'
                elif all(seed['state'] == 'complete' for seed in independent):
                    scores = [seed['nine_task_mean'] for seed in independent]
                    record.update(status='independent_confirmation_complete',
                        independent_nine_task_mean=statistics.mean(scores),
                        independent_seed_sample_SD=statistics.stdev(scores),
                        independent_exceeds_threshold=statistics.mean(scores) > threshold)
        atomic(out/'status.json', record)
        terminal = record['status'] in ('candidate_ineligible_stopped', 'candidate_did_not_qualify',
                    'independent_confirmation_ineligible_stopped', 'independent_confirmation_complete')
        if not args.watch_promote or terminal:
            print(json.dumps(record), flush=True); return
        time.sleep(30)


if __name__ == '__main__':
    main()
