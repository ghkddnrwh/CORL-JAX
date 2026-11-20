"""Audit the pre-registered paired fresh-ensemble baselines.

Uses already-planned single-member/ensemble evaluations. No training or policy
evaluation is launched, and no method/member/checkpoint maximization is allowed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

from utils.flow_iql_research import core  # SciPy compatibility before pickle.
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.audit_independent_critic_policy_v1 import audit_job as audit_V1
from utils.flow_iql_research.audit_independent_critic_policy_v2 import audit_job as audit_V2
from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job
import numpy as np


STEPS = (800000, 900000, 1000000)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def last3(records):
    result = {}
    for step in STEPS:
        found = [r for r in records if r['step'] == step]
        assert len(found) == 1, ('Missing/duplicate late evaluation', step)
        record = found[0]
        assert record['n_episodes'] == 50 and record['first_episode_seed'] == 42
        assert record['eval_device'] == 'cpu'
        assert record['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
        for name in ('episode_successes', 'episode_rewards', 'episode_lengths'):
            assert len(record[name]) == 50 and np.isfinite(record[name]).all()
        assert set(record['episode_successes']) <= {0., 1.}
        assert all(int(n) == n and n > 0 for n in record['episode_lengths'])
        assert abs(statistics.mean(record['episode_successes']) - record['success_rate']) < 1e-12
        assert abs(statistics.mean(record['episode_rewards']) - record['reward_mean']) < 1e-9
        result[str(step)] = record['success_rate']
    return result


def audit_member(run, expected_identity, expected_sources, verifier):
    path = run / 'config.json'
    if not path.exists():
        return dict(run=run.name, status='not_started', last3_mean=None)
    config = json.loads(path.read_text())
    identity = config['identity']
    assert identity == expected_identity, f'Unregistered member identity: {run}'
    assert config['source_sha256'] == expected_sources
    assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
    assert identity['steps'] == 1000000 and identity['episodes'] == 50
    assert identity['config']['num_samples'] == 9 and identity['config']['flow_steps'] == 10
    assert identity['config']['discount'] == .999 and identity['config']['expectile'] == .9
    for source, digest in expected_sources.items():
        assert verifier.digest(source) == digest, source
    assert not (run / 'stopped.json').exists(), f'Stopped member is ineligible: {run}'
    if not (run / 'completed.json').exists():
        return dict(run=run.name, status='incomplete', last3_mean=None,
            config_sha256=verifier.digest(path), exact_registered_identity_and_sources=True)
    done = json.loads((run / 'completed.json').read_text())
    assert done['kind'] == 'complete'
    assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
    evaluation = run / 'evaluations.jsonl'
    scores = last3([json.loads(line) for line in evaluation.read_text().splitlines()])
    checkpoints = {str(step): verifier.checkpoint(run / f'checkpoint_{step}.pkl', config, step)
        for step in STEPS}
    diagnostics = run / 'diagnostics.jsonl'
    probes = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert probes and probes[-1]['step'] == 1000000
    assert all(r['finite'] and r['q_updates'] == r['step'] and r['v_updates'] == 2 * r['step'] for r in probes)
    return dict(run=run.name, status='complete', environment=identity['environment'], learner_seed=identity['seed'],
        per_step=scores, last3_mean=statistics.mean(scores.values()),
        config_sha256=verifier.digest(path), evaluation_sha256=verifier.digest(evaluation),
        checkpoint_sha256=checkpoints, diagnostic_sha256=verifier.digest(diagnostics),
        numerical_probes=dict(observed_probes=len(probes), all_observed_state_probes_finite=True,
            max_fixed_query_Q_RMS_over_known_return_bound=max(r['q_rms_over_return_bound'] for r in probes),
            max_fixed_query_Q_above_upper_RMS=max(r['q_above_upper_rms'] for r in probes),
            max_fixed_query_Q_below_lower_RMS=max(r['q_below_lower_rms'] for r in probes)),
        exact_registered_identity_and_sources=True)


def audit(contract, verifier):
    for source, digest in contract['source_sha256'].items():
        assert verifier.digest(source) == digest, source
    for field in ('analysis_plan', 'training_plan', 'mean_rank_plan', 'raw_control_plan'):
        assert verifier.digest(contract[field]) == contract[field + '_sha256']
    analysis = json.loads(Path(contract['analysis_plan']).read_text())
    plan = json.loads(Path(contract['training_plan']).read_text())
    mean_plan = json.loads(Path(contract['mean_rank_plan']).read_text())
    raw_plan = json.loads(Path(contract['raw_control_plan']).read_text())
    assert analysis['groups'] == plan['fresh_groups'] == [[3, 4, 5], [6, 7, 8]]
    assert analysis['first_BC_seeds'] == [3, 6]
    assert analysis['registration_before_any_fresh_800K_outcomes']
    assert len(plan['environments']) == len(set(plan['environments'])) == 9
    root = Path(plan['root'])
    old = json.loads(Path(plan['discovery_plan']).read_text())
    assert verifier.digest(plan['discovery_plan']) == plan['discovery_plan_sha256']
    members, errors = {}, []
    for job in plan['training_jobs']:
        try:
            members[job['id']] = audit_member(root / 'runs' / job['id'],
                plan['expected_identities'][job['id']], plan['training_source_sha256'], verifier)
        except (AssertionError, KeyError, ValueError, OSError) as error:
            errors.append(dict(member=job['id'], error=repr(error)))
            members[job['id']] = dict(run=job['id'], status='invalid', last3_mean=None)
    assert len(members) == 54
    controls = dict(
        mean_rank=(mean_plan['output_directory'], mean_plan, audit_V2),
        median_rank=(plan['output_directory'], old, audit_V1),
        mean=(raw_plan['output'], raw_plan, audit_V2))
    groups, ensemble_records = [], []
    for index, seeds in enumerate(plan['fresh_groups']):
        tasks = []
        for env in plan['environments']:
            matching = [j for j in plan['training_jobs'] if j['args']['env'] == env and j['args']['seed'] in seeds]
            matching.sort(key=lambda j: seeds.index(j['args']['seed']))
            assert len(matching) == 3
            own = [members[j['id']] for j in matching]
            all_members_complete = all(row['status'] == 'complete' for row in own)
            baseline = {}
            if all_members_complete:
                first = own[0]['per_step']
                uniform = {str(s): statistics.mean(row['per_step'][str(s)] for row in own) for s in STEPS}
                baseline = dict(first_BC_own_single_critic=first, uniform_member_expected_return=uniform)
            policy = {}
            for rule, (directory, evidence, check) in controls.items():
                scores = {}
                parents = []
                for step in STEPS:
                    try:
                        job = ready_job(root, dict(plan, output_directory=directory), index, env, step, rule, verifier)
                        if job is None or not Path(job['output']).exists():
                            continue
                        actual = check(evidence, job, verifier)
                        assert actual['learner_seeds'] == seeds
                        scores[str(step)] = actual['success']
                        parents.append(actual['parent_learners_complete'])
                        ensemble_records.append(dict(id=job['id'], rule=rule, group=index + 1,
                            result_sha256=actual['result_sha256']))
                    except (AssertionError, KeyError, ValueError, OSError) as error:
                        errors.append(dict(group=index + 1, environment=env, rule=rule, step=step, error=repr(error)))
                complete = len(scores) == 3 and all(parents) and all_members_complete
                policy[rule] = dict(complete=complete, per_step=scores,
                    last3_mean=statistics.mean(scores.values()) if complete else None)
            rows = {name: dict(complete=all_members_complete, per_step=values,
                last3_mean=statistics.mean(values.values())) for name, values in baseline.items()}
            for name in ('first_BC_own_single_critic', 'uniform_member_expected_return'):
                if name not in rows:
                    rows[name] = dict(complete=False, per_step={}, last3_mean=None)
            tasks.append(dict(environment=env, member_runs=[j['id'] for j in matching],
                all_members_complete=all_members_complete, baselines=rows, ensemble_policies=policy))
        summaries = []
        names = ['first_BC_own_single_critic', 'uniform_member_expected_return', 'mean_rank', 'median_rank', 'mean']
        for name in names:
            rows = [t['baselines'].get(name, t['ensemble_policies'].get(name)) for t in tasks]
            count = sum(row['complete'] for row in rows)
            summaries.append(dict(method=name, completed_tasks=count,
                nine_task_mean=statistics.mean(row['last3_mean'] for row in rows) if count == 9 and not errors else None,
                macro_per_step={str(s): statistics.mean(row['per_step'][str(s)] for row in rows) for s in STEPS}
                    if count == 9 and not errors else None))
        contrasts = []
        pairs = [(policy, baseline) for policy in ('mean_rank', 'median_rank')
            for baseline in ('first_BC_own_single_critic', 'uniform_member_expected_return')]
        pairs += [('mean_rank', 'mean'), ('median_rank', 'mean')]
        for policy, baseline in pairs:
            first = next(s for s in summaries if s['method'] == policy)
            second = next(s for s in summaries if s['method'] == baseline)
            complete = first['nine_task_mean'] is not None and second['nine_task_mean'] is not None
            contrasts.append(dict(policy=policy, baseline=baseline, complete=complete,
                nine_task_mean_difference=first['nine_task_mean'] - second['nine_task_mean'] if complete else None,
                macro_per_step_difference={str(s): first['macro_per_step'][str(s)] - second['macro_per_step'][str(s)]
                    for s in STEPS} if complete else None))
        groups.append(dict(group=index + 1, learner_seeds=seeds, first_BC_seed=seeds[0],
            tasks=tasks, summaries=summaries, paired_contrasts=contrasts))
    pooled = []
    for name in names:
        summaries = [next(s for s in g['summaries'] if s['method'] == name) for g in groups]
        complete = all(s['nine_task_mean'] is not None for s in summaries) and not errors
        pooled.append(dict(method=name, complete=complete,
            fresh_two_group_mean=statistics.mean(s['nine_task_mean'] for s in summaries) if complete else None,
            between_group_sample_SD=statistics.stdev(s['nine_task_mean'] for s in summaries) if complete else None))
    pooled_contrasts = []
    for policy, baseline in pairs:
        matching = [next(c for c in g['paired_contrasts'] if c['policy'] == policy and c['baseline'] == baseline)
            for g in groups]
        complete = all(c['complete'] for c in matching) and not errors
        pooled_contrasts.append(dict(policy=policy, baseline=baseline, complete=complete,
            paired_fresh_two_group_mean_difference=statistics.mean(c['nine_task_mean_difference'] for c in matching)
                if complete else None))
    complete = all(row['complete'] for row in pooled) and not errors
    return dict(time=time.time(), status='audit_errors' if errors else 'complete' if complete else 'incomplete',
        members=members, completed_members=sum(m['status'] == 'complete' for m in members.values()),
        groups=groups, pooled=pooled, paired_contrasts=pooled_contrasts,
        errors=errors, ensemble_records=ensemble_records,
        actual_unique_late_models_verified=len(verifier.checkpoints), additional_GPU_jobs=0,
        additional_CPU_evaluations=0, primary_candidate_unchanged='mean_rank', goal_complete=False,
        limits=analysis['limits'] + [
            'Numerical probe finiteness/value-bound violations are not true conditional zeta or propagated B_k labels.',
            'Checkpoint fluctuation and two-group SD are descriptive, not uncertainty intervals or proof of universal stability.',
            'Raw mean is a fixed control, not an alternative eligible for candidate selection on these heldout cohorts.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contract', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = json.loads(args.contract.read_text())
    verifier = Verifier()
    while True:
        result = audit(contract, verifier)
        result.update(pid=os.getpid(), contract_sha256=verifier.digest(args.contract),
            source_sha256=verifier.digest(__file__))
        atomic(args.output / 'status.json', result)
        if result['status'] in ('complete', 'audit_errors') or not args.watch:
            atomic(args.output / 'audit.json', result)
            print(json.dumps({k: result[k] for k in ('status', 'completed_members',
                'actual_unique_late_models_verified', 'errors')}), flush=True)
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
