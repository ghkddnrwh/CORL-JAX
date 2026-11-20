"""Read actual four-seed/nine-task models and stored CPU50 evaluations.

This audit performs no updates, rollouts, queue mutations, or timing. Fresh
identities come from the immutable attempt4 receipt, not the seed0 registry.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from utils.flow_iql_research.watch_onestep_deploy_v12 import Verifier, full_confirmation
from utils.flow_iql_research.routed_policy_verifier_v1 import PolicyVerifier
from utils.flow_iql_research.routed_policy_registry_v1 import NATIVE, POLICY, MODE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    area = root / 'analysis/deployment_interim_review_v15'
    proposal_path = area / 'actual_full_confirmation_file_audit_plan_v1.json'
    proposal = json.loads(proposal_path.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, digest in proposal['source_sha256'].items():
        assert sha(path) == digest, path
    registry_path = root / 'registration_v19.json'
    fresh_path = root / 'analysis/onestep_pipeline_v1/registrations/confirmation_attempt4_fresh.json'
    assert sha(registry_path) == proposal['registration_sha256']
    assert sha(fresh_path) == proposal['fresh_receipt_sha256']
    plan = json.loads(registry_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    assert fresh['source_contract_sha256'] == proposal['execution_contract_sha256']
    state = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    attempt = state['attempts'][state['active_attempt']]
    assert attempt['number'] == 4 and attempt['recipe'] == POLICY
    assert attempt['fresh_seeds'] == [24, 25, 26]
    original = [j for j in plan['jobs'] if j['args']['recipe'] == NATIVE and j['args']['seed'] == 0]
    jobs = original + fresh['new_jobs']
    assert len(original) == 9 and len(fresh['new_jobs']) == 27
    assert len(jobs) == len({j['id'] for j in jobs}) == 36
    authorized = {**{j['id']: plan['expected_identities'][j['id']] for j in original},
                  **fresh['expected_identities']}
    current_jobs = {j['id']: j for j in state['jobs']}
    for job in jobs:
        for key, value in job.items():
            assert current_jobs[job['id']][key] == value
        assert state['identities'][job['id']] == authorized[job['id']]
    seeds = [0, 24, 25, 26]
    environments = sorted({j['args']['env'] for j in original})
    assert len(environments) == 9
    assert all(sorted(j['args']['env'] for j in jobs if j['args']['seed'] == seed) == environments for seed in seeds)
    verifier = Verifier(plan)
    policy_verifier = PolicyVerifier(plan, verifier, sha)
    points, task_rows, gate_rows, probes = [], [], [], []
    configs_by_environment = {}
    for job in sorted(jobs, key=lambda j: (j['args']['seed'], j['args']['env'])):
        rid, seed, environment = job['id'], job['args']['seed'], job['args']['env']
        run = root / 'runs' / rid
        conf = verifier.config(run, authorized[rid])
        assert conf is not None and conf['identity']['recipe'] == NATIVE
        config = conf['identity']['config']
        if environment in configs_by_environment:
            assert config == configs_by_environment[environment]
        else:
            configs_by_environment[environment] = config
        assert config['discount'] == .999 and config['expectile'] == .9
        assert config['flow_steps'] == config['num_samples'] + 1 == 10
        assert config['backup_samples'] == 9 and config['backup_Q_aggregation'] == 'mean'
        successes = {role: [] for role in ['native', 'teacher', 'routed']}
        per_step = {role: {} for role in successes}
        for step in [800000, 900000, 1000000]:
            pin = verifier.model(run, conf, step)
            routed = policy_verifier.model(run, conf, step, pin)
            metadata = routed['metadata']
            assert metadata['computed_action_heads'] == metadata['actor_forward_passes'] == 1
            assert metadata['flow_integration_steps'] == metadata['generated_candidate_actions'] == 0
            assert not metadata['requires_critic'] and not metadata['requires_BC_flow']
            results = {}
            for role, mode in [('native', 'primary'), ('teacher', 'teacher'), ('routed', MODE)]:
                path = root / f'analysis/evaluations_v1/{rid}_{step}_{mode}_n50.json'
                result = (policy_verifier.evaluation(path, job, step, routed) if role == 'routed'
                          else verifier.evaluation(path, job, step, mode, 50, pin))
                successes[role].append(result['successes'])
                per_step[role][str(step)] = result['success_rate']
                results[role] = dict(path=str(path), **result)
            points.append(dict(run=rid, seed=seed, environment=environment, step=step,
                               native_model=pin, routed_model=routed, evaluations=results))
        assert (run / 'completed.json').exists()
        raw = (run / 'diagnostics.jsonl').read_text()
        assert raw.endswith('\n')
        logged = [json.loads(line) for line in raw.splitlines()]
        assert len(logged) == 1000 and logged[-1]['step'] == 1000000
        assert all(p['finite'] and p['q_updates'] == p['actor_updates'] == p['step']
                   and p['v_updates'] == 0 for p in logged)
        probes.append(dict(run=rid, observed_probes=1000, finite_and_counters_verified=True,
                           path=str(run / 'diagnostics.jsonl'), sha256=sha(run / 'diagnostics.jsonl')))
        means = {role: float(Fraction(sum(counts), 150)) for role, counts in successes.items()}
        task_rows.append(dict(run=rid, seed=seed, environment=environment, **means,
                              routed_minus_teacher=means['routed'] - means['teacher']))
        gate_rows.append(dict(run=rid, recipe=POLICY, seed=seed, environment=environment,
                              complete=True, primary_mean=means['routed'], teacher_mean=means['teacher']))
        print(json.dumps(dict(run=rid, seed=seed, checked_last3=means)), flush=True)
    confirmation = full_confirmation(gate_rows, attempt, plan)
    assert confirmation is not None and confirmation['performance_passed']
    manual = []
    for seed in seeds:
        group = [r for r in task_rows if r['seed'] == seed]
        manual.append(dict(seed=seed, **{role: statistics.mean(r[role] for r in group)
                                        for role in ['native', 'teacher', 'routed']}))
    assert all(abs(a['routed'] - b['primary']) < 1e-12 and abs(a['teacher'] - b['teacher']) < 1e-12
               for a, b in zip(manual, confirmation['seed_means']))
    assert abs(statistics.mean(r['routed'] for r in manual[1:]) - confirmation['fresh_mean']) < 1e-12
    task_summary = []
    for environment in environments:
        group = [r for r in task_rows if r['environment'] == environment and r['seed'] != 0]
        task_summary.append(dict(environment=environment,
            **{role + '_fresh_mean': statistics.mean(r[role] for r in group) for role in ['native', 'teacher', 'routed']},
            routed_fresh_sample_SD=statistics.stdev(r['routed'] for r in group)))
    result = dict(time=time.time(), status='pass', proposal_sha256=sha(proposal_path),
        registration_sha256=sha(registry_path), fresh_receipt_sha256=sha(fresh_path),
        execution_contract_sha256=proposal['execution_contract_sha256'], source_pins_checked=len(proposal['source_sha256']),
        confirmation=confirmation, seed_means_with_native_sampler=manual, task_rows=task_rows,
        fresh_task_summary=task_summary, actual_model_points=points, training_probes=probes,
        actual_native_checkpoints=108, actual_native_actor_exports=108, actual_typed_actor_exports=108,
        stored_CPU50_points_verified=324, stored_episode_records_verified=16200,
        training_seeds=seeds, independent_fresh_training_replicates_per_environment=3,
        same_environment_config_across_all_four_training_seeds=True,
        optimizer_updates_performed=0, new_rollout_episodes=0, isolated_speed_measured_by_audit=False,
        goal_complete=False,
        scope='Complete registered nine-task last3 performance audit plus three fresh seeds. Empirical practical gates, not a population non-inferiority test. Environment regressions remain visible. Speed and final preservation audit remain required.')
    write_json(area / 'FULL_NINE_FRESH_NATIVE_MODEL_AND_EVALUATION_ROOT_AUDIT_v1.json', result)
    for name, rows in [('task_results_all_four_seeds_v1.csv', task_rows),
                       ('fresh_task_summary_v1.csv', task_summary), ('seed_means_v1.csv', manual)]:
        with (area / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(status='pass', model_points=len(points), stored_episodes=16200,
                          confirmation=confirmation)), flush=True)


if __name__ == '__main__':
    main()
