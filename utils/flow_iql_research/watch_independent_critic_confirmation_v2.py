"""Strict conditional promotion and fresh evaluation of one common ensemble rule.

No training starts until all54 discovery evaluations and81 actual member models
pass the audit. Fresh learner seeds, configs, costs and selection are preregistered.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from utils.flow_iql_research.audit_independent_critic_policy_v1 import audit, audit_job
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier, provided_thresholds
from utils.flow_iql_research.run_independent_critic_discovery_v1 import command


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)


def promote(root, plan, selected_rule, discovery, plan_sha):
    """Queue locked recipes, preserving the explicitly registered fair controls."""
    assert discovery['status'] == 'complete' and not discovery['errors']
    assert discovery['valid_evaluations'] == 54 and discovery['discovery_qualifies']
    assert discovery['selected_rule'] == selected_rule
    jobs = plan['training_jobs']; assert len(jobs) == 54
    receipt = Path(plan['output_directory'])/'promotion_receipt.json'
    with (root/'queue_mutation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved['confirmation_plan_sha256'] == plan_sha
            assert saved['selected_rule'] == selected_rule
        original = (root/'queue.json').read_bytes(); queue = json.loads(original)
        existing = {job['id']: job for job in queue}; additions = []
        for job in jobs:
            if job['id'] in existing:
                assert existing[job['id']] == job
            else:
                assert not (root/'runs'/job['id']/'config.json').exists()
                additions.append(job)
        protected = set(plan['preserve_front_control_ids'])
        front = [job for job in queue if job['id'] in protected]
        tail = [job for job in queue if job['id'] not in protected]
        assert (root/'queue.json').read_bytes() == original
        if not receipt.exists():
            atomic(receipt, dict(time=time.time(), confirmation_plan_sha256=plan_sha,
                selected_rule=selected_rule, discovery=discovery, jobs=jobs,
                rule_locked_before_fresh_training=True, newly_enqueued=[job['id'] for job in additions]))
        if additions:
            atomic(root/'queue.json', front+additions+tail)
    return receipt


def ready_job(root, plan, group_index, environment, step, rule, verifier):
    jobs = [job for job in plan['training_jobs'] if job['args']['env'] == environment
            and job['args']['seed'] in plan['fresh_groups'][group_index]]
    jobs.sort(key=lambda job: plan['fresh_groups'][group_index].index(job['args']['seed']))
    assert len(jobs) == 3
    pins = []; runs = []
    for job in jobs:
        run = root/'runs'/job['id']; cp = run/f'checkpoint_{step}.pkl'
        if (run/'stopped.json').exists():
            raise AssertionError(f'Fresh member stopped: {run}')
        if not cp.exists():
            return None
        conf = json.loads((run/'config.json').read_text())
        assert conf['identity'] == plan['expected_identities'][job['id']]
        assert conf['source_sha256'] == plan['training_source_sha256']
        pins.append(dict(run=str(run), checkpoint_sha256=verifier.checkpoint(cp, conf, step),
                         config_sha256=verifier.digest(run/'config.json')))
        runs.append(str(run))
    name = next(job['id'].split('_')[2] for job in jobs)
    key = f'group{group_index+1}_{name}_{step}_{rule}'
    return dict(id=key, environment=environment, step=step, rule=rule,
                actor_run=runs[0], critic_runs=runs, member_pins=pins,
                output=str(Path(plan['output_directory'])/(key+'.json')))


def summarize(plan, selected_rule, jobs, results, discovery):
    groups = []
    for group in range(2):
        tasks = []
        for env in plan['environments']:
            matching = [job for job in jobs.values() if job['environment'] == env
                        and job['id'].startswith(f'group{group+1}_')]
            valid = [results.get(job['id']) for job in matching]
            complete = (len(matching) == 3 and {job['step'] for job in matching} == {800000,900000,1000000}
                        and all(row is not None and row['parent_learners_complete'] for row in valid))
            tasks.append(dict(environment=env, complete=complete,
                per_step={str(job['step']): results[job['id']]['success'] if job['id'] in results else None for job in matching},
                last3_mean=statistics.mean(row['success'] for row in valid) if complete else None))
        count = sum(task['complete'] for task in tasks)
        groups.append(dict(learner_seeds=plan['fresh_groups'][group], completed_tasks=count, tasks=tasks,
            nine_task_mean=statistics.mean(task['last3_mean'] for task in tasks) if count == 9 else None))
    complete = all(group['nine_task_mean'] is not None for group in groups)
    fresh_mean = statistics.mean(group['nine_task_mean'] for group in groups) if complete else None
    all_mean = statistics.mean([discovery['selected_nine_task_mean']]+[group['nine_task_mean'] for group in groups]) if complete else None
    return dict(selected_rule=selected_rule, fresh_groups=groups, all_fresh_complete=complete,
        fresh_two_group_nine_task_mean=fresh_mean, all_three_group_nine_task_mean=all_mean,
        fresh_group_sample_SD=statistics.stdev(group['nine_task_mean'] for group in groups) if complete else None,
        fresh_exceeds_threshold=fresh_mean > plan['threshold'] if complete else None,
        all_groups_exceed_threshold=all_mean > plan['threshold'] if complete else None,
        goal_complete=False,
        interpretation='Two independent fresh ENSEMBLE groups; three learners/group; 3xbase trainingcost. Finalreviewrequired.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('plan', type=Path)
    parser.add_argument('--watch-promote', action='store_true')
    parser.add_argument('--execution-amendment', type=Path, required=True); args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); root = Path(plan['root']); out = Path(plan['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    amendment = json.loads(args.execution_amendment.read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert amendment['parent_plan_sha256'] == digest(args.plan)
    assert amendment['controller_source_sha256'] == digest(__file__)
    assert amendment['cpu_evaluation_slots'] == 4
    previous = json.loads((out/'status.json').read_text())
    assert previous['pid'] == amendment['old_controller_pid']
    assert previous['status'] == 'fresh_confirmation_running'
    assert previous['valid_evaluations'] == previous['ready_evaluations'] == 0
    assert not previous['running'] and not previous['failures']
    old_stat = Path('/proc')/str(previous['pid'])/'stat'
    assert not old_stat.exists() or old_stat.read_text().split(') ',1)[1].split()[0] == 'Z', 'Previous controller still alive'
    assert digest(out/'discovery_audit.json') == amendment['preserved_discovery_audit_sha256']
    verifier = Verifier(); plan_sha = verifier.digest(args.plan)
    discovery_path = Path(plan['discovery_plan']); discovery_plan = json.loads(discovery_path.read_text())
    assert verifier.digest(discovery_path) == plan['discovery_plan_sha256']
    assert plan['fresh_groups'] == discovery_plan['fresh_confirmation_groups'] == [[3,4,5],[6,7,8]]
    assert len({job['id'] for job in plan['training_jobs']}) == 54
    running = {}; failures = []; known = {}; validated = {}; discovery = None
    cpus = sorted(os.sched_getaffinity(0)); slots = [cpus[32+4*i:36+4*i] for i in range(4)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    evidence_plan = dict(source_sha256=discovery_plan['source_sha256'])
    while True:
        for path, digest in {**plan['training_source_sha256'], **plan['controller_source_sha256']}.items():
            assert verifier.digest(path) == digest, path
        thresholds, digest = provided_thresholds(Path(plan['criterion_file']))
        assert digest == plan['criterion_sha256'] and max(thresholds.values()) == plan['threshold'] == .47
        if discovery is None or discovery['status'] != 'complete':
            discovery = audit(discovery_plan, verifier)
            saved_discovery = json.loads((out/'discovery_audit.json').read_text())
            assert {k:v for k,v in discovery.items() if k != 'time'} == {k:v for k,v in saved_discovery.items() if k != 'time'}
            discovery = saved_discovery  # Keep the original frozen discovery audit byte-for-byte.

        base = dict(time=time.time(), pid=os.getpid(), promotion_enabled=args.watch_promote,
            confirmation_plan_sha256=plan_sha, discovery=discovery, failures=failures, goal_complete=False,
            execution_amendment_sha256=digest(args.execution_amendment),
            actual_controller_source_sha256=digest(__file__), cpu_evaluation_slots=4)
        if discovery['errors']:
            atomic(out/'status.json', base | dict(status='discovery_audit_errors')); return
        if discovery['status'] != 'complete':
            atomic(out/'status.json', base | dict(status='waiting_full54_discovery'))
            if not args.watch_promote: return
            time.sleep(30); continue
        if not discovery['discovery_qualifies']:
            atomic(out/'status.json', base | dict(status='discovery_did_not_qualify_no_fresh_jobs')); return
        rule = discovery['selected_rule']
        if not args.watch_promote:
            atomic(out/'status.json', base | dict(status='qualified_audit_only_no_queue_mutation')); return
        receipt = promote(root, plan, rule, discovery, plan_sha)
        for key, record in list(running.items()):
            code = record['process'].poll()
            if code is None: continue
            record['log'].close(); del running[key]
            if code:
                failures.append(dict(id=key, exit_code=code, time=time.time()))
        if failures:
            # Other owned evaluations finish normally; do not kill or automatically restart.
            for record in running.values(): record['process'].wait(); record['log'].close()
            atomic(out/'status.json', base | dict(status='evaluation_failed_requires_inspection',failures=failures)); return
        for group in range(2):
            for env in plan['environments']:
                for step in (800000,900000,1000000):
                    job = ready_job(root, plan, group, env, step, rule, verifier)
                    if job is not None:
                        if job['id'] in known: assert known[job['id']] == job
                        known[job['id']] = job
        atomic(out/'prepared_evaluation_jobs.json', list(known.values()))
        for key, job in known.items():
            if Path(job['output']).exists():
                validated[key] = audit_job(evidence_plan, job, verifier)
        occupied = {record['slot'] for record in running.values()}
        for slot in range(4):
            if slot in occupied: continue
            job = next((job for key,job in known.items() if key not in running and key not in validated),None)
            if job is None: break
            log = (out/(job['id']+'.log')).open('a')
            process = subprocess.Popen(command(job,slots[slot]), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            running[job['id']] = dict(process=process,log=log,slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        result = summarize(plan, rule, known, validated, discovery)
        atomic(out/'status.json', base | result | dict(status='fresh_confirmation_complete' if result['all_fresh_complete'] else 'fresh_confirmation_running',
            promotion_receipt=str(receipt), ready_evaluations=len(known), valid_evaluations=len(validated),
            running={key:dict(pid=record['process'].pid,slot=record['slot'],process_identity=record['process_identity']) for key,record in running.items()}))
        if result['all_fresh_complete']: return
        time.sleep(30)


if __name__ == '__main__': main()
