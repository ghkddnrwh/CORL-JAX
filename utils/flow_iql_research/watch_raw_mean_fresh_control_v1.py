"""Paired raw-Q-mean control for the locked rank-mean fresh confirmation.

No training or primary recipe/gate changes. This control is not an alternative
candidate eligible for promotion based on the same heldout outcomes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from utils.flow_iql_research.audit_independent_critic_policy_v1 import audit as discovery_audit
from utils.flow_iql_research.audit_independent_critic_policy_v2 import audit_job
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier, provided_thresholds
from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job, summarize
from utils.flow_iql_research.run_independent_critic_factorial_v1 import command


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('registration', type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    out = Path(registration['output'])
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verifier = Verifier()
    assert verifier.digest(registration['training_plan']) == registration['training_plan_sha256']
    plan = json.loads(Path(registration['training_plan']).read_text())
    root = Path(plan['root'])
    discovery_path = Path(plan['discovery_plan'])
    assert verifier.digest(discovery_path) == plan['discovery_plan_sha256']
    original = discovery_audit(json.loads(discovery_path.read_text()), verifier)
    assert original['status'] == 'complete' and original['valid_evaluations'] == 54 and not original['errors']
    raw = next(row for row in original['methods'] if row['rule'] == 'mean')
    assert raw['nine_task_mean'] == registration['raw_mean_discovery_mean']
    atomic(out / 'discovery_control_audit.json', original)
    plan['output_directory'] = str(out)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    known, valid, running, failures = {}, {}, {}, []
    while True:
        for source, digest in registration['source_sha256'].items():
            assert verifier.digest(source) == digest, source
        thresholds, criterion_hash = provided_thresholds(Path(plan['criterion_file']))
        assert criterion_hash == plan['criterion_sha256'] and max(thresholds.values()) == .47
        for key, row in list(running.items()):
            code = row['process'].poll()
            if code is None:
                continue
            row['log'].close()
            del running[key]
            if code:
                failures.append(dict(id=key, exit_code=code))
        if failures:
            for row in running.values():
                row['process'].wait()
                row['log'].close()
            atomic(out / 'status.json', dict(time=time.time(), pid=os.getpid(),
                status='failure_requires_inspection', failures=failures, goal_complete=False))
            return
        for group in range(2):
            for env in plan['environments']:
                for step in (800000, 900000, 1000000):
                    job = ready_job(root, plan, group, env, step, 'mean', verifier)
                    if job is not None:
                        assert job['id'] not in known or known[job['id']] == job
                        known[job['id']] = job
        atomic(out / 'prepared_evaluation_jobs.json', list(known.values()))
        for key, job in known.items():
            if Path(job['output']).exists():
                valid[key] = audit_job(registration, job, verifier)
        occupied = {row['slot'] for row in running.values()}
        for slot, cpus in enumerate(registration['CPUslots']):
            if slot in occupied:
                continue
            job = next((j for key, j in known.items() if key not in running and key not in valid), None)
            if job is None:
                continue
            log = (out / (job['id'] + '.log')).open('a')
            process = subprocess.Popen(command(job, cpus), env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            running[job['id']] = dict(process=process, log=log, slot=slot,
                process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()[19])
        summary = summarize(plan, 'mean', known, valid, dict(selected_nine_task_mean=raw['nine_task_mean']))
        result = dict(time=time.time(), pid=os.getpid(), status='complete' if summary['all_fresh_complete'] else 'running',
            **summary, ready_evaluations=len(known), valid_evaluations=len(valid), failures=failures,
            running={key: dict(pid=row['process'].pid, slot=row['slot'], process_identity=row['process_identity'])
                for key, row in running.items()}, registration_sha256=verifier.digest(args.registration),
            source_sha256=verifier.digest(__file__), primary_recipe_unchanged='mean_rank',
            control_only_not_eligible_for_candidate_promotion=True, additional_GPU_jobs=0,
            note='Paired raw-Q mean on the exact same heldout models/first BC/N9. Report alongside fixed rank rules; no per-task or cohort winning-method mixture.')
        atomic(out / 'status.json', result)
        if summary['all_fresh_complete']:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
