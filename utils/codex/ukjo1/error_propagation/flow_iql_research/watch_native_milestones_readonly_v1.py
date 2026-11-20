"""Observe exact owned learner handles; never launch/stop/change experiments.

This operational watcher records actual /proc identities and source-fixed
pipeline state while the already-authorized GPU learners train. An elapsed
observation window never implies a learner failure or triggers a restart.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from utils.flow_iql_research import inspect_onestep_runtime_v7 as runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.proposal.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path(plan['root'])
    output = Path(plan['journal'])
    assert not output.exists()
    transients = 0
    while True:
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest
        state = json.loads((root / 'analysis/onestep_pipeline_v1/status.json').read_text())
        native = {row['run']: row for row in state['rows'] if not row.get('native_training_recipe')}
        targets = []
        alerts = []
        available = []
        for target in plan['targets']:
            run = root / 'runs' / target['run']
            live = identity(target['pid']) == target['birth']
            row = native.get(target['run'], {})
            probe = row.get('last_probe') or {}
            model = run / f'checkpoint_{plan["step"]}.pkl'
            actor = run / f'actor_{plan["step"]}.pkl'
            model_ready = model.exists() and actor.exists()
            done = (run / 'completed.json').exists()
            stopped = (run / 'stopped.json').exists()
            targets.append(dict(run=target['run'], pid=target['pid'], birth=target['birth'],
                exact_handle_live=live, latest_native_probe_step=probe.get('step'),
                observed_probe_finite=probe.get('finite'), requested_step=plan['step'],
                requested_model_ready=model_ready, native_completed_marker=done,
                native_stopped_marker=stopped))
            if model_ready:
                available.append(target['run'])
            if stopped or row.get('status') in ['training_failed', 'training_nonfinite', 'evaluation_failed']:
                alerts.append(target['run'])
            if not live and not done and not model_ready:
                alerts.append(target['run'] + ':handle_missing_without_completed_model')
        record = dict(time=time.time(), observer_pid=os.getpid(), observer_birth=identity(os.getpid()),
            proposal_sha256=sha(args.proposal), targets=targets,
            native_runs_evaluated=state['complete_primary_and_teacher_runs'],
            completed_routed_tasks=state['complete_routed_policy_runs'],
            evaluation_failures=state['evaluation_failures'],
            observer_performs_no_training_evaluation_or_queue_mutation=True)
        try:
            snapshot = runtime.inspect(root)
            record.update(actual_learners_per_GPU=snapshot['live'],
                live_learner_handles=snapshot['handles'], finite_probed=snapshot['probed_running_count'],
                all_observed_finite=snapshot['finite'])
            if not snapshot['finite']:
                alerts.append('nonfinite_running_probe')
            transients = 0
        except (FileNotFoundError, ProcessLookupError, AssertionError) as error:
            transients += 1
            record['inspection_transient'] = repr(error)
            record['consecutive_inspection_transients'] = transients
        if (root / 'analysis/onestep_pipeline_v1/controller_failure.json').exists():
            alerts.append('CPU_controller_failure_record')
        record['alerts'] = alerts
        record['status'] = 'verified_wait_on_existing_native_learners'
        if available:
            record.update(status='requested_native_checkpoint_available_pending_root_review', available_runs=available)
        elif alerts or transients >= 3:
            record.update(status='requires_root_read_only_inspection', alerts=alerts)
        elif (root / 'STOP').exists():
            record['status'] = 'root_STOP_present_observer_exits_without_altering_jobs'
        with output.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        if record['status'] != 'verified_wait_on_existing_native_learners':
            return
        time.sleep(plan['poll_seconds'])


if __name__ == '__main__':
    main()
