"""Adopt running learners under R11, without signaling learners or evaluators."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    root = Path(sys.argv[1]).resolve()
    area = root / 'analysis/mixture_onestep_pipeline_v1'
    continuation = root / 'CONTINUATION.json'
    saved = json.loads(continuation.read_text())
    plan = json.loads((root / 'registration_v11.json').read_text())
    previous = json.loads((root / 'registration_v10.json').read_text())
    for path, digest in plan['all_source_sha256'].items():
        assert sha(path) == digest, path
    assert all(plan['all_source_sha256'][p] == d for p, d in previous['all_source_sha256'].items())
    assert plan['thresholds'] == previous['thresholds']
    assert not (area / 'activation_before_v1.json').exists()
    assert not (root / 'STOP').exists() and not (root / 'PAUSE_NEW_LAUNCHES.json').exists()
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    assert len(scheduler['running']) == 16
    for row in scheduler['running'].values():
        assert identity(row['pid']) == row['process_identity']
        env = dict(x.split('=', 1) for x in Path(f'/proc/{row["pid"]}/environ').read_bytes().decode().split('\0') if '=' in x)
        assert env['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.1'
        assert env['CUDA_VISIBLE_DEVICES'] == str(row['gpu'])
        assert len(os.sched_getaffinity(row['pid'])) == 6
    owners = {role: saved[role] for role in ('scheduler', 'analysis_controller', 'gradient_controller')}
    expected = dict(scheduler='scheduler_onestep_deploy_v5', analysis_controller='watch_onestep_deploy_v7',
                    gradient_controller='watch_onestep_gradients_v4')
    for role, row in owners.items():
        assert identity(row['pid']) == row['process_identity']
        command = Path(f'/proc/{row["pid"]}/cmdline').read_bytes().decode().split('\0')
        assert 'utils.flow_iql_research.' + expected[role] in command
    atomic(area / 'activation_before_v1.json', dict(time=time.time(),scheduler=scheduler,owners=owners,
        registration_sha256=sha(root / 'registration_v11.json'),activation_script_sha256=sha(__file__),
        learner_and_evaluator_signals_authorized_or_used=False))
    # Stop only these exact three orchestration processes. Their workers remain live.
    for role in ('analysis_controller', 'gradient_controller', 'scheduler'):
        row = owners[role]
        assert identity(row['pid']) == row['process_identity']
        os.kill(row['pid'], signal.SIGTERM)
        end = time.monotonic() + 5
        while identity(row['pid']) == row['process_identity'] and time.monotonic() < end:
            time.sleep(.05)
        assert identity(row['pid']) is None, (role, 'Do not signal children or force kill')
    state_path = root / 'analysis/onestep_pipeline_v1/research_state.json'
    new = plan['mixture_implementation_extension']['new_jobs']
    expected_ids = plan['mixture_implementation_extension']['new_expected_identities']
    receipt = root / 'analysis/onestep_pipeline_v1/registrations/mixture_implementation_extension_v1.json'
    with (root / 'queue_mutation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        queue = json.loads((root / 'queue.json').read_text())
        assert state['jobs'] == previous['jobs'] and state['identities'] == previous['expected_identities']
        assert state['active_attempt'] == 1 and state['phase'] == 'confirmation'
        assert state['attempts'][1]['recipe'] == 'bank_bc' and state['attempts'][1]['stage'] == 'discovery'
        assert {j['id']:j for j in queue} == {j['id']:j for j in previous['jobs']}
        assert not receipt.exists() and not any(j['id'] in state['identities'] for j in new)
        after = dict(state, jobs=plan['jobs'],identities=plan['expected_identities'])
        after['round_recipes'] = [*state['round_recipes'], 'mdn9_bc']
        after['all_recipes'] = [*state['all_recipes'], 'mdn9_bc']
        after['events'] = [*state['events'],dict(time=time.time(),label='mixture_implementation_extension_v1',jobs=5)]
        finished = json.loads((root / 'finished.json').read_text())
        started = set(scheduler['running']) | set(finished)
        started |= {j['id'] for j in state['jobs'] if (root / 'runs' / j['id'] / 'config.json').exists()}
        ordered = order_pending(plan, after, queue + new, started)
        assert [j for j in ordered if j['id'] not in expected_ids] == queue
        # Receipt precedes both mutable writes, retaining the complete prior control state.
        atomic(receipt,dict(time=time.time(),new_jobs=new,expected_identities=expected_ids,
            registration_sha256=sha(root / 'registration_v11.json'),source_contract_sha256=sha(root / 'analysis/execution_contract_v13.json'),
            previous_state=state,previous_queue=queue,after_queue=ordered,started=sorted(started),
            learner_evaluator_or_gradient_producer_signals=0))
        atomic(state_path, after)
        atomic(root / 'queue.json', ordered)
    python = '/home/ukjo1/anaconda3/envs/ogbench/bin/python'
    commands = dict(
        scheduler=['taskset','-c','0-95',python,'-u','-m','utils.flow_iql_research.scheduler_onestep_deploy_v6',str(root),'--per-gpu','8'],
        analysis_controller=['nice','-n','10','taskset','-c','36-47',python,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v8',str(root / 'analysis/execution_contract_v13.json')],
        gradient_controller=['nice','-n','15','taskset','-c','36-41',python,'-u','-m','utils.flow_iql_research.watch_onestep_gradients_v5',str(root / 'analysis/gradient_execution_contract_v5.json')])
    handles = {}
    for role, command in commands.items():
        with (area / f'activation_{role}_v1.log').open('a') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        handles[role] = dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=command)
        if role != 'scheduler':
            handles[role]['execution_contract_sha256'] = sha(command[-1])
        atomic(area / 'activation_handles_v1.json',handles)
    for role in handles:
        saved['previous_mixture_transition_' + role] = owners[role]
        saved[role] = handles[role]
    saved.update(time=time.time(),registration_sha256=sha(root / 'registration_v11.json'),
        analysis_contract=str(root / 'analysis/execution_contract_v13.json'),gradient_contract=str(root / 'analysis/gradient_execution_contract_v5.json'),
        runtime_inspector='utils.flow_iql_research.inspect_onestep_runtime_v5',current_registered_jobs=100,
        mixture_joint_learner_draft=dict(saved['mixture_joint_learner_draft'],status='registered_mdN9_BC_only_R11_V13_gradV5',
            GPU_queue_unchanged=False,no_GPU_registration_or_active_owner_change=False,activation_receipt=str(receipt)),
        next_actions=['Use R11/contractV13/gradientV5 and inspectorV5. Audit adoption of all16 original learners and live evaluators; no learner signals.',
            'Maintain8/GPU, .1 memory, natural replenishment. Pending energy3 precede MDN9_BC5 and pending Set prefetch6; active bankBC full9 confirmation retained.',
            'MDN initially NLL-only, one global K9/std_floor.001/NLL-per-dimension weight1. No directQ GPU variants selected. Frozen CPU evidence exploratory, Puzzle remains unresolved.',
            'Goal active until common recipe full9 last3CPU50, three fresh seeds and isolated CPU/GPU latency pass. Original finite-mean/lean files/models protected.'])
    atomic(continuation,saved)
    print(json.dumps(dict(status='activated_await_independent_adoption_audit',handles=handles,receipt=str(receipt))),flush=True)


if __name__ == '__main__':
    main()
