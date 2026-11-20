"""Advance two unstarted MDN screens; preserve100job definitions and all gates."""
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


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19]if fields[0]!='Z'else None
    except FileNotFoundError:return None


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/mixture_onestep_pipeline_v1'
    parent=root/'registration_v11.json';previous=root/'analysis/execution_contract_v13.json'
    registration=root/'registration_v12.json';execution=root/'analysis/execution_contract_v14.json'
    assert not registration.exists()and not execution.exists()
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    for p,d in plan['all_source_sha256'].items():assert sha(p)==d,p
    state_path=root/'analysis/onestep_pipeline_v1/research_state.json'
    state=json.loads(state_path.read_text());assert state['jobs']==plan['jobs']and len(state['jobs'])==100
    assert state['phase']=='confirmation'and state['active_attempt']==1 and state['attempts'][1]['recipe']=='bank_bc'
    moving=['osm1_cube_mdn9_bc_s0','osm1_pz3_mdn9_bc_s0'];old=list(plan['jobs'])
    for rid in moving:assert not(root/'runs'/rid).exists(),rid
    jobs={j['id']:j for j in old};remaining=[j for j in old if j['id']not in moving]
    scheduler=json.loads((root/'scheduler_status.json').read_text());assert len(scheduler['running'])==16
    finished=json.loads((root/'finished.json').read_text())
    started=set(scheduler['running'])|set(finished)|{j['id']for j in old if(root/'runs'/j['id']/'config.json').exists()}
    insertion=next(i for i,j in enumerate(remaining)if j['id']not in started)
    assert remaining[insertion]['args']['recipe']=='energy_bc'
    plan['jobs']=remaining[:insertion]+[jobs[rid]for rid in moving]+remaining[insertion:]
    assert {j['id']:j for j in plan['jobs']}==jobs
    assert [j for j in plan['jobs']if j['id']not in moving]==remaining
    old95=json.loads((root/'registration_v10.json').read_text())['jobs'];ids95={j['id']for j in old95}
    assert [j for j in plan['jobs']if j['id']in ids95]==old95
    amendment=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        moved_pending_jobs=moving,new_jobs=0,job_definitions_configurations_and_training_sources_unchanged=True,
        all_gate_thresholds_and_active_bankBC_attempt_unchanged=True,
        rationale='Start the first two jointMDN hard-task screens sooner alongside the existing energy Cube/Puzzle learners. Frozen20KCPU results do not settle joint1M extraction. Compare implementations before scheduling further variants.',
        selection_scope='Adaptive allocation only, before any jointMDN GPU outcome. No per-environment recipe/parameter selection or performance promotion.',
        queue_rule='Keep started positions and all100job definitions. Advance MDN Cube/Puzzle before remaining unstarted energy pilots; preserve relative logical order of the other98 and original95 jobs.',
        no_new_fresh_seed_allocation=True,no_goal_completion_claim=True)
    sources={**plan['all_source_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    plan.update(all_source_sha256=sources,mixture_hard_task_priority_amendment=amendment)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    atomic(registration,plan)
    contract.update(registration=str(registration),registration_sha256=sha(registration),source_sha256=sources,
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),mixture_hard_task_priority_amendment=amendment)
    atomic(execution,contract)
    continuation=root/'CONTINUATION.json';saved=json.loads(continuation.read_text());owner=saved['analysis_controller']
    assert identity(owner['pid'])==owner['process_identity']
    command=Path(f'/proc/{owner["pid"]}/cmdline').read_bytes().decode().split('\0')
    assert 'utils.flow_iql_research.watch_onestep_deploy_v8'in command and str(previous)in command
    before=area/'hard_task_priority_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),owner=owner,scheduler=scheduler,amendment=amendment))
    # Only the CPU analysis controller is replaced; manager and every learner remain live.
    os.kill(owner['pid'],signal.SIGTERM);end=time.monotonic()+5
    while identity(owner['pid'])==owner['process_identity']and time.monotonic()<end:time.sleep(.05)
    assert identity(owner['pid'])is None
    receipt=root/'analysis/onestep_pipeline_v1/registrations/mixture_hard_task_priority_v1.json'
    with(root/'queue_mutation.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(state_path.read_text());assert state['jobs']==old
        live=json.loads((root/'scheduler_status.json').read_text())
        assert live['running']==scheduler['running']
        queue=json.loads((root/'queue.json').read_text());after=dict(state,jobs=plan['jobs'])
        ordered=order_pending(plan,after,queue,started)
        assert all(jobs[j['id']]==j for j in ordered)
        assert all(a==b for a,b in zip(queue,ordered)if a['id']in started)
        assert [j['id']for j in ordered if j['id']not in started][:2]==moving
        assert not receipt.exists()
        atomic(receipt,dict(time=time.time(),registration_sha256=sha(registration),contract_sha256=sha(execution),
            previous_state=state,previous_queue=queue,after_queue=ordered,started=sorted(started),learner_signals=0))
        after['events']=[*state['events'],dict(time=time.time(),label='mixture_hard_task_priority_v1',receipt=str(receipt))]
        atomic(state_path,after);atomic(root/'queue.json',ordered)
    command=[*owner['command'][:-1],str(execution)]
    with(area/'hard_task_priority_controller_v1.log').open('a')as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    handle=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=command,execution_contract_sha256=sha(execution))
    atomic(area/'hard_task_priority_controller_handle_v1.json',handle)
    saved.update(time=time.time(),previous_hard_task_priority_CPU_controller=owner,analysis_controller=handle,
        analysis_contract=str(execution),registration_sha256=sha(registration),latest_runtime_audit=str(before))
    saved['mixture_joint_learner_draft'].update(hard_task_priority_receipt=str(receipt),status='R12_V14_priority_first2_MDN_pending_gradV5_unchanged')
    saved['next_actions']=['Authoritative R12/contractV14, schedulerV6 and gradientV5 unchanged. CPUcontrollerV8 replaced only; audit adoption.',
        'Preserve16learners,8/GPU,.1memory. Nextpending MDNCube/Puzzle,thenremainingenergy2,thenMDNthreepilots. No learner signals or gate changes.',
        'Continue bankBC full9 attempt1. MDN1M efficacy not observed; common full9/fresh/isolatedlatency still required.',
        'CPUmultilabel sweep complete negative. PlainMDN scaling65K/524K states at100KCPU updates running,notgoal.']
    atomic(continuation,saved)
    print(json.dumps(dict(status='priority_activated_await_adoption_audit',registration_sha256=sha(registration),contract_sha256=sha(execution),CPU_controller=handle)),flush=True)


if __name__=='__main__':main()
