"""Replace owned CPU controller, preserve all GPU/evaluation/gradient workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.flow_iql_research.activate_mixture_full9_prefetch_v1 import identity,atomic,sha


def stop_owned_parent(handle,module):
    assert identity(handle['pid'])==handle.get('process_identity',handle.get('birth'))
    argv=Path(f'/proc/{handle["pid"]}/cmdline').read_text().split('\0')
    assert module in argv
    os.kill(handle['pid'],signal.SIGTERM)
    end=time.monotonic()+5
    while identity(handle['pid'])==handle.get('process_identity',handle.get('birth')) and time.monotonic()<end:time.sleep(.05)
    assert identity(handle['pid'])is None,'No force kill or child signals'


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/routed_policy_pipeline_v1';area.mkdir(exist_ok=True)
    target=root/'registration_v18.json';contract=root/'analysis/execution_contract_v20.json'
    plan=json.loads(target.read_text());old=json.loads((root/'registration_v17.json').read_text())
    assert json.loads(contract.read_text())['registration_sha256']==sha(target)
    assert plan['jobs']==old['jobs'] and plan['thresholds']==old['thresholds']
    assert all(plan['all_source_sha256'][p]==digest for p,digest in old['all_source_sha256'].items())
    for path,digest in plan['all_source_sha256'].items():assert sha(path)==digest,path
    cpath=root/'CONTINUATION.json';c=json.loads(cpath.read_text())
    assert Path(c['analysis_contract'])==root/'analysis/execution_contract_v19.json'
    owner=c['analysis_controller'];assert identity(owner['pid'])==owner['process_identity']
    command=Path(f'/proc/{owner["pid"]}/cmdline').read_text().split('\0')
    assert 'utils.flow_iql_research.watch_onestep_deploy_v10' in command
    env=dict(x.split('=',1) for x in Path(f'/proc/{owner["pid"]}/environ').read_text().split('\0') if '=' in x)
    before=area/'activation_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),old_CPU_owner=owner,scheduler=c['scheduler'],gradient=c['gradient_controller'],
        scheduler_status=json.loads((root/'scheduler_status.json').read_text()),
        research_state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text()),
        new_registration_sha256=sha(target),no_learner_evaluator_or_gradient_producer_signals=True,
        all_142_job_definitions_and_identities_preserved=True))
    stop_owned_parent(owner,'utils.flow_iql_research.watch_onestep_deploy_v10')
    argv=['utils.flow_iql_research.watch_onestep_deploy_v11' if x=='utils.flow_iql_research.watch_onestep_deploy_v10' else x for x in owner['command']]
    argv[-1]=str(contract)
    with (area/'CPU_controller_activation_v1.log').open('a') as log:
        child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
    handle=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=argv,
                execution_contract_sha256=sha(contract))
    atomic(area/'activation_handle_v1.json',handle)
    # The old private observer was frozen for unspent seeds21..23. Retire only
    # that idle owned CPU parent; full-nine observer and all producers remain.
    legacy=root/'analysis/mixture_location_fresh_confirmation_v1'
    h=json.loads((legacy/'observer_handle_v1.json').read_text())
    state=json.loads((legacy/'observer_state_v1.json').read_text())
    assert state['actual_GPU_training_jobs_observed']==0 and not state['running']
    assert not Path(f'/proc/{h["pid"]}/task/{h["pid"]}/children').read_text().strip()
    retirement=area/'legacy_idle_CPU_observer_retirement_v1.json'
    atomic(retirement,dict(time=time.time(),previous_handle=h,previous_state=state,
        no_children_or_GPU_jobs=True,reason='Its planned seeds21..23 remain unspent. Production policy-specific attempt4 reserves fresh24..26 before any allocation.'))
    stop_owned_parent(h,'utils.flow_iql_research.watch_mixture_location_fresh_CPU_v1')
    c.update(time=time.time(),analysis_controller=handle,previous_routed_policy_controller=owner,
        analysis_contract=str(contract),registration_sha256=sha(target),authoritative_registration=str(target),
        routed_policy_pipeline=dict(status='activated_await_root_adoption_audit',registration=str(target),
            contract=str(contract),receipt=str(before),legacy_idle_observer_retirement=str(retirement)))
    c['next_actions'].insert(0,'AUTHORITATIVE R18/V20; CPUwatchV11 onlyproductionparent replaced. ManagerV7+gradientV6 unchanged. Policycandidate mdn9_bc__component_location reuses nativeMDN identities; attempt3default superseded ascontrol,attempt4fresh24/25/26 onlyafterfull9. Retiredidleprivate21..23CPUwatch, no producer signals. Auditallworkeradoptions.')
    atomic(cpath,c)
    print(json.dumps(dict(status='activated_await_root_adoption_audit',CPU_controller=handle,
                         legacy_idle_CPU_observer_retired=h['pid'])))


if __name__=='__main__':main()
