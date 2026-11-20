"""Update the owned CPU controller; adopt all existing producers unchanged."""
import json
from pathlib import Path
import subprocess
import sys
import time
from utils.flow_iql_research.activate_mixture_full9_prefetch_v1 import identity,atomic,sha
from utils.flow_iql_research.activate_routed_policy_pipeline_v1 import stop_owned_parent


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/benchmark_CPU_isolation_v1';area.mkdir(exist_ok=True)
    planpath=root/'registration_v19.json';contract=root/'analysis/execution_contract_v21.json'
    plan=json.loads(planpath.read_text());old=json.loads((root/'registration_v18.json').read_text())
    assert json.loads(contract.read_text())['registration_sha256']==sha(planpath)
    assert plan['jobs']==old['jobs'] and plan['thresholds']==old['thresholds']
    for path,digest in plan['all_source_sha256'].items():assert sha(path)==digest
    cpath=root/'CONTINUATION.json';c=json.loads(cpath.read_text());owner=c['analysis_controller']
    assert Path(c['analysis_contract'])==root/'analysis/execution_contract_v20.json'
    assert identity(owner['pid'])==owner['process_identity']
    assert 'utils.flow_iql_research.watch_onestep_deploy_v11'in Path(f'/proc/{owner["pid"]}/cmdline').read_text().split('\0')
    env=dict(x.split('=',1)for x in Path(f'/proc/{owner["pid"]}/environ').read_text().split('\0')if'='in x)
    before=area/'activation_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),old_CPU_owner=owner,scheduler=c['scheduler'],gradient=c['gradient_controller'],
        scheduler_status=json.loads((root/'scheduler_status.json').read_text()),
        research_state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text()),
        no_learner_evaluator_or_gradient_producer_signals=True,all_jobs_identities_and_active_attempt_preserved=True))
    stop_owned_parent(owner,'utils.flow_iql_research.watch_onestep_deploy_v11')
    command=['utils.flow_iql_research.watch_onestep_deploy_v12'if x=='utils.flow_iql_research.watch_onestep_deploy_v11'else x for x in owner['command']];command[-1]=str(contract)
    with(area/'CPU_controller_activation_v1.log').open('a')as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    handle=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=command,execution_contract_sha256=sha(contract))
    atomic(area/'activation_handle_v1.json',handle)
    c.update(time=time.time(),analysis_controller=handle,analysis_contract=str(contract),registration_sha256=sha(planpath),
        authoritative_registration=str(planpath),previous_benchmark_isolation_controller=owner,
        benchmark_CPU_isolation=dict(status='activated_await_root_adoption_audit',registration=str(planpath),contract=str(contract)))
    c['next_actions'].insert(0,'AUTHORITATIVE R19/V21 CPUwatchV12; sourcepin count is recorded in registration. ManagerV7/gradientV6 andalljobs/identities/attempt4/fresh24..26unchanged. CPUevalpool mustdrainbeforeisolatedbenchmark; admissionreceipt required. InspectorV7. Auditproduceradoption;nosignalsapartownedCPUparent.')
    atomic(cpath,c);print(json.dumps(dict(status='activated_await_root_adoption_audit',CPU_controller=handle)))


if __name__=='__main__':main()
