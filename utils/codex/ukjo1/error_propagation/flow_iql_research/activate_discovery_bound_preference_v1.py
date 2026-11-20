"""Replace only the exact owned CPU controller; all worker handles are adopted."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.flow_iql_research.activate_mixture_full9_prefetch_v1 import identity,atomic,sha


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/discovery_bound_preference_v1';area.mkdir(exist_ok=True)
    target=root/'registration_v17.json';contract=root/'analysis/execution_contract_v19.json'
    plan=json.loads(target.read_text());old=json.loads((root/'registration_v16.json').read_text())
    assert json.loads(contract.read_text())['registration_sha256']==sha(target)
    assert plan['jobs']==old['jobs']and plan['thresholds']==old['thresholds']
    assert all(plan['all_source_sha256'][p]==h for p,h in old['all_source_sha256'].items())
    for p,h in plan['all_source_sha256'].items():assert sha(p)==h,p
    cpath=root/'CONTINUATION.json';c=json.loads(cpath.read_text());assert Path(c['analysis_contract'])==root/'analysis/execution_contract_v18.json'
    owner=c['analysis_controller'];assert identity(owner['pid'])==owner['process_identity']
    command=Path(f'/proc/{owner["pid"]}/cmdline').read_text().split('\0')
    assert 'utils.flow_iql_research.watch_onestep_deploy_v9'in command
    env=dict(x.split('=',1)for x in Path(f'/proc/{owner["pid"]}/environ').read_text().split('\0')if'='in x)
    before=area/'activation_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),old_owner=owner,scheduler=c['scheduler'],gradient=c['gradient_controller'],
        scheduler_status=json.loads((root/'scheduler_status.json').read_text()),
        research_state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text()),
        new_registration_sha256=sha(target),no_learner_evaluator_or_gradient_producer_signals=True,
        no_state_or_queue_mutations=True))
    os.kill(owner['pid'],signal.SIGTERM);end=time.monotonic()+5
    while identity(owner['pid'])==owner['process_identity']and time.monotonic()<end:time.sleep(.05)
    assert identity(owner['pid'])is None,'Never force kill or signal children'
    newcommand=['utils.flow_iql_research.watch_onestep_deploy_v10'if x=='utils.flow_iql_research.watch_onestep_deploy_v9'else x for x in owner['command']]
    newcommand[-1]=str(contract)
    with(area/'CPU_controller_activation_v1.log').open('a')as log:
        child=subprocess.Popen(newcommand,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    handle=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=newcommand,execution_contract_sha256=sha(contract))
    atomic(area/'activation_handle_v1.json',handle)
    c.update(time=time.time(),analysis_controller=handle,previous_discovery_bound_controller=owner,
        analysis_contract=str(contract),registration_sha256=sha(target),authoritative_registration=str(target),
        discovery_bound_preference=dict(status='activated_await_adoption_audit',registration=str(target),contract=str(contract),receipt=str(before)))
    c['next_actions'].insert(0,'AUTHORITATIVE R17/V19; CPUwatchV10 onlyparentreplaced. ManagerV7 andgradientV6 unchanged. Auditall16learner/evaluatoradoptions. Exactempiricalbound neverstopsruns; ifSetfails,nextMDNfull9preferenceisexploratorynotpilot-qualified. Allfinalgatesunchanged.')
    atomic(cpath,c);print(json.dumps(dict(status='activated_await_adoption_audit',CPU_controller=handle)))


if __name__=='__main__':main()
