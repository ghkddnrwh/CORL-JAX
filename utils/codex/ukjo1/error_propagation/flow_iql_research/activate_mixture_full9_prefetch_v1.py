"""Adopt 142 definitions; signal only exact owned manager/CPU-controller parents.

No learner/evaluator/gradient producer is signaled. Preserve old role envs in
memory so launching this registrar in CPU mode cannot turn learners into CPU jobs.
"""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return x[19]if x[0]!='Z'else None
    except FileNotFoundError:return None


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/mixture_full9_prefetch_v1'
    previous=root/'registration_v15.json';target=root/'registration_v16.json'
    old=json.loads(previous.read_text());plan=json.loads(target.read_text())
    contract=root/'analysis/execution_contract_v18.json';c=json.loads(contract.read_text())
    assert c['registration_sha256']==sha(target)and len(plan['jobs'])==142
    assert plan['thresholds']==old['thresholds']
    for p,h in plan['all_source_sha256'].items():assert sha(p)==h,p
    assert not(root/'STOP').exists()and not(root/'PAUSE_NEW_LAUNCHES.json').exists()
    continuation=root/'CONTINUATION.json';saved=json.loads(continuation.read_text())
    assert Path(saved['analysis_contract'])==root/'analysis/execution_contract_v17.json'
    owners={k:saved[k]for k in ['scheduler','analysis_controller']};role_env={}
    expected=dict(scheduler='utils.flow_iql_research.scheduler_onestep_deploy_v7',analysis_controller='utils.flow_iql_research.watch_onestep_deploy_v9')
    scheduler=json.loads((root/'scheduler_status.json').read_text());assert len(scheduler['running'])==16
    for role,owner in owners.items():
        assert identity(owner['pid'])==owner['process_identity']
        command=Path(f'/proc/{owner["pid"]}/cmdline').read_text().split('\0')
        assert expected[role]in command
        role_env[role]=dict(x.split('=',1)for x in Path(f'/proc/{owner["pid"]}/environ').read_text().split('\0')if'='in x)
    assert role_env['scheduler'].get('JAX_PLATFORM_NAME')!='cpu'
    before=area/'activation_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),owners=owners,scheduler=scheduler,new_registration_sha256=sha(target),
                      preserved_gradient_owner=saved['gradient_controller'],no_learner_signals=True))
    # Freeze queue writers/launches by stopping their parents, leaving all children live.
    for role in ['analysis_controller','scheduler']:
        owner=owners[role];assert identity(owner['pid'])==owner['process_identity']
        os.kill(owner['pid'],signal.SIGTERM);end=time.monotonic()+5
        while identity(owner['pid'])==owner['process_identity']and time.monotonic()<end:time.sleep(.05)
        assert identity(owner['pid'])is None,'Do not force kill or signal children'
    state_path=root/'analysis/onestep_pipeline_v1/research_state.json'
    receipt=root/'analysis/onestep_pipeline_v1/registrations/mixture_full9_prefetch_extension_v1.json'
    new=plan['mixture_full9_prefetch_extension']['new_jobs'];new_ids={j['id']for j in new}
    with(root/'queue_mutation.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(state_path.read_text());queue=json.loads((root/'queue.json').read_text())
        assert state['jobs']==old['jobs']and state['identities']==old['expected_identities']
        assert state['phase']=='confirmation'and state['active_attempt']==2
        assert state['attempts'][2]['recipe']=='set_bc'
        assert {j['id']:j for j in queue}=={j['id']:j for j in old['jobs']}
        started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
        started|={j['id']for j in old['jobs']if(root/'runs'/j['id']/'config.json').exists()}
        assert not any((root/'runs'/j['id']).exists()for j in new)
        recipes=plan['mixture_full9_prefetch_extension']['added_recipes']
        after=dict(state,jobs=plan['jobs'],identities=plan['expected_identities'],
                   all_recipes=[*state['all_recipes'],*recipes],round_recipes=[*state['round_recipes'],*recipes])
        ordered=order_pending(plan,after,queue+new,started)
        assert [j for j in ordered if j['id']not in new_ids]==queue
        assert all(x==y for x,y in zip(queue,ordered)if x['id']in started)
        assert not receipt.exists()
        atomic(receipt,dict(time=time.time(),registration_sha256=sha(target),contract_sha256=sha(contract),
            previous_state=state,previous_queue=queue,after_queue=ordered,started=sorted(started),
            new_jobs=new,new_expected_identities=plan['mixture_full9_prefetch_extension']['new_expected_identities'],
            learner_evaluator_gradient_producer_signals=0,role_environments_preserved_without_recording_secrets=True))
        after['events']=[*state['events'],dict(time=time.time(),label='mixture_full9_prefetch_extension_v1',receipt=str(receipt),new_jobs=4)]
        atomic(state_path,after);atomic(root/'queue.json',ordered)
    commands=dict(scheduler=owners['scheduler']['command'],
                  analysis_controller=[*owners['analysis_controller']['command'][:-1],str(contract)])
    handles={}
    for role,command in commands.items():
        with(area/(role+'_activation_v1.log')).open('a')as log:
            child=subprocess.Popen(command,env=role_env[role],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,close_fds=True)
        handle=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=command)
        if role=='analysis_controller':handle['execution_contract_sha256']=sha(contract)
        handles[role]=handle;atomic(area/'activation_handles_v1.json',handles)
    for role,handle in handles.items():saved['previous_mixture_prefetch_'+role]=owners[role];saved[role]=handle
    saved.update(time=time.time(),analysis_contract=str(contract),registration_sha256=sha(target),
                 current_registered_jobs=142,mixture_full9_prefetch_extension=dict(status='activated_await_adoption',
                 receipt=str(receipt),registration=str(target),contract=str(contract),recipe=plan['mixture_full9_prefetch_extension']['recipe']))
    saved['next_actions'].insert(0,'AuthoritativeR16/V18; managerV7 andCPUwatchV9 samecode; gradientV6unchanged. Audit16adoptions. FourremainingMDN9tasks prefetched,notformalpromotion; preserveSetattempt2/gates/freshidentity.')
    atomic(continuation,saved)
    print(json.dumps(dict(status='activated_await_adoption_audit',handles=handles,receipt=str(receipt))))


if __name__=='__main__':main()
