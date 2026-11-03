"""Adopt owned GPU and CPU children under R14; never signal model workers."""
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
    root=Path(sys.argv[1]).resolve();area=root/'analysis/latent_selector_joint_pipeline_v1'
    oldpath=root/'registration_v13.json';target=root/'registration_v14.json'
    old=json.loads(oldpath.read_text());plan=json.loads(target.read_text())
    contract=root/'analysis/execution_contract_v16.json';gradient=root/'analysis/gradient_execution_contract_v6.json'
    assert json.loads(contract.read_text())['registration_sha256']==sha(target)and len(plan['jobs'])==130
    assert plan['thresholds']==old['thresholds']
    for p,h in plan['all_source_sha256'].items():assert sha(p)==h,p
    assert all(plan['all_source_sha256'][p]==h for p,h in old['all_source_sha256'].items())
    assert not(root/'STOP').exists()and not(root/'PAUSE_NEW_LAUNCHES.json').exists()
    continuation=root/'CONTINUATION.json';saved=json.loads(continuation.read_text())
    assert Path(saved['analysis_contract'])==root/'analysis/execution_contract_v15.json'
    roles=('scheduler','analysis_controller','gradient_controller')
    owners={k:saved[k]for k in roles};role_env={}
    expected=dict(scheduler='utils.flow_iql_research.scheduler_onestep_deploy_v6',
        analysis_controller='utils.flow_iql_research.watch_onestep_deploy_v8',
        gradient_controller='utils.flow_iql_research.watch_onestep_gradients_v5')
    replacement=dict(scheduler='utils.flow_iql_research.scheduler_onestep_deploy_v7',
        analysis_controller='utils.flow_iql_research.watch_onestep_deploy_v9',
        gradient_controller='utils.flow_iql_research.watch_onestep_gradients_v6')
    scheduler=json.loads((root/'scheduler_status.json').read_text());assert len(scheduler['running'])==16
    for row in scheduler['running'].values():
        assert identity(row['pid'])==row['process_identity']
        env=dict(x.split('=',1)for x in Path(f'/proc/{row["pid"]}/environ').read_text().split('\0')if'='in x)
        assert env['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'and env['CUDA_VISIBLE_DEVICES']==str(row['gpu'])
        assert len(os.sched_getaffinity(row['pid']))==6
    for role,owner in owners.items():
        assert identity(owner['pid'])==owner['process_identity']
        command=Path(f'/proc/{owner["pid"]}/cmdline').read_text().split('\0')
        assert expected[role]in command
        role_env[role]=dict(x.split('=',1)for x in Path(f'/proc/{owner["pid"]}/environ').read_text().split('\0')if'='in x)
    assert role_env['scheduler'].get('JAX_PLATFORM_NAME')!='cpu'
    before=area/'activation_before_v1.json';assert not before.exists()
    atomic(before,dict(time=time.time(),owners=owners,scheduler=scheduler,
        evaluation_workers=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())['evaluation_workers'],
        gradient_status=json.loads((root/'analysis/actor_gradient_diagnosis_v1/status.json').read_text()),
        registration_sha256=sha(target),activation_source_sha256=sha(__file__),worker_signals=0))
    for role in ('analysis_controller','gradient_controller','scheduler'):
        owner=owners[role];assert identity(owner['pid'])==owner['process_identity']
        os.kill(owner['pid'],signal.SIGTERM);end=time.monotonic()+5
        while identity(owner['pid'])==owner['process_identity']and time.monotonic()<end:time.sleep(.05)
        assert identity(owner['pid'])is None,'Do not force kill or signal children'
    state_path=root/'analysis/onestep_pipeline_v1/research_state.json'
    receipt=root/'analysis/onestep_pipeline_v1/registrations/latent_selector_extension_v1.json'
    new=plan['latent_selector_extension']['new_jobs'];new_ids={j['id']for j in new}
    with(root/'queue_mutation.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(state_path.read_text());queue=json.loads((root/'queue.json').read_text())
        assert state['jobs']==old['jobs']and state['identities']==old['expected_identities']
        assert state['phase']=='confirmation'and state['active_attempt']==1 and state['attempts'][1]['recipe']=='bank_bc'
        assert {j['id']:j for j in queue}=={j['id']:j for j in old['jobs']}
        started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
        started|={j['id']for j in old['jobs']if(root/'runs'/j['id']/'config.json').exists()}
        assert not any((root/'runs'/j['id']).exists()for j in new)
        recipes=plan['latent_selector_extension']['recipes']
        after=dict(state,jobs=plan['jobs'],identities=plan['expected_identities'],
            all_recipes=[*state['all_recipes'],*recipes],round_recipes=[*state['round_recipes'],*recipes])
        ordered=order_pending(plan,after,queue+new,started)
        assert [j for j in ordered if j['id']not in new_ids]==queue
        assert all(x==y for x,y in zip(queue,ordered)if x['id']in started)
        assert not receipt.exists()
        atomic(receipt,dict(time=time.time(),registration_sha256=sha(target),contract_sha256=sha(contract),gradient_contract_sha256=sha(gradient),
            previous_state=state,previous_queue=queue,after_queue=ordered,started=sorted(started),
            new_jobs=new,new_expected_identities=plan['latent_selector_extension']['new_expected_identities'],
            learner_evaluator_gradient_producer_signals=0,role_environments_preserved_without_recording_secrets=True))
        after['events']=[*state['events'],dict(time=time.time(),label='latent_selector_extension_v1',receipt=str(receipt),new_jobs=25)]
        atomic(state_path,after);atomic(root/'queue.json',ordered)
    handles={}
    for role in roles:
        command=[replacement[role]if x==expected[role]else x for x in owners[role]['command']]
        if role=='analysis_controller':command[-1]=str(contract)
        if role=='gradient_controller':command[-1]=str(gradient)
        with(area/(role+'_activation_v1.log')).open('a')as log:
            child=subprocess.Popen(command,env=role_env[role],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        handles[role]=dict(time=time.time(),pid=child.pid,process_identity=identity(child.pid),command=command)
        if role!='scheduler':handles[role]['execution_contract_sha256']=sha(command[-1])
        atomic(area/'activation_handles_v1.json',handles)
    for role,handle in handles.items():saved['previous_latent_'+role]=owners[role];saved[role]=handle
    saved.update(time=time.time(),analysis_contract=str(contract),gradient_contract=str(gradient),registration_sha256=sha(target),
        current_registered_jobs=130,runtime_inspector='utils.flow_iql_research.inspect_onestep_runtime_v6',
        latent_selector_joint_extension=dict(status='activated_await_adoption',receipt=str(receipt),registration=str(target),contract=str(contract)))
    saved['next_actions'].insert(0,'AUTHORITATIVE R14/V16/gradientV6. ManagerV7,CPUwatchV9,gradientwatchV6. Audit16learner and CPU/gradient-childadoption. New25nativejointlatentjobs exploratory; originalfinite-mean/lean and9task/fresh/speedgates unchanged.')
    atomic(continuation,saved)
    print(json.dumps(dict(status='activated_await_adoption_audit',handles=handles,receipt=str(receipt))))


if __name__=='__main__':main()
