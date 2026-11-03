"""Private V5 startup/queue receipt test with no models or GPU operations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.scheduler_v4 import process_identity
from utils.flow_iql_research.onestep_implementation_registry_v2 import MEAN_TEACHER,configuration,implementation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    private=args.output_dir.resolve();private.mkdir(parents=True,exist_ok=False)
    (private/'analysis').mkdir()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads((args.root/'registration_v4.json').read_text());plan['root']=str(private)
    plan['scheduling_amendment']=dict(performance_gates_unchanged=True,private_fixture_only=True)
    directory=Path(__file__).resolve().parent
    sources={str(path):sha(path)for path in MEAN_TEACHER.core.source_paths()+[directory/'train_mean_teacher_onestep_v1.py']}
    plan['implementation_training_source_sha256']['mean_teacher']=sources
    plan['all_source_sha256'].update(sources)
    for recipe in MEAN_TEACHER.core.RECIPES:
        for index,environment in enumerate(plan['pilot_environments']):
            rid=f'private_{index}_{recipe}';job=dict(id=rid,module=MEAN_TEACHER.runner_module,args=dict(env=environment,recipe=recipe,seed=0,steps=1000000))
            plan['jobs'].append(job);plan['expected_identities'][rid]=configuration(environment,recipe,0)
        for environment in MEAN_TEACHER.core.ENVIRONMENTS:
            identity=configuration(environment,recipe,0)
            assert identity['specification']['teacher_q_aggregation']=='mean' and implementation(recipe)==MEAN_TEACHER
    for name in ('watch_onestep_deploy_v6.py','onestep_scheduling_policy_v1.py','scheduler_onestep_queue_v1.py','scheduler_onestep_deploy_v4.py','onestep_implementation_registry_v2.py','evaluate_mean_teacher_onestep_v1.py','benchmark_mean_teacher_onestep_v1.py'):
        path=directory/name;plan['all_source_sha256'][str(path)]=sha(path)
    registration=private/'private_registration.json';registration.write_text(json.dumps(plan,indent=2))
    contract=private/'private_contract.json';contract.write_text(json.dumps(dict(registration=str(registration),
        registration_sha256=sha(registration),source_sha256=plan['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),
        GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v4'),indent=2))
    queue=plan['jobs'];(private/'queue.json').write_text(json.dumps(queue,indent=2))
    (private/'finished.json').write_text('{}')
    # This *live test process* occupies the manager handle. The controller
    # therefore has no reason to spawn a GPU manager in this private root.
    (private/'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=len(queue),finished=0)))
    for env in plan['pilot_environments'][:2]:
        job=next(job for job in queue if job['args']['recipe']=='bank_bc'and job['args']['env']==env)
        output=private/'runs'/job['id'];output.mkdir(parents=True)
        shutil.copyfile(args.root/'runs'/job['id']/'config.json',output/'config.json')
        # No source archive or weights: legitimate 'initializing', never a
        # synthetic performance score or a runnable training artifact.
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v6',str(contract)]
    with(private/'controller.log').open('a')as log:
        child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=process_identity(child.pid);deadline=time.monotonic()+20;status_path=private/'analysis/onestep_pipeline_v1/status.json'
    while time.monotonic()<deadline:
        if status_path.exists():break
        assert child.poll()is None,(private/'controller.log').read_text()
        time.sleep(.1)
    assert status_path.exists(),'Private child still owned; inspect actual PID before recovery'
    status=json.loads(status_path.read_text());state=json.loads((private/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert status['pid']==child.pid and status['phase']=='pilot'and not status['CPU_running']and not status['benchmark_running']
    assert not status['evaluation_failures']and not status['attempts']and status['complete_primary_and_teacher_runs']==0
    assert all(row['primary_mean']is None and row['teacher_mean']is None for row in status['rows'])
    assert not any(event['label']=='restart_owned_GPU_scheduler'for event in state['events'])
    new_queue=json.loads((private/'queue.json').read_text());assert {j['id']:j for j in new_queue}=={j['id']:j for j in queue}
    receipt_path=private/'analysis/onestep_pipeline_v1/registrations/pending_order0.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['source_contract_sha256']==sha(contract)
    assert receipt['before']==[j['id']for j in queue]and receipt['after']==[j['id']for j in new_queue]
    first=next(job for job in new_queue if job['id']not in receipt['started'])
    assert first['args']['recipe']=='bank_bc'and first['args']['env']==plan['pilot_environments'][2]
    assert process_identity(child.pid)==birth
    cmd=Path(f'/proc/{child.pid}/cmdline').read_bytes().decode().split('\0')
    assert cmd[cmd.index('-m')+1]=='utils.flow_iql_research.watch_onestep_deploy_v6'
    os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    result=dict(time=time.time(),status='pass',checks=[
        'Actual V5 first loop verifies frozen sources and writes status in an isolated root',
        'Registers deterministic pending-only order receipt before atomic queue change',
        'Retains every job definition and keeps all unavailable scores missing',
        'Starts zero GPU managers, learners, evaluations or benchmarks'],
        no_performance_claim=True,no_production_queue_change=True,
        source_sha256={str(path):sha(path)for path in [Path(__file__).resolve(),directory/'watch_onestep_deploy_v6.py',directory/'onestep_scheduling_policy_v1.py']})
    (private/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='pass',output=str(private/'result.json'))),flush=True)


if __name__=='__main__':main()
