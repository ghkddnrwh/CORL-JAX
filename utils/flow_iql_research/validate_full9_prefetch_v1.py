"""Private CPU integration of prefetch routing; never trains/evaluates a model."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from utils.flow_iql_research.register_full9_prefetch_v1 import build_plan
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v2 import configuration,implementation
from utils.flow_iql_research.scheduler_v4 import process_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'analysis').mkdir();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,_=build_plan(root);assert len(plan['jobs'])==78
    selected=[j for j in plan['jobs']if j['args']['recipe']=='fqln_a03'and j['args']['seed']==0]
    environments=plan['scheduling_amendment']['all_nine_environments']
    assert len(selected)==9 and {j['args']['env']for j in selected}==set(environments)
    assert all(j['args']['seed']==0 for j in plan['jobs'])
    # Compile the actual frozen controller's nested helpers without executing
    # its main loop. Exercise deduplication with all9present and one missing.
    controller=Path(__file__).resolve().with_name('watch_onestep_deploy_v6.py')
    tree=ast.parse(controller.read_text());main_node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main')
    nodes=[next(x for x in main_node.body if isinstance(x,ast.FunctionDef)and x.name==name)for name in ('add_configuration','promote')]
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(controller),'exec')
    captured=[]
    bindings=dict(core=SimpleNamespace(ENVIRONMENTS=environments),configuration=configuration,implementation=implementation,
        append_jobs=lambda new,expected,label,details:captured.append(dict(new=copy.deepcopy(new),expected=copy.deepcopy(expected),label=label)))
    def simulate(jobs):
        bindings['state']=dict(jobs=copy.deepcopy(jobs),attempts=[],phase='pilot',active_attempt=None)
        captured.clear();exec(code,bindings)
        bindings['promote'](dict(recipe='fqln_a03',private_routing_only=True))
        assert len(captured)==1
        return copy.deepcopy(captured[0]),bindings['state']
    duplicate,state=simulate(plan['jobs']);assert duplicate['new']==[]and duplicate['expected']=={}
    assert state['phase']=='confirmation'and state['attempts'][0]['stage']=='discovery'
    removed=plan['full9_prefetch_extension']['new_jobs'][0]
    control,_=simulate([j for j in plan['jobs']if j['id']!=removed['id']])
    assert len(control['new'])==1 and control['new'][0]['args']==removed['args']
    generated=control['new'][0];assert control['expected'][generated['id']]==configuration(removed['args']['env'],'fqln_a03',0)
    assert generated['module']==removed['module']
    # Run the unchanged CPU controller once against an isolated, weightless
    # root. A real live parent occupies the manager handle, so no GPU manager
    # can be launched. No fabricated model or performance scores are used.
    private=copy.deepcopy(plan);private['root']=str(out)
    registration=out/'private_registration.json';registration.write_text(json.dumps(private,indent=2))
    contract=out/'private_contract.json';contract.write_text(json.dumps(dict(registration=str(registration),
        registration_sha256=sha(registration),source_sha256=private['all_source_sha256'],
        GPU_manager_CPU_affinity=list(range(96)),GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v4'),indent=2))
    (out/'queue.json').write_text(json.dumps(private['jobs'],indent=2));(out/'finished.json').write_text('{}')
    (out/'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=78,finished=0)))
    for job in selected[:2]:
        destination=out/'runs'/job['id'];destination.mkdir(parents=True)
        shutil.copyfile(root/'runs'/job['id']/'config.json',destination/'config.json')
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v6',str(contract)]
    with(out/'controller.log').open('a')as log:
        child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=process_identity(child.pid)
    (out/'controller_launch.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command),indent=2))
    status_path=out/'analysis/onestep_pipeline_v1/status.json';end=time.monotonic()+25
    while time.monotonic()<end:
        if status_path.exists():break
        assert child.poll()is None,(out/'controller.log').read_text()
        time.sleep(.1)
    assert status_path.exists(),'Observation deadline; inspect the same private child PID before recovery'
    status=json.loads(status_path.read_text());research=json.loads((out/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert status['pid']==child.pid and status['phase']=='pilot'and len(status['rows'])==78
    assert not status['CPU_running']and not status['benchmark_running']and not status['evaluation_failures']
    assert not status['pilot_candidates']and not research['attempts']
    assert all(row['primary_mean']is None and row['teacher_mean']is None for row in status['rows'])
    assert not any(e['label']=='restart_owned_GPU_scheduler'for e in research['events'])
    after=json.loads((out/'queue.json').read_text());assert {j['id']:j for j in after}=={j['id']:j for j in private['jobs']}
    assert process_identity(child.pid)==birth
    os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    sources=[Path(__file__).resolve(),Path(__file__).resolve().with_name('register_full9_prefetch_v1.py'),controller,
        Path(__file__).resolve().with_name('onestep_scheduling_policy_v1.py')]
    result=dict(time=time.time(),status='pass',no_performance_claim=True,
        checks=['Actual controller helpers reuse all9existingconfigurations without a duplicate learner',
            'Removing one prefetched configuration schedules exactly that missing environment',
            'Actual unchanged controller boots with78registeredjobs and keeps phasepilot',
            'Zero fabricated scores, model files, evaluators, benchmarks, GPU managers or learners',
            'No formal pilot eligibility, promotion or fresh training allocation from prefetch alone'],
        source_sha256={str(p):sha(p)for p in sources},all9configurations_unique=True,
        native_promotion_added_jobs_when_all9present=0,native_control_added_jobs_when_one_missing=1,
        private_controller_terminal=True,no_production_queue_or_model_changes=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='pass',output=str(out/'result.json'))),flush=True)


if __name__=='__main__':main()
