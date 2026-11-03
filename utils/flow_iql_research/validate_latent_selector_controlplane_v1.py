"""Verify native routing, future confirmation, read-only gradients and CPU adoption."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from utils.flow_iql_research import onestep_implementation_registry_v4 as old_registry
from utils.flow_iql_research import onestep_implementation_registry_v5 as registry
from utils.flow_iql_research import scheduler_onestep_deploy_v6 as old_manager
from utils.flow_iql_research import scheduler_onestep_deploy_v7 as manager
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.register_latent_selector_onestep_v1 import build_plan,RECIPES
from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending
from utils.flow_iql_research.activate_latent_selector_onestep_v1 import identity


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output_dir.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out.mkdir(parents=True,exist_ok=False);(out/'analysis').mkdir()
    plan,_=build_plan(root);state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert len(plan['jobs'])==130 and len(state['jobs'])==105
    for job in state['jobs']:
        x=job['args'];assert registry.implementation(x['recipe'])==old_registry.implementation(x['recipe'])
        assert registry.configuration(x['env'],x['recipe'],x['seed'])==plan['expected_identities'][job['id']]
    assert manager.allowed_modules()==old_manager.allowed_modules()|{
        'utils.flow_iql_research.train_latent_selector_onestep_v1','utils.flow_iql_research.train_latent_order_guided_v1'}
    scheduler=json.loads((root/'scheduler_status.json').read_text());assert len(scheduler['running'])==16
    for row in scheduler['running'].values():
        assert identity(row['pid'])==row['process_identity']
        command=Path(f'/proc/{row["pid"]}/cmdline').read_text().split('\0')
        assert command[command.index('-m')+1]in manager.allowed_modules()
    folder=Path(__file__).resolve().parent;controller=folder/'watch_onestep_deploy_v9.py'
    main_node=next(x for x in ast.parse(controller.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='main')
    nodes=[next(x for x in main_node.body if isinstance(x,ast.FunctionDef)and x.name==name)
        for name in ('add_configuration','promote','begin_fresh')]
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(controller),'exec')
    routing=[]
    for recipe in RECIPES:
        captured=[];bindings=dict(core=SimpleNamespace(ENVIRONMENTS=registry.LATENT.core.ENVIRONMENTS),
            configuration=registry.configuration,implementation=registry.implementation,
            append_jobs=lambda new,expected,label,details:captured.append(copy.deepcopy(dict(new=new,expected=expected,label=label))))
        bindings['state']=dict(jobs=copy.deepcopy(plan['jobs']),attempts=[],phase='pilot',active_attempt=None)
        exec(code,bindings);bindings['promote'](dict(recipe=recipe,private_routing_only=True))
        assert len(captured)==1 and len(captured[0]['new'])==4
        assert all(j['module']==registry.implementation(recipe).runner_module for j in captured[0]['new'])
        bindings['state']['jobs']+=captured[0]['new'];captured.clear()
        bindings['begin_fresh'](bindings['state']['attempts'][0])
        assert len(captured)==1 and len(captured[0]['new'])==27
        assert len({(j['args']['env'],j['args']['seed'])for j in captured[0]['new']})==27
        routing.append(dict(recipe=recipe,missing_discovery_jobs=4,fresh_jobs=27))
    queue=json.loads((root/'queue.json').read_text());started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
    started|={j['id']for j in state['jobs']if(root/'runs'/j['id']/'config.json').exists()}
    extended=copy.deepcopy(state);extended.update(jobs=plan['jobs'],identities=plan['expected_identities'])
    combined=queue+plan['latent_selector_extension']['new_jobs'];ordered=order_pending(plan,extended,combined,started)
    new_ids={j['id']for j in plan['latent_selector_extension']['new_jobs']}
    assert [j for j in ordered if j['id']not in new_ids]==queue
    assert all(x==y for x,y in zip(combined,ordered)if x['id']in started)
    assert order_pending(plan,extended,ordered,started)==ordered
    # Exercise the actual gradient producer, including dictionary noise and internally
    # weighted imitation. All source/model hashes checked by the producer itself.
    readiness=json.loads(Path(plan['latent_selector_extension']['readiness']).read_text());gradients=[]
    for recipe in ('latent_order9','latent_diag9','latent_order9_q_a03'):
        row=next(x for x in readiness['rows']if x['recipe']==recipe and x['environment']=='cube-single-play-singletask-v0')
        target=out/(recipe+'_gradient.json');command=[sys.executable,'-u','-m',
            'utils.flow_iql_research.diagnose_onestep_gradients_v6','--run',row['run'],'--step','8','--output',str(target)]
        with(out/(recipe+'_gradient.log')).open('xb')as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        (out/(recipe+'_gradient_handle.json')).write_text(json.dumps(dict(pid=child.pid,birth=identity(child.pid),command=command)))
        assert child.wait()==0,(out/(recipe+'_gradient.log')).read_text()[-4000:]
        v=json.loads(target.read_text());assert v['optimizer_updates_performed']==0
        assert v['distillation_objective_key']=='distill_objective'and v['gradient_additivity_residual_L2']<1e-3
        if '_q_'in recipe:assert v['selector_NLL_coefficient']==.3 and v['Q_guidance_updates_decoder']is False
        else:assert v['Q_gradient_L2']==0
        gradients.append(dict(recipe=recipe,path=str(target),sha256=sha(target)))
    private=copy.deepcopy(plan);private['root']=str(out);registration=out/'private_registration.json'
    registration.write_text(json.dumps(private,indent=2));contract=out/'private_contract.json'
    contract.write_text(json.dumps(dict(registration=str(registration),registration_sha256=sha(registration),
        source_sha256=private['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),
        GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v7'),indent=2))
    private_state={k:copy.deepcopy(state[k])for k in ('round','round_recipes','all_recipes','phase','tested_candidates','attempts','active_attempt')}
    private_state['round_recipes']+=list(RECIPES);private_state['all_recipes']+=list(RECIPES)
    private_state.update(jobs=plan['jobs'],identities=plan['expected_identities'],evaluation_workers={},evaluation_tries={},
        evaluation_failures={},benchmark_workers={},benchmark_results={},events=[])
    directory=out/'analysis/onestep_pipeline_v1';directory.mkdir()
    (directory/'research_state.json').write_text(json.dumps(private_state,indent=2))
    (out/'queue.json').write_text(json.dumps(plan['jobs'],indent=2));(out/'finished.json').write_text('{}')
    (out/'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=130,finished=0)))
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v9',str(contract)]
    with(out/'controller.log').open('xb')as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=identity(child.pid);(out/'controller_handle.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command)))
    end=time.monotonic()+35;statuspath=directory/'status.json'
    while not statuspath.exists()and time.monotonic()<end:
        assert child.poll()is None,(out/'controller.log').read_text()[-5000:]
        time.sleep(.1)
    assert statuspath.exists(),'Observe same owned private parent on timeout; do not restart.'
    status=json.loads(statuspath.read_text());after=json.loads((directory/'research_state.json').read_text())
    assert len(status['rows'])==130 and status['phase']=='confirmation'and after['attempts']==state['attempts']
    assert not status['CPU_running']and not status['benchmark_running']and not status['evaluation_failures']
    assert all(row['primary_mean']is None and row['teacher_mean']is None for row in status['rows'])
    assert not any(x['label']=='restart_owned_GPU_scheduler'for x in after['events'])
    assert identity(child.pid)==birth;os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    result=dict(time=time.time(),status='pass',no_performance_claim=True,no_production_changes=True,
        old105routes_configurations_unchanged=True,private_pending_order_idempotent=True,started_positions_unchanged=True,
        accepts16live_learner_birth_identities=True,promotion_and_fresh_routing=routing,read_only_gradient_evidence=gradients,
        private_controller_rows=130,private_controller_terminal=True,private_GPU_learners_and_evaluators_launched=0,
        source_sha256={**{str(Path(__file__).resolve()):sha(__file__)},**plan['all_source_sha256']})
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',output=str(out/'result.json'))))


if __name__=='__main__':main()
