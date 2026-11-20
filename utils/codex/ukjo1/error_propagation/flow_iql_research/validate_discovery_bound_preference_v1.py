"""Pure bound cases and actual weightless-controller transition/readiness."""
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
from utils.flow_iql_research.discovery_bound_policy_v1 import empirical_bound,preferred_candidate
from utils.flow_iql_research.register_discovery_bound_preference_v1 import build_plan
from utils.flow_iql_research.onestep_implementation_registry_v5 import implementation,configuration
from utils.flow_iql_research.activate_mixture_full9_prefetch_v1 import identity
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    root=Path(sys.argv[1]).resolve();out=root/'analysis/discovery_bound_preference_v1/validation_v1'
    out.mkdir(parents=True,exist_ok=False);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,_=build_plan(root);state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    envs=plan['scheduling_amendment']['all_nine_environments'];steps=['800000','900000','1000000']
    def fake(counts):
        rows=[dict(run=f'private_{i}',recipe='private',seed=0,environment=e,per_step={'primary':{}})for i,e in enumerate(envs)]
        for i,count in enumerate(counts):rows[i//3]['per_step']['primary'][steps[i%3]]=count/50
        return rows
    cases=[]
    for counts,expected in [([],False),([35]+[29]*22,False),([30]+[28]*24,True),([30]*27,False),([0]*27,True)]:
        bound=empirical_bound(fake(counts),'private',plan)
        assert bound['impossible_even_if_all_missing_perfect']==expected
        assert bound['minimum_total_successes']==804
        cases.append(dict(observed_points=len(counts),observed_successes=sum(counts),expected_futile=expected,upper_bound=bound['upper_bound']))
    assert empirical_bound(fake([])[:8],'private',plan)is None
    assert preferred_candidate(state,plan,[])is None
    failed=copy.deepcopy(state);failed['attempts'][2]['status']='discovery_historical_gate_impossible';failed['phase']='pilot'
    failed['tested_candidates'].append('set_bc')
    selected=preferred_candidate(failed,plan,[])
    assert selected['recipe']=='mdn9_bc'and selected['eligible']is False and selected['no_original_pilot_pass_claim']
    trained=copy.deepcopy(failed);trained['tested_candidates'].append('mdn9_bc');assert preferred_candidate(trained,plan,[])is None
    folder=Path(__file__).resolve().parent;controller=folder/'watch_onestep_deploy_v10.py'
    main_node=next(n for n in ast.parse(controller.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nodes=[next(n for n in main_node.body if isinstance(n,ast.FunctionDef)and n.name==key)for key in ['add_configuration','promote','begin_fresh']]
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(controller),'exec')
    captured=[];bindings=dict(core=SimpleNamespace(ENVIRONMENTS=envs),configuration=configuration,implementation=implementation,
        state=copy.deepcopy(failed),append_jobs=lambda new,expected,label,details:captured.append(copy.deepcopy(dict(new=new,expected=expected,label=label))))
    exec(code,bindings);bindings['promote'](selected)
    assert bindings['state']['active_attempt']==3 and bindings['state']['attempts'][3]['fresh_seeds']==[21,22,23]
    assert len(captured)==1 and not captured[0]['new']
    captured.clear();bindings['begin_fresh'](bindings['state']['attempts'][3])
    assert len(captured)==1 and len(captured[0]['new'])==27
    assert {j['args']['seed']for j in captured[0]['new']}=={21,22,23}
    assert all(j['module']=='utils.flow_iql_research.train_mixture_onestep_v1'for j in captured[0]['new'])
    private=copy.deepcopy(plan);private['root']=str(out)
    registration=out/'private_registration.json';registration.write_text(json.dumps(private,indent=2))
    contract=out/'private_contract.json';contract.write_text(json.dumps(dict(registration=str(registration),registration_sha256=sha(registration),
        source_sha256=private['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v7'),indent=2))
    ps={k:copy.deepcopy(state[k])for k in ['round','round_recipes','all_recipes','phase','tested_candidates','attempts','active_attempt']}
    ps.update(jobs=plan['jobs'],identities=plan['expected_identities'],evaluation_workers={},evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={},events=[])
    directory=out/'analysis/onestep_pipeline_v1';directory.mkdir(parents=True)
    (directory/'research_state.json').write_text(json.dumps(ps,indent=2));(out/'queue.json').write_text(json.dumps(plan['jobs'],indent=2))
    (out/'finished.json').write_text('{}');(out/'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=142,finished=0)))
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v10',str(contract)]
    with(out/'controller.log').open('xb')as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=identity(child.pid);(out/'controller_handle.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command)))
    end=time.monotonic()+35;sp=directory/'status.json'
    while not sp.exists()and time.monotonic()<end:
        assert child.poll()is None,(out/'controller.log').read_text()[-4000:]
        time.sleep(.1)
    assert sp.exists(),'Observe same private handle on timeout; never restart solely for observation expiry.'
    status=json.loads(sp.read_text());after=json.loads((directory/'research_state.json').read_text())
    assert len(status['rows'])==142 and after['attempts']==state['attempts']and after['active_attempt']==2
    assert not status['CPU_running']and not status['benchmark_running']and not status['evaluation_failures']
    assert all(r['primary_mean']is None and r['teacher_mean']is None for r in status['rows'])
    assert not any(e['label']=='restart_owned_GPU_scheduler'for e in after['events'])
    assert identity(child.pid)==birth;os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    result=dict(time=time.time(),status='pass',no_performance_claim=True,pure_math_cases=cases,
        missing_environment_does_not_trigger_bound=True,preference_only_after_registered_failure=True,
        incomplete_pilot_eligibility_preserved_false=True,existing9mixturejobs_reused=True,
        future_fresh_routing_jobs=27,future_fresh_seed_ids=[21,22,23],no_actual_fresh_allocation=True,
        actual_weightless_controller_rows=142,actual_weightless_controller_preserved_Set_attempt=True,
        actual_private_controller_terminal=True,no_production_mutations=True,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),**plan['all_source_sha256']})
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(status='pass',output=str(out/'result.json'))))


if __name__=='__main__':main()
