"""Actual-model policy verification and a weightless controller transition."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.flow_iql_research import routed_policy_registry_v1 as registry
from utils.flow_iql_research import onestep_implementation_registry_v5 as native
from utils.flow_iql_research import watch_onestep_deploy_v11 as controller
from utils.flow_iql_research.routed_policy_verifier_v1 import PolicyVerifier,derived_rows
from utils.flow_iql_research.register_routed_policy_pipeline_v1 import build_plan
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def complete_rows(environments,primary=.7,teacher=.72):
    return [dict(run=f'{seed}_{i}',recipe=registry.POLICY,environment=env,seed=seed,
        complete=True,terminal=True,training_complete=True,primary_mean=primary,teacher_mean=teacher,
        actor_parameter_count=1) for seed in [0,24,25,26] for i,env in enumerate(environments)]


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/routed_policy_pipeline_v1/validation_v1'
    area.mkdir(parents=True,exist_ok=True);target=area/'result.json';assert not target.exists()
    plan,contract=build_plan(root);state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    envs=list(controller.core.ENVIRONMENTS);POLICY=registry.POLICY
    for recipe in state['all_recipes']:
        assert registry.implementation(recipe)==native.implementation(recipe),recipe
    configurations=[]
    for env in envs:
        for seed in [0,24,25,26]:
            actual=registry.configuration(env,POLICY,seed)
            assert actual==native.configuration(env,registry.NATIVE,seed)
            assert actual['recipe']==registry.NATIVE
            configurations.append(dict(environment=env,seed=seed,native_identity_exact=True))
    attempt=dict(number=4,recipe=POLICY,fresh_seeds=[24,25,26],stage='fresh')
    rows=complete_rows(envs);assert controller.full_confirmation(rows,attempt,plan)['performance_passed']
    cases=['complete36pass']
    missing=copy.deepcopy(rows);missing[0]['complete']=False
    assert controller.full_confirmation(missing,attempt,plan)is None;cases.append('missing_checkpoint_not_complete')
    failed=complete_rows(envs,.58,.59)
    assert not controller.full_confirmation(failed,attempt,plan)['checks']['seed0_historical_mean'];cases.append('seed0_historical_not_relaxed')
    failed=complete_rows(envs)
    for row in failed:
        if row['seed']!=0:row['primary_mean']=.55;row['teacher_mean']=.57
    result=controller.full_confirmation(failed,attempt,plan)
    assert result['checks']['fresh_paired_mean']and not result['checks']['fresh_historical_mean'];cases.append('fresh_historical_not_relaxed')
    native_leak=[dict(row,recipe=registry.NATIVE,primary_mean=1.) for row in failed]
    assert not controller.full_confirmation(failed+native_leak,attempt,plan)['performance_passed'];cases.append('native_sampler_scores_do_not_leak')
    diagnostics=complete_rows(envs,.75,.7)
    for row in diagnostics:
        if row['environment']==envs[0]:row['primary_mean']=.5
    result=controller.full_confirmation(diagnostics,attempt,plan)
    assert result['performance_passed']and not result['diagnostic_task_loss_cap_satisfied'];cases.append('per_task_cap_diagnostic_only')
    assert controller.discovery_confirmation(complete_rows(envs),attempt,plan,envs)['discovery_passed']
    assert registry.preferred_candidate(state,plan,[])['recipe']==POLICY
    # Use real native models and explicitly exported actor weights, not a toy.
    verifier=controller.Verifier(plan);pv=PolicyVerifier(plan,verifier,sha);models=[]
    for case in ['cube','pz3','cubed','ant']:
        rid=f'osm1_{case}_mdn9_bc_s0';run=root/'runs'/rid
        conf=verifier.config(run,state['identities'][rid]);assert conf
        for step in [800000,900000,1000000]:
            pin=verifier.model(run,conf,step)
            actor=root/f'analysis/mixture_location_deployment_v1/{case}_actor_{step}_component_location_v1.pkl'
            pp=pv.model(run,conf,step,pin,override_actor=actor);models.append(pp)
    rid='osm1_pz3_mdn9_bc_s0';job=next(j for j in state['jobs']if j['id']==rid)
    replay=root/'analysis/mixture_location_deployment_v1/PZ3_1000K_PORTABLE_CPU50_REPLAY_v1.json'
    pin=pv.pins[str(root/'analysis/mixture_location_deployment_v1/pz3_actor_1000000_component_location_v1.pkl')]
    score=pv.evaluation(replay,job,1000000,pin);assert score['successes']==47
    # Alias evaluation failures do not change a native learner's identity or
    # turn an unevaluated alias into a complete row.
    statuses=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())['rows']
    def result_path(rid,step,mode,episodes):return root/'analysis/evaluations_v1'/f'{rid}_{step}_{mode}_n{episodes}.json'
    policy_rows,ready,_=derived_rows(root,state['jobs'],statuses,state,verifier,pv,result_path,
        lambda rid,step,mode,episodes:f'{rid}:{step}:{mode}:{episodes}')
    assert len(policy_rows)==9 and not any(r['complete']for r in policy_rows)
    assert len(ready)>=12 and all(j['args']['recipe']==registry.NATIVE and mode==registry.MODE for j,step,mode,n in ready)
    # The unchanged scheduling helper sees real MDN recipes, including fresh
    # work. Already-started positions and exact job dictionaries are preserved.
    schedule_state=copy.deepcopy(state);schedule_state['attempts'].append(attempt)
    schedule_state['active_attempt']=4;schedule_state['phase']='confirmation'
    fresh=[]
    for seed in [24,25,26]:
        for i,env in enumerate(envs):fresh.append(dict(id=f'test{seed}_{i}',module=native.implementation(registry.NATIVE).runner_module,
            args=dict(env=env,recipe=registry.NATIVE,seed=seed,steps=1000000)))
    schedule_state['jobs']+=fresh;queue=schedule_state['jobs'];started={queue[0]['id'],queue[4]['id']}
    ordered=registry.order_pending(plan,schedule_state,queue,started)
    assert all(a==b for a,b in zip(queue,ordered)if a['id']in started)
    assert next(j for j in ordered if j['id']not in started)['id']in {j['id']for j in fresh}
    assert {j['id']for j in ordered}=={j['id']for j in queue}
    # A real new controller process gets an empty filesystem for weights.
    # It must supersede attempt3 transparently, reuse all nine existing MDN
    # definitions, and stop at discovery with zero new GPU jobs/evaluations.
    fixture=area/'controller_fixture';assert not fixture.exists()
    (fixture/'analysis/onestep_pipeline_v1').mkdir(parents=True);(fixture/'runs').mkdir()
    fixture_plan=copy.deepcopy(plan);fixture_plan['root']=str(fixture)
    fixture_state=copy.deepcopy(state)
    fixture_state.update(evaluation_workers={},evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={})
    save(fixture/'registration.json',fixture_plan)
    fixture_contract=copy.deepcopy(contract);fixture_contract.update(registration=str(fixture/'registration.json'),
        registration_sha256=sha(fixture/'registration.json'),source_sha256=plan['all_source_sha256'])
    save(fixture/'contract.json',fixture_contract)
    save(fixture/'analysis/onestep_pipeline_v1/research_state.json',fixture_state)
    save(fixture/'queue.json',fixture_state['jobs'])
    save(fixture/'scheduler_status.json',dict(pid=os.getpid(),running={}))
    save(fixture/'finished.json',{})
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v11',str(fixture/'contract.json')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (area/'fixture_controller.log').open('xb') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
    handle=dict(pid=child.pid,birth=controller.process_identity(child.pid),command=command)
    save(area/'fixture_controller_handle.json',handle)
    deadline=time.monotonic()+50;fixture_status=None
    try:
        while time.monotonic()<deadline:
            assert child.poll()is None,(area/'fixture_controller.log').read_text()[-3000:]
            status_path=fixture/'analysis/onestep_pipeline_v1/status.json'
            if status_path.exists():
                fixture_status=json.loads(status_path.read_text());break
            time.sleep(.2)
        assert fixture_status is not None,'Fixture observation timeout; handle retained for root inspection.'
        after=json.loads((fixture/'analysis/onestep_pipeline_v1/research_state.json').read_text())
        assert after['active_attempt']==4 and after['phase']=='confirmation'
        assert after['attempts'][3]['status']=='superseded_by_registered_decoder_candidate'
        assert after['attempts'][4]['recipe']==POLICY and after['attempts'][4]['fresh_seeds']==[24,25,26]
        assert after['attempts'][4]['stage']=='discovery'
        assert after['jobs']==fixture_state['jobs'] and after['identities']==fixture_state['identities']
        assert not after['evaluation_workers']and not after['benchmark_workers']
        assert len(fixture_status['rows'])==151 and fixture_status['planned_GPU_jobs']==142
        assert json.loads((fixture/'queue.json').read_text()) and len(json.loads((fixture/'queue.json').read_text()))==142
        receipt=fixture/'analysis/onestep_pipeline_v1/registrations/confirmation_attempt4.json'
        assert json.loads(receipt.read_text())['new_jobs']==[]
        fixture_result=dict(status='pass',GPU_jobs=142,policy_rows=9,total_status_rows=151,
            actual_new_GPU_jobs=0,actual_evaluation_workers=0,receipt_sha256=sha(receipt),
            all_native_identities_exact=True,alias_never_used_as_training_recipe=True)
    finally:
        # This is the exact owned CPU controller parent in the weightless
        # fixture. No learner, evaluator, or gradient producer was launched.
        if child.poll()is None:
            assert controller.process_identity(child.pid)==handle['birth']
            assert str(fixture/'contract.json') in Path(f'/proc/{child.pid}/cmdline').read_text().split('\0')
            os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    result=dict(time=time.time(),status='pass',source_sha256=plan['all_source_sha256'],no_performance_claim=True,
        actual_native_models_verified=len(models),actual_model_pins=models,actual_existing_CPU50_point=score,
        configurations=configurations,unchanged_macro_gate_cases=cases,controller_fixture=fixture_result,
        native_job_definitions_and_identities_preserved=True,no_real_GPU_jobs_started=True,
        no_real_primary_results_modified=True)
    save(target,result)
    print(json.dumps(dict(status='pass',actual_native_models=12,actual_CPU50_successes=score['successes'],
        gate_cases=len(cases),fixture=fixture_result)))


if __name__=='__main__':main()
