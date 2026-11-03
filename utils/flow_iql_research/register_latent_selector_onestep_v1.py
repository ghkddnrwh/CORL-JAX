"""Prospective exploratory native joint actors; retain all previous definitions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research import onestep_implementation_registry_v5 as registry


RECIPES=(*registry.LATENT_RECIPES,*registry.GUIDED_RECIPES)


def build_plan(root):
    parent=root/'registration_v13.json';previous=root/'analysis/execution_contract_v15.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    area=root/'analysis/latent_selector_joint_pipeline_v1';ready_path=area/'readiness_result_v1.json'
    ready=json.loads(ready_path.read_text());assert ready['status']=='pass' and ready['no_performance_claim'] and len(ready['rows'])==13
    for p,h in ready['source_sha256'].items():assert sha(p)==h,p
    for e in ready['evidence']:assert sha(e['path'])==e['sha256']
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs'] and len(state['jobs'])==105
    assert state['identities']==plan['expected_identities']
    assert state['phase']=='confirmation' and state['active_attempt']==1 and state['attempts'][1]['recipe']=='bank_bc'
    assert all(r not in state['all_recipes'] for r in RECIPES)
    cases=[('cube','cube-single-play-singletask-v0'),('pz3','puzzle-3x3-play-singletask-v0'),
           ('ant','antmaze-large-navigate-singletask-v0'),('cubed','cube-double-play-singletask-v0'),
           ('hmmed','humanoidmaze-medium-navigate-singletask-v0')]
    new=[];expected={}
    for phase in (cases[:2],cases[2:]):
        for recipe in RECIPES:
            for tag,env in phase:
                rid=f'osz1_{tag}_{recipe}_s0';assert not(root/'runs'/rid).exists()
                item=registry.implementation(recipe)
                new.append(dict(id=rid,module=item.runner_module,args=dict(env=env,recipe=recipe,seed=0,steps=1000000)))
                expected[rid]=registry.configuration(env,recipe,0)
                assert expected[rid]['config']['num_samples']==9
    old=list(plan['jobs']);first=max(i for i,j in enumerate(old)if j['id'].startswith('ospS_'))+1
    jobs=old[:first]+new[:10]+old[first:]
    after_lr=max(i for i,j in enumerate(jobs)if j['id'].startswith('osl2_'))+1
    jobs=jobs[:after_lr]+new[10:]+jobs[after_lr:]
    assert [j for j in jobs if j['id']not in expected]==old and len(jobs)==130
    plan['jobs']=jobs;plan['expected_identities'].update(expected)
    plan['initial_recipes']=[*plan['initial_recipes'],*RECIPES]
    for recipe in RECIPES:plan['initial_recipe_specs'][recipe]=registry.specification(recipe)
    for item in (registry.LATENT,registry.GUIDED):
        plan['implementation_definitions'][item.tag]=asdict(item)
        plan['implementation_training_source_sha256'][item.tag]=ready['implementation_training_source_sha256'][item.tag]
    proposal=area/'prospective_experiment_v1.json';assert proposal.exists()
    p=json.loads(proposal.read_text());assert p['recipes']==list(RECIPES) and p['no_native_latent_GPU_efficacy_observed']
    assert p['parent_registration_sha256']==sha(parent) and p['all_final_gates_unchanged']
    sources={**plan['all_source_sha256'],**ready['source_sha256']}
    folder=Path(__file__).resolve().parent
    names=['register_latent_selector_onestep_v1.py','activate_latent_selector_onestep_v1.py',
        'validate_latent_selector_controlplane_v1.py','onestep_implementation_registry_v5.py',
        'scheduler_onestep_deploy_v7.py','watch_onestep_deploy_v9.py','inspect_onestep_runtime_v6.py',
        'watch_onestep_gradients_v6.py','diagnose_onestep_gradients_v6.py']
    for path in [*(folder/n for n in names),ready_path,proposal]:sources[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in sources.items())
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),proposal=str(proposal),proposal_sha256=sha(proposal),
        readiness=str(ready_path),readiness_sha256=sha(ready_path),recipes=list(RECIPES),new_jobs=new,new_expected_identities=expected,
        selection_is_exploratory=True,not_qualified_by_private_fixed_teacher_CPU_results=True,
        active_bankBC_attempt_unchanged=True,no_fresh_seed_allocation=True,old105definitions_and_relative_order_unchanged=True,
        all_performance_speed_and_fresh_gates_unchanged=True,
        queue_rule='Retain Setfull9prefetch; insert10newhardtasks before remainingLR2pilots,15newpilotjobs afterLR2pilots; keep old relative order and all started positions.')
    plan.update(all_source_sha256=sources,latent_selector_extension=extension)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    return plan,contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();root=a.root.resolve()
    target=root/'registration_v14.json';execution=root/'analysis/execution_contract_v16.json';gradient=root/'analysis/gradient_execution_contract_v6.json'
    assert not any(p.exists()for p in [target,execution,gradient])
    plan,contract=build_plan(root)
    vpath=root/'analysis/latent_selector_joint_pipeline_v1/controlplane_validation_v1/result.json';v=json.loads(vpath.read_text())
    assert v['status']=='pass' and v['no_performance_claim']
    for p,h in v['source_sha256'].items():assert sha(p)==h,p
    plan['latent_selector_extension']['integration_validation']=dict(path=str(vpath),sha256=sha(vpath))
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    previous=root/'analysis/execution_contract_v15.json'
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=plan['all_source_sha256'],
        GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v7',
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        latent_selector_extension=plan['latent_selector_extension'])
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    oldg=root/'analysis/gradient_execution_contract_v5.json';g=json.loads(oldg.read_text())
    g.update(time=time.time(),source_sha256=plan['all_source_sha256'],previous_contract=str(oldg),
        previous_contract_sha256=sha(oldg),implementation_registry='onestep_implementation_registry_v5',
        execution='V6 supports dictionary latent labels, exact weighted NLL+decoder objective and actual selector_alpha; older live V1..V5 diagnostic producers adopted without signals.',
        implementation_validation=[*g.get('implementation_validation',[]),dict(path=str(vpath),sha256=sha(vpath),no_performance_claim=True)])
    gradient.write_text(json.dumps(g,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),
        execution=str(execution),execution_sha256=sha(execution),gradient=str(gradient),gradient_sha256=sha(gradient),jobs=130,new_jobs=25)))


if __name__=='__main__':main()
