"""Complete missing hard-task controls using the already frozen flat learner."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v5 import configuration,specification,implementation


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/best9_guidance_hard_tasks_v1';area.mkdir(exist_ok=True)
    parent=root/'registration_v14.json';previous=root/'analysis/execution_contract_v16.json'
    target=root/'registration_v15.json';execution=root/'analysis/execution_contract_v17.json'
    assert not target.exists()and not execution.exists()
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert contract['registration_sha256']==sha(parent)and state['jobs']==plan['jobs']and len(state['jobs'])==130
    assert state['phase']=='confirmation'and state['active_attempt']==2 and state['attempts'][2]['recipe']=='set_bc'
    recipes=['bank_q_a1','bank_q_a03','bank_q_a3','bank_q_a3__spreadQ']
    tasks=[('cube','cube-single-play-singletask-v0'),('pz3','puzzle-3x3-play-singletask-v0')]
    new=[];identities={}
    for recipe in recipes:
        assert implementation(recipe).tag=='flat'
        for tag,env in tasks:
            assert not any(j['args']['recipe']==recipe and j['args']['env']==env and j['args']['seed']==0 for j in state['jobs'])
            rid=f'osbq_{tag}_{recipe}_s0';assert not(root/'runs'/rid).exists()
            new.append(dict(id=rid,module=implementation(recipe).runner_module,args=dict(env=env,recipe=recipe,seed=0,steps=1000000)))
            identities[rid]=configuration(env,recipe,0)
            cfg=identities[rid]['config'];reference=configuration(env,'bank_bc',0)['config']
            assert {k:v for k,v in cfg.items()if k!='onestep_specification'}=={k:v for k,v in reference.items()if k!='onestep_specification'}
            s=cfg['onestep_specification'];assert s['target']=='best9'and s['objective']=='maximize'and s['input']=='bank'
    old=list(plan['jobs']);first=max(i for i,j in enumerate(old)if j['id'].startswith('ospS_'))+1
    jobs=old[:first]+new[:2]+old[first:]
    after=max(i for i,j in enumerate(jobs)if j['id']=='osz1_pz3_latent_order9_q_a3_s0')+1
    jobs=jobs[:after]+new[2:]+jobs[after:]
    assert [j for j in jobs if j['id']not in identities]==old and len(jobs)==138
    added=[recipe for recipe in recipes if recipe not in state['all_recipes']]
    plan['jobs']=jobs;plan['expected_identities'].update(identities)
    for recipe in added:
        plan['initial_recipes'].append(recipe);plan['initial_recipe_specs'][recipe]=specification(recipe)
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),new_jobs=new,new_expected_identities=identities,
        added_recipes=added,recipes=recipes,scope='Missing environment coverage of frozen existing best9BC+Q loss, not a new architecture or implementation.',
        rationale='Core user-proposed best9 label plus direct Q gradient was trained on Ant/CubeD/HMM only. Its Puzzle/Cube behavior was untested; private weighted-MSE/NLL studies do not replace this ablation.',
        globals='Existing alpha .3/1/3 withabsQnormalization; additional alpha3usingwithin-state BC9Q spread floor1. Same recipes in both hardtasks; no environment-specific alpha choice.',
        queue_rule='Insert2bank_q_a1 hardcontrols before unstarted new latent hardtasks; insert6remainingcontrols after the10latent hardtasks. Preserve all130old definitions,identities,relativeorder andstartedpositions.',
        original_Q_and_BC_Flow_config_unchanged=True,training_sources_unchanged=True,
        active_SetBC_full9_attempt_preserved=2,all_final_performance_fresh_and_speed_gates_unchanged=True,
        no_fresh_seed_allocation=True,no_current_hardtask_efficacy_observed=True)
    proposal=area/'prospective_experiment_v1.json';assert not proposal.exists();proposal.write_text(json.dumps(extension,indent=2))
    sources=dict(plan['all_source_sha256'])
    for p in [Path(__file__).resolve(),Path(__file__).with_name('activate_best9_guidance_hard_tasks_v1.py').resolve(),proposal]:sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items())
    plan.update(all_source_sha256=sources,best9_guidance_hard_extension=extension)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=sources,
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),best9_guidance_hard_extension=extension)
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),execution=str(execution),execution_sha256=sha(execution),jobs=138,new_jobs=8,added_recipes=added)))


if __name__=='__main__':main()
