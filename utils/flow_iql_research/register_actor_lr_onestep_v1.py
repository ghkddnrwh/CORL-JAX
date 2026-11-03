"""Register a global actor-lr multiplier using the frozen existing flat runner.

Writes prospective immutable registration/contract only; activation is separate.
"""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v4 import configuration,specification,implementation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve();area=root/'analysis/actor_lr_joint_pipeline_v1'
    parent=root/'registration_v12.json';previous=root/'analysis/execution_contract_v14.json'
    target=root/'registration_v13.json';execution=root/'analysis/execution_contract_v15.json'
    assert not target.exists()and not execution.exists()
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    ready_path=area/'readiness_result_v1.json';ready=json.loads(ready_path.read_text())
    assert ready['status']=='pass'and ready['no_performance_claim']and len(ready['rows'])==5
    for p,h in ready['source_sha256'].items():assert sha(p)==h,p
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs']and len(state['jobs'])==100
    assert state['phase']=='confirmation'and state['active_attempt']==1
    assert state['attempts'][1]['recipe']=='bank_bc'and state['attempts'][1]['stage']=='discovery'
    recipe='bank_bc__lr2';assert recipe not in state['all_recipes']
    assert implementation(recipe).tag=='flat'
    specs=[('cube','cube-single-play-singletask-v0'),('pz3','puzzle-3x3-play-singletask-v0'),
           ('ant','antmaze-large-navigate-singletask-v0'),('cubed','cube-double-play-singletask-v0'),
           ('hmmed','humanoidmaze-medium-navigate-singletask-v0')]
    new=[];expected={}
    for tag,environment in specs:
        rid=f'osl2_{tag}_bank_bc_lr2_s0';assert not(root/'runs'/rid).exists()
        job=dict(id=rid,module='utils.flow_iql_research.train_onestep_deploy_v1',
                 args=dict(env=environment,recipe=recipe,seed=0,steps=1000000))
        new.append(job);expected[rid]=configuration(environment,recipe,0)
        cfg=expected[rid]['config'];base=configuration(environment,'bank_bc',0)['config']
        assert cfg['onestep_specification']['actor_lr_multiplier']==2
        assert {k:v for k,v in cfg.items()if k!='onestep_specification'}=={k:v for k,v in base.items()if k!='onestep_specification'}
    old=list(plan['jobs']);old_ids={j['id']for j in old}
    first=next(i for i,j in enumerate(old)if j['id']=='osm1_ant_mdn9_bc_s0')
    jobs=old[:first]+new[:2]+old[first:]
    after_set=max(i for i,j in enumerate(jobs)if j['id'].startswith('ospS_'))+1
    jobs=jobs[:after_set]+new[2:]+jobs[after_set:]
    assert [j for j in jobs if j['id']in old_ids]==old and len(jobs)==105
    plan['jobs']=jobs;plan['expected_identities'].update(expected)
    plan['initial_recipes']=[*plan['initial_recipes'],recipe]
    plan['initial_recipe_specs'][recipe]=specification(recipe)
    proposal=area/'prospective_experiment_v1.json';assert not proposal.exists()
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        previous_contract=str(previous),previous_contract_sha256=sha(previous),recipe=recipe,
        new_jobs=new,new_expected_identities=expected,implementation_tag='flat',
        only_change='Global deploymentactorAdamlearningrate x2:0.0003->0.0006; oneAdamstep perQstep, alpha1purepairedMSE.',
        rationale='Current coupledactor retains conditional imitation residual and weak teacher-noise covariance. Test actor tracking rate directly; no critic action gradient or new loss/architecture.',
        scope='Configuration addition through existing frozen lr suffix; not a new training branch.',
        initial_hard_tasks=[specs[0][1],specs[1][1]],original_three_pilots_unchanged=plan['pilot_environments'],
        queue_rule='Insert2hardtaskjobs before remainingMDNpilots; insert3originalpilots after existingSetfull9prefetch. Preserveoriginal100definitions,identitiesandrelativeorder,startedpositionsandactivebankBCattempt.',
        readiness=dict(path=str(ready_path),sha256=sha(ready_path)),
        original100_training_sources_unchanged=True,all_performance_fresh_speed_gates_unchanged=True,
        no_fresh_seed_allocation=True,no_per_environment_parameter_selection=True,no_GPU_lr2_efficacy_observed=True)
    proposal.write_text(json.dumps(extension,indent=2))
    sources=dict(plan['all_source_sha256'])
    for p in [Path(__file__).resolve(),Path(__file__).with_name('activate_actor_lr_onestep_v1.py').resolve(),
              ready_path,area/'readiness_proposal_v1.json',proposal]:sources[str(p)]=sha(p)
    sources.update(ready['source_sha256'])
    for p,h in sources.items():assert sha(p)==h,p
    plan['all_source_sha256']=sources;plan['actor_lr_extension']=extension
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=sources,
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),actor_lr_extension=extension)
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),
        contract=str(execution),contract_sha256=sha(execution),new_jobs=5,total_jobs=105)))


if __name__=='__main__':main()
