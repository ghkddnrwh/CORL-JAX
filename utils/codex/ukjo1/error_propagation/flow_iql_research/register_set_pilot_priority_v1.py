"""Advance two already validated set-actor recipes into the pilot round.

No learner/evaluation implementation or success gate changes. This builder
freezes the scheduling amendment before any of these set learners starts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research.onestep_implementation_registry_v2 import (
    SET, configuration, specification)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    assert not args.registration.exists()and not args.contract.exists()
    parent=root/'registration_v5.json';previous=root/'analysis/execution_contract_v7.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    scheduler=json.loads((root/'scheduler_status.json').read_text())
    assert state['jobs']==plan['jobs']and len(state['jobs'])==66
    assert state['phase']=='pilot'and state['round']==0 and not state['attempts']and not status['pilot_candidates']
    ready_path=root/'analysis/set_actor_pipeline_v1/readiness_contract_v1.json'
    ready=json.loads(ready_path.read_text());assert ready['frozen']
    for path,digest in ready['source_sha256'].items():assert sha(path)==digest,path
    validation_path=Path(ready['validation_result']);validation=json.loads(validation_path.read_text())
    assert validation['status']=='pass'and validation['no_performance_claim']
    for row in validation['cases']:
        conf=json.loads((Path(row['run'])/'config.json').read_text())
        assert conf['source_sha256']==plan['implementation_training_source_sha256']['set']
    recipes=['set_bc','set_q_a1'];new=[];expected={}
    tags=['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene']
    for recipe in recipes:
        assert recipe in plan['second_round_recipes']and recipe not in plan['initial_recipes']
        for environment in plan['pilot_environments']:
            tag=tags[SET.core.ENVIRONMENTS.index(environment)];rid=f'oss_{tag}_{recipe}_s0'
            assert not(root/'runs'/rid).exists()
            new.append(dict(id=rid,module=SET.runner_module,args=dict(env=environment,recipe=recipe,seed=0,steps=1000000)))
            expected[rid]=configuration(environment,recipe,0)
        plan['initial_recipes'].append(recipe);plan['initial_recipe_specs'][recipe]=specification(recipe)
    started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
    started|={job['id']for job in state['jobs']if(root/'runs'/job['id']/'config.json').exists()}
    insertion=next(i for i,job in enumerate(state['jobs'])if job['id']not in started)
    old=list(plan['jobs']);plan['jobs']=old[:insertion]+new+old[insertion:]
    old_ids={job['id']for job in old}
    assert [job for job in plan['jobs']if job['id']in old_ids]==old
    assert len({job['id']for job in plan['jobs']})==72
    plan['expected_identities'].update(expected)
    evidence_paths=[root/'analysis/Ant_occupancy_diagnosis_v1/comparison_audit_v1.json',
        root/'analysis/late_extraction_diagnosis_v1/CubeD800K_three_alpha_review_v1.json']
    for path in evidence_paths:assert json.loads(path.read_text())['status']=='audited'
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),
        recipes=recipes,new_jobs=new,new_expected_identities=expected,
        readiness=dict(path=str(ready_path),sha256=sha(ready_path)),
        implementation_validation=dict(path=str(validation_path),sha256=sha(validation_path)),
        observations=[dict(path=str(path),sha256=sha(path))for path in evidence_paths],
        rationale='Best9 is a function of an unordered latent set. Shared latent features and symmetric aggregation may make this function easier to approximate than a flattened input. Try one pure imitation and one alpha1 Q-guided set actor, with frozen prior implementations.',
        selection_scope='Alpha1 selected as a common middle setting with an existing flat control, not as a per-environment optimum.',
        no_set_GPU_efficacy_observed=True,no_new_algorithm_sources=True,
        timing='Adaptive architecture prioritization after partial flat actor outcomes, before any completed3taskpilot/full9/freshsetoutcome.',
        queue_rule='Keep every prior66definition and relative logical order; insert6setjobs before first unstarted logical job. Existing confirmation and missing-third-pilot priorities still apply. Preserve started positions.',
        inserted_before_logical_index=insertion,
        next_round='The existing expansion already filters initial recipes, so these two are not duplicated when the remaining set recipes are considered.',
        paired_reference_and_all_performance_speed_gates_unchanged=True,
        deployed_policy='One set actor forward, one action. Shared latent features are not candidate actions; no BC flow or critic at inference.',
        warning='A synthetic/8update implementation pass is not a learned policy performance or isolated speed result. No completion claim.')
    sources=dict(plan['all_source_sha256'])
    for path in [Path(__file__).resolve(),*evidence_paths]:sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    plan.update(all_source_sha256=sources,set_pilot_priority_extension=extension)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=sources,set_pilot_priority_extension=extension)
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),jobs=72,added_jobs=6)))


if __name__=='__main__':main()
