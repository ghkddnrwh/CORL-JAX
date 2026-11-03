"""Freeze a teacher-selection ablation before any of its GPU training starts.

This builder writes registration/contract files only. Queue and process
transitions are separate ownership-checked operations, after validation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v2 import (
    IMPLEMENTATIONS, MEAN_TEACHER, configuration, implementation, specification)


RECIPES=(*MEAN_TEACHER.core.RECIPES,'bank_q_a1__meanQ')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    assert not args.registration.exists() and not args.contract.exists()
    parent=root/'registration_v4.json';previous=root/'analysis/execution_contract_v6.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['phase']=='pilot' and state['round']==0 and not state['attempts']
    assert state['jobs']==plan['jobs'] and len(state['jobs'])==51
    assert not status['pilot_candidates']
    ready_path=root/'analysis/mean_teacher_pipeline_v1/readiness_contract_v1.json'
    ready=json.loads(ready_path.read_text());assert ready['frozen']
    for path,digest in ready['source_sha256'].items():assert sha(path)==digest,path
    evidence=[]
    for entry in ready['validation']:
        path=Path(entry['path']);assert sha(path)==entry['sha256']
        result=json.loads(path.read_text());assert result['status']=='pass' and result['no_performance_claim']
        evidence.append(entry)
    result=json.loads((root/'analysis/mean_teacher_pipeline_v1/integration_v1/result.json').read_text())
    maps=[json.loads((Path(row['run'])/'config.json').read_text())['source_sha256'] for row in result['cases']]
    assert maps and all(value==maps[0] for value in maps)
    plan['implementation_training_source_sha256']['mean_teacher']=maps[0]
    plan['implementation_definitions']={entry.tag:asdict(entry) for entry in IMPLEMENTATIONS}
    new=[];expected={}
    tags=['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene']
    for recipe in RECIPES:
        assert recipe not in plan['initial_recipes']
        for environment in plan['pilot_environments']:
            tag=tags[MEAN_TEACHER.core.ENVIRONMENTS.index(environment)]
            rid=f'osm_{tag}_{recipe}_s0';assert not(root/'runs'/rid).exists()
            new.append(dict(id=rid,module=implementation(recipe).runner_module,
                args=dict(env=environment,recipe=recipe,seed=0,steps=1000000)))
            expected[rid]=configuration(environment,recipe,0)
        plan['initial_recipes'].append(recipe)
        plan['initial_recipe_specs'][recipe]=specification(recipe)
    old_jobs=list(plan['jobs']);plan['jobs']+=new;plan['expected_identities'].update(expected)
    assert len(new)==15 and len(plan['jobs'])==66
    sources=dict(plan['all_source_sha256'])
    for path,digest in {**ready['source_sha256'],str(ready_path):sha(ready_path),str(Path(__file__).resolve()):sha(__file__)}.items():
        assert path not in sources or sources[path]==digest,path
        sources[path]=digest
    for path,digest in sources.items():assert sha(path)==digest,path
    plan['all_source_sha256']=sources
    erratum=root/'analysis/public_alpha_midpoint_v1/FQL_ACTOR_AGGREGATION_ERRATUM_KO_v1.json'
    plan['public_raw_midpoint_extension']['official_scope']='Public alpha mapping and meanQ actor aggregation match the bundled official FQL actor. Official q_agg config controls its critic target only. Lean finite-mean backup differs intentionally; this is not full FQL.'
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),
        readiness_contract=str(ready_path),readiness_sha256=sha(ready_path),validation=evidence,
        recipes=list(RECIPES),new_jobs=new,new_expected_identities=expected,
        hypothesis='Distill mean-Q best9 rather than min-Q best9 when teacher head aggregation affects action selection. Pure BC isolates label selection; alpha1 meanQ-only control separates actor guidance aggregation.',
        label_rule='argmax over the mean of two online critic heads on the same nine BC-flow proposals; frozen label per minibatch.',
        critic_rule='Unchanged lean finite empirical-expectile bootstrap, two target critic heads averaged, projected known return range; actor gradients never update critic or BC flow.',
        deployed_policy='One flat actor forward, nine input latent vectors, one output action. Zero candidate actions, ODE integration or critic evaluation.',
        paired_reference='The original min-Q best9 deployment policy from the same checkpoint remains the paired baseline in every run.',
        global_hyperparameters='alpha .3,1,3 and pure imitation shared across environments. No environment-specific choice.',
        controls='Existing bank_bc and bank_q_a1 use min teacher/min guidance; added bank_q_a1__meanQ uses min teacher/mean guidance; bank_q_a1__meanQ__meanTeacher changes both to mean.',
        no_mean_teacher_GPU_efficacy_observed=True,
        observation_scope='Adaptive proposal after partial Ant/Cube outcomes and a single-checkpoint CPU50 teacher ablation. Neither a nine-task result nor a trained one-step mean-teacher outcome.',
        teacher_ablation=str(root/'analysis/teacher_aggregation_v1/matched_CPU6_comparison_audit_v1.json'),
        original51definitions_and_relative_order_preserved=True,performance_and_speed_gates_unchanged=True,
        pending_order='Existing confirmation/missing-pilot priority remains; append fifteen jobs after the previous51 logical jobs.',
        official_actor_metadata_erratum=dict(path=str(erratum),sha256=sha(erratum)))
    plan['mean_teacher_extension']=extension
    assert plan['jobs'][:51]==old_jobs
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=sources,GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v4',
        mean_teacher_extension=extension,public_raw_midpoint_extension=plan['public_raw_midpoint_extension'])
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),new_jobs=len(new),initial_jobs=len(plan['jobs']))))


if __name__=='__main__':main()
