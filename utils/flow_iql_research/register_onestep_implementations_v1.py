"""Versioned registration builder; never mutates the training queue."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research.onestep_implementation_registry_v1 import IMPLEMENTATIONS,SET_RECIPES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(target,source):
    for path,digest in source.items():
        assert path not in target or target[path]==digest,path
        target[path]=digest


def build_plan(root):
    parent=root/'registration_v1.json';plan=json.loads(parent.read_text())
    previous=root/'analysis/execution_contract_v3.json';old_contract=json.loads(previous.read_text())
    assert old_contract['registration_sha256']==sha(parent)
    ready_path=root/'analysis/set_actor_pipeline_v1/readiness_contract_v1.json'
    ready=json.loads(ready_path.read_text());assert ready['frozen']
    result=json.loads(Path(ready['validation_result']).read_text())
    assert result['status']=='pass' and result['no_performance_claim']
    set_sources=None
    for row in result['cases']:
        conf=json.loads((Path(row['run'])/'config.json').read_text())
        if set_sources is None:set_sources=conf['source_sha256']
        else:assert set_sources==conf['source_sha256']
    assert set_sources
    plan['implementation_training_source_sha256']={'flat':plan['training_source_sha256'],'set':set_sources}
    plan['implementation_definitions']={entry.tag:asdict(entry) for entry in IMPLEMENTATIONS}
    all_sources=dict(plan['all_source_sha256']);merge(all_sources,ready['source_sha256']);merge(all_sources,old_contract['source_sha256'])
    directory=Path(__file__).resolve().parent
    for name in ('onestep_implementation_registry_v1.py','watch_onestep_deploy_v4.py',
                 'scheduler_onestep_deploy_v2.py','inspect_onestep_runtime_v2.py',
                 'register_onestep_implementations_v1.py','validate_onestep_teacher_alignment_v1.py',
                 'validate_onestep_registry_pipeline_v1.py','validate_onestep_scheduler_adoption_v1.py'):
        path=directory/name;merge(all_sources,{str(path):sha(path)})
    for path,digest in all_sources.items():assert sha(path)==digest,path
    plan['all_source_sha256']=all_sources
    plan['second_round_recipes']=[*SET_RECIPES,*plan['adaptive_search']['second_round']]
    plan['adaptive_search']['second_round']=plan['second_round_recipes']
    plan['selection']='Preserve V2 macro eligibility and V3 maximum two exploratory full9/fresh confirmations. Task losses remain diagnostics; no task-specific recipe selection.'
    plan['implementation_extension']=dict(time=time.time(),parent_registration=str(parent),
        parent_registration_sha256=sha(parent),parent_execution_contract=str(previous),
        parent_execution_contract_sha256=sha(previous),set_readiness_contract=str(ready_path),
        set_readiness_sha256=sha(ready_path),set_efficacy_outcomes_observed=False,
        timing='Set architecture CPU validated; registration follows partial flat actor outcomes. New set jobs have no GPU efficacy data.',
        queue_policy='Keep all original48 jobs/identities/order. Add six set recipes only in the next adaptive expansion if existing confirmations do not achieve the goal.',
        unchanged='800K900K1M CPU50, nine-task macro thresholds, fresh seeds, isolated speed gates, 2x8 slots, memory fraction .1.')
    return plan,old_contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True);args=parser.parse_args()
    assert not args.registration.exists() and not args.contract.exists()
    plan,old=build_plan(args.root.resolve())
    evidence=[]
    for name in ('validation_v2','scheduler_adoption_validation_v1'):
        path=args.root.resolve()/'analysis/implementation_registry_v1'/name/'result.json'
        value=json.loads(path.read_text());assert value['status']=='pass' and value['no_performance_claim']
        evidence.append(dict(path=str(path),sha256=sha(path)))
    plan['implementation_extension']['integration_validation']=evidence
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract=dict(old,registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=plan['all_source_sha256'],GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v2',
        implementation_extension=plan['implementation_extension'])
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),contract=str(args.contract),
        registration_sha256=sha(args.registration),contract_sha256=sha(args.contract))))


if __name__=='__main__':main()
