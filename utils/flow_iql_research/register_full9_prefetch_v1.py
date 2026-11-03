"""Register six missing seed0tasks for an existing recipe, without promotion."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v2 import FLAT,configuration


def build_plan(root):
    parent=root/'registration_v6.json';previous=root/'analysis/execution_contract_v8.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    proposal_path=root/'analysis/full9_prefetch_v1/proposal_v1.json';proposal=json.loads(proposal_path.read_text())
    assert proposal['parent_registration_sha256']==sha(parent)
    assert proposal['recipe']=='fqln_a03'and proposal['alpha']==.3
    assert proposal['selection_is_exploratory']and proposal['no_pilot_eligibility_claim']and proposal['no_fresh_seed_allocation']
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    scheduler=json.loads((root/'scheduler_status.json').read_text())
    assert state['phase']=='pilot'and state['round']==0 and not state['attempts']and not status['pilot_candidates']
    assert state['jobs']==plan['jobs']and len(state['jobs'])==72
    old_recipe_jobs=[j for j in plan['jobs']if j['args']['recipe']=='fqln_a03']
    assert len(old_recipe_jobs)==3 and {j['args']['env']for j in old_recipe_jobs}==set(plan['pilot_environments'])
    expected_remaining=set(FLAT.core.ENVIRONMENTS)-set(plan['pilot_environments'])
    assert set(proposal['remaining_environments'])==expected_remaining and len(expected_remaining)==6
    for item in proposal['observations']:assert sha(item['path'])==item['sha256']
    new=[];expected={};tags=['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene']
    for environment in proposal['remaining_environments']:
        tag=tags[FLAT.core.ENVIRONMENTS.index(environment)];rid=f'osp0_{tag}_fqln_a03_s0'
        assert not(root/'runs'/rid).exists()
        new.append(dict(id=rid,module=FLAT.runner_module,args=dict(env=environment,recipe='fqln_a03',seed=0,steps=1000000)))
        identity=configuration(environment,'fqln_a03',0);spec=identity['specification']
        assert spec['effective_alpha']==.3 and spec['target']=='flow1'and spec['q_aggregation']=='mean'and spec['q_scale']=='absQ'
        expected[rid]=identity
    started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
    started|={j['id']for j in state['jobs']if(root/'runs'/j['id']/'config.json').exists()}
    insertion=next(i for i,j in enumerate(state['jobs'])if j['id']not in started)
    old=list(plan['jobs']);plan['jobs']=old[:insertion]+new+old[insertion:]
    old_ids={j['id']for j in old};assert[j for j in plan['jobs']if j['id']in old_ids]==old
    assert len({j['id']for j in plan['jobs']})==78
    plan['expected_identities'].update(expected)
    sources=dict(plan['all_source_sha256']);directory=Path(__file__).resolve().parent
    for path in (Path(__file__).resolve(),directory/'validate_full9_prefetch_v1.py',proposal_path):sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),proposal=str(proposal_path),proposal_sha256=sha(proposal_path),
        recipe='fqln_a03',new_jobs=new,new_expected_identities=expected,
        mode='exploratory_seed0_prefetch_not_formal_promotion',no_new_algorithm_sources=True,
        no_pilot_eligibility_claim=True,no_fresh_seed_allocation=True,
        unchanged_pilot_recipes=plan['initial_recipes'],unchanged_pilot_job_count=72,
        queue_rule=proposal['queue_rule'],reuse_rule=proposal['reuse_rule'],inserted_before_logical_index=insertion,
        old72definitions_and_relative_logical_order_preserved=True,
        all_performance_and_speed_gates_unchanged=True,observations=proposal['observations'])
    plan.update(all_source_sha256=sources,full9_prefetch_extension=extension)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    return plan,contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();assert not args.registration.exists()and not args.contract.exists()
    plan,contract=build_plan(args.root.resolve())
    evidence_path=args.root.resolve()/'analysis/full9_prefetch_v1/validation_v1/result.json'
    evidence=json.loads(evidence_path.read_text());assert evidence['status']=='pass'and evidence['no_performance_claim']
    for path,digest in evidence['source_sha256'].items():assert sha(path)==digest,path
    plan['full9_prefetch_extension']['validation']=dict(path=str(evidence_path),sha256=sha(evidence_path))
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=plan['all_source_sha256'],full9_prefetch_extension=plan['full9_prefetch_extension'])
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),jobs=78,new_prefetch_jobs=6,pilot_recipes_unchanged=True)))


if __name__=='__main__':main()
