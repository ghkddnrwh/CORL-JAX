"""Register one missing global raw-alpha midpoint; no algorithm changes."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v1 import configuration,specification


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();assert not args.registration.exists()and not args.contract.exists()
    parent=root/'registration_v3.json';previous=root/'analysis/execution_contract_v5.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    proposal_path=root/'analysis/public_alpha_midpoint_v1/proposal_v1.json'
    proposal=json.loads(proposal_path.read_text());assert proposal['parent_registration_sha256']==sha(parent)
    assert proposal['recipe']=='fqlraw_x3'and proposal['all_performance_and_speed_gates_unchanged']
    retro_path=Path(proposal['retrospective_review']['path']);assert sha(retro_path)==proposal['retrospective_review']['sha256']
    retro=json.loads(retro_path.read_text());assert retro['status']=='audited'and len(retro['rows'])==3
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    scheduler=json.loads((root/'scheduler_status.json').read_text())
    assert state['phase']=='pilot'and state['round']==0 and not state['attempts']and not status['pilot_candidates']
    assert state['jobs']==plan['jobs']and len(state['jobs'])==48
    assert proposal['recipe']not in state['all_recipes']and len(proposal['new_jobs'])==3
    for job in proposal['new_jobs']:
        assert job['module']=='utils.flow_iql_research.train_onestep_deploy_v1'
        assert job['args']==dict(env=job['args']['env'],recipe='fqlraw_x3',seed=0,steps=1000000)
        assert proposal['expected_identities'][job['id']]==configuration(job['args']['env'],'fqlraw_x3',0)
        assert not(root/'runs'/job['id']).exists()
    assert {j['args']['env']for j in proposal['new_jobs']}==set(plan['pilot_environments'])
    started=set(scheduler['running'])|set(json.loads((root/'finished.json').read_text()))
    started|={job['id']for job in state['jobs']if(root/'runs'/job['id']/'config.json').exists()}
    index=next(i for i,job in enumerate(state['jobs'])if job['id']not in started)
    old_jobs=list(plan['jobs']);plan['jobs']=old_jobs[:index]+proposal['new_jobs']+old_jobs[index:]
    old_ids={j['id']for j in old_jobs};assert [j for j in plan['jobs']if j['id']in old_ids]==old_jobs
    assert len({j['id']for j in plan['jobs']})==51
    plan['expected_identities'].update(proposal['expected_identities'])
    plan['initial_recipes'].append('fqlraw_x3')
    assert isinstance(plan['initial_recipe_specs'],dict)
    plan['initial_recipe_specs']['fqlraw_x3']=specification('fqlraw_x3')
    sources=dict(plan['all_source_sha256'])
    for path in (Path(__file__).resolve(),proposal_path,retro_path):sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    plan['all_source_sha256']=sources
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),proposal=str(proposal_path),proposal_sha256=sha(proposal_path),
        recipe='fqlraw_x3',global_alpha_multiplier=3,public_alpha_source='Bundled official FQL README default-task commands',
        all9_effective_actor_alpha=proposal['all9_effective_actor_alpha'],
        retrospective=proposal['retrospective_review'],retrospective_pilot_macro=retro['macro'],
        no_completed_three_task_pilot_at_registration=True,
        observed_late_evaluations={key:value for key,value in status['valid_evaluations'].items()if key.endswith(':50')},
        original48definitions_and_relative_logical_order_preserved=True,new_initial_job_count=51,
        inserted_before_first_not_started_logical_job=index,
        pending_policy='Existing missing-pilot/confirmation priority remains. Within the remaining logical order, place the new midpoint before previously unstarted jobs; preserve started queue positions.',
        scalar_scope='Only actor effective_alpha changes. MeanQ/raw-Q actor family, finite-mean lean, all critic/BC/architecture/budget settings and actual update counts remain unchanged.',
        official_scope='Public alpha mapping only; present meanQ onAnt differs from the official Ant minQ flag. Not an exact full FQL reproduction.',
        timing='Adaptive alpha addition after partial pilot observations and retrospective old guided-policy review; before any complete3task/full9/fresh outcome for the current campaign.',
        performance_and_speed_gates_unchanged=True,configuration_addition_only=True,
        validation_scope='Existing validated generic fqlraw float-multiplier runner; generated concrete factory identities checked against the same frozen implementation. No new training branch or redundant unit tests.')
    plan['public_raw_midpoint_extension']=extension
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=sources,public_raw_midpoint_extension=extension)
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),initial_jobs=len(plan['jobs']),algorithm_sources_unchanged=True)))


if __name__=='__main__':main()
