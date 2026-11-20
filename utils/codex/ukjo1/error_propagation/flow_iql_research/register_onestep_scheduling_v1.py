"""Register a scheduling-only amendment before any complete3task pilot."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research import onestep_deploy_core_v1 as core


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();assert not args.registration.exists()and not args.contract.exists()
    previous=root/'registration_v2.json';old_contract_path=root/'analysis/execution_contract_v4.json'
    plan=json.loads(previous.read_text());old_contract=json.loads(old_contract_path.read_text())
    assert old_contract['registration_sha256']==sha(previous)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    assert state['phase']=='pilot'and not state['attempts']and not status['pilot_candidates']
    assert len(state['jobs'])==48 and state['jobs']==plan['jobs']
    assert not(root/'PAUSE_NEW_LAUNCHES.json').exists()and not(root/'STOP').exists()
    unstarted=plan['pilot_environments'][2]
    assert not any((root/'runs'/job['id']/'config.json').exists()for job in state['jobs']if job['args']['env']==unstarted)
    directory=Path(__file__).resolve().parent
    new_names=('onestep_scheduling_policy_v1.py','watch_onestep_deploy_v5.py',
        'scheduler_onestep_queue_v1.py','scheduler_onestep_deploy_v3.py',
        'validate_onestep_adaptive_order_v1.py','validate_onestep_scheduler_adoption_v2.py',
        'validate_onestep_controller_startup_v1.py','register_onestep_scheduling_v1.py')
    sources=dict(plan['all_source_sha256'])
    for name in new_names:
        path=directory/name;assert str(path)not in sources;sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    evidence=[];out=root/'analysis/adaptive_scheduling_v1'
    for path in (out/'policy_validation_v3.json',out/'scheduler_validation_v2/result.json',out/'controller_startup_v2/result.json'):
        value=json.loads(path.read_text());assert value['status']=='pass'and value['no_performance_claim']
        for source,digest in value.get('source_sha256',{}).items():assert sha(source)==digest,source
        evidence.append(dict(path=str(path),sha256=sha(path),scope=value.get('scope'),checks=value['checks']))
    plan['all_source_sha256']=sources
    amendment=dict(time=time.time(),version=1,parent_registration=str(previous),parent_registration_sha256=sha(previous),
        parent_execution_contract=str(old_contract_path),parent_execution_contract_sha256=sha(old_contract_path),
        performance_gates_unchanged=True,algorithms_and_job_definitions_unchanged=True,
        all_nine_environments=list(core.ENVIRONMENTS),no_completed_pilot_recipe_at_registration=True,
        no_humanoid_pilot_training_started_at_registration=True,
        timing='Partial CubeD last3 and other100K/300K pilot observations exist; no full3task candidate or nine-task outcome exists.',
        queue_order='Preserve every job and started position. Prioritize active confirmation; otherwise fill the missing third pilot environment for recipes with two environments already started, using logical registration order for ties.',
        candidate_selection='An eligible complete3task pilot can advance before all48 jobs finish. Unqualified exploratory promotion still requires the completed round and keeps the original maximum2/diverse-family policy.',
        staged_confirmation='First finish nine seed0 tasks and require the same paired/historical macro gates. Only a pass registers all27 fresh runs on3reservedseeds. All original full9/fresh gates remain mandatory.',
        benchmarks='After all performance gates pass, PAUSE_NEW_LAUNCHES.json drains existing learners without terminating them. Measure the same isolated CPU/GPU latency gates. Resume pending work if speed fails.',
        terminal_recovery='Adopt only matching PID births; a missing previous owner with a completed marker is recorded terminal before scheduling, preventing duplicate relaunch during manager transition.',
        overrides=['R2 pending job order','wait-for-whole-pilot-round barrier for eligible recipes','simultaneous seed0/fresh allocation'],
        unchanged=['48 original jobs/configurations','all actor losses and learner sources','800K900K1M CPU50 protocol','macro thresholds and historical references','fresh3seeds mandatory','global recipe across9tasks','2x8 slots and memory.1','isolated final speed gates'],
        validation=evidence)
    plan['scheduling_amendment']=amendment
    plan['selection']=amendment['candidate_selection']
    plan['confirmation']=dict(previous=plan['confirmation'],scheduling=amendment['staged_confirmation'])
    plan['implementation_extension']['queue_policy']='Original48 definitions retained; pending scheduling follows the explicit V3 scheduling amendment. Six set recipes remain in the registered next adaptive round.'
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract=dict(old_contract,registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=sources,GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v3',
        scheduling_amendment=amendment)
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),thresholds_unchanged=plan['thresholds']==json.loads(previous.read_text())['thresholds'])))


if __name__=='__main__':main()
