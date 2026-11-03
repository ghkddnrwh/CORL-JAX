"""Register one exact discovery failure; never changes success gates or jobs."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();assert not args.registration.exists()and not args.contract.exists()
    parent=root/'registration_v7.json';previous=root/'analysis/execution_contract_v9.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text());assert contract['registration_sha256']==sha(parent)
    directory=root/'analysis/discovery_bound_attempt0_v1';proof_path=directory/'raw_success_upper_bound_proof_v1.json';proof=json.loads(proof_path.read_text())
    validation_path=directory/'validation_v2/result.json';validation=json.loads(validation_path.read_text())
    assert proof['status']=='audited'and proof['historical_gate_mathematically_impossible']
    assert validation['status']=='pass'and validation['no_performance_claim']and validation['proof_sha256']==sha(proof_path)
    assert proof['registration_sha256']==sha(parent)
    floor=plan['thresholds']['historical_lean_seed0_nine_task_reference']-plan['thresholds']['mean_loss_tolerance']
    assert proof['unchanged_historical_gate_minimum']==floor and proof['nine_task_mean_upper_bound']<floor-1e-12
    for path,digest in {**proof['raw_evidence_sha256'],**validation['source_sha256']}.items():assert sha(path)==digest,path
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['phase']=='confirmation'and state['active_attempt']==0 and len(state['attempts'])==1
    attempt=state['attempts'][0];assert attempt['recipe']==proof['recipe']and attempt['stage']=='discovery'
    assert proof['recipe']not in state['tested_candidates']
    assert not any(j['args']['seed']in attempt['fresh_seeds']for j in state['jobs'])
    assert state['jobs']==plan['jobs']and len(state['jobs'])==78
    decision=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),attempt=0,recipe=proof['recipe'],
        proof=dict(path=str(proof_path),sha256=sha(proof_path)),validation=dict(path=str(validation_path),sha256=sha(validation_path)),
        completed_environment_count=proof['completed_environment_count'],observed_successes=proof['observed_successes'],
        remaining_best_case_successes=proof['remaining_best_case_successes'],nine_task_denominator=proof['nine_task_denominator'],
        upper_bound=proof['nine_task_mean_upper_bound'],unchanged_historical_minimum=floor,
        state_transition=dict(attempt_status='discovery_historical_gate_impossible',phase='pilot',add_tested_recipe=proof['recipe']),
        no_success_gate_changes=True,no_job_definition_or_source_changes=True,no_per_environment_veto=True,
        remaining_two_learners_continue=True,no_fresh_run_allocated_for_failed_attempt=True,
        statistical_scope='Deterministic upper bound on the fixed empirical seed0last3CPU50metric. No population-performance assertion or completed9taskmean.',
        reason='Even perfect scores in both missing environments cannot meet the original seed0historical macro gate. Reject this attempt now and allow the next actually qualified pilot to advance; retain all running experiments.')
    sources=dict(plan['all_source_sha256'])
    for path in (Path(__file__).resolve(),Path(__file__).resolve().with_name('validate_discovery_bound_decision_v1.py'),proof_path,validation_path):sources[str(path)]=sha(path)
    for path,digest in sources.items():assert sha(path)==digest,path
    plan.update(all_source_sha256=sources,discovery_bound_failure_decision=decision)
    assert plan['jobs']==json.loads(parent.read_text())['jobs']and plan['thresholds']==json.loads(parent.read_text())['thresholds']
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),source_sha256=sources,discovery_bound_failure_decision=decision)
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),jobs_unchanged=78,upper_bound=decision['upper_bound'],historical_gate_minimum=floor)))


if __name__=='__main__':main()
