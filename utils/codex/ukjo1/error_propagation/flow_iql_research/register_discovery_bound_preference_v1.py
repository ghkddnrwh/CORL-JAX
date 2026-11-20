"""Register exact empirical futility and a transparent full9 mixture preference."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def build_plan(root):
    parent=root/'registration_v16.json';previous=root/'analysis/execution_contract_v18.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text());assert contract['registration_sha256']==sha(parent)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs']and len(state['jobs'])==142
    assert state['phase']=='confirmation'and state['active_attempt']==2 and state['attempts'][2]['recipe']=='set_bc'
    evidence=root/'analysis/mixture_onestep_pipeline_v1/NATIVE_THREE_HARD_TASKS_ROOT_AUDIT_v1.json'
    proof=json.loads(evidence.read_text());assert proof['status']=='pass'and len(proof['rows'])==3
    for row in proof['rows']:
        for point in row['points']:assert sha(point['path'])==point['sha256']
    preference=dict(time=time.time(),recipe='mdn9_bc',after_attempt=2,after_recipe='set_bc',
        evidence=dict(path=str(evidence),sha256=sha(evidence)),
        reason='Native joint mixture actor has complete hard-task last3 Cube.92/Puzzle.8133/CubeD.2533, unlike weak private frozen CPU screens. Give one exploratory full-nine evaluation next if currentSetattempt fails; original pilot may still be incomplete.',
        original_pilot_screen_pass_not_assumed=True,no_final_gate_changes=True,no_environment_wise_recipe_selection=True,
        no_immediate_state_transition_or_fresh_allocation=True)
    plan['full9_candidate_preference']=preference
    plan['bounded_discovery_policy']=dict(time=time.time(),policy='Missing fixedCPU50 points are each allowed50successes; rejectonlyif even this cannot satisfy unchangedhistoricalmacrofloor.',
        attempts_affected='Discovery stages only',all_remaining_learners_and_evaluations_continue=True,
        receipt_precedes_state_transition=True,all_final_full9_fresh_speed_gates_unchanged=True,
        no_per_task_veto=True,no_population_futility_claim=True)
    f=Path(__file__).resolve().parent;sources=dict(plan['all_source_sha256'])
    for name in ['discovery_bound_policy_v1.py','watch_onestep_deploy_v10.py','register_discovery_bound_preference_v1.py',
                 'activate_discovery_bound_preference_v1.py','validate_discovery_bound_preference_v1.py']:
        path=f/name;sources[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in sources.items())
    plan['all_source_sha256']=sources
    assert plan['jobs']==json.loads(parent.read_text())['jobs']and plan['thresholds']==json.loads(parent.read_text())['thresholds']
    return plan,contract


def main():
    root=Path(sys.argv[1]).resolve();target=root/'registration_v17.json';execution=root/'analysis/execution_contract_v19.json'
    assert not target.exists()and not execution.exists();plan,contract=build_plan(root)
    vp=root/'analysis/discovery_bound_preference_v1/validation_v1/result.json';v=json.loads(vp.read_text())
    assert v['status']=='pass'and v['no_performance_claim']
    for p,h in v['source_sha256'].items():assert sha(p)==h,p
    plan['bounded_discovery_policy']['validation']=dict(path=str(vp),sha256=sha(vp))
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    previous=root/'analysis/execution_contract_v18.json'
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=plan['all_source_sha256'],
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        full9_candidate_preference=plan['full9_candidate_preference'],bounded_discovery_policy=plan['bounded_discovery_policy'])
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),
        execution=str(execution),execution_sha256=sha(execution),jobs_unchanged=142)))


if __name__=='__main__':main()
