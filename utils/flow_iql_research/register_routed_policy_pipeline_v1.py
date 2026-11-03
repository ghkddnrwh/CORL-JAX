"""Register one deployment policy over unchanged native MDN training jobs."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.routed_policy_registry_v1 import POLICY,NATIVE,MODE


def build_plan(root):
    parent=root/'registration_v17.json';previous=root/'analysis/execution_contract_v19.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs'] and len(state['jobs'])==142
    assert state['phase']=='confirmation' and state['active_attempt']==3
    assert state['attempts'][3]['recipe']==NATIVE and state['attempts'][3]['stage']=='discovery'
    assert not any(j['args']['seed'] in [21,22,23,24,25,26] for j in state['jobs'])
    freeze=root/'analysis/deployment_interim_review_v10/PRIVATE_DEPLOYMENT_CANDIDATE_FREEZE_v1.json'
    frozen=json.loads(freeze.read_text());sources=dict(plan['all_source_sha256'])
    for path,digest in frozen['source_sha256'].items():
        assert sha(path)==digest,path
        sources[path]=digest
    folder=Path(__file__).resolve().parent
    names=['routed_policy_registry_v1.py','routed_policy_verifier_v1.py',
        'benchmark_routed_mixture_isolated_v1.py','watch_onestep_deploy_v11.py',
        'register_routed_policy_pipeline_v1.py','activate_routed_policy_pipeline_v1.py',
        'validate_routed_policy_pipeline_v1.py']
    for name in names:sources[str(folder/name)]=sha(folder/name)
    evidence={}
    for relative in [
        'analysis/native_mixture_locations_v1/COMPLETE_FOUR_TASK_COMMON_DECODER_ROOT_AUDIT_v1.json',
        'analysis/mixture_location_deployment_v1/TWELVE_ACTUAL_NATIVE_EXPORTS_ISOLATION_v1.json',
        'analysis/mixture_location_deployment_v1/PZ3_1000K_PORTABLE_CPU50_REPLAY_v1.json']:
        path=root/relative;data=json.loads(path.read_text())
        assert data.get('status','pass')=='pass'
        evidence[relative]=dict(path=str(path),sha256=sha(path));sources[str(path)]=sha(path)
    assert len(json.loads(Path(next(iter(evidence.values()))['path']).read_text())['rows'])==4
    sources[str(freeze)]=sha(freeze)
    evaluation_sources={str(folder/name):sha(folder/name) for name in [
        'evaluate_portable_mixture_locations_v1.py','export_mixture_location_actor_v1.py',
        'standalone_mixture_location_actor_v1.py','lazy_mixture_location_actor_v1.py',
        'train_mean_empirical_backup_v1.py']}
    extension=dict(time=time.time(),policy_key=POLICY,native_training_recipe=NATIVE,primary_mode=MODE,
        fixed_global_decoder='sampled_component_location',superseded_attempt=3,expected_attempt=4,
        fresh_training_seeds=[24,25,26],unspent_previous_reserved_seeds=[21,22,23],
        evaluation_source_sha256=evaluation_sources,evidence=evidence,
        same_native_training_identity_and_source=True,existing_nine_seed0_jobs_reused=True,
        initial_GPU_jobs_added=0,initial_fresh_training_jobs_added=0,
        default_MDN_remains_full_nine_reported_control=True,
        default_attempt_supersession_is_not_a_failed_performance_claim=True,
        location_discovery_re_evaluated_with_explicit_actor_artifacts=True,
        repeated_seed0_rollouts_are_not_independent_training_replicates=True,
        full_nine_seed0_before_any_fresh_allocation=True,all_final_performance_and_speed_gates_unchanged=True,
        no_task_or_checkpoint_specific_policy_choice=True,
        alias_is_deployment_candidate_not_learner_recipe=True,
        native_futility_rules_retained=True,routed_candidate_uses_complete_nine_task_gate=True,
        reason='One common routed component-location policy improves all four completed same-model discovery tasks while retaining a single actor-only inference path. Confirm it independently over all nine tasks and three fresh native MDN training seeds.')
    plan['routed_deployment_extension']=extension;plan['all_source_sha256']=sources
    for path,digest in sources.items():assert sha(path)==digest,path
    original=json.loads(parent.read_text())
    for key in ('jobs','expected_identities','thresholds','implementation_training_source_sha256','GPU_slots_per_device','memory_fraction'):
        assert plan[key]==original[key],key
    return plan,contract


def main():
    root=Path(sys.argv[1]).resolve();target=root/'registration_v18.json'
    execution=root/'analysis/execution_contract_v20.json';assert not target.exists() and not execution.exists()
    plan,contract=build_plan(root)
    vp=root/'analysis/routed_policy_pipeline_v1/validation_v1/result.json';validation=json.loads(vp.read_text())
    assert validation['status']=='pass' and validation['no_performance_claim']
    for path,digest in validation['source_sha256'].items():assert sha(path)==digest,path
    plan['routed_deployment_extension']['implementation_validation']=dict(path=str(vp),sha256=sha(vp))
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    previous=root/'analysis/execution_contract_v19.json'
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=plan['all_source_sha256'],
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        routed_deployment_extension=plan['routed_deployment_extension'])
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),
        execution=str(execution),execution_sha256=sha(execution),GPU_jobs_unchanged=142,
        source_and_evidence_pins=len(plan['all_source_sha256']))))


if __name__=='__main__':main()
