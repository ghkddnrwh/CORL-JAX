"""Register CPU evaluation drainage before final isolated speed measurement."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def build_plan(root):
    parent=root/'registration_v18.json';previous=root/'analysis/execution_contract_v20.json'
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs'] and len(state['jobs'])==142
    assert state['active_attempt']==4 and state['attempts'][4]['recipe']=='mdn9_bc__component_location'
    assert state['attempts'][4]['stage']=='discovery'
    assert not any(j['args']['seed']in[24,25,26]for j in state['jobs'])
    folder=Path(__file__).resolve().parent;sources=dict(plan['all_source_sha256'])
    for name in ['watch_onestep_deploy_v12.py','register_benchmark_CPU_isolation_v1.py',
        'activate_benchmark_CPU_isolation_v1.py','validate_benchmark_CPU_isolation_v1.py','inspect_onestep_runtime_v7.py',
        'validate_portable_mixture_locations_GPU_v1.py']:
        sources[str(folder/name)]=sha(folder/name)
    for name in ['FOUR_NATIVE_GPU_TF32_DISABLED_CORRECTNESS_v1.json','GPU_TF32_DISABLED_INVOCATION_v1.json']:
        path=root/'analysis/mixture_location_deployment_v1'/name;sources[str(path)]=sha(path)
    plan['all_source_sha256']=sources
    plan['benchmark_CPU_isolation_amendment']=dict(time=time.time(),
        require_owned_CPU_evaluation_pool_drained=True,require_empty_GPU_process_query=True,
        require_no_GPU_learners=True,immutable_admission_receipt_before_workers=True,
        training_jobs_and_algorithms_unchanged=True,all_performance_and_speed_gates_unchanged=True,
        registered_primary_speed_protocol='Existing batch1 resident-weight paired CPU/GPU benchmark, default runtime for both policies.',
        supplemental_precise_GPU_runtime='NVIDIA_TF32_OVERRIDE=0 is independently validated for original actor single/batch32. Any supplemental timing must use the same flag for both teacher and actor; never silently replace the default-runtime speed result.')
    for path,digest in sources.items():assert sha(path)==digest,path
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    return plan,contract


def main():
    root=Path(sys.argv[1]).resolve();target=root/'registration_v19.json';execution=root/'analysis/execution_contract_v21.json'
    assert not target.exists()and not execution.exists();plan,contract=build_plan(root)
    vp=root/'analysis/benchmark_CPU_isolation_v1/validation_v1/result.json';validation=json.loads(vp.read_text())
    assert validation['status']=='pass'and validation['no_performance_claim']
    for path,digest in validation['source_sha256'].items():assert sha(path)==digest
    plan['benchmark_CPU_isolation_amendment']['validation']=dict(path=str(vp),sha256=sha(vp))
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    previous=root/'analysis/execution_contract_v20.json'
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=plan['all_source_sha256'],
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        benchmark_CPU_isolation_amendment=plan['benchmark_CPU_isolation_amendment'])
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration_sha256=sha(target),
        execution_contract_sha256=sha(execution),GPU_jobs=142,pins=len(plan['all_source_sha256']))))


if __name__=='__main__':main()
