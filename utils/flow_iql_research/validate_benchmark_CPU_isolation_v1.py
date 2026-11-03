"""Exercise benchmark admission and resume a real weightless CPU controller."""
import ast
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.flow_iql_research import watch_onestep_deploy_v12 as controller
from utils.flow_iql_research.register_benchmark_CPU_isolation_v1 import build_plan
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/benchmark_CPU_isolation_v1/validation_v1'
    area.mkdir(parents=True,exist_ok=True);output=area/'result.json';assert not output.exists()
    plan,contract=build_plan(root);state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    ready=controller.benchmark_admission_ready
    assert ready({},'',{},{},{})
    assert not ready({},'',{'live_CPU_evaluator':object()},{},{})
    assert not ready({},'live_GPU_process',{},{},{})
    assert not ready({'live_learner':object()},'',{},{},{})
    assert not ready({},'',{}, {'live_benchmark':object()}, {})
    assert not ready({},'',{},{},{'existing_results':object()})
    assert not ready({},'live_GPU_process',{'live_CPU_evaluator':object()},{},{})
    tree=ast.parse(Path(controller.__file__).read_text())
    admission=[node for node in ast.walk(tree)if isinstance(node,ast.If)and isinstance(node.test,ast.Call)
        and isinstance(node.test.func,ast.Name)and node.test.func.id=='benchmark_admission_ready']
    assert len(admission)==1 and isinstance(admission[0].test.args[2],ast.Name)and admission[0].test.args[2].id=='workers'
    assert 'isolation_admission_v1.json'in ast.unparse(admission[0])
    # The GPU flag proof is actual execution of the original frozen client.
    gpu=root/'analysis/mixture_location_deployment_v1/FOUR_NATIVE_GPU_TF32_DISABLED_CORRECTNESS_v1.json'
    proof=json.loads(gpu.read_text());assert proof['status']=='pass'and len(proof['rows'])==4
    for row in proof['rows']:
        assert row['batch32_CPU_GPU_max_abs_difference']<1e-4 and row['single_CPU_GPU_max_abs_difference']<1e-4
        assert row['computed_action_heads']==1 and not row['critic_loaded']and not row['Flow_loaded']
    fixture=area/'fixture';assert not fixture.exists();(fixture/'runs').mkdir(parents=True)
    p=copy.deepcopy(plan);p['root']=str(fixture);save(fixture/'registration.json',p)
    execution=copy.deepcopy(contract);execution.update(registration=str(fixture/'registration.json'),
        registration_sha256=sha(fixture/'registration.json'),source_sha256=p['all_source_sha256'])
    save(fixture/'contract.json',execution)
    initial=copy.deepcopy(state);initial.update(evaluation_workers={},evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={})
    save(fixture/'analysis/onestep_pipeline_v1/research_state.json',initial)
    save(fixture/'queue.json',initial['jobs']);save(fixture/'scheduler_status.json',dict(pid=os.getpid(),running={}))
    save(fixture/'finished.json',{})
    argv=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v12',str(fixture/'contract.json')]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (area/'fixture_controller.log').open('xb')as log:
        child=subprocess.Popen(argv,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=controller.process_identity(child.pid);save(area/'fixture_controller_handle.json',dict(pid=child.pid,birth=birth,command=argv))
    try:
        deadline=time.monotonic()+45;status_path=fixture/'analysis/onestep_pipeline_v1/status.json'
        while not status_path.exists():
            assert child.poll()is None,(area/'fixture_controller.log').read_text()[-2500:]
            assert time.monotonic()<deadline,'Poll existing fixture handle if observation expires.'
            time.sleep(.2)
        status=json.loads(status_path.read_text());after=json.loads((fixture/'analysis/onestep_pipeline_v1/research_state.json').read_text())
        assert after['jobs']==initial['jobs']and after['identities']==initial['identities']
        assert after['active_attempt']==4 and after['attempts']==initial['attempts']
        assert after['phase']=='confirmation'and not after['evaluation_workers']and not after['benchmark_workers']
        assert status['planned_GPU_jobs']==142 and len(status['rows'])==151
        assert not list((fixture/'analysis/onestep_pipeline_v1').glob('benchmark_attempt*'))
    finally:
        if child.poll()is None:
            assert controller.process_identity(child.pid)==birth
            assert str(fixture/'contract.json')in Path(f'/proc/{child.pid}/cmdline').read_text().split('\0')
            os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    result=dict(time=time.time(),status='pass',source_sha256=plan['all_source_sha256'],no_performance_claim=True,
        admission_boundary_cases=7,actual_CPU_worker_gate_used=True,immutable_admission_receipt_source_verified=True,
        actual_GPU_correctness_evidence_sha256=sha(gpu),actual_GPU_actor_exports=4,
        weightless_controller_resume=dict(native_jobs=142,policy_rows=9,actual_new_GPU_jobs=0,
            actual_CPU_evaluators=0,current_attempt4_and_all_identities_preserved=True))
    save(output,result);print(json.dumps({k:result[k]for k in ['status','admission_boundary_cases','actual_GPU_actor_exports','weightless_controller_resume']}))


if __name__=='__main__':main()
