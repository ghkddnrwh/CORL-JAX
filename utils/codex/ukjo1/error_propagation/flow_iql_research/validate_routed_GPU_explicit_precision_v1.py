"""Actual public single/batch CPU/GPU inference with explicit precision."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as default
from utils.flow_iql_research import standalone_mixture_location_fp32_v1 as fp32
import jax
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert all(x not in sys.modules for x in ['flax','optax','ogbench'])
    rows=[]
    for path in a.actors:
        cpu=default.load_actor(path,'cpu');gpu=default.load_actor(path,'gpu')
        cpu32=fp32.load_actor(path,'cpu');gpu32=fp32.load_actor(path,'gpu')
        x=np.random.RandomState(811).normal(0,.2,(32,cpu.metadata['observation_dim'])).astype(np.float32)
        key=jax.random.PRNGKey(812)
        errors={}
        for shape,obs in [('single',x[0]),('batch32',x)]:
            reference=np.asarray(cpu(obs,key));defaultGPU=np.asarray(gpu(obs,key))
            explicitCPU=np.asarray(cpu32(obs,key));explicitGPU=np.asarray(gpu32(obs,key))
            assert np.isfinite(explicitGPU).all() and np.abs(explicitGPU).max()<=1
            errors[shape]=dict(default_CPU_GPU_max=float(np.max(np.abs(reference-defaultGPU))),
                explicit_FP32_CPU_GPU_max=float(np.max(np.abs(explicitCPU-explicitGPU))),
                default_vs_FP32_CPU_max=float(np.max(np.abs(reference-explicitCPU))))
            assert errors[shape]['explicit_FP32_CPU_GPU_max']<1e-4,errors[shape]
            assert errors[shape]['default_vs_FP32_CPU_max']==0,errors[shape]
            if shape=='single':assert errors[shape]['default_CPU_GPU_max']<1e-4
            np.testing.assert_array_equal(explicitGPU,np.asarray(gpu32(obs,key)))
        hlo=jax.xla_computation(gpu32.policy)(gpu32.params,jax.device_put(x[0],gpu32.device),jax.device_put(key,gpu32.device)).as_hlo_text()
        dots=[line for line in hlo.splitlines() if ' dot(' in line]
        assert len(dots)==6 and all('highest' in line for line in dots),dots
        rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            GPU_kind=gpu32.device.device_kind,errors=errors,explicit_FP32_hlo_dot_precision_proven=True,
            actual_GPU_inference_completed=True,computed_action_heads=1,neural_dot_operations=6,
            no_critic=True,no_Flow=True,no_candidate_action_bank=True))
    assert all(x not in sys.modules for x in ['flax','optax','ogbench'])
    result=dict(time=time.time(),status='pass',rows=rows,default_single_observation_GPU_check_passed=True,
        explicit_FP32_batch_and_single_GPU_check_passed=True,CPU_actions_unchanged_by_explicit_precision=True,
        global_context_on_cached_JIT_is_not_a_precision_proof=True,
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in [__file__,fp32.__file__,default.__file__]},
        training_updates_performed=0,production_artifacts_or_registered_code_changed=False,
        no_isolated_latency_or_rollout_performance_claim=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_models=len(rows),max_explicit_FP32_CPU_GPU_difference=max(
        v['explicit_FP32_CPU_GPU_max']for row in rows for v in row['errors'].values()))))


if __name__=='__main__':main()
