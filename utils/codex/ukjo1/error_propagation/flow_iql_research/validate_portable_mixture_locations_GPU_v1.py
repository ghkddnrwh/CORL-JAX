"""Bounded actual GPU inference correctness; no isolated latency claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'
    assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'
    forbidden=('flax','optax','ogbench','utils.flow_iql_research.vendor.flax_utils',
               'utils.flow_iql_research.mixture_onestep_deploy_core_v1')
    assert all(name not in sys.modules for name in forbidden)
    rows=[]
    for path in a.actors:
        cpu=portable.load_actor(path,'cpu');gpu=portable.load_actor(path,'gpu')
        assert gpu.device.platform=='gpu' and cpu.device.platform=='cpu'
        x=np.random.RandomState(811).normal(0,.2,(32,cpu.metadata['observation_dim'])).astype(np.float32)
        key=jax.random.PRNGKey(812)
        expected=np.asarray(cpu(x,key));actual=np.asarray(gpu(x,key))
        assert np.isfinite(actual).all() and np.abs(actual).max()<=1
        error=float(np.max(np.abs(expected-actual)));assert error<1e-4,error
        np.testing.assert_array_equal(actual,np.asarray(gpu(x,key)))
        single=np.asarray(gpu(x[0],key));reference=np.asarray(cpu(x[0],key))
        single_error=float(np.max(np.abs(single-reference)));assert single_error<1e-4,single_error
        hlo=jax.xla_computation(gpu.policy)(gpu.params,jax.device_put(x[0],gpu.device),jax.device_put(key,gpu.device)).as_hlo_text()
        dots=sum(' dot(' in line for line in hlo.splitlines());assert dots==6
        rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            GPU_kind=gpu.device.device_kind,actual_GPU_inference_completed=True,
            batch32_CPU_GPU_max_abs_difference=error,single_CPU_GPU_max_abs_difference=single_error,
            deterministic_given_key=True,Dense_dot_operations=dots,computed_action_heads=1,
            critic_loaded=False,Flow_loaded=False,candidate_action_bank=False))
    assert all(name not in sys.modules for name in forbidden)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forbidden_imports_absent=list(forbidden),training_updates_performed=0,
        no_isolated_latency_or_rollout_performance_claim=True,
        scope='Bounded single/batch GPU actor-only calls while16reallearners continue. Correctness/device/dependency proof, not an isolated benchmark.'),f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_GPU_actor_exports=len(rows),
        max_CPU_GPU_difference=max(max(x['single_CPU_GPU_max_abs_difference'],x['batch32_CPU_GPU_max_abs_difference'])for x in rows))))


if __name__=='__main__':main()
