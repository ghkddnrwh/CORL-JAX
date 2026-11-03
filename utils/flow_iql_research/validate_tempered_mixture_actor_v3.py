"""Separate mathematical equivalence from FP32 head-reduction roundoff.

The frozen center itself differs from the full-output FP32 reference on the
ill-conditioned Ant fixture. Preserve both earlier failures, prove the head
mapping with1e-10 FP64 agreement, report all FP32 errors, and require exact
zero-temperature compatibility. Production remains FP32 throughout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import validate_tempered_mixture_actor_v1 as reference
from utils.flow_iql_research import lazy_tempered_mixture_actor_v1 as tempered
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as center
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    forbidden=('flax','optax','ogbench','utils.flow_iql_research.mixture_onestep_deploy_core_v1')
    assert all(x not in sys.modules for x in forbidden)and all(d.platform=='cpu'for d in jax.devices())
    rows=[]
    for path in a.actors:
        for temperature in tempered.TEMPERATURES:
            loaded=tempered.load_actor(path,temperature);m=loaded.metadata
            batch=np.random.RandomState(510).normal(0,.2,(16,m['observation_dim'])).astype(np.float32)
            key=jax.random.PRNGKey(511)
            full32=jax.jit(lambda params,obs,rng:reference.reference(params,obs,rng,m,temperature),backend='cpu')
            errors=[]
            for shape,obs in [('batch16',batch),('single',batch[0])]:
                actual=np.asarray(loaded(obs,key));target=np.asarray(full32(loaded.params,jnp.asarray(obs),key))
                fp32_error=float(np.max(np.abs(actual-target)))
                assert np.isfinite(actual).all()and np.abs(actual).max()<=1
                # This is a coarse engineering sanity limit, explicitly NOT
                # a claim that the previous1e-5 FP32 comparison passed.
                assert fp32_error<1e-3,fp32_error
                if temperature==0.:
                    old=center.make_policy(m,'sampled_component_location')
                    np.testing.assert_array_equal(actual,np.asarray(old(loaded.params,jnp.asarray(obs),key)))
                with jax.experimental.enable_x64():
                    params64=jax.tree_util.tree_map(lambda x:jnp.asarray(x,dtype=jnp.float64),loaded.params)
                    obs64=jnp.asarray(obs,dtype=jnp.float64)
                    selected64=tempered.make_policy(m,temperature)
                    full64=jax.jit(lambda params,observations,rng:reference.reference(params,observations,rng,m,temperature),backend='cpu')
                    left=np.asarray(selected64(params64,obs64,key));right=np.asarray(full64(params64,obs64,key))
                    fp64_error=float(np.max(np.abs(left-right)));assert fp64_error<1e-10,fp64_error
                errors.append(dict(shape=shape,FP32_full_output_reference_max_error=fp32_error,
                    previous_1e_minus5_FP32_comparison_passed=fp32_error<1e-5,
                    FP64_mathematical_reference_max_error=fp64_error))
            hlo=jax.xla_computation(loaded.policy)(loaded.params,jnp.asarray(batch[0]),key).as_hlo_text()
            dots=sum(' dot('in line for line in hlo.splitlines());assert dots==6
            rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                temperature=temperature,errors=errors,Dense_dot_operations=dots,computed_action_heads=1,
                realized_actions=1,candidate_action_bank=False,critic_loaded=False,Flow_loaded=False))
            print(json.dumps(dict(actor=path.parent.name,temperature=temperature,
                FP32_max=max(e['FP32_full_output_reference_max_error']for e in errors),
                FP64_max=max(e['FP64_mathematical_reference_max_error']for e in errors))),flush=True)
    assert all(x not in sys.modules for x in forbidden)
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        verification='FP64 head-mapping equivalence1e-10 + original-zero-path FP32 bitwise compatibility + explicit FP32 rounding diagnostics.',
        earlier_FP32_validator_failures_preserved=True,production_remains_FP32=True,
        zero_temperature_exact_to_frozen_center=True,forbidden_dependencies_absent=list(forbidden),
        all_performance_and_latency_gates_unchanged=True,no_performance_or_latency_claim=True,
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in [__file__,reference.__file__,tempered.__file__,center.__file__]}),f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_models=len(a.actors),temperature_graphs=len(rows))))


if __name__=='__main__':main()
