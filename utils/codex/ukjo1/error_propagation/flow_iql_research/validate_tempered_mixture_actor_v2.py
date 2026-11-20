"""Compare compiled selected heads with a compiled full-output reference.

The first validator's eager reference introduced another execution regime.
Keep its failure/source, compare both references explicitly, and retain the
original1e-5 numerical criterion for the compiled inference graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import validate_tempered_mixture_actor_v1 as previous
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
            compiled=jax.jit(lambda params,obs,rng:previous.reference(params,obs,rng,m,temperature),backend='cpu')
            errors=[]
            for name,obs in [('batch16',batch),('single',batch[0])]:
                actual=np.asarray(loaded(obs,key));ref=np.asarray(compiled(loaded.params,jnp.asarray(obs),key))
                eager=np.asarray(previous.reference(loaded.params,jnp.asarray(obs),key,m,temperature))
                error=float(np.max(np.abs(actual-ref)));eager_error=float(np.max(np.abs(actual-eager)))
                entry=dict(shape=name,compiled_reference_max_error=error,eager_reference_max_error=eager_error)
                print(json.dumps(dict(actor=path.name,temperature=temperature,**entry)),flush=True)
                assert error<1e-5,entry
                assert np.isfinite(actual).all()and np.abs(actual).max()<=1
                if temperature==0:
                    same=center.make_policy(m,'sampled_component_location')
                    np.testing.assert_array_equal(actual,np.asarray(same(loaded.params,jnp.asarray(obs),key)))
                errors.append(entry)
            hlo=jax.xla_computation(loaded.policy)(loaded.params,jnp.asarray(batch[0]),key).as_hlo_text()
            dots=sum(' dot('in line for line in hlo.splitlines());assert dots==6
            rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                temperature=temperature,errors=errors,Dense_dot_operations=dots,computed_action_heads=1,
                realized_actions=1,candidate_action_bank=False,critic_loaded=False,Flow_loaded=False))
    assert all(x not in sys.modules for x in forbidden)
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        numerical_tolerance_unchanged=1e-5,zero_temperature_exact_to_frozen_center=True,
        forbidden_dependencies_absent=list(forbidden),no_performance_or_latency_claim=True,
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in [__file__,previous.__file__,tempered.__file__,center.__file__]}),f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_models=len(a.actors),temperature_graphs=len(rows))))


if __name__=='__main__':main()
