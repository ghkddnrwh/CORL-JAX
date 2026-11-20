"""Fresh-process inference proof without learner, OGBench or optimizer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from utils.flow_iql_research import standalone_latent_selector_actor_v1 as portable
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    forbidden=('ogbench','optax','utils.flow_iql_research.latent_selector_onestep_core_v1',
               'utils.flow_iql_research.onestep_deploy_core_v1','utils.flow_iql_research.vendor.flax_utils')
    assert all(x not in sys.modules for x in forbidden)
    assert all(d.platform=='cpu'for d in jax.devices())
    loaded=portable.load_actor(a.actor);metadata=loaded.metadata
    with a.actor.open('rb')as f:artifact=pickle.load(f)
    assert set(artifact['actor_params'])=={'router','decoder'}
    assert metadata['actor_forward_passes']==1 and metadata['MLP_blocks']==2
    obs=np.linspace(-.2,.2,metadata['observation_dim'],dtype=np.float32)
    fixture=np.stack([obs,np.zeros_like(obs),-obs,obs*.1])
    key=jax.random.PRNGKey(419);actual=np.asarray(loaded(fixture,key))
    noise=jax.random.normal(jax.random.split(key)[0],(4,metadata['noise_dim']))
    expected=np.asarray(portable.definition(metadata).apply({'params':loaded.params},jnp.asarray(fixture),noise))
    difference=float(np.max(np.abs(actual-expected)));assert difference<1e-6
    assert np.isfinite(actual).all()and np.abs(actual).max()<=1
    np.testing.assert_array_equal(actual,np.asarray(loaded(fixture,key)))
    hlo=jax.xla_computation(loaded.policy)(loaded.params,jnp.asarray(obs),key).as_hlo_text()
    count=sum(' dot('in line for line in hlo.splitlines());assert count==10,count
    assert all(x not in sys.modules for x in forbidden)
    result=dict(time=time.time(),status='pass',actor_sha256=hashlib.sha256(a.actor.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),metadata=metadata,
        forbidden_dependencies_absent=True,model_trees_loaded=['router','decoder'],
        MLP_blocks=2,Dense_dot_operations=count,one_sampled_latent=True,one_action=True,
        requires_critic=False,requires_BC_flow=False,numerical_flow_integration_steps=0,
        pure_and_fitting_graph_max_difference=difference,deterministic_given_key=True,
        no_performance_or_latency_claim=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    a.output.with_suffix('.hlo.txt').write_text(hlo)
    print(json.dumps(dict(status='pass',selector=metadata['selector_family'],Dense_dots=count)))


if __name__=='__main__':main()
