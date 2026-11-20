"""Fresh-process equivalence of explicit artifacts and evaluated decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy
import jax
import jax.numpy as jnp
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    forbidden=('flax','optax','ogbench','utils.flow_iql_research.vendor.flax_utils',
               'utils.flow_iql_research.onestep_deploy_core_v1')
    assert all(x not in sys.modules for x in forbidden)
    assert all(d.platform=='cpu' for d in jax.devices())
    rows=[]
    for path in a.actors:
        with path.open('rb') as stream:artifact=pickle.load(stream)
        native_path=Path(artifact['native_actor_path']);assert sha(native_path)==artifact['native_actor_sha256']
        with native_path.open('rb') as stream:native=pickle.load(stream)
        assert artifact['training_identity']==native['identity']
        left=jax.tree_util.tree_leaves(artifact['actor_params']);right=jax.tree_util.tree_leaves(native['actor_params'])
        assert len(left)==len(right) and all(np.array_equal(x,y) for x,y in zip(left,right))
        for source,digest in artifact['deployment_source_sha256'].items():assert sha(source)==digest
        actor=portable.load_actor(path);reference=lazy.make_policy(native['metadata'],'sampled_component_location')
        obs=jax.random.normal(jax.random.PRNGKey(721),(64,actor.metadata['observation_dim']))*.2
        key=jax.random.PRNGKey(722)
        actual=np.asarray(actor(obs,key));expected=np.asarray(reference(actor.params,obs,key))
        np.testing.assert_array_equal(actual,expected)
        single=np.asarray(actor(obs[0],key));single_ref=np.asarray(reference(actor.params,obs[0],key))
        np.testing.assert_array_equal(single,single_ref)
        assert np.isfinite(actual).all() and np.abs(actual).max()<=1
        hlo=jax.xla_computation(actor.policy)(actor.params,obs[0],key).as_hlo_text()
        dots=sum(' dot(' in line for line in hlo.splitlines());assert dots==6
        rows.append(dict(actor=str(path),actor_sha256=sha(path),native_actor_sha256=sha(native_path),
            step=artifact['training_step'],training_identity_preserved=True,actor_params_bitwise_exact=True,
            single_and_batch_outputs_bitwise_exact=True,Dense_dot_operations=dots,
            computed_action_heads=1,critic_loaded=False,Flow_loaded=False,candidate_action_bank=False))
    assert all(x not in sys.modules for x in forbidden)
    result=dict(time=time.time(),status='pass',rows=rows,forbidden_imports_absent=list(forbidden),
        source_sha256=sha(__file__),no_performance_or_isolated_latency_claim=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_trained_actor_exports=len(rows),neural_dot_operations=6)))


if __name__=='__main__':
    main()
