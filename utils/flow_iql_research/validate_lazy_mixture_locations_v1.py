"""Trained-model lazy-head equivalence and inference dependency/graph proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as lazy
from utils.flow_iql_research import standalone_mixture_actor_v1 as original
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    forbidden=['ogbench','optax','utils.flow_iql_research.vendor.flax_utils','utils.flow_iql_research.mixture_onestep_deploy_core_v1']
    assert all(x not in sys.modules for x in forbidden)and all(d.platform=='cpu'for d in jax.devices())
    rows=[]
    for actor in a.actors:
        loaded=original.load_actor(actor);m=loaded.metadata
        x=jax.random.normal(jax.random.PRNGKey(490),(16,m['observation_dim']))*.1;key=jax.random.PRNGKey(491)
        epsilon=jax.random.normal(jax.random.split(key)[0],(16,m['noise_dim']))
        p=original.definition(m).apply({'params':loaded.params},x,epsilon,return_parameters=True)
        for decoder in lazy.DECODERS:
            component=jnp.argmax(p['logits'],-1)if decoder=='highest_weight_location'else jnp.minimum(jnp.sum(
                ndtr(epsilon[...,0,None])>jnp.cumsum(jax.nn.softmax(p['logits'],-1),-1),-1),m['components']-1)
            indices=jnp.broadcast_to(component[...,None,None],(16,1,m['action_dim']))
            expected=jnp.clip(jnp.take_along_axis(p['means'],indices,-2)[...,0,:],-1,1)
            policy=lazy.make_policy(m,decoder);actual=policy(loaded.params,x,key)
            delta=float(jnp.max(jnp.abs(expected-actual)));assert delta<1e-5,delta
            assert np.isfinite(actual).all()and np.abs(actual).max()<=1
            hlo=jax.xla_computation(policy)(loaded.params,x[0],key).as_hlo_text()
            dots=sum(' dot('in line for line in hlo.splitlines());assert dots==len(m['hidden_dims'])+2==6
            rows.append(dict(actor=str(actor),actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),decoder=decoder,
                reference_vs_lazy_max_difference=delta,Dense_dots=dots,actor_forwards=1,computed_action_heads=1,
                requires_critic=False,requires_Flow=False,candidate_action_bank=False,optimizer_updates=0))
    assert all(x not in sys.modules for x in forbidden)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(time=time.time(),status='pass',rows=rows,forbidden_imports_absent=forbidden,
        no_performance_or_latency_claim=True,source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in [__file__,lazy.__file__]}),indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',trained_models=len(a.actors),decoder_graphs=len(rows))))


if __name__=='__main__':main()
