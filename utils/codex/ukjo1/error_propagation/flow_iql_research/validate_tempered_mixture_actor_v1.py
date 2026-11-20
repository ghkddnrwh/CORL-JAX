"""Actual-model parameter-head equivalence; pure NumPy/JAX inference proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import lazy_tempered_mixture_actor_v1 as tempered
from utils.flow_iql_research import lazy_mixture_location_actor_v1 as center
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np


def reference(params,observations,key,m,temperature):
    x=observations
    for i in range(len(m['hidden_dims'])):
        p=params[f'Dense_{i}'];x=jax.nn.gelu(jnp.dot(x,p['kernel'])+p['bias'])
    head=params[f"Dense_{len(m['hidden_dims'])}"];y=jnp.dot(x,head['kernel'])+head['bias']
    k,d=m['components'],m['action_dim'];logits=y[...,:k]
    mu=y[...,k:k+k*d].reshape((*y.shape[:-1],k,d))
    sigma=jax.nn.softplus(y[...,k+k*d:].reshape((*y.shape[:-1],k,d)))+m['std_floor']
    noise=jax.random.normal(jax.random.split(key)[0],(*observations.shape[:-1],m['noise_dim']))
    c=jnp.minimum(jnp.sum(ndtr(noise[...,0,None])>jnp.cumsum(jax.nn.softmax(logits,-1),-1),-1),k-1)
    idx=jnp.broadcast_to(c[...,None,None],(*c.shape,1,d))
    chosen_mu=jnp.take_along_axis(mu,idx,-2)[...,0,:]
    chosen_sigma=jnp.take_along_axis(sigma,idx,-2)[...,0,:]
    return jnp.clip(chosen_mu+temperature*chosen_sigma*noise[...,1:],-1,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    forbidden=('flax','optax','ogbench','utils.flow_iql_research.mixture_onestep_deploy_core_v1')
    assert all(x not in sys.modules for x in forbidden)and all(d.platform=='cpu'for d in jax.devices())
    rows=[]
    for path in a.actors:
        for t in tempered.TEMPERATURES:
            loaded=tempered.load_actor(path,t);m=loaded.metadata
            batch=np.random.RandomState(510).normal(0,.2,(16,m['observation_dim'])).astype(np.float32)
            key=jax.random.PRNGKey(511)
            for name,obs in [('batch16',batch),('single',batch[0])]:
                actual=np.asarray(loaded(obs,key));expected=np.asarray(reference(loaded.params,jnp.asarray(obs),key,m,t))
                error=float(np.max(np.abs(actual-expected)));assert error<1e-5,error
                assert np.isfinite(actual).all()and np.abs(actual).max()<=1
                if t==0:
                    original=center.make_policy(m,'sampled_component_location')
                    np.testing.assert_array_equal(actual,np.asarray(original(loaded.params,jnp.asarray(obs),key)))
            hlo=jax.xla_computation(loaded.policy)(loaded.params,jnp.asarray(batch[0]),key).as_hlo_text()
            dots=sum(' dot('in line for line in hlo.splitlines());assert dots==6
            rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                temperature=t,single_and_batch_reference_error_below_1e5=True,Dense_dot_operations=dots,
                computed_action_heads=1,realized_actions=1,candidate_action_bank=False,requires_critic=False,
                requires_Flow=False,no_optimizer_updates=True))
    assert all(x not in sys.modules for x in forbidden)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        forbidden_dependencies_absent=list(forbidden),no_performance_or_latency_claim=True,
        source_sha256={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()for path in [__file__,tempered.__file__,center.__file__]}),f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',actual_models=len(a.actors),temperature_graphs=len(rows))))


if __name__=='__main__':main()
