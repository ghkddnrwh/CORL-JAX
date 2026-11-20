"""Optional full-FP32 dot precision for vectorized CPU/GPU deployment.

The native weights and routing rule are unchanged. Explicit per-dot precision
avoids relying on a global context around an already-cached JIT function.
The registered primary benchmark remains batch1 of the original client.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as original


def decode(params,observations,key,metadata):
    x=observations;precision=jax.lax.Precision.HIGHEST
    for i in range(len(metadata['hidden_dims'])):
        layer=params[f'Dense_{i}']
        x=jax.nn.gelu(jnp.dot(x,layer['kernel'],precision=precision)+layer['bias'])
    head=params[f"Dense_{len(metadata['hidden_dims'])}"];k=metadata['components'];d=metadata['action_dim']
    logits=jnp.dot(x,head['kernel'][:,:k],precision=precision)+head['bias'][:k]
    noise=jax.random.normal(jax.random.split(key)[0],(*observations.shape[:-1],metadata['noise_dim']))
    cdf=jnp.cumsum(jax.nn.softmax(logits,-1),-1)
    component=jnp.minimum(jnp.sum(ndtr(noise[...,0,None])>cdf,-1),k-1)
    columns=k+component[...,None]*d+jnp.arange(d)
    kernel=jnp.moveaxis(jnp.take(head['kernel'],columns,axis=1),0,-2)
    bias=jnp.take(head['bias'],columns)
    return jnp.clip(jnp.einsum('...h,...hd->...d',x,kernel,precision=precision)+bias,-1,1)


def make_policy(metadata,backend='cpu'):
    assert metadata['architecture']=='routed_component_location_MLP'
    assert metadata['deployment_decoder']=='sampled_component_location'
    assert metadata['computed_action_heads']==1 and not metadata['requires_critic']
    assert not metadata['requires_BC_flow'] and metadata['generated_candidate_actions']==metadata['flow_integration_steps']==0
    return jax.jit(lambda params,obs,key:decode(params,obs,key,metadata),backend=backend)


def load_actor(path,backend='cpu'):
    loaded=original.load_actor(path,backend)
    loaded.policy=make_policy(loaded.metadata,backend)
    loaded.metadata=dict(loaded.metadata,runtime_dot_precision='full_float32_HIGHEST',
        runtime_client='standalone_mixture_location_fp32_v1',native_parameter_or_artifact_changes=False)
    return loaded
