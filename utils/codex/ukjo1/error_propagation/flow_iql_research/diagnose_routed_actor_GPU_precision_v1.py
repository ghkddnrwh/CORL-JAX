"""Resolve backend/precision differences without changing any trained model."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np


def routing(params,x,key,m):
    for i in range(len(m['hidden_dims'])):
        layer=params[f'Dense_{i}'];x=jax.nn.gelu(jnp.dot(x,layer['kernel'])+layer['bias'])
    head=params[f"Dense_{len(m['hidden_dims'])}"];k=m['components']
    logits=jnp.dot(x,head['kernel'][:,:k])+head['bias'][:k]
    noise=jax.random.normal(jax.random.split(key)[0],(*x.shape[:-1],m['noise_dim']))
    uniform=ndtr(noise[...,0]);cdf=jnp.cumsum(jax.nn.softmax(logits,-1),-1)
    index=jnp.minimum(jnp.sum(uniform[...,None]>cdf,-1),k-1)
    return dict(logits=logits,index=index,uniform=uniform,min_CDF_margin=jnp.min(jnp.abs(cdf-uniform[...,None]),-1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--actors',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert all(x not in sys.modules for x in ['flax','optax','ogbench'])
    rows=[]
    for path in a.actors:
        cpu=portable.load_actor(path,'cpu');gpu=portable.load_actor(path,'gpu')
        x=np.random.RandomState(811).normal(0,.2,(32,cpu.metadata['observation_dim'])).astype(np.float32)
        key=jax.random.PRNGKey(812)
        for precision in [None,'float32']:
            context=nullcontext() if precision is None else jax.default_matmul_precision(precision)
            with context:
                for shape,obs in [('batch32',x),('single',x[0])]:
                    expected=np.asarray(cpu(obs,key));actual=np.asarray(gpu(obs,key))
                    assert np.isfinite(actual).all() and np.abs(actual).max()<=1
                    error=float(np.max(np.abs(expected-actual)))
                    cp=jax.jit(lambda params,obs,rng:routing(params,obs,rng,cpu.metadata),backend='cpu')
                    gp=jax.jit(lambda params,obs,rng:routing(params,obs,rng,gpu.metadata),backend='gpu')
                    cr=jax.device_get(cp(cpu.params,jax.device_put(obs,cpu.device),jax.device_put(key,cpu.device)))
                    gr=jax.device_get(gp(gpu.params,jax.device_put(obs,gpu.device),jax.device_put(key,gpu.device)))
                    rows.append(dict(actor=str(path),actor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        precision=precision,shape=shape,CPU_GPU_max_abs_action_difference=error,
                        CPU_GPU_max_abs_logit_difference=float(np.max(np.abs(cr['logits']-gr['logits']))),
                        selected_component_mismatches=int(np.sum(cr['index']!=gr['index'])),
                        CPU_minimum_CDF_margin=float(np.min(cr['min_CDF_margin'])),
                        GPU_minimum_CDF_margin=float(np.min(gr['min_CDF_margin'])),
                        CPU_actions=expected.tolist(),GPU_actions=actual.tolist(),
                        actual_GPU_device=gpu.device.device_kind))
    checks=dict(default_single_all_within_1e4=all(r['CPU_GPU_max_abs_action_difference']<1e-4 for r in rows if r['precision']is None and r['shape']=='single'),
        explicit_float32_single_and_batch_all_within_1e4=all(r['CPU_GPU_max_abs_action_difference']<1e-4 for r in rows if r['precision']=='float32'),
        default_batch_component_indices_match=all(r['selected_component_mismatches']==0 for r in rows if r['precision']is None and r['shape']=='batch32'))
    assert all(x not in sys.modules for x in ['flax','optax','ogbench'])
    result=dict(time=time.time(),status='completed_precision_diagnosis',checks=checks,rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_updates_performed=0,
        production_code_or_model_changes=False,no_isolated_latency_or_rollout_performance_claim=True,
        scope='Same four actual actor exports and inputs on CPU/GPU; compare default and explicit float32 matmul precision, single and batch32. This result diagnoses backend numerical differences; it does not relax any performance/speed gate.')
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=checks,default_batch_max=max(r['CPU_GPU_max_abs_action_difference']for r in rows if r['precision']is None and r['shape']=='batch32'),
        explicit_float32_max=max(r['CPU_GPU_max_abs_action_difference']for r in rows if r['precision']=='float32'))))


if __name__=='__main__':main()
