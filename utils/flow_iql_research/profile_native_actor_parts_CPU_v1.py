"""Actual native bank actors: preliminary random generation/MLP cost split.

Shared machine, CPU only. External NumPy Gaussian sampling is timed end to
end, including generation and conversion; it changes seeded trajectories and
has no new quality evaluation. Never compare its omitted RNG cost to a full
teacher cost or treat this diagnostic as the isolated final latency gate.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import standalone_onestep_actor_v1 as portable
import jax
import jax.numpy as jnp
import numpy as np
import hashlib


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    manifest=json.loads(a.manifest.read_text())
    for path,digest in manifest['source_sha256'].items():assert sha(path)==digest,path
    a.output.mkdir(parents=True);results=[];timings={}
    for name,case in manifest['cases'].items():
        assert sha(case['actor'])==case['actor_sha256']
        with Path(case['actor']).open('rb')as f:saved=pickle.load(f)
        m=saved['metadata'];assert m['input']=='bank'and m['actor_forward_passes']==1
        actor=portable._Actor(tuple(m['hidden_dims']),m['action_dim'],m['layer_norm'])
        full=portable.make_policy(m,'cpu')
        def random_latents(key):
            action_key,_=jax.random.split(key)
            z=jax.random.normal(action_key,(9,m['action_dim']))
            return z.reshape((m['noise_dim'],))
        random_latents=jax.jit(random_latents,backend='cpu')
        given=jax.jit(lambda pp,x,z:jnp.clip(actor.apply({'params':pp},x,z),-1,1),backend='cpu')
        device=jax.devices('cpu')[0];params=jax.device_put(saved['actor_params'],device)
        observations=jax.device_put(jax.random.normal(jax.random.PRNGKey(49530),(230,m['observation_dim'])),device)
        keys=jax.device_put(jax.random.split(jax.random.PRNGKey(49531),230),device)
        xs=[observations[i]for i in range(230)];ks=[keys[i]for i in range(230)]
        zs=[random_latents(k)for k in ks]
        differences=[]
        for i in range(16):
            differences.append(float(np.max(np.abs(np.asarray(full(params,xs[i],ks[i]))-
                                                   np.asarray(given(params,xs[i],zs[i]))))))
        assert max(differences)<=1e-6
        for i in range(30):
            full(params,xs[i],ks[i]).block_until_ready()
            given(params,xs[i],zs[i]).block_until_ready()
            random_latents(ks[i]).block_until_ready()
        rng=np.random.default_rng(49532)
        def external(i):
            z=rng.standard_normal(m['noise_dim']).astype(np.float32)
            return given(params,xs[i],jax.device_put(z,device))
        methods=dict(full_JAX_RNG_and_actor=lambda i:full(params,xs[i],ks[i]),
            JAX_RNG_only=lambda i:random_latents(ks[i]),
            actor_given_resident_latent=lambda i:given(params,xs[i],zs[i]),
            external_NumPy_RNG_and_actor=external)
        times={n:[]for n in methods}
        names=list(methods)
        for trial in range(3):
            order=names[trial:]+names[:trial]
            for n in order:
                values=[]
                for i in range(30,230):
                    start=time.perf_counter();methods[n](i).block_until_ready()
                    values.append((time.perf_counter()-start)*1000)
                times[n].append(values)
        rows={n:dict(median_ms=float(np.median(v)),p90_ms=float(np.quantile(v,.9)),
            per_trial_median_ms=np.median(v,axis=1).tolist())for n,v in times.items()}
        results.append(dict(case=name,actor_sha256=sha(case['actor']),metadata=m,
            full_given_latent_max_difference=max(differences),rows=rows))
        for n,v in times.items():timings[name+'__'+n]=np.asarray(v)
        print(json.dumps(results[-1]),flush=True)
    raw=a.output/'timings_v1.npz';np.savez_compressed(raw,**timings)
    output=dict(time=time.time(),status='complete',manifest_sha256=sha(a.manifest),
        source_sha256=manifest['source_sha256'],actual_trained_native_actors=True,
        CPU_affinity=sorted(os.sched_getaffinity(0)),isolated=False,
        optimizer_updates=0,quality_evaluation=False,rows=results,
        raw_timings_sha256=sha(raw),trials=3,calls_per_method_per_trial=200,warmup_calls=30,
        resident_weights_and_observations=True,
        external_RNG_generation_conversion_and_actor_all_timed=True,
        external_RNG_changes_seeded_trajectories=True,
        actor_only_given_latent_cost_excludes_RNG=True,
        scope='Shared-machine cost decomposition, no teacher/FQL comparison or isolated final CPU/GPU latency or preserved-quality claim for changed RNG.')
    (a.output/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()
