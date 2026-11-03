"""Shared-machine CPU planning timings, not the isolated nine-task speed gate."""
import argparse
import json
import os
from pathlib import Path
import pickle
import statistics
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import standalone_mixture_actor_v2 as portable
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and len(os.sched_getaffinity(0))==6 and all(x.platform=='cpu'for x in jax.devices())
    args.output.mkdir(parents=True)
    conf=json.loads((args.teacher_run/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
    for p,d in conf['source_sha256'].items():assert sha(p)==d
    with(args.teacher_run/'actor_1000000.pkl').open('rb')as f:old=pickle.load(f)
    d,a=old['metadata']['observation_dim'],old['metadata']['action_dim']
    template=flat.create(conf['identity']['seed'],jnp.zeros((1,d)),jnp.zeros((1,a)),cfg,conf['identity']['recipe'])
    cp=args.teacher_run/'checkpoint_1000000.pkl';cp_hash=sha(cp)
    with cp.open('rb')as f:saved=pickle.load(f)
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    models=dict(teacher_best9=(flat.make_teacher_policy(teacher),flat.lean.policy_parameters(teacher.base)))
    metadata={}
    for name,dims in [('MDN_256x4',(256,)*4),('MDN_512x4',(512,)*4),('MDN_512x6',(512,)*6)]:
        m=dict(schema='single_actor_deployment_v1',hidden_dims=list(dims),observation_dim=d,action_dim=a,noise_dim=a+1,
            components=9,std_floor=.001,actor_forward_passes=1,generated_candidate_actions=0,requires_critic=False,requires_BC_flow=False,flow_integration_steps=0)
        definition=portable.definition(m);p=definition.init(jax.random.PRNGKey(49341),jnp.zeros((1,d)),jnp.zeros((1,a+1)))['params']
        models[name]=(portable.make_policy(m),p)
        metadata[name]=dict(metadata=m,parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(p)),weight_source='Random finite fixture; timings only, no learned policy performance.')
    device=jax.devices('cpu')[0];observation=jax.device_put(np.zeros(d,np.float32),device)
    keys=jax.device_put(jax.random.split(jax.random.PRNGKey(49342),230),device)
    models={k:(policy,jax.device_put(p,device))for k,(policy,p)in models.items()}
    times={k:[]for k in models}
    for policy,p in models.values():
        for i in range(30):policy(p,observation,keys[i]).block_until_ready()
    orders=[list(models),list(reversed(models)),['MDN_512x4','teacher_best9','MDN_512x6','MDN_256x4']]
    for order in orders:
        for name in order:
            policy,p=models[name];samples=[]
            for i in range(30,230):
                start=time.perf_counter();policy(p,observation,keys[i]).block_until_ready();samples.append((time.perf_counter()-start)*1000)
            times[name].append(samples)
    rows={}
    teacher_ms=statistics.median(np.asarray(times['teacher_best9']).reshape(-1))
    for name,values in times.items():
        values=np.asarray(values);rows[name]=dict(median_ms=float(np.median(values)),p90_ms=float(np.quantile(values,.9)),
            median_ratio_to_teacher=float(np.median(values)/teacher_ms),trials=3,calls_per_trial=200,per_trial_median_ms=np.median(values,axis=1).tolist(),**metadata.get(name,{}))
    np.savez_compressed(args.output/'raw_timings_v1.npz',**{k:np.asarray(v)for k,v in times.items()})
    result=dict(time=time.time(),status='complete',environment=conf['identity']['environment'],teacher_checkpoint_sha256=cp_hash,
        CPU_affinity=sorted(os.sched_getaffinity(0)),GPU_learners_active_during_measurement=16,isolated=False,
        random_actor_fixtures_no_performance_claim=True,optimizer_updates_performed=0,rows=rows,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,portable.__file__,*flat.source_paths()]},
        scope='One-task preliminary CPU compute planning on resident parameters/zero observation, synchronized calls; random actor fixtures. Not learned quality, GPU timing, or isolated final nine-task latency.')
    assert sha(cp)==cp_hash
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
