"""Isolated GPU update and CPU/GPU single-action latency; not policy training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import core, benchmark_core, mean_empirical_backup_core_v1 as mean_core
from utils.flow_iql_research.train_mean_empirical_backup_v1 import make_policy
from utils.propagation_research.reward_bounds import return_bounds
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sync(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf,'block_until_ready'):
            leaf.block_until_ready()


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    gpu=jax.devices('gpu')[0]
    cpu=jax.devices('cpu')[0]
    result=dict(time=time.time(),status='running',GPU=str(gpu),GPU_kind=gpu.device_kind,
        jax_version=jax.__version__,CPU_affinity=sorted(os.sched_getaffinity(0)),
        memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),rows=[],
        training_timing='Batch256,scan64,warm64,three timed64-update repeats. Device batches resident; synchronize all outputs. Excludescompile,dataIO,diagnostics,checkpoints,evaluation.',
        inference_timing='Singleobservation;100warmed synchronous calls per CPU/GPU; includes Python dispatch,excludesenvironment/reset.',
        performance_training_replicates=0,paper_hardware_comparison=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for short,environment in [('ant','antmaze-large-navigate-singletask-v0'),('cubed','cube-double-play-singletask-v0')]:
        env,data,_=ogbench.make_env_and_datasets(environment)
        obs,act=data['observations'][:1],data['actions'][:1]
        interval=return_bounds(environment,.999,float(data['rewards'].min()),float(data['rewards'].max()),'ogbench')
        random=np.random.RandomState(811)
        indices=random.randint(len(data['actions']),size=(64,256))
        batches={k:jax.device_put(np.asarray(data[k][indices],np.float32),gpu) for k in ('observations','actions','next_observations','rewards','masks')}
        batches['actions']=jnp.clip(batches['actions'],-1+1e-5,1-1e-5)
        cases=[('FQL_public',512),('finite_mean_original',512 if short=='ant' else 256)]
        if short=='cubed':
            cases.append(('FQL_matched_width256',256))
        for name,width in cases:
            if name.startswith('FQL'):
                module,kind=benchmark_core,'fql'
                cfg=module.config_for(environment,.999,width=width,kind=kind)
                update=module.build_update('official',interval,kind)
            else:
                module,kind=mean_core,'ifql'
                cfg=module.config_for(environment,.999,.9,width)
                cfg.update(num_samples=9,flow_steps=10,agent_name='ifql',policy_type='ifql',flow_temperature=3.,distill_coefficient=30.)
                update=module.build_update('bounded_v2_mean_empirical_backup',interval,kind)
            state=module.create(0,jax.device_put(obs,gpu),jax.device_put(act,gpu),cfg,kind)
            scan=core.make_scan(update)
            start=time.perf_counter()
            state,logs=scan(state,batches)
            sync((state,logs))
            compile_inclusive=time.perf_counter()-start
            repeats=[]
            for _ in range(3):
                start=time.perf_counter()
                state,logs=scan(state,batches)
                sync((state,logs))
                repeats.append((time.perf_counter()-start)*1000/64)
            row=dict(task=short,environment=environment,method=name,width=width,config=cfg,
                compilation_and_first64_seconds=compile_inclusive,training_ms_per_update_repeats=repeats,
                training_ms_per_update_median=statistics.median(repeats))
            for backend,device in [('gpu',gpu),('cpu',cpu)]:
                agent=state.agent
                def action(params,observation,key):
                    current=agent.replace(network=agent.network.replace(params=params))
                    return current.sample_actions(observations=observation,seed=key,temperature=0.)
                policy=jax.jit(action,backend=backend)
                params=jax.device_put(agent.network.params,device)
                observation=jax.device_put(np.asarray(obs[0],np.float32),device)
                keys=jax.device_put(jax.random.split(jax.random.PRNGKey(420),110),device)
                for index in range(10):
                    sync(policy(params,observation,keys[index]))
                times=[]
                for index in range(10,110):
                    start=time.perf_counter()
                    sync(policy(params,observation,keys[index]))
                    times.append((time.perf_counter()-start)*1000)
                row[backend+'_action_ms_mean']=statistics.mean(times)
                row[backend+'_action_ms_median']=statistics.median(times)
                row[backend+'_action_ms_p90']=float(np.quantile(times,.9))
            result['rows'].append(row)
            atomic(args.output,result)
            print(json.dumps({k:row[k] for k in ['task','method','training_ms_per_update_median','cpu_action_ms_median','gpu_action_ms_median']}),flush=True)
            del state,logs,scan,update,params,policy,agent
            if hasattr(jax,'clear_caches'):
                jax.clear_caches()
            gc.collect()
        env.close()
        del data,batches
    result['status']='complete'
    atomic(args.output,result)


if __name__=='__main__':main()
