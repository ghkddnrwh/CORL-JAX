"""Isolated timing from actual1M models; never changes source checkpoints."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import efficient_mean_core_v1 as core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(tree):
    for value in jax.tree_util.tree_leaves(tree):
        if hasattr(value,'block_until_ready'):value.block_until_ready()


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    plan=json.loads(args.input.read_text())
    gpu,cpu=jax.devices('gpu')[0],jax.devices('cpu')[0]
    result=dict(time=time.time(),status='running',rows=[],GPU_kind=gpu.device_kind,
        visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),CPU_affinity=sorted(os.sched_getaffinity(0)),
        input_sha256=sha(args.input),source_sha256=sha(__file__),
        update_protocol='Actual1M model;residentbatch256/scan64;warm64;three timed64update repeats;allsavedoutputs synchronized. No source model writes.',
        inference_protocol='Original1M model parameters;singleobservation;warm10;100synchronous calls per CPU/GPU. No environment step or optimizer call.',
        limits=['Excludescompile,IO,diagnostics andCPU50 evaluation; end-to-end times are separately reported.',
            'Tasks assigned to separate GPUs; every method/reference for a task uses the same GPU/CPU affinity. Ratios are within-task.',
            'Timing repeats are not independent training seeds.'])
    for job in plan['jobs']:
        run=Path(job['run'])
        conf=json.loads((run/'config.json').read_text())
        identity=conf['identity']
        assert identity['variant']==job['variant'] and identity['environment']==job['environment']
        for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
        checkpoint=run/'checkpoint_1000000.pkl'
        digest=sha(checkpoint)
        import pickle
        with checkpoint.open('rb') as stream:saved=pickle.load(stream)
        assert saved['step']==1000000 and saved['identity']==identity
        cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        env,data,_=ogbench.make_env_and_datasets(identity['environment'])
        interval=json.loads((run/'dataset.json').read_text())
        bounds=(interval['return_lower'],interval['return_upper'])
        obs=jax.device_put(np.asarray(data['observations'][:1],np.float32),gpu)
        act=jax.device_put(np.asarray(data['actions'][:1],np.float32),gpu)
        template=core.create(identity['seed'],obs,act,cfg,identity['variant'])
        original=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),gpu)
        assert int(original.q_updates)==1000000
        rng=np.random.RandomState(811)
        indices=rng.randint(len(data['actions']),size=(64,256))
        batches={key:jax.device_put(np.asarray(data[key][indices],np.float32),gpu) for key in ('observations','actions','next_observations','rewards','masks')}
        batches['actions']=jnp.clip(batches['actions'],-1+1e-5,1-1e-5)
        scan=core.make_scan(core.build_update(identity['variant'],bounds))
        before=time.perf_counter()
        state,logs=scan(original,batches)
        sync((state,logs))
        compilation=time.perf_counter()-before
        timings=[]
        for _ in range(3):
            before=time.perf_counter()
            state,logs=scan(state,batches)
            sync((state,logs))
            timings.append((time.perf_counter()-before)*1000/64)
        row=dict(run=run.name,variant=identity['variant'],environment=identity['environment'],seed=identity['seed'],
            checkpoint_sha256=digest,width=cfg['actor_hidden_dims'][0],source_q_updates=1000000,
            compilation_and_first64_seconds=compilation,training_ms_per_update_repeats=timings,
            training_ms_per_update_median=statistics.median(timings))
        row['benchmark_state_finite']=all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(state))
        for backend,device in [('gpu',gpu),('cpu',cpu)]:
            policy=core.make_policy(original,identity['variant'],backend=backend)
            payload=jax.device_put(core.policy_parameters(original),device)
            observation=jax.device_put(np.asarray(data['observations'][0],np.float32),device)
            keys=jax.device_put(jax.random.split(jax.random.PRNGKey(420),110),device)
            for index in range(10):sync(policy(payload,observation,keys[index]))
            measured=[]
            for index in range(10,110):
                before=time.perf_counter()
                sync(policy(payload,observation,keys[index]))
                measured.append((time.perf_counter()-before)*1000)
            row[backend+'_action_ms_median']=statistics.median(measured)
            row[backend+'_action_ms_mean']=statistics.mean(measured)
            row[backend+'_action_ms_p90']=float(np.quantile(measured,.9))
        assert sha(checkpoint)==digest
        result['rows'].append(row)
        atomic(args.output,result)
        print(json.dumps({k:row[k] for k in ['run','training_ms_per_update_median','cpu_action_ms_median','gpu_action_ms_median']}),flush=True)
        env.close()
        del saved,template,original,state,logs,scan,batches,data,policy,payload,keys
        if hasattr(jax,'clear_caches'):jax.clear_caches()
        gc.collect()
    result['status']='complete'
    atomic(args.output,result)


if __name__=='__main__':main()
