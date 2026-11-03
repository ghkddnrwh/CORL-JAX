"""Exploratory CPU latency while GPU training continues: never a speed gate.

Compare the actual portable actor with same-checkpoint flow-best9. The shared
machine is deliberately not drained; these measurements are provisional.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research import standalone_onestep_actor_v1 as portable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    plan=json.loads(args.input.read_text())
    for p,d in plan['source_sha256'].items():assert sha(p)==d,p
    result=dict(time=time.time(),status='running',input_sha256=sha(args.input),
        source_sha256=plan['source_sha256'],rows=[],CPU_affinity=sorted(os.sched_getaffinity(0)),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        isolated_machine=False,counts_toward_acceptance_speed_gate=False,
        registered_GPU_updates=0,teacher_updates=0,rollouts=0,
        protocol='Actual1Mportableactor versus same-checkpoint best9flow policy. Single zero observation; residentCPUweights/keys;10warmcalls+3x100synchronouscalls. Alternate policy order per repeat. Excludes compile/load/environment/IO. Concurrent16GPUlearners continue; no definitive latency claim.')
    for job in plan['jobs']:
        run=Path(job['run']);conf=json.loads((run/'config.json').read_text());identity=conf['identity']
        assert identity['recipe']=='bank_bc'
        for p,d in conf['source_sha256'].items():assert sha(p)==d,p
        actor_path=run/'actor_1000000.pkl';checkpoint=run/'checkpoint_1000000.pkl'
        assert sha(actor_path)==job['actor_sha256'] and sha(checkpoint)==job['checkpoint_sha256']
        with actor_path.open('rb') as stream:artifact=pickle.load(stream)
        with checkpoint.open('rb') as stream:saved=pickle.load(stream)
        assert artifact['identity']==saved['identity']==identity
        assert artifact['checkpoint_sha256']==job['checkpoint_sha256']
        metadata=artifact['metadata'];cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        template=core.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),
                             jnp.zeros((1,metadata['action_dim'])),cfg,identity['recipe'])
        state=flax.serialization.from_state_dict(template,saved['state'])
        assert int(state.base.q_updates)==int(state.actor_updates)==1000000 and int(state.base.v_updates)==0
        exported=flax.serialization.to_state_dict(state.actor.params)
        for a,b in zip(jax.tree_util.tree_leaves(exported),jax.tree_util.tree_leaves(artifact['actor_params'])):
            assert np.array_equal(a,b)
        policy={'primary':portable.make_policy(metadata,'cpu'),
                'teacher':core.make_teacher_policy(state,'cpu')}
        weights={'primary':artifact['actor_params'],'teacher':core.lean.policy_parameters(state.base)}
        weights={k:jax.device_put(v,jax.devices('cpu')[0]) for k,v in weights.items()}
        obs=jnp.zeros(metadata['observation_dim'],dtype=jnp.float32)
        keys=jax.random.split(jax.random.PRNGKey(733),310)
        measurements={mode:[] for mode in policy};medians={mode:[] for mode in policy}
        reference=core.make_actor_policy(metadata,'cpu')
        assert np.array_equal(policy['primary'](weights['primary'],obs,keys[0]),
                              reference(weights['primary'],obs,keys[0]))
        for mode in policy:
            for index in range(10):policy[mode](weights[mode],obs,keys[index]).block_until_ready()
        for repeat in range(3):
            for mode in (('primary','teacher') if repeat%2==0 else ('teacher','primary')):
                current=[]
                for index in range(10+100*repeat,110+100*repeat):
                    began=time.perf_counter()
                    policy[mode](weights[mode],obs,keys[index]).block_until_ready()
                    current.append((time.perf_counter()-began)*1000)
                measurements[mode].extend(current);medians[mode].append(statistics.median(current))
        row=dict(run=run.name,environment=identity['environment'],recipe=identity['recipe'],seed=identity['seed'],
                 actor_sha256=job['actor_sha256'],checkpoint_sha256=job['checkpoint_sha256'],metadata=metadata,
                 actual_actor_updates=1000000,actual_critic_updates=1000000,actual_V_updates=0,
                 portable_and_training_policy_actions_identical=True)
        for mode in policy:
            row[mode+'_ms_repeat_medians']=medians[mode]
            row[mode+'_ms_median']=statistics.median(medians[mode])
            row[mode+'_ms_p90']=float(np.quantile(measurements[mode],.9))
        row['CPU_latency_ratio']=row['primary_ms_median']/row['teacher_ms_median']
        row['CPU_speedup']=1/row['CPU_latency_ratio']
        assert sha(actor_path)==job['actor_sha256'] and sha(checkpoint)==job['checkpoint_sha256']
        result['rows'].append(row);write(args.output,result)
        print(json.dumps(row),flush=True)
        del state,template,saved,artifact,policy,weights,reference
        if hasattr(jax,'clear_caches'):jax.clear_caches()
        gc.collect()
    result.update(status='complete',completed=time.time())
    write(args.output,result)


if __name__=='__main__':
    main()
