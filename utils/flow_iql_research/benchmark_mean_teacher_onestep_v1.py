"""Isolated, paired actor-only versus same-checkpoint flow-best9 latency."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import statistics
import time
from utils.flow_iql_research import mean_teacher_onestep_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();jobs=json.loads(args.input.read_text())['jobs']
    result=dict(time=time.time(),status='running',rows=[],source_sha256=sha(__file__),input_sha256=sha(args.input),
        visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),CPU_affinity=sorted(os.sched_getaffinity(0)),GPU_kind=jax.devices('gpu')[0].device_kind,
        protocol='Paired samecheckpoint1000K;singleobservation,10warmcalls+3x100synchronouscallsperpolicy/backend. Residentweights/input. No optimizer/environmentstep/IO/compile included.',
        structural_requirement='Primary loads actor_params only,oneMLPforward,noODE,nocandidateactions,noQrerank.')
    for job in jobs:
        run=Path(job['run']);conf=json.loads((run/'config.json').read_text());identity=conf['identity']
        for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
        actor_path=run/'actor_1000000.pkl';checkpoint=run/'checkpoint_1000000.pkl'
        actor_hash,checkpoint_hash=sha(actor_path),sha(checkpoint)
        with actor_path.open('rb') as stream:artifact=pickle.load(stream)
        with checkpoint.open('rb') as stream:saved=pickle.load(stream)
        assert artifact['identity']==saved['identity']==identity and artifact['checkpoint_sha256']==checkpoint_hash
        assert artifact['step']==saved['step']==1000000
        metadata=artifact['metadata'];cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        template=core.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,identity['recipe'])
        state=flax.serialization.from_state_dict(template,saved['state'])
        row=dict(run=run.name,recipe=identity['recipe'],environment=identity['environment'],training_seed=identity['seed'],
            checkpoint_sha256=checkpoint_hash,actor_artifact_sha256=actor_hash,metadata=metadata,actor_artifact_bytes=actor_path.stat().st_size)
        for backend in ('cpu','gpu'):
            device=jax.devices(backend)[0]
            for mode in ('primary','teacher'):
                if mode=='primary':policy=core.make_actor_policy(metadata,backend);weights=artifact['actor_params']
                else:policy=core.make_teacher_policy(state,backend);weights=core.lean.policy_parameters(state.base)
                weights=jax.device_put(weights,device);obs=jax.device_put(np.zeros(metadata['observation_dim'],np.float32),device)
                keys=jax.device_put(jax.random.split(jax.random.PRNGKey(733),310),device)
                for index in range(10):policy(weights,obs,keys[index]).block_until_ready()
                measurements=[];repeat_medians=[]
                for repeat in range(3):
                    current=[]
                    for index in range(10+repeat*100,110+repeat*100):
                        began=time.perf_counter();policy(weights,obs,keys[index]).block_until_ready()
                        current.append((time.perf_counter()-began)*1000)
                    measurements+=current;repeat_medians.append(statistics.median(current))
                row[f'{backend}_{mode}_ms_median']=statistics.median(repeat_medians)
                row[f'{backend}_{mode}_ms_repeat_medians']=repeat_medians
                row[f'{backend}_{mode}_ms_p90']=float(np.quantile(measurements,.9))
            row[f'{backend}_latency_ratio']=row[f'{backend}_primary_ms_median']/row[f'{backend}_teacher_ms_median']
        assert sha(actor_path)==actor_hash and sha(checkpoint)==checkpoint_hash
        result['rows'].append(row);atomic(args.output,result)
        print(json.dumps({k:row[k] for k in ('run','cpu_latency_ratio','gpu_latency_ratio')}),flush=True)
        del state,template,saved,artifact,policy,weights,keys
        if hasattr(jax,'clear_caches'):jax.clear_caches()
        gc.collect()
    result['status']='complete';atomic(args.output,result)


if __name__=='__main__':main()
