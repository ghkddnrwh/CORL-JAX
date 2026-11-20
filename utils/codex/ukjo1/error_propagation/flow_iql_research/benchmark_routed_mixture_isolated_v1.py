"""Paired isolated CPU/GPU latency of an explicit routed actor and Flow-best9."""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import statistics
import time
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as core
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
from utils.flow_iql_research.routed_policy_verifier_v1 import actor_path
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();jobs=json.loads(a.input.read_text())['jobs']
    result=dict(time=time.time(),status='running',rows=[],source_sha256=sha(__file__),input_sha256=sha(a.input),
        visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),CPU_affinity=sorted(os.sched_getaffinity(0)),
        GPU_kind=jax.devices('gpu')[0].device_kind,
        protocol='Paired1000K,batch1,residentweights/input,10warmcalls+3x100synchronouscalls. Alternate primary/teacher block order. No compile/load/optimizer/environmentstep/IO included.',
        structural_requirement='Primary one actor,6Dense,one selected action head,noFlow/noQ/noactionbank.')
    for job in jobs:
        run=Path(job['run']);root=run.parent.parent;conf=json.loads((run/'config.json').read_text())
        identity=conf['identity'];assert identity['recipe']=='mdn9_bc'
        path=actor_path(root,run.name,1000000);checkpoint=run/'checkpoint_1000000.pkl';native=run/'actor_1000000.pkl'
        actor_digest,checkpoint_digest,native_digest=sha(path),sha(checkpoint),sha(native)
        with path.open('rb') as f:artifact=pickle.load(f)
        with checkpoint.open('rb') as f:saved=pickle.load(f)
        assert artifact['training_identity']==saved['identity']==identity
        assert artifact['training_checkpoint_sha256']==checkpoint_digest and artifact['native_actor_sha256']==native_digest
        assert artifact['training_step']==saved['step']==1000000
        for source,digest in artifact['deployment_source_sha256'].items():assert sha(source)==digest
        for source,digest in conf['source_sha256'].items():assert sha(source)==digest
        metadata=artifact['metadata'];cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        template=core.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,'mdn9_bc')
        state=flax.serialization.from_state_dict(template,saved['state'])
        row=dict(run=run.name,recipe='mdn9_bc__component_location',native_training_recipe='mdn9_bc',
            environment=identity['environment'],training_seed=identity['seed'],checkpoint_sha256=checkpoint_digest,
            actor_artifact_sha256=actor_digest,native_actor_artifact_sha256=native_digest,metadata=metadata,
            actor_artifact_bytes=path.stat().st_size)
        for backend in ('cpu','gpu'):
            device=jax.devices(backend)[0];obs=jax.device_put(np.zeros(metadata['observation_dim'],np.float32),device)
            keys=jax.device_put(jax.random.split(jax.random.PRNGKey(733),310),device)
            policies={'primary':(portable.make_policy(metadata,backend),artifact['actor_params']),
                      'teacher':(core.make_teacher_policy(state,backend),core.lean.policy_parameters(state.base))}
            for mode,(policy,weights) in policies.items():
                weights=jax.device_put(weights,device);policies[mode]=(policy,weights)
                for index in range(10):policy(weights,obs,keys[index]).block_until_ready()
            all_values={mode:[] for mode in policies}
            for repeat in range(3):
                for mode in (('primary','teacher') if repeat%2==0 else ('teacher','primary')):
                    policy,weights=policies[mode];current=[]
                    for index in range(10+repeat*100,110+repeat*100):
                        before=time.perf_counter();policy(weights,obs,keys[index]).block_until_ready()
                        current.append((time.perf_counter()-before)*1000)
                    all_values[mode].append(current)
            for mode,values in all_values.items():
                medians=[statistics.median(v) for v in values]
                row[f'{backend}_{mode}_ms_median']=statistics.median(medians)
                row[f'{backend}_{mode}_ms_repeat_medians']=medians
                row[f'{backend}_{mode}_ms_p90']=float(np.quantile(np.asarray(values).ravel(),.9))
            row[f'{backend}_latency_ratio']=row[f'{backend}_primary_ms_median']/row[f'{backend}_teacher_ms_median']
        assert sha(path)==actor_digest and sha(checkpoint)==checkpoint_digest and sha(native)==native_digest
        result['rows'].append(row);atomic(a.output,result)
        print(json.dumps({k:row[k] for k in ('run','cpu_latency_ratio','gpu_latency_ratio')}),flush=True)
        del state,template,saved,artifact,policies,policy,weights,keys
        if hasattr(jax,'clear_caches'):jax.clear_caches()
        gc.collect()
    result.update(status='complete',time=time.time());atomic(a.output,result)


if __name__=='__main__':
    main()
