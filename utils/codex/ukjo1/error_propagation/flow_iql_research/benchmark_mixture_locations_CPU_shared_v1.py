"""Bounded paired CPU timing on an explicitly shared training machine."""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import time
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as core
from utils.flow_iql_research import standalone_mixture_actor_v1 as original
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as routed
from utils.flow_iql_research.watch_native_mixture_decoder_full9_v1 import sha,atomic
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();area=root/'analysis/mixture_location_shared_CPU_timing_v1';area.mkdir(exist_ok=True)
    planpath=area/'prospective_timing_plan_v1.json';assert not planpath.exists()
    assert all(d.platform=='cpu' for d in jax.devices())
    cases=('cube','pz3','cubed','ant');jobs=[]
    for case in cases:
        run=root/'runs'/f'osm1_{case}_mdn9_bc_s0'
        actor=root/f'analysis/mixture_location_deployment_v1/{case}_actor_1000000_component_location_v1.pkl'
        jobs.append(dict(case=case,run=str(run),location_actor=str(actor),location_actor_sha256=sha(actor),
                         native_actor_sha256=sha(run/'actor_1000000.pkl'),checkpoint_sha256=sha(run/'checkpoint_1000000.pkl')))
    sources=[Path(__file__),Path(routed.__file__),Path(original.__file__)]+core.source_paths()
    plan=dict(time=time.time(),source_sha256={str(p.resolve()):sha(p) for p in sources},jobs=jobs,
        CPU_only=True,isolated_speed_claim=False,background_GPU_learners_continue=True,
        protocol='Batch1,residentfloat32weights/input;10warmcalls then3x100synchronouscalls per policy. Three policies rotate block order each repeat. No compile/load/IO/environmentstep included.',
        policies=['teacher_flow_best9','native_mdn_sample','routed_component_location'])
    with planpath.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False)
    result=dict(time=time.time(),status='running',plan_sha256=sha(planpath),rows=[],
                isolated_speed_claim=False,scope='Four-task CPU-only shared-machine diagnostic. Not final nine-task CPU/GPU speed confirmation.')
    target=area/'paired_CPU_results_v1.json'
    for job in jobs:
        run=Path(job['run']);conf=json.loads((run/'config.json').read_text());identity=conf['identity']
        with (run/'checkpoint_1000000.pkl').open('rb') as f:saved=pickle.load(f)
        mdn=original.load_actor(run/'actor_1000000.pkl');location=routed.load_actor(job['location_actor'])
        cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        template=core.create(identity['seed'],jnp.zeros((1,mdn.metadata['observation_dim'])),
                             jnp.zeros((1,mdn.metadata['action_dim'])),cfg,'mdn9_bc')
        state=flax.serialization.from_state_dict(template,saved['state'])
        definitions={
            'teacher_flow_best9':(core.make_teacher_policy(state,'cpu'),core.lean.policy_parameters(state.base)),
            'native_mdn_sample':(mdn.policy,mdn.params),
            'routed_component_location':(location.policy,location.params)}
        device=jax.devices('cpu')[0]
        obs=jax.device_put(np.zeros(mdn.metadata['observation_dim'],np.float32),device)
        keys=jax.device_put(jax.random.split(jax.random.PRNGKey(961),310),device)
        for name,(policy,params) in definitions.items():
            params=jax.device_put(params,device);definitions[name]=(policy,params)
            for index in range(10):policy(params,obs,keys[index]).block_until_ready()
        timings={name:[] for name in definitions};names=list(definitions)
        for repeat in range(3):
            order=names[repeat:]+names[:repeat]
            for name in order:
                policy,params=definitions[name];values=[]
                for index in range(10+100*repeat,110+100*repeat):
                    before=time.perf_counter();policy(params,obs,keys[index]).block_until_ready()
                    values.append((time.perf_counter()-before)*1000)
                timings[name].append(values)
        medians={name:statistics.median(statistics.median(v) for v in repeats) for name,repeats in timings.items()}
        row=dict(case=job['case'],environment=identity['environment'],native_training_seed=identity['seed'],
            medians_ms=medians,repeat_medians_ms={name:[statistics.median(v) for v in repeats] for name,repeats in timings.items()},
            p90_ms={name:float(np.quantile(np.asarray(repeats).ravel(),.9)) for name,repeats in timings.items()},
            routed_to_teacher_ratio=medians['routed_component_location']/medians['teacher_flow_best9'],
            routed_to_native_mdn_ratio=medians['routed_component_location']/medians['native_mdn_sample'])
        assert sha(run/'checkpoint_1000000.pkl')==job['checkpoint_sha256']
        assert sha(run/'actor_1000000.pkl')==job['native_actor_sha256'] and sha(job['location_actor'])==job['location_actor_sha256']
        result['rows'].append(row);atomic(target,result);print(json.dumps(row),flush=True)
    result.update(time=time.time(),status='complete')
    atomic(target,result)


if __name__=='__main__':
    main()
