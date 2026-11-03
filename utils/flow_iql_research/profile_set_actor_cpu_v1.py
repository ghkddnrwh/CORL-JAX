"""Architecture feasibility timing only: untrained actors, CPU under load."""
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import set_onestep_deploy_core_v1 as structured
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import jax
import jax.numpy as jnp
import numpy as np


def main():
    root=Path('logs/diagnosis/finite_mean_onestep_20261031').resolve();output=root/'analysis/set_actor_preliminary_CPU_timing_v1.json'
    assert not output.exists()and all(d.platform=='cpu'for d in jax.devices())
    rows=[]
    for tag,environment in [('ant',flat.ENVIRONMENTS[0]),('cubed',flat.ENVIRONMENTS[5])]:
        data=json.loads((root/'runs'/f'os1_{tag}_bank_bc_s0'/'dataset.json').read_text())
        observations=jnp.zeros((1,data['observation_dim']));actions=jnp.zeros((1,data['action_dim']))
        for name,module,recipe in [('flat_bank',flat,'bank_bc'),('set_attention',structured,'set_bc')]:
            cfg=module.config_for(environment,recipe);state=module.create(7,observations,actions,cfg,recipe)
            metadata=module.actor_metadata(cfg,data['observation_dim'],data['action_dim']);artifact=module.export_actor(state,metadata)
            policy=module.make_actor_policy(metadata);params=jax.device_put(artifact['actor_params'],jax.devices('cpu')[0])
            observation=observations[0];keys=jax.random.split(jax.random.PRNGKey(126),310)
            for index in range(10):policy(params,observation,keys[index]).block_until_ready()
            medians=[]
            for repeat in range(3):
                current=[]
                for index in range(10+repeat*100,110+repeat*100):
                    start=time.perf_counter();policy(params,observation,keys[index]).block_until_ready();current.append((time.perf_counter()-start)*1000)
                medians.append(statistics.median(current))
            row=dict(environment=environment,architecture=name,metadata=metadata,actor_updates=int(state.actor_updates),
                actor_parameter_count=sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(params)),CPU_ms_repeat_medians=medians,
                CPU_ms_median=statistics.median(medians),output_shape=list(np.asarray(policy(params,observation,keys[0])).shape))
            assert row['actor_updates']==0 and row['output_shape']==[data['action_dim']]
            rows.append(row);print(json.dumps({k:row[k]for k in ('environment','architecture','actor_parameter_count','CPU_ms_median')}),flush=True)
    output.write_text(json.dumps(dict(time=time.time(),status='complete',rows=rows,training_updates=0,
        source_sha256={str(Path(m.__file__).resolve()):sha(m.__file__)for m in (flat,structured)},
        scope='PreliminaryCPUarchitecturefeasibilityonly. Untrainedweights,sharedmachineload,noGPUmeasurement,noperformanceclaim,nogoalcompletionevidence.'),indent=2))


if __name__=='__main__':main()
