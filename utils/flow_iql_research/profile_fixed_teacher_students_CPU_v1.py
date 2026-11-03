"""Read-only, shared-machine timings for actual trained direct fixed32 students.

Resident inputs/weights, synchronized calls and three balanced orders. This
is not the final isolated CPU/GPU nine-task speed test.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import standalone_onestep_actor_v1 as portable
from utils.flow_iql_research.evaluate_fixed_bank_teacher_v1 import make_policy as fixed_policy
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-case', type=Path, required=True)
    parser.add_argument('--fixed-teacher-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    parent = json.loads((args.study_case/'status.json').read_text())
    entry = parent['actors']['direct_MSE']
    assert entry['status'] == 'complete' and sha(entry['private_actor_path']) == entry['private_actor_sha256']
    for p, digest in parent['source_sha256'].items():
        assert sha(p) == digest, p
    with Path(entry['private_actor_path']).open('rb') as f:
        artifact = pickle.load(f)
    assert artifact['actor_Adam_updates'] == 100000
    metadata = artifact['metadata']
    cache = args.study_case/'fixed32_replay_cache_v1.npz'
    assert sha(cache) == parent['labeled_cache_sha256']
    with np.load(cache) as z:
        observations = z['observations'][-4096:-3866].copy()
    checkpoint = Path(parent['teacher_checkpoint'])
    assert sha(checkpoint) == parent['teacher_checkpoint_sha256']
    conf = json.loads((checkpoint.parent/'config.json').read_text())
    cfg = dict(conf['identity']['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    template = flat.create(conf['identity']['seed'], jnp.asarray(observations[:1]),
        jnp.zeros((1,metadata['action_dim'])), cfg, conf['identity']['recipe'])
    with checkpoint.open('rb') as f:
        saved = pickle.load(f)
    teacher = flax.serialization.from_state_dict(template, saved['state'])
    bank_path = args.fixed_teacher_case/'fixed_noise_bank32_v1.npy'
    assert sha(bank_path) == parent['fixed_noise_bank_sha256']
    models = {
        'original_stochastic_best9': (flat.make_teacher_policy(teacher), flat.lean.policy_parameters(teacher.base)),
        'fixed32_teacher': (fixed_policy(teacher,jnp.asarray(np.load(bank_path))), flat.lean.policy_parameters(teacher.base)),
        'direct_MSE_actor_only': (portable.make_policy(metadata), artifact['actor_params']),
    }
    device = jax.devices('cpu')[0]
    obs = [jax.device_put(x,device) for x in observations]
    keys = [jax.device_put(x,device) for x in jax.random.split(jax.random.PRNGKey(49502),230)]
    models = {k:(policy,jax.device_put(params,device)) for k,(policy,params) in models.items()}
    # Validate the portable implementation against the exact fitting graph.
    definition = flat.actor_definition(metadata)
    fitting_policy = jax.jit(lambda p,x:jnp.clip(definition.apply({'params':p},x,
                        jnp.zeros((metadata['action_dim'],),dtype=x.dtype)),-1,1),backend='cpu')
    expected = jnp.stack([fitting_policy(models['direct_MSE_actor_only'][1],x) for x in obs])
    actual = jnp.stack([models['direct_MSE_actor_only'][0](models['direct_MSE_actor_only'][1],x,keys[i]) for i,x in enumerate(obs)])
    error = float(jnp.max(jnp.abs(expected-actual)))
    assert error < 1e-6
    for policy, params in models.values():
        for i in range(30):
            policy(params,obs[i],keys[i]).block_until_ready()
    times = {k:[] for k in models}
    orders = [list(models),list(reversed(models)),['fixed32_teacher','direct_MSE_actor_only','original_stochastic_best9']]
    for order in orders:
        for name in order:
            policy, params = models[name]
            values = []
            for i in range(30,230):
                start = time.perf_counter()
                policy(params,obs[i%len(obs)],keys[i]).block_until_ready()
                values.append((time.perf_counter()-start)*1000)
            times[name].append(values)
    args.output.mkdir(parents=True)
    raw = args.output/'raw_timings_v1.npz'
    np.savez_compressed(raw,**{k:np.asarray(v) for k,v in times.items()})
    reference = float(np.median(times['original_stochastic_best9']))
    rows = {name:dict(median_ms=float(np.median(x)),p90_ms=float(np.quantile(x,.9)),
        median_ratio_to_original_best9=float(np.median(x)/reference),
        per_trial_median_ms=np.median(x,axis=1).tolist()) for name,x in times.items()}
    result = dict(time=time.time(),status='complete',environment=parent['environment'],
        actual_trained_actor_sha256=sha(entry['private_actor_path']),
        teacher_checkpoint_sha256=sha(checkpoint),raw_timings_sha256=sha(raw),
        CPU_affinity=sorted(os.sched_getaffinity(0)),isolated=False,
        GPU_learners_active=16,optimizer_updates=0,portable_fitting_graph_absmax=error,
        resident_inputs_and_weights=True,synchronized=True,warmup_calls=30,
        trials=3,calls_per_trial=200,rows=rows,
        source_sha256={str(Path(p).resolve()):sha(p) for p in
                       [__file__,portable.__file__,*flat.source_paths()]},
        scope='Preliminary actual-actor CPU timings while all research jobs continue. '
              'No final isolated/GPU/nine-task latency claim.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
