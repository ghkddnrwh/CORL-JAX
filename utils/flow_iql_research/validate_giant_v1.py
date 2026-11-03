"""Actual Giant data/MDP preflight and unchanged-update adapter checks."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import giant_core_v1 as core
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def difference(a,b):
    first,ta=jax.tree_util.tree_flatten(a)
    second,tb=jax.tree_util.tree_flatten(b)
    assert ta==tb
    return max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64)))) for x,y in zip(first,second))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    env,data,_=ogbench.make_env_and_datasets(core.ENVIRONMENT)
    reset=prepare_seeded_action_space(env,42)
    spec=env.spec
    assert spec.id=='antmaze-giant-singletask-v0' and spec.kwargs['reward_task_id']==0
    assert spec.max_episode_steps==1000
    reward_min,reward_max=float(data['rewards'].min()),float(data['rewards'].max())
    assert reward_min==-1 and reward_max==0
    observations=jnp.asarray(data['observations'][:1],jnp.float32)
    actions=jnp.asarray(data['actions'][:1],jnp.float32)
    rng=np.random.RandomState(138)
    indices=rng.randint(len(data['actions']),size=(4,8))
    batches={key:jnp.asarray(data[key][indices],jnp.float32) for key in ('observations','actions','next_observations','rewards','masks')}
    batches['actions']=jnp.clip(batches['actions'],-1+1e-5,1-1e-5)
    rows=[]
    for discount in (.99,.999):
        interval=core.bounds(core.ENVIRONMENT,discount,reward_min,reward_max)
        for kind in core.KINDS:
            cfg=core.config_for(core.ENVIRONMENT,kind,discount)
            assert tuple(cfg['actor_hidden_dims'])==tuple(cfg['value_hidden_dims'])==(512,)*4
            cfg.update(actor_hidden_dims=(8,)*4,value_hidden_dims=(8,)*4)
            initial=core.create(19,observations,actions,cfg,kind)
            update=core.build_update(kind,interval)
            full,logs=core.make_scan(update)(initial,batches)
            half,_=core.make_scan(update)(initial,jax.tree_util.tree_map(lambda x:x[:2],batches))
            restored=flax.serialization.from_state_dict(initial,flax.serialization.to_state_dict(half))
            resumed,_=core.make_scan(update)(restored,jax.tree_util.tree_map(lambda x:x[2:],batches))
            assert difference(full,resumed)==0
            profile='bounded_v2_mean_empirical_backup' if kind=='mean_backup' else 'official'
            direct=core.module(kind).build_update(profile,interval,core.agent_kind(kind))
            reference,_=core.make_scan(direct)(initial,batches)
            gap=difference(full,reference)
            assert gap==0 and int(full.q_updates)==4 and int(full.v_updates)==core.V_UPDATES[kind]*4
            assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(full))
            rows.append(dict(kind=kind,discount=discount,full_state_vs_frozen_update_max_abs=gap,
                serialized2plus2_vs4_exact=True,Q_Adam_updates=4,V_total=int(full.v_updates)))
    hashes={}
    for suffix in ('','-val'):
        path=Path.home()/'.ogbench/data'/('antmaze-giant-navigate-v0'+suffix+'.npz')
        hashes[str(path)]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(time=time.time(),status='passed',environment=core.ENVIRONMENT,registered_env=spec.id,
        default_task_kwargs=spec.kwargs,actual_task_id=env.unwrapped.cur_task_id,max_episode_steps=spec.max_episode_steps,
        dataset_size=len(data['actions']),observation_dim=observations.shape[-1],action_dim=actions.shape[-1],
        reward_min=reward_min,reward_max=reward_max,reset_metadata=reset,raw_dataset_hashes=hashes,
        six_kind_discount_checks=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='RealGiant data, tiny8wide networks for equation/serialization checks; productionwidth512 verified in configs. No policy performance claims.')
    args.output.write_text(json.dumps(result,indent=2))
    env.close()
    print(json.dumps(dict(status='passed',checks=len(rows),default_task=result['actual_task_id'],dataset_size=result['dataset_size'])),flush=True)


if __name__=='__main__':main()
