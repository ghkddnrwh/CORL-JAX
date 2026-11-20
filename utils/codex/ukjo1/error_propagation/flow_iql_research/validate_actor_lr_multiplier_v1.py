"""Check the existing actor-only lr suffix on real task shapes, without training.

One synthetic update per configuration verifies that lr2 actually changes
the actor's Adam step and leaves every serialized lean-base array unchanged.
It is configuration readiness, not a policy-performance experiment.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def same(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict)and set(a)==set(b)and all(same(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b)and len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
    if a is None or b is None:
        return a is b
    return np.array_equal(np.asarray(a),np.asarray(b))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    cases=[('os1_ant_bank_bc_s0',1000000),('os1_cubed_bank_bc_s0',1000000),
           ('os1_hmmed_bank_bc_s0',1000000),('osc1_cube_bank_bc_s0',900000),
           ('osc1_pz3_bank_bc_s0',900000)]
    rows=[]
    for runname,step in cases:
        run=args.root/'runs'/runname
        conf=json.loads((run/'config.json').read_text())
        with(run/f'actor_{step}.pkl').open('rb')as f:artifact=pickle.load(f)
        metadata=artifact['metadata'];obs_dim=metadata['observation_dim'];action_dim=metadata['action_dim']
        env=conf['identity']['environment'];cfg0=core.config_for(env,'bank_bc');cfg2=core.config_for(env,'bank_bc__lr2')
        assert cfg0['lr']==cfg2['lr'] and cfg2['onestep_specification']['actor_lr_multiplier']==2.
        assert {k:v for k,v in cfg0.items()if k!='onestep_specification'}=={k:v for k,v in cfg2.items()if k!='onestep_specification'}
        random=np.random.RandomState(49505);n=16
        batch={k:jnp.asarray(random.normal(size=(n,obs_dim)).astype(np.float32))for k in ['observations','next_observations']}
        batch['actions']=jnp.asarray(np.clip(random.normal(size=(n,action_dim)),-1,1).astype(np.float32))
        data=json.loads((run/'dataset.json').read_text());batch['rewards']=jnp.asarray(random.choice([data['reward_min'],data['reward_max']],size=n).astype(np.float32))
        batch['masks']=jnp.asarray(np.r_[np.ones(n-2),np.zeros(2)].astype(np.float32))
        bounds=(data['return_lower'],data['return_upper'])
        raw=core.create(0,batch['observations'][:1],batch['actions'][:1],cfg0,'bank_bc')
        faster=core.create(0,batch['observations'][:1],batch['actions'][:1],cfg2,'bank_bc__lr2')
        assert same(flax.serialization.to_state_dict(raw),flax.serialization.to_state_dict(faster))
        before=raw.actor.params
        raw,log0=jax.jit(core.build_update('bank_bc',bounds))(raw,batch)
        faster,log2=jax.jit(core.build_update('bank_bc__lr2',bounds))(faster,batch)
        base_exact=same(flax.serialization.to_state_dict(raw.base),flax.serialization.to_state_dict(faster.base))
        assert base_exact,'Actor lr changed serialized base arrays'
        def norm(after):
            return np.sqrt(sum(np.sum((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2)
                for a,b in zip(jax.tree_util.tree_leaves(after),jax.tree_util.tree_leaves(before))))
        delta0,delta2=norm(raw.actor.params),norm(faster.actor.params)
        ratio=float(delta2/delta0)
        assert delta0>0 and abs(ratio-2)<.001,ratio
        assert int(faster.actor.opt_state[0].count)==int(faster.base.q_updates)==int(faster.actor_updates)==1
        assert int(faster.base.v_updates)==0
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(faster))
        assert core.actor_metadata(cfg0,obs_dim,action_dim)==core.actor_metadata(cfg2,obs_dim,action_dim)
        rows.append(dict(environment=env,observation_dim=obs_dim,action_dim=action_dim,
            original_actor_lr=cfg0['lr'],new_actor_lr=2*cfg2['lr'],base_arrays_bitwise_equal=base_exact,
            first_actor_parameter_step_norm_ratio=ratio,Q_Adam=1,actor_Adam=1,V_updates=0,
            deployment_metadata_unchanged=True))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(time=time.time(),status='pass',recipe='bank_bc__lr2',rows=rows,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,*core.source_paths()]},
        original_algorithm_sources_unmodified=True,no_performance_claim=True,
        scope='Existing frozen generic lr suffix, 5real taskshapes, one syntheticSGDupdate/config only. Not GPUefficacy.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
