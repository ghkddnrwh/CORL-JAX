"""Implementation tests; synthetic updates are never performance replicates."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def difference(left,right):
    a,b=jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right)
    assert len(a)==len(b)
    return max(float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64))))for x,y in zip(a,b))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert args.part in (0,1) and not args.output.exists()
    recipes=core.INITIAL_RECIPES[args.part::2]
    obs=jnp.asarray([[.1,-.2,.3],[.2,.4,-.1],[.6,.2,.5],[.2,-.4,.8]],jnp.float32)
    actions=jnp.asarray([[.2,-.1],[.4,.2],[-.2,.6],[.3,.1]],jnp.float32)
    batch=dict(observations=obs,actions=actions,next_observations=obs*.9,rewards=jnp.full((4,),-1.),masks=jnp.ones((4,)))
    results=[]
    for recipe in recipes:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8)
        state=core.create(7,obs,actions,cfg,recipe)
        step=jax.jit(core.build_update(recipe,(-1000.,0.)))
        original=state
        first,logs=step(state,batch)
        plain=jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        baseline_difference=difference(flax.serialization.to_state_dict(first.base),flax.serialization.to_state_dict(plain))
        assert baseline_difference<1e-5,baseline_difference
        perturbed=state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda p:p+.5,state.actor.params)))
        poison,_=step(perturbed,batch)
        assert difference(flax.serialization.to_state_dict(first.base),flax.serialization.to_state_dict(poison.base))==0
        state=first
        for _ in range(3):state,logs=step(state,batch)
        resumed=original
        for _ in range(2):resumed,_=step(resumed,batch)
        resumed=flax.serialization.from_state_dict(original,pickle.loads(pickle.dumps(flax.serialization.to_state_dict(jax.device_get(resumed)))))
        for _ in range(2):resumed,_=step(resumed,batch)
        assert difference(flax.serialization.to_state_dict(state),flax.serialization.to_state_dict(resumed))==0
        assert int(state.actor_updates)==int(state.base.q_updates)==4 and int(state.base.v_updates)==0
        assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
        metadata=core.actor_metadata(cfg,3,2);artifact=core.export_actor(state,metadata)
        assert set(artifact)=={'schema','metadata','actor_params','actor_updates','Q_updates'}
        assert not metadata['requires_critic'] and not metadata['requires_BC_flow'] and metadata['actor_forward_passes']==1
        cpu=jax.devices('cpu')[0]
        policy=core.make_actor_policy(metadata)
        payload=jax.device_put(artifact['actor_params'],cpu);ob=jax.device_put(obs[0],cpu);key=jax.device_put(jax.random.PRNGKey(42),cpu)
        prediction=np.asarray(policy(payload,ob,key));assert prediction.shape==(2,) and np.isfinite(prediction).all() and np.max(np.abs(prediction))<=1
        probe=core.make_probe(recipe,(-1000.,0.));summary=core.probe_summary(jax.device_get(probe(state,batch)),(-1000.,0.))
        assert summary['finite']
        item=dict(recipe=recipe,base_step_difference_from_lean=baseline_difference,actor_perturbation_base_effect=0.,serialized_resume_difference=0.,
            Q_updates=4,V_updates=0,actor_updates=4,standalone_actor_CPU_action=prediction.tolist(),deployment_metadata=metadata,
            actor_parameter_count=sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(artifact['actor_params'])),finite_probe=True)
        results.append(item);print(json.dumps(dict(recipe=recipe,status='passed')),flush=True)
        if hasattr(jax,'clear_caches'):jax.clear_caches()
    result=dict(time=time.time(),status='passed',part=args.part,cases=results,scope='Synthetic implementation/isolation/serialization/export tests, not performance experiments.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in core.source_paths()+[Path(__file__).resolve()]})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
