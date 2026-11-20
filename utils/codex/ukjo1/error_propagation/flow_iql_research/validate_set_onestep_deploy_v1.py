"""Symmetry, learner isolation and actor-only export tests, not return tests."""
import json
from pathlib import Path
import time
from utils.flow_iql_research import set_onestep_deploy_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    root=Path('logs/diagnosis/finite_mean_onestep_20261031').resolve();output=root/'analysis/set_actor_implementation_validation_v1.json'
    assert not output.exists();obs=jnp.asarray([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32);act=jnp.asarray([[.2,-.1],[-.4,.3]],jnp.float32)
    batch=dict(observations=obs,actions=act,next_observations=obs*.9,rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    rows=[]
    for recipe in core.RECIPES:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8);state=core.create(6,obs,act,cfg,recipe)
        metadata=core.actor_metadata(cfg,3,2);noise=jax.random.normal(jax.random.PRNGKey(12),(2,9,2))
        permutation=jnp.asarray([8,1,5,2,6,0,4,7,3]);left=state.actor(obs,noise.reshape(2,18));right=state.actor(obs,noise[:,permutation].reshape(2,18))
        symmetry=float(np.abs(np.asarray(left)-np.asarray(right)).max());assert symmetry<1e-6
        update=jax.jit(core.build_update(recipe,(-1000.,0.)));new,logs=update(state,batch)
        reference=jax.jit(core.base.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        base_difference=difference(flax.serialization.to_state_dict(new.base),flax.serialization.to_state_dict(reference));assert base_difference<1e-5
        poison=state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda value:value+.25,state.actor.params)))
        second,_=update(poison,batch);assert difference(flax.serialization.to_state_dict(new.base),flax.serialization.to_state_dict(second.base))==0
        exported=core.export_actor(new,metadata);assert exported['schema']=='set_actor_deployment_v1'
        policy=core.make_actor_policy(metadata);result=np.asarray(policy(exported['actor_params'],obs[0],jax.random.PRNGKey(42)))
        assert result.shape==(2,)and np.isfinite(result).all()and np.max(np.abs(result))<=1
        assert int(new.base.q_updates)==int(new.actor_updates)==1 and int(new.base.v_updates)==0
        summary=core.probe_summary(jax.device_get(core.make_probe(recipe,(-1000.,0.))(new,batch)),(-1000.,0.));assert summary['finite']
        rows.append(dict(recipe=recipe,permutation_max_difference=symmetry,base_update_difference=base_difference,actor_perturbation_base_effect=0.,
            actor_only_CPU_action_finite=True,metadata=metadata,scope='Synthetic one-update implementation check only.'))
        print(json.dumps(dict(recipe=recipe,status='passed')),flush=True)
    output.write_text(json.dumps(dict(time=time.time(),status='passed',cases=rows,source_sha256={str(p):sha(p)for p in core.source_paths()},
        no_performance_claim=True,no_production_queue_change=True),indent=2,allow_nan=False))


if __name__=='__main__':main()
