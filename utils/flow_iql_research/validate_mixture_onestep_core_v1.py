"""Synthetic logic/resume/portable checks, not performance or registration."""
import argparse
import json
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as core
from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research import mixture_guidance_losses_v1 as guidance
from utils.flow_iql_research import standalone_mixture_actor_v1 as standalone
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-directory',type=Path,required=True);args=parser.parse_args()
    assert not args.output_directory.exists() and all(d.platform=='cpu' for d in jax.devices())
    args.output_directory.mkdir(parents=True)
    obs=jnp.asarray([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32)
    action=jnp.asarray([[.2,-.1],[-.4,.3]],jnp.float32)
    batch=dict(observations=obs,actions=action,next_observations=.9*obs,rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    rows=[];initial=None
    for recipe in core.RECIPES:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8);spec=cfg['onestep_specification']
        state=core.create(6,obs,action,cfg,recipe)
        if initial is None:initial=flax.serialization.to_state_dict(state.actor.params)
        else:assert difference(initial,flax.serialization.to_state_dict(state.actor.params))==0
        rng=jax.random.PRNGKey(29001)
        ob,noise,target,heads,proposals,index=core.teacher_batch(state,batch,rng,spec)
        original=base.teacher_batch(state,batch,rng,spec)
        assert np.array_equal(target,original[2]) and np.array_equal(proposals,original[4])
        assert np.array_equal(index,heads.min(0).argmax(0)) and noise.shape==(10,2,3)
        assert not np.array_equal(noise[:9,:,:2],original[1])
        p=state.actor(ob,jnp.zeros((2,3)),return_parameters=True)
        expected,_=guidance.loss(p,target,ob,noise[:9,:,:2],state.base.agent.network.select('critic'),spec['effective_alpha'],spec['objective']=='maximize')
        actual,_=core.actor_loss(state,ob,noise,target,heads,spec,state.actor.params)
        assert float(actual-expected)==0
        update=jax.jit(core.build_update(recipe,(-1000.,0.)))
        one,_=update(state,batch)
        reference=jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        deviation=difference(flax.serialization.to_state_dict(one.base),flax.serialization.to_state_dict(reference))
        assert deviation<1e-5
        perturbed=state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda x:x+.25,state.actor.params)))
        other,_=update(perturbed,batch)
        assert difference(flax.serialization.to_state_dict(one.base),flax.serialization.to_state_dict(other.base))==0
        full=state
        for _ in range(8):full,logs=update(full,batch)
        resumed=state
        for _ in range(4):resumed,_=update(resumed,batch)
        resumed=flax.serialization.from_state_dict(state,flax.serialization.to_state_dict(resumed))
        for _ in range(4):resumed,_=update(resumed,batch)
        assert difference(flax.serialization.to_state_dict(full),flax.serialization.to_state_dict(resumed))==0
        assert int(full.base.q_updates)==int(full.base.agent.network.opt_state[0].count)==8
        assert int(full.actor_updates)==int(full.actor.opt_state[0].count)==8
        assert int(full.base.v_updates)==int(full.base.v_opt[0].count)==0
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(full))
        assert float(logs['deployment/distill_MSE_is_training_objective'])==0
        if recipe=='mdn9_bc':assert float(logs['deployment/Q_loss'])==0
        values=core.make_probe(recipe,(-1000.,0.))(full,batch)
        summary=core.probe_summary(jax.device_get(values),(-1000.,0.))
        assert summary['finite'] and np.isfinite(summary['mixture_NLL_per_dimension'])
        metadata=core.actor_metadata(cfg,3,2)
        assert metadata['noise_dim']==3 and metadata['components']==9 and metadata['realized_action_samples']==1
        artifact=core.export_actor(full,metadata)
        path=args.output_directory/(recipe+'_implementation_actor8.pkl')
        with path.open('xb') as stream:pickle.dump(artifact,stream)
        loaded=standalone.load_actor(path)
        key=jax.random.PRNGKey(42);actual=np.asarray(loaded(np.asarray(obs),key))
        action_key,_=jax.random.split(key);latent=jax.random.normal(action_key,(2,3))
        expected=np.asarray(full.actor(obs,latent))
        maxdiff=float(np.max(np.abs(actual-expected)));assert maxdiff<1e-6
        rows.append(dict(recipe=recipe,base_difference_vs_lean=deviation,actor_perturbation_effect_on_base=0.,
            Q_Adam=8,actor_Adam=8,V_Adam=0,resume4plus4_bitwise_equal=True,
            loss_matches_shared_frozen_CPU_guidance_implementation=True,portable_vs_full_model_max_difference=maxdiff,
            teacher_labels_and_proposals_exactly_match_original=True,fixture_artifact=str(path),fixture_sha256=sha(path)))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(time=time.time(),status='pass',rows=rows,all_alpha_initial_parameters_exact=True,
        scope='SyntheticCPU8update implementation/gradientwiring/resume/portable fixtures only. No RL performance claim or production registration/GPUqueue change.',
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),*core.source_paths()]})
    (args.output_directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
