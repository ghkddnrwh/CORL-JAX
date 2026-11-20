"""Synthetic joint wiring/resume/independent-Adam/portable readiness checks."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import latent_selector_onestep_core_v1 as core
from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research import standalone_latent_selector_actor_v1 as portable
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    a.output.mkdir(parents=True)
    obs=jnp.array([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32)
    action=jnp.array([[.2,-.1],[-.4,.3]],jnp.float32)
    batch=dict(observations=obs,actions=action,next_observations=.9*obs,rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    rows=[];decoder_initial=None
    for recipe in core.RECIPES:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8)
        state=core.create(6,obs,action,cfg,recipe);spec=cfg['onestep_specification']
        if decoder_initial is None:decoder_initial=flax.serialization.to_state_dict(state.actor.params['decoder'])
        else:assert difference(decoder_initial,flax.serialization.to_state_dict(state.actor.params['decoder']))==0
        rng=jax.random.PRNGKey(29001)
        x,noise,target,heads,bank,index=core.teacher_batch(state,batch,rng,spec)
        original=base.teacher_batch(state,batch,rng,spec)
        assert np.array_equal(target,original[2])and np.array_equal(bank,original[4])
        assert np.array_equal(noise['selected_latent'],base.gather(original[1],index))
        assert np.array_equal(index,heads.min(0).argmax(0))
        actor,_=state.actor.apply_loss_fn(lambda p:core.actor_loss(state,x,noise,target,heads,spec,p))
        rdef,ddef=core.definitions(state)
        rstate=TrainState.create(rdef,state.actor.params['router'],optax.adam(cfg['lr']))
        dstate=TrainState.create(ddef,state.actor.params['decoder'],optax.adam(cfg['lr']))
        def rl(p):
            pars=dict(router=p,decoder=state.actor.params['decoder'])
            info=core.actor_loss(state,x,noise,target,heads,spec,pars)[1]
            v=info['router_NLL_per_dimension'];return v,dict(loss=v)
        def dl(p):
            pars=dict(router=state.actor.params['router'],decoder=p)
            info=core.actor_loss(state,x,noise,target,heads,spec,pars)[1]
            v=info['decoder_uniform_raw_MSE'];return v,dict(loss=v)
        rstate,_=rstate.apply_loss_fn(rl);dstate,_=dstate.apply_loss_fn(dl)
        independent=difference(flax.serialization.to_state_dict(actor.params),
            flax.serialization.to_state_dict(dict(router=rstate.params,decoder=dstate.params)))
        assert independent<1e-7
        update=jax.jit(core.build_update(recipe,(-1000.,0.)))
        one,_=update(state,batch)
        ref=jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        base_difference=difference(flax.serialization.to_state_dict(one.base),flax.serialization.to_state_dict(ref))
        assert base_difference<1e-5
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
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(full))
        probe=core.make_probe(recipe,(-1000.,0.));values=jax.device_get(probe(full,batch))
        summary=core.probe_summary(values,(-1000.,0.));assert summary['finite']
        assert summary['actor_probe_MSE_is_training_objective']==0
        decomp=abs(summary['predicted_Q_regret_mean']-
            summary['selector_predicted_Q_regret']-summary['decoder_predicted_Q_regret'])
        assert decomp<1e-5
        if recipe=='latent_order9':assert summary['sampled_density_ratio_above_bestN_fraction']==0
        metadata=core.actor_metadata(cfg,3,2)
        artifact=core.export_actor(full,metadata);path=a.output/(recipe+'_implementation_actor8.pkl')
        with path.open('xb')as f:pickle.dump(artifact,f,pickle.HIGHEST_PROTOCOL)
        loaded=portable.load_actor(path)
        key=jax.random.PRNGKey(42);actual=np.asarray(loaded(np.asarray(obs),key))
        latent=jax.random.normal(jax.random.split(key)[0],(2,3))
        expected=np.asarray(full.actor(obs,latent))
        error=float(np.max(np.abs(actual-expected)));assert error<1e-6
        rows.append(dict(recipe=recipe,actual_Q_Adam=8,actual_actor_Adam=8,V_Adam=0,
            combined_vs_independent_Adam_parameter_difference=independent,
            base_difference_vs_lean=base_difference,actor_perturbation_effect_on_base=0.,
            teacher_target_and_bank_exact=True,resume4plus4_bitwise_equal=True,
            portable_model_max_difference=error,predicted_Q_regret_decomposition_error=decomp,
            fixture_actor=str(path),fixture_sha256=sha(path)))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(time=time.time(),status='pass',rows=rows,decoder_initialization_same_between_families=True,
        source_sha256={str(p.resolve()):sha(p)for p in [Path(__file__),*core.source_paths()]},
        no_performance_or_registration_claim=True,
        scope='SyntheticCPU8step joint wiring; real task shapes and runner/evaluator integration still required before registration.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
