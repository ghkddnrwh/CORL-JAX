"""CPU joint-update and gradient-isolation checks for guided latent actors."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import latent_order_guided_core_v1 as core
from utils.flow_iql_research import latent_selector_onestep_core_v1 as parent
from utils.flow_iql_research import standalone_latent_selector_actor_v1 as portable
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(tree))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and all(d.platform=='cpu' for d in jax.devices())
    a.output.mkdir(parents=True)
    obs=jnp.array([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32)
    actions=jnp.array([[.2,-.1],[-.4,.3]],jnp.float32)
    batch=dict(observations=obs,actions=actions,next_observations=.9*obs,
               rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    sd=flax.serialization.to_state_dict
    plain_cfg=parent.config_for(parent.ENVIRONMENTS[0],'latent_order9',width=8)
    plain=parent.create(6,obs,actions,plain_cfg,'latent_order9')
    plain_update=jax.jit(parent.build_update('latent_order9',(-1000.,0.)))
    plain_final=plain
    for _ in range(8):plain_final,_=plain_update(plain_final,batch)
    rows=[]
    for recipe in core.RECIPES:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8);spec=cfg['onestep_specification']
        state=core.create(6,obs,actions,cfg,recipe)
        assert difference(sd(state.actor.params),sd(plain.actor.params))==0
        x,noise,target,heads,bank,index=core.teacher_batch(state,batch,jax.random.PRNGKey(29001),spec)
        original=parent.teacher_batch(plain,batch,jax.random.PRNGKey(29001),plain_cfg['onestep_specification'])
        assert np.array_equal(target,original[2]) and np.array_equal(bank,original[4])
        for k in original[1]:assert np.array_equal(noise[k],original[1][k])
        def losses(params):return core.actor_loss(state,x,noise,target,heads,spec,params)[1]
        total_grad=jax.grad(lambda p:losses(p)['loss'])(state.actor.params)
        imitation_grad=jax.grad(lambda p:losses(p)['distill_objective'])(state.actor.params)
        qgrad=jax.grad(lambda p:losses(p)['Q_loss'])(state.actor.params)
        assert norm(qgrad['decoder'])==0 and norm(qgrad['router'])>0
        reconstruction=jax.tree_util.tree_map(lambda t,i,q:t-i-q,total_grad,imitation_grad,qgrad)
        assert norm(reconstruction)<1e-5
        raw_grad=jax.grad(lambda p:parent.actor_loss(state,x,noise,target,heads,spec,p)[1]['decoder_uniform_raw_MSE'])(state.actor.params)
        assert difference(sd(raw_grad['decoder']),sd(total_grad['decoder']))<1e-7
        rdef,ddef=parent.definitions(state)
        parameters=parent.router_parameters(rdef,state.actor.params['router'],x,spec,2)
        latents=core.guidance.component_latents(parameters,noise['guidance_gaussian'],9)
        assert latents.shape==(10,2,2) and np.array_equal(latents[0],noise['guidance_gaussian'])
        repeated=jnp.broadcast_to(x,(10,*x.shape))
        values=state.base.agent.network.select('critic')(repeated,actions=jnp.clip(ddef.apply(
            {'params':state.actor.params['decoder']},repeated,latents),-1,1)).mean(0)
        weights=jnp.moveaxis(jax.nn.softmax(parameters['logits'],-1),-1,0)
        info=losses(state.actor.params)
        assert np.allclose(info['actor_expected_Q'],jnp.mean(jnp.sum(weights*values,0)),atol=1e-7)
        assert np.isclose(float(info['loss']),float(info['decoder_uniform_raw_MSE']+
            spec['selector_alpha']*info['router_NLL_per_dimension']+info['Q_loss']))
        update=jax.jit(core.build_update(recipe,(-1000.,0.)))
        one,_=update(state,batch)
        ref=jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        assert difference(sd(one.base),sd(ref))<1e-5
        perturbed=state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda p:p+.25,state.actor.params)))
        other,_=update(perturbed,batch)
        assert difference(sd(one.base),sd(other.base))==0
        full=state
        for _ in range(8):full,logs=update(full,batch)
        resumed=state
        for _ in range(4):resumed,_=update(resumed,batch)
        resumed=flax.serialization.from_state_dict(state,sd(resumed))
        for _ in range(4):resumed,_=update(resumed,batch)
        assert difference(sd(full),sd(resumed))==0
        decoder_diff=difference(sd(full.actor.params['decoder']),sd(plain_final.actor.params['decoder']))
        assert decoder_diff<1e-7
        assert int(full.base.q_updates)==int(full.base.agent.network.opt_state[0].count)==8
        assert int(full.actor_updates)==int(full.actor.opt_state[0].count)==8
        assert int(full.base.v_updates)==int(full.base.v_opt[0].count)==0
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(full))
        summary=core.probe_summary(jax.device_get(core.make_probe(recipe,(-1000.,0.))(full,batch)),(-1000.,0.))
        assert summary['finite'] and summary['sampled_density_ratio_above_bestN_fraction']==0
        metadata=core.actor_metadata(cfg,3,2);artifact=core.export_actor(full,metadata)
        path=a.output/(recipe+'_implementation_actor8.pkl')
        with path.open('xb')as f:pickle.dump(artifact,f,pickle.HIGHEST_PROTOCOL)
        loaded=portable.load_actor(path);key=jax.random.PRNGKey(42)
        expected=np.asarray(full.actor(obs,jax.random.normal(jax.random.split(key)[0],(2,3))))
        delta=float(np.max(np.abs(np.asarray(loaded(np.asarray(obs),key))-expected)));assert delta<1e-6
        rows.append(dict(recipe=recipe,selector_alpha=spec['selector_alpha'],actual_Q_Adam=8,
            actual_actor_Adam=8,V_Adam=0,Q_gradient_decoder_norm=norm(qgrad['decoder']),
            Q_gradient_router_norm=norm(qgrad['router']),gradient_reconstruction_error=norm(reconstruction),
            decoder_parameters_vs_unguided_difference=decoder_diff,base_actor_independent=True,
            resume4plus4_bitwise=True,portable_difference=delta,training_only_mixture_components=10,
            deployment_MLP_blocks=2,deployment_realized_action_samples=1,
            fixture_actor=str(path),fixture_sha256=sha(path)))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(time=time.time(),status='pass',rows=rows,no_performance_claim=True,
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),*core.source_paths()]},
        scope='Synthetic CPU8update gradient and joint wiring; not native1M efficacy.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
