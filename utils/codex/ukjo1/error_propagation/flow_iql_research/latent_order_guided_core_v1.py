"""Joint finite-mean learner and guided, density-capped latent selector.

Decoder MSE is kept at coefficient one. Selector likelihood is weighted by
one global alpha. Q guidance updates the selector only, through the frozen
one-step decoder; no Flow BPTT, actor bootstrap or independent critic is added.
"""
from pathlib import Path
from utils.flow_iql_research import latent_selector_onestep_core_v1 as parent
from utils.flow_iql_research import latent_order_guidance_v1 as guidance
import jax


ENVIRONMENTS=parent.ENVIRONMENTS
RECIPES=('latent_order9_q_a003','latent_order9_q_a03','latent_order9_q_a3')
State=parent.State
lean=parent.lean
make_scan=parent.make_scan
actor_metadata=parent.actor_metadata
actor_definition=parent.actor_definition
make_actor_policy=parent.make_actor_policy
make_teacher_policy=parent.make_teacher_policy
export_actor=parent.export_actor
create=parent.create
make_probe=parent.make_probe
probe_summary=parent.probe_summary
diagnostic_actor_actions=parent.diagnostic_actor_actions


def specification(recipe):
    assert recipe in RECIPES
    alpha={'latent_order9_q_a003':.03,'latent_order9_q_a03':.3,'latent_order9_q_a3':3.}[recipe]
    s=parent.specification('latent_order9')
    s.update(name=recipe,objective='maximize',selector_alpha=alpha,
        # The imitation objective below already weights router NLL internally;
        # generic gradient reconstruction must not multiply decoder MSE again.
        alpha=1.,effective_alpha=1.,distillation_objective_already_weighted=True,
        q_aggregation='mean',q_scale='within_state_BC_action_spread_floor1',
        guidance_updates_decoder=False,ideal_latent_prior_density_ratio_bound=9.)
    return s


def config_for(environment,recipe,width=None):
    cfg=parent.config_for(environment,'latent_order9',width)
    cfg['onestep_specification']=specification(recipe)
    return cfg


def teacher_batch(state,batch,rng,spec):
    obs,noise,target,heads,actions,index=parent.teacher_batch(state,batch,rng,spec)
    d=target.shape[-1]
    noise=dict(noise,guidance_gaussian=jax.random.normal(jax.random.fold_in(rng,74327),(*obs.shape[:-1],d)))
    return obs,noise,target,heads,actions,index


def actor_loss(state,observations,noises,target,proposal_heads,spec,params):
    _,info=parent.actor_loss(state,observations,noises,target,proposal_heads,spec,params)
    rdef,ddef=parent.definitions(state)
    p=parent.router_parameters(rdef,params['router'],observations,spec,target.shape[-1])
    qloss,qinfo=guidance.expected_Q_loss(p,noises['guidance_gaussian'],observations,ddef,
        params['decoder'],state.base.agent.network.select('critic'),proposal_heads,spec['best_of'])
    imitation=info['decoder_uniform_raw_MSE']+spec['selector_alpha']*info['router_NLL_per_dimension']
    total=imitation+qloss
    info.update(qinfo,loss=total,distill_objective=imitation,
        selector_NLL_coefficient=spec['selector_alpha'])
    return total,info


def build_update(recipe,bounds):
    assert recipe in RECIPES
    original_update=lean.build_update('lean',bounds)
    def update(state,batch):
        next_base,logs=original_update(state.base,batch)
        _,rng=jax.random.split(state.base.agent.rng);spec=state.base.agent.config['onestep_specification']
        obs,noise,target,heads,_,_=teacher_batch(state,batch,rng,spec)
        actor,info=state.actor.apply_loss_fn(lambda p:actor_loss(state,obs,noise,target,heads,spec,p))
        logs.update({'deployment/'+k:v for k,v in info.items()})
        return state.replace(base=next_base,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


def source_paths():
    return [Path(__file__).resolve(),Path(guidance.__file__).resolve()]+parent.source_paths()
