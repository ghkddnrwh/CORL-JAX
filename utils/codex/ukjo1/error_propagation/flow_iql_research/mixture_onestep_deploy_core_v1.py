"""Joint one-step mixture actor over an unchanged finite-mean lean learner.

Best9 online-minQ labels are likelihood-distilled. Optional FQL-style online
meanQ guidance analytically enumerates mixture components during training.
Deployment is one parameter-MLP forward plus one categorical/Gaussian action.
"""
from pathlib import Path

from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research import mixture_guidance_losses_v1 as guidance
from utils.flow_iql_research import standalone_mixture_actor_v1 as portable
from utils.flow_iql_research.vendor.flax_utils import TrainState

import jax
import jax.numpy as jnp
import optax


ENVIRONMENTS=base.ENVIRONMENTS
RECIPES=('mdn9_bc','mdn9_q_a003','mdn9_q_a03','mdn9_q_a3')
State=base.State
lean=base.lean


def specification(recipe):
    assert recipe in RECIPES
    alpha={'mdn9_bc':1.,'mdn9_q_a003':.03,'mdn9_q_a03':.3,'mdn9_q_a3':3.}[recipe]
    spec=base.specification('bank_bc')
    spec.update(name=recipe,input='mixture',objective='none' if recipe=='mdn9_bc' else 'maximize',
        alpha=alpha,effective_alpha=alpha,q_aggregation='mean',q_scale='absQ' if recipe!='mdn9_bc' else 'none',
        distillation_objective_key='distill_objective',distillation_family='clipped_diagonal_mixture_likelihood',
        teacher_q_aggregation='min',mixture_components=9,std_floor=.001,
        likelihood_normalization='per_action_dimension',deployment_mode='sample_one_component_and_one_Gaussian',
        student_noise_independent_of_teacher=True)
    return spec


def config_for(environment,recipe,width=None):
    cfg=base.config_for(environment,'bank_bc',width)
    cfg['onestep_specification']=specification(recipe)
    return cfg


def actor_metadata(cfg,observation_dim,action_dim):
    spec=cfg['onestep_specification'];width=int(cfg['actor_hidden_dims'][0])
    return dict(schema='single_actor_deployment_v1',architecture='clipped_diagonal_mixture_MLP',
        observation_dim=int(observation_dim),action_dim=int(action_dim),noise_dim=int(action_dim)+1,
        hidden_dims=[width]*4,layer_norm=False,input='mixture',components=spec['mixture_components'],std_floor=spec['std_floor'],
        output='sample_one_component_then_one_Gaussian_and_clip_to_minus1_plus1',
        requires_critic=False,requires_BC_flow=False,flow_integration_steps=0,
        generated_candidate_actions=0,actor_forward_passes=1,realized_action_samples=1)


actor_definition=portable.definition
make_actor_policy=portable.make_policy
make_teacher_policy=base.make_teacher_policy
make_scan=base.make_scan
export_actor=base.export_actor


def create(seed,observations,actions,cfg,recipe):
    assert cfg['onestep_specification']['name']==recipe
    learner=lean.create(seed,observations,actions,cfg,'lean')
    metadata=actor_metadata(cfg,observations.shape[-1],actions.shape[-1]);definition=actor_definition(metadata)
    latent=jnp.zeros((*observations.shape[:-1],metadata['noise_dim']))
    parameters=definition.init(jax.random.fold_in(jax.random.PRNGKey(seed),99213),observations,latent)['params']
    actor=TrainState.create(definition,parameters,optax.adam(cfg['lr']))
    return State(learner,actor,jnp.asarray(0))


def teacher_batch(state,batch,rng,spec):
    obs,_,target,heads,actions,index=base.teacher_batch(state,batch,rng,spec)
    # Nine Gaussian draws for the optional expected-Q term, and a separate
    # D+1 draw for sampling diagnostics. NLL integrates studentnoise exactly.
    noises=jax.random.normal(jax.random.fold_in(rng,74324),(10,*obs.shape[:-1],target.shape[-1]+1))
    return obs,noises,target,heads,actions,index


def actor_loss(state,observations,noises,target,proposal_heads,spec,params):
    p=state.actor(observations,jnp.zeros((*observations.shape[:-1],target.shape[-1]+1)),params=params,return_parameters=True)
    total,info=guidance.loss(p,target,observations,noises[:9,...,:target.shape[-1]],
        state.base.agent.network.select('critic'),spec['effective_alpha'],spec['objective']=='maximize')
    raw=portable.raw_action(p,noises[-1]);sample=jnp.clip(raw,-1,1)
    info.update(distill_MSE=jnp.mean((sample-target)**2),distill_MSE_is_training_objective=jnp.asarray(0.),
                raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
    return total,info


def build_update(recipe,bounds):
    assert recipe in RECIPES
    learner_update=lean.build_update('lean',bounds)
    def update(state,batch):
        learner,logs=learner_update(state.base,batch)
        _,rng=jax.random.split(state.base.agent.rng)
        spec=state.base.agent.config['onestep_specification']
        obs,noise,target,heads,_,_=teacher_batch(state,batch,rng,spec)
        actor,info=state.actor.apply_loss_fn(lambda p:actor_loss(state,obs,noise,target,heads,spec,p))
        logs.update({'deployment/'+k:v for k,v in info.items()})
        return state.replace(base=learner,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


def diagnostic_actor_actions(state,observations,key):
    """Hook for generic diagnostics; mixture noise dimension is D+1."""
    dim=state.base.agent.config['action_dim']
    latent=jax.random.normal(key,(*observations.shape[:-1],dim+1))
    p=state.actor(observations,latent,return_parameters=True)
    return jnp.clip(portable.raw_action(p,latent),-1,1)


def make_probe(recipe,bounds):
    previous=lean.make_probe('lean',bounds)
    @jax.jit
    def probe(state,batch):
        obs=batch['observations'];agent=state.base.agent
        broadcast,noise=lean.bank(agent,obs,jax.random.PRNGKey(74191),9)
        bank=lean.flow(agent,broadcast,noise,10)
        heads=agent.network.select('critic')(broadcast,actions=bank);qbank=heads.min(0)
        index=qbank.argmax(0);best=base.gather(bank,index)
        latent=jax.random.normal(jax.random.PRNGKey(74193),(*obs.shape[:-1],agent.config['action_dim']+1))
        p=state.actor(obs,latent,return_parameters=True);raw=portable.raw_action(p,latent);action=jnp.clip(raw,-1,1)
        qheads=agent.network.select('critic')(obs,actions=action);q=qheads.min(0);best_q=qbank.max(0)
        small=obs[:16];repeated=jnp.broadcast_to(small,(8,*small.shape))
        diverse=diagnostic_actor_actions(state,repeated,jax.random.PRNGKey(74192))
        nearest=jnp.mean((bank-action[None])**2,axis=-1).argmin(0)
        weights=jax.nn.softmax(p['logits'],axis=-1)
        entropy=-jnp.sum(weights*jax.nn.log_softmax(p['logits'],axis=-1),axis=-1)
        nll=-mixture.log_probability(p,best).mean()/action.shape[-1]
        return dict(base=previous(state.base,batch),actor=action,raw=raw,best=best,flow1=bank[0],
            actor_q=q,best_q=best_q,q_disagreement=jnp.abs(qheads[0]-qheads[1]),regret=best_q-q,
            closest_candidate_is_best=(nearest==index).astype(jnp.float32),same_state_action_variance=jnp.var(diverse,axis=0).mean(),
            finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in [*jax.tree_util.tree_leaves(state),raw,q,qbank,nll]])),
            mixture_NLL_per_dimension=nll,
            mixture_effective_components=jnp.exp(entropy).mean(),
            mixture_weighted_mean_scale=jnp.mean(jnp.sum(weights[...,None]*p['scales'],axis=1)))
    return probe


def probe_summary(values,bounds):
    result=base.probe_summary(values,bounds)
    result.update(actor_probe_MSE_is_training_objective=0.,mixture_components=9.,
        **{key:float(values[key]) for key in ('mixture_NLL_per_dimension','mixture_effective_components','mixture_weighted_mean_scale')})
    return result


def source_paths():
    return [Path(__file__).resolve(),Path(mixture.__file__).resolve(),Path(guidance.__file__).resolve(),Path(portable.__file__).resolve()]+base.source_paths()
