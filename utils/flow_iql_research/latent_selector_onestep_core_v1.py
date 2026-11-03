"""Joint selected-latent density fitting and coupled one-step Flow decoding.

The original finite-mean lean critic/Flow learner is unchanged. Router NLL
and decoder MSE have disjoint parameters and one common diagonal Adam step.
The deployed actor samples one latent and decodes one action. Neither
branch supplies bootstrap proposals or maximizes an off-support critic.
"""
from pathlib import Path
from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research import latent_selected_actor_v1 as diagonal
from utils.flow_iql_research import orderstat_latent_actor_v1 as order
from utils.flow_iql_research import standalone_latent_selector_actor_v1 as portable
from utils.flow_iql_research.vendor.flax_utils import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


ENVIRONMENTS=base.ENVIRONMENTS
RECIPES=('latent_order9','latent_diag9')
State=base.State
lean=base.lean
make_scan=base.make_scan
make_teacher_policy=base.make_teacher_policy
export_actor=base.export_actor
actor_definition=portable.definition
make_actor_policy=portable.make_policy


def specification(recipe):
    assert recipe in RECIPES
    family='orderstat'if recipe=='latent_order9'else'diagonal'
    spec=base.specification('bank_bc')
    spec.update(name=recipe,input='latent_selector',objective='none',alpha=1.,effective_alpha=1.,
        selector_family=family,components=9 if family=='diagonal'else 10,
        directions=9,best_of=9,teacher_q_aggregation='min',
        distillation_family='selected_latent_likelihood_plus_coupled_behavior_decoder',
        distillation_objective_key='distill_objective',likelihood_normalization='per_latent_dimension',
        decoder_training_distribution='uniformly_selected_Gaussian_proposal',
        student_noise_independent_of_teacher=True)
    if family=='diagonal':spec['std_floor']=.1
    return spec


def config_for(environment,recipe,width=None):
    cfg=base.config_for(environment,'bank_bc',width)
    cfg['onestep_specification']=specification(recipe)
    return cfg


def actor_metadata(cfg,observation_dim,action_dim):
    s=cfg['onestep_specification'];width=int(cfg['actor_hidden_dims'][0])
    m=dict(schema='single_actor_deployment_v1',architecture='latent_selector_then_one_step_decoder',
        selector_family=s['selector_family'],observation_dim=int(observation_dim),action_dim=int(action_dim),
        hidden_dims=[width]*4,layer_norm=False,input='latent_selector',noise_dim=int(action_dim)+1,
        components=s['components'],MLP_blocks=2,dense_layers=10,
        output='one_unbounded_latent_then_one_clipped_decoded_action',
        requires_critic=False,requires_BC_flow=False,flow_integration_steps=0,
        generated_candidate_actions=0,actor_forward_passes=1,realized_latents=1,realized_action_samples=1)
    if s['selector_family']=='orderstat':
        m.update(directions=s['directions'],best_of=s['best_of'],scalar_inverse_CDF_calls=1,
            sampler_projection_tail_guard=10.,ideal_Gaussian_tail_guard_mass=1.5239706048321186e-23)
    else:m['std_floor']=s['std_floor']
    return m


def router_definition(spec,hidden,dimension):
    if spec['selector_family']=='orderstat':return order.OrderStatisticRouter(hidden,dimension,spec['directions'],spec['best_of'])
    return diagonal.MixtureActor(hidden,dimension,spec['components'],spec['std_floor'])


def router_parameters(definition,params,observations,spec,dimension):
    if spec['selector_family']=='orderstat':return definition.apply({'params':params},observations)
    return definition.apply({'params':params},observations,jnp.zeros((*observations.shape[:-1],dimension+1)),return_parameters=True)


def router_log_probability(parameters,latent,spec):
    if spec['selector_family']=='orderstat':return order.log_probability(parameters,latent,spec['best_of'])
    return diagonal.latent_log_probability(parameters,latent)


def router_sample(parameters,noise,spec):
    if spec['selector_family']=='orderstat':return order.sample_latent(parameters,noise,spec['best_of'])
    return diagonal.sample_latent(parameters,noise)


def definitions(state):
    cfg=state.base.agent.config;s=cfg['onestep_specification']
    hidden=(int(cfg['actor_hidden_dims'][0]),)*4;d=cfg['action_dim']
    return router_definition(s,hidden,d),diagonal._Actor(hidden,d,False)


def create(seed,observations,actions,cfg,recipe):
    assert cfg['onestep_specification']['name']==recipe
    assert cfg['num_samples']==9
    learner=lean.create(seed,observations,actions,cfg,'lean')
    metadata=actor_metadata(cfg,observations.shape[-1],actions.shape[-1]);hidden=tuple(metadata['hidden_dims'])
    s=cfg['onestep_specification'];d=actions.shape[-1]
    rdef=router_definition(s,hidden,d);ddef=diagonal._Actor(hidden,d,False)
    rkey=jax.random.fold_in(jax.random.PRNGKey(seed),99502)
    if s['selector_family']=='orderstat':rparams=rdef.init(rkey,observations)['params']
    else:rparams=rdef.init(rkey,observations,jnp.zeros((*observations.shape[:-1],d+1)),return_parameters=True)['params']
    dparams=ddef.init(jax.random.fold_in(jax.random.PRNGKey(seed),99501),observations,jnp.zeros_like(actions))['params']
    actor=TrainState.create(actor_definition(metadata),dict(router=rparams,decoder=dparams),optax.adam(cfg['lr']))
    return State(learner,actor,jnp.asarray(0))


def teacher_batch(state,batch,rng,spec):
    obs,z,target,heads,actions,index=base.teacher_batch(state,batch,rng,spec)
    uniform_index=jax.random.randint(jax.random.fold_in(rng,74326),obs.shape[:-1],0,9)
    noises=dict(selected_latent=jax.lax.stop_gradient(base.gather(z,index)),
        decoder_latent=base.gather(z,uniform_index),
        decoder_target=jax.lax.stop_gradient(base.gather(actions,uniform_index)))
    return obs,noises,target,heads,actions,index


def actor_loss(state,observations,noises,target,proposal_heads,spec,params):
    rdef,ddef=definitions(state);d=target.shape[-1]
    p=router_parameters(rdef,params['router'],observations,spec,d)
    nll=-router_log_probability(p,noises['selected_latent'],spec).mean()/d
    raw=ddef.apply({'params':params['decoder']},observations,noises['decoder_latent'])
    mse=jnp.mean((raw-noises['decoder_target'])**2)
    total=nll+mse
    return total,dict(loss=total,distill_objective=total,distill_MSE=mse,
        router_NLL_per_dimension=nll,decoder_uniform_raw_MSE=mse,Q_loss=jnp.asarray(0.),
        decoder_uniform_raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))


def build_update(recipe,bounds):
    assert recipe in RECIPES
    original_update=lean.build_update('lean',bounds)
    def update(state,batch):
        next_base,logs=original_update(state.base,batch)
        _,rng=jax.random.split(state.base.agent.rng);spec=state.base.agent.config['onestep_specification']
        obs,noises,target,heads,_,_=teacher_batch(state,batch,rng,spec)
        actor,info=state.actor.apply_loss_fn(lambda p:actor_loss(state,obs,noises,target,heads,spec,p))
        logs.update({'deployment/'+k:v for k,v in info.items()})
        return state.replace(base=next_base,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


def diagnostic_actor_actions(state,observations,key):
    d=state.base.agent.config['action_dim']
    noise=jax.random.normal(key,(*observations.shape[:-1],d+1))
    return state.actor(observations,noise)


def make_probe(recipe,bounds):
    previous=lean.make_probe('lean',bounds)
    @jax.jit
    def probe(state,batch):
        obs=batch['observations'];agent=state.base.agent;s=agent.config['onestep_specification'];d=agent.config['action_dim']
        broadcast,z=lean.bank(agent,obs,jax.random.PRNGKey(74191),9)
        bank=lean.flow(agent,broadcast,z,10);heads=agent.network.select('critic')(broadcast,actions=bank)
        qbank=heads.min(0);index=qbank.argmax(0);best=base.gather(bank,index);best_z=base.gather(z,index)
        rdef,ddef=definitions(state)
        p=router_parameters(rdef,state.actor.params['router'],obs,s,d)
        noise=jax.random.normal(jax.random.PRNGKey(74193),(*obs.shape[:-1],d+1))
        sampled_z=router_sample(p,noise,s)
        raw=ddef.apply({'params':state.actor.params['decoder']},obs,sampled_z);action=jnp.clip(raw,-1,1)
        prior_raw=ddef.apply({'params':state.actor.params['decoder']},broadcast,z)
        selected_raw=base.gather(prior_raw,index)
        exact_sample=lean.flow(agent,obs,sampled_z,10)
        qheads=agent.network.select('critic')(obs,actions=action);q=qheads.min(0)
        exact_q=agent.network.select('critic')(obs,actions=exact_sample).min(0);best_q=qbank.max(0)
        prior_logp=-.5*jnp.sum(sampled_z**2,axis=-1)-.5*d*jnp.log(2*jnp.pi)
        log_ratio=router_log_probability(p,sampled_z,s)-prior_logp
        small=obs[:16];repeated=jnp.broadcast_to(small,(8,*small.shape))
        diverse=diagnostic_actor_actions(state,repeated,jax.random.PRNGKey(74192))
        return dict(base=previous(state.base,batch),actor=action,raw=raw,best=best,flow1=bank[0],
            actor_q=q,best_q=best_q,q_disagreement=jnp.abs(qheads[0]-qheads[1]),regret=best_q-q,
            closest_candidate_is_best=((bank-action[None])**2).mean(-1).argmin(0)==index,
            same_state_action_variance=jnp.var(diverse,axis=0).mean(),
            router_NLL_per_dimension=-router_log_probability(p,best_z,s).mean()/d,
            decoder_prior_raw_MSE=jnp.mean((prior_raw-bank)**2),
            decoder_prior_clipped_MSE=jnp.mean((jnp.clip(prior_raw,-1,1)-bank)**2),
            decoder_selected_clipped_MSE=jnp.mean((jnp.clip(selected_raw,-1,1)-best)**2),
            decoder_sampled_clipped_MSE=jnp.mean((action-exact_sample)**2),
            selector_predicted_Q_regret=jnp.mean(best_q-exact_q),
            decoder_predicted_Q_regret=jnp.mean(exact_q-q),
            sampled_log_density_ratio_max=log_ratio.max(),
            sampled_density_ratio_above_bestN_fraction=jnp.mean((log_ratio>jnp.log(9.)+1e-4).astype(jnp.float32)),
            finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))for x in [*jax.tree_util.tree_leaves(state),raw,sampled_z,q,qbank,log_ratio]])))
    return probe


def probe_summary(values,bounds):
    result=base.probe_summary(values,bounds)
    result.update(actor_probe_MSE_is_training_objective=0.,deployment_MLP_blocks=2.)
    for key in ('router_NLL_per_dimension','decoder_prior_raw_MSE','decoder_prior_clipped_MSE',
                'decoder_selected_clipped_MSE','decoder_sampled_clipped_MSE','selector_predicted_Q_regret',
                'decoder_predicted_Q_regret','sampled_log_density_ratio_max','sampled_density_ratio_above_bestN_fraction'):
        result[key]=float(np.asarray(values[key]))
    return result


def source_paths():
    folder=Path(__file__).resolve().parent
    return [Path(__file__).resolve(),Path(portable.__file__).resolve(),Path(diagonal.__file__).resolve(),
            Path(order.__file__).resolve(),folder/'mixture_onestep_actor_v1.py',
            folder/'standalone_onestep_actor_v1.py']+base.source_paths()
