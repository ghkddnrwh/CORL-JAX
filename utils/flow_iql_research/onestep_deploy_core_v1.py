"""Actor-only deployment over an unchanged finite-mean lean learner.

The actor has its own optimizer and never supplies bootstrap proposals. Best9
targets reuse the next-state proposal bank; actor labels use ONLINE min-Q,
whereas the finite-mean backup still uses TARGET mean-Q. No deployment critic,
flow integration, candidate actions, or reranking is part of the exported actor.
"""
from pathlib import Path
from utils.flow_iql_research import efficient_mean_core_v1 as lean
from utils.flow_iql_research import benchmark_core
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.vendor.networks import ActorVectorField
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


ENVIRONMENTS=lean.ENVIRONMENTS
INITIAL_RECIPES=(
    'bank_bc','bank_q_a1','fqln_a03','fqln_a1',
    'bank_q_a03','bank_q_a3','bank_value_a1','bank_regret_a1',
    'fqln_a003','fqln_a01','fqln_a3','fqln_a10',
    'fqlraw_x10','fqlraw_x100','state_bc','state_q_a1',
)
ALPHAS={'0003':.003,'001':.01,'003':.03,'01':.1,'03':.3,
        '1':1.,'3':3.,'10':10.,'30':30.,'100':100.,'300':300.,'1000':1000.}


def specification(name):
    spec=dict(name=name,target='best9',input='bank',objective='none',alpha=1.,
        teacher_states='next_observations',q_aggregation='min',q_scale='none',
        actor_width_multiplier=1,actor_depth=4,actor_layer_norm=False,actor_lr_multiplier=1.)
    # Dynamic names allow later preregistered scalar sweeps without source edits.
    suffixes=name.split('__');stem=suffixes[0]
    if stem.startswith('fqln_a'):
        spec.update(target='flow1',input='single',objective='maximize',alpha=ALPHAS[stem[6:]],
            teacher_states='observations',q_aggregation='mean',q_scale='absQ')
    elif stem.startswith('fqlraw_x'):
        spec.update(target='flow1',input='single',objective='maximize',alpha=float(stem[8:]),
            teacher_states='observations',q_aggregation='mean',q_scale='none',public_alpha_multiplier=True)
    else:
        if stem.startswith('state_'):spec['input']='state';stem=stem[6:]
        elif stem.startswith('bank_'):stem=stem[5:]
        else:raise ValueError(name)
        if stem=='bc':pass
        elif '_a' in stem:
            objective,alpha=stem.rsplit('_a',1)
            spec.update(objective={'q':'maximize','value':'value_match','regret':'regret'}[objective],
                alpha=ALPHAS[alpha],q_scale='absQ' if objective=='q' else 'spread')
        else:raise ValueError(name)
    for suffix in suffixes[1:]:
        if suffix.startswith('w'):spec['actor_width_multiplier']=int(suffix[1:])
        elif suffix.startswith('d'):spec['actor_depth']=int(suffix[1:])
        elif suffix=='ln':spec['actor_layer_norm']=True
        elif suffix=='current':spec['teacher_states']='observations'
        elif suffix=='meanQ':spec['q_aggregation']='mean'
        elif suffix=='spreadQ':spec['q_scale']='spread'
        elif suffix.startswith('lr'):spec['actor_lr_multiplier']=float(suffix[2:])
        else:raise ValueError(suffix)
    assert spec['actor_width_multiplier'] in (1,2,3) and 2<=spec['actor_depth']<=8
    return spec


def config_for(environment,recipe,width=None):
    cfg=lean.config_for(environment,'lean',width)
    spec=specification(recipe)
    if spec.get('public_alpha_multiplier'):
        spec['effective_alpha']=spec['alpha']*benchmark_core.config_for(environment,.999,kind='fql')['alpha']
    else:spec['effective_alpha']=spec['alpha']
    cfg['onestep_specification']=spec
    return cfg


@flax.struct.dataclass
class State:
    base: object
    actor: object
    actor_updates: object


def actor_metadata(cfg,observation_dim,action_dim):
    spec=cfg['onestep_specification'];width=int(cfg['actor_hidden_dims'][0])*spec['actor_width_multiplier']
    return dict(schema='single_actor_deployment_v1',observation_dim=int(observation_dim),action_dim=int(action_dim),
        hidden_dims=[width]*spec['actor_depth'],layer_norm=spec['actor_layer_norm'],
        input=spec['input'],noise_dim=action_dim*(9 if spec['input']=='bank' else 1),
        output='clip_single_actor_action_to_minus1_plus1',
        requires_critic=False,requires_BC_flow=False,flow_integration_steps=0,generated_candidate_actions=0,actor_forward_passes=1)


def actor_definition(metadata):
    return ActorVectorField(hidden_dims=tuple(metadata['hidden_dims']),action_dim=metadata['action_dim'],layer_norm=metadata['layer_norm'])


def create(seed,observations,actions,cfg,recipe):
    assert cfg['onestep_specification']['name']==recipe
    base=lean.create(seed,observations,actions,cfg,'lean')
    metadata=actor_metadata(cfg,observations.shape[-1],actions.shape[-1])
    definition=actor_definition(metadata)
    # Same initialization for every alpha with the same architecture/input.
    tag=99211+(9 if metadata['input']=='bank' else 0 if metadata['input']=='state' else 1)
    key=jax.random.fold_in(jax.random.PRNGKey(seed),tag)
    params=definition.init(key,observations,jnp.zeros((*observations.shape[:-1],metadata['noise_dim'])))['params']
    actor=TrainState.create(definition,params,optax.adam(cfg['lr']*cfg['onestep_specification']['actor_lr_multiplier']))
    return State(base,actor,jnp.asarray(0))


def packed_input(noises,spec):
    if spec['input']=='bank':
        return jnp.moveaxis(noises,0,-2).reshape((*noises.shape[1:-1],9*noises.shape[-1]))
    if spec['input']=='state':return jnp.zeros_like(noises[0])
    return noises[0]


def gather(actions,index):
    indices=jnp.broadcast_to(index[None,...,None],(1,*actions.shape[1:]))
    return jnp.take_along_axis(actions,indices,axis=0)[0]


def teacher_batch(state,batch,rng,spec):
    agent=state.base.agent;observations=batch[spec['teacher_states']]
    if spec['teacher_states']=='next_observations':
        key=jax.random.fold_in(rng,61723)
    else:key=jax.random.fold_in(rng,99217)
    count=9 if spec['target']=='best9' or spec['input']=='bank' else 1
    ob,noises=lean.bank(agent,observations,key,count)
    actions=lean.flow(agent,ob,noises,10)
    q_heads=agent.network.select('critic')(ob,actions=actions)
    min_q=q_heads.min(0)
    best_index=jnp.argmax(min_q,axis=0)
    target=gather(actions,best_index) if spec['target']=='best9' else actions[0]
    return observations,noises,jax.lax.stop_gradient(target),q_heads,actions,best_index


def actor_loss(state,observations,noises,target,proposal_heads,spec,params):
    raw=state.actor(observations,packed_input(noises,spec),params=params)
    actions=jnp.clip(raw,-1,1)
    mse=jnp.mean((raw-target)**2)
    heads=state.base.agent.network.select('critic')(observations,actions=actions)
    aggregate=lambda q:q.min(0) if spec['q_aggregation']=='min' else q.mean(0)
    q=aggregate(heads)
    target_q=aggregate(state.base.agent.network.select('critic')(observations,actions=target))
    if spec['q_scale']=='absQ':scale=jax.lax.stop_gradient(jnp.maximum(jnp.mean(jnp.abs(q)),1e-6))
    elif spec['q_scale']=='spread':scale=jax.lax.stop_gradient(jnp.maximum(jnp.std(aggregate(proposal_heads),axis=0),1.))
    else:scale=jnp.asarray(1.)
    objective=spec['objective']
    if objective=='none':q_loss=jnp.asarray(0.)
    elif objective=='maximize':q_loss=-jnp.mean(q/scale)
    elif objective=='value_match':q_loss=jnp.mean(optax.huber_loss((q-jax.lax.stop_gradient(target_q))/scale,delta=1.))
    elif objective=='regret':q_loss=jnp.mean(jax.nn.relu((jax.lax.stop_gradient(target_q)-q)/scale))
    else:raise ValueError(objective)
    total=spec['effective_alpha']*mse+q_loss
    return total,dict(loss=total,distill_MSE=mse,Q_loss=q_loss,actor_Q=q.mean(),teacher_Q=target_q.mean(),
        predicted_Q_regret=(target_q-q).mean(),Q_normalizer=jnp.mean(scale),
        raw_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))


def build_update(recipe,bounds):
    lean_update=lean.build_update('lean',bounds)
    def update(state,batch):
        # Nothing in this call depends on actor parameters or optimizer state.
        base,logs=lean_update(state.base,batch)
        _,rng=jax.random.split(state.base.agent.rng)
        spec=state.base.agent.config['onestep_specification']
        observations,noises,target,heads,_,_=teacher_batch(state,batch,rng,spec)
        actor,info=state.actor.apply_loss_fn(lambda p:actor_loss(state,observations,noises,target,heads,spec,p))
        logs.update({'deployment/'+k:v for k,v in info.items()})
        return state.replace(base=base,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


make_scan=lean.make_scan


def make_actor_policy(metadata,backend='cpu'):
    assert metadata['actor_forward_passes']==1 and metadata['flow_integration_steps']==0
    assert metadata['requires_critic'] is False and metadata['requires_BC_flow'] is False
    definition=actor_definition(metadata)
    def policy(params,observations,key):
        count=9 if metadata['input']=='bank' else 1
        action_key,_=jax.random.split(key)
        noises=jax.random.normal(action_key,(count,*observations.shape[:-1],metadata['action_dim']))
        inputs=packed_input(noises,metadata)
        return jnp.clip(definition.apply({'params':params},observations,inputs),-1,1)
    return jax.jit(policy,backend=backend)


def make_teacher_policy(state,backend='cpu'):
    return lean.make_policy(state.base,'lean','primary',backend)


def export_actor(state,metadata):
    return dict(schema='single_actor_deployment_v1',metadata=metadata,
        actor_params=flax.serialization.to_state_dict(jax.device_get(state.actor.params)),
        actor_updates=int(state.actor_updates),Q_updates=int(state.base.q_updates))


def make_probe(recipe,bounds):
    base_probe=lean.make_probe('lean',bounds)
    @jax.jit
    def probe(state,batch):
        base_values=base_probe(state.base,batch)
        agent=state.base.agent;spec=agent.config['onestep_specification']
        obs=batch['observations'];ob,noises=lean.bank(agent,obs,jax.random.PRNGKey(74191),9)
        bank=lean.flow(agent,ob,noises,10)
        bank_heads=agent.network.select('critic')(ob,actions=bank);bank_q=bank_heads.min(0)
        index=jnp.argmax(bank_q,axis=0);best=gather(bank,index)
        raw=state.actor(obs,packed_input(noises,spec));action=jnp.clip(raw,-1,1)
        heads=agent.network.select('critic')(obs,actions=action)
        q=heads.min(0);best_q=bank_q.max(0)
        nearest=((bank-action[None])**2).mean(-1).argmin(0)
        # Same-state diversity, independent from across-state action variation.
        small=obs[:16]
        diversity_noise=jax.random.normal(jax.random.PRNGKey(74192),(9,8,small.shape[0],agent.config['action_dim']))
        repeated=jnp.broadcast_to(small,(8,*small.shape))
        repeated_action=jnp.clip(state.actor(repeated,packed_input(diversity_noise,spec)),-1,1)
        return dict(base=base_values,actor=action,raw=raw,best=best,flow1=bank[0],actor_q=q,best_q=best_q,
            q_disagreement=jnp.abs(heads[0]-heads[1]),regret=best_q-q,
            closest_candidate_is_best=(nearest==index).astype(jnp.float32),
            same_state_action_variance=jnp.var(repeated_action,axis=0).mean(),
            finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v))for v in jax.tree_util.tree_leaves(state)])))
    return probe


def probe_summary(values,bounds):
    result=lean.probe_summary(values['base'],bounds)
    actor,best,raw=(np.asarray(values[k])for k in ('actor','best','raw'))
    regret=np.asarray(values['regret']);rms=lambda x:float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
    result.update(finite=bool(np.asarray(values['finite'])),actor_to_best9_MSE=float(np.mean((actor-best)**2)),
        actor_to_flow1_MSE=float(np.mean((actor-np.asarray(values['flow1']))**2)),
        actor_Q_min_mean=float(np.asarray(values['actor_q']).mean()),teacher_best9_Q_min_mean=float(np.asarray(values['best_q']).mean()),
        actor_Q_disagreement_mean=float(np.asarray(values['q_disagreement']).mean()),
        predicted_Q_regret_mean=float(regret.mean()),predicted_Q_regret_RMS=rms(regret),predicted_Q_regret_p95=float(np.quantile(regret,.95)),
        raw_actor_saturation_fraction=float(np.mean(np.abs(raw)>1)),
        closest_candidate_is_best_fraction=float(np.asarray(values['closest_candidate_is_best']).mean()),
        same_state_action_variance=float(np.asarray(values['same_state_action_variance'])))
    return result


def source_paths():
    return [Path(__file__).resolve()]+lean.source_paths()
