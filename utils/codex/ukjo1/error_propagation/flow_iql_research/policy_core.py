"""Policy extraction families with the same official IQL Q/V objectives.

IFQL: behavior flow plus best-of-N Q selection (official implementation).
AWR flow: fit an advantage-tilted action distribution by weighted flow matching.
Distilled flow: FQL's behavior-flow teacher and differentiable one-step student,
with IQL Q/V backups. The actor objective centers Q by V and normalizes by the
observed data advantage scale, so common state-value offsets do not set the
relative strength of distillation. None of these policies enters the Q backup.
"""
import copy
from utils.flow_iql_research import core as reference
from utils.flow_iql_research.core import ENVIRONMENTS, PROFILES, State, make_scan, probe
from utils.flow_iql_research.vendor.ifql import IFQLAgent
from utils.flow_iql_research.vendor.flax_utils import ModuleDict, TrainState
from utils.flow_iql_research.vendor.networks import ActorVectorField, Value
import flax
import jax
import jax.numpy as jnp
import optax


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    c = reference.config_for(environment,discount,expectile,width,alpha)
    if 'puzzle-4x4' in environment and alpha is None:
        c['alpha'] = 3.
    c.update(num_samples=32,flow_steps=10,policy_type='ifql',
             flow_temperature=3.,distill_coefficient=30.,normalize_advantage=True)
    return c


class FlowIQLAgent(IFQLAgent):
    def compute_flow_actions(self, observations, noises):
        actions = noises
        for i in range(self.config['flow_steps']):
            t=jnp.full((*observations.shape[:-1],1),i/self.config['flow_steps'])
            actions=actions+self.network.select('actor_flow')(observations,actions,t)/self.config['flow_steps']
        return jnp.clip(actions,-1,1)

    def actor_loss(self,batch,grad_params,rng=None):
        cfg=self.config;net=self.network
        batch_size,action_dim=batch['actions'].shape
        rng,x_rng,t_rng,noise_rng=jax.random.split(rng,4)
        x0=jax.random.normal(x_rng,(batch_size,action_dim))
        t=jax.random.uniform(t_rng,(batch_size,1))
        xt=(1-t)*x0+t*batch['actions'];velocity=batch['actions']-x0
        prediction=net.select('actor_flow')(batch['observations'],xt,t,params=grad_params)
        per_example=jnp.mean((prediction-velocity)**2,axis=-1)
        data_qs=net.select('critic')(batch['observations'],actions=batch['actions'])
        qdata=data_qs.min(0)
        v=net.select('value')(batch['observations'])
        advantage=qdata-v
        logs={'data_adv_mean':advantage.mean(),'data_adv_abs_mean':jnp.abs(advantage).mean()}
        if cfg['policy_type']=='awrflow':
            weights=jnp.exp(jnp.minimum(cfg['flow_temperature']*advantage,jnp.log(100.)))
            loss=jnp.mean(weights*per_example)/jnp.maximum(weights.mean(),1e-8)
            logs.update(weight_ess_fraction=weights.mean()**2/jnp.maximum((weights**2).mean(),1e-20),
                        weight_underflow_fraction=jnp.mean((cfg['flow_temperature']*advantage < -87.).astype(jnp.float32)),
                        weight_clip_fraction=jnp.mean((cfg['flow_temperature']*advantage>=jnp.log(100.)).astype(jnp.float32)))
        else:
            loss=per_example.mean()
        logs['bc_flow_loss']=loss
        if cfg['policy_type'] in ('distilled','fql_policy'):
            noises=jax.random.normal(noise_rng,(batch_size,action_dim))
            teacher=self.compute_flow_actions(batch['observations'],noises)
            raw=net.select('actor_onestep_flow')(batch['observations'],noises,params=grad_params)
            distill=jnp.mean((raw-teacher)**2)
            actions=jnp.clip(raw,-1,1)
            # FQL uses the mean of the two heads in the policy objective.
            q=net.select('critic')(batch['observations'],actions=actions).mean(0)
            # One component reward is the common unit in all nine cost tasks.
            normalizer=jax.lax.stop_gradient(jnp.maximum(jnp.abs(advantage).mean(),1.))
            normalize=cfg['normalize_advantage'] and cfg['policy_type']!='fql_policy'
            qloss=-(q-v).mean()/(normalizer if normalize else 1.)
            loss=loss+cfg['distill_coefficient']*distill+qloss
            logs.update(distill_loss=distill,q_loss=qloss,
                        q_loss_normalizer=normalizer if normalize else jnp.asarray(1.),
                        q_loss_normalization_active=jnp.asarray(float(normalize)),
                        student_adv_mean=(q-v).mean(),student_minus_data_q=(q-data_qs.mean(0)).mean(),
                        teacher_displacement_rms=jnp.sqrt(distill),
                        student_saturation_fraction=jnp.mean((jnp.abs(raw)>1).astype(jnp.float32)))
        logs['actor_loss']=loss
        return loss,logs

    @jax.jit
    def sample_actions(self,observations,seed=None,temperature=1.):
        action_seed,_=jax.random.split(seed)
        if self.config['policy_type'] in ('distilled','fql_policy'):
            noises=jax.random.normal(action_seed,(*observations.shape[:-1],self.config['action_dim']))
            return jnp.clip(self.network.select('actor_onestep_flow')(observations,noises),-1,1)
        n=self.config['num_samples']
        # Sample dimension first, so arbitrary leading observation dimensions work.
        noises=jax.random.normal(action_seed,(n,*observations.shape[:-1],self.config['action_dim']))
        ob=jnp.broadcast_to(observations,(n,*observations.shape))
        actions=self.compute_flow_actions(ob,noises)
        if n==1:
            return actions[0]
        q=self.network.select('critic')(ob,actions=actions).min(0)
        best=jnp.argmax(q,axis=0)
        return jnp.take_along_axis(actions,jnp.broadcast_to(best[None,...,None],(1,*actions.shape[1:])),axis=0)[0]

    @classmethod
    def create(cls,seed,ex_observations,ex_actions,config):
        rng=jax.random.PRNGKey(seed);rng,init_rng=jax.random.split(rng,2)
        config=dict(config);config['action_dim']=ex_actions.shape[-1]
        value=Value(config['value_hidden_dims'],layer_norm=config['layer_norm'],num_ensembles=1)
        critic=Value(config['value_hidden_dims'],layer_norm=config['layer_norm'],num_ensembles=2)
        flow=ActorVectorField(config['actor_hidden_dims'],config['action_dim'],layer_norm=config['actor_layer_norm'])
        info=dict(value=(value,(ex_observations,)),critic=(critic,(ex_observations,ex_actions)),
                  target_critic=(copy.deepcopy(critic),(ex_observations,ex_actions)),
                  actor_flow=(flow,(ex_observations,ex_actions,ex_actions[...,:1])))
        if config['policy_type'] in ('distilled','fql_policy'):
            info['actor_onestep_flow']=(copy.deepcopy(flow),(ex_observations,ex_actions))
        definition=ModuleDict({k:v[0] for k,v in info.items()})
        params=flax.core.unfreeze(definition.init(init_rng,**{k:v[1] for k,v in info.items()})['params'])
        params['modules_target_critic']=params['modules_critic']
        network=TrainState.create(definition,params,tx=optax.adam(config['lr']))
        return cls(rng,network=network,config=flax.core.FrozenDict(**config))


def create(seed,observations,actions,config,kind='iql'):
    if kind in ('iql','gaussian'):
        return reference.create(seed,observations,actions,config,'iql')
    config=dict(config)
    if kind=='ifql':
        # Exact official IFQL forward/update are retained as an explicit control.
        return reference.create(seed,observations,actions,config,'ifql')
    config['policy_type']=kind
    agent=FlowIQLAgent.create(seed,observations,actions,config)
    vp=agent.network.params['modules_value']
    return State(agent,optax.adam(config['lr']).init(vp),jnp.asarray(0),jnp.asarray(0))


def build_update(profile_name,bounds,kind='iql'):
    update=reference.build_update(profile_name,bounds,kind)
    profile=PROFILES[profile_name]
    def with_backup_metrics(state,batch):
        new,logs=update(state,batch)
        vnet=(new if profile.get('fresh') else state).agent.network
        vn=vnet.select('value')(batch['next_observations'])
        y=batch['rewards']+state.agent.config['discount']*batch['masks']*vn
        low=jnp.mean((y<bounds[0]).astype(jnp.float32))
        high=jnp.mean((y>bounds[1]).astype(jnp.float32))
        prefix='backup/' if profile.get('bound') else 'backup/counterfactual_'
        return new,{**logs,prefix+'lower_clip_fraction':low,prefix+'upper_clip_fraction':high}
    return with_backup_metrics


def probe_summary(values,expectile,bounds,alpha):
    result=reference.probe_summary(values,expectile,bounds,alpha if alpha is not None else 0.)
    if alpha is None:
        # IFQL and the distilled student do not use exponential actor weights.
        result={k:v for k,v in result.items() if not k.startswith('actor_weight_')}
    return result
