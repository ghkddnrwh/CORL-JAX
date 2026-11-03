"""Keep IQL's AWR data objective; replace its single Gaussian by a mixture.

Primary deployment takes the mean of the highest-probability component, without
Q selection. Learned component variances are part of this policy model. The
one-component variant is a heteroscedastic Gaussian control, not the public
fixed-standard-deviation baseline.
"""
import copy
from utils.flow_iql_research import mixture_core as mixture
from utils.flow_iql_research import core as reference
from utils.flow_iql_research import policy_core
from utils.flow_iql_research.core import ENVIRONMENTS, State, make_scan, probe
from utils.flow_iql_research.policy_core import probe_summary
from utils.flow_iql_research.vendor.flax_utils import TrainState
import flax
import jax
import jax.numpy as jnp
import optax


PROFILES={name:reference.PROFILES[name]for name in ('official','bounded_v2')}


def mixture_log_prob(logits,means,logstd,actions):
    components=-.5*jnp.sum(((actions[...,None,:]-means)*jnp.exp(-logstd))**2+2*logstd+jnp.log(2*jnp.pi),axis=-1)
    joint=jax.nn.log_softmax(logits)+components
    return jax.scipy.special.logsumexp(joint,axis=-1),joint


def greedy_component_mean(logits,means):
    component=jnp.argmax(logits,axis=-1)
    index=jnp.broadcast_to(component[...,None,None],(*means.shape[:-2],1,means.shape[-1]))
    return jnp.take_along_axis(means,index,axis=-2)[...,0,:]


class AWRMixtureAgent(mixture.MixtureIQLAgent):
    def actor_loss(self,batch,grad_params,rng=None):
        net=self.network
        # Match the public Gaussian AWR's OLD ONLINE Q and OLD V convention.
        values=net.select('value')(batch['observations'])
        q=net.select('critic')(batch['observations'],actions=batch['actions']).min(0)
        advantage=q-values
        log_weight=self.config['alpha']*advantage
        weights=jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_weight,jnp.log(100.))))
        logits,means,logstd=net.select('actor')(batch['observations'],params=grad_params)
        log_prob,joint=mixture_log_prob(logits,means,logstd,batch['actions'])
        # Preserve the public AWR batch mean; no weight normalization or extra
        # entropy, Q-gradient, distillation, or flow objective is introduced.
        loss=-jnp.mean(weights*log_prob)
        probability=jax.nn.softmax(logits)
        entropy=-jnp.sum(probability*jax.nn.log_softmax(logits),axis=-1)
        posterior=jax.nn.softmax(joint)
        posterior_entropy=-jnp.sum(posterior*jax.nn.log_softmax(joint),axis=-1)
        usage=jnp.sum(weights[...,None]*posterior,axis=0)/jnp.maximum(weights.sum(),1e-20)
        usage_entropy=-jnp.sum(jnp.where(usage>0,usage*jnp.log(jnp.maximum(usage,1e-20)),0.))
        center=jnp.sum(probability[...,None]*means,axis=-2)
        spread=jnp.sqrt(jnp.mean(jnp.sum(probability[...,None]*(means-center[...,None,:])**2,axis=-2)))
        deterministic=greedy_component_mean(logits,means)
        return loss,dict(actor_loss=loss,adv=advantage.mean(),bc_log_prob=log_prob.mean(),
            mse=jnp.mean((deterministic-batch['actions'])**2),std=jnp.exp(logstd).mean(),
            AWR_weight_mean=weights.mean(),
            AWR_weight_ESS_fraction=weights.mean()**2/jnp.maximum(jnp.mean(weights**2),1e-20),
            AWR_weight_clip_fraction=jnp.mean((log_weight>=jnp.log(100.)).astype(jnp.float32)),
            AWR_weight_underflow_proxy_fraction=jnp.mean((log_weight < -87.).astype(jnp.float32)),
            AWR_zero_batch_mass=(weights.sum()==0).astype(jnp.float32),
            mixture_entropy=entropy.mean(),effective_components=jnp.exp(entropy).mean(),
            max_component_probability=probability.max(-1).mean(),
            weighted_posterior_entropy=jnp.sum(weights*posterior_entropy)/jnp.maximum(weights.sum(),1e-20),
            weighted_usage_effective_components=jnp.exp(usage_entropy),
            largest_weighted_mean_responsibility=usage.max(),mean_component_spread=spread,
            std_at_min_fraction=jnp.mean((logstd<=-5.).astype(jnp.float32)),
            std_at_max_fraction=jnp.mean((logstd>=2.).astype(jnp.float32)))

    @jax.jit
    def sample_actions(self,observations,seed=None,temperature=1.):
        if seed is None:seed=jax.random.PRNGKey(0)
        component_key,noise_key=jax.random.split(seed)
        logits,means,logstd=self.network.select('actor')(observations)
        sampled=jax.random.categorical(component_key,logits)
        component=jnp.where(jnp.asarray(temperature)>0,sampled,jnp.argmax(logits,axis=-1))
        index=jnp.broadcast_to(component[...,None,None],(*means.shape[:-2],1,means.shape[-1]))
        mean=jnp.take_along_axis(means,index,axis=-2)[...,0,:]
        scale=jnp.exp(jnp.take_along_axis(logstd,index,axis=-2)[...,0,:])
        action=mean+jnp.maximum(jnp.asarray(temperature),0.)*scale*jax.random.normal(noise_key,mean.shape)
        return jnp.clip(action,-1,1)


def config_for(environment,discount=None,expectile=.9,width=None,alpha=None):
    config=mixture.config_for(environment,discount,expectile,width,alpha)
    config.update(agent_name='awr_mixture',policy_type='awr_mixture',num_samples=1,
        mixture_components=5,const_std=False,state_dependent_std=True,
        actor_objective='public_AWR_weighted_data_mixture_log_likelihood',
        deterministic_decoder='mean_of_highest_probability_component')
    return config


def create(seed,observations,actions,config,kind='awr_mixture'):
    assert kind in ('awr_mixture','awr_gaussian')
    cfg=dict(config);cfg.update(action_dim=actions.shape[-1],policy_type=kind,agent_name=kind,
        mixture_components=5 if kind=='awr_mixture'else 1,num_samples=1)
    model=mixture.MixtureIQLAgent.create(seed,observations,actions,cfg)
    native_cfg={**cfg,'const_std':True,'state_dependent_std':False}
    native=reference.create(seed,observations,actions,native_cfg,'iql')
    params=copy.deepcopy(model.network.params)
    for name in ('modules_critic','modules_target_critic','modules_value'):
        params[name]=copy.deepcopy(native.agent.network.params[name])
    network=TrainState.create(model.network.model_def,params,tx=optax.adam(cfg['lr']))
    agent=AWRMixtureAgent(native.agent.rng,network=network,config=flax.core.FrozenDict(**cfg))
    return State(agent,optax.adam(cfg['lr']).init(params['modules_value']),jnp.asarray(0),jnp.asarray(0))


def build_update(profile_name,bounds,kind='awr_mixture'):
    assert profile_name in PROFILES and kind in ('awr_mixture','awr_gaussian')
    return policy_core.build_update(profile_name,bounds,kind)
