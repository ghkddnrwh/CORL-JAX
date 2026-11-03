"""Gaussian/mixture behavior proposals with the same best-of32 Q extraction.

K1 vs K5 isolates mixture expressivity; comparison with IFQL isolates the flow
proposal. Neither likelihood training nor action selection uses exponential
advantage weights. Q/V learning remains the official IQL/value-control update.
"""
import copy
from utils.flow_iql_research import core as reference
from utils.flow_iql_research.core import ENVIRONMENTS,PROFILES,State,make_scan,probe
from utils.flow_iql_research.policy_core import probe_summary
from utils.flow_iql_research.vendor.iql import IQLAgent
from utils.flow_iql_research.vendor.flax_utils import ModuleDict,TrainState
from utils.flow_iql_research.vendor.networks import MLP,Value,default_init
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


class MixtureActor(nn.Module):
    hidden_dims:tuple
    action_dim:int
    components:int

    @nn.compact
    def __call__(self,observations):
        features=MLP(self.hidden_dims,activate_final=True)(observations)
        logits=nn.Dense(self.components,kernel_init=nn.initializers.zeros)(features)
        means=nn.Dense(self.components*self.action_dim,kernel_init=default_init(.01),name='mean_net')(features)
        logstd=nn.Dense(self.components*self.action_dim,kernel_init=default_init(.01),name='log_std_net')(features)
        shape=(*observations.shape[:-1],self.components,self.action_dim)
        return logits,means.reshape(shape),jnp.clip(logstd.reshape(shape),-5.,2.)


class MixtureIQLAgent(IQLAgent):
    def actor_loss(self,batch,grad_params,rng=None):
        logits,means,logstd=self.network.select('actor')(batch['observations'],params=grad_params)
        log_components=-.5*jnp.sum(((batch['actions'][...,None,:]-means)*jnp.exp(-logstd))**2+2*logstd+jnp.log(2*jnp.pi),axis=-1)
        log_probs=jax.scipy.special.logsumexp(jax.nn.log_softmax(logits)+log_components,axis=-1)
        loss=-log_probs.mean()
        probabilities=jax.nn.softmax(logits)
        entropy=-jnp.sum(probabilities*jax.nn.log_softmax(logits),axis=-1)
        return loss,dict(actor_loss=loss,bc_log_prob=log_probs.mean(),mixture_entropy=entropy.mean(),
                         effective_components=jnp.exp(entropy).mean(),max_component_probability=probabilities.max(-1).mean(),
                         std=jnp.exp(logstd).mean(),std_min=jnp.exp(logstd).min())

    @jax.jit
    def sample_actions(self,observations,seed=None,temperature=1.):
        component_key,noise_key=jax.random.split(seed)
        logits,means,logstd=self.network.select('actor')(observations)
        n=self.config['num_samples'];leading=observations.shape[:-1]
        components=jax.random.categorical(component_key,logits,shape=(n,*leading))
        index=jnp.broadcast_to(components[...,None,None],(n,*leading,1,self.config['action_dim']))
        selected_means=jnp.take_along_axis(jnp.broadcast_to(means,(n,*means.shape)),index,axis=-2)[...,0,:]
        selected_std=jnp.take_along_axis(jnp.broadcast_to(logstd,(n,*logstd.shape)),index,axis=-2)[...,0,:]
        actions=jnp.clip(selected_means+jnp.exp(selected_std)*jax.random.normal(noise_key,selected_means.shape),-1,1)
        obs=jnp.broadcast_to(observations,(n,*observations.shape))
        q=self.network.select('critic')(obs,actions=actions).min(0)
        best=jnp.argmax(q,axis=0)
        return jnp.take_along_axis(actions,jnp.broadcast_to(best[None,...,None],(1,*actions.shape[1:])),axis=0)[0]

    @classmethod
    def create(cls,seed,ex_observations,ex_actions,config):
        config=dict(config);config['action_dim']=ex_actions.shape[-1]
        rng=jax.random.PRNGKey(seed);rng,init_rng=jax.random.split(rng,2)
        value=Value(config['value_hidden_dims'],layer_norm=config['layer_norm'],num_ensembles=1)
        critic=Value(config['value_hidden_dims'],layer_norm=config['layer_norm'],num_ensembles=2)
        actor=MixtureActor(config['actor_hidden_dims'],config['action_dim'],config['mixture_components'])
        info=dict(value=(value,(ex_observations,)),critic=(critic,(ex_observations,ex_actions)),
                  target_critic=(copy.deepcopy(critic),(ex_observations,ex_actions)),actor=(actor,(ex_observations,)))
        definition=ModuleDict({k:v[0] for k,v in info.items()})
        params=flax.core.unfreeze(definition.init(init_rng,**{k:v[1] for k,v in info.items()})['params'])
        params['modules_target_critic']=params['modules_critic']
        network=TrainState.create(definition,params,tx=optax.adam(config['lr']))
        return cls(rng,network=network,config=flax.core.FrozenDict(**config))


def config_for(environment,discount=None,expectile=.9,width=None,alpha=None):
    c=reference.config_for(environment,discount,expectile,width,alpha)
    if 'puzzle-4x4' in environment and alpha is None:c['alpha']=3.
    c.update(num_samples=32,mixture_components=5,policy_type='mixture_bc')
    return c


def create(seed,observations,actions,config,kind='mixture_bc'):
    assert kind in ('gaussian_bc','mixture_bc')
    agent=MixtureIQLAgent.create(seed,observations,actions,dict(config))
    vp=agent.network.params['modules_value']
    return State(agent,optax.adam(config['lr']).init(vp),jnp.asarray(0),jnp.asarray(0))


def build_update(profile_name,bounds,kind='mixture_bc'):
    # Reuse the frozen policy controller, including exhaustive backup metrics.
    from utils.flow_iql_research.policy_core import build_update
    return build_update(profile_name,bounds,kind)
