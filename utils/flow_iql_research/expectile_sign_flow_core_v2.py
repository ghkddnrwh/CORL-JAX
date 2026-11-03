"""IDQL expectile-sign endpoint weighting in flow matching, with frozen V2.

No exponential temperature, comparator-policy samples, auxiliary actor, or
action gradients through Q. Only the behavioral-flow objective is changed.
At the exact expectile this models the known IDQL implicit policy; there is
no novelty claim for its sign weights (arxiv:2304.10573, Eq9).
"""
from utils.flow_iql_research import value_sampling_core_v1 as reference
from utils.flow_iql_research.value_sampling_core_v1 import ENVIRONMENTS, make_scan, probe, probe_summary
from utils.flow_iql_research.vendor.ifql import IFQLAgent
import jax
import jax.numpy as jnp


PROFILES = {'bounded_v2_expectile_sign_flow': dict(V_fits=2, actor_weighting='expectile_sign')}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    cfg=reference.config_for(environment,discount,expectile,width,alpha)
    cfg.update(actor_weighting='expectile_sign_divided_by_tau',
        actor_weight_threshold='old_target_min_Q_minus_old_V',
        actor_weight_gradient='stopped', actor_weight_normalization='fixed_tau_no_batch_ratio')
    return cfg


def sign_weights(advantage, expectile):
    return jax.lax.stop_gradient(jnp.where(advantage>=0,1.,(1-expectile)/expectile))


class ExpectileFlowAgent(IFQLAgent):
    def actor_loss(self,batch,grad_params,rng=None):
        net=self.network; cfg=self.config
        _,xkey,tkey=jax.random.split(rng,3)
        x0=jax.random.normal(xkey,batch['actions'].shape)
        t=jax.random.uniform(tkey,(batch['actions'].shape[0],1))
        xt=(1-t)*x0+t*batch['actions']
        pred=net.select('actor_flow')(batch['observations'],xt,t,params=grad_params)
        per_example=jnp.mean((pred-(batch['actions']-x0))**2,axis=-1)
        q=net.select('target_critic')(batch['observations'],actions=batch['actions']).min(0)
        v=net.select('value')(batch['observations'])
        margin=jax.lax.stop_gradient(q-v)
        weight=sign_weights(margin,cfg['expectile'])
        loss=jnp.mean(weight*per_example); mass=weight.mean()
        terminal=1-batch['masks']
        return loss,dict(actor_loss=loss,unweighted_flow_loss=per_example.mean(),
            sign_weight_mean=mass,sign_weight_min=weight.min(),sign_weight_max=weight.max(),
            sign_weight_ESS_fraction=mass**2/jnp.mean(weight**2),
            sign_positive_fraction=jnp.mean((margin>=0).astype(jnp.float32)),
            sign_margin_abs_mean=jnp.mean(jnp.abs(margin)),
            sign_margin_within_point01_fraction=jnp.mean((jnp.abs(margin)<.01).astype(jnp.float32)),
            sign_weighted_data_margin_mean=jnp.sum(weight*margin)/jnp.sum(weight),
            sign_terminal_data_fraction=jnp.mean(terminal),
            sign_weighted_terminal_fraction=jnp.sum(weight*terminal)/jnp.sum(weight),
            sign_sample_score_mean=jnp.mean(weight*cfg['expectile']*margin))


def create(seed,observations,actions,config,kind='ifql'):
    assert kind=='ifql'
    base=reference.create(seed,observations,actions,config,kind)
    return base.replace(agent=ExpectileFlowAgent(base.agent.rng,network=base.agent.network,config=base.agent.config))


def build_update(profile_name,bounds,kind='ifql'):
    assert profile_name=='bounded_v2_expectile_sign_flow' and kind=='ifql'
    return reference.build_update('bounded_v2_repeated_batches',bounds,kind)
