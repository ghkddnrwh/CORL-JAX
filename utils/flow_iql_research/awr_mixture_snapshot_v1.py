"""Heldout mixture diagnostics, without treating likelihood as policy return."""
from utils.flow_iql_research import awr_mixture_core_v1 as core
import jax
import jax.numpy as jnp
import numpy as np


def make_inspector(agent):
    @jax.jit
    def inspect(observations,actions):
        net=agent.network
        logits,means,logstd=net.select('actor')(observations)
        log_prob,joint=core.mixture_log_prob(logits,means,logstd,actions)
        probability=jax.nn.softmax(logits)
        posterior=jax.nn.softmax(joint)
        value=net.select('value')(observations)
        q=net.select('critic')(observations,actions=actions).min(0)
        advantage=q-value
        log_weight=agent.config['alpha']*advantage
        weights=jnp.exp(jnp.minimum(log_weight,jnp.log(100.)))
        greedy=jnp.clip(core.greedy_component_mean(logits,means),-1,1)
        raw_average=jnp.sum(probability[...,None]*means,axis=-2)
        average=jnp.clip(raw_average,-1,1)
        greedy_q=net.select('critic')(observations,actions=greedy).min(0)
        average_q=net.select('critic')(observations,actions=average).min(0)
        spread=jnp.sum(probability[...,None]*(means-raw_average[...,None,:])**2,axis=-2)
        within=jnp.sum(probability[...,None]*jnp.exp(2*logstd),axis=-2)
        return dict(log_prob=log_prob,probabilities=probability,posteriors=posterior,
            log_std=logstd,advantage=advantage,log_weight=log_weight,weights=weights,
            selected_components=jnp.argmax(logits,axis=-1),component_mean=greedy,mixture_mean=average,
            greedy_Q=greedy_q,mixture_mean_Q=average_q,data_actions=actions,
            between_component_variance=spread,within_component_variance=within)
    return inspect


def summarize(found):
    data={key:np.asarray(value)for key,value in jax.device_get(found).items()}
    assert all(np.isfinite(value).all()for value in data.values())
    weights=data['weights'].astype(np.float64);mass=float(weights.sum())
    probability=data['probabilities'];posterior=data['posteriors'];components=probability.shape[-1]
    entropy=lambda p:-np.sum(p*np.log(np.maximum(p,1e-30)),axis=-1)
    usage=(weights[:,None]*posterior).sum(0)/max(mass,1e-30)
    mse=lambda actions:float(np.mean((actions-data['data_actions'])**2))
    return dict(rows=len(weights),components=components,
        unweighted_negative_log_likelihood=float(-data['log_prob'].mean()),
        raw_AWR_weighted_negative_log_likelihood=float(np.mean(-weights*data['log_prob'])),
        normalized_AWR_weighted_negative_log_likelihood_auxiliary_only=float(np.sum(-weights*data['log_prob'])/mass)if mass>0 else None,
        AWR_weight_mean=float(weights.mean()),AWR_weight_ESS_fraction=float(weights.mean()**2/max(np.mean(weights**2),1e-30)),
        AWR_zero_weight_fraction=float(np.mean(weights==0)),AWR_weight_clip_fraction=float(np.mean(data['log_weight']>=np.log(100.))),
        AWR_log_weight_quantiles=np.quantile(data['log_weight'],[0,.1,.5,.9,1]).tolist(),
        posterior_weighted_usage=usage.tolist(),highest_probability_component_frequencies=(np.bincount(data['selected_components'],minlength=components)/len(weights)).tolist(),
        mean_mixture_entropy=float(entropy(probability).mean()),
        mean_posterior_entropy=float(entropy(posterior).mean()),
        greedy_component_mean_data_MSE=mse(data['component_mean']),mixture_conditional_mean_data_MSE=mse(data['mixture_mean']),
        model_Q_greedy_minus_mixture_mean=float(np.mean(data['greedy_Q']-data['mixture_mean_Q'])),
        greedy_action_saturation_fraction=float(np.mean(np.abs(data['component_mean'])>=.98)),
        greedy_vs_mixture_mean_action_RMS=float(np.sqrt(np.mean((data['component_mean']-data['mixture_mean'])**2))),
        between_component_RMS=float(np.sqrt(data['between_component_variance'].mean())),
        within_component_std_RMS=float(np.sqrt(data['within_component_variance'].mean())),
        std_min=float(np.exp(data['log_std']).min()),std_max=float(np.exp(data['log_std']).max()),
        std_at_lower_bound_fraction=float(np.mean(data['log_std']<=-5.)),
        std_at_upper_bound_fraction=float(np.mean(data['log_std']>=2.)),
        limits=['AWR-weightnormalizationisauxiliaryreportingonly;trainingusesrawbatchmean.',
                'Thehighest-probabilitycomponentmeanisnotclaimedtobetheglobaldensitymode.',
                'MSE,likelihood,componentusageandmodelQdiagnosedifferentproxies;nonecertifiesactualpolicyreturn.'])
