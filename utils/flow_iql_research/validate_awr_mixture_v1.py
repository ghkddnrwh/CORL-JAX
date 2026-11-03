"""Check mixture AWR normalization, value isolation and deterministic decoding."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import awr_mixture_core_v1 as mixture
from utils.flow_iql_research import core as baseline
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def difference(first,second):
    a,b=jax.tree_util.tree_leaves(first),jax.tree_util.tree_leaves(second)
    assert len(a)==len(b)
    return max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64))))for x,y in zip(a,b))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rng=np.random.RandomState(61523)
    actions=jnp.asarray(rng.normal(size=(7,2)),jnp.float32)
    mean=jnp.asarray(rng.normal(size=(7,2)),jnp.float32)
    logstd=jnp.asarray(rng.uniform(-1,1,(7,2)),jnp.float32)
    logits=jnp.asarray(rng.normal(size=(7,5)),jnp.float32)
    expected=-.5*jnp.sum(((actions-mean)*jnp.exp(-logstd))**2+2*logstd+jnp.log(2*jnp.pi),axis=-1)
    one,_=mixture.mixture_log_prob(jnp.zeros((7,1)),mean[:,None],logstd[:,None],actions)
    duplicated,_=mixture.mixture_log_prob(logits,jnp.broadcast_to(mean[:,None],(7,5,2)),jnp.broadcast_to(logstd[:,None],(7,5,2)),actions)
    np.testing.assert_allclose(one,expected,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(duplicated,expected,rtol=1e-6,atol=1e-6)
    gradient=jax.grad(lambda value:mixture.mixture_log_prob(logits,jnp.broadcast_to(value[:,None],(7,5,2)),jnp.broadcast_to(logstd[:,None],(7,5,2)),actions)[0].sum())(mean)
    np.testing.assert_allclose(gradient,(actions-mean)*jnp.exp(-2*logstd),rtol=1e-6,atol=1e-6)
    candidates=jnp.asarray(rng.normal(size=(7,5,2)),jnp.float32)
    selected=mixture.greedy_component_mean(logits,candidates)
    np.testing.assert_array_equal(selected,np.asarray(candidates)[np.arange(7),np.asarray(logits).argmax(-1)])

    config=mixture.config_for('cube-single-play-singletask-v0',.999,.9,8)
    initial=mixture.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),config)
    public_config=baseline.config_for('cube-single-play-singletask-v0',.999,.9,8)
    reference=baseline.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),public_config)
    common=('modules_critic','modules_target_critic','modules_value')
    assert max(difference(initial.agent.network.params[name],reference.agent.network.params[name])for name in common)==0
    np.testing.assert_array_equal(initial.agent.rng,reference.agent.rng)
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,(4,12,2)),jnp.float32),rewards=-jnp.ones((4,12)),masks=jnp.ones((4,12)))
    first=jax.tree_util.tree_map(lambda value:value[0],batches);key=jax.random.PRNGKey(7)
    gradients=jax.grad(lambda params:initial.agent.actor_loss(first,params,key)[0])(initial.agent.network.params)
    QV_gradients={name:max(float(jnp.max(jnp.abs(value)))for value in jax.tree_util.tree_leaves(gradients[name]))for name in common}
    assert max(QV_gradients.values())==0
    update=mixture.build_update('bounded_v2',(-1000.,0.))
    scan=mixture.make_scan(update);full,_=scan(initial,batches)
    old,_=baseline.make_scan(baseline.build_update('bounded_v2',(-1000.,0.)))(reference,batches)
    common_deltas={name:difference(full.agent.network.params[name],old.agent.network.params[name])for name in common}
    assert max(common_deltas.values())<=1e-6,common_deltas
    half,_=scan(initial,jax.tree_util.tree_map(lambda value:value[:2],batches))
    restored=flax.serialization.from_state_dict(initial,copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed,_=scan(restored,jax.tree_util.tree_map(lambda value:value[2:],batches))
    assert difference(full,resumed)==0
    assert int(full.q_updates)==4 and int(full.v_updates)==8
    deterministic=full.agent.sample_actions(first['observations'],key,0.)
    np.testing.assert_array_equal(deterministic,full.agent.sample_actions(first['observations'],jax.random.PRNGKey(81),0.))
    params=copy.deepcopy(full.agent.network.params)
    for name in common:params[name]=jax.tree_util.tree_map(lambda value:value+10,params[name])
    other=full.agent.replace(network=full.agent.network.replace(params=params))
    np.testing.assert_array_equal(deterministic,other.sample_actions(first['observations'],key,0.))
    assert deterministic.shape==(12,2)and np.max(np.abs(deterministic))<=1
    assert full.agent.sample_actions(first['observations'][0],key,0.).shape==(2,)
    assert np.isfinite(np.asarray(full.agent.sample_actions(first['observations'],key,1.))).all()
    # At zero online Q and V, every AWR weight is exactly one.
    params=copy.deepcopy(initial.agent.network.params)
    for name in common:params[name]=jax.tree_util.tree_map(jnp.zeros_like,params[name])
    zero=initial.agent.replace(network=initial.agent.network.replace(params=params))
    loss,logs=zero.actor_loss(first,params,key)
    output=zero.network.select('actor')(first['observations'])
    unweighted=-mixture.mixture_log_prob(*output,first['actions'])[0].mean()
    assert abs(float(loss-unweighted))<1e-6 and float(logs['AWR_weight_mean'])==1.
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(full))
    gaussian=mixture.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),config,'awr_gaussian')
    assert gaussian.agent.config['mixture_components']==1
    assert gaussian.agent.sample_actions(first['observations'],key,0.).shape==(12,2)
    result=dict(time=time.time(),K1_normal_density_correct=True,duplicate_K5_density_and_gradient_correct=True,
        highest_component_weight_decoder_correct=True,zero_advantage_unweighted_likelihood_exact=True,
        common_initial_QV_and_RNG_exact=True,actor_QV_parameter_gradient_max_abs=QV_gradients,
        four_update_QV_vs_public_Gaussian_V2_max_abs=common_deltas,
        serialized_two_plus_two_full_state_max_abs=0.,deterministic_policy_seed_and_QV_independent=True,
        single_batch_and_stochastic_samples_finite_and_bounded=True,K1_learned_variance_control_constructed=True,
        q_updates=4,v_updates=8,
        scope='ShortCPUimplementationchecks;K5includeslearnedvariance,nolongGPUidentityorbenchmarkperformanceclaim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
