"""Verify policy-only objective, actual Adam moments, isolation and resume."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import rank_flow_core_v1 as rank
from utils.flow_iql_research import frozen_rank_policy_fit_core_v1 as fit
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def delta(first,second):
    return max(float(np.max(np.abs(np.asarray(a,np.float64)-np.asarray(b,np.float64))))for a,b in zip(jax.tree_util.tree_leaves(first),jax.tree_util.tree_leaves(second)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=rank.config_for('cube-single-play-singletask-v0',.999,.9,8)
    initial=rank.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),config)
    rng=np.random.RandomState(771)
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,8,3)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,8,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,(4,8,2)),jnp.float32),rewards=-jnp.ones((4,8)),masks=jnp.ones((4,8)))
    base,_=rank.make_scan(rank.build_update('bounded_v2_rank_flow',(-1000.,0.)))(initial,batches)
    before=flax.serialization.to_bytes(base)
    carry=fit.extract(base,jax.random.PRNGKey(81321))
    assert int(carry[1][0].count)==4
    assert delta(carry[1][0].mu,base.agent.network.opt_state[0].mu[fit.POLICY])==0
    assert max(float(jnp.max(jnp.abs(value)))for value in jax.tree_util.tree_leaves(carry[1][0].mu))>0
    first=jax.tree_util.tree_map(lambda value:value[0],batches);key=jax.random.PRNGKey(7)
    _,logs=base.agent.actor_loss(first,base.agent.network.params,key)
    loss=fit.objective(base.agent,carry[0],first,key)
    assert abs(float(loss-logs['policy_flow_loss']))<1e-6
    original_gradient=jax.grad(lambda params:base.agent.actor_loss(first,params,key)[0])(base.agent.network.params)[fit.POLICY]
    isolated_gradient=jax.grad(lambda params:fit.objective(base.agent,params,first,key))(carry[0])
    gradient_delta=delta(original_gradient,isolated_gradient);assert gradient_delta<1e-6
    update=fit.make_update(base.agent)
    scan=jax.jit(lambda state,batch:jax.lax.scan(update,state,batch))
    full,_=scan(carry,batches)
    half,_=scan(carry,jax.tree_util.tree_map(lambda value:value[:2],batches))
    restored=flax.serialization.from_state_dict(carry,copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed,_=scan(restored,jax.tree_util.tree_map(lambda value:value[2:],batches))
    assert delta(full,resumed)==0
    assert int(full[1][0].count)==8
    inference=fit.policy_only_inference_state(base,full[0])
    untouched={name:delta(base.agent.network.params[name],inference.agent.network.params[name])for name in ('modules_critic','modules_target_critic','modules_value','modules_actor_flow')}
    assert max(untouched.values())==0
    assert flax.serialization.to_bytes(base)==before
    assert int(inference.q_updates)==4 and int(inference.v_updates)==8
    assert delta(base.agent.network.opt_state,inference.agent.network.opt_state)==0
    changed=delta(base.agent.network.params[fit.POLICY],inference.agent.network.params[fit.POLICY]);assert changed>0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(time=time.time(),policy_loss_matches_original_component=True,policy_gradient_max_delta=gradient_delta,
        actual_nonzero_policy_moments_extracted_exact=True,initial_policy_Adam_count=4,final_auxiliary_policy_Adam_count=8,
        four_vs_serialized_two_plus_two_aux_state_max_delta=0.,untouched_parameter_max_deltas=untouched,
        original_whole_state_bytes_unchanged=True,original_joint_optimizer_unchanged=True,
        changed_policy_parameter_max_abs=changed,inference_q_updates=4,inference_v_updates=8,
        scope='ShortCPUcorrectnessonly;inferenceviewnotajointtrainingresumestate;notbenchmarkdata.')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
