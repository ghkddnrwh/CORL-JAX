"""Validate rank invariance, majority robustness, ties and exact single policy."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core,selection_v2,independent_critic_policy_v1 as selector
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    q=jnp.asarray([[0,1,2,3,9,4,5,6,7],[1,2,3,4,10,5,6,7,8],[1000000,4,3,2,-10,0,-1,-2,-3]],jnp.float32)
    assert int(jnp.argmax(selector.scores(q,'median_rank')))==4
    assert int(jnp.argmax(selector.scores(q,'mean')))==0
    transformed=q*jnp.asarray([2.,.25,3.])[:,None]+jnp.asarray([13.,-7.,31.])[:,None]
    np.testing.assert_array_equal(selector.midranks(q),selector.midranks(transformed))
    np.testing.assert_array_equal(selector.midranks(jnp.ones((3,9))),4.5*jnp.ones((3,9)))
    tied=jnp.asarray([[1,1,4]],jnp.float32);np.testing.assert_array_equal(selector.midranks(tied),jnp.asarray([[1.,1.,2.5]]))
    cfg=policy_core.config_for('cube-single-play-singletask-v0',.999,.9,8);cfg.update(num_samples=9,agent_name='ifql',policy_type='ifql')
    initial=policy_core.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql');agent=initial.agent
    baseline=selection_v2.make_policy(agent,'max',9,'behavior');single=selector.make_selector(agent,'median_rank',9)
    rng=np.random.RandomState(71327);observations=jnp.asarray(rng.normal(size=(7,3)),jnp.float32)
    for key in range(8):
        seed=jax.random.PRNGKey(key);action,_,_=single(agent.network.params,(agent.network.params['modules_critic'],),observations,seed)
        np.testing.assert_array_equal(action,baseline(agent.network.params,observations,seed))
    members=tuple(policy_core.create(seed,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql').agent.network.params['modules_critic']for seed in(31,32,33))
    action,values,ranks=single(agent.network.params,members,observations,jax.random.PRNGKey(9))
    assert action.shape==(7,2)and values.shape==(3,9,7)and ranks.shape==(9,7)and np.max(np.abs(action))<=1
    # num_samples is an inference-only control in the base IFQL learning update.
    cfg32=dict(cfg,num_samples=32);other=policy_core.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg32,'ifql')
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,6,3)),jnp.float32),next_observations=jnp.asarray(rng.normal(size=(4,6,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,(4,6,2)),jnp.float32),rewards=-jnp.ones((4,6)),masks=jnp.ones((4,6)))
    scan=policy_core.make_scan(policy_core.build_update('bounded_v2',(-1000.,0.),'ifql'));a,_=scan(initial,batches);b,_=scan(other,batches)
    delta=max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64))))for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)));assert delta==0
    result=dict(time=time.time(),strict_majority_survives_one_extreme_dissenting_critic=True,
        per_member_positive_affine_score_invariance_exact=True,tied_midranks_correct=True,
        single_member_policy_batch_actions_exact_for_eight_keys=True,three_member_batch_shapes_and_bounds_correct=True,
        num_samples9_vs32_four_CPU_learning_updates_full_state_max_abs=delta,
        scope='Finite-rankandshortCPUimplementationchecks;nooraclecriticrankingorlongGPUbitwiseclaim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
