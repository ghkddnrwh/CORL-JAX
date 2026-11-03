"""Test2x2 ablation semantics and unchanged original policy computations."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core,independent_critic_policy_v1 as old,independent_critic_policy_v2 as new
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    q=jnp.asarray([[9,8,1,2,3,4,5,6,7],[9,8,1,2,3,4,5,6,7],[-100,10000,1,2,3,4,5,6,7]],jnp.float32)
    choices={rule:int(jnp.argmax(new.scores(q,rule)))for rule in new.RULES}
    assert choices==dict(mean=1,median_rank=0,median_value=0,mean_rank=1)
    shifted=q+jnp.asarray([1000.,-8.,0.])[:,None]
    assert int(jnp.argmax(new.scores(shifted,'median_value')))==1
    assert int(jnp.argmax(new.scores(shifted,'mean')))==1
    assert int(jnp.argmax(new.scores(shifted,'median_rank')))==0
    assert int(jnp.argmax(new.scores(shifted,'mean_rank')))==1
    transformed=q*jnp.asarray([2.,.25,3.])[:,None]+jnp.asarray([13.,-7.,31.])[:,None]
    for rule in('median_rank','mean_rank'):np.testing.assert_array_equal(new.scores(q,rule),new.scores(transformed,rule))
    cfg=policy_core.config_for('cube-single-play-singletask-v0',.999,.9,8)
    cfg.update(num_samples=9,agent_name='ifql',policy_type='ifql')
    models=[policy_core.create(seed,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql')for seed in(31,32,33)]
    agent=models[0].agent;members=tuple(model.agent.network.params['modules_critic']for model in models)
    observations=jnp.asarray(np.random.RandomState(71327).normal(size=(7,3)),jnp.float32)
    exact={}
    for rule in('mean','median_rank'):
        previous=old.make_selector(agent,rule,9);current=new.make_selector(agent,rule,9)
        for key in range(8):
            a=previous(agent.network.params,members,observations,jax.random.PRNGKey(key))
            b=current(agent.network.params,members,observations,jax.random.PRNGKey(key))
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        exact[rule]=True
    for rule in new.RULES:
        compiled=new.make_selector(agent,rule,9)
        actions,values,scores=compiled(agent.network.params,members,observations,jax.random.PRNGKey(9))
        assert actions.shape==(7,2)and values.shape==(3,9,7)and scores.shape==(9,7)
        assert np.max(np.abs(actions))<=1 and np.isfinite(np.asarray(values)).all()
    result=dict(time=time.time(),original_rules_eight_keys_actions_values_scores_exact=exact,
        two_unique_best_critics_resisted_by_median_rank_but_not_mean_rank=True,
        raw_median_can_reverse_after_independent_state_offsets=True,
        both_rank_rules_positive_affine_scores_exact=True,all_four_batch_shapes_and_action_bounds_valid=True,
        scope='SyntheticandshortCPUinferencechecks;nooraclecorrectnessorbenchmarkperformanceclaim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
