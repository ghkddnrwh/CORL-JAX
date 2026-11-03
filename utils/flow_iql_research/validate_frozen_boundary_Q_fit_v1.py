"""Verify isolated Q moments, boundary gradients, protected blocks and resume."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import empirical_backup_core_v1 as core
from utils.flow_iql_research import frozen_boundary_Q_fit_core_v1 as fitting
from utils.flow_iql_research.fit_frozen_boundary_Q_v1 import fingerprint
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def difference(a,b):
    return max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64))))
               for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rng=np.random.RandomState(61127);cfg=core.config_for('cube-single-play-singletask-v0',.999,.9,8)
    cfg.update(num_samples=9,agent_name='ifql',policy_type='ifql')
    initial=core.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql')
    data=dict(observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,(4,12,2)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
        rewards=-jnp.ones((4,12)),masks=jnp.ones((4,12)))
    base,_=core.make_scan(core.build_update('bounded_v2_empirical_backup',(-1000.,0.)))(initial,data)
    before=fingerprint(base);carry=fitting.extract(base,jax.random.PRNGKey(9));assert int(carry[1][0].count)==4
    assert difference(carry[1][0].mu,base.agent.network.opt_state[0].mu[fitting.Q])==0
    assert max(float(jnp.max(jnp.abs(x)))for x in jax.tree_util.tree_leaves(carry[1][0].mu))>0
    goals=dict(observations=jnp.asarray(rng.normal(size=(4,32,3)),jnp.float32),actions=jnp.asarray(rng.uniform(-1,1,(4,32,2)),jnp.float32))
    first=jax.tree_util.tree_map(lambda x:x[0],goals);key=jax.random.PRNGKey(71)
    def whole(params):
        agent=base.agent.replace(network=base.agent.network.replace(params=params))
        return fitting.objective(agent,params[fitting.Q],first,key)
    gradient=jax.grad(whole)(base.agent.network.params)
    isolated_gradient=jax.grad(lambda value:fitting.objective(base.agent,value,first,key))(carry[0])
    assert difference(gradient[fitting.Q],isolated_gradient)==0
    protected_gradient={name:max(float(jnp.max(jnp.abs(x)))for x in jax.tree_util.tree_leaves(gradient[name]))
        for name in('modules_target_critic','modules_value','modules_actor_flow')}
    assert max(protected_gradient.values())==0
    update=fitting.make_update(base.agent);scan=jax.jit(lambda state,batches:jax.lax.scan(update,state,batches))
    full,_=scan(carry,goals);half,_=scan(carry,jax.tree_util.tree_map(lambda x:x[:2],goals))
    restored=flax.serialization.from_state_dict(carry,copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed,_=scan(restored,jax.tree_util.tree_map(lambda x:x[2:],goals));assert difference(full,resumed)==0
    assert int(full[1][0].count)==8 and fingerprint(base)==before
    view=fitting.inference_view(base,full[0]);assert difference(view.agent.network.opt_state,base.agent.network.opt_state)==0
    for name in protected_gradient:assert difference(view.agent.network.params[name],base.agent.network.params[name])==0
    assert difference(view.agent.network.params[fitting.Q],base.agent.network.params[fitting.Q])>0
    result=dict(time=time.time(),actual_nonzero_Q_Adam_moments_extracted_exactly=True,initial_Q_Adam_count=4,final_auxiliary_Q_Adam_count=8,
        full_vs_serialized_two_plus_two_max_abs=0.,whole_network_vs_isolated_Q_gradient_exact=True,
        protected_block_gradient_max_abs=protected_gradient,base_whole_state_unchanged=True,
        inference_view_protected_blocks_and_original_joint_optimizer_exact=True,
        scope='ShortCPUimplementationvalidation;inferenceviewisnotajointresumestate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
