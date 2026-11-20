"""Check exact target actuation, untouched BC/V, clocks and serialization."""
import argparse
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import next_boundary_backup_core_v1 as core
from utils.flow_iql_research import empirical_backup_core_v1 as original
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def difference(a,b):
    xs,ys = jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b);assert len(xs)==len(ys)
    return max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64)))) for x,y in zip(xs,ys))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cfg=core.config_for('cube-single-play-singletask-v0',.999,.9,8)
    cfg.update(num_samples=9,policy_type='ifql',agent_name='ifql')
    initial=core.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql')
    rng=np.random.RandomState(987)
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,6,3)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,6,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-1,1,size=(4,6,2)),jnp.float32),
        rewards=-jnp.ones((4,6)),masks=jnp.ones((4,6)),
        next_success=jnp.asarray(rng.randint(2,size=(4,6)),jnp.float32))
    update=core.build_update('bounded_v2_next_boundary_backup',(-1000.,0.));scan=core.make_scan(update)
    base=original.build_update('bounded_v2_empirical_backup',(-1000.,0.));base_scan=core.make_scan(base)
    zero={**batches,'next_success':jnp.zeros((4,6))}
    absent,_=scan(initial,zero);reference,_=base_scan(initial,zero)
    no_boundary=difference(absent,reference);assert no_boundary==0
    full,_=scan(initial,batches)
    first_half,_=scan(initial,jax.tree_util.tree_map(lambda x:x[:2],batches))
    restored=flax.serialization.from_state_dict(initial,copy.deepcopy(flax.serialization.to_state_dict(first_half)))
    resumed,_=scan(restored,jax.tree_util.tree_map(lambda x:x[2:],batches))
    resume=difference(full,resumed);assert resume==0
    untouched={name:difference(full.agent.network.params[name],reference.agent.network.params[name])
        for name in ('modules_actor_flow','modules_value')}
    # V targets can differ after Q has changed. Test V independence at FIRST update,
    # while BC flow remains independent over all four updates.
    assert untouched['modules_actor_flow']==0
    batch=jax.tree_util.tree_map(lambda x:x[0],batches)
    boundary={**batch,'next_success':jnp.ones((6,))}
    actual,info=jax.jit(update)(initial,boundary)
    known,known_info=jax.jit(base)(initial,{**batch,'masks':jnp.zeros((6,))})
    exact_boundary=difference(actual,known);assert exact_boundary==0
    qs=initial.agent.network.select('critic')(batch['observations'],actions=batch['actions'])
    expected_loss=float(jnp.mean(jnp.sum((qs-batch['rewards'][None])**2,axis=0)))
    assert abs(float(info['critic/critic_loss'])-expected_loss)<1e-6
    unmodified,_=jax.jit(base)(initial,batch)
    first_V=difference(actual.agent.network.params['modules_value'],unmodified.agent.network.params['modules_value']);assert first_V==0
    q_change=difference(actual.agent.network.params['modules_critic'],unmodified.agent.network.params['modules_critic']);assert q_change>0
    assert int(full.q_updates)==4 and int(full.v_updates)==8
    probe=jax.device_get(core.probe(initial,boundary));np.testing.assert_array_equal(probe[3],np.asarray(batch['rewards']))
    stats=core.probe_summary(probe,.9,(-1000.,0.),None)
    assert stats['snapshot_boundary_active_count']==6 and stats['snapshot_next_success_root_RMS_to_known_zero']is not None
    result=dict(time=time.time(),no_success_vs_original_full_state_max_abs=no_boundary,
        all_next_success_vs_exact_zero_bootstrap_full_state_max_abs=exact_boundary,
        actual_TD_loss=expected_loss,four_vs_serialized_two_plus_two_full_state_max_abs=resume,
        BC_four_update_max_abs=untouched['modules_actor_flow'],first_V_update_max_abs=first_V,
        changed_Q_first_update_max_abs=q_change,actual_probe_target_equals_reward=True,
        clocks=dict(Q=4,V=8),scope='CPUimplementationchecks;no benchmark score.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
