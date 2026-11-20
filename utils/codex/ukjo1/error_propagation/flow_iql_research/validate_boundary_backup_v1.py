"""Validate supervised boundary actuation and unchanged finite-backup controls."""
import argparse
import copy
import json
from pathlib import Path
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import boundary_empirical_backup_core_v1 as boundary
from utils.flow_iql_research import empirical_backup_core_v1 as baseline


def difference(a, b):
    aa, bb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(aa) == len(bb)
    return max(float(np.max(np.abs(np.asarray(x, np.float64)-np.asarray(y, np.float64)))) for x,y in zip(aa,bb))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rng=np.random.RandomState(61027)
    cfg=boundary.config_for('cube-single-play-singletask-v0',.999,.9,8)
    cfg.update(num_samples=9,policy_type='ifql',agent_name='ifql')
    initial=boundary.create(31,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,'ifql')
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
                 next_observations=jnp.asarray(rng.normal(size=(4,12,3)),jnp.float32),
                 actions=jnp.asarray(rng.uniform(-1,1,(4,12,2)),jnp.float32),
                 rewards=-jnp.ones((4,12)),masks=jnp.ones((4,12)),
                 boundary_observations=jnp.asarray(rng.normal(size=(4,32,3)),jnp.float32),
                 boundary_actions=jnp.asarray(rng.uniform(-1,1,(4,32,2)),jnp.float32),
                 boundary_valid=jnp.ones((4,32)))
    update=boundary.build_update('bounded_v2_boundary_empirical_backup',(-1000.,0.))
    scan=boundary.make_scan(update);full,logs=scan(initial,batches)
    half,_=scan(initial,jax.tree_util.tree_map(lambda x:x[:2],batches))
    restored=flax.serialization.from_state_dict(initial,copy.deepcopy(flax.serialization.to_state_dict(half)))
    resumed,_=scan(restored,jax.tree_util.tree_map(lambda x:x[2:],batches))
    assert difference(full,resumed)==0
    original,_=baseline.make_scan(baseline.build_update('bounded_v2_empirical_backup',(-1000.,0.)))(initial,batches)
    empty=dict(batches,boundary_valid=jnp.zeros((4,32)))
    disabled,disabled_logs=scan(initial,empty)
    empty_difference=difference(disabled,original);assert empty_difference<1e-6,empty_difference
    bc_difference=difference(full.agent.network.params['modules_actor_flow'],original.agent.network.params['modules_actor_flow'])
    assert bc_difference<1e-6
    q_difference=difference(full.agent.network.params['modules_critic'],original.agent.network.params['modules_critic'])
    assert q_difference>1e-8
    np.testing.assert_array_equal(full.agent.rng,original.agent.rng)
    assert int(full.q_updates)==4 and int(full.v_updates)==8
    assert np.min(np.asarray(logs['critic/boundary_loss']))>0
    assert np.max(np.asarray(disabled_logs['critic/boundary_loss']))==0
    first=jax.tree_util.tree_map(lambda x:x[0],batches)
    # The finite Q backup remains independent of the auxiliary learned V.
    params=copy.deepcopy(initial.agent.network.params)
    params['modules_value']=jax.tree_util.tree_map(lambda x:x+2,params['modules_value'])
    altered=initial.replace(agent=initial.agent.replace(network=initial.agent.network.replace(params=params)))
    standard,_=jax.jit(update)(initial,first);changed,_=jax.jit(update)(altered,first)
    V_isolation={name:difference(standard.agent.network.params[name],changed.agent.network.params[name]) for name in ('modules_critic','modules_target_critic','modules_actor_flow')}
    assert max(V_isolation.values())==0
    values=boundary.probe(full,first);summary=boundary.probe_summary(jax.device_get(values),.9,(-1000.,0.),None)
    assert summary['finite'] and summary['snapshot_boundary_active_fraction']==1
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(full))
    result=dict(time=time.time(),serialized_two_plus_two_full_state_max_abs=0.,
                empty_boundary_pool_vs_original_finite_full_state_max_abs=empty_difference,
                BC_four_updates_vs_original_finite_max_abs=bc_difference,
                critic_change_with_boundary_max_abs=q_difference,agent_RNG_exact=True,
                learned_V_perturbation_Q_BC_max_abs=V_isolation,
                boundary_loss_positive_when_active_and_zero_when_disabled=True,
                actual_finite_backup_and_known_boundary_snapshot_metrics_finite=True,q_updates=4,v_updates=8,
                scope='ShortCPUimplementationchecks;nointeriorQoracleorreturnimprovementclaim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
