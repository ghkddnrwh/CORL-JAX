"""CPU logic/isolation checks for an unregistered energy-score learner draft."""
import argparse
import json
from pathlib import Path
import pickle
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from utils.flow_iql_research import energy_onestep_deploy_core_v1 as core
from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research import prototype_energy_student_v1 as prototype
from utils.flow_iql_research import standalone_onestep_actor_v1 as standalone
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output_directory.exists()
    assert all(d.platform == 'cpu' for d in jax.devices())
    args.output_directory.mkdir(parents=True)
    obs = jnp.asarray([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32)
    action = jnp.asarray([[.2,-.1],[-.4,.3]],jnp.float32)
    batch = dict(observations=obs,actions=action,next_observations=.9*obs,
                 rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    cfg = core.config_for(core.ENVIRONMENTS[0],'energy_bc',width=8)
    state = core.create(6,obs,action,cfg,'energy_bc')
    spec = cfg['onestep_specification']
    rng = jax.random.PRNGKey(29001)
    ob,noises,target,heads,proposals,index = core.teacher_batch(state,batch,rng,spec)
    teacher = base.teacher_batch(state,batch,rng,spec)
    assert np.array_equal(np.asarray(target),np.asarray(teacher[2]))
    assert np.array_equal(np.asarray(proposals),np.asarray(teacher[4]))
    assert np.array_equal(np.asarray(index),np.asarray(heads.min(0).argmax(0)))
    assert noises.shape == (2,2,2) and teacher[1].shape == (9,2,2)
    assert not np.array_equal(np.asarray(noises[0]),np.asarray(noises[1]))
    assert not np.array_equal(np.asarray(noises[0]),-np.asarray(noises[1]))
    assert not np.array_equal(np.asarray(noises),np.asarray(teacher[1][:2]))
    loss,_ = core.actor_loss(state,ob,noises,target,heads,spec,state.actor.params)
    first = state.actor(ob,noises[0]);second = state.actor(ob,noises[1])
    expected = .5*jnp.mean(prototype.norm(first-target)+prototype.norm(second-target)-prototype.norm(first-second))
    assert abs(float(loss-expected)) < 1e-7
    assert np.array_equal(np.asarray(jax.grad(lambda x:core.norm(x))(jnp.zeros(2))),np.zeros(2))
    update = jax.jit(core.build_update('energy_bc',(-1000.,0.)))
    one,logs = update(state,batch)
    reference = jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
    deviation = difference(flax.serialization.to_state_dict(one.base),flax.serialization.to_state_dict(reference))
    assert deviation < 1e-5
    perturbed = state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda v:v+.25,state.actor.params)))
    other,_ = update(perturbed,batch)
    assert difference(flax.serialization.to_state_dict(one.base),flax.serialization.to_state_dict(other.base)) == 0
    full = state
    for _ in range(8):
        full,logs = update(full,batch)
    resumed = state
    for _ in range(4):
        resumed,_ = update(resumed,batch)
    resumed = flax.serialization.from_state_dict(state,flax.serialization.to_state_dict(resumed))
    for _ in range(4):
        resumed,_ = update(resumed,batch)
    assert difference(flax.serialization.to_state_dict(full),flax.serialization.to_state_dict(resumed)) == 0
    assert int(full.base.q_updates) == int(full.base.agent.network.opt_state[0].count) == 8
    assert int(full.actor_updates) == int(full.actor.opt_state[0].count) == 8
    assert int(full.base.v_updates) == int(full.base.v_opt[0].count) == 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(full))
    assert float(logs['deployment/Q_loss']) == 0 and float(logs['deployment/distill_MSE_is_training_objective']) == 0
    values = core.make_probe('energy_bc',(-1000.,0.))(full,batch)
    summary = core.probe_summary(jax.device_get(values),(-1000.,0.))
    assert summary['finite'] and np.isfinite(summary['energy_distribution_score'])
    metadata = core.actor_metadata(cfg,3,2)
    assert metadata['input'] == 'single' and metadata['noise_dim'] == 2
    artifact = core.export_actor(full,metadata)
    artifact_path = args.output_directory / 'implementation_fixture_actor8.pkl'
    with artifact_path.open('xb') as stream:
        pickle.dump(artifact,stream)
    loaded = standalone.load_actor(artifact_path)
    expected_action = np.asarray(core.make_actor_policy(metadata)(artifact['actor_params'],obs,jax.random.PRNGKey(42)))
    actual_action = np.asarray(loaded(np.asarray(obs),jax.random.PRNGKey(42)))
    assert np.allclose(expected_action,actual_action,rtol=0,atol=1e-6)
    result = dict(time=time.time(),status='pass',scope='Synthetic CPU8update logic/resume/portable fixture only; not RL performance.',
        teacher_best9_labels_and_proposals_match_original=True,teacher_Q_aggregation='online_min',
        student_draws_two_and_not_teacher_coupled_or_antithetic=True,
        production_score_matches_CPU_prototype=True,zero_norm_subgradient_finite=True,
        base_difference_vs_original_lean=deviation,actor_perturbation_effect_on_base=0.,
        Q_Adam=8,actor_Adam=8,V_Adam=0,resume4plus4_bitwise_equal=True,
        portable_actor_max_difference=float(np.abs(expected_action-actual_action).max()),
        fixture_artifact=str(artifact_path),fixture_artifact_sha256=sha(artifact_path),
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),*core.source_paths(),Path(prototype.__file__),Path(standalone.__file__)]},
        no_production_GPU_learner_or_queue_changes=True,no_performance_claim=True)
    (args.output_directory / 'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',output=str(args.output_directory / 'result.json'))),flush=True)


if __name__ == '__main__':
    main()
