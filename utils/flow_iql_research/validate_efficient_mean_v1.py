"""Small-network equations, resume, sampler and student-isolation checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import efficient_mean_core_v1 as core
from utils.flow_iql_research import selection_v2
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def difference(a,b):
    aa,ta=jax.tree_util.tree_flatten(a)
    bb,tb=jax.tree_util.tree_flatten(b)
    assert ta==tb
    return max(float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64)))) for x,y in zip(aa,bb))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    rng=np.random.RandomState(411)
    batches=dict(observations=jnp.asarray(rng.normal(size=(4,8,3)),jnp.float32),
        next_observations=jnp.asarray(rng.normal(size=(4,8,3)),jnp.float32),
        actions=jnp.asarray(rng.uniform(-.9,.9,size=(4,8,2)),jnp.float32),
        rewards=-jnp.ones((4,8)),masks=jnp.ones((4,8)))
    initial_states={}
    finals={}
    rows=[]
    for variant in core.VARIANTS:
        cfg=core.set_guidance_config(core.config_for('antmaze-large-navigate-singletask-v0',variant,8),
            'antmaze-large-navigate-singletask-v0',variant)
        initial=core.create(13,jnp.zeros((1,3)),jnp.zeros((1,2)),cfg,variant)
        initial_states[variant]=initial
        scan=core.make_scan(core.build_update(variant,(-1000.,0.)))
        full,logs=scan(initial,batches)
        half,_=scan(initial,jax.tree_util.tree_map(lambda x:x[:2],batches))
        restored=flax.serialization.from_state_dict(initial,copy.deepcopy(flax.serialization.to_state_dict(half)))
        resumed,_=scan(restored,jax.tree_util.tree_map(lambda x:x[2:],batches))
        assert difference(full,resumed)==0
        assert int(full.q_updates)==4 and int(full.v_updates)==(8 if variant=='reference' else 0)
        assert int(full.student_updates)==(4 if variant in core.BC_STUDENTS else 0)
        assert int(full.policy_updates)==(4 if variant in core.GUIDED_POLICIES else 0)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(full))
        finals[variant]=full
        if variant in core.BC_STUDENTS:
            branch=initial.replace(q_updates=jnp.asarray(core.WARMUP))
            branch,_=scan(branch,batches)
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(branch))
        rows.append(dict(variant=variant,resume4_vs2plus2_exact=True,Q_updates=4,V_updates=int(full.v_updates),
            BC_student_updates=int(full.student_updates),guided_policy_updates=int(full.policy_updates)))
        print('VALIDATED',variant,flush=True)
    control={name:difference(finals['reference'].agent.network.params[name],finals['lean'].agent.network.params[name])
        for name in ('modules_critic','modules_target_critic','modules_actor_flow')}
    assert max(control.values())<1e-5,control
    state=initial_states['lean']
    obs,noises=core.bank(state.agent,batches['next_observations'][0],jax.random.PRNGKey(31))
    new=core.flow(state.agent,obs,noises,10)
    old,_=selection_v2.proposals(state.agent,batches['next_observations'][0],jax.random.PRNGKey(31),9,'behavior')
    assert difference(new,old)==0
    state=initial_states['student_guided']
    altered=state.policy_student.replace(params=jax.tree_util.tree_map(lambda x:x+2,state.policy_student.params))
    modified=state.replace(policy_student=altered)
    scan=core.make_scan(core.build_update('student_guided',(-1000.,0.)))
    standard,_=scan(state,batches)
    perturbed,_=scan(modified,batches)
    isolation={name:difference(standard.agent.network.params[name],perturbed.agent.network.params[name])
        for name in ('modules_critic','modules_target_critic','modules_actor_flow')}
    assert max(isolation.values())==0,isolation
    for variant,state in finals.items():
        policy=core.make_policy(state,variant)
        action=policy(core.policy_parameters(state),jnp.zeros(3),jax.random.PRNGKey(42))
        assert action.shape==(2,) and np.isfinite(np.asarray(action)).all()
    result=dict(time=time.time(),status='passed',variants=rows,reference_vs_noV_Q_BC_max_abs=control,
        full10step_sampler_exact=True,guided_policy_does_not_enter_bootstrap=isolation,
        CPU_single_observation_all_variants_finite=True,source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        scope='Synthetic3observation/2action,tiny8wide networks. Actual9variant updates and serializedresume; postwarmup branch uses counterfixtures, not a production checkpoint. No performance claim.')
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
