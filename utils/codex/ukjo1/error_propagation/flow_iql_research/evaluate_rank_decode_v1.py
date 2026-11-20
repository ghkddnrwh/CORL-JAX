"""One frozen policy with a different integration grid and unchanged teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import rank_flow_core_v1 as core
from utils.flow_iql_research import rank_decode_policy_v1 as decoder
from utils.flow_iql_research.rank_flow_policy_diagnostics_v1 import energy
from utils.flow_iql_research import train_rank_flow_v2 as primary
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--decode-steps',type=int,choices=(10,20,50),required=True)
    parser.add_argument('--episodes',type=int,choices=(1,50),default=50)
    parser.add_argument('--queries',type=int,default=64)
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    config=json.loads((args.run/'config.json').read_text());identity=config['identity']
    assert identity['kind']=='rank_flow'and identity['config']['flow_steps']==10
    for source,digest in config['source_sha256'].items():assert sha(source)==digest,source
    checkpoint=args.run/f'checkpoint_{args.step}.pkl'
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    assert saved['step']==int(saved['state']['q_updates'])==args.step
    assert json.loads(json.dumps(saved['identity']))==identity
    assert saved['source_sha256']==config['source_sha256']
    assert int(saved['state']['v_updates'])==int(saved['state']['v_opt']['0']['count'])==2*args.step
    assert int(saved['state']['agent']['network']['opt_state']['0']['count'])==args.step
    cfg=dict(identity['config'])
    for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    env,_,validation=ogbench.make_env_and_datasets(identity['environment'])
    reset=prepare_seeded_action_space(env,42)
    template=core.create(identity['seed'],jnp.zeros((1,env.observation_space.shape[-1])),jnp.zeros((1,env.action_space.shape[-1])),cfg)
    state=flax.serialization.from_state_dict(template,saved['state']);agent=state.agent
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(state))
    indices=np.random.RandomState(77231).randint(len(validation['actions']),size=args.queries)
    observations=jnp.asarray(validation['observations'][indices],jnp.float32)

    @jax.jit
    def inspect(observations):
        repeated=jnp.broadcast_to(observations,(args.samples,*observations.shape))
        key=jax.random.PRNGKey(7172)
        native=agent.sample_actions(repeated,key,0.)
        ten=decoder.decode(agent,repeated,key,10)
        chosen=decoder.decode(agent,repeated,key,args.decode_steps)
        dense=decoder.decode(agent,repeated,key,50)
        teacher,teacher_q,_=agent.selected_teacher_targets(repeated,jax.random.PRNGKey(7171))
        second,_,_=agent.selected_teacher_targets(repeated,jax.random.PRNGKey(7174))
        chosen_q=agent.network.select('critic')(repeated,actions=chosen).min(0)
        return dict(native_ten_actions=native,manual_ten_actions=ten,chosen_actions=chosen,
            fifty_actions=dense,teacher_actions=teacher,independent_teacher_actions=second,
            teacher_Q=teacher_q.max(0),chosen_Q=chosen_q)
    arrays={key:np.asarray(value)for key,value in jax.device_get(inspect(observations)).items()}
    assert all(np.isfinite(value).all()for value in arrays.values())
    native_delta=float(np.max(np.abs(arrays['native_ten_actions']-arrays['manual_ten_actions'])))
    assert native_delta<=1e-6,native_delta
    policy=decoder.make_policy(agent,args.decode_steps)
    result=primary.evaluate(state,env,policy,args.step,args.episodes)
    comparison=None
    if args.reference is not None:
        assert args.decode_steps==10
        found=[json.loads(line)for line in args.reference.read_text().splitlines()if json.loads(line)['step']==args.step]
        assert len(found)==1 and found[0]['n_episodes']==args.episodes
        comparison={key:result[key]==found[0][key]for key in ('episode_successes','episode_rewards','episode_lengths')}
        assert all(comparison.values()),comparison
    rms=lambda value:float(np.sqrt(np.mean(np.asarray(value,np.float64)**2)))
    summary=dict(policy_vs_best9_teacher_energy_mean=float(energy(arrays['chosen_actions'],arrays['teacher_actions']).mean()),
        teacher_vs_independent_teacher_energy_mean=float(energy(arrays['teacher_actions'],arrays['independent_teacher_actions']).mean()),
        chosen_vs_ten_same_latent_endpoint_RMS=rms(arrays['chosen_actions']-arrays['manual_ten_actions']),
        chosen_vs_fifty_same_latent_endpoint_RMS=rms(arrays['chosen_actions']-arrays['fifty_actions']),
        chosen_action_saturation_fraction=float(np.mean(np.abs(arrays['chosen_actions'])>=.98)),
        teacher_action_saturation_fraction=float(np.mean(np.abs(arrays['teacher_actions'])>=.98)),
        chosen_within_state_action_std_mean=float(arrays['chosen_actions'].std(0).mean()),
        model_Q_teacher_minus_chosen_mean=float(np.mean(arrays['teacher_Q']-arrays['chosen_Q'])))
    result.update(time=time.time(),kind='auxiliary_student_Euler_grid_refinement',run=args.run.name,
        environment=identity['environment'],training_seed=identity['seed'],decode_steps=args.decode_steps,
        fixed_BC_teacher_steps=10,unchanged_Q_V_BC_policy_parameters=True,
        checkpoint_sha256=sha(checkpoint),config_sha256=sha(args.run/'config.json'),
        decoder_source_sha256=sha(decoder.__file__),evaluator_source_sha256=sha(__file__),
        summary=summary,queries=args.queries,samples=args.samples,query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        manual_ten_vs_native_sampler_max_abs=native_delta,self_reference_episode_arrays_exact=comparison,
        reset_metadata=reset,primary_evaluations_modified=False,
        limits=['The BC teacher remains at ten Euler steps; only student numerical integration changes.',
                'The fifty-step flow is a numerical reference, not a proven exact ODE or target distribution.',
                'Endpoint RMS compares the same vector field and initial latent across grids; unrelated teacher-policy latents are compared by distributions.',
                'Learned Q and heldout-state distribution distances are not oracle Q errors or environment returns.',
                'This posthoc single-checkpoint diagnostic is not a new training seed, final-checkpoint selection or nine-task performance claim.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),indices=indices,**arrays)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,decode_steps=args.decode_steps,success=result['success_rate'],summary=summary,self_reference=comparison)),flush=True)


if __name__=='__main__':main()
