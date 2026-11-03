"""Auxiliary CPU policy fitting against one frozen current best9 Q/BC teacher.

No primary model, optimizer, RNG, score or source is modified. Artifacts contain
only isolated policy parameters/moments, never a joint training resume state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import rank_flow_core_v1 as core
from utils.flow_iql_research import frozen_rank_policy_fit_core_v1 as fitting
from utils.flow_iql_research.rank_flow_policy_diagnostics_v1 import energy
from utils.flow_iql_research import train_rank_flow_v2 as primary
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(tree):
    result=hashlib.sha256()
    for value in jax.tree_util.tree_leaves(tree):
        array=np.asarray(value);result.update(str(array.shape).encode());result.update(str(array.dtype).encode());result.update(array.tobytes())
    return result.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());run=Path(plan['run']);out=Path(plan['output_directory']);out.mkdir(parents=True,exist_ok=True)
    assert not(out/'diagnostic_metadata.json').exists(),'Preserveexistingauxiliaryresults'
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    config=json.loads((run/'config.json').read_text());identity=config['identity']
    assert sha(run/'config.json')==plan['config_sha256']
    assert identity['kind']=='rank_flow'and identity['profile']=='bounded_v2_rank_flow'
    for path,digest in config['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=Path(plan['checkpoint']) if 'checkpoint' in plan else run/f"checkpoint_{plan['step']}.pkl";assert sha(checkpoint)==plan['checkpoint_sha256']
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    assert json.loads(json.dumps(saved['identity']))==identity and saved['source_sha256']==config['source_sha256']
    assert saved['step']==int(saved['state']['q_updates'])==plan['step']
    assert int(saved['state']['v_updates'])==int(saved['state']['v_opt']['0']['count'])==2*plan['step']
    assert int(saved['state']['agent']['network']['opt_state']['0']['count'])==plan['step']
    cfg=dict(identity['config'])
    for field in ('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
    env,raw,validation=ogbench.make_env_and_datasets(identity['environment']);reset=prepare_seeded_action_space(env,42)
    template=core.create(identity['seed'],jnp.zeros((1,env.observation_space.shape[-1])),jnp.zeros((1,env.action_space.shape[-1])),cfg)
    base=flax.serialization.from_state_dict(template,saved['state']);base=jax.device_put(base,jax.devices('cpu')[0])
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(base))
    assert cfg['flow_steps']==10
    base_fingerprint=fingerprint(base)
    protected_names=('modules_critic','modules_target_critic','modules_value','modules_actor_flow')
    protected={name:fingerprint(base.agent.network.params[name])for name in protected_names}
    carry=fitting.extract(base,jax.random.PRNGKey(88321));initial_count=int(carry[1][0].count)
    assert initial_count==plan['step']
    indices=np.random.RandomState(77231).randint(len(validation['actions']),size=plan['queries'])
    observations=jnp.asarray(validation['observations'][indices],jnp.float32)

    @jax.jit
    def prepare(observations):
        repeated=jnp.broadcast_to(observations,(plan['samples'],*observations.shape))
        selected,values,_=base.agent.selected_teacher_targets(repeated,jax.random.PRNGKey(7171))
        independent,_,_=base.agent.selected_teacher_targets(repeated,jax.random.PRNGKey(7174))
        initial=jax.random.normal(jax.random.PRNGKey(7175),selected.shape)
        time=jax.random.uniform(jax.random.PRNGKey(7176),(*selected.shape[:-1],1))
        position=(1-time)*initial+time*selected
        bc=base.agent.compute_flow_actions(repeated,jax.random.normal(jax.random.PRNGKey(7173),selected.shape))
        return dict(observations=repeated,teacher=selected,teacher_Q=values.max(0),independent_teacher=independent,
                    BC=bc,position=position,time=time,velocity=selected-initial)
    cache=prepare(observations)

    @jax.jit
    def inspect(params):
        model=fitting.policy_only_inference_state(base,params).agent
        actions=model.sample_actions(cache['observations'],jax.random.PRNGKey(7172),0.)
        prediction=model.network.select('actor_policy_flow')(cache['observations'],cache['position'],cache['time'])
        values=base.agent.network.select('critic')(cache['observations'],actions=actions).min(0)
        return dict(actions=actions,policy_FM_loss=jnp.mean((prediction-cache['velocity'])**2),policy_Q=values)

    cached=jax.device_get(cache)
    data=dict(observations=np.asarray(raw['observations'],np.float32),
              actions=np.clip(np.asarray(raw['actions'],np.float32),-1+1e-5,1-1e-5))
    replay=np.random.RandomState(88322)
    update=fitting.make_update(base.agent)
    scan=jax.jit(lambda state,batches:jax.lax.scan(update,state,batches))
    assert plan['budgets'][0]==0 and plan['budgets']==sorted(set(plan['budgets']))
    atomic(out/'diagnostic_metadata.json',dict(time=time.time(),kind='auxiliary_frozen_teacher_policy_fit_not_a_training_run',
        plan=plan,base_Q_training_step=plan['step'],initial_policy_Adam_count=initial_count,
        fixed_Q_V_BC_parameter_fingerprints=protected,base_state_fingerprint=base_fingerprint,
        fresh_training_teacher_samples_every_update=True,cached_teacher_samples_for_heldout_probes_only=True,
        joint_training_resume_state_saved=False,reset_metadata=reset))
    extra=0;rows=[];start=time.monotonic()
    for budget in plan['budgets']:
        while extra<budget:
            count=min(plan['chunk'],budget-extra)
            sampled=replay.randint(len(data['actions']),size=(count,cfg['batch_size']))
            batches={name:jnp.asarray(value[sampled])for name,value in data.items()}
            carry,losses=scan(carry,batches);extra+=count
            assert int(carry[1][0].count)==initial_count+extra
            assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(carry))
            with(out/'fit_trace.jsonl').open('a')as stream:
                stream.write(json.dumps(dict(additional_policy_steps=extra,policy_Adam_count=int(carry[1][0].count),
                    mean_chunk_FM_loss=float(jnp.mean(losses)),elapsed_seconds=time.monotonic()-start))+'\n')
        view=fitting.policy_only_inference_state(base,carry[0])
        assert fingerprint(base)==base_fingerprint
        assert all(fingerprint(view.agent.network.params[name])==digest for name,digest in protected.items())
        probed={name:np.asarray(value)for name,value in jax.device_get(inspect(carry[0])).items()}
        result=primary.evaluate(view,env,primary.make_policy(view.agent),plan['step'],plan['episodes'])
        comparison=None
        if budget==0:
            if 'reference_evaluation' in plan:
                reference_path=Path(plan['reference_evaluation'])
                assert sha(reference_path)==plan['reference_evaluation_sha256']
                records=[json.loads(reference_path.read_text())]
            else:
                records=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()]
            found=[value for value in records if value['step']==plan['step']]
            assert len(found)==1 and found[0]['n_episodes']==plan['episodes']
            comparison={name:result[name]==found[0][name]for name in ('episode_successes','episode_rewards','episode_lengths')}
            assert all(comparison.values()),comparison
        summary=dict(fixed_teacher_heldout_policy_FM_loss=float(probed['policy_FM_loss']),
            policy_vs_fixed_teacher_energy_mean=float(energy(probed['actions'],cached['teacher']).mean()),
            independent_teacher_resolution_energy_mean=float(energy(cached['independent_teacher'],cached['teacher']).mean()),
            BC_vs_fixed_teacher_energy_mean=float(energy(cached['BC'],cached['teacher']).mean()),
            model_Q_fixed_teacher_minus_policy_mean=float(np.mean(cached['teacher_Q']-probed['policy_Q'])),
            policy_action_saturation_fraction=float(np.mean(np.abs(probed['actions'])>=.98)),
            policy_within_state_action_std_mean=float(probed['actions'].std(0).mean()))
        result.update(kind='auxiliary_policy_only_frozen_teacher_fit',run=run.name,base_training_step=plan['step'],
            additional_policy_steps=extra,auxiliary_policy_Adam_count=int(carry[1][0].count),
            Q_V_BC_unchanged=True,base_whole_state_unchanged=True,summary=summary,
            self_reference_episode_arrays_exact=comparison,checkpoint_sha256=plan['checkpoint_sha256'],
            config_sha256=plan['config_sha256'],plan_sha256=sha(args.plan),evaluator_source_sha256=sha(__file__),
            query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),primary_evaluations_modified=False,
            limits=['FreshBCtargetsaredrawnduringtraining;thecacheisheldoutdiagnosticdataonly.',
                    'OnlythepolicyparameterblockadvanceswithitsextractedactualAdammoments;Q/V/BCandtheoriginaljointstatearefrozen.',
                    'Thisisaposthocfixed-checkpointmechanismdiagnostic,notanothertrainingseedorprimarylast3result.',
                    'ImprovedFMorenergydistancedoesnotcertifypolicyreturnororaclecriticaccuracy.'])
        path=out/f'additional_policy_{extra}_CPU{plan["episodes"]}.json';atomic(path,result)
        np.savez_compressed(out/f'additional_policy_{extra}_probes.npz',indices=indices,policy_actions=probed['actions'],teacher_actions=cached['teacher'],policy_Q=probed['policy_Q'],teacher_Q=cached['teacher_Q'])
        with(out/f'auxiliary_policy_only_{extra}.pkl').open('wb')as stream:
            pickle.dump(dict(kind='isolated_policy_parameters_and_Adam_not_joint_resume',base_checkpoint_sha256=plan['checkpoint_sha256'],
                policy_params=jax.device_get(carry[0]),policy_optimizer=jax.device_get(carry[1]),auxiliary_key=jax.device_get(carry[2]),
                auxiliary_replay_rng=replay.get_state(),additional_policy_steps=extra),stream,protocol=pickle.HIGHEST_PROTOCOL)
        rows.append(dict(additional_policy_steps=extra,policy_Adam_count=int(carry[1][0].count),CPU_success=result['success_rate'],summary=summary,result_sha256=sha(path)))
        atomic(out/'status.json',dict(pid=__import__('os').getpid(),time=time.time(),completed_policy_budget=extra,rows=rows,phase='running'))
        print(json.dumps(rows[-1]),flush=True)
    assert sha(checkpoint)==plan['checkpoint_sha256']and fingerprint(base)==base_fingerprint
    atomic(out/'summary.json',dict(time=time.time(),rows=rows,base_training_step=plan['step'],episodes=plan['episodes'],
        base_whole_state_unchanged=True,Q_V_BC_parameter_fingerprints_unchanged=True,plan_sha256=sha(args.plan),
        scope='Frozen-current-teacherpolicyrefittingdiagnostic;notprimarylast3benchmarkortrainingreplication.',goal_complete=False))
    atomic(out/'status.json',dict(pid=__import__('os').getpid(),time=time.time(),completed_policy_budget=extra,rows=rows,phase='complete'))


if __name__=='__main__':main()
