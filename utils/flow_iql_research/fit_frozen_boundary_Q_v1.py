"""Posthoc Q-boundary repair diagnostic with actual isolated Q Adam moments."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, selection_v2
from utils.flow_iql_research import frozen_boundary_Q_fit_core_v1 as fitting
from utils.flow_iql_research import train_empirical_backup_v1 as primary
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
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());run=Path(plan['run']);out=Path(plan['output_directory']);out.mkdir(parents=True,exist_ok=True)
    assert not(out/'metadata.json').exists()
    for source,digest in plan['source_sha256'].items():assert sha(source)==digest,source
    conf=json.loads((run/'config.json').read_text());identity=conf['identity'];cfg=dict(identity['config'])
    assert sha(run/'config.json')==plan['config_sha256'] and identity['kind']=='ifql'
    assert cfg['num_samples']==9 and cfg['backup_samples']==9
    for source,digest in conf['source_sha256'].items():assert sha(source)==digest,source
    for field in('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
    cp=run/f"checkpoint_{plan['step']}.pkl";assert sha(cp)==plan['checkpoint_sha256']
    with cp.open('rb')as stream:saved=pickle.load(stream)
    assert saved['step']==plan['step'] and json.loads(json.dumps(saved['identity']))==identity
    assert saved['source_sha256']==conf['source_sha256']
    env,training,validation=ogbench.make_env_and_datasets(identity['environment']);reset=prepare_seeded_action_space(env,42)
    template=policy_core.create(identity['seed'],jnp.zeros((1,env.observation_space.shape[-1])),jnp.zeros((1,env.action_space.shape[-1])),cfg,'ifql')
    base=flax.serialization.from_state_dict(template,saved['state']);base=jax.device_put(base,jax.devices('cpu')[0])
    assert int(base.q_updates)==int(base.agent.network.opt_state[0].count)==plan['step']
    assert int(base.v_updates)==int(base.v_opt[0].count)==2*plan['step']
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(base))
    protected_names=('modules_target_critic','modules_value','modules_actor_flow')
    protected={name:fingerprint(base.agent.network.params[name])for name in protected_names}
    base_fingerprint=fingerprint(base)
    train_goals=np.flatnonzero(np.asarray(training['masks'])==0)
    assert len(train_goals) and np.all(np.asarray(training['rewards'])[train_goals]==0)
    goal_sets={};interior_sets={};rng=np.random.RandomState(88404)
    for split,data in [('training',training),('validation',validation)]:
        goals=np.flatnonzero(np.asarray(data['masks'])==0);assert len(goals)
        assert np.all(np.asarray(data['rewards'])[goals]==0)
        if len(goals)>4096:goals=np.random.RandomState(89131).choice(goals,4096,replace=False)
        goal_sets[split]=dict(observations=jnp.asarray(data['observations'][goals],jnp.float32),
            actions=jnp.clip(jnp.asarray(data['actions'][goals],jnp.float32),-1+1e-5,1-1e-5),
            uniform=jnp.asarray(rng.uniform(-1,1,(len(goals),data['actions'].shape[-1])),jnp.float32),indices=goals)
        indices=rng.randint(len(data['actions']),size=plan['queries'])
        observations=jnp.asarray(data['observations'][indices],jnp.float32)
        proposals,_=selection_v2.proposals(base.agent,observations,jax.random.PRNGKey(88405),9,'behavior')
        interior_sets[split]=dict(observations=observations,actions=jnp.clip(jnp.asarray(data['actions'][indices],jnp.float32),-1+1e-5,1-1e-5),
            proposals=proposals,indices=indices)
    @jax.jit
    def inspect_goal(params,observations,actions,uniform):
        net=fitting.inference_view(base,params).agent.network
        return net.select('critic')(observations,actions=actions),net.select('critic')(observations,actions=uniform)
    @jax.jit
    def inspect_interior(params,observations,actions,proposals):
        net=fitting.inference_view(base,params).agent.network
        repeated=jnp.broadcast_to(observations,(9,*observations.shape))
        return net.select('critic')(observations,actions=actions),net.select('critic')(repeated,actions=proposals).min(0)
    carry=fitting.extract(base,jax.random.PRNGKey(88401));initial_count=int(carry[1][0].count)
    assert initial_count==plan['step']
    baseline_interior={key:jax.device_get(inspect_interior(carry[0],data['observations'],data['actions'],data['proposals']))for key,data in interior_sets.items()}
    update=fitting.make_update(base.agent);scan=jax.jit(lambda state,batches:jax.lax.scan(update,state,batches))
    replay=np.random.RandomState(88402);extra=0;rows=[];start=time.monotonic()
    assert plan['budgets'][0]==0 and plan['budgets']==sorted(set(plan['budgets']))
    atomic(out/'metadata.json',dict(time=time.time(),kind='auxiliary_boundary_Q_only_fit_not_joint_training',plan=plan,
        base_state_fingerprint=base_fingerprint,protected_fingerprints=protected,initial_Q_Adam_count=initial_count,
        joint_resume_state_saved=False,training_goal_count=len(train_goals),reset_metadata=reset))
    for budget in plan['budgets']:
        while extra<budget:
            count=min(plan['chunk'],budget-extra);chosen=train_goals[replay.randint(len(train_goals),size=(count,plan['boundary_batch_size']))]
            batch=dict(observations=jnp.asarray(training['observations'][chosen],jnp.float32),
                actions=jnp.clip(jnp.asarray(training['actions'][chosen],jnp.float32),-1+1e-5,1-1e-5))
            carry,losses=scan(carry,batch);extra+=count
            assert int(carry[1][0].count)==initial_count+extra
            assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(carry))
            with(out/'fit_trace.jsonl').open('a')as stream:stream.write(json.dumps(dict(additional_Q_boundary_steps=extra,
                Q_Adam_count=int(carry[1][0].count),mean_boundary_loss=float(jnp.mean(losses)),elapsed_seconds=time.monotonic()-start))+'\n')
        view=fitting.inference_view(base,carry[0]);assert fingerprint(base)==base_fingerprint
        assert all(fingerprint(view.agent.network.params[name])==digest for name,digest in protected.items())
        diagnostics={};rms=lambda value:float(np.sqrt(np.mean(np.asarray(value,np.float64)**2)))
        for split,data in goal_sets.items():
            observed,uniform=jax.device_get(inspect_goal(carry[0],data['observations'],data['actions'],data['uniform']))
            observations=interior_sets[split];q,proposed=jax.device_get(inspect_interior(carry[0],observations['observations'],observations['actions'],observations['proposals']))
            oldq,oldproposed=baseline_interior[split];centered=proposed-proposed.mean(0);oldcentered=oldproposed-oldproposed.mean(0)
            diagnostics[split]=dict(goal_observed_Q_RMS_to_known_zero=rms(observed),goal_uniform_Q_RMS_to_known_zero=rms(uniform),
                inspected_goal_states=len(data['indices']),interior_queries=len(observations['indices']),
                interior_data_Q_mean=float(q.mean()),interior_data_Q_shift_RMS=rms(q-oldq),
                interior_action_centered_Q_shift_RMS=rms(centered-oldcentered),
                interior_best9_rank_flip_fraction=float(np.mean(proposed.argmax(0)!=oldproposed.argmax(0))),
                interior_action_Q_std_mean=float(proposed.std(0).mean()))
        result=primary.evaluate(view,env,primary.make_policy(view.agent),plan['step'],plan['episodes'])
        exact=None
        if extra==0:
            records=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()];reference=next(row for row in records if row['step']==plan['step'])
            assert reference['n_episodes']==plan['episodes'];exact={key:result[key]==reference[key]for key in('episode_successes','episode_rewards','episode_lengths')};assert all(exact.values()),exact
        result.update(kind='auxiliary_frozen_boundary_Q_fit',base_training_step=plan['step'],additional_Q_boundary_steps=extra,
            auxiliary_Q_Adam_count=int(carry[1][0].count),diagnostics=diagnostics,base_whole_state_unchanged=True,
            target_Q_V_BC_and_original_joint_optimizer_unchanged=True,self_reference_episode_arrays_exact=exact,
            checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha(args.plan),primary_evaluations_modified=False,
            limits=['OnlytheonlineQblockisrefittedonknownsuccess-boundarylabels;noTDorinteriorlossisusedduringthiscounterfactual.',
                    'BC,V,targetQandtheoriginaljointoptimizerarefixed;thisdoesnotreplaythenewcoupledtrainingalgorithm.',
                    'InteriorQ/rankchangesarenotoracleerrorchanges;repairaftercollapsemaydifferfromprevention.',
                    'Posthoconecheckpointdiagnostic,notanothertrainingseedorprimarylast3result.'])
        result_path=out/f'additional_Q_boundary_{extra}_CPU{plan["episodes"]}.json';atomic(result_path,result)
        with(out/f'isolated_Q_boundary_{extra}.pkl').open('wb')as stream:pickle.dump(dict(kind='isolated_Q_parameters_and_Adam_NOT_joint_resume',
            Q_params=jax.device_get(carry[0]),Q_optimizer=jax.device_get(carry[1]),key=jax.device_get(carry[2]),replay_rng=replay.get_state(),
            additional_Q_boundary_steps=extra,base_checkpoint_sha256=plan['checkpoint_sha256']),stream,protocol=pickle.HIGHEST_PROTOCOL)
        rows.append(dict(additional_Q_boundary_steps=extra,Q_Adam_count=int(carry[1][0].count),CPU_success=result['success_rate'],diagnostics=diagnostics,
            self_reference_episode_arrays_exact=exact,result_sha256=sha(result_path)))
        atomic(out/'status.json',dict(pid=__import__('os').getpid(),phase='running',time=time.time(),rows=rows));print(json.dumps(rows[-1]),flush=True)
    assert fingerprint(base)==base_fingerprint and sha(cp)==plan['checkpoint_sha256']
    atomic(out/'summary.json',dict(time=time.time(),rows=rows,base_whole_state_unchanged=True,protected_blocks_unchanged=True,
        scope='AuxiliaryposthocQ-boundaryrefitdiagnostic;notprimarylast3benchmark.',goal_complete=False))
    atomic(out/'status.json',dict(pid=__import__('os').getpid(),phase='complete',time=time.time(),rows=rows))


if __name__=='__main__':main()
