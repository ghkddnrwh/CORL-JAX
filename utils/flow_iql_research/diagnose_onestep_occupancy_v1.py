"""Read-only paired action errors on replay and actual policy-visited states.

This is a small mechanism diagnostic, never primary success-rate evidence.
Neither counterfactual policy controls the other policy's rollouts. Parameters,
optimizer states and canonical evaluations are never changed.
"""
import argparse
import json
from pathlib import Path
import pickle
import random
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research.onestep_implementation_registry_v1 import implementation
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def summary(rows):
    keys=('action_MSE','predicted_minQ_regret','actor_Q_disagreement','teacher_Q_disagreement')
    return dict(samples=len(rows),**{key:dict(mean=float(np.mean([r[key]for r in rows])),
        median=float(np.median([r[key]for r in rows])),p90=float(np.quantile([r[key]for r in rows],.9)))for key in keys})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=5);parser.add_argument('--sample-every',type=int,default=10)
    parser.add_argument('--replay-states',type=int,default=128);parser.add_argument('--max-steps',type=int)
    args=parser.parse_args();assert not args.output.exists()
    assert args.episodes>0 and args.sample_every>0 and args.replay_states>0
    assert all(device.platform=='cpu'for device in jax.devices())
    start=time.time();conf=json.loads((args.run/'config.json').read_text());identity=conf['identity']
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    selected=implementation(identity['recipe']).core
    checkpoint=args.run/f'checkpoint_{args.step}.pkl';artifact_path=args.run/f'actor_{args.step}.pkl'
    checkpoint_hash=sha(checkpoint);artifact_hash=sha(artifact_path)
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    with artifact_path.open('rb')as stream:artifact=pickle.load(stream)
    assert saved['identity']==artifact['identity']==identity and saved['step']==artifact['step']==args.step
    assert artifact['checkpoint_sha256']==checkpoint_hash
    assert saved['source_sha256']==artifact['source_sha256']==conf['source_sha256']
    assert artifact['config_sha256']==sha(args.run/'config.json')
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    metadata=artifact['metadata'];obsdim=metadata['observation_dim'];actdim=metadata['action_dim']
    template=selected.create(identity['seed'],jnp.zeros((1,obsdim)),jnp.zeros((1,actdim)),cfg,identity['recipe'])
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.base.q_updates)==int(state.actor_updates)==args.step and int(state.base.v_updates)==0
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
    exported=jax.tree_util.tree_leaves(artifact['actor_params']);actual=jax.tree_util.tree_leaves(flax.serialization.to_state_dict(state.actor.params))
    assert len(exported)==len(actual)and all(np.array_equal(a,b)for a,b in zip(exported,actual))
    actor=selected.make_actor_policy(metadata);teacher=selected.make_teacher_policy(state)
    actor_params=jax.device_put(artifact['actor_params']);teacher_params=flat.lean.policy_parameters(state.base)
    @jax.jit
    def q_values(params,obs,actor_action,teacher_action):
        first=state.base.agent.network.select('critic')(obs,actions=actor_action,params=params)
        second=state.base.agent.network.select('critic')(obs,actions=teacher_action,params=params)
        return first,second
    rows=[];observations=[];actor_actions=[];teacher_actions=[]
    def compare(obs,key,mode,episode,t,chosen=None):
        ob=np.asarray(obs,np.float32)
        aa=np.asarray(actor(actor_params,ob,key)) if mode!='primary'else chosen
        ta=np.asarray(teacher(teacher_params,ob,key)) if mode!='teacher'else chosen
        aq,tq=map(np.asarray,q_values(teacher_params['base'],ob,aa,ta))
        row=dict(mode=mode,episode=episode,t=t,action_MSE=float(np.mean((aa-ta)**2)),
            predicted_minQ_regret=float(tq.min()-aq.min()),actor_Q_disagreement=float(abs(aq[0]-aq[1])),
            teacher_Q_disagreement=float(abs(tq[0]-tq[1])))
        assert all(np.isfinite(row[k])for k in ('action_MSE','predicted_minQ_regret','actor_Q_disagreement','teacher_Q_disagreement'))
        rows.append(row);observations.append(ob.copy());actor_actions.append(aa.copy());teacher_actions.append(ta.copy())
    env,data,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    indices=np.random.RandomState(21931).randint(len(data['observations']),size=args.replay_states)
    for index,ix in enumerate(indices):compare(data['observations'][ix],jax.random.fold_in(jax.random.PRNGKey(51932),index),'replay',-1,index)
    episodes=[];reset_records=[]
    for mode in ('primary','teacher'):
        env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
        reset_records.append(dict(mode=mode,metadata=prepare_seeded_action_space(env,42)))
        policy,payload=(actor,actor_params)if mode=='primary'else(teacher,teacher_params)
        for episode in range(args.episodes):
            seed=42+episode;np.random.seed(seed);random.seed(seed);env.action_space.seed(seed)
            obs,_=env.reset(seed=seed);key=jax.random.PRNGKey(seed);t=0;reward_sum=0.;success=0.
            while True:
                key,action_key=jax.random.split(key)
                act=np.asarray(policy(payload,np.asarray(obs,np.float32),action_key)).copy()
                if t%args.sample_every==0:compare(obs,action_key,mode,episode,t,act)
                obs,reward,terminated,truncated,info=env.step(act);t+=1;reward_sum+=reward
                success=max(success,float(info.get('success',0.)))
                implementation_cutoff=bool(args.max_steps and t>=args.max_steps and not(terminated or truncated))
                if terminated or truncated or implementation_cutoff:break
            episodes.append(dict(mode=mode,seed=seed,length=t,reward=float(reward_sum),success=success,
                terminated=bool(terminated),truncated=bool(truncated),implementation_cutoff=implementation_cutoff))
        env.close()
    groups={mode:summary([row for row in rows if row['mode']==mode])for mode in ('replay','primary','teacher')}
    horizon=[]
    for mode in ('primary','teacher'):
        for low in range(0,max(row['t']for row in rows if row['mode']==mode)+1,100):
            subset=[row for row in rows if row['mode']==mode and low<=row['t']<low+100]
            if subset:horizon.append(dict(mode=mode,start=low,end=low+99,**summary(subset)))
    assert sha(checkpoint)==checkpoint_hash and sha(artifact_path)==artifact_hash
    args.output.parent.mkdir(parents=True,exist_ok=True);trace=args.output.with_suffix('.npz');assert not trace.exists()
    np.savez_compressed(trace,observations=np.asarray(observations),actor_actions=np.asarray(actor_actions),teacher_actions=np.asarray(teacher_actions))
    result=dict(time=time.time(),seconds=time.time()-start,status='complete',run=args.run.name,recipe=identity['recipe'],
        environment=identity['environment'],step=args.step,checkpoint_sha256=checkpoint_hash,actor_artifact_sha256=artifact_hash,
        source_sha256=sha(__file__),core_source_sha256=sha(selected.__file__),trace_path=str(trace),trace_sha256=sha(trace),
        optimizer_updates_performed=0,implementation_only=bool(args.max_steps),no_primary_performance_claim=True,
        replay_index_seed=21931,counterfactual_key_seed=51932,first_episode_seed=42,episodes_per_policy=args.episodes,
        sample_every=args.sample_every,groups=groups,horizon=horizon,episodes=episodes,rows=rows,reset_records=reset_records,
        scope='Paired same-key actions on fixed replay and independently rolled-out policy states. Observational occupancy diagnostic; not a true-Q oracle, causal identification, or primary evaluation.')
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(status='complete',groups=groups,implementation_only=bool(args.max_steps))),flush=True)


if __name__=='__main__':main()
