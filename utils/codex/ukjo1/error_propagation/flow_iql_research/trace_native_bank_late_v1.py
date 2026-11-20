"""Read-only last3 joint-actor occupancy and crossed fit diagnosis.

Evaluation interactions never update any model. Each rollout must exactly
reproduce the first ten episodes of its sealed native CPU50 evaluation.
Different checkpoints are scored on the SAME visited-state sets and noise.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    proposal=json.loads(a.proposal.read_text());case=proposal['cases'][a.case]
    for path,digest in proposal['source_sha256'].items():assert sha(path)==digest,path
    run=Path(case['run']);conf=json.loads((run/'config.json').read_text())
    assert sha(run/'config.json')==case['config_sha256']
    assert conf['identity']['recipe']=='bank_bc'
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    with np.load(case['replay_cache'])as cache:heldout=cache['observations'][65536:65536+512].copy()
    assert sha(case['replay_cache'])==case['replay_cache_sha256']
    cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    dim=json.loads((run/'dataset.json').read_text())['action_dim']
    template=flat.create(0,jnp.asarray(heldout[:1]),jnp.zeros((1,dim)),cfg,'bank_bc')
    result=dict(time=time.time(),status='rollouts',environment=conf['identity']['environment'],
        proposal_sha256=sha(a.proposal),source_sha256=proposal['source_sha256'],
        steps=[800000,900000,1000000],episode_seeds=list(range(42,52)),sample_every=20,
        optimizer_updates_performed=0,evaluation_states_used_for_learning=False,
        scope='Read-only joint actor diagnostic; no new primary score, training data or causal identification.',
        rollouts={},crossed_fit={})
    a.output.mkdir(parents=True);write(a.output/'status.json',result)
    states={};occupancies=dict(heldout_replay=heldout)
    for step in result['steps']:
        entry=case['artifacts'][str(step)]
        for kind in ('checkpoint','actor','primary_evaluation'):
            assert sha(entry[kind])==entry[kind+'_sha256']
        with Path(entry['checkpoint']).open('rb')as f:saved=pickle.load(f)
        with Path(entry['actor']).open('rb')as f:artifact=pickle.load(f)
        state=flax.serialization.from_state_dict(template,saved['state'])
        assert int(state.base.q_updates)==int(state.actor_updates)==step and int(state.base.v_updates)==0
        assert int(state.actor.opt_state[0].count)==int(state.base.agent.network.opt_state[0].count)==step
        states[step]=state
        policy=flat.make_actor_policy(artifact['metadata'])
        prior=json.loads(Path(entry['primary_evaluation']).read_text())
        assert prior['n_episodes']==50 and prior['first_episode_seed']==42
        env=ogbench.make_env_and_datasets(result['environment'],env_only=True)
        prepare_seeded_action_space(env,42);samples=[];episodes=[]
        for seed in result['episode_seeds']:
            np.random.seed(seed);random.seed(seed);env.action_space.seed(seed)
            x,_=env.reset(seed=seed);key=jax.random.PRNGKey(seed)
            t=0;success=0.;reward_sum=0.;mismatch=[];events=0;previous=None
            while True:
                key,ak=jax.random.split(key)
                action=np.asarray(policy(artifact['actor_params'],np.asarray(x,np.float32),ak)).copy()
                if t%20==0:samples.append((np.asarray(x,np.float32).copy(),seed,t))
                x,reward,terminated,truncated,info=env.step(action)
                reward_sum+=float(reward);success=max(success,float(info.get('success',0.)));t+=1
                bare=env.unwrapped
                if hasattr(bare,'_cur_button_states'):
                    buttons=np.asarray(bare._cur_button_states).copy()
                    mismatch.append(int(np.count_nonzero(buttons!=bare._target_button_states)))
                    if previous is not None:events+=int(not np.array_equal(previous,buttons))
                    previous=buttons
                if terminated or truncated:break
            row=dict(seed=seed,success=success,reward=reward_sum,length=t)
            if mismatch:row.update(min_button_mismatches=min(mismatch),final_button_mismatches=mismatch[-1],button_state_change_events=events)
            episodes.append(row)
        env.close()
        for field,key in [('success','episode_successes'),('reward','episode_rewards'),('length','episode_lengths')]:
            assert [x[field]for x in episodes]==prior[key][:10],(step,field)
        obs=np.stack([x[0]for x in samples]);occupancies[str(step)]=obs
        path=a.output/f'visited_states_{step}_v1.npz'
        np.savez_compressed(path,observations=obs,episode_seeds=[x[1]for x in samples],episode_steps=[x[2]for x in samples])
        result['rollouts'][str(step)]=dict(episodes=episodes,first10_native_arrays_exact=True,
            sampled_states=len(obs),states_sha256=sha(path),success=sum(x['success']for x in episodes)/10)
        write(a.output/'status.json',result)
        print(json.dumps(dict(stage='rollouts',step=step,success=result['rollouts'][str(step)]['success'])),flush=True)
    for step,state in states.items():
        agent=state.base.agent;spec=agent.config['onestep_specification']
        @jax.jit
        def fit_metrics(x,key):
            ob,z=flat.lean.bank(agent,x,key,9);aa=flat.lean.flow(agent,ob,z,10)
            q=agent.network.select('critic')(ob,actions=aa).min(0)
            target=flat.gather(aa,q.argmax(0))
            prediction=jnp.clip(state.actor(x,flat.packed_input(z,spec)),-1,1)
            pq=agent.network.select('critic')(x,actions=prediction).min(0)
            return jnp.array([jnp.mean((prediction-target)**2),jnp.mean(q.max(0)-pq),
                jnp.mean(jnp.abs(prediction-target)),jnp.mean(jnp.abs(prediction)>=1)])
        result['crossed_fit'][str(step)]={}
        for name,observations in occupancies.items():
            idx=np.linspace(0,len(observations)-1,min(512,len(observations))).astype(int)
            x=jnp.asarray(observations[idx]);rows=[]
            for draw in range(4):
                rows.append(np.asarray(fit_metrics(x,jax.random.fold_in(jax.random.PRNGKey(49601),draw))))
            mse,regret,mae,saturation=np.stack(rows).mean(0).tolist()
            result['crossed_fit'][str(step)][name]=dict(states=len(x),matched_bank_draws=4,
                paired_teacher_action_MSE=mse,predicted_Q_regret=regret,
                action_MAE=mae,actor_endpoint_fraction=saturation)
        write(a.output/'status.json',result)
    result.update(status='complete',completed=time.time(),evaluation_episodes=30,
        caveat='Each model is compared with its own same-checkpoint teacher. Visited-state changes and teacher drift are not uniquely causal; predicted Q is not an oracle.')
    write(a.output/'result.json',result);write(a.output/'status.json',result)
    print(json.dumps(dict(status='complete',crossed_fit=result['crossed_fit'])),flush=True)


if __name__=='__main__':main()
