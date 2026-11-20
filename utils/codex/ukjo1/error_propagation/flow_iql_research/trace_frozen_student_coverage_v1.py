"""Read-only rollout coverage diagnosis. Evaluation states never train a policy."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import energy_onestep_deploy_core_v1 as energy
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from scipy.spatial import cKDTree


METHODS = ('teacher', 'bank_MSE', 'energy2', 'mixture9_NLL')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path = args.study_case / 'result.json'
    parent = json.loads(parent_path.read_text())
    assert parent['status'] == 'complete'
    teacher_run = Path(parent['teacher_run'])
    cp = Path(parent['teacher_checkpoint'])
    assert sha(cp) == parent['teacher_checkpoint_sha256']
    conf = json.loads((teacher_run / 'config.json').read_text())
    for p, digest in conf['source_sha256'].items():
        assert sha(p) == digest
    cfg = dict(conf['identity']['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    cache_path = args.study_case / 'frozen_replay_pairs_v1.npz'
    assert sha(cache_path) == parent['pair_cache_sha256']
    with np.load(cache_path) as cache:
        train = int(cache['train_size'])
        replay = cache['observations'][:train].copy()
        heldout = cache['observations'][train:train+512].copy()
    action_dim = cfg['action_dim']
    template = flat.create(conf['identity']['seed'], jnp.asarray(replay[:1]), jnp.zeros((1, action_dim)), cfg, conf['identity']['recipe'])
    with cp.open('rb') as stream:
        saved = pickle.load(stream)
    teacher = flax.serialization.from_state_dict(template, saved['state'])
    assert int(teacher.actor_updates) == int(teacher.base.q_updates) == 1000000
    teacher_policy = flat.make_teacher_policy(teacher)
    teacher_payload = flat.lean.policy_parameters(teacher.base)
    policies = {'teacher': (teacher_policy, teacher_payload)}
    artifacts = {}
    mixture_definition = None
    for method in METHODS[1:]:
        entry = parent['actors'][method]
        path = Path(entry['private_actor_path'])
        assert sha(path) == entry['private_actor_sha256']
        with path.open('rb') as stream:
            artifact = pickle.load(stream)
        assert artifact['actor_Adam_updates'] == 20000 and artifact['teacher_optimizer_updates_performed'] == 0
        params, metadata = artifact['actor_params'], artifact['metadata']
        if method == 'mixture9_NLL':
            mixture_definition = mixture.MixtureActor(tuple(metadata['hidden_dims']), action_dim, 9, .001)
            def mdn_policy(parameters, observations, key):
                action_key, _ = jax.random.split(key)
                latent = jax.random.normal(action_key, (*observations.shape[:-1], action_dim+1))
                return mixture_definition.apply({'params': parameters}, observations, latent)
            policy = jax.jit(mdn_policy, backend='cpu')
        else:
            policy = flat.make_actor_policy(metadata, 'cpu')
        policies[method] = policy, params
        artifacts[method] = dict(path=str(path), sha256=sha(path))
    result = dict(time=time.time(),status='running',environment=parent['environment'],
        schema='frozen_student_rollout_coverage_v1',parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        teacher_checkpoint_sha256=sha(cp),private_artifacts=artifacts,
        source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__,mixture.__file__,*energy.source_paths()]},
        episode_seeds=list(range(42,52)),sample_every_steps=20,optimizer_updates_performed=0,
        evaluation_interactions_used_for_learning=False,registered_GPU_updates=0,
        support_proxy='Exact Euclidean nearest-neighbor distance to65536cached training states, divided featurewise by max(replay_std,.001) and by sqrt(observation_dim). This is not an OOD oracle.',
        scope='Exploratory frozen20KCPUstudents/1Mteacher trace, not joint training, goal last3, speed or causal identification.',rollouts={})
    write(args.output / 'status.json', result)
    traces = {}
    for method in METHODS:
        policy, params = policies[method]
        env = ogbench.make_env_and_datasets(parent['environment'], env_only=True)
        prepare_seeded_action_space(env,42)
        samples, episodes = [], []
        for seed in range(42,52):
            np.random.seed(seed);random.seed(seed);env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            key = jax.random.PRNGKey(seed)
            steps, success, reward_sum = 0, 0., 0.
            mismatch, transitions, previous_buttons = [], 0, None
            while True:
                key, action_key = jax.random.split(key)
                act = np.asarray(policy(params,np.asarray(obs,np.float32),action_key)).copy()
                if steps % 20 == 0:
                    samples.append((np.asarray(obs,np.float32).copy(),seed,steps))
                obs,reward,terminated,truncated,info = env.step(act)
                reward_sum += float(reward);success=max(success,float(info.get('success',0.)));steps+=1
                bare=env.unwrapped
                if hasattr(bare,'_cur_button_states'):
                    buttons=np.asarray(bare._cur_button_states).copy()
                    mismatch.append(int(np.count_nonzero(buttons!=bare._target_button_states)))
                    if previous_buttons is not None:transitions+=int(not np.array_equal(previous_buttons,buttons))
                    previous_buttons=buttons
                if terminated or truncated:break
            row=dict(seed=seed,success=success,reward=reward_sum,length=steps)
            if mismatch:row.update(min_button_mismatches=min(mismatch),final_button_mismatches=mismatch[-1],button_state_change_events=transitions)
            episodes.append(row)
        env.close()
        if method!='teacher':
            prior=json.loads(Path(parent['actors'][method]['evaluation_path']).read_text())
        else:prior=json.loads(Path(parent['paired_teacher_raw_evaluation']).read_text())
        for field,prior_key in [('success','episode_successes'),('reward','episode_rewards'),('length','episode_lengths')]:
            assert [x[field] for x in episodes]==prior[prior_key][:10],(method,field,'Trace changed policy/reset behavior')
        observations=np.stack([x[0] for x in samples])
        traces[method]=observations
        np.savez_compressed(args.output/(method+'_visited_states_v1.npz'),observations=observations,
            episode_seeds=np.asarray([x[1] for x in samples]),episode_steps=np.asarray([x[2] for x in samples]))
        result['rollouts'][method]=dict(episodes=episodes,sampled_states=len(samples),success_rate=float(np.mean([e['success']for e in episodes])),
            first10_previous_CPU50_arrays_exact=True)
        write(args.output/'status.json',result)
        print(json.dumps(dict(stage='rollout',method=method,success=result['rollouts'][method]['success_rate'],states=len(samples))),flush=True)
    scale=np.maximum(replay.std(0),.001)
    tree=cKDTree(replay.astype(np.float64)/scale)
    result['support']={}
    for name,obs in dict(heldout_replay=heldout,**traces).items():
        distances,_=tree.query(obs.astype(np.float64)/scale,k=1,workers=1)
        distances=distances/np.sqrt(replay.shape[-1])
        result['support'][name]=dict(mean=float(distances.mean()),median=float(np.median(distances)),p90=float(np.quantile(distances,.9)),max=float(distances.max()))
    result['student_fit_by_state_distribution']={}
    teacher_batch=jax.jit(jax.vmap(teacher_policy,in_axes=(None,0,0)))
    for name,observations in dict(heldout_replay=heldout,**traces).items():
        # Same deterministic subsampling for all three students; up to512states per occupancy.
        idx=np.linspace(0,len(observations)-1,min(512,len(observations))).astype(int)
        obs=jnp.asarray(observations[idx]);n=len(obs)
        labels=[]
        for draw in range(4):
            keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(49201),draw),n)
            labels.append(teacher_batch(teacher_payload,obs,keys))
        target=jnp.stack(labels)
        yy=energy.norm(target[:,None]-target[None,:]).sum()/(4*3*n)
        result['student_fit_by_state_distribution'][name]={}
        for method in METHODS[1:]:
            policy,params=policies[method]
            batch_policy=jax.jit(jax.vmap(policy,in_axes=(None,0,0)))
            draws=[]
            for draw in range(8):
                keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(49202),draw),n)
                draws.append(batch_policy(params,obs,keys))
            actions=jnp.stack(draws)
            cross=energy.norm(actions[:,None]-target[None,:]).mean()
            xx=energy.norm(actions[:,None]-actions[None,:]).sum()/(8*7*n)
            row=dict(states=n,energy_distance_Ustat=float(2*cross-xx-yy),independent_sample_cross_MSE=float(jnp.mean((actions[:,None]-target[None,:])**2)),
                teacher_variance=float(jnp.var(target,axis=0,ddof=1).mean()),student_variance=float(jnp.var(actions,axis=0,ddof=1).mean()))
            if method=='mixture9_NLL':
                p=mixture_definition.apply({'params':params},obs,jnp.zeros((n,action_dim+1)),return_parameters=True)
                row['teacher_NLL_per_dimension']=float(-mixture.log_probability(p,target).mean()/action_dim)
            result['student_fit_by_state_distribution'][name][method]=row
        write(args.output/'status.json',result)
        print(json.dumps(dict(stage='occupancy_metrics',occupancy=name,states=n)),flush=True)
    assert sha(cp)==parent['teacher_checkpoint_sha256'] and sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),total_evaluation_episodes=40)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':
    main()
