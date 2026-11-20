"""Broad replay, frozen-teacher extraction feasibility; NOT joint training.

Use65536unique replay observations, fit three disposable students20K updates
each, and evaluate only their final actor over50episodes. No evaluation
interaction enters training. Original checkpoint and GPUqueue remain read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

# Existing legacyJAX/SciPy compatibility shim must precede optax.
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import energy_onestep_deploy_core_v1 as energy
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


METHODS=('bank_MSE','energy2','mixture9_NLL')
TRAIN=65536
HELDOUT=4096
UPDATES=20000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True);started=time.time()
    conf_path=args.teacher_run/'config.json';conf=json.loads(conf_path.read_text());identity=conf['identity']
    assert identity['recipe']=='fqln_a03' and identity['config']['encoder'] is None
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.teacher_run/'checkpoint_1000000.pkl';checkpoint_sha=sha(checkpoint)
    with checkpoint.open('rb') as stream:saved=pickle.load(stream)
    assert saved['step']==1000000 and saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
    env,data,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    assert len(data['actions'])>=TRAIN+HELDOUT
    random=np.random.RandomState(34201);selected=[];seen=set();attempted=0
    while len(selected)<TRAIN+HELDOUT:
        attempted+=4096
        assert attempted<100*(TRAIN+HELDOUT),'Insufficient distinct replay observations; do not loop indefinitely.'
        for index in random.choice(len(data['actions']),size=4096,replace=False):
            observation=np.asarray(data['next_observations'][index],np.float32)
            digest=hashlib.sha256(observation.tobytes()).digest()
            if digest not in seen:
                seen.add(digest);selected.append(int(index))
                if len(selected)==TRAIN+HELDOUT:break
    observations=np.asarray(data['next_observations'][selected],np.float32)
    action_dim=data['actions'].shape[-1]
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=flat.create(identity['seed'],jnp.asarray(observations[:1]),jnp.zeros((1,action_dim)),cfg,identity['recipe'])
    teacher=flax.serialization.from_state_dict(template,saved['state'])
    assert int(teacher.base.q_updates)==int(teacher.actor_updates)==1000000 and int(teacher.base.v_updates)==0
    del data,saved,template
    agent=teacher.base.agent
    sources=[Path(__file__),Path(mixture.__file__)]+energy.source_paths()
    result=dict(schema='broad_frozen_teacher_extraction_feasibility_v1',started=started,
        environment=identity['environment'],teacher_run=str(args.teacher_run),teacher_checkpoint=str(checkpoint),teacher_checkpoint_sha256=checkpoint_sha,
        teacher_Q_updates_loaded=1000000,teacher_optimizer_updates_performed=0,registered_GPU_updates_performed=0,
        source_sha256={str(p.resolve()):sha(p) for p in sources},methods=list(METHODS),
        train_unique_observations=TRAIN,heldout_unique_observations=HELDOUT,teacher_draws_per_training_observation=1,
        exact_observation_disjoint=True,trajectory_disjoint=False,dataset_state_seed=34201,teacher_label_seed=34202,
        train_updates_per_method=UPDATES,batch_size=256,learning_rate=.0003,minibatch_noise_seed=34211,
        actor_initialization_seeds=dict(bank_MSE=0,energy2=0,mixture9_NLL=34210),
        environment_eval_seed=42,environment_eval_episodes_per_method=50,
        evaluation_interactions_used_for_learning=False,
        policy_selection='Allthreepredeclaredmethods;onlyfinal20Kactorisrolledout,nocheckpointpeakselection.',
        goal_protocol=False,GPU_queue_change=False,
        scope='Frozen1Mteacher, CPUstudents20Kupdates only, two hard tasks. Not a jointly trained algorithm, last3, nine-task or fresh-seed result. Private actor exports are diagnostics, never canonical1Mmodels.',actors={})
    write(args.output/'status.json',result)

    @jax.jit
    def label(obs,key):
        broadcast,noises=flat.lean.bank(agent,obs,key,9)
        proposals=flat.lean.flow(agent,broadcast,noises,10)
        q=agent.network.select('critic')(broadcast,actions=proposals).min(0)
        return flat.packed_input(noises,{'input':'bank'}),flat.gather(proposals,q.argmax(0))

    packed_chunks,target_chunks=[],[]
    for chunk,offset in enumerate(range(0,len(observations),256)):
        packed,target=jax.device_get(label(jnp.asarray(observations[offset:offset+256]),jax.random.fold_in(jax.random.PRNGKey(34202),chunk)))
        packed_chunks.append(packed);target_chunks.append(target)
        if chunk%32==0:print(json.dumps(dict(stage='cache',observations=offset+256,elapsed_seconds=time.time()-started)),flush=True)
    packed=np.concatenate(packed_chunks);targets=np.concatenate(target_chunks)
    # Four independent original best9 draws for each of256heldout states.
    distribution_targets=[]
    for draw in range(4):
        _,target=label(jnp.asarray(observations[TRAIN:TRAIN+256]),jax.random.fold_in(jax.random.PRNGKey(34203),draw))
        distribution_targets.append(np.asarray(target))
    distribution_targets=np.stack(distribution_targets)
    assert np.isfinite(targets).all() and np.isfinite(packed).all()
    cache=args.output/'frozen_replay_pairs_v1.npz'
    np.savez_compressed(cache,observations=observations,packed_noises=packed,targets=targets,
        selected_replay_indices=np.asarray(selected),distribution_eval_targets=distribution_targets,train_size=TRAIN)
    result['pair_cache_sha256']=sha(cache)
    obs,z,y=(jnp.asarray(a) for a in (observations,packed,targets))
    eval_obs=obs[TRAIN:TRAIN+256];eval_y=jnp.asarray(distribution_targets)
    baseline_path=args.teacher_run.parent.parent/'analysis/evaluations_v1'/f'{args.teacher_run.name}_1000000_teacher_n50.json'
    baseline=json.loads(baseline_path.read_text());assert baseline['checkpoint_sha256']==checkpoint_sha and baseline['n_episodes']==50 and baseline['first_episode_seed']==42
    result.update(paired_teacher_CPU50=baseline['success_rate'],paired_teacher_raw_evaluation=str(baseline_path),paired_teacher_raw_evaluation_sha256=sha(baseline_path))
    write(args.output/'status.json',result)

    for method in METHODS:
        if method=='mixture9_NLL':
            definition=mixture.MixtureActor(tuple(cfg['actor_hidden_dims']),action_dim,9)
            parameters=definition.init(jax.random.PRNGKey(34210),obs[:1],jnp.zeros((1,action_dim+1)))['params']
            actor=TrainState.create(definition,parameters,optax.adam(.0003))
            metadata=dict(schema='private_frozen_mixture_actor_v1',hidden_dims=list(cfg['actor_hidden_dims']),components=9,std_floor=.001,
                observation_dim=obs.shape[-1],action_dim=action_dim,noise_dim=action_dim+1,
                actor_forward_passes=1,generated_candidate_actions=0,requires_critic=False,requires_BC_flow=False,flow_integration_steps=0)
        else:
            module,recipe=(flat,'bank_bc') if method=='bank_MSE' else (energy,'energy_bc')
            student_cfg=module.config_for(identity['environment'],recipe)
            actor=module.create(0,obs[:1],jnp.zeros((1,action_dim)),student_cfg,recipe).actor
            metadata=module.actor_metadata(student_cfg,obs.shape[-1],action_dim)
        record=dict(metadata=metadata,parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(actor.params)),snapshots=[])
        result['actors'][method]=record

        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key,noise_key=jax.random.split(key,3)
                indices=jax.random.randint(index_key,(256,),0,TRAIN)
                b,t,n=obs[indices],y[indices],z[indices]
                draws=jax.random.normal(noise_key,(2,256,action_dim))
                def loss(params):
                    if method=='bank_MSE':value=jnp.mean((actor(b,n,params=params)-t)**2)
                    elif method=='energy2':
                        first=actor(b,draws[0],params=params);second=actor(b,draws[1],params=params)
                        value=.5*jnp.mean(energy.norm(first-t)+energy.norm(second-t)-energy.norm(first-second))
                    else:
                        p=actor(b,jnp.zeros((256,action_dim+1)),params=params,return_parameters=True)
                        value=-mixture.log_probability(p,t).mean()/action_dim
                    return value,dict(objective=value)
                actor,_=actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)

        @jax.jit
        def metrics(actor):
            broadcast=jnp.broadcast_to(eval_obs,(32,*eval_obs.shape))
            if method=='mixture9_NLL':
                noise=jax.random.normal(jax.random.PRNGKey(34212),(32,256,action_dim+1))
                actions=actor(broadcast,noise)
                p=actor(obs[TRAIN:],jnp.zeros((HELDOUT,action_dim+1)),return_parameters=True)
                extra=dict(heldout_NLL_per_dimension=-mixture.log_probability(p,y[TRAIN:]).mean()/action_dim,
                           mean_scale=p['scales'].mean())
            else:
                noise=jax.random.normal(jax.random.PRNGKey(34212),(9,32,256,action_dim))
                inputs=flat.packed_input(noise,{'input':'bank' if method=='bank_MSE' else 'single'})
                actions=jnp.clip(actor(broadcast,inputs),-1,1)
                inputs=z[TRAIN:] if method=='bank_MSE' else jax.random.normal(jax.random.PRNGKey(34213),(HELDOUT,action_dim))
                extra=dict(heldout_action_MSE=jnp.mean((actor(obs[TRAIN:],inputs)-y[TRAIN:])**2))
            cross=energy.norm(actions[:,None]-eval_y[None,:]).mean()
            xx=energy.norm(actions[:,None]-actions[None,:]).sum()/(32*31*256)
            yy=energy.norm(eval_y[:,None]-eval_y[None,:]).sum()/(4*3*256)
            return dict(heldout_energy_distance_Ustat=2*cross-xx-yy,
                student_to_teacher_variance_ratio=jnp.var(actions,axis=0,ddof=1).mean()/jnp.maximum(jnp.var(eval_y,axis=0,ddof=1).mean(),1e-12),
                action_endpoint_fraction=jnp.mean((jnp.abs(actions)>=1).astype(jnp.float32)),**extra)
        previous,key=0,jax.random.PRNGKey(34211)
        for updates in (0,1000,5000,UPDATES):
            if updates:actor,key=fit(actor,key,updates-previous)
            assert int(actor.opt_state[0].count)==updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(actor.params))
            snapshot=dict(Adam_updates=updates,elapsed_seconds=time.time()-started,**{k:float(v) for k,v in jax.device_get(metrics(actor)).items()})
            record['snapshots'].append(snapshot);write(args.output/'status.json',result)
            print(json.dumps(dict(stage='fit',method=method,**snapshot)),flush=True);previous=updates
        payload=flax.serialization.to_state_dict(jax.device_get(actor.params))
        artifact_path=args.output/(method+'_private_actor_20000.pkl')
        with artifact_path.open('xb') as stream:pickle.dump(dict(schema='private_frozen_teacher_student_v1',training_mode='CPU_frozen_teacher_diagnostic_only',
            method=method,metadata=metadata,actor_params=payload,actor_Adam_updates=UPDATES,teacher_optimizer_updates_performed=0,
            teacher_checkpoint_sha256=checkpoint_sha,source_sha256=result['source_sha256']),stream,pickle.HIGHEST_PROTOCOL)
        if method=='mixture9_NLL':
            def policy(parameters,observation,key):
                action_key,_=jax.random.split(key)
                latent=jax.random.normal(action_key,(*observation.shape[:-1],action_dim+1))
                return definition.apply({'params':parameters},observation,latent)
            policy=jax.jit(policy,backend='cpu')
        else:policy=flat.make_actor_policy(metadata,'cpu')
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        environment=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
        reset_metadata=prepare_seeded_action_space(environment,42)
        evaluated=evaluate(namespace,environment,policy,UPDATES,50)
        environment.close()
        evaluated.update(method=method,environment=identity['environment'],time=time.time(),
            protocol='private_frozen_teacher_student_CPU50_v1',teacher_checkpoint_step=1000000,
            teacher_checkpoint_sha256=checkpoint_sha,actor_optimizer_updates_before_evaluation=UPDATES,
            optimizer_updates_during_evaluation=0,training_mode='frozen_teacher_CPU_diagnostic_only',
            actor_artifact_sha256=sha(artifact_path),reset_metadata=reset_metadata,
            counts_toward_goal_last3=False,counts_toward_joint_performance=False)
        evaluation_path=args.output/(method+'_CPU50_v1.json');write(evaluation_path,evaluated)
        record.update(private_actor_path=str(artifact_path),private_actor_sha256=sha(artifact_path),
            evaluation_path=str(evaluation_path),evaluation_sha256=sha(evaluation_path),final_CPU50_success=evaluated['success_rate'])
        write(args.output/'status.json',result)
        print(json.dumps(dict(stage='CPU50',method=method,success=evaluated['success_rate'],teacher=baseline['success_rate'])),flush=True)
    assert sha(checkpoint)==checkpoint_sha and sha(cache)==result['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_actor_Adam_updates=3*UPDATES,
                  evaluation_episodes=150,teacher_checkpoint_unchanged=True)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
