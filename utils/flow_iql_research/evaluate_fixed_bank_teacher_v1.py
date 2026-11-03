"""Read-only deterministic-teacher feasibility, NOT one-step deployment.

One common bank seed is chosen before outcomes. The fixed32 bank contains
the fixed9 bank. Policy still evaluates Flow/Q; this alone is not goal success.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def make_policy(template,bank):
    def policy(parameters,observations,key):
        agent=template.base.agent.replace(network=template.base.agent.network.replace(params=parameters['base']))
        broadcast=jnp.broadcast_to(observations,(len(bank),*observations.shape))
        actions=flat.lean.flow(agent,broadcast,bank,10)
        q=agent.network.select('critic')(broadcast,actions=actions).min(0)
        return actions[q.argmax()]
    return jax.jit(policy,backend='cpu')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and len(os.sched_getaffinity(0))==6 and all(d.platform=='cpu'for d in jax.devices())
    args.output.mkdir(parents=True)
    cp=args.teacher_run/'checkpoint_1000000.pkl';digest=sha(cp);conf=json.loads((args.teacher_run/'config.json').read_text())
    for p,d in conf['source_sha256'].items():assert sha(p)==d,p
    with(args.teacher_run/'actor_1000000.pkl').open('rb')as f:artifact=pickle.load(f)
    dim=artifact['metadata']['action_dim'];obs_dim=artifact['metadata']['observation_dim']
    assert artifact['checkpoint_sha256']==digest and conf['identity']['config']['encoder']is None
    cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=flat.create(conf['identity']['seed'],jnp.zeros((1,obs_dim)),jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with cp.open('rb')as f:saved=pickle.load(f)
    assert saved['step']==1000000 and saved['identity']==conf['identity']and saved['source_sha256']==conf['source_sha256']
    teacher=flax.serialization.from_state_dict(template,saved['state']);assert int(teacher.base.q_updates)==int(teacher.actor_updates)==1000000
    payload=flat.lean.policy_parameters(teacher.base)
    bank_key=jax.random.fold_in(jax.random.PRNGKey(conf['identity']['seed']),49501)
    bank32=jax.random.normal(bank_key,(32,dim));bank_path=args.output/'fixed_noise_bank32_v1.npy';np.save(bank_path,np.asarray(bank32))
    old_eval=args.teacher_run.parent.parent/'analysis/evaluations_v1'/f'{args.teacher_run.name}_1000000_teacher_n50.json'
    baseline=json.loads(old_eval.read_text());assert baseline['checkpoint_sha256']==digest and baseline['n_episodes']==50 and baseline['first_episode_seed']==42
    result=dict(time=time.time(),status='evaluating',environment=conf['identity']['environment'],teacher_run=str(args.teacher_run),teacher_checkpoint_sha256=digest,
        original_stochastic_best9_CPU50=baseline['success_rate'],original_evaluation=str(old_eval),original_evaluation_sha256=sha(old_eval),
        bank_seed='fold_in(training_seed,49501)',bank32_sha256=sha(bank_path),nested_fixed9_prefix_of_fixed32=True,
        modes=[9,32],episodes_per_mode=50,episode_seeds=list(range(42,92)),
        teacher_actor_or_GPU_optimizer_updates=0,evaluation_used_for_learning=False,mode_seed_not_tuned_per_environment=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,*flat.source_paths()]},evaluations={},
        scope='Teacher determinization screen only; still Flow10 and Q reranking at every action. Not a one-step actor, joint training, last3/nine-task/freshseed or speed result.')
    write(args.output/'status.json',result)
    for count in (9,32):
        policy=make_policy(teacher,bank32[:count])
        a=policy(payload,jnp.zeros(obs_dim),jax.random.PRNGKey(1));b=policy(payload,jnp.zeros(obs_dim),jax.random.PRNGKey(2))
        assert np.array_equal(np.asarray(a),np.asarray(b))and np.isfinite(np.asarray(a)).all()
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        env=ogbench.make_env_and_datasets(conf['identity']['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,1000000,50);env.close()
        evaluated.update(time=time.time(),mode='fixed_bank_'+str(count)+'_teacher',environment=conf['identity']['environment'],protocol='private_fixed_noise_teacher_CPU50_v1',
            teacher_checkpoint_sha256=digest,noise_bank_sha256=sha(bank_path),reset_metadata=reset,no_optimizer_updates=True,counts_toward_one_step_actor_or_goal=False)
        ep=args.output/(f'fixed{count}_teacher_CPU50_v1.json');write(ep,evaluated)
        result['evaluations'][str(count)]=dict(success=evaluated['success_rate'],path=str(ep),sha256=sha(ep),actions_independent_of_policy_PRNG_key=True)
        write(args.output/'status.json',result);print(json.dumps(dict(candidates=count,success=evaluated['success_rate'],stochastic_best9=baseline['success_rate'])),flush=True)
    assert sha(cp)==digest
    result.update(status='complete',completed=time.time(),total_evaluation_episodes=100)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
