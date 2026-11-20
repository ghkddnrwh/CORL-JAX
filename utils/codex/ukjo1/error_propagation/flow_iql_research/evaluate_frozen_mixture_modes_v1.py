"""Exploratory deployment-only modes from a frozen CPU mixture diagnostic.

The primary stochastic result is preserved. Both global alternatives emit
one action from one parameter-network forward, with no critic or ODE. They
change the teacher-matching distribution and are not joint/last3 evidence.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import jax
import jax.numpy as jnp
import numpy as np
import ogbench


MODES=('sample_component_mean','highest_weight_component_mean')


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study-case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True)
    parent_path=args.study_case/'result.json';parent=json.loads(parent_path.read_text());assert parent['status']=='complete'
    for p,digest in parent['source_sha256'].items():assert sha(p)==digest,p
    entry=parent['actors']['mixture9_NLL'];artifact_path=Path(entry['private_actor_path'])
    digest=sha(artifact_path);assert digest==entry['private_actor_sha256']
    with artifact_path.open('rb') as stream:artifact=pickle.load(stream)
    assert artifact['schema']=='private_frozen_teacher_student_v1' and artifact['actor_Adam_updates']==20000
    assert artifact['teacher_optimizer_updates_performed']==0
    metadata=artifact['metadata'];assert metadata['components']==9 and metadata['generated_candidate_actions']==0 and metadata['actor_forward_passes']==1
    definition=mixture.MixtureActor(tuple(metadata['hidden_dims']),metadata['action_dim'],metadata['components'],metadata['std_floor'])
    payload=artifact['actor_params'];assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(payload))
    result=dict(time=time.time(),schema='frozen_mixture_deployment_modes_v1',status='running',
        environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        actor_artifact=str(artifact_path),actor_artifact_sha256=digest,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(mixture.__file__).resolve()):sha(mixture.__file__)},
        primary_stochastic_CPU50=entry['final_CPU50_success'],paired_best9_teacher_CPU50=parent['paired_teacher_CPU50'],
        modes=list(MODES),optimizer_updates_performed=0,teacher_updates=0,registered_GPU_updates=0,
        episode_seeds=list(range(42,92)),CPU_affinity=sorted(os.sched_getaffinity(0)),
        scope='Secondary exploratory modes of a frozen20KCPU actor, not joint/last3/9task. Means and categorical choice change the stochastic teacher law. No global mixture-mode/MAP claim for highest-weight component.',evaluations={})
    write(args.output/'status.json',result)
    for mode in MODES:
        def policy(params,observation,key):
            p=definition.apply({'params':params},observation,jnp.zeros((*observation.shape[:-1],metadata['action_dim']+1)),return_parameters=True)
            if mode=='sample_component_mean':
                action_key,_=jax.random.split(key)
                latent=jax.random.normal(action_key,(*observation.shape[:-1],metadata['action_dim']+1))
                return mixture.sample_one(dict(p,scales=jnp.zeros_like(p['scales'])),latent)
            index=p['logits'].argmax(-1)
            idx=jnp.broadcast_to(index[...,None,None],(*index.shape,1,metadata['action_dim']))
            return jnp.clip(jnp.take_along_axis(p['means'],idx,axis=-2)[...,0,:],-1,1)
        policy=jax.jit(policy,backend='cpu')
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        env=ogbench.make_env_and_datasets(parent['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,20000,50);env.close()
        evaluated.update(time=time.time(),mode=mode,environment=parent['environment'],
            protocol='secondary_frozen_mixture_mode_CPU50_v1',actor_artifact_sha256=digest,
            no_optimizer_updates=True,reset_metadata=reset,counts_toward_goal_last3=False,
            teacher_checkpoint_step=1000000,training_mode='CPU_frozen_teacher_diagnostic_only')
        write(args.output/(mode+'_CPU50_v1.json'),evaluated)
        result['evaluations'][mode]=dict(success=evaluated['success_rate'],path=str(args.output/(mode+'_CPU50_v1.json')))
        write(args.output/'status.json',result)
        print(json.dumps(dict(mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(artifact_path)==digest
    result.update(status='complete',completed=time.time())
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
