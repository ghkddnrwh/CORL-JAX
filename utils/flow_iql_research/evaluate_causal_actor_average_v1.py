"""Private CPU test of a causal, uniform late-checkpoint actor weight mean.

At 800K use theta800; at 900K mean(theta800,theta900); at 1M mean all3.
Never use future weights, average actions, average critics, or select a peak.
The underlying actor was jointly trained; no additional SGD is performed.
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

import jax
import numpy as np
import ogbench


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def mean_parameters(trees):
    assert trees
    if len(trees)==1:
        return jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),trees[0])
    return jax.tree_util.tree_map(
        lambda *xs:np.mean(np.stack([np.asarray(x)for x in xs]),axis=0,dtype=np.float32),*trees)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu'for d in jax.devices())
    manifest=json.loads(args.manifest.read_text())
    for p,h in manifest['source_sha256'].items():
        assert sha(p)==h,p
    case=manifest['cases'][args.case]
    run=Path(case['run']);conf=json.loads((run/'config.json').read_text())
    assert sha(run/'config.json')==case['config_sha256']
    assert conf['identity']['recipe']=='bank_bc'and conf['identity']['steps']==1000000
    for p,h in conf['source_sha256'].items():
        assert sha(p)==h,p
    actors={};references={}
    for step in case['steps']:
        entry=case['artifacts'][str(step)]
        assert sha(entry['actor'])==entry['actor_sha256']
        assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
        assert sha(entry['primary_evaluation'])==entry['primary_evaluation_sha256']
        with Path(entry['actor']).open('rb')as f:actors[step]=pickle.load(f)
        a=actors[step]
        assert a['actor_updates']==step and a['Q_updates']==step
        assert a['checkpoint_sha256']==entry['checkpoint_sha256']
        references[step]=json.loads(Path(entry['primary_evaluation']).read_text())
        assert references[step]['n_episodes']==50 and references[step]['first_episode_seed']==42
    metadata=actors[800000]['metadata']
    assert all(x['metadata']==metadata for x in actors.values())
    assert metadata['actor_forward_passes']==1 and metadata['generated_candidate_actions']==0
    policy=flat.make_actor_policy(metadata)
    args.output.mkdir(parents=True)
    result=dict(time=time.time(),status='evaluating',environment=conf['identity']['environment'],
        training_seed=conf['identity']['seed'],run=str(run),method='causal_uniform_late_actor_weight_mean',
        manifest_sha256=sha(args.manifest),source_sha256=manifest['source_sha256'],
        steps=case['steps'],rule='Prefix only:800K[800],900K[800,900],1M[800,900,1M]; uniform parameter means.',
        training_actors_jointly_trained_with_unchanged_lean=True,new_optimizer_updates=0,
        only_actor_parameters_averaged=True,no_action_ensemble=True,future_weights_used_for_earlier_outputs=False,
        no_environment_specific_settings_or_peak_selection=True,metadata=metadata,evaluations={},
        scope='Private exploratory CPU50 derived policy; original learners/artifacts unchanged. '
              'Not native final selection, fresh-seed confirmation, or an isolated speed result.')
    write(args.output/'status.json',result)
    for step in case['steps']:
        prefix=sorted(t for t in actors if t<=step)
        assert prefix==[t for t in [800000,900000,1000000]if t<=step]
        params=mean_parameters([actors[t]['actor_params']for t in prefix])
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(params))
        if step==800000:
            assert all(np.asarray(a).tobytes()==np.asarray(b).tobytes()for a,b in zip(
                jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(actors[step]['actor_params'])))
        artifact=args.output/f'causal_average_actor_{step}_private_v1.pkl'
        with artifact.open('xb')as f:
            pickle.dump(dict(schema='private_causal_actor_average_v1',metadata=metadata,actor_params=params,
                source_training_step=step,averaged_checkpoint_steps=prefix,
                source_actor_sha256={str(t):case['artifacts'][str(t)]['actor_sha256']for t in prefix},
                new_optimizer_updates=0,source_sha256=result['source_sha256']),f,pickle.HIGHEST_PROTOCOL)
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
        env=ogbench.make_env_and_datasets(result['environment'],env_only=True)
        reset=prepare_seeded_action_space(env,42)
        evaluated=evaluate(namespace,env,policy,step,50);env.close()
        evaluated.update(time=time.time(),protocol='private_causal_actor_average_CPU50_v1',
            method=result['method'],averaged_checkpoint_steps=prefix,
            actor_artifact_sha256=sha(artifact),reset_metadata=reset,
            no_optimizer_updates=True,evaluation_used_for_learning=False,counts_toward_native_goal=False,
            learned_model_trees_loaded=['averaged_actor_params'],actor_forward_passes=1,
            requires_critic=False,requires_BC_flow=False,generated_candidate_actions=0)
        if step==800000:
            for key in ['episode_successes','episode_rewards','episode_lengths']:
                assert evaluated[key]==references[step][key],('800Kcontrol failed',key)
        ep=args.output/f'causal_average_{step}_CPU50_v1.json';write(ep,evaluated)
        result['evaluations'][str(step)]=dict(success=evaluated['success_rate'],
            original_actor_success=references[step]['success_rate'],averaged_steps=prefix,
            path=str(ep),sha256=sha(ep),artifact=str(artifact),artifact_sha256=sha(artifact),
            original_800K_control_exactly_reproduced=step==800000)
        write(args.output/'status.json',result)
        print(json.dumps(dict(step=step,averaged_steps=prefix,success=evaluated['success_rate'],
                             original=references[step]['success_rate'])),flush=True)
    result.update(status='complete',completed=time.time(),episodes=50*len(case['steps']))
    if case['steps']==[800000,900000,1000000]:
        result['last3_mean']=float(np.mean([x['success']for x in result['evaluations'].values()]))
        result['original_last3_mean']=float(np.mean([x['original_actor_success']for x in result['evaluations'].values()]))
    for entry in case['artifacts'].values():
        assert sha(entry['actor'])==entry['actor_sha256']
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':
    main()
