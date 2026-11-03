"""Auxiliary frozen best9 min/mean-Q control; never primary actor evidence."""
import argparse
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.onestep_implementation_registry_v1 import implementation
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def make_policy(state,aggregation):
    assert aggregation in ('min','mean')
    assert state.base.agent.config['encoder']is None
    assert state.base.agent.config['num_samples']==9 and state.base.agent.config['flow_steps']==10
    assert 'modules_actor_flow'in state.base.agent.network.params
    def policy(parameters,observations,key):
        assert observations.ndim==1
        agent=state.base.agent.replace(network=state.base.agent.network.replace(params=parameters['base']))
        action_seed,_=jax.random.split(key)
        actions=jax.random.normal(action_seed,(*observations.shape[:-1],agent.config['num_samples'],agent.config['action_dim']))
        repeated=jnp.repeat(jnp.expand_dims(observations,0),agent.config['num_samples'],axis=0)
        for index in range(agent.config['flow_steps']):
            times=jnp.full((*observations.shape[:-1],agent.config['num_samples'],1),index/agent.config['flow_steps'])
            velocity=agent.network.select('actor_flow')(repeated,actions,times,is_encoded=True)
            actions=actions+velocity/agent.config['flow_steps']
        actions=jnp.clip(actions,-1,1)
        heads=agent.network.select('critic')(repeated,actions=actions)
        scores=heads.min(axis=0)if aggregation=='min'else heads.mean(axis=0)
        return actions[jnp.argmax(scores)]
    return jax.jit(policy,backend='cpu')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--aggregation',choices=('min','mean'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();assert not args.output.exists()and all(device.platform=='cpu'for device in jax.devices())
    conf=json.loads((args.run/'config.json').read_text());identity=conf['identity'];selected=implementation(identity['recipe']).core
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.run/f'checkpoint_{args.step}.pkl';actor_path=args.run/f'actor_{args.step}.pkl'
    cp_hash=sha(checkpoint);actor_hash=sha(actor_path)
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    with actor_path.open('rb')as stream:artifact=pickle.load(stream)
    assert saved['identity']==artifact['identity']==identity and saved['step']==artifact['step']==args.step
    assert saved['source_sha256']==artifact['source_sha256']==conf['source_sha256']
    assert artifact['checkpoint_sha256']==cp_hash and artifact['config_sha256']==sha(args.run/'config.json')
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    metadata=artifact['metadata']
    template=selected.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,identity['recipe'])
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.base.q_updates)==int(state.actor_updates)==args.step and int(state.base.v_updates)==0
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(state))
    payload=core.lean.policy_parameters(state.base)
    original=selected.make_teacher_policy(state);replica=make_policy(state,'min')
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True);prepare_seeded_action_space(env,42)
    differences=[]
    for index in range(4):
        seed=42+index;np.random.seed(seed);random.seed(seed);env.action_space.seed(seed)
        obs,_=env.reset(seed=seed);obs=np.asarray(obs,np.float32);key=jax.random.PRNGKey(29421+index)
        differences.append(float(np.max(np.abs(np.asarray(original(payload,obs,key))-np.asarray(replica(payload,obs,key))))))
    env.close();assert max(differences)<1e-5,differences
    result=dict(time=time.time(),run=args.run.name,recipe=identity['recipe'],environment=identity['environment'],
        training_seed=identity['seed'],step=args.step,aggregation=args.aggregation,checkpoint_sha256=cp_hash,
        actor_artifact_sha256=actor_hash,config_sha256=sha(args.run/'config.json'),source_sha256=sha(__file__),
        min_replica_numeric_precheck=dict(cases=4,max_action_differences=differences),
        optimizer_updates_performed=0,primary_evaluation=False,deployment_candidate=False,
        generated_actions=9,flow_integration_steps=10,requires_critic=True,
        scope='Auxiliary frozen-teacher aggregation intervention; all primary last3/full9/fresh metrics remain untouched.')
    if args.validate_only:
        result.update(status='implementation_pass',no_performance_claim=True)
    else:
        env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
        reset_metadata=prepare_seeded_action_space(env,42)
        policy=make_policy(state,args.aggregation)
        namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
        result.update(evaluate(namespace,env,policy,args.step,50));env.close()
        result.update(status='complete',reset_metadata=reset_metadata,protocol='auxiliary_frozen_teacher_aggregation_cpu50_v1')
        canonical=args.run.parent.parent/f'analysis/evaluations_v1/{args.run.name}_{args.step}_teacher_n50.json'
        reference=json.loads(canonical.read_text());assert reference['checkpoint_sha256']==cp_hash
        matches={key:result[key]==reference[key]for key in ('episode_successes','episode_rewards','episode_lengths')}
        result.update(canonical_min_reference=str(canonical),canonical_min_reference_sha256=sha(canonical),
            canonical_min_success_rate=reference['success_rate'],episode_arrays_equal_to_canonical_min=matches,
            paired_success_wins=sum(a>b for a,b in zip(result['episode_successes'],reference['episode_successes'])),
            paired_success_losses=sum(a<b for a,b in zip(result['episode_successes'],reference['episode_successes'])))
    assert sha(checkpoint)==cp_hash and sha(actor_path)==actor_hash
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps({key:result[key]for key in ('status','run','aggregation','min_replica_numeric_precheck')}),flush=True)


if __name__=='__main__':main()
