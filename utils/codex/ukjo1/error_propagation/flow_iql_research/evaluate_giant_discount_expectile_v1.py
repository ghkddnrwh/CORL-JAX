"""CPU50 paired lean-best9 and actor-location target-ablation evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time
from types import SimpleNamespace
from utils.flow_iql_research import giant_discount_expectile_core_v1 as core
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--mode',choices=('actor','lean'),required=True)
    parser.add_argument('--episodes',type=int,default=50);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--implementation-only',action='store_true')
    args=parser.parse_args();assert not args.output.exists()
    assert args.episodes in (10,50) and all(d.platform=='cpu'for d in jax.devices())
    if args.episodes==50:assert args.step in (800000,900000,1000000)
    if args.episodes==50:assert not args.implementation_only
    if args.episodes==10 and args.step not in (100000,300000):assert args.implementation_only
    conf=json.loads((args.run/'config.json').read_text());identity=conf['identity']
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.run/f'checkpoint_{args.step}.pkl';digest=sha(checkpoint)
    artifact_path=args.run/f'actor_{args.step}.pkl'
    with artifact_path.open('rb') as stream:artifact=pickle.load(stream)
    assert artifact['step']==artifact['actor_updates']==artifact['Q_updates']==args.step
    assert artifact['identity']==identity and artifact['checkpoint_sha256']==digest
    assert artifact['source_sha256']==conf['source_sha256'] and artifact['config_sha256']==sha(args.run/'config.json')
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(artifact['actor_params']))
    metadata=artifact['metadata'];actor_hash=sha(artifact_path)
    if args.mode=='actor':
        # Loading the actor requires neither the learner nor its critic weights.
        policy=core.make_actor_policy(metadata);payload=artifact['actor_params']
        learned_model_trees_loaded=['actor_params']
    else:
        with checkpoint.open('rb') as stream:saved=pickle.load(stream)
        assert saved['identity']==identity and saved['source_sha256']==conf['source_sha256'] and saved['step']==args.step
        cfg=dict(identity['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        template=core.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,identity['recipe'])
        state=flax.serialization.from_state_dict(template,saved['state'])
        assert int(state.base.q_updates)==int(state.actor_updates)==args.step and int(state.base.v_updates)==0
        assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
        policy=core.make_teacher_policy(state);payload=core.lean.policy_parameters(state.base)
        learned_model_trees_loaded=['same_checkpoint_BC_flow','same_checkpoint_critic']
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
    reset_metadata=prepare_seeded_action_space(env,42)
    result=evaluate(namespace,env,policy,args.step,args.episodes)
    env.close()
    # Diagnostic timing only: final speed claims require isolated benchmarks.
    device=jax.devices('cpu')[0];parameters=jax.device_put(payload,device)
    observation=jax.device_put(np.zeros(metadata['observation_dim'],np.float32),device)
    keys=jax.device_put(jax.random.split(jax.random.PRNGKey(414),110),device)
    for index in range(10):policy(parameters,observation,keys[index]).block_until_ready()
    measurements=[]
    for index in range(10,110):
        began=time.perf_counter();policy(parameters,observation,keys[index]).block_until_ready()
        measurements.append((time.perf_counter()-began)*1000)
    result.update(time=time.time(),run=args.run.name,environment=identity['environment'],training_seed=identity['seed'],
        recipe=identity['recipe'],policy_mode=args.mode,checkpoint_sha256=digest,actor_artifact_sha256=actor_hash,
        config_sha256=sha(args.run/'config.json'),source_sha256=sha(__file__),core_source_sha256=sha(core.__file__),
        reset_metadata=reset_metadata,no_optimizer_updates=True,learned_model_trees_loaded=learned_model_trees_loaded,
        deployment_metadata=metadata if args.mode=='actor' else None,
        diagnostic_shared_machine_CPU_action_ms_median=statistics.median(measurements),
        diagnostic_shared_machine_CPU_action_ms_p90=float(np.quantile(measurements,.9)),
        isolated_speed_claim=False,actor_artifact_bytes=artifact_path.stat().st_size,
        scope='Paired same-base comparison: lean Flow-best9 and actor component-location. Not two independent training runs.')
    if args.episodes==10:result['protocol']='auxiliary_cpu10_100K300K_screening_v1'
    if args.implementation_only:result.update(protocol='implementation_smoke_only_cpu10',no_performance_claim=True)
    assert sha(checkpoint)==digest and sha(artifact_path)==actor_hash
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,mode=args.mode,episodes=args.episodes,success=result['success_rate'])),flush=True)


if __name__=='__main__':main()
