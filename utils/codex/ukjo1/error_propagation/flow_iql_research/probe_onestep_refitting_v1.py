"""Extra actor fits in an isolated in-memory copy of a saved learner.

Measures optimization/generalization to fixed labels, not policy returns or a
new jointly trained algorithm. No production model, optimizer, export or score
file is written. Actual scratch Adam counts are reported without relabeling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.onestep_implementation_registry_v1 import implementation
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def tree_hash(tree):
    digest=hashlib.sha256()
    for value in jax.tree_util.tree_leaves(tree):
        array=np.asarray(value);digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fits',type=int,default=32);parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--implementation-only',action='store_true');args=parser.parse_args()
    assert not args.output.exists()and args.fits>0 and args.batch_size>0
    assert all(device.platform=='cpu'for device in jax.devices())
    start=time.time();conf=json.loads((args.run/'config.json').read_text());identity=conf['identity']
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.run/f'checkpoint_{args.step}.pkl';artifact_path=args.run/f'actor_{args.step}.pkl'
    cp_hash=sha(checkpoint);artifact_hash=sha(artifact_path)
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    with artifact_path.open('rb')as stream:artifact=pickle.load(stream)
    assert saved['step']==artifact['step']==args.step and saved['identity']==artifact['identity']==identity
    assert artifact['checkpoint_sha256']==cp_hash and artifact['config_sha256']==sha(args.run/'config.json')
    assert saved['source_sha256']==artifact['source_sha256']==conf['source_sha256']
    selected=implementation(identity['recipe']).core;cfg=dict(identity['config'])
    for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    metadata=artifact['metadata']
    template=selected.create(identity['seed'],jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,identity['recipe'])
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.base.q_updates)==int(state.actor_updates)==args.step and int(state.base.v_updates)==0
    assert int(state.actor.opt_state[0].count)==int(state.base.agent.network.opt_state[0].count)==args.step
    base_hash=tree_hash(state.base);actor_hash=tree_hash(state.actor)
    env,data,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    spec=state.base.agent.config['onestep_specification']
    targets={};index_hashes={}
    for name,index_seed,key_seed in (('fitting',21941,51941),('heldout',21942,51942)):
        indices=np.random.RandomState(index_seed).randint(len(data['actions']),size=args.batch_size)
        index_hashes[name]=hashlib.sha256(indices.tobytes()).hexdigest()
        batch={key:jnp.asarray(data[key][indices],jnp.float32)for key in ('observations','actions','next_observations','rewards','masks')}
        batch['actions']=jnp.clip(batch['actions'],-1+1e-5,1-1e-5)
        observations,noises,target,heads,_,_=core.teacher_batch(state,batch,jax.random.PRNGKey(key_seed),spec)
        targets[name]=(observations,noises,target,heads)
    @jax.jit
    def losses(params,target_batch):
        observations,noises,target,heads=target_batch
        return core.actor_loss(state,observations,noises,target,heads,spec,params)[1]
    @jax.jit
    def fit(actor):
        observations,noises,target,heads=targets['fitting']
        return actor.apply_loss_fn(lambda params:core.actor_loss(state,observations,noises,target,heads,spec,params))[0]
    initial_targets={name:tree_hash(target)for name,target in targets.items()}
    actor=state.actor;rows=[];points={0,1,2,4,8,16,32,args.fits}
    def record(count):
        assert int(actor.opt_state[0].count)==args.step+count
        values={name:{key:float(value)for key,value in jax.device_get(losses(actor.params,target)).items()}for name,target in targets.items()}
        assert all(np.isfinite(value)for row in values.values()for value in row.values())
        rows.append(dict(scratch_actor_fits=count,actual_scratch_actor_Adam_count=int(actor.opt_state[0].count),
            frozen_Q_Adam_count=int(state.base.agent.network.opt_state[0].count),losses=values))
    record(0)
    for index in range(1,args.fits+1):
        actor=fit(actor)
        if index in points:record(index)
    assert tree_hash(state.base)==base_hash and tree_hash(state.actor)==actor_hash
    assert all(tree_hash(target)==initial_targets[name]for name,target in targets.items())
    assert int(state.actor.opt_state[0].count)==args.step
    assert sha(checkpoint)==cp_hash and sha(artifact_path)==artifact_hash
    result=dict(time=time.time(),seconds=time.time()-start,status='complete',run=args.run.name,recipe=identity['recipe'],
        environment=identity['environment'],step=args.step,checkpoint_sha256=cp_hash,actor_artifact_sha256=artifact_hash,
        source_sha256=sha(__file__),core_source_sha256=sha(selected.__file__),shared_loss_core_source_sha256=sha(core.__file__),
        source_training_sha256=conf['source_sha256'],specification=dict(spec),batch_size=args.batch_size,
        dataset_index_seeds=dict(fitting=21941,heldout=21942),teacher_key_seeds=dict(fitting=51941,heldout=51942),index_hashes=index_hashes,
        fixed_target_hashes=initial_targets,base_state_hash_unchanged=base_hash,original_actor_state_hash_unchanged=actor_hash,
        original_model_optimizer_updates_performed=0,scratch_actor_optimizer_updates_performed=args.fits,
        production_artifact=False,no_policy_evaluation_performed=True,no_performance_claim=True,
        implementation_only=args.implementation_only,rows=rows,
        scope='Carry original Adam moments into a memory-only actor copy; fixed teacher/critic/labels. A single-batch optimization diagnostic, not evidence about joint-training frequency or return improvement.')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(status='complete',run=args.run.name,fits=args.fits,rows=rows)),flush=True)


if __name__=='__main__':main()
