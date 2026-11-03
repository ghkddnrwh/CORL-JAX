"""CPU50 exact late-model policy evaluation; primary and teacher kept separate."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import efficient_mean_core_v1 as core
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--mode',choices=('primary','teacher'),default='primary')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    assert all(device.platform=='cpu' for device in jax.devices())
    conf=json.loads((args.run/'config.json').read_text())
    identity=conf['identity']
    for path,digest in conf['source_sha256'].items():
        assert sha(path)==digest,path
    path=args.run/f'checkpoint_{args.step}.pkl'
    digest=sha(path)
    with path.open('rb') as stream:
        saved=pickle.load(stream)
    assert saved['step']==args.step and saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key]=tuple(cfg[key])
    dataset=json.loads((args.run/'dataset.json').read_text())
    template=core.create(identity['seed'],jnp.zeros((1,dataset['observation_dim'])),
        jnp.zeros((1,dataset['action_dim'])),cfg,identity['variant'])
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.q_updates)==args.step
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    policy=core.make_policy(state,identity['variant'],args.mode)
    payload=core.policy_parameters(state)
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    result=evaluate(namespace,env,policy,args.step,50)
    env.close()
    result.update(time=time.time(),run=args.run.name,variant=identity['variant'],environment=identity['environment'],
        training_seed=identity['seed'],policy_mode=args.mode,checkpoint_sha256=digest,config_sha256=sha(args.run/'config.json'),
        source_sha256=sha(__file__),core_source_sha256=sha(core.__file__),reset_metadata=reset,
        Q_updates=int(state.q_updates),V_updates=int(state.v_updates),BC_student_updates=int(state.student_updates),
        guided_policy_updates=int(state.policy_updates),no_optimizer_updates=True,
        scope='Primaryfixed800K/900K/1M model CPU50. Teacher mode is an auxiliary same-Q deployment control,never mixed into primary averages.')
    assert sha(path)==digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False))
    temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,mode=args.mode,success=result['success_rate'])),flush=True)


if __name__=='__main__':main()
