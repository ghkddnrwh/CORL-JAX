"""Saved-model CPU check of label/deployment noise and selection alignment."""
import argparse
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
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert all(device.platform=='cpu' for device in jax.devices())
    cfg_file=args.run/'config.json';conf=json.loads(cfg_file.read_text());identity=conf['identity']
    for p,h in conf['source_sha256'].items():assert sha(p)==h,p
    cp=args.run/f'checkpoint_{args.step}.pkl';ap=args.run/f'actor_{args.step}.pkl';ch,ah=sha(cp),sha(ap)
    with cp.open('rb') as stream:saved=pickle.load(stream)
    with ap.open('rb') as stream:artifact=pickle.load(stream)
    assert saved['identity']==artifact['identity']==identity and saved['step']==artifact['step']==args.step
    assert artifact['checkpoint_sha256']==ch and artifact['config_sha256']==sha(cfg_file)
    meta=artifact['metadata'];cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=core.create(identity['seed'],jnp.zeros((1,meta['observation_dim'])),jnp.zeros((1,meta['action_dim'])),cfg,identity['recipe'])
    state=flax.serialization.from_state_dict(template,saved['state']);spec=cfg['onestep_specification']
    assert spec['target']=='best9' and spec['input']=='bank'
    teacher=core.make_teacher_policy(state,'cpu');actor=core.make_actor_policy(meta,'cpu')
    params=core.lean.policy_parameters(state.base)
    env,data,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    indices=np.random.RandomState(21931).randint(len(data['actions']),size=4)
    observations=np.asarray(data['observations'][indices],np.float32);rows=[]
    label=jax.jit(lambda st,obs,key:core.teacher_batch(st,{'observations':obs,'next_observations':obs},key,spec))
    for i,ob in enumerate(observations):
        rng=jax.random.PRNGKey(51932+i);fold=jax.random.fold_in(rng,61723)
        ob_batch,noises,target,heads,actions,index=label(state,jnp.asarray(ob[None]),rng)
        deployed_teacher=np.asarray(teacher(params,ob,fold))
        label_difference=float(np.max(np.abs(np.asarray(target[0])-deployed_teacher)))
        trained_action=jnp.clip(state.actor(ob_batch,core.packed_input(noises,spec)),-1,1)[0]
        deployed_actor=actor(artifact['actor_params'],ob,fold)
        packing_difference=float(np.max(np.abs(np.asarray(trained_action)-np.asarray(deployed_actor))))
        assert label_difference<1e-5 and packing_difference<1e-5
        rows.append(dict(dataset_index=int(indices[i]),teacher_label_deployment_max_difference=label_difference,
                         actor_training_deployment_max_difference=packing_difference))
    assert sha(cp)==ch and sha(ap)==ah
    result=dict(time=time.time(),status='pass',run=args.run.name,step=args.step,cases=rows,
        source_sha256=sha(__file__),checkpoint_sha256=ch,actor_sha256=ah,
        no_optimizer_updates=True,no_performance_claim=True,
        scope='Four saved-model CPU observations,B1,samefoldedkey. Code and shape check, not efficacy or a proof for all inputs.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
