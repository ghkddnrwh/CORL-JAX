"""Read-only auxiliary BC-flow1 quality, separated from CPU50 final results.

No best-of-N selection or one-step approximation: generate one full10-step
BC flow action. This localizes the loss of critic selection versus amortizing
the BC flow. The private CPU10 diagnostic never enters selection gates.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    conf_path=args.run/'config.json';conf=json.loads(conf_path.read_text())
    identity=conf['identity'];assert identity['recipe']=='fqln_a03'
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    checkpoint=args.run/'checkpoint_1000000.pkl';digest=sha(checkpoint)
    artifact_path=args.run/'actor_1000000.pkl';artifact_digest=sha(artifact_path)
    with checkpoint.open('rb') as stream:saved=pickle.load(stream)
    with artifact_path.open('rb') as stream:artifact=pickle.load(stream)
    assert saved['identity']==artifact['identity']==identity and saved['step']==1000000
    assert saved['source_sha256']==conf['source_sha256'] and artifact['checkpoint_sha256']==digest
    cfg=dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    meta=artifact['metadata']
    template=core.create(identity['seed'],jnp.zeros((1,meta['observation_dim'])),
                         jnp.zeros((1,meta['action_dim'])),cfg,identity['recipe'])
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.base.q_updates)==int(state.actor_updates)==1000000 and int(state.base.v_updates)==0
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(state))
    def policy(parameters,observations,key):
        agent=state.base.agent.replace(network=state.base.agent.network.replace(params=parameters))
        ob,noise=core.lean.bank(agent,observations,key,1)
        return core.lean.flow(agent,ob,noise,10)[0]
    policy=jax.jit(policy,backend='cpu')
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=state.base.agent.network.params)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
    reset_metadata=prepare_seeded_action_space(env,42)
    result=evaluate(namespace,env,policy,1000000,10)
    env.close()
    result.update(time=time.time(),run=args.run.name,environment=identity['environment'],
        recipe=identity['recipe'],training_seed=identity['seed'],
        policy_mode='BC_flow1_without_Q_selection',flow_steps=10,generated_actions=1,
        uses_critic_for_action=False,is_one_step_actor=False,
        checkpoint_sha256=digest,actor_artifact_sha256=artifact_digest,
        config_sha256=sha(conf_path),source_sha256=sha(__file__),
        shared_evaluation_source_sha256=sha(Path(core.__file__).with_name('train_mean_empirical_backup_v1.py')),
        original_training_source_sha256=conf['source_sha256'],reset_metadata=reset_metadata,
        no_optimizer_updates=True,CPU_affinity=sorted(os.sched_getaffinity(0)),
        protocol='exploratory_BCflow1_CPU10_1M_extraction_diagnostic_v1',
        counts_toward_performance_acceptance=False,isolated_speed_claim=False,
        scope='Single checkpoint and10episodes; compare stored first10episodes of matching primary/teacher CPU50. Not a last3, nine-task or fresh-seed conclusion. Noise shape differs frombest9, so same episode seeds do not imply the same initial flow candidate.')
    assert sha(checkpoint)==digest and sha(artifact_path)==artifact_digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,flow1_success=result['success_rate'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
