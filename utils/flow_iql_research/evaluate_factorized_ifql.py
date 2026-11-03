"""Hold a behavioral flow fixed while swapping independently trained IQL Q/V.

IFQL's behavioral flow loss has no Q/V dependence. This counterfactual therefore
isolates the value-controller comparison, using real equal-step checkpoints and
no extra optimizer updates. It is kept separate from primary training scores.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core,train
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal_tree(a,b):
    aa=jax.tree_util.tree_leaves(a);bb=jax.tree_util.tree_leaves(b)
    return len(aa)==len(bb) and all(np.array_equal(x,y) for x,y in zip(aa,bb))


def read(run,step):
    conf=json.loads((run/'config.json').read_text())
    for source,digest in conf['source_sha256'].items():assert sha(source)==digest,source
    checkpoint=run/f'checkpoint_{step}.pkl'
    if not checkpoint.exists():checkpoint=run.parent.parent/'analysis/checkpoint_archive'/run.name/f'checkpoint_{step}.pkl'
    with checkpoint.open('rb') as f:saved=pickle.load(f)
    assert saved['step']==step and int(saved['state']['q_updates'])==step
    return conf,saved,checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--value-run',type=Path,required=True)
    parser.add_argument('--density-run',type=Path,required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing factorized evidence')
    qc,qs,qp=read(args.value_run,args.step);bc,bs,bp=read(args.density_run,args.step)
    qi,bi=qc['identity'],bc['identity']
    assert qi['environment']==bi['environment'] and qi['seed']==bi['seed']
    assert bi['kind']=='ifql'
    for field in ('lr','batch_size','actor_hidden_dims','value_hidden_dims','layer_norm','actor_layer_norm'):
        assert qi['config'][field]==bi['config'][field],field
    assert equal_tree(qs['replay_rng'],bs['replay_rng']), 'Replay streams differ'
    cfg=dict(bi['config'])
    for field in ('discount','expectile','tau','value_hidden_dims','layer_norm'):cfg[field]=qi['config'][field]
    for field in ('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
    qstate=qs['state'];bstate=bs['state'];merged=copy.deepcopy(qstate)
    qnet=merged['agent']['network'];bnet=bstate['agent']['network']
    actor_keys=[k for k in bnet['params'] if k.startswith('modules_actor')]
    assert actor_keys==['modules_actor_flow']
    for key in list(qnet['params']):
        if key.startswith('modules_actor'):del qnet['params'][key]
    for key in actor_keys:qnet['params'][key]=copy.deepcopy(bnet['params'][key])
    qadam=qnet['opt_state']['0'];badam=bnet['opt_state']['0']
    assert int(qadam['count'])==int(badam['count'])==args.step
    for field in ('mu','nu'):
        for key in list(qadam[field]):
            if key.startswith('modules_actor'):del qadam[field][key]
        for key in actor_keys:qadam[field][key]=copy.deepcopy(badam[field][key])
    merged['agent']['rng']=copy.deepcopy(bstate['agent']['rng'])
    same_params=equal_tree(qnet['params'],bnet['params'])
    same_dynamic=equal_tree(merged,bstate)
    env=ogbench.make_env_and_datasets(qi['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    template=policy_core.create(qi['seed'],jnp.zeros((1,env.observation_space.shape[-1])),jnp.zeros((1,env.action_space.shape[-1])),cfg,'ifql')
    state=flax.serialization.from_state_dict(template,merged)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    result=train.evaluate(state,env,train.make_policy(state.agent),args.step,50)
    existing=[]
    if (args.density_run/'evaluations.jsonl').exists():
        existing=[json.loads(s) for s in (args.density_run/'evaluations.jsonl').read_text().splitlines() if json.loads(s)['step']==args.step]
    comparison=None
    if same_params and existing:
        comparison={key:existing[0].get(key)==result.get(key) for key in ('episode_successes','episode_rewards','episode_lengths')}
        assert all(comparison.values()),comparison
    result.update(kind='auxiliary_factorized_IFQL_value_controller_control',time=time.time(),
        value_run=args.value_run.name,density_run=args.density_run.name,
        value_profile=qi['profile'],value_discount=qi['config']['discount'],expectile=qi['config']['expectile'],
        retained_Q_updates=int(state.q_updates),retained_V_updates=int(state.v_updates),retained_BC_flow_updates=args.step,
        all_network_params_equal_density_model=same_params,full_dynamic_state_equal_density_model=same_dynamic,
        exact_same_policy_episode_comparison=comparison,
        value_checkpoint_sha256=sha(qp),density_checkpoint_sha256=sha(bp),
        source_sha256=sha(__file__),evaluation_helper_sha256=sha(train.__file__),reset_metadata=reset,
        caveats=['The Q/V and behavioral flow are assembled from independently trained equal-step sources. No primary score, source checkpoint, replay state or optimizer trajectory is edited.',
                 'The behavioral flow objective is independent of Q/V; unused Gaussian training in the value source is a research-compute overhead, not an input to this policy.',
                 'This is an auxiliary common-proposal comparison. Exact equivalence to a separately compiled direct IFQL trajectory must be checked, not assumed.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(value_run=args.value_run.name,step=args.step,success=result['success_rate'],same_policy=same_params,episode_comparison=comparison)),flush=True)


if __name__=='__main__':main()
