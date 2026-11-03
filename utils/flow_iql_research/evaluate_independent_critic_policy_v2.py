"""Frozen independent-critic ensemble evaluation; no mixed learning state."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core,independent_critic_policy_v2 as selector
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research import train as primary
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actor-run',type=Path,required=True)
    parser.add_argument('--critic-runs',type=Path,nargs='+',required=True);parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--rule',choices=selector.RULES,default='median_rank')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();assert not args.output.exists()and len(args.critic_runs)in(1,3)
    actor_config,saved,actor_path=read(args.actor_run,args.step);identity=actor_config['identity'];cfg=dict(identity['config'])
    assert identity['kind']=='ifql'and identity['profile']=='bounded_v2'
    for name in('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    template=policy_core.create(identity['seed'],jnp.zeros((1,env.observation_space.shape[-1])),jnp.zeros((1,env.action_space.shape[-1])),cfg,'ifql')
    state=flax.serialization.from_state_dict(template,saved['state']);assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
    members=[];evidence=[]
    for run in args.critic_runs:
        conf,member,path=read(run,args.step);ident=conf['identity']
        assert ident['environment']==identity['environment']and ident['kind']=='ifql'and ident['profile']=='bounded_v2'
        assert {k:v for k,v in ident['config'].items()if k!='num_samples'}=={k:v for k,v in identity['config'].items()if k!='num_samples'}
        assert int(member['state']['v_updates'])==2*args.step
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(member['state']))
        members.append(jax.device_put(member['state']['agent']['network']['params']['modules_critic'],jax.devices('cpu')[0]))
        evidence.append(dict(run=run.name,training_seed=ident['seed'],checkpoint=str(path),checkpoint_sha256=sha(path),
            config_sha256=sha(run/'config.json'),actual_Q_Adam_count=args.step,actual_V_Adam_count=2*args.step))
    assert len({x['training_seed']for x in evidence})==len(evidence)
    compiled=selector.make_selector(state.agent,args.rule,9);diagnostics=[]
    def policy(params,observation,key):
        action,values,combined=compiled(params,tuple(members),observation,key)
        q=np.asarray(values,np.float64);s=np.asarray(combined,np.float64);assert np.isfinite(q).all()
        chosen=int(np.argmax(s));agreement=float(np.mean(q.argmax(1)==chosen))
        diagnostics.append([agreement,float(np.sum(s==s.max())>1),float(np.mean(np.std(q-q.mean(1,keepdims=True),axis=0)))])
        return action
    result=primary.evaluate(state,env,policy,args.step,50);exact=None
    if args.reference is not None:
        ref=json.loads(args.reference.read_text());assert ref.get('actor_checkpoint_sha256',ref.get('checkpoint_sha256'))==sha(actor_path)and ref['selection_samples']==9
        if len(members)==3:
            assert ref['num_independent_learners']==3 and ref['aggregation_rule']==args.rule
            assert [x['checkpoint_sha256']for x in ref['critic_members']]==[x['checkpoint_sha256']for x in evidence]
        exact={key:result[key]==ref[key]for key in('episode_successes','episode_rewards','episode_lengths')};assert all(exact.values()),exact
    result.update(time=time.time(),kind='independent_critic_ensemble_policy_discovery',environment=identity['environment'],
        aggregation_rule=args.rule,selection_samples=9,num_independent_learners=len(members),num_Q_heads=2*len(members),
        learner_update_step=args.step,total_Q_updates_across_members=args.step*len(members),
        total_V_updates_across_members=2*args.step*len(members),critic_members=evidence,actor_training_seed=identity['seed'],
        actor_checkpoint_sha256=sha(actor_path),self_reference_episode_arrays_exact=exact,
        deployed_policy_diagnostics=dict(zip(('member_argmax_agreement','combined_score_tie_fraction','action_centered_member_std_mean'),np.mean(diagnostics,axis=0).tolist())),
        evaluator_source_sha256=sha(__file__),selector_source_sha256=sha(selector.__file__),reset_metadata=reset,
        primary_evaluations_modified=False,mixed_joint_learning_state_saved=False,
        limits=['Three independent learners form ONE ensemble observation, not three independent ensemble training seeds.',
                'Training cost is approximately three base learners; each member has the reported update budget.',
                'The same current BC of the first learner supplies all nine proposals; only critic aggregation changes.',
                'Median ranks are robust to one dissenting critic only when two critics agree on a unique best action.',
                'No oracle rank correctness, neural convergence or seed independence of errors is assumed.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(dict(rule=args.rule,step=args.step,success=result['success_rate'],exact=exact)),flush=True)


if __name__=='__main__':main()
