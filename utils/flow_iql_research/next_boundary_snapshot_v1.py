"""Inspect all certified entry edges and known-zero continuation target errors."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from utils.flow_iql_research import policy_core,selection_v2,next_success_labels_v1
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.propagation_research.reward_bounds import return_bounds
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    config,saved,cp=read(args.run,args.step);identity=config['identity'];cfg=dict(identity['config'])
    assert identity['kind']=='ifql'and identity['profile']in('bounded_v2_empirical_backup','bounded_v2_next_boundary_backup')
    env,train,val=ogbench.make_env_and_datasets(identity['environment'])
    for field in('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
    initial=policy_core.create(identity['seed'],jnp.asarray(train['observations'][:1]),jnp.asarray(train['actions'][:1]),cfg,'ifql')
    state=flax.serialization.from_state_dict(initial,saved['state'])
    assert int(state.q_updates)==int(state.agent.network.opt_state[0].count)==args.step
    assert int(state.v_updates)==int(state.v_opt[0].count)==2*args.step
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
    bounds=return_bounds(identity['environment'],cfg['discount'],float(np.min(train['rewards'])),float(np.max(train['rewards'])),'ogbench')
    @jax.jit
    def predict(obs,act,next_obs,key):
        proposals,_=selection_v2.proposals(state.agent,next_obs,key,9,'behavior')
        repeated=jnp.broadcast_to(next_obs,(9,*next_obs.shape));net=state.agent.network
        values=net.select('target_critic')(repeated,actions=proposals).min(0)
        root=selection_v2.empirical_expectile(values,cfg['expectile'])
        return root,net.select('critic')(obs,actions=act),net.select('value')(next_obs)
    summaries={};override=identity['profile']=='bounded_v2_next_boundary_backup'
    for split,data in [('train',train),('validation',val)]:
        labels,metadata=next_success_labels_v1.labels(data)
        indices=np.flatnonzero((labels==1)&(np.asarray(data['masks'])==1))
        if not len(indices):
            summaries[split]=dict(available=False,coverage=metadata,entry_edges=0);continue
        outputs=[]
        for start in range(0,len(indices),1024):
            chosen=indices[start:start+1024]
            outputs.append(jax.device_get(predict(jnp.asarray(data['observations'][chosen]),
                jnp.clip(jnp.asarray(data['actions'][chosen]),-1+1e-5,1-1e-5),
                jnp.asarray(data['next_observations'][chosen]),jax.random.fold_in(jax.random.PRNGKey(19402),start))))
        root=np.concatenate([row[0]for row in outputs]).astype(np.float64)
        qs=np.concatenate([row[1]for row in outputs],axis=1).astype(np.float64)
        learned=np.concatenate([row[2]for row in outputs]).astype(np.float64)
        rewards=np.asarray(data['rewards'])[indices].astype(np.float64)
        raw=rewards+cfg['discount']*root;raw_projected=np.clip(raw,*bounds)
        actual=rewards if override else raw_projected
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        summaries[split]=dict(available=True,coverage=metadata,entry_edges=len(indices),inspected_all=True,
            index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),next_success_finite_root_RMS_to_known_zero=rms(root),
            next_success_learned_V_RMS_to_known_zero=rms(learned),root_mean=float(root.mean()),
            unmodified_unprojected_target_error_RMS=rms(raw-rewards),
            unmodified_projected_target_error_RMS=rms(raw_projected-rewards),
            actual_projected_target_error_RMS=rms(actual-rewards),
            current_Q_to_observed_exact_boundary_target_RMS=rms(qs-rewards[None]),
            current_Q_to_actual_training_target_RMS=rms(qs-actual[None]))
    env.close()
    result=dict(time=time.time(),kind='certified_next_success_boundary_target_snapshot',run=args.run.name,
        environment=identity['environment'],profile=identity['profile'],step=args.step,sets=summaries,
        next_boundary_override=override,checkpoint_sha256=sha(cp),config_sha256=sha(args.run/'config.json'),
        actual_Q_Adam_count=args.step,actual_V_Adam_count=2*args.step,source_sha256=sha(__file__),
        helper_source_sha256=sha(next_success_labels_v1.__file__),primary_scores_unchanged=True,
        limits=['Onlycertifiedentryedges;notallstates,notallactions,andnotapolicyreturnestimate.',
            'Continuationattheknownsuccessstateisexactzero;predecessorQiscomparedtoanOBSERVEDtransitiontarget,notanoracleexpectedQlabel.',
            'Validationedgesareevaluatedbutneverusedfortraining.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,sets=summaries)),flush=True)


if __name__=='__main__':main()
