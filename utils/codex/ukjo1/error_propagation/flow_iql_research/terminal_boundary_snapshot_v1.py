"""Independent observed/uniform-action boundary audit on a frozen IFQL model."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import ogbench.relabel_utils


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    conf=json.loads((args.run/'config.json').read_text());ident=conf['identity'];cfg=dict(ident['config'])
    assert ident['kind']=='ifql'
    for source,digest in conf['source_sha256'].items():assert sha(source)==digest,source
    env,train,val=ogbench.make_env_and_datasets(ident['environment'])
    for key in('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=policy_core.create(ident['seed'],jnp.zeros_like(jnp.asarray(train['observations'][:1])),
        jnp.zeros_like(jnp.asarray(train['actions'][:1])),cfg,'ifql')
    cp=args.run/f'checkpoint_{args.step}.pkl'
    with cp.open('rb')as stream:saved=pickle.load(stream)
    assert saved['step']==args.step and json.loads(json.dumps(saved['identity']))==ident
    assert saved['source_sha256']==conf['source_sha256']
    state=flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.q_updates)==int(state.agent.network.opt_state[0].count)==args.step
    assert int(state.v_updates)==int(state.v_opt[0].count)==2*args.step
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(state))
    @jax.jit
    def predict(observations,actions,uniform_actions):
        net=state.agent.network
        return dict(Q=net.select('critic')(observations,actions=actions),
                    uniform_Q=net.select('critic')(observations,actions=uniform_actions),
                    target_minQ=net.select('target_critic')(observations,actions=actions).min(0),
                    V=net.select('value')(observations))
    summaries={}
    for split,data in [('independent_train',train),('validation',val)]:
        indices=np.flatnonzero(np.asarray(data['masks'])==0);total=len(indices)
        assert np.all(np.asarray(data['rewards'])[indices]==0)
        if len(indices)>4096:indices=np.random.RandomState(89131).choice(indices,4096,replace=False)
        if not len(indices):
            summaries[split]=dict(total_goal_transitions=0,inspected_goal_transitions=0,available=False)
            continue
        rng=np.random.RandomState(89413);outputs=[]
        # Draw all uniforms first, so the action set does not depend on batching.
        uniform=rng.uniform(-1,1,(len(indices),data['actions'].shape[-1])).astype(np.float32)
        for start in range(0,len(indices),1024):
            chosen=indices[start:start+1024]
            outputs.append(jax.device_get(predict(jnp.asarray(data['observations'][chosen],jnp.float32),
                jnp.clip(jnp.asarray(data['actions'][chosen],jnp.float32),-1+1e-5,1-1e-5),
                jnp.asarray(uniform[start:start+1024]))))
        out={name:np.concatenate([row[name]for row in outputs],axis=1 if name in('Q','uniform_Q')else 0).astype(np.float64)
             for name in outputs[0]}
        delta=out['target_minQ']-out['V'];score=np.where(delta>=0,cfg['expectile'],1-cfg['expectile'])*delta
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        summaries[split]=dict(available=True,total_goal_transitions=total,inspected_goal_transitions=len(indices),
            inspected_all=len(indices)==total,index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            uniform_actions_sha256=hashlib.sha256(uniform.tobytes()).hexdigest(),
            observed_Q_RMS_to_known_zero=rms(out['Q']),uniform_Q_RMS_to_known_zero=rms(out['uniform_Q']),
            V_RMS_to_known_zero=rms(out['V']),observed_Q_mean=float(out['Q'].mean()),
            uniform_Q_mean=float(out['uniform_Q'].mean()),V_mean=float(out['V'].mean()),
            target_minQ_minus_V_RMS=rms(delta),expectile_score_proxy_RMS=rms(score),
            both_observed_Q_heads_within1_fraction=float(np.mean((np.abs(out['Q'])<=1).all(0))),
            both_uniform_Q_heads_within1_fraction=float(np.mean((np.abs(out['uniform_Q'])<=1).all(0))),
            V_within1_fraction=float(np.mean(np.abs(out['V'])<=1)))
    env.close()
    result=dict(time=time.time(),run=args.run.name,environment=ident['environment'],step=args.step,
                kind='known_success_boundary_snapshot',sets=summaries,checkpoint_sha256=sha(cp),
                config_sha256=sha(args.run/'config.json'),diagnostic_source_sha256=sha(__file__),
                relabeler_sha256=sha(ogbench.relabel_utils.__file__),actual_Q_Adam_count=int(state.agent.network.opt_state[0].count),
                actual_V_Adam_count=int(state.v_opt[0].count),primary_scores_unchanged=True,
                limits=['Zero labels use verified OGBench state-based masked-task semantics; no oracle interior Q labels.',
                        'Independent train is a fresh diagnostic draw from training data, not a heldout training seed.',
                        'Observed and uniform actions share goal states; they are not independent generalization datasets.',
                        'Boundary fit and action coverage do not imply improved interior ranking or policy return.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,sets=summaries)),flush=True)


if __name__=='__main__':main()
