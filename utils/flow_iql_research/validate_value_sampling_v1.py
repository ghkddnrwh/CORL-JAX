"""Actual1M-state checks before registering coupled minibatch-reuse pilots."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core,value_sampling_core_v1 as changed
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def difference(a,b):
    aa,ta=jax.tree_util.tree_flatten(a);bb,tb=jax.tree_util.tree_flatten(b)
    assert ta==tb and len(aa)==len(bb)
    errors=[float(np.max(np.abs(np.asarray(x,np.float64)-np.asarray(y,np.float64)))) for x,y in zip(aa,bb)]
    return dict(array_leaves=len(aa),max_abs=max(errors),exact=all(e==0 for e in errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and all(d.platform=='cpu' for d in jax.devices())
    verifier=Verifier();run=args.run.resolve();config=json.loads((run/'config.json').read_text());identity=config['identity'];cfg=dict(identity['config'])
    for path,digest in config['source_sha256'].items():assert verifier.digest(path)==digest,path
    cp=run/'checkpoint_1000000.pkl';cp_sha=verifier.checkpoint(cp,config,1000000)
    with cp.open('rb')as stream:saved=pickle.load(stream)
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    env,raw,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
    template=policy_core.create(identity['seed'],jnp.zeros((1,raw['observations'].shape[-1])),jnp.zeros((1,raw['actions'].shape[-1])),cfg,'ifql')
    initial=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),jax.devices('cpu')[0])
    data=json.loads((run/'dataset.json').read_text());bounds=(data['return_lower'],data['return_upper'])
    original=jax.jit(policy_core.build_update('bounded_v2',bounds,'ifql'),backend='cpu')
    repeated=jax.jit(changed.build_update('bounded_v2_repeated_batches',bounds,'ifql'),backend='cpu')
    independent=jax.jit(changed.build_update('bounded_v2_fresh_batches',bounds,'ifql'),backend='cpu')
    rng=np.random.RandomState(19695)
    def sample():
        indices=rng.randint(len(raw['actions']),size=256)
        batch={k:jnp.asarray(raw[k][indices],jnp.float32)for k in ('observations','actions','next_observations','rewards','masks')}
        batch['actions']=jnp.clip(batch['actions'],-1+1e-5,1-1e-5)
        return batch
    states={name:initial for name in ('original','repeat','fresh_same_data','fresh_new_data')};trace=[]
    for index in range(4):
        batch=sample();extra=sample()
        new={**batch,'extra_value_observations':extra['observations'],'extra_value_actions':extra['actions']}
        same={**batch,'extra_value_observations':batch['observations'],'extra_value_actions':batch['actions']}
        states['original'],base_logs=original(states['original'],batch)
        states['repeat'],repeat_logs=repeated(states['repeat'],new)
        states['fresh_same_data'],_=independent(states['fresh_same_data'],same)
        states['fresh_new_data'],_=independent(states['fresh_new_data'],new)
        checks=dict(repeated_vs_original=difference(states['repeat'],states['original']),
            same_data_algebraic_control=difference(states['fresh_same_data'],states['original']),
            fresh_data_effect=difference(states['fresh_new_data'],states['original']),
            repeated_logs_vs_original=difference(repeat_logs,base_logs))
        assert checks['repeated_vs_original']['exact'] and checks['repeated_logs_vs_original']['exact']
        assert checks['same_data_algebraic_control']['exact']
        assert checks['fresh_data_effect']['max_abs']>0
        for name,state in states.items():
            assert int(state.q_updates)==1000001+index and int(state.v_updates)==2000002+2*index
            assert int(state.v_opt[0].count)==2000002+2*index
            assert int(state.agent.network.opt_state[0].count)==1000001+index
            assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
            assert difference(state.agent.network.params['modules_actor_flow'],states['original'].agent.network.params['modules_actor_flow'])['exact']
            assert np.array_equal(np.asarray(state.agent.rng),np.asarray(states['original'].agent.rng))
        trace.append(dict(extra_joint_updates=index+1,checks=checks,all_BC_parameters_exact=True,all_learner_RNGs_exact=True))
    result=dict(time=time.time(),status='passed',actual_source_checkpoint_sha256=cp_sha,
        run=run.name,trace=trace,config_sha256=verifier.digest(run/'config.json'),
        source_sha256={str(p):verifier.digest(p)for p in (Path(__file__),Path(changed.__file__),Path('utils/flow_iql_research/train_value_sampling_v1.py'))},
        limits=['CPU arithmetic on one actual1Mmodel; no claim of GPU bitwise equality or policy improvement.',
            'Resumable runner and private replay-stream state require a separate end-to-end check.'],
        new_benchmark_evaluations=0,new_GPU_training=0,goal_complete=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(status='passed',trace=trace)),flush=True)


if __name__=='__main__':main()
