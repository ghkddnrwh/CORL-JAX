"""Actual-state differential test of the learned mean-head V teacher."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core,mean_value_teacher_core_v1 as changed
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.validate_value_sampling_v1 import difference
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    verifier=Verifier();results=[];source=Path(changed.__file__).read_text()
    line="target_q = net.select('target_critic')(batch['observations'], actions=batch['actions']).mean(0)"
    assert source.count(line)==1
    control_source=source.replace(line,line.replace('.mean(0)','.min(0)'))
    namespace={'__name__':'validation_only_min_teacher','__file__':changed.__file__}
    exec(compile(control_source,'<validation_only_min_teacher>','exec'),namespace)
    for run in args.runs:
        conf=json.loads((run/'config.json').read_text());identity=conf['identity'];cfg=dict(identity['config'])
        for path,sha in conf['source_sha256'].items():assert verifier.digest(path)==sha,path
        checkpoint=run/'checkpoint_1000000.pkl';sha=verifier.checkpoint(checkpoint,conf,1000000)
        with checkpoint.open('rb')as stream:saved=pickle.load(stream)
        for key in('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        env,raw,_=ogbench.make_env_and_datasets(identity['environment']);env.close()
        template=policy_core.create(identity['seed'],jnp.zeros((1,raw['observations'].shape[-1])),
            jnp.zeros((1,raw['actions'].shape[-1])),cfg,'ifql')
        initial=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),jax.devices('cpu')[0])
        data=json.loads((run/'dataset.json').read_text());bounds=(data['return_lower'],data['return_upper'])
        baseline=jax.jit(policy_core.build_update('bounded_v2',bounds,'ifql'),backend='cpu')
        control=jax.jit(namespace['build_update']('bounded_v2_mean_value_teacher',bounds,'ifql'),backend='cpu')
        mean=jax.jit(changed.build_update('bounded_v2_mean_value_teacher',bounds,'ifql'),backend='cpu')
        states={key:initial for key in('baseline','control','mean')};trace=[];rng=np.random.RandomState(721483)
        for index in range(4):
            indices=rng.randint(len(raw['actions']),size=256)
            batch={k:jnp.asarray(raw[k][indices],jnp.float32)for k in('observations','actions','next_observations','rewards','masks')}
            batch['actions']=jnp.clip(batch['actions'],-1+1e-5,1-1e-5)
            for name,fn in [('baseline',baseline),('control',control),('mean',mean)]:states[name],_=fn(states[name],batch)
            same=difference(states['baseline'],states['control']);assert same['exact'],same
            effect=difference(states['baseline'],states['mean']);assert effect['max_abs']>0
            checks={}
            for name,state in states.items():
                assert int(state.q_updates)==int(state.agent.network.opt_state[0].count)==1000001+index
                assert int(state.v_updates)==int(state.v_opt[0].count)==2000002+2*index
                assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
                checks[name]=dict(BC_params=difference(state.agent.network.params['modules_actor_flow'],states['baseline'].agent.network.params['modules_actor_flow'])['exact'],
                    BC_Adam_mu=difference(state.agent.network.opt_state[0].mu['modules_actor_flow'],states['baseline'].agent.network.opt_state[0].mu['modules_actor_flow'])['exact'],
                    BC_Adam_nu=difference(state.agent.network.opt_state[0].nu['modules_actor_flow'],states['baseline'].agent.network.opt_state[0].nu['modules_actor_flow'])['exact'],
                    agent_RNG=bool(np.array_equal(np.asarray(state.agent.rng),np.asarray(states['baseline'].agent.rng))))
                assert all(checks[name].values()),checks
            trace.append(dict(extra_updates=index+1,min_teacher_reversal_vs_original=same,mean_teacher_effect=effect,checks=checks))
        results.append(dict(run=run.name,environment=identity['environment'],checkpoint_sha256=sha,
            actual_widths=cfg['value_hidden_dims'],trace=trace))
    result=dict(time=time.time(),status='passed',results=results,actual_updates=len(results)*12,
        validation_min_control_generated_by_only_reduction_reversal=True,
        sources={str(p):verifier.digest(p)for p in(Path(__file__),Path(changed.__file__),Path('utils/flow_iql_research/train_mean_value_teacher_v1.py'))},
        new_GPU_training=0,new_policy_evaluations=0,goal_complete=False,
        limits=['Actual full-width1M source states on CPU, four updates per variant; not a performance or GPU bitwise-equivalence claim.',
            'Only the V teacher head reduction is reversed in the validation-only in-memory control; no training file is patched.',
            'Full runner replay/resume must be checked separately; actual late BC equality is not assumed from this short test.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='passed',cases=len(results),actual_updates=result['actual_updates'])),flush=True)


if __name__=='__main__':main()
