"""Analysis-only fixtures and real eight-update checkpoint audit; no scores."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import watch_efficiency_v1 as watcher
from utils.flow_iql_research import efficient_mean_core_v1 as core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    root=Path('logs/diagnosis/finite_mean_efficiency_20261030').resolve()
    output=root/'analysis/controller_validation_v1.json'
    assert not output.exists()
    plan=json.loads((root/'registration_v1.json').read_text())
    verifier=watcher.Verifier(plan)
    receipts=[]
    for variant in ('reference','fql','student_guided'):
        run=root/'analysis/runner_smoke_v1'/variant
        conf=json.loads((run/'config.json').read_text())
        assert verifier.config(run,conf['identity']) is not None
        pin=verifier.model(run,conf,8)
        with (run/'checkpoint_8.pkl').open('rb') as stream:saved=pickle.load(stream)
        cfg=dict(conf['identity']['config'])
        for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
        dataset=json.loads((run/'dataset.json').read_text())
        obs=jnp.zeros((1,dataset['observation_dim']))
        act=jnp.zeros((1,dataset['action_dim']))
        state=flax.serialization.from_state_dict(core.create(0,obs,act,cfg,variant),saved['state'])
        for mode in ('primary','teacher'):
            policy=core.make_policy(state,variant,mode)
            action=np.asarray(policy(core.policy_parameters(state),obs[0],jax.random.PRNGKey(42)))
            assert action.shape==(dataset['action_dim'],) and np.isfinite(action).all() and np.max(np.abs(action))<=1
        receipts.append(dict(variant=variant,real_checkpoint_Q_updates=8,checkpoint_sha256=pin['sha256'],
            finite_primary_and_teacher_CPU_actions=True))
    pilots=plan['pilot_environments'];criteria=plan['promotion_rule']
    groups=[];timing={}
    for env in pilots:
        for variant,train,inference in [('reference',10.,4.),('bc_student',7.,1.),('guided_actor',9.,.5)]:
            groups.append(dict(variant=variant,environment=env,seed=0,complete=True,last3_mean=.6 if variant=='reference' else .59))
            timing[(variant,env)]=dict(benchmark_state_finite=True,measurement_GPU='fixture',measurement_CPUs=[0],
                training_ms_per_update_median=train,cpu_action_ms_median=inference)
    selected,evidence=watcher.select_candidate(groups,timing,pilots,criteria)
    assert selected['variant']=='bc_student'
    assert not next(row for row in evidence if row['variant']=='guided_actor')['eligible']
    bad=copy.deepcopy(groups)
    next(row for row in bad if row['variant']=='bc_student')['last3_mean']=.1
    assert watcher.select_candidate(bad,timing,pilots,criteria)[0] is None
    missing=[row for row in groups if not(row['variant']=='bc_student' and row['environment']==pilots[0])]
    assert watcher.select_candidate(missing,timing,pilots,criteria)[0] is None
    unfinished=copy.deepcopy(groups)
    next(row for row in unfinished if row['variant']=='bc_student')['complete']=False
    assert watcher.select_candidate(unfinished,timing,pilots,criteria)[0] is None
    nonfinite=copy.deepcopy(timing)
    nonfinite[('bc_student',pilots[0])]['benchmark_state_finite']=False
    assert watcher.select_candidate(groups,nonfinite,pilots,criteria)[0] is None
    wrong_device=copy.deepcopy(timing)
    wrong_device[('bc_student',pilots[0])]['measurement_GPU']='different'
    try:watcher.select_candidate(groups,wrong_device,pilots,criteria)
    except AssertionError:pass
    else:raise AssertionError('Different-device comparison accepted')
    result=dict(time=time.time(),status='passed',real_smoke_model_audits=receipts,
        selector_fixture_cases=['eligible_balanced_speedup','reject_slow_training','reject_task_collapse',
            'withhold_missing_task','withhold_unfinished_task','reject_nonfinite_benchmark','reject_mismatched_device'],
        scope='Synthetic selector fixtures are explicitly NOT performance results. Real saved models have eight updates only.',
        source_sha256={str(Path(module.__file__).resolve()):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (watcher,core)})
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
