"""Root review of a complete common recipe: raw27CPU50points per policy."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
from utils.flow_iql_research.watch_onestep_deploy_v9 import Verifier
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--recipe',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();plan=json.loads(a.registration.read_text())
    state=json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    jobs=[j for j in state['jobs']if j['args']['recipe']==a.recipe and j['args']['seed']==a.seed]
    assert len(jobs)==9 and len({j['args']['env']for j in jobs})==9
    assert {j['args']['env']for j in jobs}==set(plan['scheduling_amendment']['all_nine_environments'])
    verifier=Verifier(plan);rows=[];points=[]
    for job in jobs:
        run=a.root/'runs'/job['id'];assert(run/'completed.json').exists()
        assert state['identities'][job['id']]==plan['expected_identities'][job['id']]
        conf=verifier.config(run,state['identities'][job['id']]);assert conf
        counts=dict(primary=0,teacher=0);snapshots=[]
        for step in (800000,900000,1000000):
            pin=verifier.model(run,conf,step);snapshots.append(pin)
            for mode in ('primary','teacher'):
                path=a.root/'analysis/evaluations_v1'/f"{job['id']}_{step}_{mode}_n50.json"
                score=verifier.evaluation(path,job,step,mode,50,pin)
                counts[mode]+=score['successes'];points.append(dict(run=job['id'],step=step,mode=mode,path=str(path),**score))
        primary,teacher=Fraction(counts['primary'],150),Fraction(counts['teacher'],150)
        rows.append(dict(run=job['id'],environment=job['args']['env'],primary_mean=float(primary),teacher_mean=float(teacher),
            primary_successes=counts['primary'],teacher_successes=counts['teacher'],paired_difference=float(primary-teacher),
            snapshots=snapshots))
    primary=Fraction(sum(r['primary_successes']for r in rows),1350)
    teacher=Fraction(sum(r['teacher_successes']for r in rows),1350);t=plan['thresholds']
    gates=dict(paired_macro=float(primary)>=float(teacher)-t['mean_loss_tolerance']-1e-12,
        historical_lean_macro=float(primary)>=t['historical_lean_seed0_nine_task_reference']-t['mean_loss_tolerance']-1e-12)
    result=dict(time=time.time(),status='verified_complete_last3',recipe=a.recipe,training_seed=a.seed,
        source_sha256=sha(__file__),registration_sha256=sha(a.registration),
        primary_nine_task_mean=float(primary),primary_nine_task_mean_exact=str(primary),
        teacher_nine_task_mean=float(teacher),teacher_nine_task_mean_exact=str(teacher),paired_difference=float(primary-teacher),
        unchanged_gates=gates,discovery_performance_passed=all(gates.values()),rows=rows,raw_points=points,
        actual_evaluated_episodes=2700,actual_models=27,actual_primary_exports=27,
        fresh_training_seeds_in_this_review=0,isolated_speed_in_this_review=False,
        scope='One common recipe, nine actual environments, every800K900K1MCPU50 point from model/actor-hash-verified artifacts. No task/peak mixing. Single trainingseed, no final-goal claim.')
    with a.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k]for k in ['recipe','primary_nine_task_mean','teacher_nine_task_mean','paired_difference','unchanged_gates','discovery_performance_passed']}))


if __name__=='__main__':main()
