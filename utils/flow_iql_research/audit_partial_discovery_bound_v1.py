"""Read-only deterministic bound on a partially observed fixed empirical gate.

Every missing primary CPU50 point is allowed fifty successes. This is not a
population bound, a forecast, a queue transition, or a reason to kill runs.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from utils.flow_iql_research.watch_onestep_deploy_v8 import Verifier
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--recipe',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    plan_path=a.root/'registration_v13.json';plan=json.loads(plan_path.read_text())
    state=json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    verifier=Verifier(plan);rows=[];missing=[]
    jobs=[x for x in state['jobs']if x['args']['recipe']==a.recipe and x['args']['seed']==0]
    assert len(jobs)==9 and len({x['args']['env']for x in jobs})==9
    for job in jobs:
        run=a.root/'runs'/job['id']
        for step in (800000,900000,1000000):
            path=a.root/'analysis/evaluations_v1'/f"{job['id']}_{step}_primary_n50.json"
            if not path.exists():
                missing.append(dict(run=job['id'],step=step));continue
            conf=verifier.config(run,state['identities'][job['id']]);assert conf
            pin=verifier.model(run,conf,step)
            score=verifier.evaluation(path,job,step,'primary',50,pin)
            rows.append(dict(run=job['id'],environment=job['args']['env'],step=step,**score))
    denominator=9*3*50;observed=sum(x['successes']for x in rows)
    remaining=50*len(missing);upper=Fraction(observed+remaining,denominator)
    t=plan['thresholds'];threshold=t['historical_lean_seed0_nine_task_reference']-t['mean_loss_tolerance']
    minimum=math.ceil((threshold-1e-12)*denominator)
    output=dict(time=time.time(),status='verified_partial_metric_bound',recipe=a.recipe,
        source_sha256=sha(__file__),registration_sha256=sha(plan_path),
        observed_points=len(rows),rows=rows,missing_points=missing,
        observed_successes=observed,remaining_best_case_successes=remaining,
        denominator=denominator,upper_bound_exact=str(upper),upper_bound=float(upper),
        unchanged_historical_minimum=threshold,minimum_total_successes_to_pass=minimum,
        successes_needed_from_remaining=max(0,minimum-observed),
        impossible_even_if_all_missing_perfect=observed+remaining<minimum,
        no_state_queue_model_or_gate_mutations=True,
        scope='Deterministic bound on this seed0 fixed empirical last3CPU50 metric only. Not a predicted score, completed9task mean, population claim, or learner termination instruction.')
    with a.output.open('x')as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:output[k]for k in ['recipe','observed_points','observed_successes',
        'remaining_best_case_successes','upper_bound','unchanged_historical_minimum',
        'successes_needed_from_remaining','impossible_even_if_all_missing_perfect']}))


if __name__=='__main__':main()
