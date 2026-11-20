"""Root audit of complete symmetric set-actor pilots and paired lean controls."""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import time
from utils.flow_iql_research.watch_onestep_deploy_v8 import Verifier, arrays
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    plan_path=a.root/'registration_v13.json';plan=json.loads(plan_path.read_text())
    state=json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    verifier=Verifier(plan);rows=[]
    for env in plan['pilot_environments']:
        entries={recipe:next(j for j in state['jobs']if j['args']['recipe']==recipe and
                    j['args']['env']==env and j['args']['seed']==0)for recipe in ('set_bc','bank_bc')}
        points=[]
        for step in (800000,900000,1000000):
            loaded={};point=dict(step=step,models={})
            teacher_evaluations={}
            for recipe,job in entries.items():
                run=a.root/'runs'/job['id'];conf=verifier.config(run,state['identities'][job['id']]);assert conf
                pin=verifier.model(run,conf,step)
                with (run/f'checkpoint_{step}.pkl').open('rb')as f:loaded[recipe]=pickle.load(f)
                scores={}
                for mode in ('primary','teacher'):
                    path=a.root/'analysis/evaluations_v1'/f"{job['id']}_{step}_{mode}_n50.json"
                    scores[mode]=verifier.evaluation(path,job,step,mode,50,pin)
                    if mode=='teacher':teacher_evaluations[recipe]=json.loads(path.read_text())
                point['models'][recipe]=dict(pin=pin,evaluations=scores)
            left=dict(arrays(loaded['set_bc']['state']['base']))
            right=dict(arrays(loaded['bank_bc']['state']['base']))
            assert left.keys()==right.keys()
            for k,v in left.items():assert v.dtype==right[k].dtype and v.shape==right[k].shape and v.tobytes()==right[k].tobytes(),k
            for key in ('episode_successes','episode_rewards','episode_lengths'):
                assert teacher_evaluations['set_bc'][key]==teacher_evaluations['bank_bc'][key]
            point.update(all_base_leaves_dtype_shape_bytes_exact=True,teacher_CPU50_episode_arrays_exact=True)
            points.append(point)
        means={recipe:statistics.mean(x['models'][recipe]['evaluations']['primary']['success_rate']for x in points)for recipe in entries}
        teacher=statistics.mean(x['models']['set_bc']['evaluations']['teacher']['success_rate']for x in points)
        rows.append(dict(environment=env,points=points,primary_last3=means,teacher_last3=teacher))
    set_score=statistics.mean(r['primary_last3']['set_bc']for r in rows)
    control=statistics.mean(r['primary_last3']['bank_bc']for r in rows)
    teacher=statistics.mean(r['teacher_last3']for r in rows)
    thresholds=plan['thresholds']
    checks=dict(paired_pilot_mean=set_score>=teacher-thresholds['mean_loss_tolerance']-1e-12,
        historical_pilot_mean=set_score>=thresholds['pilot_historical_mean_reference']-thresholds['mean_loss_tolerance']-1e-12)
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        registration_sha256=sha(plan_path),source_sha256=sha(__file__),
        set_pilot_mean=set_score,bank_pilot_mean=control,teacher_pilot_mean=teacher,
        original_pilot_screen_checks=checks,pilot_screen_passed=all(checks.values()),
        all9paired_base_states_bytes_exact=True,all9teacher_episode_arrays_exact=True,
        no_queue_or_attempt_changes=True,
        scope='Controlled three-pilot evidence only. Current bankBC full9 attempt and all success gates unchanged; no set full9/fresh/latency goal claim.'),f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',set_pilot_mean=set_score,bank_pilot_mean=control,
        teacher_mean=teacher,checks=checks,all9paired_bases_byte_exact=True)))


if __name__=='__main__':main()
