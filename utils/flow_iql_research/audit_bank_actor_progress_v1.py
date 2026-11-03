"""Read-only root verification of completed bank actors and causal averages.

This is evidence collection, never a native selection or queue mutation.
"""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import time

from utils.flow_iql_research.watch_onestep_deploy_v8 import Verifier, arrays
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import numpy as np


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    plan_path=a.root/'registration_v13.json'
    plan=json.loads(plan_path.read_text())
    state=json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    verifier=Verifier(plan)
    rows=[]
    for job in state['jobs']:
        if job['args']['recipe']!='bank_bc' or job['args']['seed']!=0:
            continue
        rid=job['id']; run=a.root/'runs'/rid
        evdir=a.root/'analysis/evaluations_v1'
        paths=[evdir/f'{rid}_{step}_{mode}_n50.json'
               for step in (800000,900000,1000000) for mode in ('primary','teacher')]
        if not all(p.exists()for p in paths):
            continue
        conf=verifier.config(run,state['identities'][rid]); assert conf
        assert (run/'completed.json').exists()
        points=[]
        for step in (800000,900000,1000000):
            pin=verifier.model(run,conf,step)
            point=dict(step=step,pin=pin)
            for mode in ('primary','teacher'):
                path=evdir/f'{rid}_{step}_{mode}_n50.json'
                point[mode]=verifier.evaluation(path,job,step,mode,50,pin)
            points.append(point)
        row=dict(run=rid,environment=job['args']['env'],points=points,
                 primary_last3=statistics.mean(x['primary']['success_rate']for x in points),
                 teacher_last3=statistics.mean(x['teacher']['success_rate']for x in points))
        rows.append(row)
    assert len({x['environment']for x in rows})==len(rows)
    save(a.output/'native_completed_bank_actors_v1.json',dict(
        time=time.time(),status='pass',registration_sha256=sha(plan_path),
        source_sha256=sha(__file__),completed_environments=len(rows),rows=rows,
        nine_task_complete=len(rows)==9,
        scope='Every completed seed0 bankBC case; missing cases excluded, no subset goal claim.'))

    old_dir=a.root/'analysis/causal_actor_average_CPU_v1'
    new_dir=a.root/'analysis/causal_actor_average_CPU_1M_v1'
    old_audit=json.loads((old_dir/'CAUSAL_AVERAGE_AUDIT_v1.json').read_text())
    manifest=json.loads((new_dir/'manifest_v1.json').read_text())
    merged=[dict(x)for x in old_audit['rows']]
    for case in ('cube','puzzle3'):
        result=json.loads((new_dir/case/'result.json').read_text())
        assert result['status']=='complete'and result['steps']==[1000000]
        assert result['manifest_sha256']==sha(new_dir/'manifest_v1.json')
        entry=result['evaluations']['1000000']
        assert sha(entry['path'])==entry['sha256']
        assert sha(entry['artifact'])==entry['artifact_sha256']
        with Path(entry['artifact']).open('rb')as f:artifact=pickle.load(f)
        assert artifact['averaged_checkpoint_steps']==[800000,900000,1000000]
        assert artifact['new_optimizer_updates']==0
        trees=[]
        for step in (800000,900000,1000000):
            ref=manifest['cases'][case]['artifacts'][str(step)]
            assert sha(ref['actor'])==ref['actor_sha256']
            with Path(ref['actor']).open('rb')as f:original=pickle.load(f)
            assert artifact['source_actor_sha256'][str(step)]==ref['actor_sha256']
            trees.append(dict(arrays(original['actor_params'])))
        mean=dict(arrays(artifact['actor_params']))
        assert all(t.keys()==mean.keys()for t in trees)
        for k,v in mean.items():
            expected=np.mean(np.stack([t[k]for t in trees]),axis=0,dtype=np.float32)
            assert expected.dtype==v.dtype and np.array_equal(expected,v)
        evaluated=json.loads(Path(entry['path']).read_text())
        assert evaluated['n_episodes']==50 and evaluated['first_episode_seed']==42
        assert evaluated['averaged_checkpoint_steps']==[800000,900000,1000000]
        assert evaluated['learned_model_trees_loaded']==['averaged_actor_params']
        assert evaluated['actor_forward_passes']==1
        assert not evaluated['requires_critic']and not evaluated['requires_BC_flow']
        assert evaluated['generated_candidate_actions']==0
        assert all(x in (0,1)for x in evaluated['episode_successes'])
        assert len(evaluated['episode_successes'])==50
        successes=sum(evaluated['episode_successes'])
        ref=manifest['cases'][case]['artifacts']['1000000']
        assert sha(ref['primary_evaluation'])==ref['primary_evaluation_sha256']
        raw=json.loads(Path(ref['primary_evaluation']).read_text())
        assert successes/50==entry['success']==evaluated['success_rate']
        assert sum(raw['episode_successes'])/50==entry['original_actor_success']
        merged.append(dict(case=case,step=1000000,prefix=[800000,900000,1000000],
            average_successes=successes,original_successes=sum(raw['episode_successes']),
            episodes=50,average=successes/50,original=raw['success_rate'],
            actor_sha256=sha(entry['artifact']),evaluation_sha256=sha(entry['path']),
            independently_verified_exact_parameter_mean=True))
    summaries=[]
    for case in ('ant','cubed','hmmed','cube','puzzle3'):
        group=sorted([x for x in merged if x['case']==case],key=lambda x:x['step'])
        assert [x['step']for x in group]==[800000,900000,1000000]
        summaries.append(dict(case=case,
            average_last3=sum(x['average_successes']for x in group)/150,
            raw_last3=sum(x['original_successes']for x in group)/150))
    save(a.output/'causal_average_merged5_audit_v1.json',dict(
        time=time.time(),status='pass',old_audit_sha256=sha(old_dir/'CAUSAL_AVERAGE_AUDIT_v1.json'),
        new_evaluation_episodes=100,total_evaluation_episodes=750,
        old_evaluations_not_repeated=True,rows=merged,case_summaries=summaries,
        success_difference=sum(x['average_successes']-x['original_successes']for x in merged),
        no_per_environment_choice=True,no_nine_task_fresh_or_speed_success_claim=True))
    print(json.dumps(dict(native_cases=len(rows),native_scores=[
        [x['environment'],x['primary_last3'],x['teacher_last3']]for x in rows],
        causal_average=summaries),allow_nan=False))


if __name__=='__main__':main()
