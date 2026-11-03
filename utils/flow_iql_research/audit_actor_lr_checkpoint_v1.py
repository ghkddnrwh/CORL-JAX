"""Verify a real LR-only joint checkpoint against its paired raw learner."""
import argparse
import json
from pathlib import Path
import pickle
import time

from utils.flow_iql_research.watch_onestep_deploy_v8 import Verifier, arrays
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--run',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    plan_path=a.root/'registration_v13.json';plan=json.loads(plan_path.read_text())
    state=json.loads((a.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    verifier=Verifier(plan);pins={};saved={};configs={}
    for name in (a.run,a.reference):
        run=a.root/'runs'/name;conf=verifier.config(run,state['identities'][name]);assert conf
        pins[name]=verifier.model(run,conf,a.step);configs[name]=conf
        with (run/f'checkpoint_{a.step}.pkl').open('rb')as f:saved[name]=pickle.load(f)
    left=dict(configs[a.run]['identity']['config']);right=dict(configs[a.reference]['identity']['config'])
    lspec=left.pop('onestep_specification');rspec=right.pop('onestep_specification')
    assert left==right
    assert lspec['actor_lr_multiplier']==2 and rspec['actor_lr_multiplier']==1
    differences={k:[rspec.get(k),v]for k,v in lspec.items()if v!=rspec.get(k)}
    assert set(differences)=={'name','actor_lr_multiplier'}
    base_left=dict(arrays(saved[a.run]['state']['base']))
    base_right=dict(arrays(saved[a.reference]['state']['base']))
    assert base_left.keys()==base_right.keys()
    for k,l in base_left.items():
        r=base_right[k];assert l.dtype==r.dtype and l.shape==r.shape and l.tobytes()==r.tobytes(),k
    actor_left=dict(arrays(saved[a.run]['state']['actor']['params']))
    actor_right=dict(arrays(saved[a.reference]['state']['actor']['params']))
    assert actor_left.keys()==actor_right.keys()
    maximum=max(float(np.max(np.abs(actor_left[k]-actor_right[k])))for k in actor_left)
    assert maximum>0
    probes=[]
    for name in (a.run,a.reference):
        lines=(a.root/'runs'/name/'diagnostics.jsonl').read_text().splitlines()
        probes.append({x['step']:x for x in map(json.loads,lines)if x['step']<=a.step})
    steps=sorted(probes[0].keys()&probes[1].keys())
    assert a.step in steps
    metrics=('q_mean','q_rms','td_rms','BC_loss','critic_loss')
    maximum_base_probe_difference={k:max(abs(probes[0][s][k]-probes[1][s][k])for s in steps)for k in metrics}
    assert all(v==0 for v in maximum_base_probe_difference.values())
    auxiliary={}
    for name in (a.run,a.reference):
        path=a.root/'analysis/evaluations_v1'/f'{name}_{a.step}_primary_n10.json'
        if path.exists():
            job=next(x for x in state['jobs']if x['id']==name)
            auxiliary[name]=verifier.evaluation(path,job,a.step,'primary',10,pins[name])
    output=dict(time=time.time(),status='pass',run=a.run,reference=a.reference,step=a.step,
        source_sha256=sha(__file__),registration_sha256=sha(plan_path),pins=pins,
        configured_actor_learning_rates=[left['lr']*2,right['lr']],
        configuration_changes=differences,serialized_base_leaf_count=len(base_left),
        entire_base_leaf_dtype_shape_bytes_exact=True,
        actor_parameter_max_difference=maximum,matched_diagnostic_steps=len(steps),
        maximum_base_probe_difference=maximum_base_probe_difference,
        actor_MSE_at_step={name:probes[i][a.step]['actor_to_best9_MSE']for i,name in enumerate((a.run,a.reference))},
        auxiliary_CPU10=auxiliary,
        scope='Real saved optimizer and whole lean-base equality check. CPU10 and replay MSE are diagnostic only; no last3/nine-task/fresh/speed efficacy claim.')
    with a.output.open('x')as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items()if k not in ('pins',)}))


if __name__=='__main__':main()
