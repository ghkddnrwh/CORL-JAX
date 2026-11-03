"""Actual saved-model routing plus synthetic selection fixtures; no efficacy claim."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from utils.flow_iql_research import watch_onestep_deploy_v4 as current
from utils.flow_iql_research import watch_onestep_deploy_v2 as macro
from utils.flow_iql_research import watch_onestep_deploy_v3 as exploration
from utils.flow_iql_research.register_onestep_implementations_v1 import build_plan
from utils.flow_iql_research.onestep_implementation_registry_v1 import implementation,configuration


def rejected(call):
    try:call()
    except (AssertionError,KeyError):return True
    raise AssertionError('Invalid fixture accepted')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not args.output_dir.exists()
    args.output_dir.mkdir(parents=True);root=args.root.resolve();plan,_=build_plan(root);checks=[]
    verifier=current.Verifier(plan);run=root/'runs/os1_cubed_fqln_a1_s0'
    identity=plan['expected_identities'][run.name];conf=verifier.config(run,identity);assert conf
    for step in current.STEPS:
        pin=verifier.model(run,conf,step)
        job=next(j for j in plan['jobs'] if j['id']==run.name)
        for mode in ('primary','teacher'):
            path=root/'analysis/evaluations_v1'/f'{run.name}_{step}_{mode}_n50.json'
            verifier.evaluation(path,job,step,mode,50,pin)
    checks.append('Actual flat 1M model, all three CPU50 checkpoints, source/archive/counters/export/evaluation routing')
    private=root/'analysis/set_actor_pipeline_v1/validation_run_v1/set_bc/full8'
    conf=json.loads((private/'config.json').read_text());set_verifier=current.Verifier(plan)
    actual=set_verifier.config(private,conf['identity']);assert actual
    pin=set_verifier.model(private,actual,8);assert pin['metadata']['schema']=='set_actor_deployment_v1'
    checks.append('Actual set 8-update model, source/archive/counters/export/schema routing; not efficacy')
    wrong=copy.deepcopy(plan);wrong['implementation_training_source_sha256']['set']=wrong['implementation_training_source_sha256']['flat']
    assert rejected(lambda:current.Verifier(wrong).config(private,conf['identity']))
    checks.append('Reject set model with flat training-source map')
    smoke=private.parent/'evaluation_primary.json';job=dict(id=private.name,args=dict(recipe='set_bc',env=conf['identity']['environment'],seed=6))
    assert rejected(lambda:set_verifier.evaluation(smoke,job,8,'primary',10,pin))
    checks.append('Reject implementation-only episode protocol as production evidence')
    fixture=json.loads(smoke.read_text());fixture['protocol']='auxiliary_cpu10_100K300K_screening_v1'
    fixture['test_fixture_only']=True
    path=args.output_dir/'FIXTURE_ONLY_set_source_routing.json';path.write_text(json.dumps(fixture))
    set_verifier.evaluation(path,job,8,'primary',10,pin)
    checks.append('Set evaluator/core SHA route using actual8 fields and a PRIVATE modified protocol fixture, not performance')
    bad=copy.deepcopy(fixture);bad['source_sha256']=plan['all_source_sha256'][str(implementation('bank_bc').evaluator_path)]
    wrong_path=args.output_dir/'FIXTURE_ONLY_wrong_evaluator.json';wrong_path.write_text(json.dumps(bad))
    assert rejected(lambda:set_verifier.evaluation(wrong_path,job,8,'primary',10,pin))
    checks.append('Reject flat evaluator SHA for set model')
    assert configuration(conf['identity']['environment'],'set_bc',6,steps=8,chunk=2,probe_every=2,train_device='cpu')==conf['identity']
    checks.append('Set runner identity matches actual CPU validation config')
    rows=[]
    for environment,primary,teacher in zip(plan['pilot_environments'],(.30,.85,.70),(.80,.65,.40)):
        rows.append(dict(recipe='bank_bc',environment=environment,seed=0,complete=True,
            primary_mean=primary,teacher_mean=teacher,actor_parameter_count=844296))
    before=macro.pilot_candidates(rows,['bank_bc'],plan)[0]
    after=current.pilot_candidates(rows,['bank_bc'],plan)[0]
    assert before['eligible']==after['eligible']==True and before['pilot_primary_mean']==after['pilot_primary_mean']
    checks.append('Preserve macro acceptance despite large diagnostic per-task loss')
    candidates=[dict(recipe='bank_bc',eligible=True,pilot_primary_mean=.60,actor_parameter_count=200,objective='none'),
        dict(recipe='fqln_a03',eligible=True,pilot_primary_mean=.605,actor_parameter_count=300,objective='maximize')]
    for tested,eligible in [([],True),([],False),(['bank_bc'],False),(['bank_bc','fqln_a03'],False)]:
        group=[dict(x,eligible=eligible) for x in candidates]
        x=exploration.pick_candidate(group,tested);y=current.pick_candidate(group,tested)
        assert (None if x is None else (x['recipe'],x['selection_kind']))==(None if y is None else (y['recipe'],y['selection_kind']))
    checks.append('Preserve qualified, diverse exploratory, and maximum-two exploration selection')
    attempt=dict(recipe='bank_bc',fresh_seeds=[12,13,14]);confirmation=[]
    for seed in [0,12,13,14]:
        for index,environment in enumerate(current.core.ENVIRONMENTS):
            confirmation.append(dict(recipe='bank_bc',environment=environment,seed=seed,complete=True,
                primary_mean=0. if index==0 else .8,teacher_mean=.68))
    old=macro.full_confirmation(confirmation,attempt,plan);new=current.full_confirmation(confirmation,attempt,plan)
    assert old['checks']==new['checks'] and old['performance_passed']==new['performance_passed']==True
    assert not new['diagnostic_task_loss_cap_satisfied']
    low=[dict(x,primary_mean=.4) for x in confirmation]
    assert not current.full_confirmation(low,attempt,plan)['performance_passed']
    assert current.full_confirmation(confirmation[:-1],attempt,plan) is None
    checks.append('Preserve full9/fresh macro gates; reject low/missing confirmation, no hard task cap')
    output=args.output_dir/'result.json';output.write_text(json.dumps(dict(time=time.time(),status='pass',checks=checks,
        no_performance_claim=True,scope='Actual flat production files and actual set8 implementation artifacts; explicitly private synthetic criterion/protocol fixtures. No GPU jobs or queue changes.'),indent=2))
    print(json.dumps(dict(status='pass',checks=len(checks),result=str(output))))


if __name__=='__main__':main()
