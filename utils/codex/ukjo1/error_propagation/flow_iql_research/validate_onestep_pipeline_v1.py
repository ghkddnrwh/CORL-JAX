"""Actual export audit plus synthetic selection/confirmation edge cases."""
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import watch_onestep_deploy_v1 as watch
from utils.flow_iql_research import onestep_deploy_core_v1 as core


def main():
    root=Path('logs/diagnosis/finite_mean_onestep_20261031').resolve();output=root/'analysis/controller_validation_v1.json'
    assert not output.exists();plan=json.loads((root/'registration_v1.json').read_text());verifier=watch.Verifier(plan)
    actual=[]
    for recipe in ('bank_bc','fqln_a1'):
        run=root/'analysis/runner_smoke_v1'/f'{recipe}_full';conf=json.loads((run/'config.json').read_text())
        assert verifier.config(run,conf['identity'])is not None;pin=verifier.model(run,conf,8)
        assert pin['exported_actor_params_exact'] and pin['actual_Q_Adam']==pin['actual_actor_Adam']==8
        actual.append(dict(recipe=recipe,pin=pin))
    rows=[]
    for env in plan['pilot_environments']:
        for recipe,loss in [('bank_bc',.01),('bank_q_a1',.005),('fqln_a1',.25)]:
            rows.append(dict(recipe=recipe,environment=env,seed=0,complete=True,primary_mean=plan['pilot_reference'][env]-loss,
                teacher_mean=plan['pilot_reference'][env],actor_parameter_count=100))
    candidates=watch.pilot_candidates(rows,['bank_bc','bank_q_a1','fqln_a1'],plan)
    assert watch.pick_candidate(candidates,[])['recipe']=='bank_bc'
    assert watch.pick_candidate(candidates,['bank_bc'])['recipe']=='bank_q_a1'
    assert watch.pick_candidate(candidates,['bank_bc','bank_q_a1'])is None
    assert not next(r for r in candidates if r['recipe']=='fqln_a1')['eligible']
    missing=[r for r in rows if not(r['recipe']=='bank_bc'and r['environment']==plan['pilot_environments'][0])]
    assert 'bank_bc'not in [r['recipe']for r in watch.pilot_candidates(missing,['bank_bc'],plan)]
    attempt=dict(recipe='bank_bc',fresh_seeds=[12,13,14]);full=[]
    for seed in (0,12,13,14):
        for env in core.ENVIRONMENTS:
            full.append(dict(recipe='bank_bc',environment=env,seed=seed,complete=True,primary_mean=.6 if seed==0 else .59,teacher_mean=.6))
    assert watch.full_confirmation(full,attempt,plan)['performance_passed']
    assert watch.full_confirmation(full[:-1],attempt,plan)is None
    damaged=copy.deepcopy(full)
    for r in damaged:
        if r['seed']!=0 and r['environment']==core.ENVIRONMENTS[0]:r['primary_mean']=0
    result=watch.full_confirmation(damaged,attempt,plan)
    assert not result['performance_passed']and not result['checks']['fresh_max_task_loss']
    damaged=copy.deepcopy(full)
    for r in damaged:
        if r['seed']==0:r['primary_mean']=.58
    assert not watch.full_confirmation(damaged,attempt,plan)['checks']['seed0_historical_mean']
    result=dict(time=time.time(),status='passed',actual8update_model_and_actor_export_audits=actual,
        selector_cases=['paired_loss_gate','historical_loss_gate','simplicity_tie','tested_candidate_exclusion','missing_task_withheld'],
        confirmation_cases=['full9plus3freshseeds_pass','missing_environment_withheld','reject_one_task_collapse','reject_historical_seed0_loss'],
        scope='Synthetic selection fixtures are not experiment performance. Real model tests contain8updates only.',
        source_sha256={str(Path(watch.__file__).resolve()):watch.sha(watch.__file__),str(Path(core.__file__).resolve()):watch.sha(core.__file__)})
    output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(status='passed',actual_saved_model_audits=len(actual),selection_fixture_cases=len(result['selector_cases'])+len(result['confirmation_cases'])),indent=2))


if __name__=='__main__':main()
