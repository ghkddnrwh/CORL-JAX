"""Pure/private scheduling and unchanged acceptance-gate validation."""
import argparse
import ast
import copy
import json
from pathlib import Path
import time
from utils.flow_iql_research import watch_onestep_deploy_v4 as previous
from utils.flow_iql_research import watch_onestep_deploy_v5 as candidate
from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending,discovery_confirmation
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    plan=json.loads((args.root/'registration_v2.json').read_text())
    state=json.loads((args.root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    queue=json.loads((args.root/'queue.json').read_text());scheduler=json.loads((args.root/'scheduler_status.json').read_text())
    assert state['phase']=='pilot'and not state['attempts']
    started=set(scheduler['running'])|set(json.loads((args.root/'finished.json').read_text()))
    started|={job['id']for job in state['jobs']if(args.root/'runs'/job['id']/'config.json').exists()}
    checks=[]
    ordered=order_pending(plan,state,queue,started)
    assert ordered!=queue and order_pending(plan,state,ordered,started)==ordered
    assert {job['id']:job for job in ordered}=={job['id']:job for job in queue}
    assert all(left==right for left,right in zip(queue,ordered)if left['id']in started)
    first=next(job for job in ordered if job['id']not in started)
    assert first['args']['recipe']=='bank_bc'and first['args']['env']==plan['pilot_environments'][2]
    checks.append('Actual48-job snapshot: prioritize missing pilot coverage; preserve every job definition and started position; idempotent order')
    simulated=copy.deepcopy(state);simulated['phase']='confirmation'
    attempt=dict(number=0,recipe='bank_bc',fresh_seeds=[12,13,14],stage='discovery')
    simulated['attempts']=[attempt];simulated['active_attempt']=0
    current=set(started)
    current|={job['id']for job in state['jobs']if job['args']['recipe']=='bank_bc'}
    new=[]
    for env in candidate.core.ENVIRONMENTS:
        if env in plan['pilot_environments']:continue
        new.append(dict(id=f'private_seed0_{len(new)}',module='utils.flow_iql_research.train_onestep_deploy_v1',args=dict(env=env,recipe='bank_bc',seed=0,steps=1000000)))
    assert len(new)==6;simulated['jobs']+=new;discovery_queue=queue+new
    ordered=order_pending(plan,simulated,discovery_queue,current)
    pending=[job for job in ordered if job['id']not in current]
    assert {job['id']for job in pending[:6]}=={job['id']for job in new}
    checks.append('Discovery stage prioritizes remaining6 seed0 environments; no fresh seeds registered yet')
    current|={job['id']for job in new};attempt['stage']='fresh';fresh=[]
    for seed in attempt['fresh_seeds']:
        for env in candidate.core.ENVIRONMENTS:
            fresh.append(dict(id=f'private_fresh_{len(fresh)}',module='utils.flow_iql_research.train_onestep_deploy_v1',args=dict(env=env,recipe='bank_bc',seed=seed,steps=1000000)))
    simulated['jobs']+=fresh;ordered=order_pending(plan,simulated,discovery_queue+fresh,current)
    assert {job['id']for job in[j for j in ordered if j['id']not in current][:27]}=={job['id']for job in fresh}
    checks.append('Fresh stage prioritizes all27 new runs with mandatory3seeds and9tasks')
    rows=[dict(recipe='bank_bc',environment=env,seed=seed,complete=True,primary_mean=.60,teacher_mean=.61)for seed in [0,12,13,14]for env in candidate.core.ENVIRONMENTS]
    discovery=discovery_confirmation(rows,attempt,plan,candidate.core.ENVIRONMENTS)
    whole=previous.full_confirmation(rows,attempt,plan)
    assert discovery['discovery_passed']and whole['performance_passed']
    assert discovery['checks']=={key:whole['checks'][key]for key in discovery['checks']}
    low=copy.deepcopy(rows)
    for row in low:
        if row['seed']==0:row['primary_mean']=.58
    discovery=discovery_confirmation(low,attempt,plan,candidate.core.ENVIRONMENTS)
    whole=previous.full_confirmation(low,attempt,plan)
    assert not discovery['discovery_passed']and not whole['performance_passed']
    assert discovery['checks']=={key:whole['checks'][key]for key in discovery['checks']}
    incomplete=[row for row in rows if not(row['seed']==0 and row['environment']==candidate.core.ENVIRONMENTS[0])]
    assert discovery_confirmation(incomplete,attempt,plan,candidate.core.ENVIRONMENTS)is None
    assert candidate.full_confirmation(incomplete,attempt,plan)is None
    checks.append('Discovery reuses the original paired/historical seed0 macro gates; reject low or missing data; fresh gates remain mandatory')
    examples=[dict(recipe='bank_bc',eligible=False,pilot_primary_mean=.40,pilot_teacher_mean=.60,actor_parameter_count=100,objective='none'),
              dict(recipe='fqln_a1',eligible=False,pilot_primary_mean=.39,pilot_teacher_mean=.60,actor_parameter_count=100,objective='maximize')]
    assert candidate.select_available(examples,[],False)is None
    qualified=copy.deepcopy(examples);qualified[1]['eligible']=True
    picked=candidate.select_available(qualified,[],False);assert picked['recipe']==previous.qualified_pick(qualified,[])['recipe']=='fqln_a1'
    first=candidate.select_available(examples,[],True);assert first['recipe']==previous.pick_candidate(examples,[])['recipe']
    second=candidate.select_available(examples,[first['recipe']],True)
    assert previous.family(first['recipe'])!=previous.family(second['recipe'])
    assert candidate.select_available(examples,[first['recipe'],second['recipe']],True)is None
    checks.append('Early selection requires a completed eligible3task pilot; exploratory promotion still waits for the full round and keeps the original maximum2/diversity rule')
    def definitions(module):
        return {node.name:ast.dump(node,include_attributes=False)for node in ast.parse(Path(module.__file__).read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    old,new=definitions(previous),definitions(candidate)
    unchanged=['Verifier','pilot_candidates','qualified_pick','family','pick_candidate','full_confirmation']
    assert all(old[name]==new[name]for name in unchanged)
    checks.append('Verifier, pilot eligibility, qualified selection, family/diversity/exploration and full9/fresh acceptance definitions are AST-identical to activeV4')
    for path,digest in plan['all_source_sha256'].items():assert sha(path)==digest,path
    checks.append('All previously frozen V2 registration sources and algorithm implementations remain unchanged')
    sources=[Path(__file__).resolve(),Path(candidate.__file__),Path(__file__).with_name('onestep_scheduling_policy_v1.py'),Path(__file__).with_name('scheduler_onestep_queue_v1.py'),Path(__file__).with_name('scheduler_onestep_deploy_v3.py')]
    result=dict(time=time.time(),status='pass',checks=checks,source_sha256={str(path.resolve()):sha(path)for path in sources},
        no_performance_claim=True,no_production_queue_change=True,scope='Pure/private policy fixtures and a read-only actual queue snapshot; no real learner/controller or evaluation changes.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',checks=len(checks),output=str(args.output))),flush=True)


if __name__=='__main__':main()
