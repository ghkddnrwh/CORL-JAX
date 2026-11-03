"""Prospective remaining-four coverage for a native mixture actor; gates fixed."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v5 import configuration,implementation


def main():
    root=Path(sys.argv[1]).resolve();area=root/'analysis/mixture_full9_prefetch_v1';area.mkdir(exist_ok=True)
    parent=root/'registration_v15.json';previous=root/'analysis/execution_contract_v17.json'
    target=root/'registration_v16.json';execution=root/'analysis/execution_contract_v18.json'
    assert not target.exists()and not execution.exists()
    plan=json.loads(parent.read_text());contract=json.loads(previous.read_text())
    assert contract['registration_sha256']==sha(parent)
    state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert state['jobs']==plan['jobs']and len(state['jobs'])==138
    assert state['phase']=='confirmation'and state['active_attempt']==2 and state['attempts'][2]['recipe']=='set_bc'
    evidence=root/'analysis/mixture_onestep_pipeline_v1/NATIVE_THREE_HARD_TASKS_ROOT_AUDIT_v1.json'
    proof=json.loads(evidence.read_text());assert proof['status']=='pass'and len(proof['rows'])==3
    assert proof['no_full9_or_fresh_or_speed_claim']and not proof['original_pilot_screen_complete']
    for row in proof['rows']:
        for point in row['points']:assert sha(point['path'])==point['sha256']
    recipe='mdn9_bc';existing=[j for j in state['jobs']if j['args']['recipe']==recipe and j['args']['seed']==0]
    assert len(existing)==5
    specs=[('hmlarge','humanoidmaze-large-navigate-singletask-v0'),('soccer','antsoccer-arena-navigate-singletask-v0'),
           ('pz4','puzzle-4x4-play-singletask-v0'),('scene','scene-play-singletask-v0')]
    new=[];expected={}
    for tag,env in specs:
        assert env not in {j['args']['env']for j in existing}
        rid=f'ospM_{tag}_mdn9_bc_s0';assert not(root/'runs'/rid).exists()
        new.append(dict(id=rid,module=implementation(recipe).runner_module,args=dict(env=env,recipe=recipe,seed=0,steps=1000000)))
        expected[rid]=configuration(env,recipe,0)
        spec=expected[rid]['specification'];assert spec['objective']=='none'and spec['mixture_components']==9 and spec['std_floor']==.001
    assert {j['args']['env']for j in [*existing,*new]}==set(plan['scheduling_amendment']['all_nine_environments'])
    old=list(plan['jobs']);index=max(i for i,j in enumerate(old)if j['id'].startswith('ospS_'))+1
    plan['jobs']=old[:index]+new+old[index:]
    assert [j for j in plan['jobs']if j['id']not in expected]==old and len(plan['jobs'])==142
    plan['expected_identities'].update(expected)
    extension=dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),recipe=recipe,new_jobs=new,new_expected_identities=expected,
        added_recipes=[],evidence=dict(path=str(evidence),sha256=sha(evidence)),
        selection_is_exploratory=True,no_original_pilot_eligibility_claim=True,no_formal_candidate_promotion=True,
        rationale='Actual native1M clipped-mixtureNLL actor preserves Cube .92, Puzzle .8133 and CubeD .2533. Fill remainingfour tasks now; fixed-teacherCPU negative screens do not replace jointlearning evidence.',
        queue_rule='Prioritize four unstarted full9mixture jobs, preserve138old definitions/identities/relative order and every started position.',
        hyperparameters='One globalK9/std_floor.001/unweightedclippedNLL andofficialdefaultwidths; no per-environment recipe tuning.',
        all_final_performance_fresh_and_speed_gates_unchanged=True,active_SetBC_attempt_preserved=2,
        no_fresh_seed_allocation=True,no_new_algorithm_sources=True)
    proposal=area/'prospective_experiment_v1.json';assert not proposal.exists();proposal.write_text(json.dumps(extension,indent=2,allow_nan=False))
    sources=dict(plan['all_source_sha256'])
    for p in [Path(__file__).resolve(),Path(__file__).with_name('activate_mixture_full9_prefetch_v1.py').resolve(),proposal,evidence]:sources[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in sources.items())
    plan.update(all_source_sha256=sources,mixture_full9_prefetch_extension=extension)
    assert plan['thresholds']==json.loads(parent.read_text())['thresholds']
    target.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(target),registration_sha256=sha(target),source_sha256=sources,
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),mixture_full9_prefetch_extension=extension)
    execution.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(status='prepared_not_activated',registration=str(target),registration_sha256=sha(target),
        execution=str(execution),execution_sha256=sha(execution),new_jobs=4,jobs=142)))


if __name__=='__main__':main()
