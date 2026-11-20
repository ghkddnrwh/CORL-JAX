"""Register one proper-score student experiment with all existing gates intact."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v4 import MIXTURE,configuration


def build_plan(root):
    parent = root / 'registration_v10.json'
    previous = root / 'analysis/execution_contract_v12.json'
    plan = json.loads(parent.read_text())
    contract = json.loads(previous.read_text())
    assert contract['registration_sha256'] == sha(parent)
    area = root / 'analysis/mixture_onestep_pipeline_v1'
    proposal_path = area / 'joint_experiment_proposal_v1.json'
    proposal = json.loads(proposal_path.read_text())
    assert proposal['parent_registration_sha256'] == sha(parent)
    assert proposal['parent_contract_sha256'] == sha(previous)
    assert proposal['selection_is_exploratory'] and proposal['all_performance_speed_and_fresh_seed_gates_unchanged']
    readiness_path = area / 'readiness_contract_v1.json'
    readiness = json.loads(readiness_path.read_text())
    assert readiness['status'] == 'pass' and readiness['no_performance_claim']
    for item in [*proposal['evidence'],*readiness['evidence']]:
        assert sha(item['path']) == item['sha256']
    for path,digest in readiness['source_sha256'].items():
        assert sha(path) == digest,path
    acceptance = json.loads((area / 'acceptance_source_identity_v1.json').read_text())
    for path,digest in acceptance['draft_source_sha256'].items():
        assert sha(path) == digest,path
    state = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert len(state['jobs']) == 95 and state['jobs'] == plan['jobs']
    assert state['phase'] == 'confirmation' and state['active_attempt'] == 1
    assert state['attempts'][1]['recipe'] == 'bank_bc' and state['attempts'][1]['stage'] == 'discovery'
    assert plan['expected_identities'] == state['identities']
    assert 'mdn9_bc' not in state['all_recipes'] and 'mdn9_bc' not in plan['initial_recipes']
    old = list(plan['jobs'])
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    started = set(scheduler['running']) | set(json.loads((root / 'finished.json').read_text()))
    started |= {j['id'] for j in old if (root / 'runs' / j['id'] / 'config.json').exists()}
    tags = ['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene']
    envs = plan['scheduling_amendment']['all_nine_environments']
    new,expected = [],{}
    for env in proposal['initial_seed0_environments_in_order']:
        rid = 'osm1_' + tags[envs.index(env)] + '_mdn9_bc_s0'
        assert not (root / 'runs' / rid).exists()
        new.append(dict(id=rid,module=MIXTURE.runner_module,args=dict(env=env,recipe='mdn9_bc',seed=0,steps=1000000)))
        expected[rid] = configuration(env,'mdn9_bc',0)
    assert len(new) == 5 and set(plan['pilot_environments']) <= {j['args']['env'] for j in new}
    insertion = max(i for i,j in enumerate(old) if j['args']['recipe'] == 'energy_bc')+1
    plan['jobs'] = old[:insertion] + new + old[insertion:]
    assert [j for j in plan['jobs'] if j['id'] not in expected] == old
    assert len({j['id'] for j in plan['jobs']}) == 100
    plan['expected_identities'].update(expected)
    plan['initial_recipes'].append('mdn9_bc')
    plan['initial_recipe_specs']['mdn9_bc'] = MIXTURE.core.specification('mdn9_bc')
    plan['implementation_definitions']['mixture'] = asdict(MIXTURE)
    plan['implementation_training_source_sha256']['mixture'] = readiness['training_source_sha256']
    directory = Path(__file__).resolve().parent
    names = ['register_mixture_onestep_v1.py','validate_mixture_controlplane_v1.py',
             'onestep_implementation_registry_v4.py','scheduler_onestep_deploy_v6.py',
             'watch_onestep_deploy_v8.py','inspect_onestep_runtime_v5.py',
             'watch_onestep_gradients_v5.py','diagnose_onestep_gradients_v5.py']
    sources = {**plan['all_source_sha256'],**readiness['source_sha256']}
    for path in [*(directory / name for name in names),proposal_path,readiness_path]:
        sources[str(path)] = sha(path)
    assert all(sha(path) == digest for path,digest in sources.items())
    extension = dict(time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),proposal=str(proposal_path),proposal_sha256=sha(proposal_path),
        readiness=str(readiness_path),readiness_sha256=sha(readiness_path),new_jobs=new,new_expected_identities=expected,
        recipe='mdn9_bc',new_implementation_tag='mixture',selection_is_exploratory=True,
        no_formal_promotion_or_fresh_seed_allocation=True,no_joint_mixture_GPU_efficacy_observed=True,
        old95definitions_identities_and_relative_logical_order_preserved=True,old25recipes_preserved=True,
        active_bankBC_discovery_attempt_preserved=1,all_performance_speed_and_fresh_gates_unchanged=True,
        inserted_before_logical_index=insertion,queue_rule=proposal['queue_rule'],
        initial_extra_manipulation_screens=['cube-single-play-singletask-v0','puzzle-3x3-play-singletask-v0'],
        original_three_pilot_environments_unchanged=plan['pilot_environments'])
    plan.update(all_source_sha256=sources,mixture_implementation_extension=extension)
    assert plan['thresholds'] == json.loads(parent.read_text())['thresholds']
    return plan,contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--gradient-contract',type=Path,required=True)
    args = parser.parse_args()
    assert not any(p.exists() for p in (args.registration,args.contract,args.gradient_contract))
    root = args.root.resolve()
    plan,contract = build_plan(root)
    path = root / 'analysis/mixture_onestep_pipeline_v1/controlplane_validation_v1/result.json'
    validation = json.loads(path.read_text())
    assert validation['status'] == 'pass' and validation['no_performance_claim']
    for source,digest in validation['source_sha256'].items():
        assert sha(source) == digest
    plan['mixture_implementation_extension']['integration_validation'] = dict(path=str(path),sha256=sha(path))
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    previous = root / 'analysis/execution_contract_v12.json'
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
        source_sha256=plan['all_source_sha256'],GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v6',
        previous_execution_contract=str(previous),previous_execution_contract_sha256=sha(previous),
        mixture_implementation_extension=plan['mixture_implementation_extension'])
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    old_gradient = root / 'analysis/gradient_execution_contract_v4.json'
    gradient = json.loads(old_gradient.read_text())
    gradient.update(time=time.time(),source_sha256=plan['all_source_sha256'],previous_contract=str(old_gradient),
        previous_contract_sha256=sha(old_gradient),implementation_registry='onestep_implementation_registry_v4',
        execution='V5 producer uses actual NLL/energy/MSE distillation objective and implementation-specific noise dimension. All old gradient math preserved, live old producers adopted. Teacher minQ and optional guidance meanQ recorded separately.',
        implementation_validation=[*gradient.get('implementation_validation',[]),dict(path=str(path),sha256=sha(path),no_performance_claim=True)])
    args.gradient_contract.write_text(json.dumps(gradient,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
        contract=str(args.contract),contract_sha256=sha(args.contract),gradient_contract=str(args.gradient_contract),
        gradient_contract_sha256=sha(args.gradient_contract),jobs=100,new_jobs=5)))


if __name__ == '__main__':
    main()
