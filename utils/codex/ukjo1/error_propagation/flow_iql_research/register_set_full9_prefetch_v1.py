"""Versioned exploratory Set-BC coverage; no algorithm or acceptance changes."""
import argparse
import copy
import json
from pathlib import Path
import time

from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.onestep_implementation_registry_v2 import configuration, implementation


def build_plan(root):
    parent = root / 'registration_v8.json'
    previous = root / 'analysis/execution_contract_v10.json'
    plan = json.loads(parent.read_text())
    contract = json.loads(previous.read_text())
    assert contract['registration_sha256'] == sha(parent)
    proposal_path = root / 'analysis/set_full9_prefetch_v1/proposal_v1.json'
    proposal = json.loads(proposal_path.read_text())
    assert proposal['parent_registration_sha256'] == sha(parent)
    assert proposal['parent_contract_sha256'] == sha(previous)
    assert proposal['recipe'] == 'set_bc' and proposal['selection_is_exploratory']
    assert proposal['no_pilot_eligibility_claim'] and proposal['no_fresh_seed_allocation']
    for item in proposal['observations']:
        assert sha(item['path']) == item['sha256']
    state = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert len(state['jobs']) == 84 and state['active_attempt'] == 1
    assert state['phase'] == 'confirmation' and state['attempts'][1]['recipe'] == 'bank_bc'
    assert state['attempts'][1]['stage'] == 'discovery'
    # The six bankBC jobs were registered natively after R8. Preserve them
    # alongside all78 top-level R8 jobs and their exact expected identities.
    previous_jobs = {j['id']:j for j in plan['jobs']}
    actual_jobs = {j['id']:j for j in state['jobs']}
    assert all(actual_jobs[k] == v for k,v in previous_jobs.items())
    receipt = root / 'analysis/onestep_pipeline_v1/registrations/confirmation_attempt1.json'
    native = json.loads(receipt.read_text())
    assert len(native['new_jobs']) == 6
    assert {j['id'] for j in state['jobs']} == set(previous_jobs) | {j['id'] for j in native['new_jobs']}
    assert all(actual_jobs[j['id']] == j for j in native['new_jobs'])
    old = copy.deepcopy(state['jobs'])
    plan['expected_identities'] = copy.deepcopy(state['identities'])
    selected = [j for j in old if j['args']['recipe'] == 'set_bc']
    assert len(selected) == 3 and {j['args']['env'] for j in selected} == set(plan['pilot_environments'])
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    started = set(scheduler['running']) | set(json.loads((root / 'finished.json').read_text()))
    started |= {j['id'] for j in old if (root / 'runs' / j['id'] / 'config.json').exists()}
    tags = ['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene']
    environments = plan['scheduling_amendment']['all_nine_environments']
    remaining = set(environments) - set(plan['pilot_environments'])
    assert set(proposal['remaining_environments']) == remaining and len(remaining) == 6
    new, expected = [], {}
    for env in proposal['remaining_environments']:
        rid = 'ospS_' + tags[environments.index(env)] + '_set_bc_s0'
        assert rid not in actual_jobs and not (root / 'runs' / rid).exists()
        new.append(dict(id=rid,module=implementation('set_bc').runner_module,
                        args=dict(env=env,recipe='set_bc',seed=0,steps=1000000)))
        expected[rid] = configuration(env, 'set_bc', 0)
        spec = expected[rid]['specification']
        assert spec['target'] == 'best9' and spec['objective'] == 'none'
        assert spec['architecture'] == 'permutation_invariant_attention'
        assert spec['effective_alpha'] == 1.0
    insertion = next(i for i,j in enumerate(old) if j['id'] not in started)
    plan['jobs'] = old[:insertion] + new + old[insertion:]
    assert [j for j in plan['jobs'] if j['id'] in actual_jobs] == old
    assert len({j['id'] for j in plan['jobs']}) == 90
    plan['expected_identities'].update(expected)
    sources = dict(plan['all_source_sha256'])
    for path in [Path(__file__).resolve(), Path(__file__).with_name('validate_set_full9_prefetch_v1.py').resolve(), proposal_path]:
        sources[str(path)] = sha(path)
    assert all(sha(path) == digest for path,digest in sources.items())
    plan.update(all_source_sha256=sources,set_full9_prefetch_extension=dict(
        time=time.time(),parent_registration=str(parent),parent_registration_sha256=sha(parent),
        parent_contract=str(previous),parent_contract_sha256=sha(previous),
        proposal=str(proposal_path),proposal_sha256=sha(proposal_path),
        native_prior_six_job_receipt=str(receipt),native_prior_six_job_receipt_sha256=sha(receipt),
        recipe='set_bc',new_jobs=new,new_expected_identities=expected,
        mode='exploratory_seed0_prefetch_not_formal_promotion',
        no_new_algorithm_sources=True,no_pilot_eligibility_claim=True,no_fresh_seed_allocation=True,
        active_discovery_attempt_preserved=1,old84definitions_and_relative_logical_order_preserved=True,
        inserted_before_logical_index=insertion,queue_rule=proposal['queue_rule'],reuse_rule=proposal['reuse_rule'],
        all_performance_and_speed_gates_unchanged=True,observations=proposal['observations']))
    assert plan['thresholds'] == json.loads(parent.read_text())['thresholds']
    return plan, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    args = parser.parse_args()
    assert not args.registration.exists() and not args.contract.exists()
    plan, contract = build_plan(args.root.resolve())
    path = args.root.resolve() / 'analysis/set_full9_prefetch_v1/validation_v1/result.json'
    validation = json.loads(path.read_text())
    assert validation['status'] == 'pass' and validation['no_performance_claim']
    for source,digest in validation['source_sha256'].items():
        assert sha(source) == digest
    plan['set_full9_prefetch_extension']['validation'] = dict(path=str(path),sha256=sha(path))
    args.registration.write_text(json.dumps(plan,indent=2,allow_nan=False))
    contract.update(registration=str(args.registration.resolve()),registration_sha256=sha(args.registration),
                    source_sha256=plan['all_source_sha256'],set_full9_prefetch_extension=plan['set_full9_prefetch_extension'])
    args.contract.write_text(json.dumps(contract,indent=2,allow_nan=False))
    print(json.dumps(dict(registration=str(args.registration),registration_sha256=sha(args.registration),
                          contract=str(args.contract),contract_sha256=sha(args.contract),jobs=90)))


if __name__ == '__main__':
    main()
