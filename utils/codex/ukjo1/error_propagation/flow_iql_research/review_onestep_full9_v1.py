"""Read-only full9 last3CPU50 model/counter/export/raw-episode audit."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import pickle
import statistics
import time

import numpy as np

from utils.flow_iql_research.review_onestep_checkpoint_v1 import sha,leaves,equal_trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.root.resolve()
    continuation = json.loads((root / 'CONTINUATION.json').read_text())
    contract_path = Path(continuation['analysis_contract'])
    contract = json.loads(contract_path.read_text())
    registration = Path(contract['registration'])
    assert sha(registration) == contract['registration_sha256']
    plan = json.loads(registration.read_text())
    for source,digest in plan['all_source_sha256'].items():
        assert sha(source) == digest,source
    research = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    jobs = [j for j in research['jobs'] if j['args']['recipe'] == args.recipe and j['args']['seed'] == args.seed]
    environments = plan['scheduling_amendment']['all_nine_environments']
    assert len(jobs) == 9 and {j['args']['env'] for j in jobs} == set(environments)
    rows,pins,hashes,alphas,discounts,expectiles = [],[],{},set(),set(),set()
    for env in environments:
        job = next(j for j in jobs if j['args']['env'] == env)
        rid = job['id']
        run = root / 'runs' / rid
        config_path = run / 'config.json'
        conf = json.loads(config_path.read_text())
        identity = conf['identity']
        assert identity == research['identities'][rid]
        assert identity['environment'] == env and identity['recipe'] == args.recipe and identity['seed'] == args.seed
        assert identity['steps'] == 1000000 and identity['training_device'] == 'gpu'
        implementation = next(v for v in plan['implementation_definitions'].values() if v['runner_module'] == job['module'])
        assert conf['source_sha256'] == plan['implementation_training_source_sha256'][implementation['tag']]
        for source,digest in conf['source_sha256'].items():
            assert sha(source) == digest
            p = Path(source)
            archive = run / 'source' / ('vendor_'+p.name if p.parent.name == 'vendor' else p.name)
            assert sha(archive) == digest
        alphas.add(identity['specification']['effective_alpha'])
        discounts.add(identity['config']['discount'])
        expectiles.add(identity['config']['expectile'])
        successes = {'primary':[],'teacher':[]}
        for step in (800000,900000,1000000):
            checkpoint = run / f'checkpoint_{step}.pkl'
            actor_path = run / f'actor_{step}.pkl'
            with checkpoint.open('rb') as stream:
                saved = pickle.load(stream)
            with actor_path.open('rb') as stream:
                artifact = pickle.load(stream)
            assert saved['identity'] == artifact['identity'] == identity
            assert saved['source_sha256'] == artifact['source_sha256'] == conf['source_sha256']
            assert saved['step'] == artifact['step'] == artifact['Q_updates'] == artifact['actor_updates'] == step
            state = saved['state']
            assert int(state['base']['q_updates']) == int(state['base']['agent']['network']['opt_state']['0']['count']) == step
            assert int(state['actor_updates']) == int(state['actor']['opt_state']['0']['count']) == step
            assert int(state['base']['v_updates']) == int(state['base']['v_opt']['0']['count']) == 0
            assert all(np.isfinite(v).all() for _,v in leaves(state))
            assert equal_trees(dict(leaves(state['actor']['params'])),dict(leaves(artifact['actor_params'])))
            metadata = artifact['metadata']
            assert metadata['actor_forward_passes'] == 1 and metadata['flow_integration_steps'] == metadata['generated_candidate_actions'] == 0
            assert not metadata['requires_critic'] and not metadata['requires_BC_flow']
            checkpoint_hash,actor_hash,config_hash = sha(checkpoint),sha(actor_path),sha(config_path)
            assert artifact['checkpoint_sha256'] == checkpoint_hash and artifact['config_sha256'] == config_hash
            pins.append(dict(run=rid,step=step,Q_Adam=step,actor_Adam=step,V_Adam=0,all_saved_state_finite=True,
                exported_params_exact=True,checkpoint_sha256=checkpoint_hash,actor_sha256=actor_hash))
            hashes[str(checkpoint)],hashes[str(actor_path)],hashes[str(config_path)] = checkpoint_hash,actor_hash,config_hash
            for mode in ('primary','teacher'):
                path = root / 'analysis/evaluations_v1' / f'{rid}_{step}_{mode}_n50.json'
                raw = json.loads(path.read_text())
                assert raw['run'] == rid and raw['recipe'] == args.recipe and raw['training_seed'] == args.seed and raw['environment'] == env
                assert raw['step'] == step and raw['policy_mode'] == mode and raw['n_episodes'] == 50
                assert raw['eval_device'] == 'cpu' and raw['first_episode_seed'] == 42 and raw['no_optimizer_updates']
                assert raw['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                assert raw['checkpoint_sha256'] == checkpoint_hash and raw['actor_artifact_sha256'] == actor_hash and raw['config_sha256'] == config_hash
                evaluator = Path(__file__).resolve().with_name(implementation['evaluator_module'].rsplit('.',1)[1]+'.py')
                core = evaluator.with_name(implementation['core_module'].rsplit('.',1)[1]+'.py')
                assert raw['source_sha256'] == plan['all_source_sha256'][str(evaluator)]
                assert raw['core_source_sha256'] == conf['source_sha256'][str(core)]
                assert all(len(raw[k]) == 50 for k in ('episode_successes','episode_rewards','episode_lengths'))
                assert all(v in (0,1) for v in raw['episode_successes'])
                assert all(math.isfinite(float(v)) for k in ('episode_rewards','episode_lengths') for v in raw[k])
                assert all(v > 0 and int(v) == v for v in raw['episode_lengths'])
                total = sum(int(v) for v in raw['episode_successes'])
                assert abs(raw['success_rate'] - total/50) < 1e-12
                assert abs(raw['reward_mean'] - statistics.mean(raw['episode_rewards'])) < 1e-8
                if mode == 'primary':
                    assert raw['learned_model_trees_loaded'] == ['actor_params'] and raw['deployment_metadata'] == metadata
                successes[mode].append(total)
                hashes[str(path)] = sha(path)
        primary,teacher = (Fraction(sum(successes[k]),150) for k in ('primary','teacher'))
        rows.append(dict(environment=env,run=rid,recipe=args.recipe,training_seed=args.seed,
            primary_successes800K900K1M=successes['primary'],teacher_successes800K900K1M=successes['teacher'],
            primary_last3=float(primary),teacher_last3=float(teacher),teacher_minus_primary=float(teacher-primary)))
    primary = Fraction(sum(sum(r['primary_successes800K900K1M']) for r in rows),1350)
    teacher = Fraction(sum(sum(r['teacher_successes800K900K1M']) for r in rows),1350)
    tolerance = plan['thresholds']['mean_loss_tolerance']
    checks = None
    if args.seed == 0:
        checks = dict(paired_mean=float(primary) >= float(teacher)-tolerance-1e-12,
            historical_mean=float(primary) >= plan['thresholds']['historical_lean_seed0_nine_task_reference']-tolerance-1e-12)
    result = dict(time=time.time(),status='audited',recipe=args.recipe,training_seed=args.seed,
        environment_count=9,evaluation_steps=[800000,900000,1000000],episodes_per_checkpoint=50,
        denominator=1350,primary_total_successes=primary.numerator*(1350//primary.denominator),
        teacher_total_successes=teacher.numerator*(1350//teacher.denominator),
        primary_macro_fraction=str(primary),teacher_macro_fraction=str(teacher),
        primary_macro=float(primary),teacher_macro=float(teacher),mean_teacher_minus_primary=float(teacher-primary),
        discovery_checks=checks,discovery_passed=all(checks.values()) if checks is not None else None,
        unique_effective_actor_alphas=sorted(alphas),discounts=sorted(discounts),expectiles=sorted(expectiles),rows=rows,
        actual_model_counter_export_pins=pins,raw_evidence_sha256=hashes,
        registration=str(registration),registration_sha256=sha(registration),review_source_sha256=sha(__file__),
        active_goal_not_marked_complete=True,
        scope='One global recipe, one training seed, full9last3CPU50. Fresh3seed confirmation and isolated CPU/GPU latency are separate mandatory goal checks.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    table = args.output.with_suffix('.csv')
    with table.open('x') as stream:
        fields=['environment','run','recipe','training_seed','primary_last3','teacher_last3','teacher_minus_primary']
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({k:result[k] for k in ('recipe','training_seed','primary_macro','teacher_macro','discovery_checks','discovery_passed')}),flush=True)


if __name__ == '__main__':
    main()
