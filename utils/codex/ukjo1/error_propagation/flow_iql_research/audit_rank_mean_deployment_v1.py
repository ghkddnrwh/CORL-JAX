"""Audit all9 deployment parameter exports against their27 actual source models."""
import argparse
import json
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import core  # Compatibility before unpickling.
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'analysis/rank_mean_discovery_deployment_v1'
    assert not (output / 'strict_audit_v1.json').exists()
    verifier = Verifier()
    manifest = json.loads((output / 'manifest.json').read_text())
    plan_path = root / 'rank_mean_discovery_deployment_registration_v1.json'
    plan = json.loads(plan_path.read_text())
    assert manifest['plan_sha256'] == verifier.digest(plan_path)
    assert manifest['status'] == 'complete' and manifest['completed_tasks'] == 9 and not manifest['failures']
    assert not manifest['running'] and not manifest['goal_complete']
    inputs = {str(output / 'manifest.json'): verifier.digest(output / 'manifest.json'),
              str(plan_path): verifier.digest(plan_path)}
    rows = []
    for job in plan['jobs']:
        artifact = next(a for a in manifest['artifacts'] if a['environment'] == job['environment'])
        model = Path(artifact['bundle'])
        assert verifier.digest(model) == artifact['bundle_sha256']
        with model.open('rb') as stream:
            bundle = pickle.load(stream)
        assert bundle['kind'] == 'independent_critic_rank_mean_deployment_v1'
        assert bundle['step'] == 1000000 and bundle['environment'] == job['environment']
        assert bundle['recipe']['rule'] == 'mean_rank' and bundle['recipe']['samples'] == 9
        assert bundle['recipe']['learners'] == 3 and len(bundle['other_critic_params']) == 2
        assert not bundle['resumable_training_checkpoint'] and not {'state', 'opt_state', 'replay_rng'} & bundle.keys()
        for source, digest in bundle['source_sha256'].items():
            assert verifier.digest(source) == digest, source
        parameters = [bundle['first_network_params'], *bundle['other_critic_params']]
        for index, (run_name, actual, evidence, pin) in enumerate(zip(
                job['critic_runs'], parameters, bundle['members'], job['member_pins'])):
            run = Path(run_name)
            assert (run / 'completed.json').exists() and not (run / 'stopped.json').exists()
            conf, saved, checkpoint = read(run, 1000000)
            digest = verifier.checkpoint(checkpoint, conf, 1000000)
            assert evidence['identity'] == conf['identity']
            assert digest == pin['checkpoint_sha256'] == evidence['checkpoint_sha256']
            assert verifier.digest(run / 'config.json') == pin['config_sha256'] == evidence['config_sha256']
            original = saved['state']['agent']['network']['params']
            if index:
                original = original['modules_critic']
            a, ta = jax.tree_util.tree_flatten(actual)
            b, tb = jax.tree_util.tree_flatten(original)
            assert ta == tb and len(a) == len(b)
            assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
            assert all(np.isfinite(np.asarray(x)).all() for x in a)
            del saved
        record_path = Path(artifact['validation_record'])
        record = json.loads(record_path.read_text())
        assert verifier.digest(record_path) == artifact['validation_sha256']
        inputs[str(record_path)] = verifier.digest(record_path)
        if record.get('status') == 'complete':
            assert record['no_source_model_reads_during_load']
            raw_path = record_path.parent / 'CPU50_reference_equivalence.json'
            assert verifier.digest(raw_path) == record['result_sha256']
            inputs[str(raw_path)] = verifier.digest(raw_path)
            record = json.loads(raw_path.read_text())
        reference_path = Path(job['reference'])
        reference = json.loads(reference_path.read_text())
        inputs[str(reference_path)] = verifier.digest(reference_path)
        assert record['environment'] == reference['environment'] == job['environment']
        assert record['step'] == reference['step'] == 1000000
        assert record['bundle_sha256'] == artifact['bundle_sha256']
        assert record['source_sha256'] == bundle['source_sha256']['utils/flow_iql_research/evaluate_rank_mean_bundle_v1.py']
        assert reference['aggregation_rule'] == 'mean_rank' and reference['selection_samples'] == 9
        assert [m['checkpoint_sha256'] for m in bundle['members']] == [m['checkpoint_sha256'] for m in reference['critic_members']]
        assert record['n_episodes'] == reference['n_episodes'] == 50
        assert record['first_episode_seed'] == reference['first_episode_seed'] == 42
        assert record['eval_device'] == reference['eval_device'] == 'cpu'
        assert record['protocol'] == reference['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
        for field in ('episode_successes', 'episode_rewards', 'episode_lengths'):
            assert len(record[field]) == 50 and np.isfinite(record[field]).all()
            assert record[field] == reference[field]
        assert all(record['reference_episode_arrays_exact'].values())
        rows.append(dict(environment=job['environment'], bundle=str(model), bytes=model.stat().st_size,
            bundle_sha256=artifact['bundle_sha256'], learner_seeds=[m['identity']['seed'] for m in bundle['members']],
            all_parameter_trees_exact=True, all_CPU50_episode_arrays_exact=True))
    assert len(rows) == 9 and len(verifier.checkpoints) == 27
    result = dict(time=time.time(), status='complete', rows=rows, tasks=9, actual_source_checkpoints_verified=27,
        all_parameter_trees_exact=True, all_CPU50_success_reward_length_arrays_exact=True,
        total_bundle_bytes=sum(r['bytes'] for r in rows), inputs_sha256=inputs,
        source_sha256=verifier.digest(__file__), goal_complete=False,
        scope='Fixed1M deployment reproduction only, not a last3 benchmark average or fresh training/ensemble replication.')
    (output / 'strict_audit_v1.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ('status', 'tasks', 'actual_source_checkpoints_verified',
        'all_parameter_trees_exact', 'all_CPU50_success_reward_length_arrays_exact', 'total_bundle_bytes')}), flush=True)


if __name__ == '__main__':
    main()
