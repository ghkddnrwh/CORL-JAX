"""Registration and audits for the default-task gamma x seed completion study.

Learning and evaluation use the existing, unmodified experiment modules.
This module only identifies compatible runs and records their provenance.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics

PREFIX = 'utils.codex.ukjo1.error_propagation.flow_iql_research.'
STUDY = 'finite_mean_default10_gamma_pair_n9_seed0123_20261124_v1'
FAMILIES = (
    ('aml', 'antmaze-large-navigate-singletask-v0'),
    ('amg', 'antmaze-giant-navigate-singletask-v0'),
    ('hmm', 'humanoidmaze-medium-navigate-singletask-v0'),
    ('hml', 'humanoidmaze-large-navigate-singletask-v0'),
    ('soccer', 'antsoccer-arena-navigate-singletask-v0'),
    ('cube1', 'cube-single-play-singletask-v0'),
    ('cube2', 'cube-double-play-singletask-v0'),
    ('puzzle3', 'puzzle-3x3-play-singletask-v0'),
    ('puzzle4', 'puzzle-4x4-play-singletask-v0'),
    ('scene', 'scene-play-singletask-v0'),
)
STEPS = [800000, 900000, 1000000]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_matrix(plan):
    assert plan['training_seeds'] == [0, 1, 2, 3]
    assert plan['discounts'] == [.995, .999] and plan['sample_counts'] == [9]
    assert plan['methods'] == ['finite_mean_n9']
    assert set(plan['environments']) == {e for _, e in FAMILIES}
    cells = plan['jobs'] + plan['reused_jobs']
    assert len(cells) == plan['total_cells'] == 80
    assert len({j['id'] for j in cells}) == 80
    keys = [(j['environment'], j['gamma'], j['seed']) for j in cells]
    assert len(set(keys)) == 80
    assert set(keys) == {(e, g, s) for _, e in FAMILIES for g in plan['discounts'] for s in range(4)}
    assert all(j['method'] == 'finite_mean_n9' and j['bootstrap_samples'] == 9 for j in cells)
    assert all(j['head_aggregation'] == ('min' if j['environment'].startswith('antmaze-') else 'mean') for j in cells)


def audit_config(job, conf, reference=None):
    ident, cfg = conf['identity'], conf['identity']['config']
    assert ident['environment'] == job['environment'] and ident['seed'] == job['seed']
    assert ident['recipe'] == job['recipe'] and ident['steps'] == 1000000
    assert cfg['discount'] == job['gamma'] and cfg['expectile'] == .9
    assert cfg['batch_size'] == cfg.get('critic_batch_size', 256) == 256
    assert cfg['tau'] == .005 and cfg['lr'] == .0003 and cfg.get('dr3_coefficient', 0.) == 0.
    assert cfg['backup_samples'] == cfg['num_samples'] == 9 and cfg['flow_steps'] == 10
    assert cfg['target_clipping'] and cfg['backup_Q_aggregation'] == job['head_aggregation']
    spec = cfg['onestep_specification']
    assert spec['target'] == 'best9' and spec['reference_training_recipe'] == 'mdn9_bc'
    assert spec['deployment_decoder'] == 'sampled_component_location' and spec['q_scale'] == 'none'
    assert spec.get('stability_intervention', 'control') == 'control'
    if reference is not None:
        # Compare every operative config key; ignore only redundant control fields.
        clean = {k: v for k, v in cfg.items() if k not in ('onestep_specification', 'critic_batch_size', 'dr3_coefficient')}
        expected = json.loads(json.dumps(reference))
        expected['discount'] = job['gamma']
        assert clean == {k: v for k, v in expected.items() if k != 'onestep_specification'}, job['id']
        ignored = {'name', 'reference_target_recipe', 'giant_search', 'discount',
                   'backup_samples', 'actor_label_samples', 'deployment_flow_samples',
                   'stability_intervention', 'target_EMA', 'critic_batch_size',
                   'BC_and_actor_batch_size', 'dr3_coefficient'}
        assert {k: v for k, v in spec.items() if k not in ignored} == {
            k: v for k, v in expected['onestep_specification'].items() if k not in ignored}, job['id']


def audit_evaluation(job, conf, step, mode, path, evidence=None):
    run = Path(job['output'])
    value = json.loads(Path(path).read_text())
    assert value['environment'] == job['environment'] and value['training_seed'] == job['seed']
    assert value['recipe'] == conf['identity']['recipe'] and value['step'] == step and value['policy_mode'] == mode
    assert value['first_episode_seed'] == 42 and value['n_episodes'] == len(value['episode_successes']) == 50
    assert value['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1' and value['no_optimizer_updates']
    assert abs(value['success_rate'] - statistics.mean(value['episode_successes'])) < 1e-12
    assert value['config_sha256'] == sha(run / 'config.json')
    if evidence is not None:
        for artifact, digest in ((run / f'checkpoint_{step}.pkl', value['checkpoint_sha256']),
                                 (run / f'actor_{step}.pkl', value['actor_artifact_sha256'])):
            if str(artifact) not in evidence:
                evidence[str(artifact)] = sha(artifact)
            assert evidence[str(artifact)] == digest, artifact
        evidence[str(path)] = sha(path)
    return 100 * value['success_rate']
