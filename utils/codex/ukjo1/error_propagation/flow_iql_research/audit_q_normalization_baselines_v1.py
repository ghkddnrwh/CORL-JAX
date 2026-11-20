"""Inventory matched Finite-mean N9 controls; does not launch training.

Verify archived source bytes rather than paths that changed during migration.
No new normalization algorithm is selected by this inventory.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    workspace = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
    logs = workspace / 'logs/codex/ukjo1/error_propagation/diagnosis'
    output = workspace / 'utils/codex/ukjo1/error_propagation/reports/q_normalization_literature_20261120'
    output.mkdir(parents=True, exist_ok=True)
    controls = []
    for gamma, study, registration in (
        (.999, 'finite_mean_target_ablation_20261105', 'registration_v1.json'),
        (.995, 'finite_mean_gamma0995_n9_9task_2seed_20261119_v2', 'registration_v2.json'),
        (.995, 'finite_mean_giant_gamma_tau_20261107', 'registration_v1.json'),
    ):
        root = logs / study
        plan = json.loads((root / registration).read_text())
        for job in plan['jobs']:
            if job['seed'] not in (0, 1):
                continue
            expected_recipe = 'ant_min_clip' if job['environment'].startswith('antmaze-') else 'mean_clip'
            if gamma == .999 and job['recipe'] != expected_recipe:
                continue
            if 'giant_gamma_tau' in study and job['recipe'] != 'g995_t90':
                continue
            run = root / 'runs' / job['id']
            config_path = run / 'config.json'
            conf = json.loads(config_path.read_text())
            identity, cfg = conf['identity'], conf['identity']['config']
            spec = cfg['onestep_specification']
            assert identity['environment'] == job['environment'] and identity['seed'] == job['seed']
            assert cfg['discount'] == gamma and cfg['expectile'] == .9
            assert cfg['backup_samples'] == cfg['num_samples'] == 9
            assert cfg['batch_size'] == 256 and cfg['target_clipping']
            assert cfg['backup_Q_aggregation'] == ('min' if job['environment'].startswith('antmaze-') else 'mean')
            assert spec['objective'] == 'none' and spec['q_scale'] == 'none'
            assert spec['target'] == 'best9' and spec['teacher_q_aggregation'] == 'min'
            assert spec['mixture_components'] == 9
            done = json.loads((run / 'completed.json').read_text())
            assert done['step'] == done['q_updates'] == done['actor_updates'] == 1000000
            saved_hashes = {digest(p) for p in (run / 'source').rglob('*') if p.is_file()}
            missing_sources = [p for p, sha in conf['source_sha256'].items() if sha not in saved_hashes]
            assert not missing_sources, (run, missing_sources)
            config_sha = digest(config_path)
            evaluations = []
            for mode in ('lean', 'actor'):
                for step in (800000, 900000, 1000000):
                    path = root / 'evaluations' / job['id'] / f'{step}_{mode}.json'
                    value = json.loads(path.read_text())
                    assert value['training_seed'] == job['seed'] and value['environment'] == job['environment']
                    assert value['step'] == step and value['policy_mode'] == mode
                    assert value['n_episodes'] == len(value['episode_successes']) == 50
                    assert value['first_episode_seed'] == 42 and value['eval_device'] == 'cpu'
                    assert value['config_sha256'] == config_sha
                    score = statistics.mean(value['episode_successes'])
                    assert abs(score - value['success_rate']) < 1e-12
                    evaluations.append(dict(mode=mode, step=step, success_pct=100 * score,
                                            path=str(path), sha256=digest(path)))
            controls.append(dict(environment=job['environment'], seed=job['seed'], gamma=gamma,
                                 run_directory=str(run), config_sha256=config_sha,
                                 archived_sources_verified=len(conf['source_sha256']),
                                 actor_objective='likelihood_distillation_without_Q_maximization',
                                 evaluations=evaluations))
    assert len(controls) == 40
    assert len({(r['environment'], r['seed'], r['gamma']) for r in controls}) == 40
    result = dict(status='PASS', requested_cells=40, completed_controls=len(controls),
                  evaluated_points=sum(len(r['evaluations']) for r in controls),
                  new_training_launched=False, normalization_intervention='awaiting_user_choice',
                  checkpoint_payload_hashes_recomputed=False, controls=controls)
    (output / 'finite_mean_control_inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = []
    for r in controls:
        for mode in ('lean', 'actor'):
            points = {str(e['step']): e['success_pct'] for e in r['evaluations'] if e['mode'] == mode}
            rows.append(dict(environment=r['environment'], seed=r['seed'], gamma=r['gamma'], mode=mode,
                             **points, last3_success_pct=statistics.mean(points.values()),
                             run_directory=r['run_directory']))
    with (output / 'finite_mean_control_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'controls'}, indent=2))


if __name__ == '__main__':
    main()
