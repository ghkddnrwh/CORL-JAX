"""Compare gamma .995/.999 on all ten tasks with exactly training seeds0/1."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

STEPS = (800000, 900000, 1000000)
LABELS = ('AntMaze Large', 'HumanoidMaze Medium', 'HumanoidMaze Large', 'AntSoccer',
          'Cube Single', 'Cube Double', 'Puzzle 3×3', 'Puzzle 4×4', 'Scene', 'AntMaze Giant')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    workspace = Path(__file__).resolve().parents[2]
    old = (workspace / 'logs/diagnosis/finite_mean_target_ablation_20261105').resolve()
    giant = (workspace / 'logs/diagnosis/finite_mean_giant_gamma_tau_20261107').resolve()
    old_plan = json.loads((old / 'registration_v1.json').read_text())
    new_plan = json.loads((root / 'registration_v2.json').read_text())
    environments = old_plan['environments']
    assert len(environments) == 10 and new_plan['training_seeds'] == [0, 1]
    evidence = {}
    for p in (old / 'registration_v1.json', giant / 'registration_v1.json', root / 'registration_v2.json'):
        evidence[str(p)] = sha(p)
    records = []
    for env in environments:
        for gamma in (.995, .999):
            for seed in (0, 1):
                if gamma == .999:
                    recipe = 'ant_min_clip' if env.startswith('antmaze-') else 'mean_clip'
                    job = next(j for j in old_plan['jobs'] if j['environment'] == env and j['seed'] == seed and j['recipe'] == recipe)
                    run, evaluation = old / 'runs' / job['id'], old / 'evaluations' / job['id']
                elif env == 'antmaze-giant-navigate-singletask-v0':
                    identity = f'gt1_g995_t90_s{seed}'
                    run, evaluation = giant / 'runs' / identity, giant / 'evaluations' / identity
                else:
                    job = next(j for j in new_plan['jobs'] if j['environment'] == env and j['seed'] == seed)
                    run, evaluation = Path(job['output']), root / 'evaluations' / job['id']
                config_path = run / 'config.json'
                conf = json.loads(config_path.read_text())
                identity, cfg = conf['identity'], conf['identity']['config']
                assert identity['environment'] == env and identity['seed'] == seed
                assert cfg['discount'] == gamma and cfg['expectile'] == .9 and cfg['tau'] == .005
                assert cfg['backup_samples'] == cfg['num_samples'] == 9 and cfg['flow_steps'] == 10
                assert cfg['lr'] == .0003 and cfg['target_clipping']
                assert cfg['backup_Q_aggregation'] == ('min' if env.startswith('antmaze-') else 'mean')
                assert cfg['onestep_specification']['teacher_q_aggregation'] == 'min'
                assert cfg['onestep_specification']['deployment_decoder'] == 'sampled_component_location'
                assert cfg['onestep_specification']['mixture_components'] == 9
                for path, expected in conf['source_sha256'].items():
                    if path not in evidence:
                        evidence[path] = sha(path)
                    assert evidence[path] == expected
                completion = run / 'completed.json'
                completed = json.loads(completion.read_text())
                assert completed['step'] == completed['q_updates'] == completed['actor_updates'] == 1000000
                assert completed['v_updates'] == 0
                evidence[str(config_path)], evidence[str(completion)] = sha(config_path), sha(completion)
                for mode in ('lean', 'actor'):
                    scores = {}
                    for step in STEPS:
                        path = evaluation / f'{step}_{mode}.json'
                        v = json.loads(path.read_text())
                        assert v['environment'] == env and v['training_seed'] == seed
                        assert v['step'] == step and v['policy_mode'] == mode and v['recipe'] == identity['recipe']
                        assert v['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                        assert v['n_episodes'] == len(v['episode_successes']) == 50 and v['first_episode_seed'] == 42
                        assert v['no_optimizer_updates'] and v['config_sha256'] == evidence[str(config_path)]
                        score = statistics.mean(v['episode_successes'])
                        assert abs(score - v['success_rate']) < 1e-12
                        scores[step] = score * 100.
                        evidence[str(path)] = sha(path)
                    records.append(dict(environment=env, gamma=gamma, training_seed=seed, deployment=mode,
                        bootstrap_samples=9, success_800K_pct=scores[800000], success_900K_pct=scores[900000],
                        success_1M_pct=scores[1000000], last3_success_pct=statistics.mean(scores.values()),
                        run_directory=str(run), evaluation_directory=str(evaluation)))
    assert len(records) == 80
    index = {(r['environment'], r['gamma'], r['training_seed'], r['deployment']): r for r in records}
    summary, wide = [], []
    for env, label in zip(environments, LABELS):
        row = dict(environment=env, label=label)
        for mode in ('lean', 'actor'):
            for gamma in (.995, .999):
                values = [index[env, gamma, seed, mode]['last3_success_pct'] for seed in (0, 1)]
                summary.append(dict(environment=env, gamma=gamma, deployment=mode, seed0_pct=values[0], seed1_pct=values[1],
                    mean_pct=statistics.mean(values), sample_sd_pp=statistics.stdev(values)))
                tag = f'{mode}_gamma{str(gamma).replace(".", "p")}'
                row[tag + '_mean_pct'], row[tag + '_sample_sd_pp'] = statistics.mean(values), statistics.stdev(values)
        wide.append(row)
    macro = []
    for mode in ('lean', 'actor'):
        for gamma in (.995, .999):
            values = [statistics.mean(index[env, gamma, seed, mode]['last3_success_pct'] for env in environments) for seed in (0, 1)]
            macro.append(dict(gamma=gamma, deployment=mode, environment_count=10, seed0_pct=values[0], seed1_pct=values[1],
                mean_pct=statistics.mean(values), sample_sd_pp=statistics.stdev(values)))
    out = root / 'analysis/discount_comparison_10task_seed01_20261120_v1'
    out.mkdir(exist_ok=True)
    write_csv(out / 'per_seed_results.csv', records)
    write_csv(out / 'task_results_long.csv', summary)
    write_csv(out / 'task_results_wide.csv', wide)
    write_csv(out / 'macro_results.csv', macro)
    description = dict(status='PASS', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_seeds=[0, 1], gammas=[.995, .999], bootstrap_samples=9, expectile=.9,
        actor_label_candidates=9, lean_deployment_candidates=9, evaluation_steps=list(STEPS),
        episodes_per_point=50, first_episode_seed=42, evaluation_points=240, episode_records=12000,
        metric='Success percentage. Average800K/900K/1M separately per training seed, then average seeds0/1; sample SD.',
        macro_metric='Equal-weight ten-task average separately per training seed, then mean/sample SD across seeds0/1.',
        baseline='Canonical original target-ablation N9 min+clip on AntMaze, mean+clip elsewhere; exactly seeds0/1.',
        existing_giant_0995='Original completed giant_discount_expectile g995_t90 seeds0/1; no new run.',
        limitations=['Only two training seeds; smaller SD alone does not establish stabilization.',
                     'The ten-task average includes Giant and therefore differs from the previous nine-task average.'],
        source_sha256=sha(__file__), evidence_sha256=evidence)
    (out / 'audit.json').write_text(json.dumps(description, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), tasks=wide, macro=macro), indent=2), flush=True)


if __name__ == '__main__':
    main()
