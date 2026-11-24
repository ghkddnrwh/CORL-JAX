"""Audit 80 requested cells, reusing completed runs and resuming baseline controls."""
import argparse
import ast
import csv
import datetime
import importlib
import json
import os
from pathlib import Path
import pickle
import shutil

from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import (
    PREFIX, STUDY, FAMILIES, STEPS, sha, atomic, write_csv, assert_matrix, audit_config, audit_evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert not (root / 'registration_v1.json').exists()
    workspace = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
    logbase = workspace / 'logs/codex/ukjo1/error_propagation/diagnosis'
    reports = workspace / ('logs/codex/ukjo1/error_propagation/reports/diagnosis/'
                           'finite_mean_gamma0995_n9_9task_2seed_20261119_v2/analysis')
    folder = Path(__file__).resolve().parent
    base = importlib.import_module(PREFIX + 'sample_budget_core_v1')
    sources, evidence, references = {}, {}, {}
    checked_sources = {}

    def pin(path):
        path = Path(path).resolve()
        digest = sha(path)
        assert str(path) not in sources or sources[str(path)] == digest
        sources[str(path)] = digest

    def remap(text):
        p = Path(text)
        if p.exists():
            return p.resolve()
        for marker in ('/runs/', '/evaluations/'):
            if marker in text:
                before, after = text.split(marker, 1)
                p = logbase / Path(before).name / marker.strip('/') / after
                if p.exists():
                    return p.resolve()
        raise FileNotFoundError(text)

    candidates = {}
    first = reports / 'discount_comparison_10task_seed01_20261120_v1/per_seed_results.csv'
    second = reports / 'all_equivalent_seeds_gamma0999_20261120_v1/included_unique_seed_results.csv'
    for path in (first, second):
        references[str(path)] = sha(path)
        with path.open() as stream:
            for row in csv.DictReader(stream):
                seed = int(row['training_seed'])
                if path == second and seed != 2:
                    continue
                gamma = float(row['gamma']) if path == first else .999
                run = remap(row['run_directory'])
                ev = remap(row['evaluation_directory']) if path == first else remap(row['raw_files'].split(';')[0]).parent
                key = (row['environment'], gamma, seed)
                assert key not in candidates or candidates[key] == (run, ev)
                candidates[key] = (run, ev)
    assert len(candidates) == 44
    reused = []
    for (environment, gamma, seed), (run, ev) in candidates.items():
        conf = json.loads((run / 'config.json').read_text())
        code = next(code for code, env in FAMILIES if env == environment)
        job = dict(id=f'g{str(gamma)[2:]}_{code}_s{seed}', code=code, environment=environment,
                   gamma=gamma, seed=seed, method='finite_mean_n9', bootstrap_samples=9,
                   head_aggregation='min' if environment.startswith('antmaze-') else 'mean',
                   recipe=conf['identity']['recipe'], output=str(run), evaluation_directory=str(ev),
                   reused=True, action='reuse_completed')
        audit_config(job, conf, base.config_for(environment, 'n9'))
        done = json.loads((run / 'completed.json').read_text())
        assert done['step'] == done['q_updates'] == done['actor_updates'] == 1000000 and done['v_updates'] == 0
        for name, digest in conf['source_sha256'].items():
            old = Path(name)
            current = Path(name.replace('/utils/flow_iql_research/', '/utils/codex/ukjo1/error_propagation/flow_iql_research/')
                           .replace('/utils/propagation_research/', '/utils/codex/ukjo1/error_propagation/propagation_research/'))
            if current == old:
                if name not in checked_sources:
                    checked_sources[name] = sha(current)
                assert checked_sources[name] == digest, name
            else:
                # Pre-move completed runs retain their original immutable config/hash.
                archived = run / 'source' / ('vendor_' + old.name if old.parent.name == 'vendor' else old.name)
                assert sha(archived) == digest, archived
                text = archived.read_text()
                for legacy, canonical in (
                    ('utils.flow_iql_research', 'utils.codex.ukjo1.error_propagation.flow_iql_research'),
                    ('utils.propagation_research', 'utils.codex.ukjo1.error_propagation.propagation_research'),
                    ('utils/flow_iql_research', 'utils/codex/ukjo1/error_propagation/flow_iql_research'),
                    ('utils/propagation_research', 'utils/codex/ukjo1/error_propagation/propagation_research')):
                    text = text.replace(legacy, canonical)
                assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(current.read_text())), current
                evidence[str(archived)] = digest
            pin(current)
        for name in ('config.json', 'completed.json', 'dataset.json'):
            evidence[str(run / name)] = sha(run / name)
        for step in STEPS:
            for mode in ('lean', 'actor'):
                audit_evaluation(job, conf, step, mode, ev / f'{step}_{mode}.json', evidence)
        reused.append(job)

    old_controls = dict(
        cube2_s2='stab1_cube2_control_s2', aml_s3='stab1_aml_control_s3',
        hmm_s3='stab1_hmm_control_s3', cube2_s3='stab1_cube2_control_s3',
        puzzle4_s3='stab1_puzzle4_control_s3')
    jobs = []
    for seed in (2, 3):
        for gamma in (.995, .999):
            for code, environment in FAMILIES:
                if (environment, gamma, seed) in candidates:
                    continue
                identity = f'g{str(gamma)[2:]}_{code}_s{seed}'
                resume_name = old_controls.get(f'{code}_s{seed}') if gamma == .999 else None
                train = 'train_stability_study_v1' if resume_name else (
                    'train_discount_0995_n9_v1' if gamma == .995 else 'train_sample_budget_v1')
                evaluate = train.replace('train_', 'evaluate_', 1)
                recipe = 'control' if resume_name else 'g995_n9' if gamma == .995 else 'n9'
                run = (logbase / 'finite_mean_stability_4task_20261120_v1/runs' / resume_name).resolve() if resume_name else root / 'runs' / identity
                job = dict(id=identity, code=code, environment=environment, gamma=gamma, seed=seed,
                           method='finite_mean_n9', bootstrap_samples=9,
                           head_aggregation='min' if environment.startswith('antmaze-') else 'mean',
                           output=str(run), evaluation_directory=str(root / 'evaluations' / identity), recipe=recipe,
                           train_module=PREFIX + train, eval_module=PREFIX + evaluate, reused=False,
                           action='resume_checkpoint' if resume_name else 'train_fresh', resume_step=0)
                module = importlib.import_module(PREFIX + train)
                cfg = module.configuration(environment, recipe, seed)
                audit_config(job, dict(identity=cfg), base.config_for(environment, 'n9'))
                for path in module.core.source_paths() + [Path(module.__file__).resolve(), folder / (evaluate + '.py')]:
                    pin(path)
                if resume_name:
                    conf = json.loads((run / 'config.json').read_text())
                    assert conf['identity'] == cfg
                    assert not (run / 'completed.json').exists() and not (run / 'stopped.json').exists()
                    for path, digest in conf['source_sha256'].items():
                        assert sha(path) == digest
                        assert sources[path] == digest
                    with (run / 'latest.pkl').open('rb') as stream:
                        saved = pickle.load(stream)
                    assert saved['identity'] == cfg and saved['source_sha256'] == conf['source_sha256']
                    assert 'replay_rng' in saved and 'extra_replay_rng' in saved
                    job['resume_step'] = int(saved['step'])
                    assert 0 < job['resume_step'] < 1000000
                    for name in ('latest.pkl', 'config.json', 'dataset.json'):
                        evidence[str(run / name)] = sha(run / name)
                    # Immutable copy protects the pre-resume state; original latest.pkl is mutable.
                    backup = root / 'resume_backups' / identity
                    backup.mkdir(parents=True, exist_ok=True)
                    for name in ('latest.pkl', 'config.json', 'dataset.json', 'cancelled.json'):
                        if (run / name).exists():
                            shutil.copyfile(run / name, backup / name)
                    job['resume_backup'] = str(backup)
                jobs.append(job)
    # Resume the five saved controls immediately, then round-robin fresh gamma/seed groups.
    jobs.sort(key=lambda j: (j['action'] != 'resume_checkpoint', j['seed'],
                             FAMILIES.index((j['code'], j['environment'])), j['gamma']))
    assert len(jobs) == 36 and sum(j['resume_step'] > 0 for j in jobs) == 5
    for name in ('default_gamma_pair_n9_v1.py', 'prepare_default_gamma_pair_n9_v1.py',
                 'run_default_gamma_pair_n9_v1.py', 'supervise_default_gamma_pair_n9_v1.py',
                 'analyze_default_gamma_pair_n9_v1.py', 'validate_default_gamma_pair_n9_v1.py'):
        pin(folder / name)
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 96
    plan = dict(client_date_KST='2026-11-24', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                purpose='Fill all missing DEFAULT-task baseline gamma=.995/.999 x seed0/1/2/3 N9 cells.',
                environments=[e for _, e in FAMILIES], training_seeds=[0, 1, 2, 3], discounts=[.995, .999],
                methods=['finite_mean_n9'], sample_counts=[9], total_cells=80, jobs=jobs, reused_jobs=reused,
                steps=1000000, evaluation_steps=STEPS, deployment_modes=['lean', 'actor'], episodes_per_point=50,
                target_EMA=.005, batch_size=256, expectile=.9, learning_rate=.0003, target_clipping=True,
                head_rule='AntMaze min; all other tasks mean.', actor_label_samples=9, lean_deployment_samples=9,
                source_sha256=sources, reused_evidence_sha256=evidence, report_reference_sha256=references,
                resources=dict(train_slots=[[g, s] for g in (0, 1) for s in range(8)],
                               train_cpus=[cpus[4*i:4*i+4] for i in range(16)],
                               eval_cpus=[cpus[64+4*i:68+4*i] for i in range(8)]),
                aggregation='Average 800K/900K/1M success within each seed, then mean/sample SD across seeds0/1/2/3.',
                learner_changes='None: existing frozen trainers and evaluators are used unchanged.',
                cancellation='STOP file pauses this matrix; checkpointed state is retained. No other study is restarted.')
    assert_matrix(plan)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'analysis').mkdir(exist_ok=True)
    (root / 'validation').mkdir(exist_ok=True)
    atomic(root / 'registration_v1.json', plan)
    write_csv(root / 'analysis/experiment_matrix_v1.csv', [
        {k: j.get(k) for k in ('id', 'code', 'environment', 'gamma', 'seed', 'action', 'resume_step', 'output', 'evaluation_directory')}
        for j in reused + jobs])
    atomic(root / 'validation/reuse_audit_v1.json', dict(status='PASS', completed_cells=44,
        completed_evaluations=264, scheduled_cells=36, resume_cells=5, fresh_cells=31,
        source_files=len(sources), evidence_files=len(evidence), resume_steps={j['id']: j['resume_step'] for j in jobs if j['resume_step']}))
    for path in sources:
        target = root / 'source' / Path(path).relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(json.dumps(dict(root=str(root), reused=44, scheduled=36, resume=5, fresh=31)), flush=True)


if __name__ == '__main__':
    main()
