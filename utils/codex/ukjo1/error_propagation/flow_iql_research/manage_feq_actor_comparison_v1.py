"""Register, schedule, select alpha and summarize the FEQ actor comparison.

No training dependencies are imported by this controller. Pilot seed 4 uses
held-out evaluation seeds; final seeds 0--3 never influence alpha selection.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import (
    FAMILIES, PREFIX, STEPS, atomic, audit_config, sha, write_csv,
)

ROOT = Path('logs/codex/ukjo1/error_propagation/diagnosis/feq_actor_fql_cvae_g0995_20261125_v1')
BASELINE = Path('logs/codex/ukjo1/error_propagation/diagnosis/finite_mean_default10_gamma_pair_n9_seed0123_20261124_v1')
PUBLIC_FQL_ALPHA = dict(aml=10., amg=10., hmm=30., hml=30., soccer=10.,
                        cube1=300., cube2=300., puzzle3=1000., puzzle4=1000., scene=300.)


def dump_job(root, job):
    path = root/'jobs'/f"{job['id']}.json"
    if path.exists():
        assert json.loads(path.read_text()) == job
    else:
        atomic(path, job)


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'registration.json').exists():
        return
    with (BASELINE/'analysis/per_seed_results_final_v1.csv').open() as f:
        baselines = [r for r in csv.DictReader(f) if float(r['gamma']) == .995 and r['deployment'] == 'actor']
    assert len(baselines) == 40
    prior = json.loads((BASELINE/'registration_v1.json').read_text())
    original_jobs = {Path(j['output']).resolve(): j for j in prior['jobs'] + prior['reused_jobs']}
    refs = []; architectures = {}
    for row in baselines:
        run = Path(row['run_directory'])
        cfg = json.loads((run/'config.json').read_text())
        audit_config(original_jobs[run.resolve()], cfg)
        architectures[row['environment']] = dict(actor_hidden_dims=cfg['identity']['config']['actor_hidden_dims'],
                                                value_hidden_dims=cfg['identity']['config']['value_hidden_dims'])
        assert json.loads((run/'completed.json').read_text())['step'] == 1000000
        evaluations = []
        for step in STEPS:
            path = Path(row['evaluation_directory'])/f'{step}_actor.json'
            result = json.loads(path.read_text())
            assert result['n_episodes'] == 50 and result['first_episode_seed'] == 42
            assert result['training_seed'] == int(row['seed']) and result['environment'] == row['environment']
            assert result['config_sha256'] == sha(run/'config.json')
            assert abs(result['success_rate']*100-float(row[f'success_{step//1000}K_pct' if step != 1000000 else 'success_1M_pct'])) < 1e-8
            evaluations.append(dict(path=str(path), sha256=sha(path)))
        refs.append(dict(code=row['code'], environment=row['environment'], seed=int(row['seed']),
            run=str(run), config_sha256=sha(run/'config.json'), evaluations=evaluations,
            last3_success_pct=float(row['last3_success_pct'])))
    pilots = []
    for code, env in FAMILIES:
        for method in ('fql', 'cvae'):
            alphas = [PUBLIC_FQL_ALPHA[code]*x for x in (.1, 1., 10.)] if method == 'fql' else [.25, 2.5, 25.]
            name = f'tune_{code}_{method}_s4'
            job = dict(id=name, stage='tuning', code=code, environment=env, seed=4,
                output=str(root/'runs'/name), steps=1000000,
                actors={f'{method}_{tag}': dict(method=method, alpha=a) for tag, a in zip(('low', 'mid', 'high'), alphas)})
            dump_job(root, job); pilots.append(name)
    plan = dict(study='FEQ deployment actor comparison v1', gamma=.995, bootstrap_samples=9,
        environments=[e for _, e in FAMILIES], training_steps=1000000,
        final_training_seeds=[0, 1, 2, 3], tuning_training_seeds=[4],
        evaluation_steps=STEPS, episodes_per_point=50, tuning_first_episode_seed=10042,
        final_first_episode_seed=42, selection='per-environment/method average of 800K/900K/1M; ties prefer middle then low then high alpha',
        alpha_grid_fql='public environment alpha times [0.1, 1, 10]', alpha_grid_cvae=[.25, 2.5, 25.],
        public_fql_alpha=PUBLIC_FQL_ALPHA, kl_coefficient=.5, cvae_latent_dim='2 * action_dim',
        cvae_encoder='first two layers of the environment baseline width, GELU',
        deployment_network='four GELU layers at the environment baseline width, no layer norm',
        architectures=architectures,
        cvae_prior='untruncated standard normal; Q1 guidance applied to prior decoder actions',
        fql_loss='alpha * mean((actor(s,z)-BCFlow(s,z))^2) - mean(mean_heads_Q(s,clip(actor)))',
        cvae_loss='mean(reconstruction^2) + 0.5 * mean(KL per latent dimension) - alpha * mean(Q1(s,decoder(s,z_prior))) / stop_gradient(max(mean(abs(Q1)),1e-6))',
        update_ratio='all new actors one update per FEQ Q update; TD3+BC loss, not its delayed update schedule',
        base_control='unmodified FEQ gamma .995 N9; AntMaze min target heads, others mean; EMA .005; batch256; lr3e-4; target clipping',
        base_independence='actors never supply proposals or gradients to the Q/Flow learner; final actors share a base state',
        pilot_jobs=pilots, expected_training_jobs=60, expected_new_actor_trajectories=140,
        expected_evaluations=420, reused_mdn=refs,
        baseline_summary_sha256=sha(BASELINE/'analysis/per_seed_results_final_v1.csv'),
        sources=dict(FQL='https://github.com/seohongpark/fql/blob/master/agents/fql.py',
                     TD3_BC='https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py',
                     BCQ_CVAE='https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py'),
        train_slots=[[g, s] for s in range(8) for g in (0, 1)],
        train_cpus=[list(range(i*4, i*4+4)) for i in range(16)],
        eval_cpus=[list(range(64+i*4, 68+i*4)) for i in range(8)])
    atomic(root/'registration.json', plan)


def eval_path(root, job, actor, step):
    return root/'evaluations'/job['id']/f'{step}_{actor}.json'


def read_eval(path, job, actor, step):
    result = json.loads(path.read_text())
    assert result['stage'] == job['stage'] and result['training_seed'] == job['seed']
    assert result['actor'] == actor and result['step'] == step
    assert result['method'] == job['actors'][actor]['method'] and result['alpha'] == job['actors'][actor]['alpha']
    assert result['environment'] == job['environment']
    assert result['first_episode_seed'] == (10042 if job['stage'] == 'tuning' else 42)
    assert result['n_episodes'] == len(result['episode_successes']) == 50
    assert result['no_optimizer_updates'] and result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
    assert abs(statistics.mean(result['episode_successes']) - result['success_rate']) < 1e-12
    return result


def select_and_enqueue(root, jobs):
    for code, env in FAMILIES:
        selections = {}
        for method in ('fql', 'cvae'):
            job = jobs[f'tune_{code}_{method}_s4']
            candidates = []
            for name, spec in job['actors'].items():
                paths = [eval_path(root, job, name, step) for step in STEPS]
                if not all(p.exists() for p in paths):
                    continue
                values = [read_eval(p, job, name, step)['success_rate'] for p, step in zip(paths, STEPS)]
                candidates.append(dict(actor=name, alpha=spec['alpha'], mean_success=statistics.mean(values),
                    checkpoint_successes=values, evidence=[dict(path=str(p), sha256=sha(p)) for p in paths]))
            if len(candidates) == 3:
                preference = {'mid': 0, 'low': 1, 'high': 2}
                winner = min(candidates, key=lambda c: (-c['mean_success'], preference[c['actor'].rsplit('_', 1)[1]]))
                selections[method] = dict(selected=winner, candidates=candidates,
                    boundary_selected=not winner['actor'].endswith('_mid'))
        if len(selections) != 2:
            continue
        selected = dict(code=code, environment=env, tuning_seed=4, selection=selections,
                        no_final_seed_results_used=True)
        path = root/'selections'/f'{code}.json'
        if path.exists():
            assert json.loads(path.read_text()) == selected
        else:
            atomic(path, selected)
        for seed in range(4):
            name = f'final_{code}_s{seed}'
            job = dict(id=name, stage='final', code=code, environment=env, seed=seed,
                output=str(root/'runs'/name), steps=1000000,
                actors={m: dict(method=m, alpha=selections[m]['selected']['alpha']) for m in ('fql', 'cvae')},
                selection_sha256=sha(path))
            dump_job(root, job); jobs[name] = job


def analyze(root):
    plan = json.loads((root/'registration.json').read_text())
    jobs = {p.stem: json.loads(p.read_text()) for p in (root/'jobs').glob('*.json')}
    rows = []
    for job in jobs.values():
        if job['stage'] != 'final':
            continue
        for name, spec in job['actors'].items():
            paths = [eval_path(root, job, name, step) for step in STEPS]
            if all(p.exists() for p in paths):
                values = [100*read_eval(p, job, name, s)['success_rate'] for p, s in zip(paths, STEPS)]
                rows.append(dict(code=job['code'], environment=job['environment'], method=spec['method'], seed=job['seed'],
                    alpha=spec['alpha'], success_800K=values[0], success_900K=values[1], success_1M=values[2],
                    last3_success_pct=statistics.mean(values), source=job['output']))
    for ref in plan['reused_mdn']:
        values = [100*json.loads(Path(e['path']).read_text())['success_rate'] for e in ref['evaluations']]
        rows.append(dict(code=ref['code'], environment=ref['environment'], method='mdn', seed=ref['seed'], alpha='',
            success_800K=values[0], success_900K=values[1], success_1M=values[2],
            last3_success_pct=statistics.mean(values), source=ref['run']))
    summary = []
    for code, env in FAMILIES:
        for method in ('mdn', 'fql', 'cvae'):
            selected = [r for r in rows if r['code'] == code and r['method'] == method]
            vals = [r['last3_success_pct'] for r in selected]
            summary.append(dict(code=code, environment=env, method=method, seeds=len(vals),
                success_mean_pct=statistics.mean(vals) if vals else '',
                success_sample_sd_pct=statistics.stdev(vals) if len(vals) > 1 else '',
                alpha=selected[0]['alpha'] if selected else '', complete=len(vals) == 4))
    out = root/'analysis'; out.mkdir(exist_ok=True)
    write_csv(out/'per_seed.csv', sorted(rows, key=lambda r: (r['code'], r['method'], r['seed'])))
    write_csv(out/'task_mean_sd.csv', summary)
    complete = len(rows) == 120 and all(r['complete'] for r in summary)
    lines = ['FEQ actor comparison: gamma=.995, bootstrap N=9',
             f'Final comparison complete: {complete}',
             'Score: mean of 800K/900K/1M, 50 episodes each; mean +/- sample SD over four training seeds.',
             'Alpha selected using separate training seed4 and episode seeds10042..10091.',
             'Final evaluation seeds42..91. MDN results reused from the audited default gamma study.', '']
    for code, _ in FAMILIES:
        cells = []
        for method in ('mdn', 'fql', 'cvae'):
            r = next(r for r in summary if r['code'] == code and r['method'] == method)
            cells.append(f"{method}: {r['success_mean_pct']:.2f} +/- {r['success_sample_sd_pct']:.2f}" if r['complete'] else f"{method}: pending ({r['seeds']}/4)")
        lines.append(code + ' | ' + ' | '.join(cells))
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    atomic(out/'summary.json', dict(complete=complete, final_actor_seed_results=len(rows), expected=120,
        mdn_reused=40, time=time.time()))
    if complete:
        paired = []
        for code, env in FAMILIES:
            for method in ('fql', 'cvae'):
                differences = []
                for seed in range(4):
                    new = next(r for r in rows if r['code'] == code and r['method'] == method and r['seed'] == seed)
                    old = next(r for r in rows if r['code'] == code and r['method'] == 'mdn' and r['seed'] == seed)
                    differences.append(new['last3_success_pct'] - old['last3_success_pct'])
                paired.append(dict(code=code, method=method, mean_delta_vs_mdn_pp=statistics.mean(differences),
                    paired_sample_sd_pp=statistics.stdev(differences), **{f'seed{s}_delta_pp': d for s, d in enumerate(differences)}))
        write_csv(out/'paired_differences_vs_mdn.csv', paired)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(FAMILIES)); width = .25
        for i, (method, label, color) in enumerate((('mdn', 'MDN (existing)', '#0072b2'),
                ('fql', 'Noise MLP / FQL loss', '#d55e00'), ('cvae', 'CVAE / TD3+BC-style loss', '#009e73'))):
            selected = [next(r for r in summary if r['code'] == code and r['method'] == method) for code, _ in FAMILIES]
            ax.bar(x+(i-1)*width, [r['success_mean_pct'] for r in selected], width,
                yerr=[r['success_sample_sd_pct'] for r in selected], capsize=2, label=label, color=color, alpha=.8)
            for j, (code, _) in enumerate(FAMILIES):
                vals = [r['last3_success_pct'] for r in rows if r['code'] == code and r['method'] == method]
                ax.scatter(j+(i-1)*width+np.linspace(-.04, .04, 4), vals, s=9, color='black', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([c for c, _ in FAMILIES])
        ax.set_ylim(0, 105); ax.set_ylabel('Success (%)'); ax.grid(axis='y', alpha=.15)
        ax.set_title('FEQ, gamma=0.995, N=9; training seeds 0–3; alpha tuned on seed 4')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3)
        fig.tight_layout()
        (root/'figures').mkdir(exist_ok=True)
        fig.savefig(root/'figures/actor_comparison.png', dpi=200, bbox_inches='tight')
        fig.savefig(root/'figures/actor_comparison.pdf', bbox_inches='tight')
        plt.close(fig)
    return complete


def birth(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if text[0] == 'Z' else text[19]
    except FileNotFoundError:
        return None


def worker_env(gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', XLA_PYTHON_CLIENT_MEM_FRACTION='0.1')
    if gpu is None:
        env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    else:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='true')
        env.pop('JAX_PLATFORMS', None); env.pop('JAX_PLATFORM_NAME', None)
    return env


def run(root):
    plan = json.loads((root/'registration.json').read_text())
    lock = (root/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root/'controller_status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    workers = previous.get('workers', {}); attempts = previous.get('attempts', {}); failed = previous.get('failed', {})
    children = {}; last_analysis = 0.; stopping = [False]
    def stop(signum, frame): stopping[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def event(kind, **kw):
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind, time=time.time(), **kw))+'\n')
        print(kind, kw, flush=True)
    def spawn(token, command, cpus, env, logpath, kind, slot, output):
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('a') as log:
            process = subprocess.Popen(['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-u', *command],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[token] = process
        attempts[token] = attempts.get(token, 0)+1
        workers[token] = dict(pid=process.pid, birth=birth(process.pid), kind=kind, slot=slot,
            started=time.time(), output=str(output), command=command)
        event('START', token=token, pid=process.pid, kind_of_job=kind, slot=slot, attempt=attempts[token])
        # Persist every launch so a controller restart can adopt live workers.
        atomic(status_path, dict(phase='running', workers=workers, attempts=attempts, failed=failed, time=time.time()))
    while True:
        stopping[0] |= (root/'STOP').exists()
        jobs = {p.stem: json.loads(p.read_text()) for p in (root/'jobs').glob('*.json')}
        select_and_enqueue(root, jobs)
        for token, worker in list(workers.items()):
            process = children.get(token)
            code = process.poll() if process is not None else None
            alive = birth(worker['pid']) == worker['birth'] and worker['birth'] is not None
            if alive:
                if stopping[0]:
                    os.kill(worker['pid'], signal.SIGTERM)
                continue
            completed = Path(worker['output']).exists()
            event('FINISH', token=token, complete=completed, exit_code=code)
            workers.pop(token)
            if not completed and not stopping[0] and attempts[token] >= 3:
                failed[token] = dict(exit_code=code, attempts=attempts[token], time=time.time())
        train_done = [k for k, j in jobs.items() if (Path(j['output'])/'completed.json').exists()]
        if not stopping[0]:
            pending = [j for k, j in jobs.items() if k not in train_done and 'train:'+k not in workers and 'train:'+k not in failed]
            # Final jobs can start as soon as that environment's tuning finishes.
            pending.sort(key=lambda j: (j['stage'] != 'final', j['id']))
            used = {w['slot'] for w in workers.values() if w['kind'] == 'train'}
            for slot, (gpu, local_slot) in enumerate(plan['train_slots']):
                if slot in used or not pending:
                    continue
                job = pending.pop(0); out = Path(job['output'])
                spawn('train:'+job['id'], ['-m', PREFIX+'train_feq_actor_comparison_v1', '--job', str(root/'jobs'/f"{job['id']}.json")],
                    plan['train_cpus'][slot], worker_env(gpu), out/'stdout.log', 'train', slot, out/'completed.json')
        eval_done = 0; eval_pending = []
        for job in jobs.values():
            for name in job['actors']:
                for step in STEPS:
                    output = eval_path(root, job, name, step)
                    token = f"eval:{job['id']}:{name}:{step}"
                    if output.exists():
                        read_eval(output, job, name, step); eval_done += 1
                    elif token not in workers and token not in failed and (Path(job['output'])/f'actor_{name}_{step}.pkl').exists():
                        eval_pending.append((token, job, name, step, output))
        # Prioritize pilot results to release final jobs without an idle barrier.
        eval_pending.sort(key=lambda x: (x[1]['stage'] != 'tuning', x[3], x[1]['id'], x[2]))
        if not stopping[0]:
            used = {w['slot'] for w in workers.values() if w['kind'] == 'eval'}
            for slot, cpus in enumerate(plan['eval_cpus']):
                if slot in used or not eval_pending:
                    continue
                token, job, name, step, output = eval_pending.pop(0)
                spawn(token, ['-m', PREFIX+'evaluate_feq_actor_comparison_v1', '--run', job['output'],
                    '--actor', name, '--step', str(step), '--output', str(output)], cpus, worker_env(),
                    output.with_suffix('.log'), 'eval', slot, output)
        complete = len(train_done) == 60 and eval_done == 420 and not workers
        active_training = sum(w['kind'] == 'train' for w in workers.values())
        phase = 'complete' if complete else 'stopping' if stopping[0] else 'failed' if failed and not workers else 'running'
        status = dict(phase=phase, pid=os.getpid(), birth=birth(os.getpid()), workers=workers, attempts=attempts, failed=failed,
            training_complete=len(train_done), expected_training=60, registered_training=len(jobs),
            active_training=active_training, active_evaluation=len(workers)-active_training,
            evaluations_complete=eval_done, expected_evaluations=420,
            selections_complete=len(list((root/'selections').glob('*.json'))), time=time.time())
        atomic(status_path, status)
        if time.time()-last_analysis > 600 or complete:
            analysis_complete = analyze(root); last_analysis = time.time()
            if complete:
                assert analysis_complete
        if complete or (stopping[0] and not workers) or phase == 'failed':
            return
        time.sleep(5)


def supervise(root):
    lock = (root/'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (root/'STOP').exists():
        status_path = root/'controller_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('phase') in ('complete', 'failed'):
            atomic(root/'supervisor_status.json', dict(phase=status['phase'], time=time.time(), pid=os.getpid()))
            return
        if status.get('pid') and birth(status['pid']) == status.get('birth'):
            time.sleep(10); continue
        with (root/'controller_stdout.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', PREFIX+'manage_feq_actor_comparison_v1', 'run', '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(root/'supervisor_status.json', dict(phase='running', controller_pid=process.pid, pid=os.getpid(), time=time.time()))
        while process.poll() is None:
            time.sleep(10)
        if process.returncode:
            with (root/'supervisor_stdout.log').open('a') as log:
                log.write(f'Controller exit={process.returncode}; retry in 15s\n')
        time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('register', 'run', 'supervise', 'analyze'))
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args(); root = args.root.resolve()
    {'register': register, 'run': run, 'supervise': supervise, 'analyze': analyze}[args.action](root)


if __name__ == '__main__':
    main()
