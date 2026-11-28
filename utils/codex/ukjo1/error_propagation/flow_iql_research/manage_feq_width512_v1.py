"""Audit, schedule and report the default10 FEQ width-512 comparison.

New training is needed only for five formerly width-256 manipulation tasks.
Existing width-512 tasks are reused. Training seeds 0/1 and both discounts
are fixed in advance; no checkpoint or seed selection based on performance.
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
    FAMILIES, PREFIX, STEPS, atomic, audit_config, audit_evaluation, sha, write_csv,
)

ROOT = Path('logs/codex/ukjo1/error_propagation/diagnosis/feq_width512_default10_g995_g999_seed01_20261127_v1')
BASELINE = Path('logs/codex/ukjo1/error_propagation/diagnosis/finite_mean_default10_gamma_pair_n9_seed0123_20261124_v1')
OTHER = Path('logs/codex/ukjo1/error_propagation/diagnosis/feq_actor_fql_cvae_g0995_20261125_v1')
TRAIN_MODULE = PREFIX + 'train_feq_width512_v1'
EVAL_MODULE = PREFIX + 'evaluate_feq_width512_v1'
THIS_MODULE = PREFIX + 'manage_feq_width512_v1'


def birth(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
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
        env.pop('JAX_PLATFORMS', None)
        env.pop('JAX_PLATFORM_NAME', None)
    return env


def register(root):
    # Loaded only during registration, under the CPU environment.
    from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_width512_core_v1 as core
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'registration.json').exists():
        return
    with (BASELINE / 'analysis/per_seed_results_final_v1.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['seed']) in (0, 1) and r['deployment'] == 'lean']
    assert len(rows) == 40
    prior = json.loads((BASELINE / 'registration_v1.json').read_text())
    previous_jobs = {Path(j['output']).resolve(): j for j in prior['jobs'] + prior['reused_jobs']}
    baselines, jobs, reused = [], [], []
    evidence = {}
    for row in rows:
        run, evaluations = Path(row['run_directory']), Path(row['evaluation_directory'])
        original = previous_jobs[run.resolve()]
        conf = json.loads((run / 'config.json').read_text())
        cfg = conf['identity']['config']
        gamma, seed = float(row['gamma']), int(row['seed'])
        recipe = 'g995_w512_n9' if gamma == .995 else 'g999_w512_n9'
        expected = core.config_for(row['environment'], recipe)
        width = cfg['value_hidden_dims'][0]
        manipulation = row['code'] in ('cube1', 'cube2', 'puzzle3', 'puzzle4', 'scene')
        assert width == (256 if manipulation else 512)
        assert cfg['value_hidden_dims'] == cfg['actor_hidden_dims'] == [width] * 4
        expected['value_hidden_dims'] = expected['actor_hidden_dims'] = (width,) * 4
        audit_config(original, conf, reference=expected)
        assert json.loads((run / 'completed.json').read_text())['step'] == 1000000
        checked = []
        for mode in ('lean', 'actor'):
            for step in STEPS:
                path = evaluations / f'{step}_{mode}.json'
                score = audit_evaluation(original, conf, step, mode, path, evidence)
                checked.append(dict(mode=mode, step=step, path=str(path), sha256=sha(path), success_pct=score))
        ref = dict(id=f"g{str(gamma)[2:]}_{row['code']}_s{seed}", code=row['code'],
                   environment=row['environment'], gamma=gamma, seed=seed, width=width,
                   output=str(run), config_sha256=sha(run / 'config.json'), evaluations=checked)
        baselines.append(ref)
        if not manipulation:
            reused.append(ref)
            continue
        name = f"g{str(gamma)[2:]}_{row['code']}_w512_s{seed}"
        job = dict(id=name, code=row['code'], environment=row['environment'], gamma=gamma,
                   seed=seed, width=512, recipe=recipe, output=str(root / 'runs' / name),
                   evaluation_directory=str(root / 'evaluations' / name),
                   head_aggregation='mean', baseline_id=ref['id'])
        atomic(root / 'jobs' / f'{name}.json', job)
        jobs.append(job)
    assert len(jobs) == len(reused) == 20
    assert len({(r['environment'], r['gamma'], r['seed']) for r in baselines}) == 40
    sources = {str(p): sha(p) for p in core.source_paths() + [
        Path(core.__file__).parent / 'train_discount_0995_n9_v1.py']}
    for name in sources:
        p = Path(name)
        target = root / 'source' / ('vendor_' + p.name if p.parent.name == 'vendor' else p.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (root / 'source' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    plan = dict(study='FEQ default10 width512, gamma .995/.999, seeds 0/1',
                training_steps=1000000, training_seeds=[0, 1], discounts=[.995, .999],
                target_architecture=[512] * 4, changed_networks=['online_Q', 'target_Q', 'BC_Flow', 'MDN_actor'],
                backup_samples=9, deployment_flow_samples=9, actor='unchanged MDN9 likelihood actor',
                training_control='unchanged FEQ N9; batch256; lr3e-4; EMA .005; expectile .9; clipping',
                evaluation_steps=STEPS, episodes=50, first_episode_seed=42,
                aggregation='mean of 800K/900K/1M within each seed, then mean and sample SD across seeds 0/1',
                baseline=baselines, reused_width512=reused, jobs=jobs, source_sha256=sources,
                baseline_evidence=evidence, expected_training=20, expected_new_evaluations=120,
                train_slots=[[gpu, slot] for slot in range(8) for gpu in (0, 1)],
                train_cpus=[list(range(i * 4, i * 4 + 4)) for i in range(16)],
                eval_cpus=[list(range(64 + i * 4, 68 + i * 4)) for i in range(8)],
                other_study=str(OTHER.resolve()),
                other_study_policy='preserve workers; use free training slots; defer CPU evaluation until other study completes',
                paper='https://arxiv.org/html/2502.02538v1#A5.SS2',
                limitation='Width ablation, not proof of overfitting. Both Q/Flow and MDN capacity change; report Lean and actor separately.')
    atomic(root / 'registration.json', plan)
    print(json.dumps(dict(new_training=len(jobs), reused_width512=len(reused), baseline_cells=len(baselines))))


def analyze(root):
    plan = json.loads((root / 'registration.json').read_text())
    rows = []
    for ref in plan['baseline']:
        assert sha(Path(ref['output']) / 'config.json') == ref['config_sha256']
        for mode in ('lean', 'actor'):
            checks = sorted((r for r in ref['evaluations'] if r['mode'] == mode), key=lambda r: r['step'])
            for r in checks:
                assert sha(r['path']) == r['sha256']
            vals = [r['success_pct'] for r in checks]
            rows.append(dict(code=ref['code'], environment=ref['environment'], gamma=ref['gamma'], seed=ref['seed'],
                             width=ref['width'], deployment=mode, origin='existing', points=3,
                             success_800K=vals[0], success_900K=vals[1], success_1M=vals[2],
                             last3_success_pct=statistics.mean(vals), run=ref['output']))
    new_evals = 0
    for job in plan['jobs']:
        run = Path(job['output'])
        if not (run / 'config.json').exists():
            continue
        conf = json.loads((run / 'config.json').read_text())
        audit_config(job, conf)
        assert conf['identity']['config']['value_hidden_dims'] == [512] * 4
        assert conf['identity']['config']['actor_hidden_dims'] == [512] * 4
        for mode in ('lean', 'actor'):
            values = {}
            for step in STEPS:
                path = Path(job['evaluation_directory']) / f'{step}_{mode}.json'
                if path.exists():
                    values[step] = audit_evaluation(job, conf, step, mode, path)
                    new_evals += 1
            rows.append(dict(code=job['code'], environment=job['environment'], gamma=job['gamma'], seed=job['seed'],
                             width=512, deployment=mode, origin='new', points=len(values),
                             success_800K=values.get(800000, ''), success_900K=values.get(900000, ''),
                             success_1M=values.get(1000000, ''),
                             last3_success_pct=statistics.mean(values.values()) if len(values) == 3 else '', run=job['output']))
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out / 'per_seed_results.csv', rows)
    summary, paired = [], []
    lines = ['FEQ 신경망 폭 비교: default10, seed 0/1, gamma .995/.999',
             '성공률: 각 시드의 800K/900K/1M 50episode 평균. ±는 두 학습 시드의 표본 표준편차.',
             '기존 512인 5환경은 재사용. 원래 256인 5환경만 512를 새로 학습.',
             '폭 확대는 Q/Flow/MDN 전체에 적용. Lean과 actor를 따로 비교하며 과적합 자체의 직접 증명은 아님.', '']
    for code, env in FAMILIES:
        for gamma in plan['discounts']:
            for mode in ('lean', 'actor'):
                groups = {}
                for width in (256, 512):
                    selected = sorted([r for r in rows if r['code'] == code and r['gamma'] == gamma
                                       and r['deployment'] == mode and r['width'] == width and r['points'] == 3],
                                      key=lambda r: r['seed'])
                    if len(selected) != 2:
                        continue
                    groups[width] = selected
                    vals = [r['last3_success_pct'] for r in selected]
                    item = dict(code=code, environment=env, gamma=gamma, deployment=mode, width=width,
                                seeds=2, mean_success_pct=statistics.mean(vals), sample_sd_pct=statistics.stdev(vals),
                                seed0_success_pct=vals[0], seed1_success_pct=vals[1])
                    summary.append(item)
                    lines.append(f"{code} gamma={gamma} {mode} width={width}: {item['mean_success_pct']:.2f} ± {item['sample_sd_pct']:.2f}% (s0={vals[0]:.2f}, s1={vals[1]:.2f})")
                if 256 in groups and 512 in groups:
                    delta = [b['last3_success_pct'] - a['last3_success_pct'] for a, b in zip(groups[256], groups[512])]
                    paired.append(dict(code=code, gamma=gamma, deployment=mode, delta_512_minus_256_pp=statistics.mean(delta),
                                       paired_sample_sd_pp=statistics.stdev(delta), seed0_delta_pp=delta[0], seed1_delta_pp=delta[1]))
    write_csv(out / 'task_mean_sd.csv', summary)
    if paired:
        write_csv(out / 'paired_width_differences.csv', paired)
    complete = new_evals == 120 and all((Path(j['output']) / 'completed.json').exists() for j in plan['jobs'])
    lines.insert(1, '상태: 완료' if complete else f'상태: 진행 중 (새 평가 {new_evals}/120); 일부 결과만으로 폭의 효과를 결론내리지 않음.')
    (out / 'RESULTS_KO.txt').write_text('\n'.join(lines) + '\n')
    atomic(out / 'summary.json', dict(complete=complete, evaluations_complete=new_evals, expected=120, time=time.time()))
    if complete:
        # Validate all final evaluation artifacts once before claiming completion.
        evidence = {}
        for job in plan['jobs']:
            conf = json.loads((Path(job['output']) / 'config.json').read_text())
            for mode in ('lean', 'actor'):
                for step in STEPS:
                    audit_evaluation(job, conf, step, mode, Path(job['evaluation_directory']) / f'{step}_{mode}.json', evidence)
        atomic(out / 'new_evaluation_evidence.json', evidence)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
        codes = ['cube1', 'cube2', 'puzzle3', 'puzzle4', 'scene']
        for gi, gamma in enumerate(plan['discounts']):
            for mi, mode in enumerate(('lean', 'actor')):
                ax = axes[gi, mi]
                for wi, width in enumerate((256, 512)):
                    data = [next(r for r in summary if r['code'] == code and r['gamma'] == gamma
                                 and r['deployment'] == mode and r['width'] == width) for code in codes]
                    x = [i + (wi - .5) * .34 for i in range(len(codes))]
                    ax.bar(x, [r['mean_success_pct'] for r in data], .32,
                           yerr=[r['sample_sd_pct'] for r in data], capsize=3, label=f'Width {width}', alpha=.8)
                    for s in (0, 1):
                        ax.scatter(x, [r[f'seed{s}_success_pct'] for r in data], s=16, color='black', marker='o' if s == 0 else 'x', zorder=3)
                ax.set_xticks(range(len(codes))); ax.set_xticklabels(codes)
                ax.set_ylim(0, 105); ax.set_ylabel('Success (%)')
                ax.set_title(f'gamma={gamma}, {mode}'); ax.legend(); ax.grid(axis='y', alpha=.15)
        fig.tight_layout()
        fig.savefig(out / 'width_comparison.png', dpi=200)
        fig.savefig(out / 'width_comparison.pdf')
        plt.close(fig)
    return complete


def other_usage(plan):
    path = Path(plan['other_study']) / 'controller_status.json'
    if not path.exists():
        return set(), True
    status = json.loads(path.read_text())
    used = {w['slot'] for w in status.get('workers', {}).values()
            if w['kind'] == 'train' and w.get('birth') is not None and birth(w['pid']) == w['birth']}
    # All old jobs are already registered. Do not fill slots if its controller
    # still has an unlaunched training job: it does not know our new queue.
    old_train_done = status.get('training_complete', 0)
    all_old_accounted = old_train_done + len(used) == status.get('expected_training', 60)
    if not all_old_accounted and status.get('phase') != 'complete':
        used = set(range(16))
    return used, status.get('phase') == 'complete'


def run(root):
    plan = json.loads((root / 'registration.json').read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / 'controller_status.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    workers, attempts, failed = old.get('workers', {}), old.get('attempts', {}), old.get('failed', {})
    children, stopping, last_analysis = {}, [False], 0.
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def event(kind, **values):
        value = dict(kind=kind, time=time.time(), **values)
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)
    def persist(**extra):
        atomic(path, dict(phase='running', pid=os.getpid(), birth=birth(os.getpid()), workers=workers,
                         attempts=attempts, failed=failed, time=time.time(), **extra))
    def spawn(token, command, cpus, gpu, logpath, kind, slot, output):
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open('a') as log:
            process = subprocess.Popen(['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-u', *command],
                                       env=worker_env(gpu), stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        children[token] = process
        attempts[token] = attempts.get(token, 0) + 1
        workers[token] = dict(pid=process.pid, birth=birth(process.pid), kind=kind, slot=slot, gpu=gpu,
                              started=time.time(), output=str(output), command=command)
        event('START', token=token, pid=process.pid, slot=slot, gpu=gpu, attempt=attempts[token])
        persist()
    while True:
        stopping[0] |= (root / 'STOP').exists()
        for token, worker in list(workers.items()):
            child = children.get(token)
            code = child.poll() if child is not None else None
            if worker['birth'] is not None and birth(worker['pid']) == worker['birth']:
                if stopping[0]:
                    os.kill(worker['pid'], signal.SIGTERM)
                continue
            done = Path(worker['output']).exists()
            event('FINISH', token=token, complete=done, exit_code=code)
            workers.pop(token)
            if not done and not stopping[0] and attempts[token] >= 3:
                failed[token] = dict(attempts=attempts[token], exit_code=code, time=time.time())
        train_done = [j['id'] for j in plan['jobs'] if (Path(j['output']) / 'completed.json').exists()]
        external_slots, evaluation_available = other_usage(plan)
        if not stopping[0]:
            pending = [j for j in plan['jobs'] if j['id'] not in train_done
                       and 'train:' + j['id'] not in workers and 'train:' + j['id'] not in failed]
            # Start both gammas/seeds across tasks, instead of completing one
            # gamma before starting the other. Reproducible queue order.
            pending.sort(key=lambda j: (j['seed'], j['code'], j['gamma']))
            used = external_slots | {w['slot'] for w in workers.values() if w['kind'] == 'train'}
            for slot, (gpu, local_slot) in enumerate(plan['train_slots']):
                if slot in used or not pending:
                    continue
                j = pending.pop(0); out = Path(j['output'])
                spawn('train:' + j['id'], ['-m', TRAIN_MODULE, '--env', j['environment'], '--recipe', j['recipe'],
                       '--seed', str(j['seed']), '--output', str(out)], plan['train_cpus'][slot], gpu,
                      out / 'stdout.log', 'train', slot, out / 'completed.json')
        eval_done, eval_pending = 0, []
        for j in plan['jobs']:
            for step in STEPS:
                for mode in ('lean', 'actor'):
                    output = Path(j['evaluation_directory']) / f'{step}_{mode}.json'
                    token = f"eval:{j['id']}:{step}:{mode}"
                    if output.exists():
                        eval_done += 1
                    elif token not in workers and token not in failed and (Path(j['output']) / f'actor_{step}.pkl').exists():
                        eval_pending.append((token, j, step, mode, output))
        if not stopping[0] and evaluation_available:
            used = {w['slot'] for w in workers.values() if w['kind'] == 'eval'}
            for slot, cpus in enumerate(plan['eval_cpus']):
                if slot in used or not eval_pending:
                    continue
                token, j, step, mode, output = eval_pending.pop(0)
                spawn(token, ['-m', EVAL_MODULE, '--run', j['output'], '--step', str(step), '--mode', mode,
                             '--output', str(output)], cpus, None, output.with_suffix('.log'), 'eval', slot, output)
        complete = len(train_done) == 20 and eval_done == 120 and not workers
        phase = 'complete' if complete else 'stopping' if stopping[0] else 'failed' if failed and not workers else 'running'
        status = dict(phase=phase, pid=os.getpid(), birth=birth(os.getpid()), workers=workers, attempts=attempts, failed=failed,
                      active_training=sum(w['kind'] == 'train' for w in workers.values()),
                      active_evaluation=sum(w['kind'] == 'eval' for w in workers.values()),
                      external_training_slots=sorted(external_slots), evaluations_deferred_for_other_study=not evaluation_available,
                      training_complete=len(train_done), expected_training=20, reused_training=20,
                      evaluations_complete=eval_done, expected_evaluations=120, time=time.time())
        if time.time() - last_analysis > 600 or complete:
            analysis_complete = analyze(root)
            last_analysis = time.time()
            if complete:
                assert analysis_complete
        atomic(path, status)
        if complete or (stopping[0] and not workers) or phase == 'failed':
            return
        time.sleep(5)


def supervise(root):
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (root / 'STOP').exists():
        path = root / 'controller_status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        if status.get('phase') in ('complete', 'failed'):
            atomic(root / 'supervisor_status.json', dict(phase=status['phase'], pid=os.getpid(), time=time.time()))
            return
        if status.get('birth') is not None and birth(status['pid']) == status['birth']:
            time.sleep(10)
            continue
        with (root / 'controller_stdout.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', THIS_MODULE, 'run', '--root', str(root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(root / 'supervisor_status.json', dict(phase='running', controller_pid=process.pid, pid=os.getpid(), time=time.time()))
        while process.poll() is None:
            time.sleep(10)
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'supervise', 'analyze'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    {'register': register, 'run': run, 'supervise': supervise, 'analyze': analyze}[args.action](args.root.resolve())


if __name__ == '__main__':
    main()
