"""Registered 800K -> 1M paired Q-stabilization experiment (no baseline edits)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, sha

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())


def prepare(root):
    from utils.codex.ukjo1.error_propagation.flow_iql_research import critic_stabilization_core_v1 as core
    oldroot = PROJECT / 'logs/diagnosis/finite_mean_sample_budget_multiseed_20261115'
    oldpath = oldroot / 'registration_v1.json'
    old = json.loads(oldpath.read_text())
    cases, jobs, hashes = [], [], {}
    for short, env, n in (
        ('hmmed', 'humanoidmaze-medium-navigate-singletask-v0', 18),
        ('pz4', 'puzzle-4x4-play-singletask-v0', 27),
    ):
        for seed in range(3):
            parent = next(j for j in old['jobs'] + old['reused_jobs']
                          if (j['environment'], j['bootstrap_samples'], j['seed']) == (env, n, seed))
            run = Path(parent['output']).resolve()
            config = json.loads((run / 'config.json').read_text())
            assert config['identity']['config']['discount'] == .999
            for p, digest in config['source_sha256'].items():
                assert sha(p) == digest, p
                hashes[p] = digest
            ev = Path(parent['evaluation_directory']).resolve() if parent.get('reused') else oldroot / 'evaluations' / parent['id']
            case = dict(id=f'{short}_N{n}_s{seed}', environment=env, seed=seed,
                bootstrap_samples=n, recipe=f'n{n}', parent_run=str(run),
                config_sha256=sha(run / 'config.json'),
                parent_checkpoints={str(s): sha(run / f'checkpoint_{s}.pkl') for s in (800000, 900000, 1000000)},
                baseline_evaluation_directory=str(ev.resolve()),
                baseline_evaluation_sha256={f'{s}_{mode}': sha(ev / f'{s}_{mode}.json')
                    for s in (800000, 900000, 1000000) for mode in ('lean', 'actor')})
            cases.append(case)
            for method in ('q_lr_0p1', 'q_cosine'):
                label = case['id'] + '_' + method
                jobs.append(dict(id=label, case=case['id'], method=method,
                                 output=str(root / 'runs' / label)))
    paths = core.source_paths() + [Path(__file__), PROJECT / 'utils/codex/ukjo1/error_propagation/flow_iql_research/evaluate_sample_budget_v1.py',
        PROJECT / 'utils/codex/ukjo1/error_propagation/flow_iql_research/validate_critic_stabilization_v1.py']
    hashes.update({str(p.resolve()): sha(p) for p in paths})
    plan = dict(protocol='late_Q_stabilization_v1', created_unix=time.time(),
        parent_registration_sha256=sha(oldpath), cases=cases, jobs=jobs, source_sha256=hashes,
        start=800000, end=1000000, training_seeds=[0, 1, 2],
        methods={'baseline': 'original completed continuation, Q lr=3e-4',
            'q_lr_0p1': 'Q Adam updates scaled by 0.1: Q lr=3e-5 throughout 800K-1M',
            'q_cosine': 'Q-only half cosine from lr=3e-4 at 800001 to zero at 1M'},
        fixed={'discount': .999, 'expectile': .9, 'batch_size': 256,
            'BC_Flow_lr': 3e-4, 'actor_lr': 3e-4, 'target_Q_EMA': .005,
            'target_Q_timing': 'reference old-online Q', 'actor_teacher_candidates': 9,
            'lean_deployment_candidates': 9, 'actor_decoder': 'single selected mixture component location'},
        evaluation_steps=[800000, 900000, 1000000],
        new_evaluation_steps=[900000, 1000000], evaluation_episodes=50,
        evaluation_reset_seeds=list(range(42, 92)), evaluation_CPU_cores_per_worker=4,
        primary_metrics=['per-task mean and sample SD of seed-level 800K/900K/1M success averages',
            'per-task mean and sample SD of 1M success', 'paired 1M-minus-800K success',
            'same-state same-nine-action-bank argmax change, 900K to 1M'],
        limits=['Two tasks and N contexts selected after observing instability, not held-out benchmark selection.',
            'Three original training seeds; each intervention shares its entire 800K prefix.',
            'Tests late training only; cannot repair or diagnose instability accumulated before 800K.',
            'Reduced ranking churn without high return does not count as a performance improvement.',
            'No best-checkpoint or per-seed method selection. Evaluation episodes are not training seeds.'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'analysis').mkdir()
    (root / 'source').mkdir()
    for index, (name, digest) in enumerate(hashes.items()):
        target = root / 'source' / f'{index:02d}_{Path(name).name}'
        target.write_bytes(Path(name).read_bytes())
        assert sha(target) == digest
    atomic(root / 'registration_v1.json', plan)
    print(json.dumps(dict(root=str(root), training_jobs=len(jobs), new_updates=2400000)), flush=True)


def load_plan(path):
    plan = json.loads(path.read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(name) == digest, name
    return plan


def train(plan_path, job_id, stop_at=None):
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ogbench
    from utils.codex.ukjo1.error_propagation.flow_iql_research import critic_stabilization_core_v1 as core
    from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as ref
    from utils.codex.ukjo1.error_propagation.flow_iql_research.train_sample_budget_v1 import sync

    plan = load_plan(plan_path)
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    case = next(c for c in plan['cases'] if c['id'] == job['case'])
    out, parent = Path(job['output']), Path(case['parent_run'])
    out.mkdir(parents=True, exist_ok=True)
    assert sha(parent / 'config.json') == case['config_sha256']
    original = json.loads((parent / 'config.json').read_text())
    identity = dict(original['identity'])
    identity['stabilization'] = dict(method=job['method'], start=plan['start'], end=plan['end'],
                                    parent_checkpoint_sha256=case['parent_checkpoints']['800000'])
    conf = dict(identity=identity, source_sha256=plan['source_sha256'], job=job,
                registration_sha256=sha(plan_path))
    if (out / 'config.json').exists():
        assert json.loads((out / 'config.json').read_text()) == conf
    else:
        atomic(out / 'config.json', conf)
    if (out / 'completed.json').exists():
        return
    device = jax.devices('gpu')[0]
    cfg = ref.config_for(case['environment'], case['recipe'])
    assert json.loads(json.dumps(cfg)) == original['identity']['config']
    env, raw, validation = ogbench.make_env_and_datasets(case['environment'])
    env.close()
    data = {k: np.asarray(raw[k], np.float32) for k in ('observations', 'actions', 'next_observations', 'rewards', 'masks')}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    del raw, validation
    dataset = json.loads((parent / 'dataset.json').read_text())
    atomic(out / 'dataset.json', dataset)
    bounds = (dataset['return_lower'], dataset['return_upper'])
    template = ref.create(case['seed'], jax.device_put(data['observations'][:1], device),
                          jax.device_put(data['actions'][:1], device), cfg, case['recipe'])
    latest = out / 'latest.pkl'
    source = latest if latest.exists() else parent / 'checkpoint_800000.pkl'
    if source != latest:
        assert sha(source) == case['parent_checkpoints']['800000']
    with source.open('rb') as stream:
        saved = pickle.load(stream)
    if source == latest:
        assert saved['identity'] == identity and saved['source_sha256'] == plan['source_sha256']
    else:
        assert saved['identity'] == original['identity'] and saved['step'] == 800000
    state = jax.device_put(flax.serialization.from_state_dict(template, saved['state']), device)
    step = saved['step']
    rng = np.random.RandomState(case['seed'])
    rng.set_state(saved['replay_rng'])
    scan = ref.make_scan(core.build_update(case['recipe'], bounds, job['method']))
    probe = ref.make_probe(case['recipe'], bounds)
    fixed_idx = np.random.RandomState(case['seed']+19401).randint(len(data['actions']), size=512)
    fixed = {k: jax.device_put(v[fixed_idx], device) for k, v in data.items()}
    bankpath = out / 'ranking_bank.npz'
    if bankpath.exists():
        with np.load(bankpath) as a:
            rank_obs, rank_actions = jax.device_put(a['observations'], device), jax.device_put(a['actions'], device)
    else:
        anchor = jax.device_put(flax.serialization.from_state_dict(template, saved['state']), device)
        rank_obs, noise = ref.lean.bank(anchor.base.agent, fixed['observations'], jax.random.PRNGKey(74191), 9)
        rank_actions = ref.lean.flow(anchor.base.agent, rank_obs, noise, 10)
        np.savez_compressed(bankpath, observations=np.asarray(rank_obs), actions=np.asarray(rank_actions), replay_indices=fixed_idx)

    @jax.jit
    def rank_heads(params):
        return template.base.agent.network.replace(params=params).select('critic')(rank_obs, actions=rank_actions)

    def ranking_snapshot():
        path = out / f'ranking_{step}.npz'
        if not path.exists():
            np.savez_compressed(path, heads=np.asarray(rank_heads(state.base.agent.network.params)),
                                step=step, bank_sha256=sha(bankpath))

    metadata = ref.actor_metadata(cfg, dataset['observation_dim'], dataset['action_dim'])
    began = time.monotonic()
    count, scan_seconds, prepare_seconds = 0, 0., 0.
    endpoint = stop_at or plan['end']

    def save():
        payload = dict(step=step, state=flax.serialization.to_state_dict(jax.device_get(state)),
            replay_rng=rng.get_state(), identity=identity, source_sha256=plan['source_sha256'],
            parent_checkpoint_sha256=case['parent_checkpoints']['800000'])
        temporary = latest.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(latest)
        target = out / f'checkpoint_{step}.pkl'
        assert not target.exists()
        os.link(latest, target)
        artifact = ref.export_actor(state, metadata)
        artifact.update(identity=identity, source_sha256=plan['source_sha256'],
            checkpoint_sha256=sha(target), config_sha256=sha(out / 'config.json'), step=step)
        actor_path = out / f'actor_{step}.pkl'
        with actor_path.with_suffix('.tmp').open('wb') as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        actor_path.with_suffix('.tmp').replace(actor_path)

    ranking_snapshot()
    while step < endpoint:
        chunk = min(100, endpoint-step)
        before = time.monotonic()
        indices = rng.randint(len(data['actions']), size=(chunk, cfg['batch_size']))
        batches = {k: jax.device_put(v[indices], device) for k, v in data.items()}
        sync(batches)
        prepare_seconds += time.monotonic()-before
        before = time.monotonic()
        state, logs = scan(state, batches)
        sync((state, logs))
        scan_seconds += time.monotonic()-before
        step += chunk
        count += chunk
        if step % 5000 == 0 or step == endpoint:
            values = jax.device_get(probe(state, fixed))
            stats = ref.probe_summary(values, bounds)
            stats.update(step=step, method=job['method'], session_updates=count,
                session_seconds=time.monotonic()-began, scan_seconds=scan_seconds,
                batch_prepare_seconds=prepare_seconds,
                last_chunk_q_lr_multiplier=float(np.asarray(logs['stabilization/q_lr_multiplier']).mean()),
                q_updates=int(state.base.q_updates), actor_updates=int(state.actor_updates), v_updates=int(state.base.v_updates))
            assert stats['finite'] and stats['v_updates'] == 0
            assert stats['q_updates'] == stats['actor_updates'] == step
            with (out / 'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(stats, allow_nan=False)+'\n')
            atomic(out / 'runtime.json', stats)
            print(json.dumps(dict(job=job_id, step=step, finite=stats['finite'],
                seconds=stats['session_seconds'], q_lr_scale=stats['last_chunk_q_lr_multiplier'])), flush=True)
        if step % 50000 == 0 or step == endpoint:
            ranking_snapshot()
        if step in (900000, 1000000) or step == endpoint:
            save()
    atomic(out / ('completed.json' if step == plan['end'] else 'screened.json'), dict(
        step=step, session_updates=count, session_seconds=time.monotonic()-began,
        method=job['method'], finite=True, registration_sha256=sha(plan_path),
        evaluation_status='separate CPU50 controller', timestamp_unix=time.time()))


def controller(plan_path):
    plan = load_plan(plan_path)
    root = plan_path.parent
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    valid = json.loads((root / 'analysis/implementation_validation_v1.json').read_text())
    assert valid['status'] == 'PASS'
    # Four jobs per GPU; each process has the same four-core training affinity.
    slots = [(gpu, lane) for lane in range(4) for gpu in (0, 1)]
    train_running, eval_running, failed = {}, {}, []
    train_done, eval_done = [], []
    logsdir = root / 'worker_logs'
    logsdir.mkdir(exist_ok=True)
    env_common = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl')
    while True:
        for group, done, kind in ((train_running, train_done, 'train'), (eval_running, eval_done, 'eval')):
            for key, item in list(group.items()):
                if item['process'].poll() is None:
                    continue
                item['log'].close()
                if item['process'].returncode:
                    failed.append(dict(kind=kind, id=key, code=item['process'].returncode))
                else:
                    done.append(key)
                del group[key]
        for index, slot in enumerate(slots):
            if failed or any(w['slot'] == index for w in train_running.values()):
                continue
            job = next((j for j in plan['jobs'] if not (Path(j['output']) / 'completed.json').exists()
                        and j['id'] not in train_running), None)
            if job is None:
                continue
            cores = list(range(index*4, index*4+4))
            log = (logsdir / f'train_{job["id"]}.log').open('a')
            env = dict(env_common, CUDA_VISIBLE_DEVICES=str(slot[0]), XLA_PYTHON_CLIENT_MEM_FRACTION='0.20')
            command = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, '-u', '-m',
                'utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1', 'train',
                '--plan', str(plan_path), '--job', job['id']]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            train_running[job['id']] = dict(process=process, log=log, slot=index)
        for index in range(8):
            if failed or any(w['slot'] == index for w in eval_running.values()):
                continue
            pending = []
            for job in plan['jobs']:
                for step in plan['new_evaluation_steps']:
                    for mode in ('lean', 'actor'):
                        label = f'{job["id"]}_{step}_{mode}'
                        output = root / 'evaluations' / job['id'] / f'{step}_{mode}.json'
                        if (Path(job['output']) / f'actor_{step}.pkl').exists() and not output.exists() and label not in eval_running:
                            pending.append((label, job, step, mode, output))
            if not pending:
                continue
            label, job, step, mode, output = pending[0]
            log = (logsdir / f'eval_{label}.log').open('a')
            env = dict(env_common, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
            command = ['taskset', '-c', ','.join(map(str, range(64+4*index, 68+4*index))),
                sys.executable, '-u', '-m', 'utils.codex.ukjo1.error_propagation.flow_iql_research.evaluate_sample_budget_v1',
                '--run', job['output'], '--step', str(step), '--mode', mode, '--episodes', '50', '--output', str(output)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            eval_running[label] = dict(process=process, log=log, slot=index)
        complete_training = sum((Path(j['output']) / 'completed.json').exists() for j in plan['jobs'])
        complete_eval = sum((root / 'evaluations' / j['id'] / f'{s}_{m}.json').exists()
            for j in plan['jobs'] for s in plan['new_evaluation_steps'] for m in ('lean', 'actor'))
        phase = 'failed' if failed else 'complete' if complete_training == len(plan['jobs']) and complete_eval == len(plan['jobs'])*4 else 'running'
        atomic(root / 'controller_status_v1.json', dict(phase=phase, pid=os.getpid(), timestamp_unix=time.time(),
            training_complete=complete_training, evaluation_complete=complete_eval, failures=failed,
            training_running={k:dict(pid=w['process'].pid, slot=w['slot']) for k,w in train_running.items()},
            evaluation_running={k:dict(pid=w['process'].pid, slot=w['slot']) for k,w in eval_running.items()}))
        if phase == 'complete':
            return
        if failed and not train_running and not eval_running:
            raise RuntimeError(failed)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    for mode in ('train', 'run'):
        p = sub.add_parser(mode)
        p.add_argument('--plan', type=Path, required=True)
        if mode == 'train':
            p.add_argument('--job', required=True)
            p.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output.resolve())
    elif args.mode == 'train':
        train(args.plan.resolve(), args.job, args.stop_at)
    else:
        controller(args.plan.resolve())


if __name__ == '__main__':
    main()
