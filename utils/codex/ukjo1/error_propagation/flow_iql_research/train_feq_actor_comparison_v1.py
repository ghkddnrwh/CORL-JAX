"""Resumable FEQ learner with independent deployment actor optimizers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time

import flax.serialization
import jax
import numpy as np
import ogbench

from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_actor_comparison_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import atomic, sha

STEPS = (800000, 900000, 1000000)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def sync(tree):
    for value in jax.tree_util.tree_leaves(tree):
        if hasattr(value, 'block_until_ready'):
            value.block_until_ready()


def tree_digest(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        x = np.asarray(leaf)
        digest.update(str((x.shape, x.dtype.str)).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    out = Path(job['output']); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = {str(p): sha(p) for p in core.source_paths() + [Path(__file__).resolve()]}
    cfg = core.config_for(job['environment'], 'g995_n9', job.get('width'))
    identity = json.loads(json.dumps(dict(job=job, config=cfg, batch_size=256,
        chunk=100, steps=job.get('steps', 1000000), optimizer='adam', actor_lr=cfg['lr'],
        q_and_flow_source='unmodified_sample_budget_core_v1.base_update(n9)',
        actor_states='observations', actor_Q_timing='pre_update_online',
        actor_updates_per_Q_update=1, evaluation_steps=list(STEPS))))
    if (out/'config.json').exists():
        old = json.loads((out/'config.json').read_text())
        assert old['identity'] == identity and old['source_sha256'] == sources
    else:
        atomic(out/'config.json', dict(identity=identity, source_sha256=sources))
        (out/'source').mkdir(exist_ok=True)
        for path in sources:
            p = Path(path)
            (out/'source'/('vendor_' + p.name if p.parent.name == 'vendor' else p.name)).write_bytes(p.read_bytes())
    if (out/'completed.json').exists():
        return
    started = time.monotonic()
    device = jax.devices('gpu')[0]
    env, raw, validation = ogbench.make_env_and_datasets(job['environment'])
    data = {k: np.asarray(raw[k], np.float32) for k in ('observations', 'actions', 'next_observations', 'rewards', 'masks')}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    bounds = core.return_bounds(job['environment'], .995, float(data['rewards'].min()), float(data['rewards'].max()))
    obs_dim, act_dim = data['observations'].shape[-1], data['actions'].shape[-1]
    atomic(out/'dataset.json', dict(size=len(data['actions']), observation_dim=obs_dim,
        action_dim=act_dim, return_bounds=bounds, action_clip_eps=1e-5, normalize_states=False))
    env.close(); del raw, validation
    specs = job['actors']
    state = core.create(job['seed'], jax.device_put(data['observations'][:1], device),
                        jax.device_put(data['actions'][:1], device), cfg, specs)
    metas = {name: core.actor_metadata(cfg, obs_dim, act_dim, **spec) for name, spec in specs.items()}
    rng = np.random.RandomState(job['seed'])
    step = 0
    latest = out/'latest.pkl'
    if latest.exists():
        with latest.open('rb') as stream:
            saved = pickle.load(stream)
        assert saved['identity'] == identity and saved['source_sha256'] == sources
        state = jax.device_put(flax.serialization.from_state_dict(state, saved['state']), device)
        rng.set_state(saved['replay_rng']); step = saved['step']
        path = out/'diagnostics.jsonl'
        if path.exists():
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            if any(r['step'] > step for r in lines):
                atomic(out/f'resume_tail_{time.time_ns()}.json', [r for r in lines if r['step'] > step])
                path.write_text(''.join(json.dumps(r)+'\n' for r in lines if r['step'] <= step))
        print('RESUME', step, flush=True)
    endpoint = args.stop_at or identity['steps']
    assert step <= endpoint <= identity['steps']
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    scan = core.make_scan(core.build_update(specs, bounds))

    @jax.jit
    def health(s):
        return jax.numpy.all(jax.numpy.stack([jax.numpy.all(jax.numpy.isfinite(x)) for x in jax.tree_util.tree_leaves(s)]))

    def save(export=False):
        assert int(state.base.q_updates) == int(state.actor_updates) == step and int(state.base.v_updates) == 0
        payload = dict(step=step, state=flax.serialization.to_state_dict(jax.device_get(state)),
            replay_rng=rng.get_state(), identity=identity, source_sha256=sources)
        temp = latest.with_suffix('.tmp')
        with temp.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(latest)
        if export:
            checkpoint = out/f'checkpoint_{step}.pkl'
            if checkpoint.exists():
                assert sha(checkpoint) == sha(latest)
            else:
                os.link(latest, checkpoint)
            base_hash = tree_digest(jax.device_get(state.base))
            for name in specs:
                artifact = core.export_actor(state, name, metas[name])
                artifact.update(step=step, identity=identity, name=name, base_state_sha256=base_hash,
                    checkpoint_sha256=sha(checkpoint), config_sha256=sha(out/'config.json'), source_sha256=sources)
                path = out/f'actor_{name}_{step}.pkl'
                with path.with_suffix('.tmp').open('wb') as stream:
                    pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
                path.with_suffix('.tmp').replace(path)
        append(out/'runtime_events.jsonl', dict(kind='checkpoint', step=step, time=time.time(), exported=export))

    first = True; sums = {}; count = 0; scan_seconds = 0.
    while step < endpoint and not stopping[0]:
        n = min(100, endpoint-step)
        indices = rng.randint(len(data['actions']), size=(n, cfg['batch_size']))
        batch = {k: jax.device_put(v[indices], device) for k, v in data.items()}
        began = time.monotonic()
        state, logs = scan(state, batch); sync((state, logs))
        elapsed = time.monotonic()-began
        step += n; count += n; scan_seconds += elapsed
        for k, v in jax.device_get(logs).items():
            sums[k] = sums.get(k, 0.) + float(np.asarray(v, np.float64).sum())
        if first:
            print('FIRST_SCAN', step, elapsed, flush=True)
            append(out/'runtime_events.jsonl', dict(kind='first_scan', step=step, seconds=elapsed, time=time.time()))
            first = False
        if step % 1000 == 0 or step == endpoint or stopping[0]:
            finite = bool(health(state)) and all(np.isfinite(v) for v in sums.values())
            record = dict(step=step, time=time.time(), finite=finite, actor_updates=int(state.actor_updates),
                q_updates=int(state.base.q_updates), v_updates=int(state.base.v_updates),
                metric_window_updates=count, scan_ms_per_update=1000*scan_seconds/count,
                **{k: v/count for k, v in sums.items()})
            if not finite:
                # Do not overwrite the last finite checkpoint.
                atomic(out/'failed.json', dict(reason='nonfinite', step=step, time=time.time()))
                raise FloatingPointError(f'Non-finite state at {step}')
            append(out/'diagnostics.jsonl', record)
            atomic(out/'runtime.json', dict(step=step, time=time.time(), phase='training',
                session_seconds=time.monotonic()-started, scan_ms_per_update=record['scan_ms_per_update']))
            sums = {}; count = 0; scan_seconds = 0.
            if step % 10000 == 0:
                print('TRAIN', step, 'MS', record['scan_ms_per_update'], flush=True)
        if step % 25000 == 0 or step == endpoint or stopping[0]:
            save(export=step in STEPS or step == endpoint)
    if stopping[0] and step < endpoint:
        if step % 25000 != 0:
            save()
        atomic(out/'interrupted.json', dict(step=step, time=time.time()))
        return
    complete = step == identity['steps']
    atomic(out/('completed.json' if complete else 'screened.json'), dict(step=step,
        q_updates=int(state.base.q_updates), actor_updates=int(state.actor_updates), v_updates=int(state.base.v_updates),
        time=time.time(), session_seconds=time.monotonic()-started, evaluation_status='pending_CPU_evaluation'))


if __name__ == '__main__':
    main()
