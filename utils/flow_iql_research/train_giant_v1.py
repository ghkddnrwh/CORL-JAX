"""Same last3CPU50 protocol for IQL, FQL and finite-mean backup on Giant."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import giant_core_v1 as core
from utils.flow_iql_research.train_mean_empirical_backup_v1 import make_policy, evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import numpy as np
import ogbench


STEPS = (800000, 900000, 1000000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def identity_for(args):
    cfg = core.config_for(args.env, args.kind, args.discount)
    return dict(environment=args.env, config=cfg, kind=args.kind,
        agent_kind=core.agent_kind(args.kind), seed=args.seed, steps=args.steps,
        chunk=args.chunk, episodes=args.episodes, training_device=args.train_device,
        evaluation_steps=list(STEPS), probe_every=args.probe_every,
        online_metrics='all_updates_in_each_probe_window_v2',
        evaluation='last3_cpu50_independent_seeded_episodes_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=(core.ENVIRONMENT,), default=core.ENVIRONMENT)
    parser.add_argument('--kind', choices=core.KINDS, required=True)
    parser.add_argument('--discount', type=float, choices=(.99, .999), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--probe-every', type=int, default=1000)
    parser.add_argument('--train-device', choices=('cpu','gpu'), default='gpu')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = identity_for(args)
    cfg = identity['config']
    sources = {str(p): sha(p) for p in core.source_paths()+[Path(__file__).resolve()]}
    conf = dict(identity=identity, source_sha256=sources, arguments=vars(args)|{'output':str(out)})
    conf['arguments'] = {k:str(v) if isinstance(v,Path) else v for k,v in conf['arguments'].items()}
    if (out/'config.json').exists():
        previous = json.loads((out/'config.json').read_text())
        assert previous['identity'] == json.loads(json.dumps(identity)) and previous['source_sha256'] == sources
    else:
        atomic_json(out/'config.json', conf)
        frozen = out/'source'
        frozen.mkdir()
        for name in sources:
            path = Path(name)
            label = 'vendor_'+path.name if path.parent.name == 'vendor' else path.name
            (frozen/label).write_bytes(path.read_bytes())
    if (out/'completed.json').exists():
        return
    device = jax.devices(args.train_device)[0]
    env, raw, validation = ogbench.make_env_and_datasets(args.env)
    reset = prepare_seeded_action_space(env, 42)
    fields = ('observations','actions','next_observations','rewards','masks')
    data = {key:np.asarray(raw[key], np.float32) for key in fields}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    interval = core.bounds(args.env, args.discount, float(data['rewards'].min()), float(data['rewards'].max()))
    atomic_json(out/'dataset.json', dict(size=len(data['actions']), reward_min=float(data['rewards'].min()),
        reward_max=float(data['rewards'].max()), return_lower=interval[0], return_upper=interval[1],
        normalize_states=False, action_clip_eps=1e-5, reset_metadata=reset,
        observation_dim=data['observations'].shape[-1], action_dim=data['actions'].shape[-1]))
    del raw, validation
    state = core.create(args.seed, jax.device_put(data['observations'][:1],device),
        jax.device_put(data['actions'][:1],device), cfg, args.kind)
    rng = np.random.RandomState(args.seed)
    latest = out/'latest.pkl'
    step = 0
    if latest.exists():
        with latest.open('rb') as stream:
            saved = pickle.load(stream)
        assert saved['identity'] == json.loads(json.dumps(identity)) and saved['source_sha256'] == sources
        state = jax.device_put(flax.serialization.from_state_dict(state, saved['state']), device)
        step = saved['step']
        rng.set_state(saved['replay_rng'])
        for name in ('diagnostics.jsonl','evaluations.jsonl'):
            path = out/name
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            tail = [row for row in rows if row['step'] > step]
            if tail:
                atomic_json(out/(name+'.resume_tail_'+str(time.time_ns())+'.json'), tail)
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step'] <= step))
        print('RESUME',step,flush=True)
    scan = core.make_scan(core.build_update(args.kind, interval))
    policy = make_policy(state.agent)
    probe_rng = np.random.RandomState(args.seed+19401)
    indices = probe_rng.randint(len(data['actions']),size=1024)
    fixed = {key:jax.device_put(value[indices],device) for key,value in data.items()}
    evaluated = {json.loads(line)['step'] for line in (out/'evaluations.jsonl').read_text().splitlines()} if (out/'evaluations.jsonl').exists() else set()
    endpoint = args.stop_at or args.steps
    start = time.monotonic()
    sums, count = None, 0

    @jax.jit
    def add_logs(total, logs):
        return {key:total[key]+value.sum() for key,value in logs.items()}

    def save():
        payload = dict(step=step, state=flax.serialization.to_state_dict(jax.device_get(state)),
            replay_rng=rng.get_state(), identity=json.loads(json.dumps(identity)), source_sha256=sources)
        temp = latest.with_suffix('.tmp')
        with temp.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp, latest)
        if step in (100000,300000,*STEPS) or step == endpoint:
            target = out/f'checkpoint_{step}.pkl'
            assert not target.exists() or sha(target) == sha(latest)
            if not target.exists():
                os.link(latest, target)

    def do_eval():
        print('EVALUATE_START',step,flush=True)
        result = evaluate(state, env, policy, step, args.episodes)
        append(out/'evaluations.jsonl',result)
        evaluated.add(step)
        print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)

    if step in STEPS and step not in evaluated:
        do_eval()
    while step < endpoint:
        n = min(args.chunk,endpoint-step)
        indices = rng.randint(len(data['actions']),size=(n,cfg['batch_size']))
        batches = {key:jax.device_put(value[indices],device) for key,value in data.items()}
        state, logs = scan(state,batches)
        step += n
        if sums is None:
            sums = {key:np.float32(0) for key in logs}
        sums = add_logs(sums,logs)
        count += n
        if step % args.probe_every == 0 or step == endpoint:
            stats = core.summarize(jax.device_get(core.probe(state,fixed,args.kind)),cfg,interval,args.kind)
            stats.update({key:float(value)/count for key,value in jax.device_get(sums).items()})
            stats.update(step=step, elapsed_seconds=time.monotonic()-start, metric_window_updates=count,
                q_updates=int(state.q_updates),v_updates=int(state.v_updates))
            assert stats['q_updates'] == step and stats['v_updates'] == core.V_UPDATES[args.kind]*step
            if not stats['finite']:
                save()
                atomic_json(out/'stopped.json',dict(reason='nonfinite_state',step=step))
                print('NONFINITE',step,flush=True)
                return
            append(out/'diagnostics.jsonl',stats)
            sums, count = {key:np.float32(0) for key in logs}, 0
            if step % 10000 == 0:
                print('TRAIN',step,'Q_RMS',stats['q_rms'],flush=True)
        if step % 100000 == 0 or step == endpoint:
            save()
        if step in STEPS and step not in evaluated:
            do_eval()
    complete = step == args.steps
    atomic_json(out/('completed.json' if complete else 'screened.json'),dict(kind='complete' if complete else 'screened',
        step=step,q_updates=int(state.q_updates),v_updates=int(state.v_updates)))
    env.close()


if __name__ == '__main__':
    main()
