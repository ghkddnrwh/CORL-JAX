"""Resumable research runner; primary score is fixed 800K/900K/1M CPU50 mean."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

from utils.flow_iql_research import rank_flow_core_v1 as core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.reward_bounds import return_bounds
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_pickle(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record, allow_nan=False)+'\n')


def make_policy(agent):
    # Pass the trainable arrays as arguments; keep the definition/config static.
    def policy(params, observation, key):
        a = agent.replace(network=agent.network.replace(params=params))
        return a.sample_actions(observations=observation, seed=key, temperature=0.)
    return jax.jit(policy, backend='cpu')


def evaluate(state, env, policy, step, episodes, seed=42):
    params = jax.device_put(state.agent.network.params, jax.devices('cpu')[0])
    rewards, successes, lengths = [], [], []
    start = time.monotonic()
    # Fixed independent episode seeds also control navigation's global np RNG.
    for episode in range(episodes):
        es = seed+episode
        np.random.seed(es)
        random.seed(es)
        env.action_space.seed(es)
        obs, _ = env.reset(seed=es)
        key = jax.device_put(jax.random.PRNGKey(es), jax.devices('cpu')[0])
        reward_sum, success, steps = 0., 0., 0
        while True:
            key, action_key = jax.random.split(key)
            act = np.asarray(policy(params, np.asarray(obs, np.float32), action_key)).copy()
            obs, reward, terminated, truncated, info = env.step(act)
            reward_sum += reward
            success = max(success, float(info.get('success', 0.)))
            steps += 1
            if terminated or truncated:
                break
        rewards.append(float(reward_sum));successes.append(success);lengths.append(steps)
    return dict(step=step, success_rate=float(np.mean(successes)),
                reward_mean=float(np.mean(rewards)), episode_successes=successes,
                episode_rewards=rewards, episode_lengths=lengths, n_episodes=episodes,
                eval_device='cpu', protocol='last3_cpu50_independent_seeded_episodes_v1',
                first_episode_seed=seed, seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=core.ENVIRONMENTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-device', choices=('gpu','cpu'), default='gpu')
    parser.add_argument('--profile', choices=core.PROFILES, default='bounded_v2_rank_flow')
    parser.add_argument('--kind', choices=('rank_flow',), default='rank_flow')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expectile', type=float, default=.9)
    parser.add_argument('--discount', type=float)
    parser.add_argument('--width', type=int)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--aux-eval-steps', default='100000,300000',
                        help='Intermediate policy evaluations; primary800K/900K/1M always run.')
    parser.add_argument('--probe-every', type=int, default=1000)
    args = parser.parse_args()
    auxiliary_steps={int(s) for s in args.aux_eval_steps.split(',') if s}
    assert all(s > 0 for s in auxiliary_steps)
    evaluation_steps={800000,900000,1000000} | auxiliary_steps
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    cfg = core.config_for(args.env, args.discount, args.expectile, args.width, args.alpha)
    assert args.samples in (None, 1), 'Rank-flow deployment uses one policy-flow sample'
    cfg.update(num_samples=1, flow_steps=10, agent_name='rank_flow', policy_type='rank_flow')
    identity = dict(environment=args.env, config=cfg, profile=args.profile, kind=args.kind,
                    seed=args.seed, steps=args.steps, chunk=args.chunk, episodes=args.episodes,
                    training_device=args.train_device,
                    evaluation_steps=sorted(evaluation_steps), probe_every=args.probe_every,
                    online_metrics='all_updates_in_each_probe_window_v2',
                    evaluation='last3_cpu50_independent_seeded_episodes_v1')
    source_paths = [Path(core.__file__), Path(__file__), Path('utils/propagation_research/seeded_reset.py'),
                    Path('utils/propagation_research/reward_bounds.py')]
    source_paths.append(Path(core.__file__).with_name('core.py'))
    source_paths.extend(sorted(Path(core.__file__).parent.joinpath('vendor').glob('*.py')))
    sources = {str(p): sha(p) for p in source_paths}
    config_path = out/'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        assert previous['identity'] == json.loads(json.dumps(identity)), 'Run identity changed'
        assert previous['source_sha256'] == sources, 'Source changed: create a new versioned run'
    else:
        config_path.write_text(json.dumps(dict(identity=identity, source_sha256=sources,
                                               arguments=vars(args) | {'output':str(out)}), default=str, indent=2))
        frozen = out/'source';frozen.mkdir(exist_ok=True)
        for p in source_paths:
            name = p.name if p.parent.name != 'vendor' else 'vendor_'+p.name
            (frozen/name).write_bytes(p.read_bytes())
    if (out/'completed.json').exists():
        return
    gpu = jax.devices(args.train_device)[0]
    env, raw, _ = ogbench.make_env_and_datasets(args.env)
    reset_metadata = prepare_seeded_action_space(env, 42)
    fields = ('observations','actions','next_observations','rewards','masks')
    data = {k:np.asarray(raw[k], np.float32) for k in fields}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    bounds = return_bounds(args.env, cfg['discount'], float(data['rewards'].min()),
                           float(data['rewards'].max()), 'ogbench')
    (out/'dataset.json').write_text(json.dumps(dict(size=len(data['actions']),
        reward_min=float(data['rewards'].min()), reward_max=float(data['rewards'].max()),
        return_lower=bounds[0], return_upper=bounds[1], normalize_states=False,
        action_clip_eps=1e-5, reset_metadata=reset_metadata), indent=2))
    obs = jax.device_put(data['observations'][:1], gpu)
    act = jax.device_put(data['actions'][:1], gpu)
    state = core.create(args.seed, obs, act, cfg, args.kind)
    rng = np.random.RandomState(args.seed)
    step = 0
    latest = out/'latest.pkl'
    if latest.exists():
        with latest.open('rb') as f: saved = pickle.load(f)
        state = flax.serialization.from_state_dict(state, saved['state'])
        state = jax.device_put(state, gpu)
        rng.set_state(saved['replay_rng']);step = saved['step']
        for name in ('diagnostics.jsonl', 'evaluations.jsonl'):
            path = out/name
            if path.exists():
                records=[json.loads(x) for x in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(x)+'\n' for x in records if x['step']<=step))
        print('RESUME',step,flush=True)
    scan = core.make_scan(core.build_update(args.profile, bounds, args.kind))
    policy = make_policy(state.agent)
    prng = np.random.RandomState(args.seed+19401)
    idx = prng.randint(len(data['actions']),size=1024)
    fixed = {k:jax.device_put(v[idx],gpu) for k,v in data.items()}
    evaluated = set()
    if (out/'evaluations.jsonl').exists():
        evaluated = {json.loads(x)['step'] for x in (out/'evaluations.jsonl').read_text().splitlines()}
    start=time.monotonic();endpoint=args.stop_at or args.steps
    log_sums=None;log_steps=0
    @jax.jit
    def add_logs(total,logs):
        return {k:total[k]+jnp.sum(v) for k,v in logs.items()}
    def save():
        payload=dict(step=step,state=flax.serialization.to_state_dict(jax.device_get(state)),
                     replay_rng=rng.get_state(), identity=identity, source_sha256=sources)
        atomic_pickle(latest,payload)
        if step in (100000,300000,800000,900000,1000000) or step == endpoint:
            target=out/f'checkpoint_{step}.pkl'
            if not target.exists(): os.link(latest,target)
    def do_eval():
        record=evaluate(state,env,policy,step,args.episodes)
        append(out/'evaluations.jsonl',record);evaluated.add(step)
        print(json.dumps({k:v for k,v in record.items() if not isinstance(v,list)}),flush=True)
    if step and step in evaluation_steps and step not in evaluated:
        do_eval()
    while step < endpoint:
        n=min(args.chunk,endpoint-step)
        indices=rng.randint(len(data['actions']),size=(n,cfg['batch_size']))
        batches={k:jax.device_put(v[indices],gpu) for k,v in data.items()}
        state,logs=scan(state,batches);step+=n
        if log_sums is None:
            log_sums={k:jnp.zeros((),v.dtype) for k,v in logs.items()}
        log_sums=add_logs(log_sums,logs);log_steps+=n
        if step % args.probe_every == 0 or step == endpoint:
            vals=jax.device_get(core.probe(state,fixed))
            probe_alpha=cfg['flow_temperature'] if args.kind=='awrflow' else cfg['alpha'] if args.kind=='iql' else None
            stats=core.probe_summary(vals,cfg['expectile'],bounds,probe_alpha)
            metrics={k:float(np.asarray(v))/log_steps for k,v in jax.device_get(log_sums).items()}
            stats['metric_window_updates']=log_steps
            log_sums={k:jnp.zeros((),v.dtype) for k,v in logs.items()};log_steps=0
            stats.update(step=step,elapsed_seconds=time.monotonic()-start,
                         q_updates=int(state.q_updates),v_updates=int(state.v_updates),**metrics)
            # Nonfinite state is recorded explicitly; never assigned a zero score.
            if not stats['finite']:
                save();(out/'stopped.json').write_text(json.dumps(dict(reason='nonfinite_state',step=step)))
                print('NONFINITE',step,flush=True);return
            append(out/'diagnostics.jsonl',stats)
            if step % 10000 == 0: print('TRAIN',step,'Q_RMS',stats['q_rms'],flush=True)
        if step % 100000 == 0 or step == endpoint:
            save()
        if step in evaluation_steps and step not in evaluated:
            do_eval()
    completion=dict(step=step,kind='complete' if step==args.steps else 'screened',
                    q_updates=int(state.q_updates),v_updates=int(state.v_updates))
    name='completed.json' if step==args.steps else 'screened.json'
    (out/name).write_text(json.dumps(completion,indent=2))


if __name__ == '__main__':
    main()
