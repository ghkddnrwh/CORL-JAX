"""Same-state/same-noise ranking drift for the frozen component crossover."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from utils.flow_iql_research import sample_budget_core_v1 as core
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, read_source, sha


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64))**2))


def ranks(q):
    return np.argsort(np.argsort(q, axis=0), axis=0).astype(np.float64)


def rank_statistics(qa, qb, bank):
    ia, ib = qa.argmax(0), qb.argmax(0)
    ra, rb = ranks(qa), ranks(qb)
    ra -= ra.mean(0)
    rb -= rb.mean(0)
    correlation = np.sum(ra * rb, axis=0) / np.sqrt(np.sum(ra**2, axis=0) * np.sum(rb**2, axis=0))
    gap_a = np.sort(qa, axis=0)[-1] - np.sort(qa, axis=0)[-2]
    gap_b = np.sort(qb, axis=0)[-1] - np.sort(qb, axis=0)[-2]
    delta = qb - qa
    centered = delta - delta.mean(0, keepdims=True)
    states = np.arange(qa.shape[1])
    return dict(argmax_flip_fraction=float(np.mean(ia != ib)),
        mean_spearman_across_nine_actions=float(correlation.mean()),
        q_change_mean=float(delta.mean()),
        q_change_action_centered_rms=float(np.sqrt(np.mean(centered**2))),
        A_top_second_margin_median=float(np.median(gap_a)),
        B_top_second_margin_median=float(np.median(gap_b)),
        selected_action_change_MSE=mse(bank[ia, states], bank[ib, states]))


def run(args):
    assert all(d.platform == 'cpu' for d in jax.devices())
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    directory = args.plan.parent / f'rankings_{args.states}_v1'
    directory.mkdir(exist_ok=True)
    sources = {}
    for name, spec in plan['sources'].items():
        conf, state = read_source(spec)
        sources[name] = state
    replay = {}
    if args.states == 'replay':
        for env in sorted({s['environment'] for s in plan['sources'].values()}):
            instance, data, validation = ogbench.make_env_and_datasets(env)
            instance.close()
            indices = np.random.default_rng(20261118).choice(len(data['observations']), 512, replace=False)
            replay[env] = (np.asarray(data['observations'][indices], np.float32), indices)
            del instance, data, validation
    results = []
    for pair in plan['pairs']:
        a, b = pair['A'], pair['B']
        path = directory / f'{pair["id"]}.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        sa, sb = sources[a], sources[b]
        env = plan['sources'][a]['environment']
        evidence = {}
        if args.states == 'replay':
            obs, indices = replay[env]
            coordinates = indices[:, None]
        else:
            pools, labels = [], []
            for label in (a, b):
                raw = args.plan.parent / 'evaluations' / f'self_{label}.npz'
                result_path = raw.with_suffix('.json')
                value = json.loads(result_path.read_text())
                assert sha(raw) == value['trajectory_arrays_sha256']
                with np.load(raw) as arrays:
                    pool = arrays['observations']
                    chosen = np.random.default_rng(20261118).choice(len(pool), min(512, len(pool)), replace=False)
                    pools.append(pool[chosen])
                    labels.append(np.c_[np.full(len(chosen), len(labels)), arrays['episode_step'][chosen]])
                evidence[str(raw)] = sha(raw)
            obs, coordinates = np.concatenate(pools), np.concatenate(labels)
        observations = jnp.asarray(obs)
        action_dim = sa.base.agent.config['action_dim']
        noise = jax.random.normal(jax.random.PRNGKey(74191), (9, len(obs), action_dim))
        broad = jnp.broadcast_to(observations, (9, *observations.shape))
        definition = sa.base.agent.network

        @jax.jit
        def generate(params):
            agent = sa.base.agent.replace(network=definition.replace(params=params))
            return core.lean.flow(agent, broad, noise, 10)

        @jax.jit
        def score(params, actions):
            return definition.replace(params=params).select('critic')(broad, actions=actions)

        ba = np.asarray(generate(sa.base.agent.network.params))
        bb = np.asarray(generate(sb.base.agent.network.params))
        heads = {}
        for qname, state in ((a, sa), (b, sb)):
            for fname, bank in ((a, ba), (b, bb)):
                heads[qname, fname] = np.asarray(score(state.base.agent.network.params, jnp.asarray(bank)))
        stats = {}
        for fname, bank in ((a, ba), (b, bb)):
            ha, hb = heads[a, fname], heads[b, fname]
            qa, qb = ha.min(0), hb.min(0)
            stats[fname] = rank_statistics(qa, qb, bank)
            for name, h, q in ((a, ha, qa), (b, hb, qb)):
                stats[fname][name] = dict(q_mean=float(q.mean()),
                    within_state_action_Q_std_mean=float(q.std(0).mean()),
                    twin_head_disagreement_rms=float(np.sqrt(np.mean((h[0] - h[1])**2))),
                    twin_head_argmax_agreement=float(np.mean(h[0].argmax(0) == h[1].argmax(0))))
        qa, qb = heads[a, a].min(0), heads[b, b].min(0)
        states = np.arange(len(obs))
        selected_a, selected_b = ba[qa.argmax(0), states], bb[qb.argmax(0), states]
        raw_path = path.with_suffix('.npz')
        np.savez_compressed(raw_path, observations=obs, state_coordinates=coordinates,
            common_noise=np.asarray(noise), flow_A_actions=ba, flow_B_actions=bb,
            Q_A_flow_A=heads[a, a], Q_A_flow_B=heads[a, b],
            Q_B_flow_A=heads[b, a], Q_B_flow_B=heads[b, b])
        result = dict(pair=pair, state_source=args.states, state_count=len(obs),
            same_states_and_gaussian_noises=True, noise_seed=74191, candidates=9,
            flow_same_noise_action_MSE=mse(ba, bb),
            actual_policy_selected_action_MSE=mse(selected_a, selected_b),
            fixed_bank_critic_comparisons=stats, trajectory_evidence=evidence,
            arrays_sha256=sha(raw_path), source_sha256=sha(__file__),
            frozen_evaluator_source_sha256=sha(Path(core.__file__)), timestamp_unix=time.time(),
            optimizer_updates=0, limit='Ranking drift is descriptive. It does not identify true action values '
                'or prove that a flipped ranking is incorrect. Return attribution requires crossover rollouts.')
        atomic(path, result)
        results.append(result)
        print(json.dumps(dict(pair=pair['id'], states=args.states,
            old_bank_flip=stats[a]['argmax_flip_fraction'], new_bank_flip=stats[b]['argmax_flip_fraction'],
            flow_MSE=result['flow_same_noise_action_MSE'])), flush=True)
    atomic(directory / 'summary_v1.json', dict(results=results, plan_sha256=sha(args.plan),
        source_sha256=sha(__file__), new_training_runs=0, new_rollouts=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--states', choices=('replay', 'visitation'), required=True)
    run(parser.parse_args())
