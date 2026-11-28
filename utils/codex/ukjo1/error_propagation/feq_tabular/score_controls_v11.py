"""Supplementary matched score/output controls, with fixed loss normalization.

The same numeric eta in a score equation and a function output is NOT an
invariant notion of equal error. Both that comparison and output-matched
negative controls are retained. No production algorithm is modified.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import brentq
from .matched_error_v11 import pick, save_json, save_csv
from .mdp import chain, expectile, solve, evaluate_policy
from .behavior_flow import offline_counts, fit


def forced_reference(m, channel, eta, lq=0.):
    """Backward solve of population IQL with a declared forcing channel."""
    q = np.zeros_like(m.reward)
    v = np.zeros(len(q))
    for s in range(len(q)-1, -1, -1):
        def q_at(x):
            nv = np.array([x if ns == s else v[ns] for ns in m.successor[s]])
            if channel == 'v_output':
                nv = nv - eta
            return (m.reward[s] + m.gamma*m.continuation[s]*nv)/(1+lq)
        def equation(x):
            d = q_at(x) - x - (eta if channel == 'q_output' else 0.)
            g = np.sum(m.behavior[s]*np.where(d >= 0, m.tau, 1-m.tau)*d)
            return g - (eta if channel == 'v_score' else 0.)
        v[s] = brentq(equation, -2000., 2000., xtol=1e-13)
        q[s] = q_at(v[s])
    return q, v


@njit(cache=True)
def train(reward, sn, cont, beta, tau, gamma, method, channel, eta,
          steps, seed, lq=0., lv=0., alpha=.1, log_every=10000):
    # method: 0=IQL9, 1=FEQ9, 2=canonical single-action IQL.
    # channel: 0=clean, 1=Q readout, 2=V score, 3=V backup readout.
    np.random.seed(7100011+seed)
    states, actions = reward.shape
    cdf = beta.copy()
    for s in range(states):
        for a in range(1, actions):
            cdf[s, a] += cdf[s, a-1]
    q = np.zeros_like(reward)
    target = q.copy()
    v = np.zeros(states)
    size = 1+(steps+log_every-1)//log_every
    ts = np.zeros(size, np.int64)
    qs = np.zeros((size, states, actions))
    vs = np.zeros((size, states))
    idx = 0
    counts = np.zeros(actions)
    for t in range(steps+1):
        if t == 0 or t % log_every == 0 or t == steps:
            ts[idx], qs[idx], vs[idx] = t, q, v
            idx += 1
        if t == steps:
            break
        s = np.random.randint(states)
        u = np.random.random(10)
        a = pick(cdf[s], u[0])
        ns = sn[s, a]
        if method == 1:
            counts[:] = 0
            for k in range(9):
                counts[pick(cdf[ns], u[k+1])] += 1
            boot = expectile(target[ns], counts, tau) - (eta if channel == 1 else 0.)
        else:
            boot = v[ns] - (eta if channel == 3 else 0.)
        old = q[s, a]
        q[s, a] += alpha*(reward[s, a]+gamma*cont[s, a]*boot-(1+lq)*old)
        if method != 1:
            grad = 0.
            n = 1 if method == 2 else 9
            for k in range(n):
                av = a if method == 2 else pick(cdf[s], u[k+1])
                d = target[s, av] - (eta if channel == 1 else 0.) - v[s]
                grad += (tau if d >= 0 else 1-tau)*d/n
            v[s] += alpha*(grad-lv*v[s]-(eta if channel == 2 else 0.))
        for ss in range(states):
            for aa in range(actions):
                online = old if ss == s and aa == a else q[ss, aa]
                target[ss, aa] = .995*target[ss, aa]+.005*online
    return ts, qs, vs


def worker(job):
    root, c = job
    key = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(root)/'raw'/key
    folder.mkdir(parents=True, exist_ok=True)
    m = chain(c['length'], .25, .999, .9, c['safe'])
    m.behavior = fit(offline_counts(m.behavior, c['seed']))
    t, q, v = train(m.reward, m.successor, m.continuation, m.behavior,
                    m.tau, m.gamma, c['method'], c['channel'], c['eta'],
                    c['steps'], c['seed'], c.get('lq', 0.), c.get('lv', 0.))
    qt, vt, _ = solve(m)
    tail = t >= .9*c['steps']
    scores = np.array([evaluate_policy(m, qq) for qq in q[tail]])
    row = {**c, 'return': float(scores[:, 0].mean()),
           'success': float(scores[:, 1].mean()),
           'q_rmse': float(np.sqrt(((q[tail]-qt)**2).mean((1, 2))).mean())}
    np.savez_compressed(folder/'data.npz', step=t, q=q, v=v, beta=m.behavior,
                        q_truth=qt, v_truth=vt)
    save_json(folder/'result.json', dict(config=c, metrics=row,
        data_sha256=hashlib.sha256((folder/'data.npz').read_bytes()).hexdigest()))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in range(2500, 2516):
        for variant, method, channel in (
                ('iql9_clean', 0, 0), ('iql9_q_output', 0, 1),
                ('iql9_v_score', 0, 2), ('iql9_v_output', 0, 3),
                ('feq9_clean', 1, 0), ('feq9_q_output', 1, 1)):
            jobs.append(dict(suite='score', variant=variant, length=8, safe=.9,
                             seed=seed, method=method, channel=channel, eta=.005,
                             steps=2000000))
        for lv in (0., .003):
            jobs.append(dict(suite='canonical_ridge', variant=f'iql1_lv{lv}',
                             length=8, safe=.9, seed=seed, method=2, channel=0,
                             eta=0., lq=.003, lv=lv, steps=2000000))
    save_json(a.root/'registration.json', dict(jobs=jobs,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='eta=.005 selected after exact exploratory sweep; fresh seeds 2500:2515'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(worker, [(str(a.root), j) for j in jobs]))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    save_csv(a.root/'per_seed.csv', [{k: r.get(k, '') for k in keys} for r in rows])
    print(json.dumps(dict(status='complete', trajectories=len(rows))))


if __name__ == '__main__':
    main()
