"""Discrete oracle checks of the best-of-N/expectile comparison, not RL scores."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def expectile(x, p, tau):
    left, right = float(x.min()), float(x.max())
    for _ in range(90):
        v = (left+right)/2
        score = np.sum(p*np.where(x >= v, tau, 1-tau)*(x-v))
        if score > 0:
            left = v
        else:
            right = v
    return (left+right)/2


def max_mean(x, p, n):
    order = np.argsort(x); x, p = x[order], p[order]
    cdf = np.cumsum(p); cdf[-1] = 1.
    mass = np.diff(np.r_[0., cdf**n])
    return float(x@mass)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    rng = np.random.RandomState(91837); rows = []
    for tau, n in ((.5, 1), (.75, 3), (.9, 9), (.95, 19), (.99, 99)):
        differences = []
        for _ in range(2000):
            k = rng.randint(2, 65)
            x = rng.normal(size=k)*10**rng.uniform(-2, 2)
            p = rng.dirichlet(np.full(k, .3))
            diff = max_mean(x, p, n)-expectile(x, p, tau)
            assert diff >= -2e-10, (tau, diff)
            differences.append(diff)
        rows.append(dict(expectile=tau, proposals=n, distributions=2000,
                         minimum_estimated_value_gap=min(differences)))
    rare = []
    for tau, n in ((.9, 9), (.99, 9), (.99, 99)):
        p = np.array([.99, .01]); x = np.array([-1., 0.])
        rare.append(dict(expectile=tau, proposals=n, exact_expectile=expectile(x,p,tau),
            exact_expected_selected_Q=max_mean(x,p,n),
            ideal_implicit_good_probability=tau*.01/(tau*.01+(1-tau)*.99),
            best_N_good_probability=1-.99**n))
    sharpness = []
    for tau, n in ((.75, 2), (.9, 8), (.95, 18), (.99, 98)):
        p = np.array([1-1e-7, 1e-7]); x = np.array([-1., 0.])
        gap = max_mean(x,p,n)-expectile(x,p,tau)
        assert gap < 0
        sharpness.append(dict(expectile=tau, proposals=n, gap=gap))
    result = dict(time=time.time(), status='complete', oracle_checks=rows,
        rare_action_example=rare, below_threshold_counterexamples=sharpness,
        proof='v=e_tau(X), r=tau/(1-tau), A=E(X-v)+, B=E(v-X)+=rA. '
              'Let p=P(X<v). For M=max of N IID draws: '
              'E(M-v)+ >= N*p**(N-1)*A, E(v-M)+ <= p**(N-1)*B. '
              'Thus E M-v >= (N-r)*p**(N-1)*A >=0 when N>=r. '
              'For p=0 or p=1 the inequalities remain valid with N=1 treated separately. '
              'Integrals ignore CDF point discontinuities, so atoms at v do not invalidate it.',
        assumptions=['IID scalar draws, finite first absolute moment, tau>=.5.',
            'The implicit actor identity requires the exact expectile under the proposal law.',
            'This does not guarantee true environment improvement under learned Q/V/behavior.',
            'More candidates may amplify critic error; runtime cost grows.',
            'The derivation is posthoc relative to old N9 experiments; no novelty claim.'],
        goal_complete=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='complete',discrete_checks=10000,examples=rare)),flush=True)


if __name__ == '__main__':
    main()
