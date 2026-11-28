"""Paired tabular backups with a shared Q-error field and NO external V error.

All methods see the same replay, action uniforms and entire Q-error field.
The field is fixed within a step: repeated actions never resample Q noise.
Action-common errors are a controlled approximation-error mode, not a claim
that exact tabular representations intrinsically have readout noise.
"""
import numpy as np
from numba import njit
from .mdp import expectile

METHODS = ('iql', 'iql_batch9', 'iql_full', 'feq9', 'population', 'repair_half',
           'repair_zero', 'iql_full9')
PARTS = ('prior_q', 'q_readout', 'v_residual', 'finite_sampling', 'model', 'total')


@njit(cache=True)
def pick(cdf, u):
    for a in range(len(cdf)):
        if u < cdf[a]:
            return a
    return len(cdf)-1


@njit(cache=True)
def simulate(method, reward, successor, continuation, behavior, model, gamma, tau,
             updates, seed, q_truth, v_truth, sigma=0., common=1., rho=0.,
             alpha_q=.1, alpha_v=.1, ema=.005, log_every=5000,
             frozen=False, init_truth=False, anneal=False):
    states, actions = reward.shape
    np.random.seed(900001+seed)
    visited = np.random.randint(0, states, updates)
    uniforms = np.random.random((updates, 10))
    # Seeded independently of replay and algorithm; generate lazily in the loop.
    np.random.seed(1900003+seed)
    latent = np.random.standard_normal((states, actions+1))
    field = np.zeros_like(reward)
    q = q_truth.copy() if init_truth else np.zeros_like(reward)
    target = q.copy()
    value = v_truth.copy() if init_truth else np.zeros(states)
    cdf = behavior.copy()
    mcdf = model.copy()
    for s in range(states):
        for a in range(1, actions):
            cdf[s,a] += cdf[s,a-1]
            mcdf[s,a] += mcdf[s,a-1]
    count = 1+(updates+log_every-1)//log_every
    times = np.zeros(count, np.int64)
    qs = np.zeros((count, states, actions))
    vs = np.zeros((count, states))
    targets = np.zeros_like(qs)
    # Per-window signed mean and second moment, retaining cross terms.
    parts = np.zeros((count, len(PARTS), 2))
    cross = np.zeros((count, len(PARTS)-1, len(PARTS)-1))
    sums = np.zeros(len(PARTS))
    squares = np.zeros(len(PARTS))
    products = np.zeros((len(PARTS)-1, len(PARTS)-1))
    root_residual = np.zeros((count, states))
    noise_statistics = np.zeros((count, 2))
    nsum = nsquare = 0.
    window = idx = 0
    counts = np.zeros(actions)
    components = np.zeros(len(PARTS))
    worst_identity = 0.
    for t in range(updates+1):
        if t == 0 or t % log_every == 0 or t == updates:
            if t:
                idx += 1
                parts[idx,:,0] = sums/window
                parts[idx,:,1] = squares/window
                cross[idx] = products/window
                noise_statistics[idx,0] = nsum/(window*states*actions)
                noise_statistics[idx,1] = nsquare/(window*states*actions)
                sums[:] = 0.
                squares[:] = 0.
                products[:] = 0.
                nsum = nsquare = 0.
                window = 0
            times[idx] = t
            qs[idx] = q
            vs[idx] = value
            targets[idx] = target
            for s in range(states):
                root_residual[idx,s] = value[s]-expectile(target[s], behavior[s], tau)
        if t == updates:
            break
        if t:
            latent = rho*latent+np.sqrt(1-rho*rho)*np.random.standard_normal(latent.shape)
        for ss in range(states):
            for aa in range(actions):
                field[ss,aa] = sigma*(np.sqrt(common)*latent[ss,0]+np.sqrt(1-common)*latent[ss,aa+1])
                nsum += field[ss,aa]
                nsquare += field[ss,aa]**2
        s = visited[t]
        a = pick(cdf[s], uniforms[t,0])
        sn = successor[s,a]
        observed_next = target[sn]+field[sn]
        observed_current = target[s]+field[s]
        clean = expectile(target[sn], behavior[sn], tau)
        observed = expectile(observed_next, behavior[sn], tau)
        model_root = expectile(observed_next, model[sn], tau)
        components[:] = 0.
        components[0] = clean-v_truth[sn]
        components[1] = observed-clean
        if method in (0,1,2,7):
            bootstrap = value[sn]
            components[2] = bootstrap-observed
        elif method in (5,6):
            k = .5 if method == 5 else 0.
            bootstrap = observed+k*(value[sn]-observed)
            components[2] = k*(value[sn]-observed)
        elif method == 4:
            bootstrap = model_root
            components[4] = model_root-observed
        else:
            counts[:] = 0.
            for k in range(9):
                counts[pick(mcdf[sn],uniforms[t,k+1])] += 1.
            bootstrap = expectile(observed_next, counts, tau)
            components[3] = bootstrap-model_root
            components[4] = model_root-observed
        y = reward[s,a]+gamma*continuation[s,a]*bootstrap
        components[:5] *= gamma*continuation[s,a]
        components[5] = y-q_truth[s,a]
        worst_identity = max(worst_identity, abs(components[:5].sum()-components[5]))
        sums += components
        squares += components**2
        for u in range(5):
            for v in range(5):
                products[u,v] += components[u]*components[v]
        window += 1
        old = q[s,a]
        schedule = 1. if not anneal else 1./np.sqrt(1.+t/10000.)
        if not frozen:
            q[s,a] += alpha_q*schedule*(y-q[s,a])
        if method in (0,5,6):
            delta = observed_current[a]-value[s]
            value[s] += 2*alpha_v*schedule*(tau if delta >= 0 else 1-tau)*delta
        elif method == 1:
            gradient = 0.
            for k in range(9):
                av = pick(cdf[s], uniforms[t,k+1])
                delta = observed_current[av]-value[s]
                gradient += (tau if delta >= 0 else 1-tau)*delta/9.
            value[s] += 2*alpha_v*schedule*gradient
        elif method in (2,7):
            for k in range(9 if method == 7 else 1):
                gradient = 0.
                for av in range(actions):
                    delta = observed_current[av]-value[s]
                    gradient += behavior[s,av]*(tau if delta >= 0 else 1-tau)*delta
                value[s] += 2*alpha_v*schedule*gradient
        # Old-online target EMA, matching the existing FEQ implementation.
        for ss in range(states):
            for aa in range(actions):
                online = old if (ss == s and aa == a) else q[ss,aa]
                target[ss,aa] = (1-ema)*target[ss,aa]+ema*online
    return times, qs, vs, targets, parts, cross, root_residual, noise_statistics, worst_identity
