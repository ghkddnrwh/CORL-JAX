"""Known episodic MDPs and exact expectile-Bellman reference solutions."""
from dataclasses import dataclass
from math import factorial
import numpy as np
from numba import njit


@dataclass
class MDP:
    reward: np.ndarray
    successor: np.ndarray
    continuation: np.ndarray
    behavior: np.ndarray
    gamma: float
    tau: float
    name: str


@njit(cache=True)
def expectile(values, probabilities, tau):
    order = np.argsort(values)
    total = 0.
    weighted = 0.
    for j in range(len(values)):
        total += probabilities[j]
        weighted += probabilities[j] * values[j]
    if total <= 0:
        raise ValueError('Probability mass must be positive')
    below_p = below_q = 0.
    for k in range(len(values) + 1):
        den = (1-tau)*below_p + tau*(total-below_p)
        root = ((1-tau)*below_q + tau*(weighted-below_q))/den
        lower = values[order[k-1]] if k else -np.inf
        upper = values[order[k]] if k < len(values) else np.inf
        if lower - 1e-12 <= root <= upper + 1e-12:
            return root
        if k < len(values):
            j = order[k]
            below_p += probabilities[j]
            below_q += probabilities[j]*values[j]
    raise ValueError('Expectile root not bracketed')


@njit(cache=True)
def roots(q, behavior, tau):
    return np.array([expectile(q[s], behavior[s], tau) for s in range(len(q))])


def chain(length=12, progress=.25, gamma=.995, tau=.9, safe_fraction=.8):
    """A safe exit competes with a delayed goal; non-progress actions wait.

    Every state/action is supported. Rewards/transitions are deterministic.
    The state-0 safe reward is a declared fraction of the population-expectile
    continuation value of the chain, not tuned using experimental outcomes.
    """
    states, actions = length+1, 4
    reward = np.zeros((states, actions))
    successor = np.repeat(np.arange(states)[:, None], actions, axis=1)
    continuation = np.ones((states, actions))
    behavior = np.full((states, actions), (1-progress)/3.)
    behavior[:, 0] = progress
    for s in range(states):
        successor[s, 0] = min(s+1, states-1)
    reward[-1, 0] = 1.
    continuation[-1, 0] = 0.
    effective = tau*progress/(tau*progress+(1-tau)*(1-progress))
    ratio = gamma*effective/(1-gamma*(1-effective))
    v_last = effective/(1-gamma*(1-effective))
    path_q = gamma * ratio**(length-1) * v_last
    reward[0, 1] = safe_fraction*path_q
    continuation[0, 1] = 0.
    return MDP(reward, successor, continuation, behavior, gamma, tau,
               f'chain_L{length}_p{progress:g}_g{gamma:g}_t{tau:g}_safe{safe_fraction:g}')


def multinomial_counts(n, actions):
    values = []
    def recurse(prefix, remaining):
        if len(prefix) == actions-1:
            values.append(prefix + [remaining]); return
        for k in range(remaining+1): recurse(prefix+[k], remaining-k)
    recurse([], n)
    counts = np.asarray(values, dtype=float)
    coefficients = np.asarray([factorial(n)/np.prod([factorial(int(k)) for k in row]) for row in counts])
    return counts, coefficients


@njit(cache=True)
def finite_roots(q, behavior, tau, counts, coefficients):
    result = np.zeros(len(q))
    for s in range(len(q)):
        for k in range(len(counts)):
            probability = coefficients[k]
            for a in range(q.shape[1]): probability *= behavior[s, a]**counts[k, a]
            if probability > 0:
                result[s] += probability*expectile(q[s], counts[k], tau)
    return result


def solve(mdp, behavior=None, samples=None, optimal=False, tol=2e-12):
    behavior = mdp.behavior if behavior is None else behavior
    q = np.zeros_like(mdp.reward)
    if samples is not None: counts, coef = multinomial_counts(samples, q.shape[1])
    for iteration in range(100000):
        if optimal: v = q.max(-1)
        elif samples is None: v = roots(q, behavior, mdp.tau)
        else: v = finite_roots(q, behavior, mdp.tau, counts, coef)
        next_q = mdp.reward + mdp.gamma*mdp.continuation*v[mdp.successor]
        if np.max(np.abs(next_q-q)) < tol: break
        q = next_q
    else: raise RuntimeError('Reference did not converge')
    q = next_q
    if optimal: v = q.max(-1)
    elif samples is None: v = roots(q, behavior, mdp.tau)
    else: v = finite_roots(q, behavior, mdp.tau, counts, coef)
    residual = np.max(np.abs(q-mdp.reward-mdp.gamma*mdp.continuation*v[mdp.successor]))
    assert residual < 1e-10
    return q, v, iteration+1


def policy_value(mdp, q):
    actions = np.argmax(q, axis=1)
    states = np.arange(len(actions))
    p = np.zeros((len(actions), len(actions)))
    p[states, mdp.successor[states, actions]] = mdp.continuation[states, actions]
    return np.linalg.solve(np.eye(len(actions))-mdp.gamma*p, mdp.reward[states, actions])


def best_of_n_policy(q, behavior, n=9):
    """Exact action law of highest-Q among n iid behavior-flow proposals.

    Equal-valued actions share their winning-group mass proportionally to
    behavior probability, matching first-index argmax over iid candidates.
    """
    pi = np.zeros_like(behavior)
    for s in range(len(q)):
        below = 0.
        for value in np.unique(q[s]):
            group = q[s] == value
            mass = behavior[s, group].sum()
            if mass:
                pi[s, group] = ((below+mass)**n-below**n)*behavior[s, group]/mass
            below += mass
    np.testing.assert_allclose(pi.sum(-1), 1., atol=1e-12)
    return pi


def evaluate_policy(mdp, q, behavior=None, n=9, horizon=1000):
    behavior = mdp.behavior if behavior is None else behavior
    pi = best_of_n_policy(q, behavior, n)
    states = len(q)
    p = np.zeros((states, states))
    for s in range(states):
        for a in range(q.shape[1]):
            p[s, mdp.successor[s,a]] += pi[s,a]*mdp.continuation[s,a]
    reward = np.sum(pi*mdp.reward, axis=-1)
    value = np.linalg.solve(np.eye(states)-mdp.gamma*p, reward)
    # Goal is the terminal transition with reward=1; safe exit is not success.
    goal = np.sum(pi*((mdp.reward == 1.) & (mdp.continuation == 0.)), axis=-1)
    augmented = np.zeros((states+1, states+1))
    augmented[:states,:states] = p
    augmented[:states,-1] = goal
    augmented[-1,-1] = 1.
    success = np.linalg.matrix_power(augmented, horizon)[0,-1]
    return float(value[0]), float(np.clip(success, 0., 1.))
