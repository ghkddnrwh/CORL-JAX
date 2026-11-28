"""Coupled data/noise tapes for tabular IQL and finite-expectile Q updates.

All algorithms use the same Q step, target EMA, data, and Q-readout field.
The actor/behavior model is fitted only from action counts, never from Q.
The optional V-readout field explicitly models an additional approximation
error; noise-free experiments are recorded separately.
"""
import numpy as np
from numba import njit
from .mdp import expectile

METHODS = ('iql', 'feq_n9', 'feq_population', 'iql_v9', 'iql_v81')
METRICS = ('q_rmse', 'q_max_error', 'v_residual_rms', 'v_total_error_rms',
           'q_bias', 'start_action', 'population_bellman_rms')


@njit(cache=True)
def choose(cdf, u):
    for i in range(len(cdf)):
        if u < cdf[i]: return i
    return len(cdf)-1


@njit(cache=True)
def run(method, reward, successor, continuation, behavior, behavior_model,
        gamma, tau, updates, alpha_q, alpha_v, ema, seed, q_truth, v_truth,
        sigma_q=0., sigma_v=0., rho=0., spatial=False, log_every=1000,
        v_steps_override=0, init_truth=False, v_pulse=0., q_pulse=0.,
        pulse_state=1, clip=False):
    np.random.seed(2000003+seed)
    states, actions = reward.shape
    # Shared streams are generated BEFORE method-dependent updates.
    visited = np.random.randint(0, states, updates)
    uniforms = np.random.random((updates, 82))
    zq = np.random.standard_normal((updates, 8)).astype(np.float32)
    zv = np.random.standard_normal((updates, 8)).astype(np.float32)
    q = q_truth.copy() if init_truth else np.zeros_like(reward)
    target = q.copy()
    value = v_truth.copy() if init_truth else np.zeros(states)
    value[pulse_state] += v_pulse
    q[pulse_state, 0] += q_pulse
    target[pulse_state, 0] += q_pulse
    cdf = behavior.copy()
    model_cdf = behavior_model.copy()
    for ss in range(states):
        for aa in range(1, actions):
            cdf[ss, aa] += cdf[ss, aa-1]
            model_cdf[ss, aa] += model_cdf[ss, aa-1]
    phi_q = np.empty((states, actions, 8)); phi_v = np.empty((states, 8))
    for s in range(states):
        for d in range(8): phi_v[s,d] = np.cos((d+1)*s/max(states-1,1)*3.7 + .31*d)
        phi_v[s] /= np.sqrt((phi_v[s]**2).sum())
        for a in range(actions):
            for d in range(8): phi_q[s,a,d] = np.cos((d+1)*s/max(states-1,1)*3.7 + .61*a*(d+1) + .31*d)
            phi_q[s,a] /= np.sqrt((phi_q[s,a]**2).sum())
    latent_q = zq[0].astype(np.float64); latent_v = zv[0].astype(np.float64)
    scale = np.sqrt(1-rho*rho)
    logs = np.zeros((1+(updates+log_every-1)//log_every, len(METRICS)))
    steps = np.zeros(len(logs), dtype=np.int64)
    snapshots = np.zeros((len(logs), states, actions))
    v_snapshots = np.zeros((len(logs), states))
    observed_q = np.zeros(actions); v_receiver = np.zeros(actions)
    clean_target_root = np.zeros(states)
    sample_values = np.zeros(actions); counts = np.zeros(actions)
    private_v_steps = v_steps_override if v_steps_override else (9 if method == 3 else 81 if method == 4 else 1)
    logged = 0
    for t in range(updates+1):
        if t == 0 or t % log_every == 0 or t == updates:
            if t > 0: logged += 1
            error = q-q_truth
            for s in range(states): clean_target_root[s] = expectile(target[s], behavior[s], tau)
            residual_sum = value_sum = bellman_sum = 0.
            for s in range(states):
                residual_sum += (value[s]-clean_target_root[s])**2
                value_sum += (value[s]-v_truth[s])**2
                for a in range(actions):
                    exact_root = expectile(q[successor[s,a]], behavior[successor[s,a]], tau)
                    bellman_sum += (q[s,a]-reward[s,a]-gamma*continuation[s,a]*exact_root)**2
            logs[logged,0] = np.sqrt((error**2).mean())
            logs[logged,1] = np.abs(error).max()
            logs[logged,2] = np.sqrt(residual_sum/states) if method in (0,3,4) else 0.
            logs[logged,3] = np.sqrt(value_sum/states) if method in (0,3,4) else np.nan
            logs[logged,4] = error.mean()
            logs[logged,5] = np.argmax(q[0])
            logs[logged,6] = np.sqrt(bellman_sum/(states*actions))
            steps[logged] = t; snapshots[logged] = q; v_snapshots[logged] = value
        if t == updates: break
        s = visited[t]
        a = choose(cdf[s], uniforms[t,0])
        sn = successor[s,a]
        if t > 0:
            latent_q = rho*latent_q + scale*zq[t]
            latent_v = rho*latent_v + scale*zv[t]
        if spatial:
            eps_vs = sigma_v*np.dot(phi_v[s], latent_v)
            eps_vn = sigma_v*np.dot(phi_v[sn], latent_v)
            for act in range(actions):
                v_receiver[act] = target[s,act]+sigma_q*np.dot(phi_q[s,act],latent_q)
                observed_q[act] = target[sn,act]+sigma_q*np.dot(phi_q[sn,act],latent_q)
        else:
            eps_vs = sigma_v*zv[t,0]
            eps_vn = eps_vs if sn == s else sigma_v*zv[t,1]
            for act in range(actions):
                v_receiver[act] = target[s,act]+sigma_q*zq[t,act]
                observed_q[act] = target[sn,act]+sigma_q*zq[t,act if sn == s else act+4]
        if method in (0,3,4): bootstrap = value[sn]+eps_vn
        elif method == 2: bootstrap = expectile(observed_q, behavior_model[sn], tau)
        else:
            counts[:] = 0.
            for k in range(9): counts[choose(model_cdf[sn],uniforms[t,k+1])] += 1.
            bootstrap = expectile(observed_q, counts, tau)
        y = reward[s,a]+gamma*continuation[s,a]*bootstrap
        if clip: y = min(1., max(0., y))
        old_q = q[s,a]
        q[s,a] += alpha_q*(y-q[s,a])
        if method in (0,3,4):
            for k in range(private_v_steps):
                av = a if k == 0 else choose(cdf[s],uniforms[t,k+1])
                delta = v_receiver[av] - (value[s]+eps_vs)
                weight = tau if delta >= 0 else 1-tau
                value[s] += 2*alpha_v*weight*delta
        # Match the FEQ implementation's old-online-Q EMA convention.
        target[s,a] = (1-ema)*target[s,a] + ema*old_q
        for ss in range(states):
            for aa in range(actions):
                if ss != s or aa != a: target[ss,aa] = (1-ema)*target[ss,aa]+ema*q[ss,aa]
    return steps, logs, q, target, value, snapshots, v_snapshots
