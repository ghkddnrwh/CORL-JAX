"""Independent numerical checks and provenance audit for the persistent study."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
from math import comb
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.special import expit
from .persistent_features_v10 import setup, references, exact_v_fit, simulate, METHODS
from .persistent_matched_features_v10 import setup as matched_setup
from .mdp import chain, solve, roots, finite_roots, multinomial_counts, best_of_n_policy
from .fair_population_v7 import expected_score, expected_iql
from .fair_study_v7 import sha, write_json


def independent_root(q, beta, tau):
    def equation(v):
        d = q - v
        return np.dot(beta, np.where(d >= 0, tau, 1-tau)*d)
    return brentq(equation, q.min()-1, q.max()+1, xtol=1e-14)


def scalar_training(method, m, phi, expanded, offline_s, offline_a, steps, seed):
    """Slow scalar oracle: Brent expectiles; explicit old-state updates."""
    indices = np.random.RandomState(8100001+seed).randint(len(offline_s), size=steps)
    rng = np.random.RandomState(9100003+seed)
    vp = expanded if method == 5 else phi
    theta = np.zeros((3, 4)); target = theta.copy(); vtheta = np.zeros(vp.shape[1])
    traces = []
    for t in range(steps+1):
        q = phi @ theta; tq = phi @ target; v = vp @ vtheta
        traces.append((q.copy(), tq.copy(), v.copy()))
        if t == steps:
            break
        u = rng.random_sample(9)
        s, a = offline_s[indices[t]], offline_a[indices[t]]
        ns = m.successor[s, a]
        boot = 0.
        if s == 0:
            if method == 2:
                sampled = np.searchsorted(m.behavior[ns].cumsum(), u)
                boot = independent_root(tq[ns], np.bincount(sampled, minlength=4)/9, m.tau)
            elif method in (3, 4, 6):
                exact = independent_root(tq[ns], m.behavior[ns], m.tau)
                boot = exact + (.5 if method == 3 else 0.)*(v[ns]-exact)
            else:
                boot = v[ns]
        old = theta.copy()
        theta[:, a] += .02 * (m.reward[s, a]+m.gamma*boot-q[s, a])*phi[s]
        if method not in (2, 6):
            if method == 1:
                aa = np.searchsorted(m.behavior[s].cumsum(), u)
                delta = tq[s, aa]-v[s]
            else:
                delta = np.array([tq[s, a]-v[s]])
            gradient = np.mean(np.where(delta >= 0, m.tau, 1-m.tau)*delta)
            vtheta += .04*gradient*vp[s]
        target = .995*target + .005*old
    return [np.stack([r[k] for r in traces]) for k in range(3)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    checks = {}; worst_fit = worst_finite = 0.
    for seed in (1300, 1305, 1315):
        for tau in (.7, .9, .99):
            m, phi, expanded, ds, da, occ, model, _ = setup(seed, tau)
            v, theta, projected = exact_v_fit(m.reward, m.behavior, occ, phi, tau)
            x = phi[1:, 1:]; q = m.reward[1:]; beta = m.behavior[1:]
            def obj(z):
                d = q-(x@z)[:, None]
                w = np.where(d >= 0, tau, 1-tau)
                loss = np.sum(occ[1:, None]*beta*w*d*d)
                grad = -2*x.T@(occ[1:] * np.sum(beta*w*d, -1))
                return loss, grad
            # A separate generic convex optimizer, initialized away from the solution.
            fit = minimize(obj, np.array([-.2, .4]), jac=True, method='BFGS',
                           options={'gtol': 1e-12, 'maxiter': 1000})
            error = float(abs(fit.x-theta).max()); worst_fit = max(worst_fit, error)
            assert error < 2e-8, (seed, tau, fit.message, error)
            assert abs(projected).max() < 1e-12
            ref = references(m, phi, occ, model)
            np.testing.assert_allclose(ref['leaf_score_residual'],
                ref['leaf_value_gap']*ref['leaf_secant'], atol=1e-13)
            # Binary-binomial reduction independently checks the 4-action multinomial sum.
            counts, coeff = multinomial_counts(9, 4)
            actual = finite_roots(m.reward, model, tau, counts, coeff)
            for s in (1, 2, 3):
                low, high = q[s-1].min(), q[s-1].max()
                prob = model[s, m.reward[s] == high].sum()
                expected = 0.
                for k in range(10):
                    value = ((1-tau)*(9-k)*low+tau*k*high)/((1-tau)*(9-k)+tau*k)
                    expected += comb(9, k)*prob**k*(1-prob)**(9-k)*value
                error = abs(actual[s]-expected); worst_finite = max(worst_finite, error)
                assert error < 1e-12
    checks['convex_fit_cases'] = 9; checks['convex_fit_max_error'] = worst_fit
    checks['binomial_cases'] = 27; checks['binomial_max_error'] = worst_finite

    m, phi, expanded, ds, da, occ, model, _ = setup(42)
    worst_trace = 0.
    for method in range(len(METHODS)):
        actual = simulate(method, m.reward, m.successor, m.behavior, model, phi, expanded,
                          ds, da, m.tau, m.gamma, 300, 42, False, 1)
        expected = scalar_training(method, m, phi, expanded, ds, da, 300, 42)
        for a, e in zip((actual[1], actual[2], actual[3]), expected):
            worst_trace = max(worst_trace, float(abs(a-e).max()))
            np.testing.assert_allclose(a, e, atol=2e-13, rtol=1e-11)
        longer = simulate(method, m.reward, m.successor, m.behavior, model, phi, expanded,
                           ds, da, m.tau, m.gamma, 400, 42, False, 1)
        for i in (1, 2, 3):
            np.testing.assert_array_equal(actual[i], longer[i][:301])
    checks['scalar_training_methods'] = len(METHODS)
    checks['scalar_trace_max_error'] = worst_trace
    checks['horizon_prefix_invariance'] = 'all seven feature methods, 300 vs 400 steps'

    worst_gaussian = 0.
    for tau in (.5, .9, .99):
        for sigma in (.002, .1):
            for delta in (-2*sigma, -.1*sigma, 0., .8*sigma, 3*sigma):
                def integrand(z):
                    d = delta+sigma*z
                    return (tau if d >= 0 else 1-tau)*d*norm.pdf(z)
                # Split at the kink; integrate over standard Normal coordinates.
                kink = -delta/sigma
                integral = quad(integrand, -np.inf, kink, epsabs=1e-13)[0]
                integral += quad(integrand, kink, np.inf, epsabs=1e-13)[0]
                error = abs(integral-expected_score(delta, sigma, tau))
                worst_gaussian = max(worst_gaussian, error); assert error < 2e-12
    checks['gaussian_quadrature_cases'] = 30
    checks['gaussian_score_max_error'] = worst_gaussian
    m = chain(4, .25, .999, .9, .8); m.reward[0, 1] = 0.; m.continuation[0, 1] = 1.
    exact_q, exact_v = expected_iql(m, .002)
    v = np.zeros(5)
    # Independent whole-system fixed-point iteration (not backward scalar solving).
    for it in range(50000):
        q = m.reward+m.gamma*m.continuation*v[m.successor]
        new = np.array([brentq(lambda z: np.dot(m.behavior[s], expected_score(q[s]-z, .002, .9)),
                             q[s].min()-1, q[s].max()+1, xtol=1e-14) for s in range(5)])
        if abs(new-v).max() < 5e-14:
            break
        v = new
    np.testing.assert_allclose(new, exact_v, atol=6e-10)
    checks['expected_loss_iteration_count'] = it+1
    checks['expected_loss_solver_max_error'] = float(abs(new-exact_v).max())

    # Enumerate candidate sequences to independently check best-of-N policy law,
    # including ties. N=3 suffices to validate the grouping/formula.
    import itertools
    b = np.array([[.1, .2, .3, .4]]); q = np.array([[1., 2., 2., -1.]])
    pi = np.zeros(4)
    for seq in itertools.product(range(4), repeat=3):
        winner = seq[int(np.argmax(q[0, list(seq)]))]
        pi[winner] += float(np.prod(b[0, list(seq)]))
    np.testing.assert_allclose(pi, best_of_n_policy(q, b, 3)[0], atol=1e-13)
    checks['policy_enumeration_sequences'] = 64

    model_max_error = matched_v_error = 0.
    for seed in (1500,1507,1515):
        m,phi,expanded,ds,da,occ,model,true_beta,theta = matched_setup(seed)
        # Independent Newton likelihood fit using an explicit 3x3 Hessian.
        counts=np.zeros((4,4))
        for s,a in zip(ds,da):counts[s,a]+=1
        n=counts.sum(-1);k=counts[:,[0,2]].sum(-1);z=np.zeros(3)
        for iteration in range(100):
            prob=expit(phi@z);gradient=phi.T@(n*prob-k)
            hessian=phi.T@((n*prob*(1-prob))[:,None]*phi)
            if abs(gradient).max()<1e-10:break
            z-=np.linalg.solve(hessian,gradient)
        assert abs(gradient).max()<1e-8
        model_max_error=max(model_max_error,float(abs(z-theta).max()))
        np.testing.assert_allclose(z,theta,atol=3e-7)
        np.testing.assert_allclose(model[:,[0,2]].sum(-1),expit(phi@theta),atol=1e-13)
        for beta in (m.behavior,model):
            v,coef,projected=exact_v_fit(m.reward,beta,occ,phi,.99)
            x=phi[1:,1:]
            def loss(u):
                delta=m.reward[1:]-(x@u)[:,None]
                weighted=occ[1:,None]*beta[1:]*np.where(delta>=0,.99,.01)
                return np.sum(weighted*delta**2),-2*x.T@np.sum(weighted*delta,-1)
            fit=minimize(loss,np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-12})
            error=float(abs(fit.x-coef).max());matched_v_error=max(matched_v_error,error)
            assert error<1e-7
    checks['matched_behavior_newton_cases']=3
    checks['matched_behavior_coefficient_max_error']=model_max_error
    checks['matched_V_independent_fit_cases']=6
    checks['matched_V_independent_fit_max_error']=matched_v_error

    runs = {}; source_files = set()
    for stage in ('tabular', 'features', 'matched_features', 'matched_tau'):
        reg = json.loads((root/stage/'registration.json').read_text())
        records = list((root/stage/'raw').glob('*/result.json'))
        expected_count = sum(len(c['seeds']) for c in reg['cases'])
        assert len(records) == expected_count
        stage_runs = {}
        for meta in records:
            record = json.loads(meta.read_text()); data = meta.with_name('data.npz')
            assert record['sources'] == reg['sources']
            assert record['sha256'] == sha(data)
            identity = record['identity']; c = identity['case']; seed = identity['seed']
            assert c in reg['cases'] and seed in c['seeds']
            key = (c['name'], c['method'], seed)
            assert key not in stage_runs
            stage_runs[key] = data
        for file, checksum in reg['sources'].items():
            assert sha(file) == checksum
            source_files.add(file)
        runs[stage] = stage_runs
        checks[stage+'_audited_runs'] = len(records)
    for seed in range(1100, 1116):
        base = np.load(runs['tabular']['main', 'iql', seed])
        for method in ('iql_batch9', 'feq9', 'repair_zero'):
            other = np.load(runs['tabular']['main', method, seed])
            for key in ('noise_statistics', 'behavior', 'model', 'step'):
                np.testing.assert_array_equal(base[key], other[key])
    for seed in range(1300, 1316):
        base = np.load(runs['features']['main', 'iql', seed])
        for method in METHODS:
            other = np.load(runs['features']['main', method, seed])
            for key in ('offline_states', 'offline_actions', 'behavior', 'model', 'phi', 'step'):
                np.testing.assert_array_equal(base[key], other[key])
        a = np.load(runs['features']['main', 'population', seed])
        b = np.load(runs['features']['main', 'repair_zero', seed])
        np.testing.assert_array_equal(a['q'], b['q'])
    checks['paired_inputs'] = 'PASS: matched source, data and noise streams at equal horizons'
    checks['feature_repair_zero_equals_population'] = '16/16 trajectories bitwise equal'
    for seed in range(1500,1516):
        base=np.load(runs['matched_features']['main','iql',seed])
        for method in ('iql9','feq9','iql_expanded','repair_zero'):
            other=np.load(runs['matched_features']['main',method,seed])
            for key in ('offline_states','offline_actions','behavior','model','behavior_parameters','phi','step'):
                np.testing.assert_array_equal(base[key],other[key])
    checks['matched_feature_paired_inputs']='16/16 seeds, every method, identical buffer and fitted model'
    for seed in range(1700,1708):
        for case in ('tau0.9','tau0.95','tau0.98','tau0.99'):
            base=np.load(runs['matched_tau'][case,'iql',seed])
            for method in ('iql9','feq9'):
                other=np.load(runs['matched_tau'][case,method,seed])
                for key in ('offline_states','offline_actions','model','behavior_parameters','step'):
                    np.testing.assert_array_equal(base[key],other[key])
    checks['matched_tau_paired_inputs']='8/8 held-out seeds x4 levels, identical buffer and fitted model'
    checks['source_files'] = sorted(source_files)
    write_json(root/'VALIDATION.json', dict(status='PASS', checks=checks))
    print(json.dumps(dict(status='PASS', checks=checks), indent=2))


if __name__ == '__main__':
    main()
