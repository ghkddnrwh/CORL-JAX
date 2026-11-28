"""Independent scalar/reference and saved-artifact checks for the v11 study."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from .matched_error_v11 import (regularized_root, train_ridge, train_channels,
                               branch_mdp, ridge_reference, save_json)
from .mdp import chain, expectile, finite_roots, multinomial_counts, roots
from .score_controls_v11 import train as score_train, forced_reference


def scalar_root(q, b, tau, penalty=0.):
    def f(v):
        d = q-v
        return np.dot(b, np.where(d >= 0, tau, 1-tau)*d)-penalty*v
    return brentq(f, min(-100., q.min()-100.), max(100., q.max()+100.), xtol=1e-14)


def reference(m, steps, seed, family, method=0, channel=0, sigma=.1,
              rho=.9, batch9=True, lq=.003, lv=.003, eta=.005):
    offset = {'ridge':4100009, 'channel':5100001, 'score':7100011}[family]
    rng = np.random.RandomState(offset+seed)
    q = np.zeros_like(m.reward); target = q.copy(); v = np.zeros(len(q))
    z = np.zeros(len(q)); qh = [q.copy()]; vh = [v.copy()]
    for t in range(steps):
        s = rng.randint(len(q)); u = rng.random_sample(10)
        if family == 'channel':
            innovation = rng.standard_normal(len(q))
            z = sigma*innovation if t == 0 else rho*z+sigma*np.sqrt(1-rho*rho)*innovation
        a = np.searchsorted(np.cumsum(m.behavior[s]), u[0])
        ns = m.successor[s, a]
        old_q = q.copy(); old_v = v.copy(); old_target = target.copy()
        if (family == 'ridge' and method == 2) or (family == 'score' and method == 1):
            aa = np.searchsorted(np.cumsum(m.behavior[ns]), u[1:])
            boot = scalar_root(old_target[ns, aa], np.full(9, 1/9), m.tau)
            if family == 'score' and channel == 1: boot -= eta
        elif family == 'ridge' and method == 3:
            boot = scalar_root(old_target[ns], m.behavior[ns], m.tau)
        else:
            boot = old_v[ns]
            if family == 'channel' and channel == 2: boot += z[ns]
            if family == 'score' and channel == 3: boot -= eta
        penalty_q = 0. if family == 'channel' else lq
        q[s, a] = old_q[s, a]+.1*(m.reward[s, a]+m.gamma*m.continuation[s, a]*boot-(1+penalty_q)*old_q[s, a])
        needs_v = family == 'channel' or (family == 'ridge' and method in (0,1,4)) or (family == 'score' and method != 1)
        if needs_v:
            full = (family == 'channel' and not batch9) or (family == 'ridge' and method == 4)
            if full:
                aa = np.arange(4); weights = m.behavior[s]
            else:
                single = family == 'ridge' and method == 0
                if family == 'score' and method == 2: aa = np.array([a])
                else: aa = np.searchsorted(np.cumsum(m.behavior[s]), u[1:2] if single else u[1:])
                weights = np.full(len(aa), 1/len(aa))
            observed = old_target[s, aa].copy()
            if family == 'channel' and channel == 1: observed += z[s]
            if family == 'score' and channel == 1: observed -= eta
            d = observed-old_v[s]
            score = np.dot(weights, np.where(d >= 0, m.tau, 1-m.tau)*d)
            penalty_v = 0. if family == 'channel' else lv
            if family == 'score' and channel == 2: score -= eta
            v[s] = old_v[s]+.1*(score-penalty_v*old_v[s])
        target = .995*old_target+.005*old_q
        qh.append(q.copy()); vh.append(v.copy())
    return np.array(qh), np.array(vh)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    rng = np.random.default_rng(891)
    errors = []
    for _ in range(100):
        q = rng.normal(size=4); b = rng.dirichlet(np.ones(4))
        tau = rng.uniform(.05,.95); penalty = rng.uniform(0,.05)
        errors.append(abs(regularized_root(q,b,tau,penalty)-scalar_root(q,b,tau,penalty)))
    assert max(errors) < 1e-11
    q = np.array([-.5,.2,1.,2.]); b = np.array([.1,.2,.3,.4]); tau=.9
    enumeration = sum(np.prod(b[list(a)])*scalar_root(q[list(a)],np.full(3,1/3),tau)
                      for a in itertools.product(range(4),repeat=3))
    counts, coef = multinomial_counts(3,4)
    enumeration_error=abs(enumeration-finite_roots(q[None],b[None],tau,counts,coef)[0])
    assert enumeration_error < 1e-12
    trace_count=0; maximum=0.
    m=chain(3,.25,.999,.9,.9)
    for method in range(5):
        out=train_ridge(m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.tau,m.gamma,
                        method,.003,.003,137,7,log_every=1)
        qr,vr=reference(m,137,7,'ridge',method=method)
        maximum=max(maximum,float(abs(qr-out[1]).max()),float(abs(vr-out[2]).max()))
        trace_count+=1
    m=branch_mdp(.02,.9)
    for channel,rho,batch in itertools.product(range(3),(0.,.9,1.),(False,True)):
        out=train_channels(m.reward,m.successor,m.continuation,m.behavior,m.tau,m.gamma,
                           channel,.1,rho,137,7,batch,log_every=1)
        qr,vr=reference(m,137,7,'channel',channel=channel,rho=rho,batch9=batch)
        maximum=max(maximum,float(abs(qr-out[1]).max()),float(abs(vr-out[2]).max()))
        trace_count+=1
    m=chain(3,.25,.999,.9,.9)
    for method,channel in itertools.product(range(3),range(4)):
        out=score_train(m.reward,m.successor,m.continuation,m.behavior,m.tau,m.gamma,
                        method,channel,.005,137,7,.003,.003,log_every=1)
        qr,vr=reference(m,137,7,'score',method=method,channel=channel)
        maximum=max(maximum,float(abs(qr-out[1]).max()),float(abs(vr-out[2]).max()))
        trace_count+=1
    assert maximum < 1e-11, maximum
    # An identical state-dependent static output error has the same fixed point
    # through Q readout and V readout. The score perturbation need not.
    qa,_=forced_reference(m,'q_output',.005)
    qb,_=forced_reference(m,'v_output',.005)
    static_error=float(abs(qa-qb).max()); assert static_error < 1e-10
    fixed_residuals=[]
    for lq,lv,finite in [(.003,.003,False),(.003,0.,False),(.003,0.,True)]:
        q,v=ridge_reference(m,lq,lv,finite)
        fixed_residuals.append(float(abs((1+lq)*q-m.reward-m.gamma*m.continuation*v[m.successor]).max()))
        if not finite:
            d=q-v[:,None];g=(m.behavior*np.where(d>=0,m.tau,1-m.tau)*d).sum(-1)
            fixed_residuals.append(float(abs(g-lv*v).max()))
    assert max(fixed_residuals)<1e-10
    by_pair={}; counts={}; total_updates=0
    for stage in ('discovery','ridge','channels','controls','score'):
        registration=json.loads((root/stage/'registration.json').read_text())
        source=Path(__file__).with_name('score_controls_v11.py' if stage=='score' else 'matched_error_v11.py')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==registration['source_sha256']
        expected={hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16] for c in registration['jobs']}
        paths=list((root/stage/'raw').glob('*/result.json'))
        assert {p.parent.name for p in paths}==expected
        counts[stage]=len(paths)
        for path in paths:
            obj=json.loads(path.read_text()); data=path.with_name('data.npz')
            assert hashlib.sha256(data.read_bytes()).hexdigest()==obj['data_sha256']
            c=obj['config']; total_updates+=c['steps']
            with np.load(data) as h:
                assert h['step'][-1]==c['steps'] and np.isfinite(h['q']).all()
                if c['suite']=='channel' and stage!='discovery' and c['channel'] in (1,2):
                    key=tuple((k,c[k]) for k in sorted(c) if k!='channel')
                    by_pair.setdefault(key,{})[c['channel']]=(h['noise_moments'].copy(),h['q'][h['step']>=.5*c['steps']].copy())
    static_tail_difference=0.
    for key,pair in by_pair.items():
        assert set(pair)=={1,2}
        np.testing.assert_array_equal(pair[1][0],pair[2][0])
        if dict(key)['rho']==1.:
            static_tail_difference=max(static_tail_difference,float(abs(pair[1][1]-pair[2][1]).max()))
    assert static_tail_difference<1e-10
    result=dict(status='passed',root_checks=100,root_max_error=max(errors),
        independent_traces=trace_count,trace_max_error=maximum,
        finite_expectile_enumeration_error=enumeration_error,
        fixed_equation_max_residual=max(fixed_residuals),matched_noise_pairs=len(by_pair),
        static_fixed_Q_difference=static_error,static_tail_Q_difference=static_tail_difference,
        artifacts=counts,trajectories=sum(counts.values()),total_updates=total_updates)
    save_json(root/'validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__': main()
