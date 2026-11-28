"""Independent numerical checks for the score/value distinction and dynamics."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from numba import njit
from .mdp import chain, solve, expectile
from .theory_channels_v9 import score, score_root, fixed_forcing, secant_diagnostics
from .fair_engine_v7 import simulate, METHODS


@njit
def bisect(q,b,tau,zeta):
    lo=q.min()-1-abs(zeta)/min(tau,1-tau)
    hi=q.max()+1+abs(zeta)/min(tau,1-tau)
    for _ in range(70):
        v=(lo+hi)/2
        g=zeta
        for a in range(len(q)):
            d=q[a]-v;g+=b[a]*(tau if d>=0 else 1-tau)*d
        if g>0:lo=v
        else:hi=v
    return (lo+hi)/2


@njit
def iterate(reward,sn,cont,b,gamma,tau,d,critic):
    q=np.zeros_like(reward)
    for t in range(100000):
        v=np.empty(len(q))
        for s in range(len(q)):
            v[s]=bisect(q[s]+(d if critic else 0.),b[s],tau,0. if critic else d)
        new=np.empty_like(q)
        for s in range(len(q)):
            for a in range(q.shape[1]):
                new[s,a]=reward[s,a]+gamma*cont[s,a]*v[sn[s,a]]
        error=np.max(np.abs(new-q));q=new
        if error<1e-13:break
    assert error<1e-13
    return q


def python_short(method,m,seed,steps):
    """Plain scalar Python update, independent of Numba training implementation."""
    rng=np.random.RandomState(900001+seed)
    visited=rng.randint(0,len(m.reward),steps);u=rng.random_sample((steps,10))
    q=np.zeros_like(m.reward);target=q.copy();v=np.zeros(len(q))
    cdf=np.cumsum(m.behavior,-1)
    for t in range(steps):
        s=visited[t];a=np.searchsorted(cdf[s],u[t,0]);sn=m.successor[s,a]
        if method=='feq9':
            actions=np.searchsorted(cdf[sn],u[t,1:])
            counts=np.bincount(actions,minlength=4).astype(float)
            boot=bisect(target[sn],counts/9,m.tau,0.)
        else:boot=v[sn]
        old=q.copy()
        q[s,a]+=.1*(m.reward[s,a]+m.gamma*m.continuation[s,a]*boot-q[s,a])
        if method=='iql':
            d=target[s,a]-v[s];v[s]+=.2*(m.tau if d>=0 else 1-m.tau)*d
        elif method=='iql_batch9':
            actions=np.searchsorted(cdf[s],u[t,1:]);d=target[s,actions]-v[s]
            v[s]+=.2*np.mean(np.where(d>=0,m.tau,1-m.tau)*d)
        target=.995*target+.005*old
    return q,target,v


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(991);worst=0.
    for _ in range(200):
        q=rng.normal(size=4);b=rng.dirichlet(np.ones(4));tau=rng.uniform(.5,.999);z=rng.normal()*.001
        np.testing.assert_allclose(score_root(q,b,tau,z),bisect(q,b,tau,z),atol=2e-12)
        v=score_root(q,b,tau,z)
        d=secant_diagnostics(q[None],np.array([v]),b[None],tau)
        np.testing.assert_allclose(d['secant']*d['gap'],z,atol=2e-12)
        assert min(tau,1-tau)-1e-8<=d['secant'][0]<=max(tau,1-tau)+1e-8
    m=chain(4,.25,.999,.99,.8);m.reward[0,1]=0.;m.continuation[0,1]=1.
    for delta in (0.,-3e-5,3e-5,1e-4):
        for channel in ('critic','score'):
            q,v=fixed_forcing(m,channel,delta)
            qi=iterate(m.reward,m.successor,m.continuation,m.behavior,m.gamma,m.tau,delta,channel=='critic')
            worst=max(worst,float(abs(q-qi).max()))
            np.testing.assert_allclose(q,qi,atol=2e-9)
    m=chain(4,.1,.999,.99,.8);qt,vt,_=solve(m)
    for method in ('iql','iql_batch9','feq9'):
        q,t,v=python_short(method,m,876,200)
        out=simulate(METHODS.index(method),m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.gamma,m.tau,200,876,qt,vt,log_every=200)
        np.testing.assert_allclose(q,out[1][-1],atol=2e-12)
        np.testing.assert_allclose(v,out[2][-1],atol=2e-12)
        np.testing.assert_allclose(t,out[3][-1],atol=2e-12)
        assert np.all(out[-2]==0)
    result=dict(status='PASS',random_root_and_secant_checks=200,independent_mdp_fixed_points=8,
                independent_short_training_methods=3,max_fixed_point_discrepancy=worst)
    (args.root/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
