"""Matched error-path and fully tabular regularized-regression experiments.

All output artifacts belong under logs/. No production algorithms are changed.
Ridge is an explicit experimental intervention, NOT a default of IQL or FEQ.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import brentq
from .mdp import (MDP, chain, expectile, roots, solve, multinomial_counts,
                  finite_roots, evaluate_policy, best_of_n_policy)
from .behavior_flow import offline_counts, fit


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def save_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@njit(cache=True)
def regularized_root(q, beta, tau, penalty):
    """Exact minimizer of .5 E[w(q-v)^2] + .5 penalty*v^2."""
    order = np.argsort(q)
    total = beta.sum(); weighted = (beta*q).sum()
    low_mass = low_sum = 0.
    for k in range(len(q)+1):
        den = (1-tau)*low_mass+tau*(total-low_mass)+penalty*total
        v = ((1-tau)*low_sum+tau*(weighted-low_sum))/den
        lower = q[order[k-1]] if k else -np.inf
        upper = q[order[k]] if k<len(q) else np.inf
        if lower-1e-12 <= v <= upper+1e-12:
            return v
        if k<len(q):
            a=order[k];low_mass+=beta[a];low_sum+=beta[a]*q[a]
    raise ValueError('Root not bracketed')


@njit(cache=True)
def pick(cdf, u):
    for a in range(len(cdf)):
        if u < cdf[a]: return a
    return len(cdf)-1


def ridge_reference(m, lq, lv, finite=False, model=None):
    """Independent backward scalar fixed-point solve, including self loops."""
    model = m.behavior if model is None else model
    counts, coeff = multinomial_counts(9,4)
    q = np.zeros_like(m.reward); v = np.zeros(len(q))
    for s in range(len(q)-1,-1,-1):
        def q_at(x):
            nv = np.array([x if ns==s else v[ns] for ns in m.successor[s]])
            return (m.reward[s]+m.gamma*m.continuation[s]*nv)/(1+lq)
        def value(x):
            qq=q_at(x)
            if finite:
                return finite_roots(qq[None],model[s:s+1],m.tau,counts,coeff)[0]
            return regularized_root(qq,m.behavior[s],m.tau,lv)
        bound = 2/(1-m.gamma)
        v[s] = brentq(lambda x:x-value(x),-bound,bound,xtol=1e-13)
        q[s] = q_at(v[s])
    return q,v


@njit(cache=True)
def train_ridge(reward,sn,cont,beta,model,tau,gamma,method,lq,lv,
                steps,seed,alpha=.1,ema=.005,log_every=10000):
    # RNG draws per step never depend on method or horizon.
    np.random.seed(4100009+seed)
    states,actions=reward.shape
    cdf=beta.copy()
    mcdf=model.copy()
    for s in range(states):
        for a in range(1,actions):cdf[s,a]+=cdf[s,a-1];mcdf[s,a]+=mcdf[s,a-1]
    q=np.zeros_like(reward);target=q.copy();v=np.zeros(states)
    count=1+(steps+log_every-1)//log_every
    times=np.zeros(count,np.int64);qs=np.zeros((count,states,actions))
    vs=np.zeros((count,states));targets=qs.copy();idx=0
    counts=np.zeros(actions)
    for t in range(steps+1):
        if t==0 or t%log_every==0 or t==steps:
            times[idx]=t;qs[idx]=q;vs[idx]=v;targets[idx]=target;idx+=1
        if t==steps:break
        s=np.random.randint(states);u=np.random.random(10);a=pick(cdf[s],u[0]);ns=sn[s,a]
        boot=0.
        if cont[s,a]:
            if method==2:
                counts[:]=0.
                for k in range(9):counts[pick(mcdf[ns],u[k+1])]+=1
                boot=expectile(target[ns],counts,tau)
            elif method==3:boot=expectile(target[ns],beta[ns],tau)
            else:boot=v[ns]
        old=q[s,a]
        q[s,a]+=alpha*(reward[s,a]+gamma*cont[s,a]*boot-(1+lq)*old)
        if method in (0,1,4):
            grad=0.
            if method==4:
                for av in range(actions):
                    d=target[s,av]-v[s];grad+=beta[s,av]*(tau if d>=0 else 1-tau)*d
            else:
                n=9 if method==1 else 1
                for k in range(n):
                    av=pick(cdf[s],u[k+1]);d=target[s,av]-v[s]
                    grad+=(tau if d>=0 else 1-tau)*d/n
            v[s]+=alpha*(grad-lv*v[s])
        for ss in range(states):
            for aa in range(actions):
                online=old if ss==s and aa==a else q[ss,aa]
                target[ss,aa]=(1-ema)*target[ss,aa]+ema*online
    return times,qs,vs,targets


def branch_mdp(gap=.01,tau=.9):
    # Both leaf action profiles differ only by an offset. All values tabular.
    r=np.zeros((3,4));r[1]=[1.,0.,1.,0.];r[2]=r[1]-gap
    sn=np.zeros((3,4),dtype=np.int64);sn[0]=[1,2,1,2]
    cont=np.zeros((3,4));cont[0]=1.
    return MDP(r,sn,cont,np.full((3,4),.25),.999,tau,'two_branch_offset')


@njit(cache=True)
def train_channels(reward,sn,cont,beta,tau,gamma,channel,sigma,rho,
                   steps,seed,batch9=True,alpha_q=.1,alpha_v=.1,log_every=1000):
    """Same scalar field Z_t(s) in Q-target read OR V-bootstrap read.

    Q field is common across actions, so state-action RMS equals state RMS.
    V readout is an intervention only in the Q target, not its regression loss.
    This tests propagation/temporal filtering, not a natural V optimizer error.
    """
    np.random.seed(5100001+seed)
    states,actions=reward.shape
    q=np.zeros_like(reward);target=q.copy();v=np.zeros(states)
    cdf=beta.copy()
    for s in range(states):
        for a in range(1,actions):cdf[s,a]+=cdf[s,a-1]
    field=np.zeros(states)
    count=1+(steps+log_every-1)//log_every
    qs=np.zeros((count,states,actions));vs=np.zeros((count,states));idx=0
    times=np.zeros(count,np.int64);noise_mom=np.zeros((count,2));s1=s2=0.;nw=0
    for t in range(steps+1):
        if t==0 or t%log_every==0 or t==steps:
            times[idx]=t;qs[idx]=q;vs[idx]=v
            if nw:noise_mom[idx,0]=s1/nw;noise_mom[idx,1]=s2/nw
            s1=s2=0.;nw=0;idx+=1
        if t==steps:break
        s=np.random.randint(states);u=np.random.random(10)
        z=np.random.standard_normal(states)
        field=sigma*z if t==0 else rho*field+sigma*np.sqrt(1-rho*rho)*z
        for ss in range(states):s1+=field[ss];s2+=field[ss]**2;nw+=1
        a=pick(cdf[s],u[0]);ns=sn[s,a]
        boot=v[ns]+(field[ns] if channel==2 else 0.)
        old=q[s,a]
        q[s,a]+=alpha_q*(reward[s,a]+gamma*cont[s,a]*boot-q[s,a])
        grad=0.
        if batch9:
            for k in range(9):
                av=pick(cdf[s],u[k+1])
                d=target[s,av]+(field[s] if channel==1 else 0.)-v[s]
                grad+=(tau if d>=0 else 1-tau)*d/9.
        else:
            for av in range(actions):
                d=target[s,av]+(field[s] if channel==1 else 0.)-v[s]
                grad+=beta[s,av]*(tau if d>=0 else 1-tau)*d
        v[s]+=alpha_v*grad
        for ss in range(states):
            for aa in range(actions):
                online=old if ss==s and aa==a else q[ss,aa]
                target[ss,aa]=.995*target[ss,aa]+.005*online
    return times,qs,vs,noise_mom


def branch_metrics(m,q):
    pi=best_of_n_policy(q,m.behavior,9)
    correct=float(pi[0,[0,2]].sum())
    ret=float(np.dot(pi[0],m.gamma*np.sum(pi*m.reward,-1)[m.successor[0]]))
    return correct,ret


def job_worker(job):
    folder,c=job;folder=Path(folder)
    key=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
    out=folder/'raw'/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())['metrics']
    if c['suite']=='ridge':
        m=chain(c['length'],.25,.999,c['tau'],c['safe'])
        m.behavior=fit(offline_counts(m.behavior,c['seed']))
        model=m.behavior.copy()
        if c.get('model_noise',0):
            z=np.random.default_rng(6100007+c['seed']).normal(size=model.shape)
            model*=np.exp(c['model_noise']*z);model/=model.sum(-1,keepdims=True)
        qt,vt,_=solve(m)
        t,q,v,target=train_ridge(m.reward,m.successor,m.continuation,m.behavior,model,m.tau,m.gamma,
                                c['method'],c['lq'],c['lv'],c['steps'],c['seed'])
        ref,rv=ridge_reference(m,c['lq'],c['lv'],c['method']==2,model)
        tail=t>=.9*c['steps']
        metrics=np.array([evaluate_policy(m,qq,model) for qq in q[tail]])
        qr=np.sqrt(np.mean((q[tail]-qt)**2,axis=(1,2)))
        row={**c,'success':float(metrics[:,1].mean()),'return':float(metrics[:,0].mean()),
             'q_rmse':float(qr.mean()),'exact_success':evaluate_policy(m,ref,model)[1],
             'exact_return':evaluate_policy(m,ref,model)[0],
             'model_tv':float(np.abs(model-m.behavior).sum(-1).mean()/2)}
        np.savez_compressed(out/'data.npz',step=t,q=q,v=v,target=target,q_truth=qt,v_truth=vt,
                            q_fixed=ref,v_fixed=rv,beta=m.behavior,model=model)
    else:
        m=branch_mdp(c['gap'],c['tau']);qt,_,_=solve(m)
        t,q,v,nm=train_channels(m.reward,m.successor,m.continuation,m.behavior,m.tau,m.gamma,
            c['channel'],c['sigma'],c['rho'],c['steps'],c['seed'],c['batch9'])
        tail=t>=.5*c['steps'];metrics=np.array([branch_metrics(m,qq) for qq in q[tail]])
        row={**c,'success':float(metrics[:,0].mean()),'return':float(metrics[:,1].mean()),
             'q_rmse':float(np.sqrt(((q[tail]-qt)**2).mean((1,2))).mean()),
             'noise_rms':float(np.sqrt(nm[1:,1].mean()))}
        np.savez_compressed(out/'data.npz',step=t,q=q,v=v,noise_moments=nm,q_truth=qt)
    save_json(out/'result.json',dict(config=c,metrics=row,data_sha256=hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()))
    return row


def exploration(root,workers):
    rows=[]
    for length in (8,16,24):
        for safe in (.8,.9,.95):
            for penalty in (.0003,.001,.003):
                m=chain(length,.25,.999,.9,safe)
                qi,_=ridge_reference(m,penalty,penalty)
                qf,_=ridge_reference(m,penalty,0.,True)
                qf2,_=ridge_reference(m,2*penalty,0.,True)
                vi,si=evaluate_policy(m,qi);vf,sf=evaluate_policy(m,qf);vf2,sf2=evaluate_policy(m,qf2)
                rows.append(dict(length=length,safe=safe,penalty=penalty,iql=si,feq=sf,feq_double_q_penalty=sf2,
                                 iql_return=vi,feq_return=vf,feq_double_return=vf2))
    save_csv(root/'ridge_discovery.csv',rows)
    jobs=[]
    for tau in (.5,.9):
        for gap in (.005,.02):
            for sigma in (.01,.04,.1):
                for batch in (False,True):
                    for channel in (0,1,2):
                        for seed in (2100,2101):
                            jobs.append(dict(suite='channel',tau=tau,gap=gap,sigma=sigma,rho=0.,
                                batch9=batch,channel=channel,seed=seed,steps=200000))
    save_json(root/'registration.json',dict(stage='exploration',jobs=jobs,
        ridge_grid=dict(lengths=[8,16,24],safe=[.8,.9,.95],penalty=[.0003,.001,.003]),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(job_worker,[(str(root),c) for c in jobs]))
    save_csv(root/'channel_discovery.csv',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--jobs',type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    if a.jobs:
        jobs=json.loads(a.jobs.read_text())
        save_json(a.root/'registration.json',dict(jobs=jobs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows=list(pool.map(job_worker,[(str(a.root),c) for c in jobs]))
        keys=list(dict.fromkeys(k for r in rows for k in r));save_csv(a.root/'per_seed.csv',[{k:r.get(k,'') for k in keys} for r in rows])
    else:exploration(a.root,a.workers)
    print(json.dumps(dict(status='complete',seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
