"""Exact small-MDP checks: min/mean twin backups share the ideal fixed point.

Enumerate the binomial action counts for9 IID behavior proposals. This tests
mathematical operators, not neural training, oracle residual recovery or returns.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def finite_value(q,p,tau,n=9):
    k=np.arange(n+1,dtype=np.float64)[None,:]
    q0,q1=q[:,0,None],q[:,1,None]
    w0=np.where(q0<=q1,1-tau,tau)*(n-k)
    w1=np.where(q0<=q1,tau,1-tau)*k
    root=(w0*q0+w1*q1)/(w0+w1)
    prob=np.stack([math.comb(n,j)*p**j*(1-p)**(n-j)for j in range(n+1)],axis=1)
    assert np.max(np.abs(prob.sum(1)-1))<1e-12
    return np.sum(prob*root,axis=1)


def scalar_update(q,rewards,masks,transition,p,tau,gamma,bounds):
    value=finite_value(q,p,tau)
    target=rewards+gamma*masks*np.einsum('sat,t->sa',transition,value)
    return np.clip(target,*bounds)


def twin_update(q,rule,*args):
    combined=q.min(0)if rule=='min'else q.mean(0)
    one=scalar_update(combined,*args)
    return np.stack((one,one))


def fixed(initial,update):
    q=initial
    for step in range(20000):
        new=update(q)
        if np.max(np.abs(new-q))<1e-11:return new,step+1
        q=new
    raise AssertionError('Exact operator did not reach the numerical tolerance')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cases=[];rng=np.random.RandomState(72531)
    for seed in range(4):
        for gamma in(.9,.99,.999):
            for tau in(.9,.99):
                s=4;a=2;bounds=(-2/(1-gamma),0.)
                transition=rng.uniform(.1,1,(s,a,s));transition/=transition.sum(2,keepdims=True)
                rewards=-rng.uniform(.1,2,(s,a));masks=np.ones((s,a));rewards[-1]=0;masks[-1]=0
                p=rng.uniform(.05,.95,s);opargs=(rewards,masks,transition,p,tau,gamma,bounds)
                q=rng.uniform(bounds[0],0,(2,s,a));u=rng.uniform(bounds[0],0,(2,s,a))
                for rule in('min','mean'):
                    first=twin_update(q,rule,*opargs)
                    np.testing.assert_array_equal(first[0],first[1])
                    distance=np.max(np.abs(first-twin_update(u,rule,*opargs)))
                    assert distance<=gamma*np.max(np.abs(q-u))+1e-9
                fmin=finite_value(q.min(0),p,tau);fmean=finite_value(q.mean(0),p,tau)
                head_gap=.5*np.abs(q[0]-q[1]);root_gap=fmean-fmin
                assert np.min(root_gap-head_gap.min(1))>=-1e-9
                assert np.max(root_gap-head_gap.max(1))<=1e-9
                scalar,nscalar=fixed(np.zeros((s,a)),lambda x:scalar_update(x,*opargs))
                minimum,nmin=fixed(q,lambda x:twin_update(x,'min',*opargs))
                average,nmean=fixed(u,lambda x:twin_update(x,'mean',*opargs))
                err=max(np.max(np.abs(minimum[0]-scalar)),np.max(np.abs(average[0]-scalar)))
                assert err<1e-8 and np.max(np.abs(scalar[-1]))==0
                cases.append(dict(seed=seed,gamma=gamma,expectile=tau,proposals=9,
                    min_mean_common_scalar_fixed_point_max_abs=float(err),
                    iterations=dict(scalar=nscalar,min=nmin,mean=nmean),
                    maximum_frozen_root_gap=float(root_gap.max()),
                    one_step_head_equality_exact=True,contraction_inequality_passed=True,
                    root_gap_bounded_by_pointwise_half_head_disagreement=True))
    propagation=[]
    for gamma in(.9,.99,.999):
        d=.001;q=0.;perturbed=0.;k=5000
        for _ in range(k):q=-.2+gamma*q;perturbed=-.2+gamma*perturbed+d
        predicted=d*(1-gamma**k)/(1-gamma)
        assert abs((perturbed-q)-predicted)<1e-9
        propagation.append(dict(gamma=gamma,per_update_external_bias=d,steps=k,
            observed_gap=perturbed-q,geometric_sum_gap=predicted,asymptotic_gap=d/(1-gamma)))
    result=dict(time=time.time(),kind='exact_finite_expectile_twin_operator_checks',cases=cases,
        cases_passed=len(cases),proposals=9,exact_binomial_count_enumeration=True,
        all_common_fixed_points_match=True,external_bias_propagation_examples=propagation,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['ExactknownsmallMDPsandfixedbehaviordistributions,notneural/stochasticoptimizationconvergence.',
            'Thesharedfixedpointisforthefinite-Noperator,notnecessarilythepopulationexpectileIQLoperator.',
            'Externalbiasexamplesassumeapersistentaddedbias;actualheadgaplogsarenotoraclebiaslabelsorconstantperturbations.'],goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(cases_passed=len(cases),max_fixed_point_difference=max(row['min_mean_common_scalar_fixed_point_max_abs']for row in cases),external_bias_examples=propagation)),flush=True)


if __name__=='__main__':main()
