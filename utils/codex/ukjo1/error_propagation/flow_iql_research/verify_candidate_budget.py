"""Check a sufficient candidate budget and its residual/alignment connection.

IDQL supplies the expectile sign-weight policy. The CDF bound below is a simple
derived sufficient condition, not a claim of a new RL optimality theorem.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def expectile(q,p,tau):
    lo,hi=float(q.min()),float(q.max())
    for _ in range(80):
        mid=(lo+hi)/2
        score=np.sum(p*np.where(q>=mid,tau,1-tau)*(q-mid))
        if score>0:lo=mid
        else:hi=mid
    return (lo+hi)/2


def maximum_mean(q,p,n):
    order=np.argsort(q);q=q[order];cdf=np.cumsum(p[order]);cdf[-1]=1.
    mass=cdf**n-np.concatenate([[0.],cdf[:-1]**n])
    return float(np.dot(q,mass))


def main():
    rng=np.random.RandomState(9423);checks=[]
    for tau in (.5,.9,.99):
        ratio=tau/(1-tau);budget=int(np.ceil(ratio-1e-12))
        min_root_margin=float('inf');min_residual_margin=float('inf');identity_error=0.
        for _ in range(1000):
            k=rng.randint(2,20);q=rng.randn(k)*100-300;p=rng.dirichlet(np.full(k,.3))
            e=expectile(q,p,tau);v=e+rng.randn()*100
            weights=np.where(q>=v,tau,1-tau);kappa=np.dot(p,weights)
            score=np.dot(p,weights*(q-v));policy_mean=np.dot(p,weights*q)/kappa
            best=maximum_mean(q,p,budget)
            min_root_margin=min(min_root_margin,best-e)
            min_residual_margin=min(min_residual_margin,best-v-score/kappa)
            identity_error=max(identity_error,abs(policy_mean-v-score/kappa))
        assert min_root_margin>=-1e-8 and min_residual_margin>=-1e-8 and identity_error<1e-8
        grid=np.linspace(0,1,100001)
        envelope=grid/(grid+ratio*(1-grid))
        cdf_violation=float(np.max(grid**budget-envelope))
        assert cdf_violation<1e-12
        checks.append(dict(tau=tau,weight_ratio=ratio,sufficient_integer_budget=budget,
                           random_distributions=1000,minimum_expected_max_minus_expectile=min_root_margin,
                           minimum_expected_max_minus_V_minus_G_over_kappa=min_residual_margin,
                           alignment_identity_max_error=identity_error,cdf_grid_max_violation=cdf_violation))
    p=np.array([.999,.001]);q=np.array([0.,1.]);e=expectile(q,p,.99)
    counter=dict(tau=.99,rare_high_action_probability=.001,expectile=e,
                 expected_best32=maximum_mean(q,p,32),expected_best128=maximum_mean(q,p,128))
    assert counter['expected_best32']<e<counter['expected_best128']
    result=dict(time=time.time(),passed=True,checks=checks,high_tau_insufficient_budget_counterexample=counter,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=dict(title='IDQL Theorem4.1 andEq9: expectile sign-weight policy',url='https://arxiv.org/pdf/2304.10573'),
        derivation=[
            'For any fixed proposal distribution mu and positive weights in[a,b], letR=b/a. The normalized weighted policy has CDF at t at least p/(p+R(1-p)), where p=F_mu(t).',
            'The CDF of the best ofN iidQ draws is p^N. IfN>=R, p^(N-1)*(R-(R-1)*p)<=1, since its derivative is nonnegative on[0,1] and its value at1 is1.',
            'Therefore best-ofN Q first-order stochastically dominates the weighted Q distribution forN>=R.',
            'For expectile sign weights, R=tau/(1-tau). At any learnedV, the weightedpolicy meanQ minusV equalsG(V)/kappa(V) exactly; no optimalV assumption is needed for this algebraic identity.',
            'Consequently E_mu[max_N Q]-V >= G(V)/kappa(V). At an exact conditional expectileG=0, this is a predicted-value policy-improvement condition.',
            'The sufficient thresholds are9 for tau.9 and99 for tau.99. N32 satisfies the former; N128 satisfies the latter. This bound is sufficient, not necessary.'],
        limitations=[
            'This result concerns a fixed Q and fixed proposal distribution; it does not prove accuracy ofQ, fidelity of a learned behavior model or neural training contraction.',
            'The actual IQL value fit uses targetQ, while deployment selects onlineQ; residuals must use the sameQ in each identity.',
            'If learnedV is overoptimistic andG<0, the lower bound can be negative even when the candidate budget is sufficient.',
            'IncreasingN can expose more critic estimation errors; no unconditional real-return improvement follows.',
            'Finite generated-action diagnostics approximate the unknown behavior distribution. Their zero empirical residual is not a true conditional oracle.'])
    out=Path('logs/diagnosis/flow_iql_research_20261026/analysis/candidate_budget_derivation_and_checks.json')
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(passed=True,checks=checks,counterexample=counter),indent=2))


if __name__=='__main__':main()
