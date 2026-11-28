"""Exact expected-loss witness: shared Q noise only, zero V noise.

For action-common centered Gaussian Q error Z, finite/population FEQ obeys
e_tau(Q(A)+Z | Z) = e_tau(Q(A)) + Z. Fitting one persistent IQL V to the
noisy target instead minimizes the loss over BOTH A and Z. These two
operations do not commute. This file solves that expected-loss fixed point
without SGD noise, incomplete V optimization, or finite training time.
It is not asserted to equal the mean stochastic learning trajectory.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from .mdp import chain, solve, roots, evaluate_policy


def expected_score(delta, sigma, tau):
    if sigma == 0:
        return np.where(delta>=0,tau,1-tau)*delta
    z=delta/sigma
    return delta*((1-tau)+(2*tau-1)*ndtr(z))+(2*tau-1)*sigma*np.exp(-z*z/2)/np.sqrt(2*np.pi)


def noisy_root(q,beta,sigma,tau):
    margin=max(1.,10*sigma)
    return brentq(lambda v:np.dot(beta,expected_score(q-v,sigma,tau)),q.min()-margin,q.max()+margin,xtol=1e-14)


def expected_iql(mdp,sigma):
    # Chain is triangular except self-loops; solve one monotone scalar equation
    # per state, from the goal backwards. No iterative Bellman stopping error.
    states=len(mdp.reward);v=np.zeros(states)
    for s in range(states-1,-1,-1):
        def score(x):
            next_v=np.where(mdp.successor[s]==s,x,v[mdp.successor[s]])
            q=mdp.reward[s]+mdp.gamma*mdp.continuation[s]*next_v
            return np.dot(mdp.behavior[s],expected_score(q-x,sigma,mdp.tau))
        v[s]=brentq(score,-1.,2.+10*sigma/(1-mdp.gamma),xtol=1e-13)
    q=mdp.reward+mdp.gamma*mdp.continuation*v[mdp.successor]
    for s in range(states):
        assert abs(np.dot(mdp.behavior[s],expected_score(q[s]-v[s],sigma,mdp.tau)))<1e-11
    return q,v


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    out=args.root/'population';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    cases=[(False,4,.999,.002),(False,12,.999,.001),(False,24,.999,.001),(False,12,.995,.005),
           (True,4,.999,.002),(True,12,.999,.001),(True,12,.999,.002)]
    for plain,length,gamma,sigma in cases:
        mdp=chain(length,.25,gamma,.9,.8)
        if plain:
            mdp.reward[0,1]=0.;mdp.continuation[0,1]=1.
        qt,vt,_=solve(mdp)
        qn,vn,_=solve(mdp,samples=9)
        qi,vi=expected_iql(mdp,sigma)
        residual=vi-roots(qi,mdp.behavior,mdp.tau)
        row=dict(plain=plain,length=length,gamma=gamma,sigma_Q=sigma,sigma_V=0.,
            IQL_Q_RMSE=float(np.sqrt(((qi-qt)**2).mean())),IQL_success=evaluate_policy(mdp,qi)[1],
            FEQ9_Q_RMSE=float(np.sqrt(((qn-qt)**2).mean())),FEQ9_success=evaluate_policy(mdp,qn)[1],
            population_FEQ_success=evaluate_policy(mdp,qt)[1],
            V_residual_RMS=float(np.sqrt((residual**2).mean())),V_residual_max=float(abs(residual).max()),
            Q_error_bias=float((qi-qt).mean()),V_residual_mean=float(residual.mean()),
            Q_readout_RMS=sigma,FEQ_expected_Q_readout_shift=0.)
        rows.append(row)
        np.savez_compressed(out/f'{"plain" if plain else "safe"}_L{length}_g{gamma}_s{sigma}.npz',q_truth=qt,v_truth=vt,iql_q=qi,iql_v=vi,
            residual=residual,feq9_q=qn,feq9_v=vn,behavior=mdp.behavior)
    # A one-state/one-action self-loop gives a closed-form amplification ratio.
    sigma=.001;gamma=.999;tau=.9
    c=noisy_root(np.array([0.]),np.array([1.]),1.,tau)
    witness=dict(normal_expectile=c,Q_readout_RMS=sigma,V_residual=c*sigma,
        IQL_fixed_Q=gamma*c*sigma/(1-gamma),FEQ_expected_fixed_Q=0.,
        equal_scalar_error_control='A constant +d added to every Q action yields the same +d bootstrap error as an additive V residual. Label alone cannot change sensitivity.')
    # Symmetric loss and noiseless cases must agree with the clean operator.
    for tau,sigma in ((.5,.01),(.9,0.)):
        mdp=chain(4,.25,.995,tau,.8);q,v=expected_iql(mdp,sigma);qt,vt,_=solve(mdp)
        np.testing.assert_allclose(q,qt,atol=2e-9)
    payload=dict(status='PASS',rows=rows,one_action_witness=witness,
        interpretation='The learned-V residual can be smaller in RMS than the shared raw Q error yet have larger accumulated harm, because temporal regression creates a systematic bias. Matching bias and post-expectile error would remove this distinction.',
        limitations='Expected-loss fixed points, not exact stochastic SGD means. Q readout field is an explicit approximation-error surrogate. Finite FEQ uses9 actions; population FEQ is a diagnostic oracle.')
    (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
