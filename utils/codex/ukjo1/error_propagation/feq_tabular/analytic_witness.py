"""Exact population-loss witness isolating the extra V approximation-error path.

One state, one action, reward0, deterministic self-loop. This is an analytic
mechanism appendix, not the non-degenerate goal-task performance benchmark.
Independent centered Gaussian Q/V readout errors define a surrogate expected
loss. Ordinary IQL without such errors has the same zero fixed point as FEQ.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,math
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.special import ndtr


def score(z,tau):
    phi=np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return (2*tau-1)*phi-z*(tau*(1-ndtr(z))+(1-tau)*ndtr(z))


@njit(cache=True)
def integrate(gamma,tau,sigma,steps):
    q=v=target=0.;curve=np.zeros((1+steps//1000,4));slot=0
    for t in range(steps):
        old=q
        q+=.1*(gamma*v-q)
        z=(v-target)/sigma
        phi=math.exp(-z*z/2)/math.sqrt(2*math.pi)
        cdf=.5*(1+math.erf(z/math.sqrt(2)))
        gradient=sigma*((2*tau-1)*phi-z*(tau*(1-cdf)+(1-tau)*cdf))
        v+=.2*gradient
        target=.995*target+.005*old
        if (t+1)%1000==0:
            slot+=1;curve[slot]=np.array([t+1,q,v,target])
    return curve


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();out=r/'analysis';out.mkdir(exist_ok=True)
    tau=.9;sigma_q=sigma_v=.001;sigma=math.hypot(sigma_q,sigma_v)
    e=brentq(lambda z:score(z,tau),-8,8,xtol=1e-14)
    rows=[]
    for gamma in (.9,.99,.995,.999):
        eps=sigma*e;expected_q=gamma*eps/(1-gamma)
        steps=5000000
        trace=integrate(gamma,tau,sigma,steps)
        # Exact zero is preserved by the FEQ population mean update Q<-gamma Q.
        assert abs(trace[-1,1]-expected_q)<max(.01*expected_q,1e-10)
        rows.append(dict(gamma=gamma,value_residual=eps,IQL_exact_fixed_point=expected_q,
                         IQL_after1M_expected_gradient_updates=float(trace[1000,1]),
                         IQL_after5M_expected_gradient_updates=float(trace[-1,1]),
                         FEQ_exact_fixed_point=0.))
        np.savez_compressed(out/f'analytic_population_g{gamma:g}.npz',IQL_trace=trace,FEQ_Q=np.zeros(len(trace)))
    # Symmetric-loss control removes the effect.
    assert abs(brentq(lambda z:score(z,.5),-8,8))<1e-12
    payload=dict(status='PASS',normal_expectile=e,sigma_q=sigma_q,sigma_v=sigma_v,tau=tau,rows=rows,
        assumptions=['Deterministic expected-gradient dynamics with independent mean-zero Gaussian readout errors; this is not the mean trajectory of the nonlinear stochastic algorithm.',
                     'V loss observes targetQ+xiQ-(V+xiV); Q target observes V+xiV.',
                     'One action means FEQ evaluates the same noisy Q function; candidate repeats do not resample noise.',
                     'No reward, support, policy-improvement or flow-model comparison is claimed by this one-state appendix.'],
        derivation=['V=Q_target+e_tau(N(0,sigma_Q^2+sigma_V^2)) at expected V-loss optimum.',
                    'Q=gamma V and Q_target=Q at fixed point.',
                    'Thus Q_IQL=gamma/(1-gamma)*e_tau(N(0,sigma_Q^2+sigma_V^2)); FEQ mean fixed point0.'])
    (out/'analytic_witness.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
