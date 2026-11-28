"""User-theory audit: Q output perturbations versus expectile SCORE residuals.

G(q,v)=sum beta*w_tau(q-v)*(q-v), with no loss rescaling. A score
residual zeta is defined by G(q,v)+zeta=0. It is NOT the value gap
v-e_tau(q). Both discrete, away-from-kink witnesses and an atomless
Uniform[0,1] witness are included. Persistent-error MDPs are diagnostic
operators, not ordinary IQL/FEQ training runs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from .mdp import chain, expectile, roots, solve, evaluate_policy


def score(q, v, beta, tau):
    u = q-v
    return np.sum(beta*np.where(u >= 0, tau, 1-tau)*u, axis=-1)


def score_root(q, beta, tau, zeta=0.):
    margin = 1+abs(zeta)/min(tau, 1-tau)
    return brentq(lambda v: score(q, v, beta, tau)+zeta,
                  q.min()-margin, q.max()+margin, xtol=1e-14)


def secant_diagnostics(q, v, beta, tau):
    """Exact zeta=kappa_secant*value_gap, including sign crossings.

    Local kappa at the exact root is a first-order approximation only.
    The secant coefficient integrates slope along the displacement.
    """
    exact = roots(q, beta, tau)
    gap = v-exact
    zeta = -score(q, v[:, None], beta, tau)
    kappa = np.sum(beta*np.where(q >= exact[:, None], tau, 1-tau), -1)
    secant = np.divide(zeta, gap, out=kappa.copy(), where=abs(gap)>1e-12)
    return dict(exact=exact, gap=gap, zeta=zeta, kappa=kappa, secant=secant)


def fixed_forcing(mdp, channel, delta):
    """Exact triangular-chain fixed point, including all nonlinear crossings."""
    v = np.zeros(len(mdp.reward))
    for s in range(len(v)-1, -1, -1):
        def equation(x):
            nv = np.where(mdp.successor[s] == s, x, v[mdp.successor[s]])
            q = mdp.reward[s]+mdp.gamma*mdp.continuation[s]*nv
            return score(q+(delta if channel=='critic' else 0.), x,
                         mdp.behavior[s], mdp.tau)+(delta if channel=='score' else 0.)
        bound = 10+10*abs(delta)/((1-mdp.gamma)*min(mdp.tau,1-mdp.tau))
        v[s] = brentq(equation, -bound, bound, xtol=1e-13)
    q = mdp.reward+mdp.gamma*mdp.continuation*v[mdp.successor]
    residual = score(q+(delta if channel=='critic' else 0.), v[:,None],
                     mdp.behavior,mdp.tau)+(delta if channel=='score' else 0.)
    assert np.max(abs(residual)) < 2e-11
    if channel=='critic':
        np.testing.assert_allclose(v,roots(q,mdp.behavior,mdp.tau)+delta,atol=2e-11)
    return q,v


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/'channels';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for tau in (.5,.9,.99,.999):
        delta=1e-7
        v=np.sqrt(tau)/(np.sqrt(tau)+np.sqrt(1-tau))
        kappa=(1-tau)+(2*tau-1)*(1-v)
        uniform_score=lambda x: .5*(tau*(1-x)**2-(1-tau)*x*x)
        vr=brentq(lambda x:uniform_score(x)+delta,0.,1.,xtol=1e-14)
        rows.append(dict(distribution='uniform_atomless',p=-1.,tau=tau,
                         kappa=kappa,critic_gain=1.,score_gain=(vr-v)/delta,
                         predicted=1/kappa,delta=delta))
        for p in (.25,.1,.01):
            q=np.array([0.,1.]);beta=np.array([1-p,p]);v=expectile(q,beta,tau)
            kappa=(1-tau)*(1-p)+tau*p
            vc=expectile(q+delta,beta,tau);vr=score_root(q,beta,tau,delta)
            np.testing.assert_allclose((vr-v)/delta,1/kappa,rtol=1e-7)
            np.testing.assert_allclose((vc-v)/delta,1.,rtol=1e-7)
            rows.append(dict(distribution='two_mass_discrete',p=p,tau=tau,
                             kappa=kappa,critic_gain=(vc-v)/delta,
                             score_gain=(vr-v)/delta,predicted=1/kappa,delta=delta))
    write_csv(out/'local_sensitivity.csv',rows)
    settings=[]
    for tau,p in ((.9,.25),(.99,.25),(.99,.05)):
        for gamma in (.995,.999):
            m=chain(4,p,gamma,tau,.8);m.reward[0,1]=0.;m.continuation[0,1]=1.
            qt,vt,_=solve(m)
            for d in (0.,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3):
                for sign in ((1,) if d==0 else (-1,1)):
                    delta=sign*d
                    for channel in ('critic','score'):
                        q,v=fixed_forcing(m,channel,delta)
                        diagnostics=secant_diagnostics(q,v,m.behavior,tau)
                        np.testing.assert_allclose(diagnostics['gap']*diagnostics['secant'],diagnostics['zeta'],atol=1e-12)
                        ret,success=evaluate_policy(m,q)
                        row=dict(tau=tau,p=p,gamma=gamma,delta=delta,channel=channel,
                                 q_rmse=float(np.sqrt(((q-qt)**2).mean())),
                                 q_max_error=float(abs(q-qt).max()),success=success,discounted_return=ret,
                                 value_gap_rms=float(np.sqrt((diagnostics['gap']**2).mean())),
                                 score_residual_rms=float(np.sqrt((diagnostics['zeta']**2).mean())),
                                 min_local_kappa=float(diagnostics['kappa'].min()),
                                 min_secant_kappa=float(diagnostics['secant'].min()))
                        settings.append(row)
    write_csv(out/'mdp_forcing.csv',settings)
    # Arbitrary finite vectors: equal value displacement must be equivalent.
    rng=np.random.default_rng(900)
    for _ in range(100):
        q=rng.normal(size=4);b=rng.dirichlet(np.ones(4));tau=rng.uniform(.5,.999)
        d=rng.uniform(-.03,.03);v=expectile(q,b,tau)
        z=-score(q,v+d,b,tau)
        np.testing.assert_allclose(score_root(q,b,tau,z),expectile(q+d,b,tau),atol=2e-12)
    (out/'audit.json').write_text(json.dumps(dict(status='PASS',local_cases=len(rows),
        operator_cases=len(settings),equal_value_gap_controls=100,
        normalization='G=sum beta*w*(q-v); J_prime=-2G; zeta=-G.',
        distinction='Equal score residual and Q output error are distinct primitive coordinates. Equal post-expectile value disturbances are equivalent.',
        finite_actions='Local derivative holds off action-value kinks. The atomless tau->1 divergence is illustrated separately; a finite-action max atom can prevent kappa->0.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for distribution,p,label in [('uniform_atomless',-1.,'Uniform [0,1]'),('two_mass_discrete',.01,'Rare high action (p=.01)')]:
        r=[x for x in rows if x['distribution']==distribution and x['p']==p]
        axes[0].plot([x['tau'] for x in r],[x['score_gain'] for x in r],'o-',label=label)
    axes[0].axhline(1,color='black',linestyle='--',label='Q-output gain = 1')
    axes[0].set(xlabel='Expectile level',ylabel='Local value displacement\n/ primitive error',yscale='log')
    axes[0].legend(fontsize=8)
    for channel,label in [('critic','Q output error'),('score','Expectile score residual')]:
        r=[x for x in settings if x['tau']==.99 and x['p']==.25 and x['gamma']==.999 and x['delta']>0 and x['channel']==channel]
        axes[1].plot([x['delta'] for x in r],[100*x['success'] for x in r],'o-',label=label)
    axes[1].set(xlabel='Positive primitive error (same number)',ylabel='Goal success (%)',xscale='log',ylim=(-3,103))
    axes[1].legend(fontsize=8)
    for a in axes:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'theory_channels.png',dpi=180);fig.savefig(out/'theory_channels.pdf');plt.close(fig)
    print(json.dumps(dict(status='PASS',local_cases=len(rows),operator_cases=len(settings))))


if __name__=='__main__':main()
