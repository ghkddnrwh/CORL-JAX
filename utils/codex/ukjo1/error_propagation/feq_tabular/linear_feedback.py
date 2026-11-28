"""Nominal expected-update Jacobians; distinguish V feedback from Q recursion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from .mdp import chain,roots


def jacobians(mdp,q,behavior,aq=.1,av=.1,ema=.005):
    s,a=q.shape;m=s*a;state_indices=np.arange(s)
    v=roots(q,behavior,mdp.tau);weights=np.where(q>=v[:,None],mdp.tau,1-mdp.tau)*behavior
    h=np.zeros((s,m));hn=h.copy()
    for ss in range(s):h[ss,ss*a:(ss+1)*a]=weights[ss];hn[ss]=h[ss]/weights[ss].sum()
    transition=np.zeros((m,s))
    for ss in range(s):
        for aa in range(a):transition[ss*a+aa,mdp.successor[ss,aa]]=mdp.continuation[ss,aa]
    d=np.diag(behavior.ravel()/s);identity=np.eye(m)
    iql=np.zeros((2*m+s,2*m+s));feq=np.zeros((2*m,2*m))
    iql[:m,:m]=identity-aq*d;iql[:m,m:m+s]=aq*d@transition*mdp.gamma
    iql[m:m+s,m:m+s]=np.eye(s)-2*av/s*np.diag(weights.sum(-1));iql[m:m+s,m+s:]=2*av/s*h
    iql[m+s:,:m]=ema*identity;iql[m+s:,m+s:]=(1-ema)*identity
    feq[:m,:m]=identity-aq*d;feq[:m,m:]=aq*d@transition@hn*mdp.gamma
    feq[m:,:m]=ema*identity;feq[m:,m:]=(1-ema)*identity
    return iql,feq


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();rows=[]
    for case,length in (('main_iid',12),('main_field',12),('main_clean_long',24)):
        with np.load(r/'raw'/case/'seed100/iql.npz') as z:
            mdp=chain(length,.25,.999,.9,.8);beta=z['data_behavior']
            for point,q in (('truth',z['q_tau_true']),('learned_iql',z['q'])):
                iq,fq=jacobians(mdp,q,beta)
                rows.append(dict(case=case,point=point,IQL_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(iq)))),
                    population_FEQ_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(fq)))),
                    IQL_state_dimension=len(iq),FEQ_state_dimension=len(fq)))
    payload=dict(scope='Noiseless local mean-update linearization, including old-online target EMA; not a stochastic stability proof.',
                 rows=rows,feedback='IQL has Q_target->V->Q->Q_target. FEQ has Q_target->exact_expectile->Q->Q_target.',
                 behavior='Behavior counts depend only on offline actions. NoQ->behavior gradient exists; IQL can also fit an independent behavior model.',
                 equal_target_error='At equal additive bootstrap-value perturbation delta, gamma P delta is identical regardless of whether delta is called V residual or model error.')
    (r/'analysis/linear_feedback.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
