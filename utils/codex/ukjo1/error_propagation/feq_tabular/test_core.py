"""Numerical tests for reference truth, exact root, coupling and decomposition."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
from itertools import product
import numpy as np
from scipy.optimize import brentq
from .mdp import chain, solve, expectile, finite_roots, multinomial_counts, best_of_n_policy, evaluate_policy
from .engine import run as old_run
from .engine_v2 import run
from .behavior_flow import fit, offline_counts, transport, region


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
 rng=np.random.default_rng(174)
 for _ in range(200):
  q=rng.normal(size=4);prob=rng.dirichlet(np.ones(4));tau=rng.uniform(.05,.99)
  score=lambda v:np.sum(prob*np.where(q-v>=0,tau,1-tau)*(q-v))
  root=expectile(q,prob,tau);oracle=brentq(score,q.min()-1,q.max()+1,xtol=1e-14)
  assert abs(root-oracle)<1e-11 and abs(score(root))<1e-12
 checks.append('200 weighted roots match independent Brent solver')
 q=np.array([[0.,1.]]);prob=np.array([[.8,.2]]);counts,c=multinomial_counts(9,2)
 expected=0.
 for draws in product((0,1),repeat=9):
  pdraw=np.prod(prob[0,list(draws)]);expected+=pdraw*expectile(q[0],np.bincount(draws,minlength=2).astype(float),.9)
 assert abs(finite_roots(q,prob,.9,counts,c)[0]-expected)<1e-12
 checks.append('finite-N reference equals exhaustive 2^9 enumeration')
 m=chain(4,.25,.99,.9,.8);qt,vt,_=solve(m)
 common=(m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.gamma,m.tau,2000,.1,.1,.005,17,qt,vt)
 for method in (0,1,2):
  before=old_run(method,*common,sigma_q=.001,sigma_v=.001,spatial=True,rho=.99,log_every=500)
  after=run(method,*common,sigma_q=.001,sigma_v=.001,spatial=True,rho=.99,log_every=500,reference_behavior=m.behavior)
  for k in (0,1,2,3,4,5,6):np.testing.assert_array_equal(before[k],after[k])
  assert after[-1]<1e-10
  # Covariance terms are retained: component RMS values must not simply be added.
  np.testing.assert_allclose(after[8][1:].sum((1,2)),after[7][1:,-1,1]**2,atol=1e-11)
 checks.append('v2 diagnostics preserve v1 Q/V trajectories bitwise; exact decomposition')
 a2=run(2,*common,log_every=500,reference_behavior=m.behavior)
 repair=run(6,*common,log_every=500,reference_behavior=m.behavior)
 np.testing.assert_array_equal(a2[2],repair[2]);np.testing.assert_array_equal(a2[5],repair[5])
 checks.append('exact V repair matches population FEQ Q trajectory bitwise')
 frozen=run(0,*common,log_every=500,reference_behavior=m.behavior,freeze_q=True,init_truth=True)
 np.testing.assert_array_equal(frozen[2],qt)
 checks.append('frozen-Q intervention does not update Q')
 probs=fit(offline_counts(m.behavior,5));z=rng.normal(size=100000)
 for s in range(len(probs)):
  bins=region(transport(probs[s],z));estimated=np.bincount(bins,minlength=4)/len(z)
  assert np.max(np.abs(estimated-probs[s]))<.008
 checks.append('Gaussian-to-action monotone flow realizes learned behavior probabilities')
 for _ in range(3):
  q=rng.normal(size=qt.shape);pi=best_of_n_policy(q,m.behavior,9)
  trans=np.zeros((len(q),len(q)))
  for ss in range(len(q)):
   for aa in range(4):trans[ss,m.successor[ss,aa]]+=pi[ss,aa]*m.continuation[ss,aa]
  goal=np.sum(pi*((m.reward==1)&(m.continuation==0)),axis=-1)
  exact=np.zeros(len(q))
  for _ in range(1000):exact=goal+trans@exact
  _,fast=evaluate_policy(m,q,m.behavior)
  assert abs(fast-exact[0])<1e-11
 checks.append('exact best9 success via matrix power matches1000-step recursion')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASS',checks=checks,source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py') if p.name in ('engine.py','engine_v2.py','mdp.py','behavior_flow.py')}),indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=checks)),flush=True)
if __name__=='__main__':main()
