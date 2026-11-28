"""Independent identities needed for the fair noise experiment."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from .mdp import chain,solve,expectile
from .fair_engine_v7 import simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    checks=[];rng=np.random.default_rng(733)
    for _ in range(100):
        q=rng.normal(size=4);b=rng.dirichlet(np.ones(4));d=rng.normal();tau=rng.uniform(.05,.95)
        assert abs(expectile(q+d,b,tau)-expectile(q,b,tau)-d)<1e-12
        # Any scalar error after expectile is EXACTLY a common-mode critic error.
        counts=rng.multinomial(9,b).astype(float)
        assert abs(expectile(q+d,counts,tau)-expectile(q,counts,tau)-d)<1e-12
    checks.append('Population and finite-sample translation equivariance; identical scalar Q/V errors are equivalent.')
    m=chain(3,.25,.995,.9,.8);q,v,_=solve(m)
    positional=(m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.gamma,m.tau,3000,45,q,v)
    results=[simulate(k,*positional,sigma=.003,log_every=1000) for k in (0,1,2,3,4,5,6,7)]
    for r in results:
        np.testing.assert_array_equal(r[7],results[0][7])
        assert r[-1]<1e-10
        np.testing.assert_allclose(r[5][1:].sum((1,2)),r[4][1:,-1,1],atol=1e-11)
    np.testing.assert_array_equal(results[4][1],results[6][1])
    checks.append('All8 methods consume bit-identical Q noise; zero residual repair equals population FEQ bitwise.')
    checks.append('Per-update decomposition and second moments/cross terms reconstruct actual TD target error.')
    # With action-common noise the post-expectile Q error does not have positive expectation.
    assert abs(results[3][4][1:,1,0].mean())<.0002
    clean=simulate(4,*positional,sigma=0.,init_truth=True,log_every=1000)
    np.testing.assert_allclose(clean[1],np.broadcast_to(q,clean[1].shape),atol=1e-10)
    frozen=simulate(0,*positional,sigma=.003,init_truth=True,frozen=True,log_every=1000)
    np.testing.assert_array_equal(frozen[1],np.broadcast_to(q,frozen[1].shape))
    checks.append('Exact fixed point is invariant under noiseless population backup; frozen-Q intervention stays fixed.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=checks)))


if __name__=='__main__':main()
