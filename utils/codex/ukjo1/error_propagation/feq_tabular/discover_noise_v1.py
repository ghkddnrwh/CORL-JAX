"""Noise discovery; preserve every condition, including FEQ failures."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv,json,time
from pathlib import Path
import numpy as np
from .engine import run,METHODS
from .mdp import chain,solve,policy_value


def job(args):
    mode,sq,sv,seed=args
    m=chain(12,.25,.999,.9,.8);qt,vt,_=solve(m)
    rows=[]
    for method in (0,1,2):
        out=run(method,m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.gamma,m.tau,200000,.1,.1,.005,seed,qt,vt,sq,sv,.99 if mode=='field' else 0.,mode=='field',log_every=10000)
        q=out[2];v=policy_value(m,q)
        rows.append(dict(mode=mode,sigma_q=sq,sigma_v=sv,seed=seed,method=METHODS[method],q_rmse=float(out[1][-1,0]),q_bias=float(out[1][-1,4]),v_residual=float(out[1][-1,2]),return_start=float(v[0]),finite=bool(np.isfinite(q).all())))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[(mode,sq,sv,seed) for mode in ('iid','field') for sq,sv in ((0.,0.),(0.,.005),(.005,0.),(.005,.005),(.005,.02),(.02,.005),(.02,.02),(.05,.05)) for seed in (3,4,5)]
    before=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for group in pool.map(job,cases):
            rows+=group;print(json.dumps(group),flush=True)
    with (args.output/'all_discovery_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'metadata.json').write_text(json.dumps(dict(seconds=time.monotonic()-before,seeds=[3,4,5],cases=len(cases),confirmation=False),indent=2))
if __name__=='__main__':main()
