"""Transparent first discovery sweep: do not pool its seeds into confirmation."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from .mdp import chain, solve, policy_value
from .engine import METHODS, METRICS, run


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];started=time.monotonic()
    for length in (4,12,24):
      for progress in (.1,.25,.5):
       for gamma in (.99,.999):
        mdp=chain(length,progress,gamma,.9,.8)
        qt,vt,_=solve(mdp)
        for seed in (0,1,2):
         for method in (0,1,2):
          out=run(method,mdp.reward,mdp.successor,mdp.continuation,mdp.behavior,mdp.behavior,gamma,.9,100000,.1,.1,.005,seed,qt,vt,log_every=5000)
          steps,logs,q=out[:3]
          for i,t in enumerate(steps):
           if t not in (10000,25000,50000,100000):continue
           policy=policy_value(mdp,out[5][i])
           row=dict(length=length,progress=progress,gamma=gamma,seed=seed,method=METHODS[method],steps=int(t),q_rmse=float(logs[i,0]),v_residual=float(logs[i,2]),return_start=float(policy[0]),optimal_return=gamma**length)
           rows.append(row)
          print(json.dumps(rows[-1]),flush=True)
    with (args.output/'all_discovery_results.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'metadata.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,rows=len(rows),seeds=[0,1,2],confirmation=False),indent=2))
if __name__=='__main__':main()
