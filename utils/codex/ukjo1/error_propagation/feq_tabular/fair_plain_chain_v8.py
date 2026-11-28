"""Additional MDP: progress or wait, with only a unit terminal goal reward.

No safe-exit reward derived from the true value. Register the complete grid
before execution; all fresh-seed outcomes, including failures, are retained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,json
from pathlib import Path
import numpy as np
from .mdp import chain,solve,evaluate_policy
from .behavior_flow import fit,offline_counts
from .fair_engine_v7 import simulate,METHODS
from .fair_study_v7 import sha,write_json


def worker(job):
    root,length,sigma,seed=job;root=Path(root)
    m=chain(length,.25,.999,.9,.8)
    m.reward[0,1]=0.;m.continuation[0,1]=1.
    beta=fit(offline_counts(m.behavior,seed));qt,vt,_=solve(m,beta)
    folder=root/'raw'/f'L{length}_s{sigma}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in ('iql','iql_batch9','feq9'):
        out=simulate(METHODS.index(method),m.reward,m.successor,m.continuation,beta,beta,
            m.gamma,m.tau,200000,seed,qt,vt,sigma=sigma,log_every=5000)
        times,qhist,vhist,thist,parts,cross,residual,noise,worst=out
        assert worst<1e-10 and np.isfinite(qhist).all()
        policy=np.array([evaluate_policy(m,q,beta) for q in qhist]);tail=times>=180000
        rmse=np.sqrt(((qhist-qt)**2).mean((1,2)))
        row=dict(length=length,sigma=sigma,seed=seed,method=method,
            tail_success=float(policy[tail,1].mean()),tail_return=float(policy[tail,0].mean()),
            tail_q_rmse=float(rmse[tail].mean()),noise_rms=float(np.sqrt(noise[1:,1].mean())))
        path=folder/f'{method}.npz'
        np.savez_compressed(path,step=times,q=qhist,value=vhist,target=thist,decomposition=parts,
            cross=cross,v_residual=residual,policy=policy,q_truth=qt,v_truth=vt,behavior=beta,
            reward=m.reward,successor=m.successor,continuation=m.continuation)
        write_json(path.with_suffix('.json'),dict(metrics=row,sha256=sha(path)))
        rows.append(row)
    print(json.dumps(rows),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    folder=Path(__file__).parent
    plan=dict(lengths=[4,12],sigmas=[0.,.001,.002,.003],seeds=list(range(500,516)),
        methods=['iql','iql_batch9','feq9'],steps=200000,gamma=.999,tau=.9,extra_V_noise=0,
        MDP='L+1 nonterminal states,4 actions; action0 advances or terminates with reward1; all other actions wait with reward0. Uniform true behavior;1024 offline draws/state.',
        sources={str(folder/f):sha(folder/f) for f in ('fair_plain_chain_v8.py','fair_engine_v7.py','mdp.py','behavior_flow.py')})
    write_json(a.root/'registration.json',plan)
    jobs=[(str(a.root),L,s,seed) for L in plan['lengths'] for s in plan['sigmas'] for seed in plan['seeds']]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(pending):rows.extend(f.result())
    with (a.root/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(a.root/'complete.json',dict(status='complete',trajectories=len(rows)))


if __name__=='__main__':main()
