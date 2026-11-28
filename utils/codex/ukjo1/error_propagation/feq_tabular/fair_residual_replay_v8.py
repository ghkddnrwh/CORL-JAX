"""Replay endogenous IQL residuals, with state/time/RMS matched controls.

Two injection sites (post-expectile V residual vs action-common Q error)
MUST yield the same trajectory. A mean-removal arm preserves statewise RMS
and normalized temporal autocovariance while removing the signed bias.
This is a diagnostic with a recorded disturbance tape, not a trainable policy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
from pathlib import Path
import numpy as np
from numba import njit
from .mdp import chain,solve,expectile,evaluate_policy
from .behavior_flow import fit,offline_counts
from .fair_engine_v7 import simulate,pick


@njit(cache=True)
def record(reward,successor,continuation,beta,gamma,tau,updates,seed,qt,vt,sigma,log_every):
    states,actions=reward.shape
    np.random.seed(900001+seed)
    visited=np.random.randint(0,states,updates)
    uniforms=np.random.random((updates,10))
    np.random.seed(1900003+seed)
    latent=np.random.standard_normal((states,actions+1))
    q=qt.copy();target=q.copy();value=vt.copy();cdf=beta.copy()
    for s in range(states):
        for a in range(1,actions):cdf[s,a]+=cdf[s,a-1]
    tape=np.zeros((updates,states))
    history=np.zeros((1+updates//log_every,states,actions));history[0]=q
    for t in range(updates):
        if t:latent=np.random.standard_normal(latent.shape)
        for s in range(states):tape[t,s]=value[s]-expectile(target[s],beta[s],tau)
        s=visited[t];a=pick(cdf[s],uniforms[t,0]);sn=successor[s,a]
        old=q[s,a]
        q[s,a]+=.1*(reward[s,a]+gamma*continuation[s,a]*value[sn]-q[s,a])
        delta=target[s,a]+sigma*latent[s,0]-value[s]
        value[s]+=.2*(tau if delta>=0 else 1-tau)*delta
        for ss in range(states):
            for aa in range(actions):
                online=old if (ss==s and aa==a) else q[ss,aa]
                target[ss,aa]=.995*target[ss,aa]+.005*online
        if (t+1)%log_every==0:history[(t+1)//log_every]=q
    return tape,visited,uniforms,history


@njit(cache=True)
def replay(reward,successor,continuation,beta,gamma,tau,qt,tape,visited,uniforms,site,log_every):
    q=qt.copy();target=q.copy();cdf=beta.copy();states,actions=q.shape
    for s in range(states):
        for a in range(1,actions):cdf[s,a]+=cdf[s,a-1]
    history=np.zeros((1+len(visited)//log_every,states,actions));history[0]=q
    for t in range(len(visited)):
        s=visited[t];a=pick(cdf[s],uniforms[t,0]);sn=successor[s,a]
        if site==0:
            bootstrap=expectile(target[sn],beta[sn],tau)+tape[t,sn]
        else:
            bootstrap=expectile(target[sn]+tape[t,sn],beta[sn],tau)
        old=q[s,a]
        q[s,a]+=.1*(reward[s,a]+gamma*continuation[s,a]*bootstrap-q[s,a])
        for ss in range(states):
            for aa in range(actions):
                online=old if (ss==s and aa==a) else q[ss,aa]
                target[ss,aa]=.995*target[ss,aa]+.005*online
        if (t+1)%log_every==0:history[(t+1)//log_every]=q
    return history


def worker(job):
    root,seed,plain=job;root=Path(root);folder=root/f'seed{seed}';folder.mkdir(exist_ok=True)
    m=chain(4,.25,.999,.9,.8)
    if plain:
        m.reward[0,1]=0.;m.continuation[0,1]=1.
    beta=fit(offline_counts(m.behavior,seed));qt,vt,_=solve(m,beta)
    updates=200000;interval=5000;sigma=.002
    tape,visited,uniforms,source=record(m.reward,m.successor,m.continuation,beta,m.gamma,m.tau,
                                     updates,seed,qt,vt,sigma,interval)
    check=simulate(0,m.reward,m.successor,m.continuation,beta,beta,m.gamma,m.tau,
        updates,seed,qt,vt,sigma=sigma,log_every=interval,init_truth=True)
    np.testing.assert_allclose(source,check[1],rtol=0,atol=1e-12)
    mean=tape.mean(0);rms=np.sqrt((tape**2).mean(0))
    centered=tape-mean
    scale=rms/np.sqrt((centered**2).mean(0))
    centered*=scale
    np.testing.assert_allclose(np.sqrt((centered**2).mean(0)),rms,rtol=1e-12)
    assert abs(centered.mean(0)).max()<1e-12
    histories=[];rows=[]
    cases=[('recorded_V',tape,0),('same_error_Q_common',tape,1),
           ('centered_same_RMS',centered,0),('zero_residual',np.zeros_like(tape),0)]
    for name,errors,site in cases:
        history=replay(m.reward,m.successor,m.continuation,beta,m.gamma,m.tau,
                       qt,errors,visited,uniforms,site,interval)
        histories.append(history)
        policy=np.array([evaluate_policy(m,q,beta) for q in history])
        tail=np.arange(len(history))*interval>=.9*updates
        row=dict(seed=seed,arm=name,tail_success=float(policy[tail,1].mean()),
            tail_return=float(policy[tail,0].mean()),
            tail_q_rmse=float(np.sqrt(((history[tail]-qt)**2).mean((1,2))).mean()),
            disturbance_RMS=float(np.sqrt((errors**2).mean())),
            mean_bias_RMS=float(np.sqrt((errors.mean(0)**2).mean())))
        rows.append(row)
    np.testing.assert_allclose(histories[0],source,rtol=0,atol=1e-11)
    np.testing.assert_allclose(histories[0],histories[1],rtol=0,atol=1e-10)
    np.savez_compressed(folder/'diagnostics.npz',residual=tape,mean=mean,rms=rms,center_scale=scale,
        source_q=source,replayed_q=np.asarray(histories),q_truth=qt,v_truth=vt,behavior=beta)
    (folder/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--plain',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    plan=dict(seeds=list(range(650,666) if a.plain else range(600,616)),plain=a.plain,length=4,gamma=.999,tau=.9,sigma_Q=.002,sigma_V=0,
        init='All arrays initialized to known expectile fixed point, isolating perturbation propagation.',
        arms=['Recorded endogenous V residual','Same residual as common Q error','Centered residual rescaled to same per-state RMS','Zero residual'],
        expected_identity='First2 arms must match every Q snapshot; first arm must reconstruct source IQL.',
        limitation='Replay freezes the source disturbance sequence; the mean correction is an oracle diagnostic, not a proposed learning algorithm.')
    (a.root/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,(str(a.root),seed,a.plain)) for seed in plan['seeds']]
        for f in as_completed(futures):rows.extend(f.result())
    with (a.root/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.root/'complete.json').write_text(json.dumps(dict(status='PASS',seeds=16,arms=4,all_identities_verified=True),indent=2)+'\n')


if __name__=='__main__':main()
