"""Held-out expectile-level sweep for the fully shared-feature witness.

Retains all tested levels, including settings where success or global Q RMSE
does not rank the methods in the same way as discounted return.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .persistent_matched_features_v10 import setup,evaluate
from .persistent_features_v10 import simulate,METHODS,references
from .fair_study_v7 import sha,write_json
from .mdp import MDP


def worker(job):
    root,c,seed,sources=job;root=Path(root);identity=dict(case=c,seed=seed)
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'raw'/token;out.mkdir(parents=True,exist_ok=True);meta=out/'result.json';path=out/'data.npz'
    if meta.exists():
        record=json.loads(meta.read_text());assert record['identity']==identity and record['sources']==sources and record['sha256']==sha(path)
        return record['metrics']
    m,phi,expanded,ds,da,occ,model,true_beta,theta=setup(seed);m.tau=c['tau']
    ref=references(m,phi,occ,model)
    mm=MDP(m.reward,m.successor,m.continuation,model,m.gamma,m.tau,m.name)
    model_ref=references(mm,phi,occ,model)
    beta=model if c['method']=='iql9' else m.behavior
    t,q,target,v=simulate(METHODS.index(c['method']),m.reward,m.successor,beta,model,phi,expanded,
                         ds,da,m.tau,m.gamma,c['steps'],seed)
    policy=np.array([evaluate(m,qq,model) for qq in q]);tail=t>=.9*c['steps']
    rmse=np.sqrt(((q-ref['q_truth'])**2).mean((1,2)))
    row=dict(case=c['name'],method=c['method'],seed=seed,steps=c['steps'],tau=m.tau,
        success=float(policy[tail,1].mean()),deployed_success=float(policy[tail,1].mean()),
        discounted_return=float(policy[tail,0].mean()),q_rmse=float(rmse[tail].mean()),
        terminal_q_rmse=float(np.sqrt(((q[tail,1:]-m.reward[None,1:])**2).mean())),
        model_tv=float(abs(model-m.behavior).sum(-1).mean()/2))
    np.savez_compressed(path,step=t,q=q,target=target,value=v,policy=policy,model_policy=policy,rmse=rmse,
        reward=m.reward,successor=m.successor,continuation=m.continuation,behavior=m.behavior,model=model,
        true_behavior=true_beta,phi=phi,behavior_parameters=theta,offline_states=ds,offline_actions=da,
        occupancy=occ,q_iql9=model_ref['q_iql'],**ref)
    write_json(meta,dict(identity=identity,sources=sources,sha256=sha(path),metrics=row))
    print(json.dumps(row),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);folder=Path(__file__).parent
    names=('sweep_matched_v10.py','persistent_matched_features_v10.py','persistent_features_v10.py',
           'mdp.py','theory_channels_v9.py','noise_models_v5.py','fair_study_v7.py')
    sources={str(folder/f):sha(folder/f) for f in names}
    cases=[dict(name=f'tau{tau:g}',tau=tau,method=method,steps=1000000,seeds=list(range(1700,1708)))
           for tau in (.9,.95,.98,.99) for method in ('iql','iql9','feq9')]
    # Analytic exploratory sweep, all values recorded before the held-out seeds run.
    m,phi,ex,ds,da,occ,model,tb,theta=setup(1500)
    m.behavior=tb;occ=np.array([.5,.125,.25,.125]);analytic=[]
    for tau in (.9,.93,.95,.96,.97,.98,.99):
        m.tau=tau;r=references(m,phi,occ,tb)
        for method,key in (('iql','q_iql'),('feq9','q_feq9')):
            ret,suc=evaluate(m,r[key],tb)
            analytic.append(dict(tau=tau,method=method,discounted_return=ret,success=suc,
                q_rmse=float(np.sqrt(((r[key]-r['q_truth'])**2).mean()))))
    reg=dict(cases=cases,sources=sources,analytic_sweep=analytic,
        origin='Follow-up to matched_features: policy return advantage persisted, but FEQ Q_tau RMSE was worse at tau=.99. Sweep retains this reversal.',
        hypothesis='Q_tau RMSE and policy success need not order methods identically; sweep tau without selecting seeds.',
        parameters='Same 3-coefficient behavior and V classes, same fixed offline buffer and deployment sampler; no external noise.')
    f=a.root/'registration.json'
    if f.exists():assert json.loads(f.read_text())==reg
    else:write_json(f,reg)
    jobs=[(str(a.root),c,s,sources) for c in cases for s in c['seeds']];start=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            f.result();done+=1;write_json(a.root/'progress.json',dict(completed=done,total=len(jobs),seconds=time.monotonic()-start))
    write_json(a.root/'complete.json',dict(status='complete',runs=done,seconds=time.monotonic()-start))


if __name__=='__main__':main()
