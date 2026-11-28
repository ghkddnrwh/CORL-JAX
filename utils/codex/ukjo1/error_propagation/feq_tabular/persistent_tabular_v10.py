"""Long-run full-tabular study: identical persistent Q errors, no V noise."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from .mdp import chain,solve,evaluate_policy
from .behavior_flow import fit,offline_counts
from .noise_models_v5 import behavior_model
from .fair_engine_v7 import simulate,METHODS
from .fair_study_v7 import sha,write_json


def plan():
    cases=[]
    for method in ('iql','iql_batch9','feq9','repair_zero'):
        cases.append(dict(name='main',method=method,steps=2000000,seeds=list(range(1100,1116))))
    for name,extra in [('annealed',dict(anneal=True)),('noiseless',dict(sigma=0.)),
                       ('mixed_Q_error',dict(common=.5)),('independent_Q_error',dict(common=0.))]:
        for method in ('iql','iql_batch9','feq9'):
            cases.append(dict(name=name,method=method,steps=2000000,seeds=list(range(1100,1108)),**extra))
    cases.append(dict(name='behavior_misspec',method='feq9',steps=2000000,seeds=list(range(1100,1108)),model_strength=.5))
    for method in ('iql','iql_batch9','feq9'):
        cases.append(dict(name='five_million',method=method,steps=5000000,seeds=list(range(1200,1204))))
    return cases


def worker(job):
    root,c,seed,sources=job;root=Path(root)
    identity=dict(case=c,seed=seed)
    name=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'raw'/name;out.mkdir(parents=True,exist_ok=True)
    meta=out/'result.json';path=out/'data.npz'
    if meta.exists():
        m=json.loads(meta.read_text());assert m['identity']==identity and m['sources']==sources
        assert m['sha256']==sha(path);return m['metrics']
    m=chain(4,.25,.999,.9,.8);m.reward[0,1]=0.;m.continuation[0,1]=1.
    beta=fit(offline_counts(m.behavior,seed));model=behavior_model(beta,'logit',c.get('model_strength',0.),seed)
    qt,vt,_=solve(m,beta)
    results=simulate(METHODS.index(c['method']),m.reward,m.successor,m.continuation,beta,model,
        m.gamma,m.tau,c['steps'],seed,qt,vt,sigma=c.get('sigma',.002),common=c.get('common',1.),
        rho=0.,alpha_q=.1,alpha_v=.1,ema=.005,log_every=25000,anneal=c.get('anneal',False))
    t,q,v,target,parts,cross,residual,noise,worst=results
    assert worst<1e-10 and np.isfinite(q).all()
    np.testing.assert_allclose(cross[1:].sum((1,2)),parts[1:,-1,1],atol=1e-10)
    policy=np.array([evaluate_policy(m,x,beta) for x in q])
    deployed=np.array([evaluate_policy(m,x,model) for x in q])
    rmse=np.sqrt(((q-qt)**2).mean((1,2)));tail=t>=.9*c['steps']
    row=dict(case=c['name'],method=c['method'],seed=seed,steps=c['steps'],
             success=float(policy[tail,1].mean()),deployed_success=float(deployed[tail,1].mean()),
             q_rmse=float(rmse[tail].mean()),q_bias=float((q[tail]-qt).mean()),
             model_tv=float(abs(model-beta).sum(-1).mean()/2))
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,step=t,q=q,value=v,target=target,decomposition=parts,cross=cross,
        v_residual=residual,noise_statistics=noise,policy=policy,model_policy=deployed,rmse=rmse,
        q_truth=qt,v_truth=vt,behavior=beta,model=model,reward=m.reward,successor=m.successor,continuation=m.continuation)
    tmp.replace(path);write_json(meta,dict(identity=identity,sources=sources,sha256=sha(path),metrics=row))
    print(json.dumps(row),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);folder=Path(__file__).parent
    sources={str(folder/f):sha(folder/f) for f in ('persistent_tabular_v10.py','fair_engine_v7.py','mdp.py','behavior_flow.py','noise_models_v5.py','fair_study_v7.py')}
    cases=plan();registration=dict(cases=cases,sources=sources,gamma=.999,tau=.9,N=9,
        MDP='5 nonterminal states; action0 progresses, all3 others wait; final progress reward1, terminal.',
        extra_V_noise=0,sigma_Q=.002,default_common_Q_fraction=1.,
        hypothesis='Persistent identical Q readout errors can create a systematic learned-V score residual even after expected-loss minimization.',
        primary_window='last10% of2M; additionally report each 200K block and 5M fresh-seed check.',
        fairness='Replay, Q-error field, Q lr, EMA, initialization and policy extraction matched at equal horizon. IQL-9 has9 V action queries.',
        limitations='Persistent Q errors are a controlled surrogate for approximation error. They are assumed not to vanish. Finite sampler is a categorical/transport surrogate, not a neural flow model.')
    reg=a.root/'registration.json'
    if reg.exists():assert json.loads(reg.read_text())==registration
    else:write_json(reg,registration)
    jobs=[(str(a.root),c,s,sources) for c in cases for s in c['seeds']];started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result());write_json(a.root/'progress.json',dict(completed=len(rows),total=len(jobs),seconds=time.monotonic()-started))
    write_json(a.root/'complete.json',dict(status='complete',runs=len(rows),seconds=time.monotonic()-started))


if __name__=='__main__':main()
