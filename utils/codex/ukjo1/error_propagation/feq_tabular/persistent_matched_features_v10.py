"""Capacity control: Q, V and behavior logits share the same state features.

V has three coefficients; the fitted behavior model has three coefficients.
No tabular behavior lookup is used at deployment. The true behavior is in
this logistic class, fitted only by likelihood on the common offline buffer.
Q is linear per action in exactly the same features and is realizable.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from .persistent_features_v10 import simulate, METHODS, references
from .mdp import MDP,best_of_n_policy,roots,multinomial_counts,finite_roots
from .fair_study_v7 import sha,write_json


def setup(seed):
    phi=np.array([[1.,0.,0.],[0.,1.,-1.],[0.,1.,0.],[0.,1.,1.]])
    expanded=np.c_[phi,[0.,1.,0.,1.]]
    reward=np.zeros((4,4));reward[1:]=np.array([1.,.6,.2])[:,None]*[1.,0.,1.,0.]
    successor=np.zeros((4,4),np.int64);successor[0]=[3,2,1,2]
    continuation=np.zeros((4,4));continuation[0]=1.
    prob=expit(phi@np.array([0.,0.,4.6]))
    true_beta=np.stack([prob/2,(1-prob)/2,prob/2,(1-prob)/2],-1)
    rng=np.random.default_rng(7100003+seed);episodes=4096
    ds=np.empty(2*episodes,np.int64);da=np.empty_like(ds)
    counts=np.zeros((4,4))
    for k in range(episodes):
        a=rng.choice(4,p=true_beta[0]);s=successor[0,a];b=rng.choice(4,p=true_beta[s])
        ds[2*k:2*k+2]=[0,s];da[2*k:2*k+2]=[a,b];counts[0,a]+=1;counts[s,b]+=1
    total=counts.sum(-1);positive=counts[:,[0,2]].sum(-1)
    def nll(theta):
        logits=phi@theta
        loss=np.dot(total,np.logaddexp(0,logits))-np.dot(positive,logits)
        grad=phi.T@(total*expit(logits)-positive)
        return loss/total.sum(),grad/total.sum()
    opt=minimize(nll,np.zeros(3),jac=True,method='BFGS',options={'gtol':1e-12,'maxiter':2000})
    assert abs(nll(opt.x)[1]).max()<1e-8
    p=expit(phi@opt.x)
    model=np.stack([p/2,(1-p)/2,p/2,(1-p)/2],-1)
    empirical=counts/total[:,None]
    m=MDP(reward,successor,continuation,empirical,.999,.99,'shared_logistic_features')
    return m,phi,expanded,ds,da,total/total.sum(),model,true_beta,opt.x


def evaluate(m,q,model):
    pi=best_of_n_policy(q,model,9)
    leaf_return=np.sum(pi*m.reward,-1)
    leaf_success=np.sum(pi*(m.reward>0),-1)
    ret=np.dot(pi[0],m.gamma*leaf_return[m.successor[0]])
    success=np.dot(pi[0],leaf_success[m.successor[0]])
    return float(ret),float(success)


def plan():
    cases=[]
    for method in ('iql','iql9','feq9','iql_expanded','repair_zero'):
        cases.append(dict(name='main',method=method,steps=1000000,seeds=list(range(1500,1516))))
    for method in ('iql','iql9','feq9'):
        cases.append(dict(name='five_million',method=method,steps=5000000,seeds=list(range(1600,1604))))
        cases.append(dict(name='annealed',method=method,steps=1000000,seeds=list(range(1500,1508)),anneal=True))
    return cases


def worker(job):
    root,c,seed,sources=job;root=Path(root);identity=dict(case=c,seed=seed)
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'raw'/token;out.mkdir(parents=True,exist_ok=True);meta=out/'result.json';path=out/'data.npz'
    if meta.exists():
        record=json.loads(meta.read_text());assert record['identity']==identity and record['sources']==sources and record['sha256']==sha(path)
        return record['metrics']
    m,phi,expanded,ds,da,occ,model,true_beta,theta=setup(seed)
    ref=references(m,phi,occ,model)
    # IQL-9 gets the same fitted sampler as FEQ. The ordinary IQL baseline
    # keeps the real replay action. Exact references distinguish the two.
    model_m=MDP(m.reward,m.successor,m.continuation,model,m.gamma,m.tau,m.name)
    model_ref=references(model_m,phi,occ,model)
    training_beta=model if c['method']=='iql9' else m.behavior
    t,q,target,v=simulate(METHODS.index(c['method']),m.reward,m.successor,training_beta,model,
        phi,expanded,ds,da,m.tau,m.gamma,c['steps'],seed,c.get('anneal',False))
    policy=np.array([evaluate(m,qq,model) for qq in q]);same=policy.copy()
    rmse=np.sqrt(((q-ref['q_truth'])**2).mean((1,2)));tail=t>=.9*c['steps']
    row=dict(case=c['name'],method=c['method'],seed=seed,steps=c['steps'],
        success=float(policy[tail,1].mean()),deployed_success=float(policy[tail,1].mean()),
        discounted_return=float(policy[tail,0].mean()),q_rmse=float(rmse[tail].mean()),
        terminal_q_rmse=float(np.sqrt(((q[tail,1:]-m.reward[None,1:])**2).mean())),
        exact_iql_success=evaluate(m,ref['q_iql'],model)[1],
        exact_iql9_success=evaluate(m,model_ref['q_iql'],model)[1],
        exact_feq9_success=evaluate(m,ref['q_feq9'],model)[1],
        model_tv=float(abs(model-m.behavior).sum(-1).mean()/2),
        model_true_tv=float(abs(model-true_beta).sum(-1).mean()/2))
    np.savez_compressed(path,step=t,q=q,target=target,value=v,policy=policy,model_policy=same,rmse=rmse,
        reward=m.reward,successor=m.successor,continuation=m.continuation,behavior=m.behavior,model=model,
        true_behavior=true_beta,phi=phi,v_phi=expanded if c['method']=='iql_expanded' else phi,
        behavior_parameters=theta,offline_states=ds,offline_actions=da,occupancy=occ,
        q_iql9=model_ref['q_iql'],v_iql9=model_ref['v_iql'],**ref)
    write_json(meta,dict(identity=identity,sources=sources,sha256=sha(path),metrics=row))
    print(json.dumps(row),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);folder=Path(__file__).parent
    names=('persistent_matched_features_v10.py','persistent_features_v10.py','mdp.py',
           'theory_channels_v9.py','noise_models_v5.py','fair_study_v7.py')
    sources={str(folder/f):sha(folder/f) for f in names};cases=plan()
    registration=dict(cases=cases,sources=sources,gamma=.999,tau=.99,N=9,
        hypothesis='A shared three-feature logistic behavior model is realizable, as is Q; V expectile need not be in the same linear span.',
        parameters='Q:3x4; V:3; behavior logistic:3. Root and leaf features identical for every model.',
        behavior_fit='Maximum likelihood on4096 common two-step offline episodes; no Q inputs; high/low groups split evenly.',
        errors='No external noise; learned behavior has finite-data estimation error.',
        metric='All methods deploy the same fitted best-of9 sampler. Success=any positive terminal reward; discounted return also reported.',
        design='Analytic construction examined at population probabilities before registration, not training-seed selection.',
        comparison='IQL-9 gets the same learned behavior sampler, queried9 times per V update; Q and replay match.')
    path=a.root/'registration.json'
    if path.exists():assert json.loads(path.read_text())==registration
    else:write_json(path,registration)
    jobs=[(str(a.root),c,s,sources) for c in cases for s in c['seeds']];start=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(pending):
            f.result();done+=1;write_json(a.root/'progress.json',dict(completed=done,total=len(jobs),seconds=time.monotonic()-start))
    write_json(a.root/'complete.json',dict(status='complete',runs=done,seconds=time.monotonic()-start))


if __name__=='__main__':main()
