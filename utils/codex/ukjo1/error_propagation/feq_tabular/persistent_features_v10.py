"""Four-state known MDP with identical state features for Q and learned V.

This is deliberately a FUNCTION-APPROXIMATION witness on a finite MDP,
not unrestricted tabular learning. Every true Q_a is in the critic class.
State-dependent behavior makes e_tau(Q) leave the shared feature span.
No external Q/V noise. An exact convex optimum and a realizable-V control
separate persistent projection residual from incomplete optimization.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,itertools,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from numba import njit
from .mdp import MDP,expectile,roots,multinomial_counts,finite_roots,best_of_n_policy
from .fair_study_v7 import sha,write_json
from .noise_models_v5 import behavior_model
from .theory_channels_v9 import score,score_root,secant_diagnostics

METHODS=('iql','iql9','feq9','repair_half','repair_zero','iql_expanded','population')


def setup(seed,tau=.9,center_high=.01,bonus=.15,model_strength=0.):
    phi=np.array([[1.,0.,0.],[0.,1.,-1.],[0.,1.,0.],[0.,1.,1.]])
    expanded=np.c_[phi,np.array([0.,1.,0.,1.])]
    reward=np.zeros((4,4));reward[0]=[0.,bonus,0.,bonus/2]
    reward[1:]=.5+(.3+.1*np.array([-1.,0.,1.]))[:,None]*np.array([1.,-1.,1.,-1.])
    successor=np.zeros((4,4),dtype=np.int64);successor[0]=[3,2,1,2]
    continuation=np.zeros((4,4));continuation[0]=1.
    true_beta=np.full((4,4),.25);true_beta[2]=[center_high/2,(1-center_high)/2,center_high/2,(1-center_high)/2]
    rng=np.random.default_rng(7100003+seed);episodes=4096
    states=np.empty(2*episodes,np.int64);actions=np.empty_like(states)
    for k in range(episodes):
        a=int(rng.choice(4,p=true_beta[0]));s=successor[0,a]
        states[2*k]=0;actions[2*k]=a;states[2*k+1]=s
        actions[2*k+1]=rng.choice(4,p=true_beta[s])
    counts=np.zeros((4,4))
    for s,a in zip(states,actions):counts[s,a]+=1
    beta=counts/counts.sum(-1,keepdims=True)
    assert (beta>0).all()
    model=behavior_model(beta,'logit',model_strength,seed)
    m=MDP(reward,successor,continuation,beta,.999,tau,'two_step_shared_features')
    return m,phi,expanded,states,actions,counts.sum(-1)/counts.sum(),model,true_beta


def exact_v_fit(q,beta,occupancy,phi,tau):
    """Enumerate convex quadratic regions; certify zero projected gradient.

    Each terminal state has exactly two distinct action values. The three
    possible intervals per state give 27 active sets. No optimizer tolerance
    or finite training budget is used to generate the residual.
    """
    x=phi[1:,1:];qq=q[1:];b=beta[1:];d=occupancy[1:]
    solutions=[]
    for regions in itertools.product(range(3),repeat=3):
        w=np.empty_like(qq)
        for s,region in enumerate(regions):
            w[s]=tau if region==0 else (1-tau if region==2 else np.where(qq[s]>qq[s].min(),tau,1-tau))
        wb=d[:,None]*b*w
        mat=x.T@(wb.sum(-1)[:,None]*x);rhs=x.T@(wb*qq).sum(-1)
        theta=np.linalg.solve(mat,rhs);v=x@theta
        valid=True
        for s,region in enumerate(regions):
            low,high=qq[s].min(),qq[s].max()
            valid &= (v[s]<=low+1e-10) if region==0 else ((v[s]>=high-1e-10) if region==2 else (low-1e-10<=v[s]<=high+1e-10))
        if valid:solutions.append(theta)
    assert solutions
    theta=solutions[0];v=x@theta
    projected=x.T@(d*score(qq,v[:,None],b,tau))
    assert abs(projected).max()<1e-12
    return v,theta,projected


def references(m,phi,occupancy,model):
    q=m.reward.copy();v=roots(q,m.behavior,m.tau)
    qt=q.copy();qt[0]+=m.gamma*v[m.successor[0]]
    vi,theta,projected=exact_v_fit(q,m.behavior,occupancy,phi,m.tau)
    qi=q.copy();qi[0]+=m.gamma*vi[m.successor[0]-1]
    counts,coeff=multinomial_counts(9,4)
    vf=finite_roots(q,model,m.tau,counts,coeff)
    qf=q.copy();qf[0]+=m.gamma*vf[m.successor[0]]
    # Q_t is realizable in exactly the same phi used by V.
    weights=np.linalg.lstsq(phi,qt,rcond=None)[0]
    np.testing.assert_allclose(phi@weights,qt,atol=1e-12)
    full_v=np.r_[expectile(qi[0],m.behavior[0],m.tau),vi]
    d=secant_diagnostics(qi,full_v,m.behavior,m.tau)
    return dict(q_truth=qt,q_iql=qi,q_feq9=qf,v_iql=full_v,
                leaf_score_residual=d['zeta'][1:],leaf_value_gap=d['gap'][1:],
                leaf_kappa=d['kappa'][1:],leaf_secant=d['secant'][1:],projected_gradient=projected)


def evaluate(m,q,beta):
    pi=best_of_n_policy(q,beta,9)
    ret=float(np.dot(pi[0],m.reward[0]+m.gamma*np.sum(pi*m.reward,-1)[m.successor[0]]))
    # Success = choose either endpoint, then its high-reward action; the
    # central state is the distractor. Discounted return is also reported.
    success=0.
    for a in (0,2):
        s=m.successor[0,a]
        success+=pi[0,a]*np.sum(pi[s,m.reward[s]==m.reward[s].max()])
    return ret,float(success)


@njit(cache=True)
def simulate(method,reward,sn,beta,model,phi,expanded,offline_s,offline_a,tau,gamma,steps,seed,anneal=False,log_every=10000):
    # Separate streams make all horizons prefix invariant.
    np.random.seed(8100001+seed);indices=np.random.randint(0,len(offline_s),steps)
    np.random.seed(9100003+seed)
    qtheta=np.zeros((3,4));target=qtheta.copy()
    vp=expanded if method==5 else phi
    vtheta=np.zeros(vp.shape[1]);bcdf=np.cumsum(beta,axis=1) if False else beta.copy()
    mcdf=model.copy()
    for s in range(4):
        for a in range(1,4):bcdf[s,a]+=bcdf[s,a-1];mcdf[s,a]+=mcdf[s,a-1]
    count=1+(steps+log_every-1)//log_every
    times=np.zeros(count,np.int64);qh=np.zeros((count,4,4));th=qh.copy();vh=np.zeros((count,4))
    idx=0
    for t in range(steps+1):
        if t==0 or t%log_every==0 or t==steps:
            if t:idx+=1
            times[idx]=t;qh[idx]=phi@qtheta;th[idx]=phi@target;vh[idx]=vp@vtheta
        if t==steps:break
        uniforms=np.random.random(9)
        s=offline_s[indices[t]];a=offline_a[indices[t]];next_s=sn[s,a]
        online=phi@qtheta;tq=phi@target;value=vp@vtheta
        boot=0.
        if s==0:
            if method==2:
                counts=np.zeros(4)
                for u in uniforms:
                    aa=0
                    while aa<3 and u>=mcdf[next_s,aa]:aa+=1
                    counts[aa]+=1
                boot=expectile(tq[next_s],counts,tau)
            elif method==6:
                boot=expectile(tq[next_s],model[next_s],tau)
            elif method in (3,4):
                exact=expectile(tq[next_s],beta[next_s],tau)
                boot=exact+(.5 if method==3 else 0.)*(value[next_s]-exact)
            else:boot=value[next_s]
        rate=.02/(np.sqrt(1+t/10000.) if anneal else 1.)
        old=qtheta.copy()
        error=reward[s,a]+gamma*boot-online[s,a]
        for k in range(3):qtheta[k,a]+=rate*error*phi[s,k]
        if method not in (2,6):
            grad=0.
            if method==1:
                for u in uniforms:
                    aa=0
                    while aa<3 and u>=bcdf[s,aa]:aa+=1
                    delta=tq[s,aa]-value[s]
                    grad+=(tau if delta>=0 else 1-tau)*delta/9.
            else:
                delta=tq[s,a]-value[s]
                grad=(tau if delta>=0 else 1-tau)*delta
            vtheta+=2*rate*grad*vp[s]
        target=.995*target+.005*old
    return times,qh,th,vh


def plan():
    cases=[]
    for method in METHODS:
        cases.append(dict(name='main',method=method,steps=1000000,seeds=list(range(1300,1316))))
    for name,extra in [('tau099',dict(tau=.99)),('uniform_behavior_control',dict(center_high=.5)),('annealed',dict(anneal=True))]:
        for method in ('iql','iql9','feq9'):
            cases.append(dict(name=name,method=method,steps=1000000,seeds=list(range(1300,1308)),**extra))
    for strength in (.25,.5):
        cases.append(dict(name=f'model_error_{strength}',method='feq9',steps=1000000,seeds=list(range(1300,1316)),model_strength=strength))
    for method in ('iql','iql9','feq9'):
        cases.append(dict(name='five_million',method=method,steps=5000000,seeds=list(range(1400,1404))))
    return cases


def worker(job):
    root,c,seed,sources=job;root=Path(root);identity=dict(case=c,seed=seed)
    name=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'raw'/name;out.mkdir(parents=True,exist_ok=True);path=out/'data.npz';meta=out/'result.json'
    if meta.exists():
        m=json.loads(meta.read_text());assert m['sources']==sources and m['identity']==identity and m['sha256']==sha(path)
        return m['metrics']
    m,phi,expanded,ds,da,occupancy,model,true_beta=setup(seed,c.get('tau',.9),c.get('center_high',.01),.15,c.get('model_strength',0.))
    ref=references(m,phi,occupancy,model)
    t,q,target,v=simulate(METHODS.index(c['method']),m.reward,m.successor,m.behavior,model,phi,expanded,ds,da,m.tau,m.gamma,c['steps'],seed,c.get('anneal',False))
    assert np.isfinite(q).all()
    policy=np.array([evaluate(m,qq,m.behavior) for qq in q]);deployed=np.array([evaluate(m,qq,model) for qq in q])
    rmse=np.sqrt(((q-ref['q_truth'])**2).mean((1,2)));tail=t>=.9*c['steps']
    row=dict(case=c['name'],method=c['method'],seed=seed,steps=c['steps'],
        success=float(policy[tail,1].mean()),deployed_success=float(deployed[tail,1].mean()),
        discounted_return=float(policy[tail,0].mean()),q_rmse=float(rmse[tail].mean()),
        terminal_q_rmse=float(np.sqrt(((q[tail,1:]-m.reward[None,1:])**2).mean())),
        exact_iql_success=evaluate(m,ref['q_iql'],m.behavior)[1],exact_feq9_success=evaluate(m,ref['q_feq9'],m.behavior)[1],
        model_tv=float(abs(model-m.behavior).sum(-1).mean()/2))
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,step=t,q=q,target=target,value=v,policy=policy,model_policy=deployed,rmse=rmse,
        reward=m.reward,successor=m.successor,continuation=m.continuation,behavior=m.behavior,model=model,
        true_behavior=true_beta,phi=phi,v_phi=expanded if c['method']=='iql_expanded' else phi,
        offline_states=ds,offline_actions=da,occupancy=occupancy,**ref)
    tmp.replace(path);write_json(meta,dict(identity=identity,sources=sources,sha256=sha(path),metrics=row))
    print(json.dumps(row),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);folder=Path(__file__).parent
    sources={str(folder/f):sha(folder/f) for f in ('persistent_features_v10.py','mdp.py','theory_channels_v9.py','noise_models_v5.py','fair_study_v7.py')}
    cases=plan();registration=dict(cases=cases,sources=sources,gamma=.999,tau=.9,N=9,
        features='Q_a and V share phi=[root_indicator,leaf_indicator,leaf_x]. Four states,3 features. Expanded V adds leaf_abs_x.',
        data='4096 fixed two-step trajectories per seed; same offline buffer for every method; behavior counts fitted from that buffer.',
        errors='No external noise in Q or V. Critic class exactly realizes the population Q reference. V class may not realize its expectile.',
        primary='900K-1M average; 5M and annealed controls; exact global projected V optimum is computed independently.',
        limitations='Restricted function approximation on finite MDP, not unrestricted tabular. Giving V a realizable representation should remove the disadvantage.')
    f=a.root/'registration.json'
    if f.exists():assert json.loads(f.read_text())==registration
    else:write_json(f,registration)
    jobs=[(str(a.root),c,s,sources) for c in cases for s in c['seeds']];rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(pending):
            rows.append(f.result());write_json(a.root/'progress.json',dict(completed=len(rows),total=len(jobs),seconds=time.monotonic()-start))
    write_json(a.root/'complete.json',dict(status='complete',runs=len(rows),seconds=time.monotonic()-start))


if __name__=='__main__':main()
