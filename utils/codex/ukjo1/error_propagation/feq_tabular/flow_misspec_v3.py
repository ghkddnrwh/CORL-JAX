"""Supplement: explicitly misspecified behavior-flow stress tests.

The original confirmation's uniform-mixture cases are shrinkage controls,
not adverse misspecification: true behavior happens to be uniform. This
separate registration tests actual loss of the progress-action mode and
retains all intermediate outcomes; original confirmation sources stay frozen.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .confirm_v2 import sha,atomic
from .engine_v2 import run,METHODS,METRICS,COMPONENTS
from .mdp import chain,solve,evaluate_policy,policy_value
from .behavior_flow import offline_counts,fit

def worker(payload):
    root_text,case,seed,source_sha=payload;root=Path(root_text)
    folder=root/'raw'/case['name']/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    mdp=chain(case['length'],.25,case['gamma'],.9,.8)
    qt,vt,_=solve(mdp);qs,vs,_=solve(mdp,optimal=True)
    counts=offline_counts(mdp.behavior,seed,1024);data_behavior=fit(counts)
    eps=case.get('flow_contamination',0.)
    wrong=np.array([0.,1/3,1/3,1/3])
    model=(1-eps)*data_behavior+eps*wrong
    rows=[]
    for method in case['methods']:
        label=METHODS[method]
        if method==1 and case.get('num_candidates',9)!=9:label=f"feq_n{case['num_candidates']}"
        path=folder/f'{label}.npz';meta_path=path.with_suffix('.json')
        identity=dict(case=case,seed=seed,method=method,label=label)
        if path.exists() and meta_path.exists():
            saved=json.loads(meta_path.read_text());assert saved['identity']==identity and saved['source_sha256']==source_sha
            assert sha(path)==saved['data_sha256'];rows.append(saved['metrics']);continue
        before=time.monotonic()
        out=run(method,mdp.reward,mdp.successor,mdp.continuation,data_behavior,model,mdp.gamma,mdp.tau,
                case['steps'],.1,case.get('alpha_v',.1),.005,seed,qt,vt,case['sigma_q'],case['sigma_v'],
                .99 if case['mode']=='field' else 0.,case['mode']=='field',log_every=5000,
                clip=case.get('clip',False),reference_behavior=mdp.behavior,num_candidates=case.get('num_candidates',9),
                freeze_q=case.get('freeze_q',False),init_truth=case.get('init_truth',False))
        steps,logs,q,target,value,qhistory,vhistory,decomp,cross,identity_error=out
        assert np.isfinite(q).all() and np.isfinite(target).all() and identity_error<1e-10
        np.testing.assert_allclose(cross[1:].sum((1,2)),decomp[1:,-1,1]**2,atol=1e-10)
        returns=[];successes=[]
        for snapshot in qhistory:
            ret,suc=evaluate_policy(mdp,snapshot,data_behavior);returns.append(ret);successes.append(suc)
        tail=steps>=.9*case['steps'];early=(steps>0)&(steps<=100000)
        metrics=dict(case=case['name'],seed=seed,method=label,steps=case['steps'],gamma=mdp.gamma,
            sigma_q=case['sigma_q'],sigma_v=case['sigma_v'],mode=case['mode'],
            final_q_rmse=float(logs[-1,0]),tail_q_rmse=float(logs[tail,0].mean()),
            early_100k_q_rmse=float(logs[early,0].mean()),tail_return=float(np.mean(np.array(returns)[tail])),
            tail_success=float(np.mean(np.array(successes)[tail])),final_return=float(returns[-1]),
            final_success=float(successes[-1]),q_rmse_optimal=float(np.sqrt(np.mean((q-qs)**2))),
            exact_policy_optimal_return=float(vs[0]),greedy_return=float(policy_value(mdp,q)[0]),
            behavior_TV=float(np.mean(np.abs(model-mdp.behavior).sum(-1)/2)),
            extra_flow_TV=float(np.mean(np.abs(model-data_behavior).sum(-1)/2)),
            identity_error=float(identity_error),seconds=time.monotonic()-before)
        for i,name in enumerate(COMPONENTS):metrics['target_'+name+'_rms']=float(decomp[tail,i,1].mean())
        for checkpoint in (50000,100000,200000,500000,1000000):
            where=np.flatnonzero(steps==checkpoint)
            if len(where):
                i=int(where[0]);metrics[f'q_rmse_{checkpoint}']=float(logs[i,0]);metrics[f'return_{checkpoint}']=float(returns[i]);metrics[f'success_{checkpoint}']=float(successes[i])
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,step=steps,metrics=logs,metric_names=np.array(METRICS),q=q,target=target,value=value,
            q_history=qhistory,v_history=vhistory,decomposition=decomp,component_cross=cross,component_names=np.array(COMPONENTS),
            policy_return=np.array(returns),goal_success=np.array(successes),q_tau_true=qt,v_tau_true=vt,q_optimal=qs,v_optimal=vs,
            offline_counts=counts,data_behavior=data_behavior,flow_behavior=model,true_behavior=mdp.behavior,
            reward=mdp.reward,successor=mdp.successor,continuation=mdp.continuation)
        temporary.replace(path)
        atomic(meta_path,dict(identity=identity,source_sha256=source_sha,data_sha256=sha(path),metrics=metrics))
        rows.append(metrics);print(json.dumps(dict(case=case['name'],seed=seed,method=label,tail_rmse=metrics['tail_q_rmse'],tail_success=metrics['tail_success'])),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();r=a.root.resolve()
    names=('engine.py','engine_v2.py','mdp.py','behavior_flow.py','confirm_v2.py','flow_misspec_v3.py')
    folder=Path(__file__).parent;sources={str((folder/n).resolve()):sha(folder/n) for n in names}
    cases=[]
    for label,mode,sigma in (('clean','iid',0.),('iid','iid',.001),('field','field',.001)):
        for eps in (.05,.5,.95,.995):
            cases.append(dict(name=f'misspec_{label}_eps{eps:g}',length=12,mode=mode,sigma_q=sigma,sigma_v=sigma,
                steps=200000,seeds=list(range(100,108)),methods=[0,1],gamma=.999,flow_contamination=eps))
    plan=dict(date_KST='2026-11-24',conditions=cases,source_sha256=sources,
        purpose='Adverse behavior-model stress, distinct from the pre-existing uniform-shrinkage control.',
        wrong_behavior=[0.,1/3,1/3,1/3],scope='Controlled model misspecification, not a new flow training algorithm.',
        trajectories=sum(len(c['seeds'])*len(c['methods']) for c in cases))
    target=r/'flow_misspec_registration_v3.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:
        atomic(target,plan);out=r/'flow_misspec_source_v3';out.mkdir(exist_ok=True)
        for n in names:shutil.copyfile(folder/n,out/n)
    begun=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,(str(r),c,s,sources)) for c in cases for s in c['seeds']]
        for future in as_completed(futures):
            future.result();done+=1;atomic(r/'flow_misspec_progress_v3.json',dict(completed_seed_groups=done,total_seed_groups=len(futures),seconds=time.monotonic()-begun))
    atomic(r/'flow_misspec_completed_v3.json',dict(status='complete',trajectories=plan['trajectories'],seconds=time.monotonic()-begun))
if __name__=='__main__':main()
