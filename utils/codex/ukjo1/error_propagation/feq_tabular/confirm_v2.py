"""Frozen independent-seed confirmation and mechanism ablations for FEQ."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import fcntl
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from .mdp import chain, solve, evaluate_policy, policy_value
from .engine_v2 import run, METHODS, METRICS, COMPONENTS
from .behavior_flow import offline_counts, fit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)


def conditions():
    main=[]
    for name,length,mode,sigma in (('iid',12,'iid',.001),('field',12,'field',.001),('clean_long',24,'iid',0.)):
        main.append(dict(name='main_'+name,length=length,mode=mode,sigma_q=sigma,sigma_v=sigma,
                         steps=1000000,seeds=list(range(100,132)),methods=[0,1,2],gamma=.999))
    extra=[]
    for mode in ('iid','field'):
        for name,sq,sv in (('Q_only',.001,0.),('V_only',0.,.001),('large_noise',.002,.002),('zero',0.,0.),('balanced',.001,.001)):
            extra.append(dict(name=f'control_{mode}_{name}',length=12,mode=mode,sigma_q=sq,sigma_v=sv,
                              steps=200000,seeds=list(range(100,108)),methods=[0,1,2],gamma=.999))
        for kind,kwargs,methods in (
            ('V_fit',{},[0,1,3,4,5,6]),
            ('clip',dict(clip=True),[0,1]),
            ('N81',dict(num_candidates=81),[0,1]),
            ('N9_control',{},[1]),
            ('fast_V',dict(alpha_v=.5),[0,1]),
        ):
            extra.append(dict(name=f'ablation_{mode}_{kind}',length=12,mode=mode,sigma_q=.001,sigma_v=.001,
                              steps=500000,seeds=list(range(100,108)),methods=methods,gamma=.999))
            extra[-1].update(kwargs)
        for epsilon in (.01,.05,.2):
            extra.append(dict(name=f'flow_{mode}_eps{epsilon:g}',length=12,mode=mode,sigma_q=.001,sigma_v=.001,
                              steps=200000,seeds=list(range(100,108)),methods=[0,1],gamma=.999,flow_contamination=epsilon))
        extra.append(dict(name=f'openloop_{mode}',length=12,mode=mode,sigma_q=.001,sigma_v=.001,
                          steps=200000,seeds=list(range(100,108)),methods=[0,1,2],gamma=.999,freeze_q=True,init_truth=True))
    extra.append(dict(name='clean_long_V_fit',length=24,mode='iid',sigma_q=0.,sigma_v=0.,steps=1000000,
                      seeds=list(range(100,108)),methods=[0,1,3,4,5,6],gamma=.999))
    return main+extra


def worker(payload):
    root_text,case,seed,source_sha=payload;root=Path(root_text)
    folder=root/'raw'/case['name']/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    mdp=chain(case['length'],.25,case['gamma'],.9,.8)
    qt,vt,_=solve(mdp);qs,vs,_=solve(mdp,optimal=True)
    counts=offline_counts(mdp.behavior,seed,1024);data_behavior=fit(counts)
    eps=case.get('flow_contamination',0.)
    model=(1-eps)*data_behavior+eps/4.
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lease=(root/'confirmation.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validation=json.loads((root/'validation/core_v2.json').read_text())
    assert validation['status']=='PASS'
    assert all(sha(p)==h for p,h in validation['source_sha256'].items())
    folder=Path(__file__).parent
    sources={str(p.resolve()):sha(p) for name in ('engine.py','engine_v2.py','mdp.py','behavior_flow.py','confirm_v2.py') for p in [folder/name]}
    cases=conditions()
    plan=dict(date_KST='2026-11-24',name='FEQ known-MDP residual mechanism confirmation',conditions=cases,
        source_sha256=sources,discovery_seeds=list(range(6)),confirmation_seeds=list(range(100,132)),
        dataset_per_state=1024,alpha_q=.1,alpha_v=.1,target_ema=.005,tau=.9,behavior_progress=.25,safe_fraction=.8,
        main_metrics=['last10percent_Q_RMSE_to_population_expectile_fixed_point','last10percent_exact_best9_goal_success_1000steps'],
        noise_free_metric='first100K time-averaged Q RMSE; also report1M reversals',
        decision='FEQ-N9 lower tail Q RMSE and higher goal success in both noise modes on32 unseen seeds; paired95percent bootstrap CI excludes0.',
        modeling_limits=['Piecewise-constant tabular Q with1D continuous action and learned monotone-CDF flow, not a neural flow-matching experiment.',
        'Readout error fields are explicitly injected approximation-error surrogates; clean tabular conditions reported separately.',
        'Shared data and common best9 deployment isolate value updates; MDN actor is not part of this experiment.',
        'True optimal Q is not the expectile fixed point: both are saved and scored separately.',
        'No claim that FEQ eliminates Q bootstrap error or that IQL behavior fitting must depend on Q.'],
        tasks=sum(len(c['seeds'])*len(c['methods']) for c in cases),updates=sum(len(c['seeds'])*len(c['methods'])*c['steps'] for c in cases))
    path=root/'confirmation_registration_v2.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:
        atomic(path,plan)
        for source in sources:
            target=root/'confirmation_source_v2'/Path(source).name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    jobs=[(str(root),case,seed,sources) for case in cases for seed in case['seeds']]
    done=0;began=time.monotonic();all_rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(worker,j):j for j in jobs}
        for future in as_completed(pending):
            all_rows+=future.result();done+=1
            atomic(root/'confirmation_progress_v2.json',dict(completed_seed_groups=done,total_seed_groups=len(jobs),
                completed_trajectories=len(all_rows),expected_trajectories=plan['tasks'],seconds=time.monotonic()-began))
    atomic(root/'confirmation_completed_v2.json',dict(status='complete',seed_groups=done,trajectories=len(all_rows),
        seconds=time.monotonic()-began,python=platform.python_version(),numpy=np.__version__))
    print('CONFIRMATION_COMPLETE',flush=True)
if __name__=='__main__':main()
