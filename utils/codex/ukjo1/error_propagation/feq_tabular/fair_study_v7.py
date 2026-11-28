"""Registered, resumable fair-comparison search; outputs belong under logs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .mdp import chain, solve, evaluate_policy, roots
from .behavior_flow import fit, offline_counts
from .noise_models_v5 import behavior_model
from .fair_engine_v7 import simulate, METHODS, PARTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def task(job):
    root, case, seed, sources = job
    root = Path(root)
    identity = dict(case=case, seed=seed)
    code = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = root/'raw'/code
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder/'result.json'
    path = folder/'data.npz'
    if meta.exists():
        saved = json.loads(meta.read_text())
        assert saved['identity'] == identity and saved['sources'] == sources
        assert saved['sha256'] == sha(path)
        return saved['metrics']
    started = time.monotonic()
    mdp = chain(case.get('length',12), case.get('progress',.25), case.get('gamma',.999),
                case.get('tau',.9), case.get('safe',.8))
    data = fit(offline_counts(mdp.behavior, seed, case.get('dataset_size',1024)))
    if case.get('true_behavior',False):
        data = mdp.behavior.copy()
    model = behavior_model(data, case.get('model_kind','none'),case.get('model_strength',0.),seed)
    qt,vt,_ = solve(mdp, data)
    qs,vs,_ = solve(mdp, optimal=True)
    out = simulate(METHODS.index(case['method']),mdp.reward,mdp.successor,mdp.continuation,
        data,model,mdp.gamma,mdp.tau,case['steps'],seed,qt,vt,
        sigma=case.get('sigma',0.),common=case.get('common',1.),rho=case.get('rho',0.),
        alpha_q=case.get('alpha_q',.1),alpha_v=case.get('alpha_v',.1),ema=case.get('ema',.005),
        log_every=case.get('log_every',5000),frozen=case.get('frozen',False),
        init_truth=case.get('init_truth',False),anneal=case.get('anneal',False))
    times,qhist,vhist,thist,parts,cross,residual,noise,worst = out
    assert worst < 1e-10 and np.isfinite(qhist).all()
    np.testing.assert_allclose(cross[1:].sum((1,2)),parts[1:,-1,1],atol=1e-11)
    evaluations = np.array([evaluate_policy(mdp,q,data) for q in qhist])
    model_evaluations = np.array([evaluate_policy(mdp,q,model) for q in qhist])
    tail = times >= .9*case['steps']
    rmse = np.sqrt(((qhist-qt)**2).mean((1,2)))
    row = dict(case=case['name'],seed=seed,method=case['method'],length=mdp.reward.shape[0]-1,
        gamma=mdp.gamma,tau=mdp.tau,sigma=case.get('sigma',0.),common=case.get('common',1.),
        rho=case.get('rho',0.),alpha_v=case.get('alpha_v',.1),steps=case['steps'],
        tail_success=float(evaluations[tail,1].mean()),tail_return=float(evaluations[tail,0].mean()),
        tail_model_success=float(model_evaluations[tail,1].mean()),tail_q_rmse=float(rmse[tail].mean()),
        final_success=float(evaluations[-1,1]),final_q_rmse=float(rmse[-1]),
        tail_v_residual=float(np.sqrt((residual[tail]**2).mean())),
        noise_mean=float(noise[1:,0].mean()),noise_rms=float(np.sqrt(noise[1:,1].mean())),
        behavior_TV=float(np.abs(model-data).sum(-1).mean()/2),identity_error=float(worst),
        seconds=time.monotonic()-started)
    for i,name in enumerate(PARTS):
        row[name+'_mean'] = float(parts[tail,i,0].mean())
        row[name+'_rms'] = float(np.sqrt(parts[tail,i,1].mean()))
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,step=times,q=qhist,value=vhist,target=thist,decomposition=parts,
        cross=cross,v_residual=residual,noise_statistics=noise,policy=evaluations,
        model_policy=model_evaluations,q_truth=qt,v_truth=vt,q_optimal=qs,v_optimal=vs,
        behavior=data,model=model,reward=mdp.reward,successor=mdp.successor,
        continuation=mdp.continuation,rmse=rmse)
    temp.replace(path)
    write_json(meta,dict(identity=identity,sources=sources,sha256=sha(path),metrics=row))
    print(json.dumps({k:row[k] for k in ('case','seed','method','tail_success','tail_q_rmse')}),flush=True)
    return row


def discovery():
    cases=[]
    for length in (4,12,24):
        for gamma in (.995,.999):
            for sigma in (0.,.0005,.001,.002,.005):
                for method in ('iql','iql_batch9','feq9','population'):
                    cases.append(dict(name=f'L{length}_g{gamma}_s{sigma}',length=length,
                        gamma=gamma,sigma=sigma,method=method,steps=100000,seeds=[300,301]))
    return cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    cases=json.loads(args.plan.read_text()) if args.plan else discovery()
    folder=Path(__file__).parent
    sources={str(folder/p):sha(folder/p) for p in ('fair_engine_v7.py','fair_study_v7.py','mdp.py','behavior_flow.py','noise_models_v5.py')}
    registration=dict(cases=cases,sources=sources,external_V_noise=0,
        fairness=['Identical replay/action uniforms and full Q-error field across methods.',
        'Same Q learning rate, target EMA, transitions, initialization, and evaluation proposals.',
        'IQL batch9 uses9 action queries to match FEQ9 candidate budget.',
        'Behavior MLE uses the same fixed finite offline action counts; no Q-to-model gradients.',
        'All searched settings saved, including ties/reversals; discovery seeds held out from confirmation.',
        'Primary truth is the population expectile fixed point for the empirical behavior; Q* also saved.'],
        limitations=['Injected Q error is an approximation-error surrogate, not ordinary exact tabular learning.',
        'Equal post-expectile target errors have identical propagation regardless of their label.',
        'Policy extraction is common best9, not IQL AWBC or neural FEQ actor.'])
    path=args.root/'registration.json'
    if path.exists():
        assert json.loads(path.read_text())==registration
    else:
        write_json(path,registration)
    jobs=[(str(args.root),c,seed,sources) for c in cases for seed in c['seeds']]
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task,j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(args.root/'progress.json',dict(completed=len(rows),total=len(jobs),seconds=time.monotonic()-started))
    rows.sort(key=lambda r:(r['case'],r['method'],r['seed']))
    with (args.root/'per_seed.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.root/'complete.json',dict(status='complete',runs=len(rows),seconds=time.monotonic()-started))


if __name__=='__main__':
    main()
