"""Supplement matching private errors at EACH state, not only global RMS.

This tests the spatial-allocation confound in the global v5 comparison.
Temporal persistence and error signs still differ and are explicitly retained.
"""
import os
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .behavior_flow import fit, offline_counts
from .confirm_v2 import atomic, sha
from .engine_state_v6 import run
from .engine_v2 import run as scalar_run
from .mdp import chain, solve, evaluate_policy
from .noise_models_v5 import behavior_model, reference_error
from .noise_sweep_v5 import SEEDS, EPSILONS


def validate():
    m=chain(4,.25,.999,.9,.8);q,v,_=solve(m)
    for spatial in (False,True):
        common=(0,m.reward,m.successor,m.continuation,m.behavior,m.behavior,m.gamma,m.tau,
                2000,.1,.1,.005,71,q,v)
        kw=dict(sigma_q=0.,rho=.99 if spatial else 0.,spatial=spatial,log_every=500,
                reference_behavior=m.behavior,clip=False,num_candidates=9)
        a=scalar_run(*common,sigma_v=.001,**kw)
        b=run(*common,sigma_v=np.full(len(q),.001),**kw)
        for k in range(9):
            np.testing.assert_array_equal(a[k],b[k])
    return ['Constant state-noise vector preserves scalar IQL trajectory bitwise in IID and field modes']


def worker(payload):
    root_text,seed,hashes=payload;root=Path(root_text)
    m=chain(12,.25,.999,.9,.8);qt,vt,_=solve(m);qo,vo,_=solve(m,optimal=True)
    counts=offline_counts(m.behavior,seed,1024);data=fit(counts);qref,vref,_=solve(m,data)
    completed=0
    for epsilon in EPSILONS:
        model=behavior_model(data,'deplete',epsilon,seed)
        delta,shift,weights=reference_error(m,qref,data,model)
        sv=np.abs(shift)
        np.testing.assert_allclose(m.gamma*np.sqrt(np.dot(weights,sv**2)),delta,atol=1e-14)
        for mode in ('iid','field'):
            for clip in (False,True):
                name=f'eps{epsilon:g}_{mode}_clip{int(clip)}'
                path=root/'raw'/f'seed{seed}'/(name+'.npz');path.parent.mkdir(exist_ok=True,parents=True)
                meta=path.with_suffix('.json');identity=dict(seed=seed,epsilon=epsilon,mode=mode,clip=clip)
                if meta.exists():
                    saved=json.loads(meta.read_text());assert saved['identity']==identity and saved['source_sha256']==hashes
                    assert sha(path)==saved['data_sha256'];completed+=1;continue
                out=run(0,m.reward,m.successor,m.continuation,data,data,m.gamma,m.tau,
                        200000,.1,.1,.005,seed,qt,vt,sigma_q=0.,sigma_v=sv,
                        rho=.99 if mode=='field' else 0.,spatial=mode=='field',log_every=5000,
                        reference_behavior=m.behavior,clip=clip,num_candidates=9)
                steps,logs,q,target,v,qh,vh,decomp,cross,err=out
                assert err<1e-10 and np.isfinite(q).all()
                np.testing.assert_allclose(cross[1:].sum((1,2)),decomp[1:,-1,1]**2,atol=1e-10)
                policy=np.array([evaluate_policy(m,s,data) for s in qh]);tail=steps>=180000
                row=dict(seed=seed,epsilon=epsilon,mode=mode,clip=clip,reference_target_rms=delta,
                    reference_error_state0_energy_fraction=float(weights[0]*shift[0]**2/np.dot(weights,shift**2)) if delta else 0.,
                    tail_success=float(policy[tail,1].mean()),tail_return=float(policy[tail,0].mean()),
                    tail_q_rmse=float(logs[tail,0].mean()),
                    target_V_fit_mean=float(decomp[tail,2,0].mean()),target_V_fit_rms=float(decomp[tail,2,1].mean()))
                temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,step=steps,q=q,target=target,value=v,q_history=qh,v_history=vh,
                    metrics=logs,decomposition=decomp,component_cross=cross,policy=policy,
                    sigma_v=sv,reference_flow_shift=shift,reference_weights=weights,
                    q_ref_data=qref,q_tau_true=qt,q_optimal=qo,data_behavior=data,model_behavior=model)
                temp.replace(path);atomic(meta,dict(identity=identity,source_sha256=hashes,data_sha256=sha(path),metrics=row))
                completed+=1
        print(json.dumps(dict(seed=seed,completed=completed,total=40)),flush=True)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=a.root.resolve()/'state_profile_v6';root.mkdir(exist_ok=True)
    names=('engine_state_v6.py','state_profile_v6.py','engine_v2.py','noise_models_v5.py',
           'noise_sweep_v5.py','mdp.py','behavior_flow.py','confirm_v2.py')
    sources={str((Path(__file__).parent/n).resolve()):sha(Path(__file__).parent/n) for n in names}
    plan=dict(date_KST='2026-11-25',seeds=SEEDS,epsilons=EPSILONS,steps=200000,trajectories=640,
        source_sha256=sources,intervention='sigma_V(s)=abs(e_tau(Qref,mu_model)-e_tau(Qref,mu_data));sigma_Q=0',
        limits=['Statewise marginal source MSE matches; signs and temporal persistence still differ.',
                'Supplement added to investigate the spatial-allocation confound in global RMS matching.'],
        checks=validate())
    path=root/'registration.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:
        atomic(path,plan);(root/'source_snapshot').mkdir(exist_ok=True)
        for s in sources:shutil.copyfile(s,root/'source_snapshot'/Path(s).name)
    begun=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,(str(root),seed,sources)) for seed in SEEDS]
        for f in as_completed(futures):
            done+=f.result();atomic(root/'progress.json',dict(completed=done,total=640))
    assert done==640 and all(sha(s)==h for s,h in sources.items())
    atomic(root/'completed.json',dict(status='complete',trajectories=done,seconds=time.monotonic()-begun))
    print('STATE_PROFILE_COMPLETE',flush=True)
if __name__=='__main__':main()
