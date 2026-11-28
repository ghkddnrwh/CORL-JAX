"""Gamma=.99 penalty controls: fixed rewards versus matched relative safe reward."""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .mdp import chain, solve, evaluate_policy
from .behavior_flow import offline_counts, fit
from .score_controls_v11 import train
from .matched_error_v11 import ridge_reference, save_csv, save_json
from .penalty_sweep_v12 import METHODS, MODES, MAIN_SEEDS, digest, config_key, source_hashes
from .refine_penalty_v13 import grouped
from .validate_matched_v11 import reference

GRID=(0.,.00005,.0001,.0002,.0003,.0005,.00075,.001,.0015,.002,.003,.005,.007,.01,.02,.03)
PROTOCOLS=('fixed_reward','matched_gap')


def environment(protocol):
    m=chain(8,.25,.999 if protocol=='fixed_reward' else .99,.9,.9)
    m.gamma=.99
    return m


def make_config(algorithm,mode,strength,seed,protocol):
    return dict(stage='main',algorithm=algorithm,penalty_mode=mode,strength=strength,
        seed=seed,steps=2000000,gamma=.99,reward_protocol=protocol,
        lq=strength if mode in ('q','qv') else 0.,lv=strength if mode in ('v','qv') else 0.)


def gamma_worker(payload):
    output,c=payload
    directory=Path(output)/c['reward_protocol']/'raw'/config_key(c)
    directory.mkdir(parents=True,exist_ok=True)
    metadata=directory/'result.json'
    if metadata.exists():
        stored=json.loads(metadata.read_text())
        assert stored['config']==c and digest(directory/'data.npz')==stored['data_sha256']
        return
    m=environment(c['reward_protocol'])
    counts=offline_counts(m.behavior,c['seed'])
    m.behavior=fit(counts)
    truth,vt,_=solve(m)
    step,q,v=train(m.reward,m.successor,m.continuation,m.behavior,m.tau,m.gamma,
        METHODS[c['algorithm']],0,0.,c['steps'],c['seed'],c['lq'],c['lv'])
    q_fixed,v_fixed=ridge_reference(m,c['lq'],c['lv'],c['algorithm']=='feq9')
    tail=step>=.9*c['steps']
    scores=np.array([evaluate_policy(m,qq) for qq in q[tail]])
    qr=np.sqrt(np.mean((q-truth)**2,axis=(1,2)))
    exact_return,exact_success=evaluate_policy(m,q_fixed)
    row={**c,'success':float(scores[:,1].mean()),'return':float(scores[:,0].mean()),
        'q_rmse':float(qr[tail].mean()),'exact_success':exact_success,'exact_return':exact_return,
        'exact_q_rmse':float(np.sqrt(np.mean((q_fixed-truth)**2))),
        'learned_to_fixed_q_rmse':float(np.sqrt(np.mean((q[tail]-q_fixed)**2))),
        'safe_reward':float(m.reward[0,1]),'root_action_gap':float((q[tail,0,0]-q[tail,0,1]).mean())}
    assert np.isfinite(q).all() and np.isfinite(v).all()
    np.savez_compressed(directory/'data.tmp.npz',step=step,q=q,v=v,q_truth=truth,v_truth=vt,
        q_fixed=q_fixed,v_fixed=v_fixed,counts=counts,beta=m.behavior,q_rmse=qr,
        evaluated_steps=step[tail],evaluated_success=scores[:,1],evaluated_return=scores[:,0],
        reward=m.reward,successor=m.successor,continuation=m.continuation,gamma=m.gamma)
    (directory/'data.tmp.npz').replace(directory/'data.npz')
    save_json(directory/'result.tmp.json',dict(config=c,metrics=row,data_sha256=digest(directory/'data.npz')))
    (directory/'result.tmp.json').replace(metadata)


def records_at(root,protocol):
    records=[]
    for path in sorted((root/protocol/'raw').glob('*/result.json')):
        obj=json.loads(path.read_text())
        records.append(dict(**obj,origin='v14',artifact=str(path.parent.resolve())))
    return records


def refine_jobs(root):
    jobs=[]
    for protocol in PROTOCOLS:
        groups=grouped(records_at(root,protocol))
        points={mode:set() for mode in ('q','v','qv')}
        scale=.1 if protocol=='fixed_reward' else 1.
        floors={'q':.000025*scale,'v':.0000125*scale,'qv':.00000625*scale}
        for algorithm in METHODS:
            for mode in MODES[algorithm]:
                values=sorted((strength,np.mean([r['success'] for r in rs]))
                    for (alg,pmode,strength),rs in groups.items() if alg==algorithm and pmode==mode)
                for (a,ya),(b,yb) in zip(values,values[1:]):
                    if abs(yb-ya)>.08 and b-a>=2*floors[mode]-1e-13:
                        points[mode].add(round((a+b)/2,12))
        for mode,strengths in points.items():
            for strength in sorted(strengths):
                for algorithm in METHODS:
                    if mode in MODES[algorithm] and (algorithm,mode,strength) not in groups:
                        jobs += [make_config(algorithm,mode,strength,seed,protocol) for seed in MAIN_SEEDS]
    return jobs


def check_kernel():
    maximum=0.
    for protocol in PROTOCOLS:
        m=environment(protocol)
        for algorithm,method in METHODS.items():
            for mode in MODES[algorithm]:
                lq=.003 if mode in ('q','qv') else 0.
                lv=.003 if mode in ('v','qv') else 0.
                actual=train(m.reward,m.successor,m.continuation,m.behavior,m.tau,m.gamma,
                    method,0,0.,173,2600,lq,lv,log_every=1)
                q,v=reference(m,173,2600,'score',method=method,channel=0,eta=0.,lq=lq,lv=lv)
                maximum=max(maximum,float(abs(q-actual[1]).max()),float(abs(v-actual[2]).max()))
    assert maximum<1e-11
    return dict(status='passed',independent_traces=14,max_error=maximum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    a=parser.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    sources={**source_hashes(),Path(__file__).name:digest(__file__)}
    registration=json.loads(json.dumps(dict(source_sha256=sources,gamma=.99,steps=2000000,seeds=MAIN_SEEDS,
        initial_grid=GRID,protocols={name:dict(safe_reward=environment(name).reward[0,1],
            meaning='Only discount changes from v12' if name=='fixed_reward' else
                    'Safe reward recalibrated to 90% of clean population continuation value') for name in PROTOCOLS},
        refinement='At most 6 bisections of >8pp adjacent changes, shared lambda within penalty mode',
        selected_after='Uniform-behavior exact fixed-point exploratory check at gamma .99; all conditions retained')))
    path=a.root/'registration.json'
    if path.exists() and json.loads(path.read_text())!=registration:
        raise ValueError('Existing output has different settings or source')
    save_json(path,registration)
    save_json(a.root/'preflight_validation.json',check_kernel())
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for round_index in range(7):
            path=a.root/f'round_{round_index:02}.json'
            if path.exists():
                jobs=json.loads(path.read_text())['jobs']
            elif round_index==0:
                jobs=[]
                for protocol in PROTOCOLS:
                    for strength in GRID:
                        for algorithm in METHODS:
                            for mode in (('none',) if strength==0 else MODES[algorithm]):
                                jobs += [make_config(algorithm,mode,strength,seed,protocol) for seed in MAIN_SEEDS]
            else:
                jobs=refine_jobs(a.root)
            if not jobs:break
            if not path.exists():save_json(path,dict(round=round_index,jobs=jobs))
            print(json.dumps(dict(round=round_index,jobs=len(jobs))),flush=True)
            for done,future in enumerate(as_completed([pool.submit(gamma_worker,(str(a.root),c)) for c in jobs]),1):
                future.result()
                if done%512==0 or done==len(jobs):
                    print(json.dumps(dict(round=round_index,completed=done,total=len(jobs),seconds=round(time.monotonic()-start,1))),flush=True)
    expected={config_key(c) for p in a.root.glob('round_*.json') for c in json.loads(p.read_text())['jobs']}
    actual=set()
    for protocol in PROTOCOLS:
        records=records_at(a.root,protocol)
        grouped(records)
        by_seed={}
        for obj in records:
            c=obj['config'];actual.add(config_key(c));p=Path(obj['artifact'])/'data.npz'
            assert digest(p)==obj['data_sha256']
            with np.load(p) as h:
                assert h['step'][-1]==2000000 and np.isfinite(h['q']).all()
                np.testing.assert_allclose(h['evaluated_success'].mean(),obj['metrics']['success'],atol=1e-13)
                if c['seed'] not in by_seed:by_seed[c['seed']]=h['beta'].copy()
                else:np.testing.assert_array_equal(h['beta'],by_seed[c['seed']])
        save_json(a.root/protocol/'artifact_index.json',records)
        save_json(a.root/protocol/'artifact_validation.json',dict(status='passed',runs=len(records)))
        save_csv(a.root/protocol/'per_seed.csv',[r['metrics'] for r in records])
    assert actual==expected
    save_json(a.root/'artifact_validation.json',dict(status='passed',runs=len(actual),total_updates=2000000*len(actual)))
    save_json(a.root/'run_metadata.json',dict(status='complete',runs=len(actual),seconds=time.monotonic()-start))
    print(json.dumps(dict(status='complete',runs=len(actual),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
