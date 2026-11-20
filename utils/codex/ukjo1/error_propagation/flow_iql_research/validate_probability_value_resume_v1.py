"""Check exact resumed unit-MSE and unit-KL trajectories without GPU training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np


def identical(a,b):
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(identical(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)is type(b)and len(a)==len(b)and all(identical(x,y)for x,y in zip(a,b))
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path('utils/flow_iql_research/train_probability_value_v1.py'),Path('utils/flow_iql_research/probability_value_core_v1.py')]
    sources={str(p):sha(p)for p in paths}
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    rows=[]
    for mode in ('mse','kl'):
        base=[sys.executable,'-u','-m','utils.flow_iql_research.train_probability_value_v1',
            '--env','puzzle-3x3-play-singletask-v0','--kind','ifql','--profile',f'bounded_v2_unit_{mode}',
            '--seed','3','--discount','0.999','--expectile','0.9','--samples','9','--steps','1000000',
            '--episodes','50','--train-device','cpu','--chunk','2','--aux-eval-steps','']
        for label,endpoint in (('full',8),('split',4),('split',8)):
            for p,digest in sources.items():assert sha(p)==digest,p
            target=output/f'{mode}_{label}';log=output/f'{mode}_{label}_{endpoint}.log'
            with log.open('w')as stream:
                subprocess.run(base+['--output',str(target),'--stop-at',str(endpoint)],env=environment,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
        with(output/f'{mode}_full/latest.pkl').open('rb')as stream:full=pickle.load(stream)
        with(output/f'{mode}_split/latest.pkl').open('rb')as stream:split=pickle.load(stream)
        assert full['step']==split['step']==8
        assert identical(full,split)
        state=full['state'];assert int(state['q_updates'])==int(state['agent']['network']['opt_state']['0']['count'])==8
        assert int(state['v_updates'])==int(state['v_opt']['0']['count'])==16
        assert sha(output/f'{mode}_full/latest.pkl')==sha(output/f'{mode}_split/latest.pkl')
        for label in ('full','split'):assert not(output/f'{mode}_{label}/evaluations.jsonl').exists()
        assert 'RESUME 4'in(output/f'{mode}_split_8.log').read_text()
        rows.append(dict(mode=mode,continuous_vs_four_plus_four_checkpoint_SHA_exact=True,
            latest_sha256=sha(output/f'{mode}_full/latest.pkl'),Q_Adam=8,V_Adam=16,replay_rng_exact=True))
        print(json.dumps(dict(mode=mode,status='passed')),flush=True)
    record=dict(time=time.time(),status='passed',rows=rows,sources=sources,new_GPU_training=0,
        new_policy_evaluations=0,goal_complete=False,
        limits=['EightCPUupdates per trajectory test source/serialization/RNG/clocks, not benchmark convergence.'])
    (output/'validation.json').write_text(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':main()
