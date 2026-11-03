"""High-tau auxiliary CPU resume check; no benchmark learning run."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import numpy as np


def identical(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False);v=Verifier()
    sources={p:v.digest(p) for p in ('utils/flow_iql_research/train_value_sampling_v1.py',
        'utils/flow_iql_research/value_sampling_core_v1.py',__file__)}
    base=[sys.executable,'-u','-m','utils.flow_iql_research.train_value_sampling_v1',
        '--env','puzzle-3x3-play-singletask-v0','--kind','ifql','--profile','bounded_v2_fresh_batches',
        '--seed','3','--discount','0.999','--expectile','0.99','--samples','9','--steps','1000000',
        '--episodes','50','--train-device','cpu','--chunk','2','--aux-eval-steps','']
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    for name,endpoint in (('full',8),('split',4),('split',8)):
        for p,d in sources.items():assert v.digest(p)==d,p
        command=base+['--output',str(out/name),'--stop-at',str(endpoint)]
        with(out/f'{name}_{endpoint}.log').open('w')as log:
            subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    with(out/'full/latest.pkl').open('rb')as stream:full=pickle.load(stream)
    with(out/'split/latest.pkl').open('rb')as stream:split=pickle.load(stream)
    assert full['identity']==split['identity'] and full['source_sha256']==split['source_sha256']
    assert full['step']==split['step']==8
    assert identical(full['state'],split['state'])
    assert identical(full['replay_rng'],split['replay_rng'])
    assert identical(full['value_replay_rng'],split['value_replay_rng'])
    assert int(full['state']['q_updates'])==8 and int(full['state']['v_updates'])==16
    assert int(full['state']['agent']['network']['opt_state']['0']['count'])==8
    assert int(full['state']['v_opt']['0']['count'])==16
    dataset=json.loads((out/'full/dataset.json').read_text());count=dataset['size']
    for seed,key in ((3,'replay_rng'),(273512,'value_replay_rng')):
        rng=np.random.RandomState(seed)
        for _ in range(4):rng.randint(count,size=(2,256))
        assert identical(rng.get_state(),full[key])
    assert not identical(full['replay_rng'],full['value_replay_rng'])
    assert not(out/'full/evaluations.jsonl').exists() and not(out/'split/evaluations.jsonl').exists()
    assert 'RESUME 4' in (out/'split_8.log').read_text()
    assert full['identity']['config']['expectile']==.99
    record=dict(time=time.time(),status='passed',expectile=.99,sources=sources,full_state_arrays_exact=True,
        independent_replay_streams_exact=True,private_replay_stream_advance_exact=True,checkpoint_steps=[4,8],
        Q_Adam_updates=8,V_Adam_updates=16,new_benchmark_evaluations=0,new_GPU_training=0,
        full_latest_sha256=v.digest(out/'full/latest.pkl'),split_latest_sha256=v.digest(out/'split/latest.pkl'),
        limits=['EightCPUupdates are a runner/state test, not a trained benchmark model or GPU-convergence claim.'],goal_complete=False)
    (out/'validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
