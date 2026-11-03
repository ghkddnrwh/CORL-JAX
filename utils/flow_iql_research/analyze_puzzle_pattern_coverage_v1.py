"""Read-only pattern-stratified support diagnostic, never training data selection."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patterns(obs):
    assert obs.shape[-1]==55
    button=obs[:,19:].reshape((-1,9,4))[:,:,:2]
    assert np.isin(button,[0.,1.]).all() and (button.sum(-1)==1).all()
    return (button[:,:,1]*(2**np.arange(9))).sum(-1).astype(int)


def main():
    root=Path(sys.argv[1]).resolve()
    area=root/'analysis/frozen_student_coverage_v1'
    output=area/'puzzle_pattern_coverage_v1.json'
    assert not output.exists()
    case=root/'analysis/broad_frozen_teacher_feasibility_v1/puzzle3_v1'
    parent=json.loads((case/'result.json').read_text())
    cache=case/'frozen_replay_pairs_v1.npz'
    assert sha(cache)==parent['pair_cache_sha256']
    with np.load(cache)as x:
        train=int(x['train_size']);reference=x['observations'][:train].copy();heldout=x['observations'][train:].copy()
    ids=patterns(reference);counts=np.bincount(ids,minlength=512)
    scale=np.maximum(reference.std(0),.001)
    groups={int(i):cKDTree(reference[ids==i].astype(np.float64)/scale) for i in np.unique(ids)}
    visited=dict(heldout_replay=heldout)
    inputs=[cache,case/'result.json',area/'puzzle3_v2/result.json']
    for name in ('teacher','bank_MSE','energy2','mixture9_NLL'):
        p=area/'puzzle3_v2'/(name+'_visited_states_v1.npz');inputs.append(p)
        with np.load(p)as x:visited[name]=x['observations'].copy()
    rows={}
    for name,obs in visited.items():
        pid=patterns(obs);distance=np.full(len(obs),np.nan)
        for pattern in np.unique(pid):
            if int(pattern)not in groups:continue
            mask=pid==pattern
            d,_=groups[int(pattern)].query(obs[mask].astype(np.float64)/scale,k=1,workers=1)
            distance[mask]=d/np.sqrt(reference.shape[-1])
        finite=distance[np.isfinite(distance)]
        rows[name]=dict(states=len(obs),unique_button_patterns=len(np.unique(pid)),unseen_pattern_fraction=float(np.mean(counts[pid]==0)),
            mean_training_examples_with_same_pattern=float(np.mean(counts[pid])),min_training_examples_with_same_pattern=int(np.min(counts[pid])),
            same_pattern_NN_mean=float(finite.mean())if len(finite)else None,
            same_pattern_NN_median=float(np.median(finite))if len(finite)else None,
            same_pattern_NN_p90=float(np.quantile(finite,.9))if len(finite)else None)
    env=Path('/home/ukjo1/anaconda3/envs/ogbench/lib/python3.10/site-packages/ogbench/manipspace/envs/puzzle_env.py')
    result=dict(time=time.time(),status='complete',inputs=[dict(path=str(p),sha256=sha(p))for p in inputs],
        source_sha256=sha(__file__),observation_layout_source=str(env),observation_layout_source_sha256=sha(env),
        layout='OGBenchPuzzle3state55:19proprioception+9*(onehot2,scaled_buttonpos1,buttonvel1). Every observation validated as onehot.',
        reference_states=len(reference),reference_patterns=int(np.count_nonzero(counts)),training_pattern_counts=counts.tolist(),rows=rows,
        optimizer_or_teacher_updates=0,training_uses_rollout_states=False,
        scope='Stratified descriptive support proxy from frozen-policy10episode traces. Not an OOD oracle, causal conclusion or goal performance. Environment-specific features used only for diagnosis.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
