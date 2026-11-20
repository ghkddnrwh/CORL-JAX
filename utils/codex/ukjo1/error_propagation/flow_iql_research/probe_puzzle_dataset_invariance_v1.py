"""Current loader invariance and reward coverage under two global RNG states.

No rollout evaluation or learning. This does not retrospectively hash in-memory
arrays used by past training processes, which did not save those fingerprints.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import ogbench
import ogbench.utils


def sha(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):hasher.update(block)
    return hasher.hexdigest()


def describe(raw):
    data={k:np.asarray(raw[k],np.float32)for k in ('observations','actions','next_observations','rewards','masks')}
    data['actions']=np.clip(data['actions'],-1+1e-5,1-1e-5)
    fields={}
    for key,value in data.items():
        assert np.isfinite(value).all()
        contiguous=np.ascontiguousarray(value)
        fields[key]=dict(shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest())
    reward,mask=data['rewards'],data['masks']
    assert np.array_equal(mask==0,reward==0)
    values,counts=np.unique(reward,return_counts=True)
    return dict(fields=fields,size=len(reward),goal_zero_reward_rows=int(np.sum(reward==0)),
        zero_bootstrap_mask_rows=int(np.sum(mask==0)),
        goal_zero_reward_fraction=float(np.mean(reward==0)),
        reward_histogram={str(float(v)):int(n)for v,n in zip(values,counts)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    root=args.root.resolve();prior=root/'analysis/reproduction_environment_v1'
    manifest=json.loads((prior/'manifest.json').read_text())
    for name in ('datasets.json','installed_ogbench_sources_assets.json'):
        assert sha(prior/name)==manifest['files'][name]['sha256']
    datasets=json.loads((prior/'datasets.json').read_text())
    installed=json.loads((prior/'installed_ogbench_sources_assets.json').read_text())
    source_root=Path(ogbench.__file__).resolve().parent
    selected=('utils.py','relabel_utils.py','manipspace/envs/puzzle_env.py','manipspace/envs/manipspace_env.py')
    sources={}
    for name in selected:
        file=source_root/name;digest=sha(file)
        reference=next(row for row in installed['files']if row['path']=='ogbench/'+name)
        assert digest==reference['sha256'],name
        sources[str(file)]=digest
    rows=[];raw_pins={}
    for environment in ('puzzle-3x3-play-singletask-v0','puzzle-4x4-play-singletask-v0'):
        for row in datasets['datasets']:
            if row['environment']==environment:
                assert sha(row['path'])==row['sha256'],row['path']
                raw_pins[row['path']]=row['sha256']
        repetitions=[]
        for seed in (0,3):
            np.random.seed(seed);random.seed(seed)
            env,train,val=ogbench.make_env_and_datasets(environment)
            task=dict(resolved_reward_task_id=int(env.unwrapped._reward_task_id),
                task_name=env.unwrapped.cur_task_info['task_name'],
                goal_button_states=np.asarray(env.unwrapped._target_button_states).tolist())
            repetitions.append(dict(global_seed=seed,task=task,train=describe(train),validation=describe(val)))
            env.close();del env,train,val;gc.collect()
        for field in ('task','train','validation'):
            assert repetitions[0][field]==repetitions[1][field],(environment,field)
        rows.append(dict(environment=environment,current_two_loads_exact=True,repetitions=repetitions))
        print(json.dumps(dict(environment=environment,train_goal_fraction=repetitions[0]['train']['goal_zero_reward_fraction'],
            validation_goal_fraction=repetitions[0]['validation']['goal_zero_reward_fraction'],two_loads_exact=True)),flush=True)
    result=dict(time=time.time(),status='complete',rows=rows,source_sha256=sources,raw_dataset_sha256=raw_pins,
        diagnostic_source_sha256=sha(__file__),earlier_provenance_manifest_sha256=sha(prior/'manifest.json'),
        learning_updates=0,benchmark_evaluations=0,goal_complete=False,
        limits=['Currentloader/source/data fingerprints, not retrospective proof of all historical in-memory training inputs.',
            'Two global-RNG states were tested; source shows fixed reward-task targets. This does not test every possible reset path.',
            'Dataset goal coverage alone does not establish why a particular trained critic or policy fails.',
            'Rare or absent terminal rows do not make learned conditional residuals observable.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
