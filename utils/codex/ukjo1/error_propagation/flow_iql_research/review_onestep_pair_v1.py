"""Read-only last-three review with raw-episode/model provenance checks."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


def leaves(tree,path=''):
    if isinstance(tree,dict):
        for key,value in sorted(tree.items()):yield from leaves(value,path+'/'+str(key))
    elif isinstance(tree,(tuple,list)):
        for index,value in enumerate(tree):yield from leaves(value,path+'/'+str(index))
    elif tree is not None:yield path,np.asarray(tree)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--tag',required=True);parser.add_argument('--recipes',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();rows=[];audits=[];bases={};teachers={};hashes={}
    for recipe in args.recipes:
        run=args.root/'runs'/f'os1_{args.tag}_{recipe}_s0'
        conf=json.loads((run/'config.json').read_text());identity=conf['identity']
        for source,digest in conf['source_sha256'].items():assert sha(source)==digest,source
        scores={'primary':[],'teacher':[]};hashes[str(run/'config.json')]=sha(run/'config.json')
        for step in (800000,900000,1000000):
            checkpoint=run/f'checkpoint_{step}.pkl';digest=sha(checkpoint);hashes[str(checkpoint)]=digest
            with checkpoint.open('rb')as stream:saved=pickle.load(stream)
            assert saved['step']==step and saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
            state=saved['state'];base=state['base']
            assert int(state['actor_updates'])==int(base['q_updates'])==step and int(base['v_updates'])==0
            for optimizer in (state['actor']['opt_state'],base['agent']['network']['opt_state']):
                counts=[int(v)for k,v in leaves(optimizer)if k.endswith('/count')]
                assert counts and all(value==step for value in counts),counts
            assert all(np.isfinite(value).all()for _,value in leaves(state))
            actor_path=run/f'actor_{step}.pkl';actor_hash=sha(actor_path);hashes[str(actor_path)]=actor_hash
            with actor_path.open('rb')as stream:actor=pickle.load(stream)
            assert actor['checkpoint_sha256']==digest and actor['identity']==identity
            assert actor['step']==actor['actor_updates']==actor['Q_updates']==step
            exported=dict(leaves(actor['actor_params']));trained=dict(leaves(state['actor']['params']))
            assert exported.keys()==trained.keys()and all(np.array_equal(value,trained[k])for k,value in exported.items())
            bases[recipe,step]=dict(leaves(base['agent']['network']['params']))
            for mode in ('primary','teacher'):
                path=args.root/'analysis/evaluations_v1'/f'{run.name}_{step}_{mode}_n50.json'
                result=json.loads(path.read_text());hashes[str(path)]=sha(path)
                assert result['step']==step and result['n_episodes']==50 and result['policy_mode']==mode
                assert result['checkpoint_sha256']==digest and result['actor_artifact_sha256']==actor_hash
                assert result['config_sha256']==sha(run/'config.json')and result['no_optimizer_updates']
                assert len(result['episode_successes'])==50
                assert abs(statistics.mean(result['episode_successes'])-result['success_rate'])<1e-12
                scores[mode].append(result['success_rate'])
                if mode=='teacher':teachers[recipe,step]={k:result[k]for k in ('episode_successes','episode_rewards','episode_lengths')}
            audits.append(dict(recipe=recipe,step=step,Q_Adam=step,actor_Adam=step,V_updates=0,
                actor_export_parameters_exact=True,full_state_finite=True))
        row=dict(environment=identity['environment'],recipe=recipe,training_seed=0,
            actor_800K=scores['primary'][0],actor_900K=scores['primary'][1],actor_1M=scores['primary'][2],
            actor_last3_mean=statistics.mean(scores['primary']),teacher_800K=scores['teacher'][0],
            teacher_900K=scores['teacher'][1],teacher_1M=scores['teacher'][2],teacher_last3_mean=statistics.mean(scores['teacher']))
        row['paired_extraction_gap']=row['actor_last3_mean']-row['teacher_last3_mean'];rows.append(row)
    left,right=args.recipes;base_exact=True;teacher_exact=True
    for step in (800000,900000,1000000):
        a,b=bases[left,step],bases[right,step]
        base_exact&=a.keys()==b.keys()and all(np.array_equal(value,b[k])for k,value in a.items())
        teacher_exact&=teachers[left,step]==teachers[right,step]
    result=dict(time=time.time(),status='audited',rows=rows,actual_model_counter_audits=audits,
        baseNN_bitwise_equal_between_alphas_all3checkpoints=base_exact,teacher_all50_arrays_exact_between_alphas=teacher_exact,
        raw_evidence_sha256=hashes,review_source_sha256=sha(__file__),
        scope='One environment,seed0,last3CPU50. Not a nine-task or fresh-seed performance or isolated speed result.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    with args.output.with_suffix('.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(rows=rows,base_exact=base_exact,teacher_episodes_exact=teacher_exact)),flush=True)


if __name__=='__main__':main()
