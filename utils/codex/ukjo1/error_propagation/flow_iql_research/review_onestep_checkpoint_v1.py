"""Independent read-only fixed-checkpoint CPU50/model provenance review.

This utility reports one evaluation checkpoint, never a last-three mean or a
full-goal pass. It changes no learner, queue, threshold or canonical score.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time
import numpy as np


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def leaves(tree,path=''):
    if isinstance(tree,dict):
        for key,value in sorted(tree.items()):yield from leaves(value,path+'/'+str(key))
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree):yield from leaves(value,path+'/'+str(index))
    elif tree is not None:yield path,np.asarray(tree)


def equal_trees(first,second):
    return first.keys()==second.keys()and all(np.array_equal(value,second[key])for key,value in first.items())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--step',type=int,choices=(800000,900000,1000000),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();root=args.root.resolve();step=args.step
    continuation=json.loads((root/'CONTINUATION.json').read_text())
    contract_path=Path(continuation['analysis_contract']);contract=json.loads(contract_path.read_text())
    registration=Path(contract['registration']);assert sha(registration)==contract['registration_sha256']
    plan=json.loads(registration.read_text());research=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    jobs={row['id']:row for row in research['jobs']};rows=[];bases=[];networks=[];episodes=[];configs=[];hashes={};probes={}
    source_directory=Path(__file__).resolve().parent
    for rid in args.runs:
        assert Path(rid).name==rid and rid in jobs
        run=root/'runs'/rid;config_path=run/'config.json';conf=json.loads(config_path.read_text())
        identity=conf['identity'];assert identity==research['identities'][rid]
        entry=next(value for value in plan['implementation_definitions'].values()if value['runner_module']==jobs[rid]['module'])
        assert conf['source_sha256']==plan['implementation_training_source_sha256'][entry['tag']]
        for path,digest in conf['source_sha256'].items():
            assert sha(path)==digest,path
            source=Path(path);copy=run/'source'/('vendor_'+source.name if source.parent.name=='vendor'else source.name)
            assert sha(copy)==digest,copy
        checkpoint=run/f'checkpoint_{step}.pkl';digest=sha(checkpoint)
        with checkpoint.open('rb')as stream:saved=pickle.load(stream)
        assert saved['step']==step and saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
        state=saved['state'];base=state['base']
        assert int(base['q_updates'])==int(state['actor_updates'])==step and int(base['v_updates'])==0
        for optimizer,want in ((base['agent']['network']['opt_state'],step),(state['actor']['opt_state'],step),(base['v_opt'],0)):
            counts=[int(value)for key,value in leaves(optimizer)if key.endswith('/count')]
            assert counts and all(value==want for value in counts),(rid,counts,want)
        assert all(np.isfinite(value).all()for _,value in leaves(state))
        actor_path=run/f'actor_{step}.pkl';actor_digest=sha(actor_path)
        with actor_path.open('rb')as stream:actor=pickle.load(stream)
        assert actor['identity']==identity and actor['source_sha256']==conf['source_sha256']
        assert actor['step']==actor['actor_updates']==actor['Q_updates']==step and actor['checkpoint_sha256']==digest
        assert actor['config_sha256']==sha(config_path)
        assert equal_trees(dict(leaves(actor['actor_params'])),dict(leaves(state['actor']['params'])))
        meta=actor['metadata'];assert meta['actor_forward_passes']==1 and meta['flow_integration_steps']==0
        assert meta['generated_candidate_actions']==0 and not meta['requires_critic']and not meta['requires_BC_flow']
        evaluation={}
        for mode in ('primary','teacher'):
            path=root/'analysis/evaluations_v1'/f'{rid}_{step}_{mode}_n50.json';raw=json.loads(path.read_text())
            assert raw['run']==rid and raw['recipe']==identity['recipe']and raw['environment']==identity['environment']
            assert raw['step']==step and raw['n_episodes']==50 and raw['policy_mode']==mode
            assert raw['training_seed']==identity['seed']and raw['first_episode_seed']==42 and raw['eval_device']=='cpu'
            assert raw['protocol']=='last3_cpu50_independent_seeded_episodes_v1'and raw['no_optimizer_updates']
            assert raw['checkpoint_sha256']==digest and raw['actor_artifact_sha256']==actor_digest and raw['config_sha256']==sha(config_path)
            for key in ('episode_successes','episode_rewards','episode_lengths'):assert len(raw[key])==50
            assert all(x in (0,1)for x in raw['episode_successes'])and abs(sum(raw['episode_successes'])/50-raw['success_rate'])<1e-12
            assert abs(statistics.mean(raw['episode_rewards'])-raw['reward_mean'])<1e-8
            assert all(np.isfinite(value)for key in ('episode_rewards','episode_lengths')for value in raw[key])
            assert all(int(value)==value and value>0 for value in raw['episode_lengths'])
            core_path=source_directory/(entry['core_module'].rsplit('.',1)[1]+'.py')
            eval_path=source_directory/(entry['evaluator_module'].rsplit('.',1)[1]+'.py')
            assert raw['core_source_sha256']==conf['source_sha256'][str(core_path)]
            assert raw['source_sha256']==plan['all_source_sha256'][str(eval_path)]==sha(eval_path)
            if mode=='primary':assert raw['learned_model_trees_loaded']==['actor_params']and raw['deployment_metadata']==meta
            else:episodes.append({key:raw[key]for key in ('episode_successes','episode_rewards','episode_lengths')})
            hashes[str(path)]=sha(path);evaluation[mode]=raw
        primary=evaluation['primary']['episode_successes'];teacher=evaluation['teacher']['episode_successes']
        row=dict(run=rid,recipe=identity['recipe'],environment=identity['environment'],seed=identity['seed'],step=step,
            actor_CPU50=evaluation['primary']['success_rate'],original_best9_CPU50=evaluation['teacher']['success_rate'],
            actor_wins=sum(a>b for a,b in zip(primary,teacher)),teacher_wins=sum(a<b for a,b in zip(primary,teacher)),
            Q_Adam=step,actor_Adam=step,V_Adam=0,export_parameters_exact=True,full_state_finite=True)
        text=(run/'diagnostics.jsonl').read_text()
        if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
        probe=next(json.loads(line)for line in text.splitlines()if json.loads(line)['step']==step)
        probes[rid]={key:probe[key]for key in ('step','actor_to_best9_MSE','actor_to_flow1_MSE','raw_actor_saturation_fraction','same_state_action_variance','predicted_Q_regret_mean')}
        rows.append(row);bases.append(dict(leaves(base)));networks.append(dict(leaves(base['agent']['network']['params'])))
        configs.append({key:value for key,value in identity['config'].items()if key!='onestep_specification'})
        hashes.update({str(checkpoint):digest,str(actor_path):actor_digest,str(config_path):sha(config_path)})
    assert len({(row['environment'],row['seed'])for row in rows})==1
    result=dict(time=time.time(),status='audited',rows=rows,all_baseNN_bitwise_equal=all(equal_trees(networks[0],x)for x in networks[1:]),
        all_full_base_states_bitwise_equal=all(equal_trees(bases[0],x)for x in bases[1:]),non_actor_configs_equal=all(x==configs[0]for x in configs[1:]),
        original_teacher_all50_arrays_equal=all(x==episodes[0]for x in episodes[1:]),fixed_probe_records=probes,
        raw_evidence_sha256=hashes,review_source_sha256=sha(__file__),registration_sha256=sha(registration),
        scope='One checkpoint/environment/seed, CPU50. Not a last3 mean, nine-task/fresh-seed or isolated-speed result. No model, source, queue, threshold or score changes.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    with args.output.with_suffix('.csv').open('w')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({key:result[key]for key in ('rows','all_baseNN_bitwise_equal','all_full_base_states_bitwise_equal','original_teacher_all50_arrays_equal')}),flush=True)


if __name__=='__main__':main()
