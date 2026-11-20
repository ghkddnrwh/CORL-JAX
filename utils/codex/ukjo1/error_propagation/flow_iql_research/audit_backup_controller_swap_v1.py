"""Strict same-seed2x2 deployed BC/critic-source intervention audit."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def episodes(row):
    assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
    assert row['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
    for field in('episode_successes','episode_rewards','episode_lengths'):
        assert len(row[field])==50 and np.isfinite(row[field]).all()
    assert set(row['episode_successes'])<={0.,1.}
    assert abs(statistics.mean(row['episode_successes'])-row['success_rate'])<1e-12


def tree_difference(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max(tree_difference(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return max(tree_difference(x,y)for x,y in zip(a,b))
    return float(np.max(np.abs(np.asarray(a,np.float64)-np.asarray(b,np.float64))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    plan=json.loads(args.plan.read_text());v=Verifier();configs={};rows=[];valid={};replays=[]
    for source,digest in plan['source_sha256'].items():assert v.digest(source)==digest,source
    for label,pin in plan['pins'].items():
        run=Path(pin['run']);assert v.digest(run/'config.json')==pin['config_sha256']
        config=json.loads((run/'config.json').read_text());configs[label]=config
        assert config['identity']['seed']==0 and config['identity']['kind']=='ifql'
        assert config['identity']['steps']==1000000 and config['identity']['episodes']==50
        assert(run/'completed.json').exists()and not(run/'stopped.json').exists()
        done=json.loads((run/'completed.json').read_text())
        assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
        for path,digest in config['source_sha256'].items():assert v.digest(path)==digest,path
        primary=[json.loads(x)for x in(run/'evaluations.jsonl').read_text().splitlines()]
        for record in plan['diagonal_primary_records'][label]:
            assert record==next(x for x in primary if x['step']==record['step']);episodes(record)
            cp=run/f"checkpoint_{record['step']}.pkl";assert v.checkpoint(cp,config,record['step'])==pin['checkpoints'][str(record['step'])]
            valid[(label,label,record['step'])]=record['success_rate']
    common=[{key:value for key,value in configs[label]['identity']['config'].items()
             if key not in('backup_kind','backup_Q_aggregation')}for label in('min','mean')]
    assert common[0]==common[1]
    for step in(800000,900000,1000000):
        states=[]
        for label in('min','mean'):
            with(Path(plan['pins'][label]['run'])/f'checkpoint_{step}.pkl').open('rb')as stream:states.append(pickle.load(stream))
        for x,y in zip(states[0]['replay_rng'],states[1]['replay_rng']):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(states[0]['state']['agent']['rng'],states[1]['state']['agent']['rng'])
        first=states[0]['state']['agent']['network']['params'];second=states[1]['state']['agent']['network']['params']
        replays.append(dict(step=step,replay_RNG_exact=True,agent_RNG_exact=True,
            actual_BC_parameters_max_abs_difference=tree_difference(first['modules_actor_flow'],second['modules_actor_flow']),
            actual_Q_parameters_max_abs_difference=tree_difference(first['modules_critic'],second['modules_critic'])))
    for job in plan['jobs']:
        path=Path(job['output'])
        if not path.exists():continue
        result=json.loads(path.read_text());episodes(result);step=job['step']
        critic=next(label for label,pin in plan['pins'].items()if pin['run']==job['critic_run'])
        actor=next(label for label,pin in plan['pins'].items()if pin['run']==job['actor_run'])
        assert result['critic_run']==Path(job['critic_run']).name and result['actor_run']==Path(job['actor_run']).name
        assert result['training_seed']==0 and result['step']==result['actor_training_step']==step
        assert result['selection_samples']==9 and result['environment']=='cube-double-play-singletask-v0'
        assert result['actor_checkpoint_sha256']==plan['pins'][actor]['checkpoints'][str(step)]
        assert result['actor_config_sha256']==plan['pins'][actor]['config_sha256']
        assert result['critic_config_sha256']==plan['pins'][critic]['config_sha256']
        assert len(result['critic_members'])==1;member=result['critic_members'][0]
        assert member['step']==member['Q_updates']==step and member['V_updates']==2*step
        assert member['sha256']==plan['pins'][critic]['checkpoints'][str(step)]
        assert Path(member['checkpoint']).resolve()==(Path(job['critic_run'])/f'checkpoint_{step}.pkl').resolve()
        assert result['evaluator_source_sha256']==plan['source_sha256']['utils/flow_iql_research/evaluate_common_actor_temporal_v1.py']
        assert result['selector_source_sha256']==plan['source_sha256']['utils/flow_iql_research/temporal_selection_v1.py']
        assert not result['primary_evaluations_modified']and not result['mixed_learning_state_saved_or_trained']
        valid[(critic,actor,step)]=result['success_rate']
        rows.append(dict(id=job['id'],critic=critic,BC=actor,step=step,success=result['success_rate'],result_sha256=v.digest(path)))
    self_result=json.loads(Path(plan['self_reference']['output']).read_text());reference=json.loads(Path(plan['self_reference']['reference']).read_text())
    assert all(self_result['self_reference_episode_arrays_exact'].values())
    for field in('episode_successes','episode_rewards','episode_lengths'):assert self_result[field]==reference[field]
    table=[]
    for critic in('min','mean'):
        for actor in('min','mean'):
            cells={str(step):valid.get((critic,actor,step))for step in(800000,900000,1000000)}
            complete=all(value is not None for value in cells.values())
            table.append(dict(critic=critic,BC=actor,complete=complete,per_step=cells,last3_mean=statistics.mean(cells.values())if complete else None))
    complete=all(row['complete']for row in table);effects=None
    if complete:
        means={(row['critic'],row['BC']):row['last3_mean']for row in table}
        effects=dict(critic_mean_minus_min_with_min_BC=means[('mean','min')]-means[('min','min')],
            critic_mean_minus_min_with_mean_BC=means[('mean','mean')]-means[('min','mean')],
            BC_mean_vs_min_with_min_critic=means[('min','mean')]-means[('min','min')],
            BC_mean_vs_min_with_mean_critic=means[('mean','mean')]-means[('mean','min')],
            interaction=means[('mean','mean')]-means[('mean','min')]-means[('min','mean')]+means[('min','min')])
    result=dict(time=time.time(),status='complete'if complete else'incomplete',table=table,conditional_effects=effects,
        crossed_evaluations_audited=len(rows),crossed_rows=rows,self_reference_all50_arrays_exact=True,
        replay_and_parameter_checks=replays,unique_actual_model_checkpoints_audited=len(v.checkpoints),
        plan_sha256=v.digest(args.plan),source_sha256=v.digest(__file__),nine_task_mean=None,
        interpretation='ActualBC/Qparameterscrossedonlyforinference;conditionalcritic-sourceeffectwithBCfixedisidentifiedforthesecheckpointmodels/resetseeds.',
        limits=['Oneenvironment/seed;notanine-taskimprovementorindependenttrainingconfirmation.',
            'Thisisolatesdeployedparameters,nottrueexpectileresidual,oracleQerrororwhichtrainingupdatefirstcausedthegain.',
            'Critic-sourceandBC-sourceeffectscaninteract;unchangedreplay/RNGdoesnotimplybitwiseidenticalGPUlearnedBCparams.'],goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    with args.output.with_suffix('.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['critic','BC','800K','900K','1M','last3_mean'])
        for row in table:writer.writerow([row['critic'],row['BC'],*[row['per_step'][str(s)]for s in(800000,900000,1000000)],row['last3_mean']])
    print(json.dumps(dict(status=result['status'],table=table,effects=effects)),flush=True)


if __name__=='__main__':main()
