"""Audit the original/.999 discount by native/V2 Gaussian comparison."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit import published_alpha
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


TASKS=('ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene')
STEPS=(800000,900000,1000000)


def finite(value):
    if isinstance(value,dict):return all(finite(item)for item in value.values())
    if isinstance(value,(tuple,list)):return all(finite(item)for item in value)
    return value is None or bool(np.isfinite(np.asarray(value)).all())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();verifier=Verifier();groups=[];configs={};checkpoint_count=0
    for discount in ('original',.999):
        for profile in ('official','bounded_v2'):
            rows=[]
            for task in TASKS:
                prefix='w1'if discount=='original'or profile=='bounded_v2'or task in ('cube','scene')else'w3'
                label='orig'if discount=='original'else'g999'
                name=f'{prefix}_{task}_{profile}_{label}_t90_s0';run=root/'runs'/name;path=run/'config.json'
                if not path.exists():
                    rows.append(dict(task=task,run=name,status='not_started',last3_mean=None));continue
                config=json.loads(path.read_text());identity=config['identity'];cfg=identity['config'];env=identity['environment']
                assert identity['kind']=='iql'and identity['seed']==0 and identity['profile']==profile
                assert cfg['expectile']==.9 and cfg['actor_loss']=='awr'and cfg['const_std']
                assert not cfg['actor_layer_norm']and cfg['layer_norm']and cfg['alpha']==published_alpha(env)
                width=256 if task in ('cube','cubed','pz3','pz4','scene')else 512
                assert list(cfg['actor_hidden_dims'])==list(cfg['value_hidden_dims'])==[width]*4
                expected=.995 if task in ('hmmed','hmlarge','soccer')else .99
                assert cfg['discount']==(expected if discount=='original'else .999)
                for source,digest in config['source_sha256'].items():assert verifier.digest(source)==digest,source
                configs[name]=config
                status='complete'if(run/'completed.json').exists()else'staged_resumable'if(run/'staging_request.json').exists()else'incomplete'
                if status!='complete':
                    rows.append(dict(task=task,run=name,status=status,last3_mean=None));continue
                done=json.loads((run/'completed.json').read_text());multiplier=2 if profile=='bounded_v2'else 1
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==multiplier*1000000
                records=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()];scores={};hashes={}
                for step in STEPS:
                    found=[record for record in records if record['step']==step];assert len(found)==1
                    record=found[0];assert record['n_episodes']==50 and record['first_episode_seed']==42
                    assert record['eval_device']=='cpu'and record['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
                    for field in ('episode_successes','episode_rewards','episode_lengths'):
                        assert len(record[field])==50 and np.isfinite(record[field]).all()
                    assert set(record['episode_successes'])<={0.,1.}
                    assert abs(statistics.mean(record['episode_successes'])-record['success_rate'])<1e-12
                    checkpoint=run/f'checkpoint_{step}.pkl';hashes[step]=verifier.digest(checkpoint)
                    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
                    assert json.loads(json.dumps(saved['identity']))==identity
                    assert saved['source_sha256']==config['source_sha256']
                    state=saved['state'];assert saved['step']==int(state['q_updates'])==step
                    assert int(state['v_updates'])==multiplier*step
                    assert int(state['agent']['network']['opt_state']['0']['count'])==step
                    assert int(state['v_opt']['0']['count'])==(2*step if profile=='bounded_v2'else 0)
                    assert finite(state);checkpoint_count+=1;scores[step]=record['success_rate']
                rows.append(dict(task=task,environment=env,run=name,status=status,last3_mean=statistics.mean(scores.values()),
                    per_step=scores,checkpoint_sha256=hashes,evaluation_sha256=verifier.digest(run/'evaluations.jsonl'),config_sha256=verifier.digest(path)))
            completed=sum(row['last3_mean']is not None for row in rows)
            groups.append(dict(discount=discount,profile=profile,rows=rows,completed_tasks=completed,
                nine_task_mean=statistics.mean(row['last3_mean']for row in rows)if completed==9 else None))
    paired=[]
    for task in TASKS:
        values={}
        for group in groups:
            row=next(row for row in group['rows']if row['task']==task)
            values[(group['discount'],group['profile'])]=row['last3_mean']
        if all(value is not None for value in values.values()):
            original=values[('original','bounded_v2')]-values[('original','official')]
            stress=values[(.999,'bounded_v2')]-values[(.999,'official')]
            paired.append(dict(task=task,V2_minus_native_original_gamma=original,V2_minus_native_gamma999=stress,
                               discount_by_value_intervention_interaction=stress-original))
    output=dict(time=time.time(),kind='Gaussian_discount_value_factorial_raw_audit',training_seed=0,
        primary_steps=STEPS,episodes=50,groups=groups,complete_four_cell_task_interactions=paired,
        retained_checkpoints_audited=checkpoint_count,
        limits=['OriginalgammaV2isnotnine-taskcompleteuntilallmissing/stagedjobsfinish. No zero or partial mean substitutes for it.',
                'Eachcellhasonecommontrainingseed;taskinteractionsaredescriptive,notindependent-seedconfirmation.',
                'Source/config,publicactorhyperparameters,actualAdammoments/countersandallstatefinitenesswereverifiedatallretainedlatecheckpoints.'],goal_complete=False)
    target=root/'analysis/Gaussian_discount_value_factorial_v1.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(output,indent=2,allow_nan=False));temporary.replace(target)
    with(root/'analysis/Gaussian_discount_value_factorial_v1.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['discount','profile','task','status','last3_mean'])
        for group in groups:
            for row in group['rows']:writer.writerow([group['discount'],group['profile'],row['task'],row['status'],row['last3_mean']if row['last3_mean']is not None else'pending'])
    print(json.dumps(dict(groups=[dict(discount=group['discount'],profile=group['profile'],complete=group['completed_tasks'],mean=group['nine_task_mean'])for group in groups],checkpoints=checkpoint_count,interactions=paired)),flush=True)


if __name__=='__main__':main()
