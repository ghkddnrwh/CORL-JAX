"""Last-three scoring without zero filling, mixed task seeds or best-step selection."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


ENVS=('antmaze-large-navigate-singletask-v0','humanoidmaze-medium-navigate-singletask-v0',
      'humanoidmaze-large-navigate-singletask-v0','antsoccer-arena-navigate-singletask-v0',
      'cube-single-play-singletask-v0','cube-double-play-singletask-v0',
      'puzzle-3x3-play-singletask-v0','puzzle-4x4-play-singletask-v0','scene-play-singletask-v0')
STEPS=(800000,900000,1000000)


def records(path):
    if not path.exists():return []
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines]


def published_alpha(env):
    return 3. if 'puzzle-4x4' in env else .3 if 'cube-double' in env else 1. if any(s in env for s in ('cube-single','antsoccer')) else 10.


def audit(root):
    rows=[]
    for run in sorted((root/'runs').glob('*')):
        cp=run/'config.json'
        if not cp.exists():continue
        conf=json.loads(cp.read_text());ident=conf['identity'];args=conf['arguments'];cfg=ident['config']
        env=ident['environment'];kind=ident['kind']
        group=dict(kind=kind,profile=ident['profile'],discount=args.get('discount') or 'original',
            expectile=cfg['expectile'],width=args.get('width') or 'paper')
        if kind not in ('fql','rebrac'):
            group['gaussian_alpha']='paper' if cfg['alpha']==published_alpha(env) else cfg['alpha']
        else:
            group['competitor_hyperparameters']='paper_Table6'
        if kind=='awrflow':group.update(flow_temperature=cfg['flow_temperature'],num_samples=cfg['num_samples'])
        if kind=='distilled':group.update(distill_coefficient=cfg['distill_coefficient'],normalize_advantage=cfg['normalize_advantage'])
        if kind=='fql_policy':group.update(distill_coefficient='paper_Table6_FQL',normalize_advantage=False)
        if kind=='ifql':group.update(num_samples=cfg['num_samples'])
        if kind in ('gaussian_bc','mixture_bc'):group.update(num_samples=cfg['num_samples'],mixture_components=cfg['mixture_components'])
        es=records(run/'evaluations.jsonl');ds=records(run/'diagnostics.jsonl')
        selected=[];errors=[]
        for step in STEPS:
            found=[e for e in es if e['step']==step]
            if len(found)>1:errors.append(f'duplicate evaluation at{step}')
            if len(found)!=1:continue
            e=found[0]
            try:
                assert e['n_episodes']==50 and len(e['episode_successes'])==50
                assert set(e['episode_successes']) <= {0.,1.}
                assert abs(statistics.mean(e['episode_successes'])-e['success_rate'])<1e-12
                assert e['protocol']=='last3_cpu50_independent_seeded_episodes_v1' and e['eval_device']=='cpu'
                assert (run/f'checkpoint_{step}.pkl').exists()
                selected.append(e)
            except AssertionError:errors.append(f'invalid primary evaluation at{step}')
        complete=(run/'completed.json').exists()
        stopped=json.loads((run/'stopped.json').read_text()) if (run/'stopped.json').exists() else None
        if complete:
            done=json.loads((run/'completed.json').read_text())
            expected=0 if kind in ('fql','rebrac') else 2000000 if 'v2' in ident['profile'] else 1000000
            if done['step']!=1000000 or done['q_updates']!=1000000 or done['v_updates']!=expected:
                errors.append('unexpected optimizer update counts')
        valid=complete and len(selected)==3 and not errors
        row=dict(run=run.name,environment=env,seed=ident['seed'],group=group,
                 complete=complete,valid_primary=valid,errors=errors,
                 stopped=stopped,
                 terminal_state_known_nonfinite=bool(stopped and stopped.get('reason')=='nonfinite_state'),
                 last_training_step=ds[-1]['step'] if ds else 0,
                 last_evaluation_step=es[-1]['step'] if es else None,
                 last_evaluation_success=es[-1]['success_rate'] if es else None,
                 last3_mean=statistics.mean(e['success_rate'] for e in selected) if valid else None,
                 successes_800k_900k_1m={str(e['step']):e['success_rate'] for e in selected},
                 q_rms=ds[-1].get('q_rms') if ds else None,
                 all_observed_probes_finite=all(d['finite'] for d in ds) if ds else None,
                 max_observed_q_rms_ratio=max(d['q_rms_over_return_bound'] for d in ds) if ds else None,
                 evaluation_sha256=hashlib.sha256((run/'evaluations.jsonl').read_bytes()).hexdigest() if es else None)
        rows.append(row)
    groups=collections.defaultdict(list)
    for r in rows:groups[json.dumps(r['group'],sort_keys=True)].append(r)
    macros=[]
    for g,gr in groups.items():
        valid=[r for r in gr if r['valid_primary']]
        bycell=collections.defaultdict(list)
        for r in valid:bycell[(r['environment'],r['seed'])].append(r)
        duplicates=[dict(environment=e,seed=s,runs=[r['run'] for r in rr]) for (e,s),rr in bycell.items() if len(rr)>1]
        common=set.intersection(*[{r['seed'] for r in valid if r['environment']==env} for env in ENVS])
        seed_scores={}
        if not duplicates:
            for s in sorted(common):seed_scores[str(s)]=statistics.mean(bycell[(e,s)][0]['last3_mean'] for e in ENVS)
        scores=list(seed_scores.values())
        macros.append(dict(group=json.loads(g),completed_cells=len(valid),planned_or_started_cells=len(gr),
            duplicates=duplicates,common_complete_seeds=sorted(common),per_seed_nine_task_mean=seed_scores,
            nine_task_mean=statistics.mean(scores) if scores else None,
            sample_SD=statistics.stdev(scores) if len(scores)>1 else None,
            exceeds_all_provided_ReBRAC_FQL=statistics.mean(scores)>.47 if scores else None))
    return dict(time=time.time(),primary_steps=STEPS,rows=rows,macros=macros,
        valid_primary_cells=sum(r['valid_primary'] for r in rows),
        caveat='Intermediate scores are progress only. Nine-task means require all nine environments in each common complete training seed; missing runs are unobserved.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=audit(args.root);out=args.root/'analysis';out.mkdir(exist_ok=True)
    temp=out/'live_audit.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(out/'live_audit.json')
    fields=('run','environment','seed','last_training_step','last_evaluation_step','last_evaluation_success','last3_mean','valid_primary','q_rms')
    with (out/'live_table.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result['rows'])
    print(json.dumps(dict(runs=len(result['rows']),valid_primary_cells=result['valid_primary_cells'],complete_macros=[m for m in result['macros'] if m['nine_task_mean'] is not None])))


if __name__=='__main__':main()
