"""Timing evidence: clipping saturation cannot explain every early failure."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();assert not args.output.exists()
    parent_path=root/'analysis/mean_backup_online_mechanism_v1.json';parent=json.loads(parent_path.read_text());rows=[]
    for row in parent['rows']:
        families={}
        for mode,data in row['families'].items():
            run=root/'runs'/data['run'];path=run/'diagnostics.jsonl';assert helper.digest(path)==data['log_sha256']
            logs=[json.loads(x)for x in path.read_text().splitlines()];by_step={d['step']:d for d in logs};assert len(by_step)==1000
            evaluations=[json.loads(x)for x in(run/'evaluations.jsonl').read_text().splitlines()];policy=[]
            for e in evaluations:
                helper.check_episodes(e,e['step']);d=by_step[e['step']]
                policy.append(dict(step=e['step'],success=e['success_rate'],is_primary=e['step']in helper.STEPS,
                    negative_mean_Q_over_abs_lower=-d['q_mean']/abs(data['known_return_lower']),lower_clip_fraction=d['backup/lower_clip_fraction']))
            assert sorted(e['step']for e in policy if e['is_primary'])==list(helper.STEPS)
            families[mode]=dict(run=data['run'],log_sha256=data['log_sha256'],evaluation_sha256=helper.digest(run/'evaluations.jsonl'),
                series=dict(step=[d['step']for d in logs],negative_mean_Q_over_abs_lower=[-d['q_mean']/abs(data['known_return_lower'])for d in logs],
                    lower_clip_fraction=[d['backup/lower_clip_fraction']for d in logs]),policy_points=policy)
        rows.append(dict(short=row['short'],environment=row['environment'],families=families,
            min_primary_last3=row['min_last3'],mean_primary_last3=row['mean_last3']))
    hmm=next(r for r in rows if r['short']=='hmmed');early=next(p for p in hmm['families']['min']['policy_points']if p['step']==100000)
    assert early['success']==0 and early['lower_clip_fraction']==0
    result=dict(time=time.time(),status='complete',rows=rows,parent_analysis_sha256=helper.digest(parent_path),source_sha256=helper.digest(__file__),
        nine_task_min_mean=parent['nine_task_min_mean'],nine_task_mean_mean=parent['nine_task_mean_mean'],primary_steps=list(helper.STEPS),
        auxiliary_policy_points_in_primary_metric=False,new_GPU_jobs=0,new_policy_evaluations=0,goal_complete=False,
        findings=['HMM minimum-backup success is already0at100K while lower clipping is0 and normalized negative meanQ~.417. Late floor saturation is not the sole cause of failure.',
            'Soccer minimum-backup success falls from.16at300K to near0 at late checkpoints while Q approaches its floor; this is a descriptive trajectory, not a separately randomized clipping intervention.',
            'Ant mean aggregation lowers magnitude/clipping yet has lower late success. Less saturation does not guarantee better policy ranking.',
            'The controlled aggregation effect remains established by exact BC/moment/RNG and actor-log checks; the causal pathway through intermediate saturation is not independently isolated.'],
        limits=['Auxiliary100K/300K outcomes are plotted only for temporal diagnosis. The primary last3 metric is unchanged.',
            'One discovery training seed, shared50resetseeds across checkpoints, no independent temporal replications.',
            'Fixed-query Q magnitudes, finite root fitting scores and clipping are not oracle true-Q error or B_k.',
            'No claim that removing the projection, changing bounds or changing aggregation universally improves performance.'])
    helper.atomic(args.output,result);print(json.dumps(dict(status='complete',tasks=len(rows),early_failure_precedes_clipping=True)),flush=True)


if __name__=='__main__':main()
