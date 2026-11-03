"""Recompute calibration mechanics from both cohorts' raw fixed-query Q tables."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import core
from utils.flow_iql_research.review_rank_mean_calibration_v1 import describe
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    root=args.root.resolve();old_path=root/'analysis/rank_mean_calibration_review_v1/review_v1.json'
    new_path=root/'analysis/fresh_independent_critic_calibration_v1/group1/rank_mean_review.json'
    old=json.loads(old_path.read_text());fresh=json.loads(new_path.read_text())
    assert old['status']==fresh['status']=='complete'
    assert old['source_sha256']==sha('utils/flow_iql_research/review_rank_mean_calibration_v1.py')
    assert fresh['source_sha256']==sha('utils/flow_iql_research/watch_fresh_critic_calibration_v1.py')
    for path,digest in old['inputs_sha256'].items():assert sha(path)==digest,path
    for path,digest in fresh['raw_table_sha256'].items():assert sha(path)==digest,path
    files=[{Path(p).name:Path(p)for p in old['inputs_sha256']if p.endswith('_values.npz')},
           {Path(p).name:Path(p)for p in fresh['raw_table_sha256']}]
    assert len(files[0])==len(files[1])==9 and set(files[0])==set(files[1])
    rows=[];index_checks=[]
    for name in sorted(files[0]):
        short=name.removesuffix('_values.npz')
        old_rows=[r for r in old['rows']if r['short']==short and r['split']=='validation']
        assert len(old_rows)==3
        environment=old_rows[0]['environment']
        with np.load(files[0][name],allow_pickle=False)as a,np.load(files[1][name],allow_pickle=False)as b:
            assert np.array_equal(a['validation_indices'],b['validation_indices'])
            index_checks.append(dict(environment=environment,validation_indices_exact=True,
                index_sha256=hashlib.sha256(a['validation_indices'].tobytes()).hexdigest()))
            for label,table,stored in [('discovery',a,old['rows']),('first_fresh',b,fresh['query_rows'])]:
                for step in (800000,900000,1000000):
                    q=table[f'validation_{step}_Q'];assert q.shape==(3,9,512)and np.isfinite(q).all()
                    actual=describe(q)
                    prior=next(r for r in stored if r['environment']==environment and r['split']=='validation'and r['step']==step)
                    assert actual=={key:prior[key]for key in actual},(label,environment,step)
                    rows.append(dict(cohort=label,environment=environment,step=step,**actual))
    summary=[]
    for label in ('discovery','first_fresh'):
        selected=[r for r in rows if r['cohort']==label];assert len(selected)==27
        summary.append(dict(cohort=label,raw_mean_choice_difference=statistics.mean(r['contrasts']['raw_mean']['choice_difference_fraction']for r in selected),
            rank_median_choice_difference=statistics.mean(r['contrasts']['rank_median']['choice_difference_fraction']for r in selected),
            rank_mean_top_tie_fraction=statistics.mean(r['rank_mean_top_tie_fraction']for r in selected),
            unique_top_majority_override_fraction=statistics.mean(r['rank_mean_overrides_unique_top_majority_fraction']for r in selected)))
    first_path=root/'analysis/fresh_group1_confirmation_summary_v1.json'
    first=json.loads(first_path.read_text());assert first['actual_first_group_models_verified']==81
    methods=first['nine_task_means']
    result=dict(time=time.time(),status='complete',cohorts=[[0,1,2],[3,4,5]],validation_rows=rows,
        validation_mechanics=summary,shared_validation_query_indices=index_checks,raw_tables_verified=18,
        discovery_rank_mean_minus_raw_mean=old['validation_aggregate']['nine_task_gain'],
        first_fresh_rank_mean_minus_raw_mean=methods['mean_rank']-methods['mean'],
        first_fresh_mean_rank=methods['mean_rank'],first_fresh_exceeds_plain_FQL=False,
        inputs_sha256={str(old_path):sha(old_path),str(new_path):sha(new_path),str(first_path):sha(first_path),
            **{str(p):sha(p)for group in files for p in group.values()}},
        source_sha256=sha(__file__),new_GPU_jobs=0,new_policy_evaluations=0,goal_complete=False,
        interpretation='The rank transformation changes a similar fraction of choices in both groups; the observed raw-to-rank return gain is much smaller in the fresh group. Mechanical invariance/change rates cannot certify correct action ranking.',
        limits=['One ensemble group per cohort, not three ensemble replications or a seed-level significance test.',
            'Same512validationindicesperenvironmentareusedacrosscohortsandsteps,notindependentstate-replicationrows.',
            'BC0andBC3generate different action proposals; this is not an entrywise Q-error comparison on the same actions.',
            'No oracle action ranking, conditional expectile residual or propagated B_k label is observed.'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(mechanics=summary,return_gain_discovery=result['discovery_rank_mean_minus_raw_mean'],return_gain_fresh=result['first_fresh_rank_mean_minus_raw_mean'],raw_tables=18)),flush=True)


if __name__=='__main__':main()
