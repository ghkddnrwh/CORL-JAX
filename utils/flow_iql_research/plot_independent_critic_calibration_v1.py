"""Display descriptive calibration evidence without oracle-rank claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    out=args.root/'analysis/independent_critic_calibration_v1';source=out/'summary.json';data=json.loads(source.read_text())
    assert data['status']=='complete'and data['models_audited']==81 and len(data['rows'])==54
    envs=['antmaze-large-navigate-singletask-v0','humanoidmaze-medium-navigate-singletask-v0',
        'humanoidmaze-large-navigate-singletask-v0','antsoccer-arena-navigate-singletask-v0',
        'cube-single-play-singletask-v0','cube-double-play-singletask-v0','puzzle-3x3-play-singletask-v0',
        'puzzle-4x4-play-singletask-v0','scene-play-singletask-v0']
    labels=['AM-L','HM-M','HM-L','AS-A','Cube-S','Cube-D','Pz-3','Pz-4','Scene'];rows=[]
    for env,label in zip(envs,labels):
        cells=[row for row in data['rows']if row['environment']==env and row['split']=='validation'];assert len(cells)==3
        diff=statistics.mean(row['raw_mean_vs_rank_choice_difference_fraction']for row in cells)
        minority=statistics.mean(row['raw_mean_minority_magnitude_domination_fraction']for row in cells)
        rows.append(dict(environment=env,label=label,mean_vs_rank_choice_difference_fraction=diff,
            minority_magnitude_domination_fraction=minority,minority_fraction_of_differences=minority/diff if diff else None,
            three_argmax_agreement_fraction=statistics.mean(row['all_three_argmax_agreement_fraction']for row in cells),
            unique_majority_guarantee_fraction=statistics.mean(row['two_unique_argmax_majority_guarantee_fraction']for row in cells),
            median_score_top_tie_fraction=statistics.mean(row['median_score_top_tie_fraction']for row in cells),
            per_checkpoint_median_scale_ratio={str(row['step']):row['action_std_max_over_min_quantiles']['p50']for row in cells}))
    fig,(left,right)=plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={'width_ratios':[1.5,1.]})
    x=np.arange(9);minority=np.array([row['minority_magnitude_domination_fraction']for row in rows])
    rest=np.array([row['mean_vs_rank_choice_difference_fraction']for row in rows])-minority
    left.bar(x,minority,color='#b96925',label='One critic scale-dominates two opposite preferences')
    left.bar(x,rest,bottom=minority,color='#c9d3db',label='Other choice differences')
    left.set(xticks=x,xticklabels=labels,ylim=(0,.45),ylabel='Fraction of fixed validation queries')
    left.set_title('Raw mean vs median rank: action-choice differences',fontsize=10)
    left.grid(axis='y',alpha=.2);left.set_axisbelow(True)
    for step,offset,color in zip((800000,900000,1000000),(-.16,0,.16),('#2a6fbb','#b96925','#197b61')):
        right.scatter(x+offset,[row['per_checkpoint_median_scale_ratio'][str(step)]for row in rows],s=26,color=color,label=f'{step//1000}K')
    right.set(xticks=x,xticklabels=labels,ylim=(1,3),ylabel='Median max/min action-value SD across 3 critics')
    plt.setp(right.get_xticklabels(),rotation=30,ha='right')
    right.set_title('Within-state relative scales of the same 9 proposals',fontsize=10)
    right.grid(axis='y',alpha=.2);right.set_axisbelow(True);right.legend(loc='upper left',ncol=3,fontsize=8,frameon=False)
    fig.legend(*left.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=1,frameon=False,fontsize=8)
    fig.suptitle('Frozen critic calibration: descriptive evidence, no oracle action labels',fontsize=12)
    fig.text(.5,.02,'512 fixed validation states/task, reused at 800K/900K/1M. Same BC0 candidates for all 3 critics.\nBoth raw mean and ranks ignore state offsets; ranks also ignore per-critic monotone/relative-scale changes.\nThe minority may be correct. These diagnostics alone do not explain the causal source of a return gain.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.21,1,.95))
    for ext in('png','pdf'):fig.savefig(out/('comparison_v1.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    with(out/'comparison_v1.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['environment','mean_vs_rank_choice_difference_fraction',
            'minority_magnitude_domination_fraction','minority_fraction_of_differences','three_argmax_agreement_fraction','unique_majority_guarantee_fraction','median_score_top_tie_fraction'])
        for row in rows:writer.writerow([row[key]for key in('environment','mean_vs_rank_choice_difference_fraction','minority_magnitude_domination_fraction','minority_fraction_of_differences','three_argmax_agreement_fraction','unique_majority_guarantee_fraction','median_score_top_tie_fraction')])
    cells=[row for row in data['rows']if row['split']=='validation']
    diff=statistics.mean(row['raw_mean_vs_rank_choice_difference_fraction']for row in cells)
    minority=statistics.mean(row['raw_mean_minority_magnitude_domination_fraction']for row in cells)
    summary=dict(time=time.time(),rows=rows,equal_task_step_mean_choice_difference_fraction=diff,
        equal_task_step_mean_minority_magnitude_domination_fraction=minority,
        minority_fraction_among_differences=minority/diff,
        mean_three_argmax_agreement_fraction=statistics.mean(row['all_three_argmax_agreement_fraction']for row in cells),
        mean_unique_majority_guarantee_fraction=statistics.mean(row['two_unique_argmax_majority_guarantee_fraction']for row in cells),
        mean_top_rank_tie_fraction=statistics.mean(row['median_score_top_tie_fraction']for row in cells),
        all_actual_affine_rank_tests_passed=all(all(t['rank_choice_unchanged']for t in row['positive_affine_counterfactuals'])for row in data['rows']),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Descriptivecommonproposalsonseenmodels;notanoraclecriticrankorpolicyimprovementtest.',goal_complete=False)
    (out/'review_v1.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({key:summary[key]for key in('equal_task_step_mean_choice_difference_fraction','minority_fraction_among_differences','mean_three_argmax_agreement_fraction','mean_unique_majority_guarantee_fraction','mean_top_rank_tie_fraction')}))


if __name__=='__main__':main()
