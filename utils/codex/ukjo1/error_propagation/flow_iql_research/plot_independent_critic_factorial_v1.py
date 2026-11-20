"""Review all four common aggregation rules on the same discovery models."""
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
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    original=root/'analysis/independent_critic_fresh_confirmation_v1/discovery_audit.json'
    new=root/'analysis/independent_critic_factorial_v1/strict_audit_complete_v1.json'
    audits=[json.loads(path.read_text())for path in(original,new)]
    assert all(a['status']=='complete'and a['valid_evaluations']==54 and not a['errors']and a['unique_model_checkpoints_verified']==81 for a in audits)
    rawplans=[root/'independent_critic_policy_discovery_registration_v1.json',root/'independent_critic_factorial_registration_v1.json']
    plans=[json.loads(path.read_text())for path in rawplans]
    for env in plans[0]['environments']:
        for step in(800000,900000,1000000):
            original_job=next(j for j in plans[0]['jobs']if j['environment']==env and j['step']==step)
            other=next(j for j in plans[1]['jobs']if j['environment']==env and j['step']==step)
            assert original_job['member_pins']==other['member_pins']and original_job['actor_run']==other['actor_run']
    order=['mean','median_value','mean_rank','median_rank']
    names=['Raw Q mean','Raw Q median','Rank mean','Rank median (primary)'];methods={m['rule']:m for a in audits for m in a['methods']}
    envs=['antmaze-large-navigate-singletask-v0','humanoidmaze-medium-navigate-singletask-v0',
        'humanoidmaze-large-navigate-singletask-v0','antsoccer-arena-navigate-singletask-v0',
        'cube-single-play-singletask-v0','cube-double-play-singletask-v0','puzzle-3x3-play-singletask-v0',
        'puzzle-4x4-play-singletask-v0','scene-play-singletask-v0']
    labels=['AM-L','HM-M','HM-L','AS-A','Cube-S','Cube-D','Pz-3','Pz-4','Scene'];rows=[]
    for rule,name in zip(order,names):
        method=methods[rule];assert method['completed_tasks']==9
        perstep={str(s):statistics.mean(row['per_step'][str(s)]for row in method['tasks'])for s in(800000,900000,1000000)}
        rows.append(dict(rule=rule,name=name,nine_task_mean=method['nine_task_mean'],macro_per_step=perstep,
            macro_last3_sample_SD=statistics.stdev(perstep.values()),
            values=[next(row['last3_mean']for row in method['tasks']if row['environment']==env)for env in envs]))
    values={row['rule']:row['nine_task_mean']for row in rows}
    effects=dict(rank_transform_at_mean=values['mean_rank']-values['mean'],rank_transform_at_median=values['median_rank']-values['median_value'],
        median_at_raw_values=values['median_value']-values['mean'],median_at_ranks=values['median_rank']-values['mean_rank'],
        interaction=values['median_rank']-values['mean_rank']-values['median_value']+values['mean'])
    fig,axes=plt.subplots(1,3,figsize=(16,5.1),gridspec_kw={'width_ratios':[1,2.2,1.3]})
    matrix=np.asarray([[values['mean'],values['median_value']],[values['mean_rank'],values['median_rank']]])
    axes[0].imshow(matrix,vmin=0,vmax=.65,cmap='Blues');axes[0].set(xticks=[0,1],xticklabels=['Mean','Median'],yticks=[0,1],yticklabels=['Raw Q','Midranks'])
    for i in range(2):
        for j in range(2):axes[0].text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',fontsize=12,color='white'if matrix[i,j]>.42 else'black')
    axes[0].add_patch(plt.Rectangle((.5,.5),1,1,fill=False,edgecolor='#b96925',lw=3));axes[0].set_title('Equal nine-task mean',fontsize=10)
    x=np.arange(9);colors=['#7d8e9c','#597c37','#528a9e','#b96925'];width=.19
    for i,(row,color)in enumerate(zip(rows,colors)):
        axes[1].bar(x+(i-1.5)*width,row['values'],width,color=color,label=row['name'])
        axes[2].plot([800,900,1000],list(row['macro_per_step'].values()),'-o',color=color,label=row['name'])
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,1.05),ylabel='Last3 CPU50 success');plt.setp(axes[1].get_xticklabels(),rotation=25,ha='right')
    axes[1].set_title('Same 3 critics and first BC: all tasks',fontsize=10)
    axes[2].axhline(.47,color='#999999',ls='--',lw=1);axes[2].set(xlabel='Updates per member (thousands)',ylabel='Nine-task checkpoint mean',ylim=(.25,.7))
    axes[2].set_title('Temporal fluctuation remains visible',fontsize=10)
    for axis in axes[1:]:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.10),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Aggregation2x2 discovery ablation: same learners0/1/2, one ensemble group',fontsize=12)
    fig.text(.5,.02,'Each rule uses N9, tau=.9, gamma=.999 and identical retained models. All108 raw CPU50 evaluations and81 unique member models are audited.\nOrange outline is the previously locked primary recipe; this mechanism ablation does not change it. Each policy costs approximately3 base learners.\nThese are discovery outcomes and3 checkpoint values, not independent ensemble training replications or uncertainty intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.21,1,.94));out=root/'analysis/independent_critic_factorial_v1'
    for ext in('png','pdf'):fig.savefig(out/('comparison_v1.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    with(out/'comparison_v1.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*envs,'nine_task_mean','macro800K','macro900K','macro1M','macro_last3_sample_SD'])
        for row in rows:writer.writerow([row['name'],*row['values'],row['nine_task_mean'],*row['macro_per_step'].values(),row['macro_last3_sample_SD']])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result=dict(time=time.time(),rows=rows,factorial_contrasts=effects,same_actual_members_and_BC_verified=True,
        primary_rule_unchanged='median_rank',ensemble_groups=1,additional_GPU_training=0,
        inputs_sha256={str(p):sha(p)for p in[original,new,*rawplans]},source_sha256=sha(__file__),
        limits=['Onealready-seendiscoveryensemblegroup;notnewtrainingseedsorindependentconfirmation.',
            'Contrastsarepolicy-aggregationinterventionsonthesefrozenmodels,notoraclecritic-errorlabels.',
            'Allfourcellsreported;noglobalmeanfromper-environmentwinners.'],goal_complete=False)
    (out/'review_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(means=values,contrasts=effects,primary_unchanged='median_rank')),flush=True)


if __name__=='__main__':main()
