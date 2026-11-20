"""Show full last3 discovery; never present3members as3ensemble replications."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    auditpath=root/'analysis/independent_critic_fresh_confirmation_v1/discovery_audit.json'
    audit=json.loads(auditpath.read_text());assert audit['status']=='complete'and audit['valid_evaluations']==54 and not audit['errors']
    assert audit['unique_model_checkpoints_verified']==81
    reviewpath=root/'analysis/method_review_v16.json';review=json.loads(reviewpath.read_text())
    envs=[row['environment']for row in review['independent_confirmation']['seeds'][0]['rows']]
    labels=['AM-L','HM-M','HM-L','AS-A','Cube-S','Cube-D','Pz-3','Pz-4','Scene']
    rows=[]
    for label in('V2 IFQL, best9 (s0)','FQL, fresh reference (s0)'):
        method=next(row for row in review['methods']if row['label']==label)
        assert method['completed_tasks']==9
        rows.append(dict(label=label,values=method['values'],mean=method['nine_task_mean'],cost='1baselearner'))
    for method in audit['methods']:
        assert method['completed_tasks']==9
        rows.append(dict(label='Independent3 / '+method['rule'],
            values=[next(task['last3_mean']for task in method['tasks']if task['environment']==env)for env in envs],
            mean=method['nine_task_mean'],cost='3baselearners;ONEensemblegroup'))
    fig,(ax,macro)=plt.subplots(1,2,figsize=(13.5,5.2),gridspec_kw={'width_ratios':[4.2,1.5]})
    colors=['#729b9b','#888888','#2a6fbb','#b96925'];x=np.arange(9);width=.19
    for i,(row,color)in enumerate(zip(rows,colors)):
        ax.bar(x+(i-1.5)*width,row['values'],width,color=color,label=row['label'])
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.06),ylabel='CPU50 success: mean800K /900K /1M')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    y=np.arange(len(rows));macro.barh(y,[row['mean']for row in rows],color=colors)
    macro.set(yticks=y,yticklabels=['V2 best9','Fresh FQL','3critics mean','3critics ranks'],xlim=(0,.59))
    macro.invert_yaxis();macro.axvline(.47,color='#b82b32',ls='--',lw=1,label='Provided FQL.47')
    for i,row in enumerate(rows):macro.text(row['mean']+.006,i,f"{row['mean']:.4f}",va='center',fontsize=9)
    macro.set_xlabel('Equal nine-task mean');macro.grid(axis='x',alpha=.15);macro.set_axisbelow(True)
    fig.legend(loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Independent-critic discovery: one ensemble group, three base learners',fontsize=12)
    fig.text(.5,.018,'All IQL variants: tau=.9, gamma=.999, BC best9. Only current-step critic aggregation differs for the two ensemble rows.\nDiscovery reuses observed learner seeds0/1/2; fresh groups[3,4,5]and[6,7,8]are required. Ensemble training cost is approximately3x.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.19,1,.95))
    out=root/'analysis/independent_critic_discovery_v1'
    for suffix in('png','pdf'):fig.savefig(out/('comparison_v1.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    with(out/'comparison_v1.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['method',*envs,'nine_task_mean','training_cost'])
        for row in rows:writer.writerow([row['label'],*row['values'],row['mean'],row['cost']])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record=dict(time=time.time(),rows=rows,selected_rule=audit['selected_rule'],selected_nine_task_mean=audit['selected_nine_task_mean'],
        inputs_sha256={str(p):sha(p)for p in(auditpath,reviewpath)},source_sha256=sha(__file__),
        scope='Strictcomplete54CPU50/81actualmodels;onealready-observedensemblegroup;noindependentconfirmationclaim.',goal_complete=False)
    (out/'comparison_review_v1.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps(dict(methods=[dict(label=row['label'],mean=row['mean'])for row in rows],selected=audit['selected_rule'])))


if __name__=='__main__':main()
