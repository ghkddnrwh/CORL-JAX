"""Show qualified discovery and still-incomplete independent confirmation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('review',type=Path);args=parser.parse_args()
    review=json.loads(args.review.read_text());assert not review['goal_complete']
    choices=[
        ('Gaussian IQL, original gamma (s0)','Gaussian IQL, original gamma (seed0)'),
        ('Native IFQL, best9 (s0)','Native IFQL, N9 (seed0)'),
        ('Native IFQL Q with common V2 BC, best9 (s0)','Native Q + common V2 BC, N9 (seed0)'),
        ('V2 IFQL, best9 (s0)','V2 Q + same V2 BC, N9 (seed0)'),
        ('FQL, fresh reference (s0)','FQL reference (one training seed)'),
        ('ReBRAC, fresh reference (s0)','ReBRAC reference (one training seed)'),
        ('Finite expectile backup, best9 (s0)','Finite backup: twin minimum, N9 (seed0)'),
        ('Mean-head finite expectile backup, best9 (s0)','Finite backup: twin mean, N9 (seed0 discovery)'),
        ('Mean-head finite expectile backup fresh seed9','Same mean backup, fresh seed9'),
        ('Mean-head finite expectile backup fresh seed10','Same mean backup, fresh seed10'),
        ('Mean-head finite expectile backup fresh seed11','Same mean backup, fresh seed11'),
        ('V2 IFQL independent3 mean_rank (discovery group0)','Earlier mean ranks: discovery group0/1/2'),
        ('Independent3 mean_rank fresh two-group average (failed confirmation)','Earlier mean ranks: TWO fresh groups (failed)'),
    ]
    selected=[(next(m for m in review['methods']if m['label']==key),label)for key,label in choices]
    values=np.array([[np.nan if v is None else v for v in row['values']+[row['nine_task_mean']]]for row,_ in selected])
    fig,axis=plt.subplots(figsize=(14.5,8));cmap=plt.cm.viridis.copy();cmap.set_bad('#e7e9eb')
    shown=axis.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=1,aspect='auto')
    axis.set_xticks(range(10),['AM-L','HM-M','HM-L','AS-A','Cube-S','Cube-D','Pz-3','Pz-4','Scene','Avg'])
    axis.set_yticks(range(len(selected)),[label for _,label in selected]);axis.xaxis.tick_top();axis.tick_params(length=0,pad=8,labelsize=9)
    for i in range(len(selected)):
        for j,value in enumerate(values[i]):
            label='pending'if np.isnan(value)else f'{value:.4f}'if j==9 else f'{value:.2f}'
            axis.text(j,i,label,ha='center',va='center',fontsize=8.5,fontweight='bold'if j==9 else'normal',
                color='#666666'if np.isnan(value)else'black'if value>.55 else'white')
    axis.axvline(8.5,color='white',lw=2)
    axis.add_patch(plt.Rectangle((-.5,6.5),10,1,fill=False,edgecolor='#d17e25',lw=2.4))
    axis.add_patch(plt.Rectangle((-.5,11.5),10,1,fill=False,edgecolor='#cb3e43',lw=2.4))
    axis.axhline(10.5,color='white',lw=2)
    fig.colorbar(shown,ax=axis,fraction=.024,pad=.02,label='Success rate')
    fig.suptitle('Uniform mean-backup discovery=.6170; independent seed9/10/11 confirmation is running',fontsize=12)
    fig.text(.5,.02,'Equal means of800K /900K /1M CPU50 and allnine tasks. Except row1, gamma=.999; IQL expectile=.9. Pending is not zero.\nFinite min/mean control:54actual late models; BC parameters, moments, replay and agent RNG all EXACT. The two common-BC rows also fix actual BC.\nFresh confirmation uses one learner per seed. Earlier mean ranks use THREE learners per ensemble group; discovery was not independently confirmed.\nProvided plain FQL criterion=.47, ReBRAC=.44; DD-ReBRAC=.63. No final independent win, equal-compute superiority, or SOTA claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.16,.99,.94))
    for ext in('png','pdf'):
        p=args.review.parent/f'common9_overview_v26.{ext}';assert not p.exists();fig.savefig(p,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
