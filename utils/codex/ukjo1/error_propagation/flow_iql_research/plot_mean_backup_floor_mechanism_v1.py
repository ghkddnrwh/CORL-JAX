"""Scientific figure of allnine seed0 aggregation controls, no policy selection."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);args=parser.parse_args()
    result=json.loads(args.analysis.read_text());assert result['status']=='complete'and len(result['rows'])==9
    rows=result['rows'];y=np.arange(9);labels={'ant':'AM-L','hmmed':'HM-M','hmlarge':'HM-L','soccer':'AS-A','cube':'Cube-S','cubed':'Cube-D','pz3':'Pz-3','pz4':'Pz-4','scene':'Scene'}
    fig,axes=plt.subplots(1,3,figsize=(13.5,6.6),sharey=True)
    for mode,label,color,offset in [('min','Target twin minimum','#cb8439',-.18),('mean','Target twin mean','#247ba0',.18)]:
        probes=[r['families'][mode]['phases']['late']['fixed_query_probe_time_means']for r in rows]
        updates=[r['families'][mode]['phases']['late']['online_update_weighted_metrics']for r in rows]
        q=np.array([p['negative_mean_Q_over_abs_return_lower']for p in probes]);clip=np.array([100*p['backup/lower_clip_fraction']for p in updates])
        success=np.array([r[f'{mode}_last3']for r in rows]);assert np.all((clip>=0)&(clip<=100))
        axes[0].barh(y+offset,q,height=.33,color=color,label=label)
        axes[1].barh(y+offset,clip,height=.33,color=color)
        axes[2].barh(y+offset,success,height=.33,color=color)
    axes[0].set_yticks(y,[labels[r['short']]for r in rows]);axes[0].invert_yaxis()
    axes[0].axvline(1,color='#777777',ls='--',lw=1);axes[0].set_xlim(0,1.08)
    axes[0].set_title('Fixed-query Q relative to return floor');axes[0].set_xlabel('Negative mean Q / absolute lower bound')
    axes[1].set_title('Actual target saturation');axes[1].set_xlabel('Targets clipped to lower bound (%)')
    axes[2].set_title('Policy return');axes[2].set_xlabel('Last3 CPU50 success rate');axes[2].set_xlim(0,1.05)
    for axis in axes:
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
    axes[0].legend(fontsize=8,loc='lower right')
    fig.suptitle(f"Finite expectile residuals are small in both methods; nine-task success {result['nine_task_min_mean']:.3f} to {result['nine_task_mean_mean']:.3f}",fontsize=12)
    fig.text(.5,.02,'Allnine environments, one discovery seed0, tau=.9, gamma=.999,9behavior-flow candidates. Left/middle use800K<step<=1M logs.\nLeft: time means of fixed1024-state probes. Middle: update-weighted actual Bellman-target clipping. Right: equal800K/900K/1M CPU50 means.\n54actual late models have EXACT BC parameters/moments/replay/agent RNG; all1000actor-log windows also match per environment.\nFloor saturation is a diagnostic, not an oracle true-Q error or B_k measurement. Ant illustrates that changing aggregation need not improve every task.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.17,1,.94))
    for extension in('png','pdf'):
        path=args.analysis.parent/f'mean_backup_floor_mechanism_v1.{extension}';assert not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
