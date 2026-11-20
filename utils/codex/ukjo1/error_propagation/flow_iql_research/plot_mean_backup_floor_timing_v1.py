"""Show contrasting temporal examples while preserving the primary late metric."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);args=parser.parse_args()
    result=json.loads(args.analysis.read_text());assert result['status']=='complete'
    chosen=[next(r for r in result['rows']if r['short']==short)for short in('hmmed','soccer','ant')]
    fig,axes=plt.subplots(3,3,figsize=(12.7,8.8),sharex=True)
    for i,row in enumerate(chosen):
        for mode,label,color in [('min','Target twin minimum','#cb8439'),('mean','Target twin mean','#247ba0')]:
            data=row['families'][mode];series=data['series'];x=np.asarray(series['step'])/1000000
            axes[i,0].plot(x,series['negative_mean_Q_over_abs_lower'],color=color,lw=1.4,label=label)
            axes[i,1].plot(x,100*np.asarray(series['lower_clip_fraction']),color=color,lw=1.4)
            points=data['policy_points'];xx=np.array([p['step']for p in points])/1000000;yy=np.array([p['success']for p in points])
            axes[i,2].plot(xx,yy,color=color,lw=1.2)
            for primary in(False,True):
                subset=[p for p in points if p['is_primary']==primary]
                axes[i,2].scatter([p['step']/1000000 for p in subset],[p['success']for p in subset],s=28,
                    edgecolors=color,facecolors=color if primary else'white',zorder=3)
        title={'hmmed':'Humanoid-medium','soccer':'Antsoccer','ant':'Ant-large'}[row['short']]
        axes[i,0].set_ylabel(f'{title}\nNegative mean Q / |L|');axes[i,0].set_ylim(0,1.08);axes[i,0].axhline(1,color='#888888',ls='--',lw=.8)
        axes[i,1].set_ylabel('Lower-clipped targets (%)');axes[i,1].set_ylim(-1,75)
        axes[i,2].set_ylabel('CPU50 success');axes[i,2].set_ylim(-.02,1.04)
        for axis in axes[i]:
            axis.axvspan(.8,1,color='#888888',alpha=.08);axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.18);axis.set_xlim(0,1.02)
    for axis in axes[-1]:axis.set_xlabel('Training updates (millions)')
    axes[0,0].legend(fontsize=8,loc='lower right');axes[0,0].set_title('Value trajectory');axes[0,1].set_title('Actual online target clipping');axes[0,2].set_title('Auxiliary and primary evaluations')
    fig.suptitle('Early failures can precede floor clipping; late saturation is not a sufficient causal explanation',fontsize=11)
    fig.text(.5,.015,'Posthoc contrasting examples from the allnine seed0 controlled comparison. tau=.9, gamma=.999,9flow candidates; actual BC/moments/replay/RNG match.\nHollow100K/300K points are auxiliary diagnostics; filled800K/900K/1M points alone define the primary score (gray late region).\nHMM fails before lower clipping; Soccer later deteriorates; Ant shows that lower magnitude/clipping need not increase return.\nThese traces do not isolate clipping as a causal mediator, measure true-Q bias/B_k, or replace independent seed9/10/11 confirmation.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95))
    for ext in('png','pdf'):
        path=args.analysis.parent/f'mean_backup_floor_timing_v1.{ext}';assert not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
