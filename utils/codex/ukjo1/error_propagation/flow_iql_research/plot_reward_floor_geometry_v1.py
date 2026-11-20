"""Export the reward-floor and boundary-frequency diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    data=json.loads((args.directory/'summary.json').read_text())
    rows=[row for row in data['rows']if row['split']=='training']
    labels=['AM-L','HM-M','HM-L','Soccer','Cube-S','Cube-D','Pz-3','Pz-4','Scene'];x=np.arange(9)
    progress=[100*row['conditional_nonterminal_progress_fraction']for row in rows]
    goals=[256*row['full_success_fraction']for row in rows]
    fig,axes=plt.subplots(2,1,figsize=(10.6,6.8),sharex=True)
    for ax,values,color,ylabel in zip(axes,[progress,goals],['#356f9a','#d28a35'],
        ['Nonterminal progress signal\n(% of nonterminal transitions)','Success transitions\nper uniform batch of256']):
        ax.bar(x,values,color=color,width=.65);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.text(i,value+max(values)*.025,f'{value:.2f}',ha='center',fontsize=8)
        ax.set_ylim(0,max(values)*1.18)
    axes[1].set_xticks(x,labels);axes[1].set_xlabel('Training dataset')
    fig.suptitle('Signals that break a constant return-floor critic',fontsize=13)
    fig.text(.5,.018,'Constructed floor L = -C/(1-gamma): nonterminal target gap is r+C; success-boundary target is0.\n'
        'Dataset geometry only; the constant floor is not a full Bellman fixed point. No algorithm performance is plotted.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.075,1,.95])
    for extension in('png','pdf'):fig.savefig(args.directory/('geometry_v1.'+extension),dpi=180)


if __name__=='__main__':main()
