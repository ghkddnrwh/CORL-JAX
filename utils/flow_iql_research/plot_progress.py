"""Standalone early learning curves; never uses a best-step score."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils.flow_iql_research.audit import records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex='col')
    colors={'official':'#66737c','bound':'#277da1','bounded_v2':'#e1812c'}
    for run in sorted((args.root/'runs').glob('*')):
        if not (run/'config.json').exists():continue
        c=json.loads((run/'config.json').read_text());i=c['identity'];a=c['arguments']
        if i['kind']!='iql' or i['profile'] not in colors:continue
        col=0 if 'cube-single' in i['environment'] else 1 if 'scene-play' in i['environment'] else None
        if col is None:continue
        high=i['config']['expectile']==.99
        color='#38965b' if high else colors[i['profile']]
        style=':' if high else '-' if a.get('discount')==.999 else '--'
        es=records(run/'evaluations.jsonl');ds=records(run/'diagnostics.jsonl')
        axes[0,col].plot([e['step']/1000 for e in es],[e['success_rate'] for e in es],color=color,ls=style,marker='o',ms=3,lw=1.5)
        axes[1,col].plot([e['step']/1000 for e in ds],[e['q_rms_over_return_bound'] for e in ds],color=color,ls=style,lw=1.3)
    for col,title in enumerate(('Cube-single','Scene')):
        axes[0,col].set_title(title);axes[0,col].set_ylim(-.02,1.02)
        axes[0,col].set_ylabel('CPU50 success (seed0)')
        axes[1,col].set_ylabel('Fixed-probe Q RMS / return scale');axes[1,col].set_xlabel('Q/actor updates (K)')
        for row in range(2):axes[row,col].grid(alpha=.2)
    handles=[Line2D([0],[0],color=c,label=p) for p,c in colors.items()]
    handles.extend([Line2D([0],[0],color='#38965b',ls=':',label='bounded_v2, expectile0.99'),
                    Line2D([0],[0],color='black',ls='--',label='original discount'),
                    Line2D([0],[0],color='black',ls='-',label='discount0.999')])
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.04))
    fig.suptitle('Official-style IQL: interim curves; primary score requires 800K, 900K and 1M',fontsize=12)
    fig.text(.5,.015,'One exploratory training seed. Probe scale is diagnostic, not a proof of bounded outputs or stable learning.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.95));out=args.root/'analysis'
    for ext in ('png','pdf'):fig.savefig(out/f'interim_value_controls.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
