"""Plot all four diagnostic routing/generation factorial tables."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    audit=json.loads(args.audit.read_text())
    assert audit['status']=='pass' and audit['total_diagnostic_rollout_episodes']==400
    args.output.mkdir(parents=True)
    fig,axes=plt.subplots(2,2,figsize=(9.3,7.6))
    fig.subplots_adjust(left=.13,right=.86,bottom=.20,top=.82,hspace=.90,wspace=.65)
    matrices={}
    for row,case in enumerate(('puzzle3_v1','cube_single_v1')):
        for col,method in enumerate(('CE_a01','rankMSE_a01')):
            item=next(x for x in audit['rows']if x['case']==case and x['method']==method)
            values=np.asarray([[item['primary'],item['hybrids']['learned_route_exact_flow']['success']],
                               [item['hybrids']['teacher_route_learned_head']['success'],item['teacher']]])
            assert np.all((values>=0)&(values<=1))
            ax=axes[row,col]
            image=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
            ax.set_xticks([0,1],['Learned head','Original Flow'])
            ax.set_yticks([0,1],['Learned selector','Teacher selector'])
            ax.tick_params(axis='both',labelsize=9,length=0,pad=7)
            ax.set_xlabel('Action generator',fontsize=10)
            label='Puzzle 3x3' if row==0 else 'Cube single'
            objective='index CE' if col==0 else 'relative-Q MSE'
            ax.set_title(label+' | '+objective+'\ncoefficient 0.1',fontsize=11,pad=10)
            for i in range(2):
                for j in range(2):
                    v=values[i,j]
                    n=int(round(50*v))
                    assert abs(n/50-v)<1e-9
                    ax.text(j,i,f'{n}/50\n{100*v:.0f}%',ha='center',va='center',fontsize=13,
                            color='white'if v>.55 else 'black')
            ax.add_patch(Rectangle((-.49,-.49),.98,.98,fill=False,edgecolor='#111111',linewidth=2.2))
            matrices[case+'/'+method]=values.tolist()
    cax=fig.add_axes([.90,.28,.022,.48])
    fig.colorbar(image,cax=cax,label='Success rate (50 episodes)')
    fig.suptitle('Separating selector and action-generator approximation',fontsize=15,y=.97)
    fig.text(.5,.91,'Frozen 1M teacher; 100K CPU students; 524K offline states; evaluation seeds 42-91',ha='center',fontsize=10)
    fig.text(.07,.09,'Black border: one-step actor only. Other cells are diagnostic hybrids using Flow and/or Q.\n'
             'Teacher baseline (lower right) is reused. Results are descriptive and do not enter the joint nine-task goal.',
             fontsize=9,linespacing=1.6)
    png=args.output/'routing_generation_factorial_v2.png'
    pdf=args.output/'routing_generation_factorial_v2.pdf'
    fig.savefig(png,dpi=180,bbox_inches='tight')
    fig.savefig(pdf,bbox_inches='tight')
    plt.close(fig)
    receipt=dict(time=time.time(),audit_sha256=sha(args.audit),source_sha256=sha(__file__),
                 files={str(p):sha(p)for p in [png,pdf]},matrices=matrices,
                 status='complete',not_joint_goal_performance=True)
    (args.output/'provenance_v2.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
