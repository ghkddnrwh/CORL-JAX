"""Plot signed value-control effects; incomplete original-discount cells stay gray."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    directory=args.root/'analysis';audit=json.loads((directory/'Gaussian_discount_value_factorial_v1.json').read_text())
    lookup={(group['discount'],group['profile']):group for group in audit['groups']}
    matrix=np.full((10,2),np.nan)
    for column,discount in enumerate(('original',.999)):
        native=lookup[(discount,'official')];corrected=lookup[(discount,'bounded_v2')]
        for index,(old,new)in enumerate(zip(native['rows'],corrected['rows'])):
            assert old['task']==new['task']
            if old['last3_mean']is not None and new['last3_mean']is not None:
                matrix[index,column]=new['last3_mean']-old['last3_mean']
        if native['nine_task_mean']is not None and corrected['nine_task_mean']is not None:
            matrix[-1,column]=corrected['nine_task_mean']-native['nine_task_mean']
    figure,axis=plt.subplots(figsize=(6.4,6.6))
    cmap=plt.cm.RdBu.copy();cmap.set_bad('#e5e7eb')
    im=axis.imshow(np.ma.masked_invalid(matrix),vmin=-.8,vmax=.8,cmap=cmap,aspect='auto')
    axis.set_xticks([0,1]);axis.set_xticklabels(['Original discount\n0.99 / 0.995','Stress discount\n0.999'])
    axis.xaxis.tick_top();axis.tick_params(length=0,pad=8)
    axis.set_yticks(range(10));axis.set_yticklabels(['AM-L','HM-M','HM-L','AS-A','Cube-S','Cube-D','Pz-3','Pz-4','Scene','Nine-task avg'])
    for index in range(10):
        for column in range(2):
            value=matrix[index,column]
            text='pending'if np.isnan(value)else'0.000'if abs(value)<1e-12 else f'{value:+.3f}'
            axis.text(column,index,text,ha='center',va='center',fontsize=10,color='#555555'if np.isnan(value)else'white'if abs(value)>.5 else'black')
    axis.axhline(8.5,color='white',linewidth=2)
    figure.colorbar(im,ax=axis,fraction=.05,pad=.05,label='V2 minus native success')
    figure.suptitle('Gaussian IQL value intervention: discount interactions',fontsize=12,y=.99)
    figure.text(.5,.025,'Exact 800K / 900K / 1M CPU50 means, training seed0, expectile0.9.\nV2: two V fits, fresh bootstrap, known-return target projection.\nOriginal-discount V2: only3/9 complete. Gray cells are missing, never zero-filled.\nThese are whole-intervention effects, not individual-component or independent-seed claims.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.15,.99,.95))
    for extension in ('png','pdf'):
        figure.savefig(directory/f'Gaussian_discount_value_factorial_v1.{extension}',dpi=180,bbox_inches='tight')
    plt.close(figure)


if __name__=='__main__':main()
