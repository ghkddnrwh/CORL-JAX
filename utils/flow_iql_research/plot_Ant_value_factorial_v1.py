"""Export completed Ant value-package/projection comparisons from audited data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    parser=argparse.ArgumentParser();parser.add_argument('analysis',type=Path);args=parser.parse_args()
    data=json.loads((args.analysis/'Ant_value_component_full1M_v2.json').read_text())
    values={row['label']:row for row in data['rows']}
    layout=[['native','one_V_fit_old_bootstrap_projected'],['two_V_fits_fresh_unbounded','two_V_fits_fresh_projected']]
    performance=np.asarray([[values[key]['last3_mean']for key in row]for row in layout])
    error=np.asarray([[values[key]['known_return_Q_error_RMS_lower_bound']for key in row]for row in layout])
    fig,axes=plt.subplots(1,2,figsize=(10.6,5.6))
    images=[axes[0].imshow(performance,cmap='Blues',vmin=0,vmax=1),axes[1].imshow(error,cmap='Oranges',norm=LogNorm(vmin=.1,vmax=1e6))]
    axes[0].set_title('Success: mean of800K /900K /1M',fontsize=11)
    axes[1].set_title('Known-bound Q error lower bound at1M',fontsize=11)
    for index,(ax,array)in enumerate(zip(axes,[performance,error])):
        ax.set_xticks([0,1],['Unprojected target','Projected target'])
        ax.set_yticks([0,1],['1 V fit\nOld V bootstrap','2 V fits\nFresh V bootstrap'])
        for i in range(2):
            for j in range(2):
                text=f'{array[i,j]:.4f}'if index==0 else f'{array[i,j]:.3g}'
                ax.text(j,i,text,ha='center',va='center',fontsize=13,
                    color='white'if (performance[i,j]>.65 if index==0 else error[i,j]>1e4)else'#111111')
        fig.colorbar(images[index],ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Ant: target projection and the V-fitting / bootstrap package',fontsize=13)
    fig.text(.5,.025,'One task and training seed; common hyperparameters and replay RNG. Rows bundle fit count and freshness.\n'
        'Error certificate uses both Q heads on a fixed1024-state data probe; a small lower bound does not certify accurate values.',
        ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,.92])
    for extension in('png','pdf'):fig.savefig(args.analysis/('Ant_value_factorial_v1.'+extension),dpi=180)


if __name__=='__main__':main()
