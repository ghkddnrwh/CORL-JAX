"""Show every frozen-Q counterfit point, without treating it as a policy result."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);args=parser.parse_args()
    result=json.loads(args.result.read_text());assert result['status']=='complete'and result['unit_return_model']
    with np.load(result['raw_arrays'])as arrays:
        n=len(arrays['validation_uniform_indices']);assert n==8192
        goals=len(arrays['validation_goal_indices']);assert goals>0
        initial=arrays['initial_V_cost'];target=arrays['target_fit_Q_cost']
        score=lambda v:float(np.mean(np.where(target[:n]>=v[:n],.9,.1)*(target[:n]-v[:n])**2))
        boundary=lambda v:float(np.sqrt(np.mean(v[n:]**2)))
        points={}
        for geometry in ('mse','bernoulli_kl'):
            values=[initial]+[arrays[f'counterfit_{geometry}_{step}_V_cost']for step in (2,8,32)]
            points[geometry]=dict(fit=[score(v)for v in values],boundary=[boundary(v)for v in values])
        teacher=float(np.sqrt(np.mean(target[n:]**2)))
    fig,(left,right)=plt.subplots(1,2,figsize=(10,4.5))
    x=np.arange(4)
    for geometry,label,color in [('mse','Probability MSE','#247ba0'),('bernoulli_kl','Asymmetric Bernoulli KL','#df7831')]:
        left.plot(x,points[geometry]['fit'],'o-',color=color,label=label,lw=1.7)
        right.plot(x,points[geometry]['boundary'],'o-',color=color,label=label,lw=1.7)
    right.axhline(teacher,color='#666666',ls='--',lw=1.2,label='Frozen target-Q boundary RMSE')
    left.set_title('Held-out fit to the frozen teacher');left.set_ylabel('Expectile fitting loss\n(cost units squared)')
    right.set_title('Calibration at the known zero-value boundary');right.set_ylabel('V RMSE in original cost units')
    for axis in (left,right):
        axis.set_xticks(x,['0','2','8','32']);axis.set_xlabel('Private extra V Adam updates')
        axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.2);axis.set_ylim(bottom=0)
    left.legend(fontsize=8,loc='lower left');right.legend(fontsize=7.5,loc='lower right')
    fig.suptitle('Pz3 100K diagnostic: lower average fitting loss can coexist with worse boundary accuracy',fontsize=11)
    fig.text(.5,.015,f'One actual unit-MSE checkpoint; fixed 8192 uniform validation rows and all {goals} success-boundary rows, reported separately.\nFrozen Q, private Adam reset for each objective, identical fresh training batches; no model files, GPU training, or policy rollouts changed.\nAll 2/8/32 points are shown. Teacher calibration is a reference, not a V-error lower bound. This does not establish OGBench performance improvement.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.16,1,.93))
    for extension in ('png','pdf'):
        path=args.result.parent/f'unit_value_fitting_mechanism_Pz3_100K_v2.{extension}';assert not path.exists()
        fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
