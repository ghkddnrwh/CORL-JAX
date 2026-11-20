"""Publication-exportable plot of the completed frozen Euler-grid diagnostic."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    directory=args.root/'analysis/Ant100K_student_decode_grid_v1'
    result=json.loads((directory/'summary.json').read_text());rows=result['rows']
    steps=[row['decode_steps']for row in rows]
    fig,axes=plt.subplots(2,2,figsize=(10.6,7.2))
    axes[0,0].plot(steps,[row['CPU50_success']for row in rows],'-o',color='#00798c',label='Student policy')
    axes[0,0].axhline(result['own_fixed_ten_step_BC_best9_CPU50'],linestyle='--',color='#d1495b',label='Fixed best9 BC reference')
    axes[0,0].set(title='Actual environment return',ylabel='CPU50 success fraction',ylim=(0,.65))
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].plot(steps,[row['summary']['policy_vs_best9_teacher_energy_mean']for row in rows],'-o',color='#00798c',label='Student vs reference')
    axes[0,1].axhline(rows[0]['summary']['teacher_vs_independent_teacher_energy_mean'],linestyle='--',color='#d1495b',label='Independent reference vs reference')
    axes[0,1].set(title='Heldout action distributions',ylabel='Energy distance',ylim=(0,.35))
    axes[0,1].legend(frameon=False,fontsize=7.5)
    axes[1,0].plot(steps,[100*row['summary']['chosen_action_saturation_fraction']for row in rows],'-o',color='#00798c')
    axes[1,0].axhline(100*rows[0]['summary']['teacher_action_saturation_fraction'],linestyle='--',color='#d1495b')
    axes[1,0].set(title='Heldout saturation: |action| >= 0.98',ylabel='Action saturation (%)',ylim=(0,9))
    axes[1,1].plot(steps,[row['summary']['chosen_vs_fifty_same_latent_endpoint_RMS']for row in rows],'-o',color='#00798c')
    axes[1,1].set(title='Same-latent numerical comparison',ylabel='Endpoint RMS from 50-step grid',ylim=(-.002,.042))
    for axis in axes.flat:
        axis.set_xlabel('Student Euler steps');axis.set_xticks(steps);axis.grid(alpha=.2)
    fig.suptitle('Frozen Antmaze-large policy at 100K: student integration-grid diagnostic',fontsize=13)
    fig.text(.5,.025,'Q, V, BC and policy parameters are fixed; BC decoding stays at 10 steps. All 50 original 10-step episode arrays reproduced.\nOne posthoc checkpoint and one training seed; these are not last-three or nine-task benchmark results. Distribution probes use 64 states and 16 samples.\nReference-reference energy estimates Monte Carlo resolution, not a confidence interval. The 50-step grid is not an oracle ODE solution.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.96))
    for extension in ('png','pdf'):
        fig.savefig(directory/f'grid_diagnostic_v1.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
