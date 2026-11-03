"""Draw audited source crossovers and paired frozen-value-fit trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=root/'analysis'
    cross_path=out/'cohort_BC_critic_cross_v1/summary.json';fit_path=out/'frozen_value_sampling_v1/summary.json'
    cross=json.loads(cross_path.read_text());fit=json.loads(fit_path.read_text())
    assert cross['status']==fit['status']=='complete'
    assert cross['actual_unique_models_verified']==36 and fit['actual_unique_1M_models_verified']==18
    assert cross['source_sha256']==sha('utils/flow_iql_research/run_cohort_BC_critic_cross_v1.py')
    assert fit['source_sha256']==sha('utils/flow_iql_research/run_frozen_value_sampling_v1.py')
    assert cross['registration_sha256']==sha(root/'cohort_BC_critic_cross_registration_v1.json')
    assert fit['registration_sha256']==sha(root/'frozen_value_sampling_registration_v1.json')
    fig,axes=plt.subplots(1,2,figsize=(10.7,5.6))
    for axis,task,title in zip(axes,('pz3','ant'),('Puzzle3x3','Antmaze-large')):
        rows=[row for row in cross['cells']if row['task']==task]
        matrix=np.zeros((2,2))
        for i,ag in enumerate(('seen','fresh')):
            for j,cg in enumerate(('seen','fresh')):
                row=next(r for r in rows if r['actor_group']==ag and r['critic_group']==cg)
                matrix[i,j]=row['last3_mean']
        image=axis.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i,ag in enumerate(('seen','fresh')):
            for j,cg in enumerate(('seen','fresh')):
                row=next(r for r in rows if r['actor_group']==ag and r['critic_group']==cg)
                color='black'if matrix[i,j]>.55 else'white'
                axis.text(j,i-.08,f'{matrix[i,j]:.3f}',ha='center',va='center',color=color,fontsize=17)
                values='/'.join(f"{row['per_step'][str(s)]:.2f}"for s in (800000,900000,1000000))
                axis.text(j,i+.2,values,ha='center',va='center',color=color,fontsize=8)
        axis.set_xticks((0,1),('Q seeds0/1/2','Q seeds3/4/5'))
        axis.set_yticks((0,1),('BC seed0','BC seed3'))
        axis.set_title(title)
    fig.colorbar(image,ax=axes.tolist(),fraction=.025,pad=.03,label='Last3 CPU50 success')
    fig.suptitle('Frozen BC / critic source crossover: failure mechanisms differ by task',fontsize=12)
    fig.subplots_adjust(left=.09,right=.88,top=.84,bottom=.28,wspace=.37)
    fig.text(.5,.075,'Each cell averages800K /900K /1M; small text shows all three steps. Same N9, mean ranks, tau.9 and gamma.999.\nOff-diagonal inference uses3critic sources plus a separate BC source:4distinct trained learners, not a new3-learner candidate.\nThese paired two-task diagnostics cannot supply a nine-task benchmark or oracle action-value errors.',ha='center',fontsize=8)
    for extension in ('png','pdf'):
        path=out/f'cohort_BC_critic_cross_mechanism_v1.{extension}';assert not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.8))
    trajectories=[]
    for axis,task,title in zip(axes,('ant','hmmed','pz3'),('Antmaze-large','Humanoidmaze-medium','Puzzle3x3')):
        rows=sorted([row for row in fit['rows'].values()if row['task']==task],key=lambda r:r['seed'])
        assert [r['seed']for r in rows]==list(range(6))
        for mode,color,label in (('repeated_minibatch','#c34a45','Repeat one minibatch'),('fresh_minibatch','#147c9a','Fresh minibatches')):
            lines=[]
            for row in rows:
                trace=row['modes'][mode];steps=[t['extra_V_Adam_steps']for t in trace]
                values=np.asarray([t['sets']['validation']['expectile_objective']for t in trace]);assert values[0]>0
                change=100*(values/values[0]-1);lines.append(change)
                axis.plot(steps,change,color=color,alpha=.17,lw=.8)
                trajectories.append(dict(task=task,seed=row['seed'],mode=mode,steps=steps,relative_validation_change_percent=change.tolist()))
            axis.plot(steps,np.mean(lines,axis=0),'-o',color=color,lw=2,ms=3,label=label)
        axis.axhline(0,color='#888888',lw=.8);axis.set(title=title,xlabel='Extra V Adam steps')
        axis.set_xticks((0,1,2,4,8));axis.grid(alpha=.2)
    axes[0].set_ylabel('Validation objective change (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.115),frameon=False)
    fig.suptitle('Frozen target-Q: less minibatch reuse helps some heldout fits, with mixed Ant effects',fontsize=11)
    fig.text(.5,.025,'Thin lines:6learner seeds per task. Thick lines: mean of within-model relative changes; axes have separate scales.\nIdentical initial V/Adam and query sets within each pair; first fitting batch shared; CPU auxiliary fits only.\nObserved-action expectile objectives are not conditional-residual or B_k labels, and these curves are not policy returns.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.22,1,.93))
    for extension in ('png','pdf'):
        path=out/f'frozen_value_sampling_mechanism_v1.{extension}';assert not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    record=dict(time=time.time(),cross_summary_sha256=sha(cross_path),fit_summary_sha256=sha(fit_path),
        source_sha256=sha(__file__),cross_cells=cross['cells'],fit_trajectories=trajectories,
        off_diagonal_distinct_training_sources=4,primary_candidate_unchanged=True,goal_complete=False)
    (out/'value_sampling_diagnostic_figures_v1.json').write_text(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':main()
