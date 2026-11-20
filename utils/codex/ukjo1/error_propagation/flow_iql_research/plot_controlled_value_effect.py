"""Last3 value-controller comparisons, keeping policy extraction fixed."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('logs/diagnosis/flow_iql_research_20261026');steps=(800000,900000,1000000)
    def scores(name):
        es=[json.loads(s) for s in (root/'runs'/name/'evaluations.jsonl').read_text().splitlines()]
        found=[next(e for e in es if e['step']==step) for step in steps]
        assert all(e['n_episodes']==50 for e in found)
        return [e['success_rate'] for e in found]
    control=json.loads((root/'analysis/factorized_IFQL_controls/scene_last3_control_summary.json').read_text())
    groups=[dict(label='Cube-single\nGaussian AWR',native=scores('w1_cube_official_g999_t90_s0'),corrected=scores('w1_cube_bounded_v2_g999_t90_s0')),
            dict(label='Scene\nGaussian AWR',native=scores('w1_scene_official_g999_t90_s0'),corrected=scores('w1_scene_bounded_v2_g999_t90_s0')),
            dict(label='Scene\nBC flow + best32',native=control['native_per_step'],corrected=control['corrected_per_step'])]
    x=np.arange(len(groups));fig,ax=plt.subplots(figsize=(9,5.5));width=.34
    for offset,key,color,label in [(-width/2,'native','#7c8993','Original value update'),(width/2,'corrected','#277e8e','Two V fits + fresh projected backup')]:
        means=[statistics.mean(g[key]) for g in groups]
        ax.bar(x+offset,means,width,color=color,label=label,alpha=.85)
        for j,g in enumerate(groups):
            ax.scatter(x[j]+offset+np.array([-.055,0,.055]),g[key],c='black',s=20,zorder=4)
            ax.text(x[j]+offset,max(g[key])+.03,f'{means[j]:.3f}',ha='center',fontsize=10)
    ax.set_xticks(x,[g['label'] for g in groups]);ax.set_ylim(0,1.07)
    ax.set_ylabel('Mean success at 800K, 900K and 1M')
    ax.set_title('Last-three evaluations: value control and policy extraction')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=9)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.025,'One training seed; each point is a CPU50 evaluation, not an independent training seed. All use discount0.999.\n'
             'Flow comparison holds the BC flow exactly fixed using equal-step checkpoints; corrected-state and episode sanity check passed.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ('png','pdf'):fig.savefig(root/f'analysis/controlled_value_effect_last3.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (root/'analysis/controlled_value_effect_last3.json').write_text(json.dumps(dict(steps=steps,groups=groups,
        caveat='Paired extraction/control examples, not a nine-task mean or a statistical-significance result. Three timepoints of one training seed are not three independent training seeds.'),indent=2))


if __name__=='__main__':main()
