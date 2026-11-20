"""Report auxiliary original learning curves at five retained checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,sha


def run(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=load_plan(path)
    root=path.parent
    steps=(100000,300000,800000,900000,1000000)
    rows,evidence=[],{}
    for case in plan['cases']:
        for mode in ('lean','actor'):
            row=dict(case=case['id'],environment=case['environment'],seed=case['seed'],deployment=mode)
            for step in steps:
                file=(root/'analysis/early_curve_evaluations'/f'{case["id"]}_{step}_{mode}.json' if step<800000
                    else Path(case['baseline_evaluation_directory'])/f'{step}_{mode}.json')
                value=json.loads(file.read_text())
                assert value['n_episodes']==50 and value['first_episode_seed']==42
                row[f'success_{step}']=100*value['success_rate']
                evidence[str(file)]=sha(file)
            rows.append(row)
    with (root/'analysis/early_curve_per_seed_v1.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    envs=sorted({r['environment'] for r in rows})
    for ei,env in enumerate(envs):
        for mi,mode in enumerate(('lean','actor')):
            ax=axes[ei,mi]
            selected=[r for r in rows if (r['environment'],r['deployment'])==(env,mode)]
            for row in sorted(selected,key=lambda r:r['seed']):
                ax.plot(np.asarray(steps)/1000,[row[f'success_{s}'] for s in steps],marker='o',label=f'Seed {row["seed"]}')
            ax.axvline(800,color='gray',linestyle='--',linewidth=.8)
            ax.set_title(('HM-M N=18' if env.startswith('humanoid') else 'Puzzle 4x4 N=27')+' / '+mode)
            ax.set_ylim(0,100);ax.set_xticks([100,300,800,900,1000]);ax.grid(alpha=.2);ax.legend()
            if mi==0:ax.set_ylabel('Original policy success (%)')
            if ei==1:ax.set_xlabel('Training updates (thousands)')
    fig.suptitle('Original runs: five retained checkpoints, CPU50; dashed line = intervention start',fontsize=11)
    fig.tight_layout()
    directory=root/'figures';directory.mkdir(exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(directory/f'baseline_full_learning_curves_v1.{extension}',dpi=200)
    plt.close(fig)
    atomic(root/'analysis/early_curve_summary_v1.json',dict(status='complete',rows=rows,
        checkpoints=list(steps),evaluation_evidence_sha256=evidence,source_sha256=sha(__file__),
        scope='Auxiliary diagnosis, not used to select a checkpoint or modify ongoing interventions; '
            'unobserved times between retained checkpoints are not measured.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();run(a.plan.resolve())
