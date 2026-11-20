"""Matched original 1M models: min heads, temporal score mean, mean heads."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,sha


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(path):
    plan=load_plan(path)
    root=path.parent
    rows,evidence=[],{}
    methods=('baseline','temporal_Q_average','online_mean_selector')
    for case in plan['cases']:
        files={
            'baseline':Path(case['baseline_evaluation_directory'])/'1000000_lean.json',
            'temporal_Q_average':root/'temporal_average_evaluations'/case['id']/'1000000_lean.json',
            'online_mean_selector':root/'analysis/mean_selector_control_v1'/f'{case["id"]}.json'}
        values={m:json.loads(p.read_text()) for m,p in files.items()}
        assert sha(files['baseline'])==case['baseline_evaluation_sha256']['1000000_lean']
        for method in methods:
            value=values[method]
            assert value['n_episodes']==50 and value['first_episode_seed']==42
            delta=np.asarray(value['episode_successes'])-np.asarray(values['baseline']['episode_successes'])
            rows.append(dict(case=case['id'],environment=case['environment'],N=case['bootstrap_samples'],
                seed=case['seed'],method=method,success_1M=100*value['success_rate'],
                paired_change_from_original_pp=float(100*delta.mean()),
                paired_episode_wins=int(np.sum(delta>0)),paired_episode_losses=int(np.sum(delta<0))))
            evidence[str(files[method])]=sha(files[method])
    summary=[]
    for env in sorted({r['environment'] for r in rows}):
        for method in methods:
            group=[r for r in rows if (r['environment'],r['method'])==(env,method)]
            assert len(group)==3
            values=np.array([r['success_1M'] for r in group])
            summary.append(dict(environment=env,N=group[0]['N'],method=method,seeds=3,
                success_1M_mean=float(values.mean()),success_1M_sd=float(values.std(ddof=1)),
                worst_seed=float(values.min()),best_seed=float(values.max())))
    write_csv(root/'analysis/selector_comparison_per_seed_v1.csv',rows)
    write_csv(root/'analysis/selector_comparison_summary_v1.csv',summary)
    atomic(root/'analysis/selector_comparison_v1.json',dict(status='complete',rows=rows,summary=summary,
        evidence_sha256=evidence,source_sha256=sha(__file__),
        limits='Same original 1M models, no new training. Temporal-average and mean-head controls '
            'do not refit the actor. Mean-head is an exploratory follow-up, not an 800K/900K/1M training experiment.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),sharey=True)
    for ax,env in zip(axes,sorted({r['environment'] for r in rows})):
        for seed in range(3):
            points=[next(r['success_1M'] for r in rows if (r['environment'],r['seed'],r['method'])==(env,seed,m)) for m in methods]
            ax.plot(range(3),points,marker='o',label=f'Seed {seed}',alpha=.8)
        group=[next(r for r in summary if (r['environment'],r['method'])==(env,m)) for m in methods]
        ax.errorbar(range(3),[r['success_1M_mean'] for r in group],yerr=[r['success_1M_sd'] for r in group],
            fmt='ks',markersize=5,capsize=4,label='Mean +/- sample SD',zorder=4)
        ax.set_title('HM-M N=18' if env.startswith('humanoid') else 'Puzzle 4x4 N=27')
        ax.set_xticks(range(3));ax.set_xticklabels(['Min heads\n(original)','800K/current\nQ-score average','Mean heads\n(current)'])
        ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('1M Lean success (%)')
    fig.suptitle('Deployment-only controls on original models; same current Flow9 and CPU50',fontsize=10)
    fig.tight_layout()
    directory=root/'figures';directory.mkdir(exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(directory/f'original_1M_selector_controls_v1.{extension}',dpi=200)
    plt.close(fig)
    print(json.dumps(dict(status='complete',summary=summary)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();run(args.plan.resolve())
