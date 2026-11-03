"""Pair the exact critic/BC swap with corrected TD and known-goal error."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    auditpath=root/'analysis/CubeD_backup_controller_swap_v1/audit_complete_v1.json';audit=json.loads(auditpath.read_text())
    assert audit['status']=='complete'and audit['crossed_evaluations_audited']==6
    assert all(row['actual_BC_parameters_max_abs_difference']==0 for row in audit['replay_and_parameter_checks'])
    planpath=root/'CubeD_backup_controller_swap_registration_v1.json';plan=json.loads(planpath.read_text())
    exact={}
    for job in plan['jobs']:
        result=json.loads(Path(job['output']).read_text());label='mean'if job['critic_run']==plan['pins']['mean']['run']else'min'
        reference=next(row for row in plan['diagonal_primary_records'][label]if row['step']==job['step'])
        exact[job['id']]={field:result[field]==reference[field]for field in('episode_successes','episode_rewards','episode_lengths')}
    assert all(all(row.values())for row in exact.values())
    snapshots={};goal={};inputs=[auditpath,planpath]
    for method,label in(('min','finite'),('mean','mean')):
        path=root/f'analysis/CubeD_finite_{method}_actual_target_snapshot_v2.json';d=json.loads(path.read_text())
        snapshots[method]=next(row for row in d['rows']if row['step']==1000000);inputs.append(path)
        path=root/f'analysis/terminal_boundary_snapshots_v1/cubed_{label}_1000000.json';g=json.loads(path.read_text())
        assert g['checkpoint_sha256']==snapshots[method]['checkpoint_sha256']
        goal[method]=g['sets']['validation']['observed_Q_RMS_to_known_zero'];inputs.append(path)
    array=np.asarray([[next(row['last3_mean']for row in audit['table']if row['critic']==critic and row['BC']==actor)
        for actor in('min','mean')]for critic in('min','mean')])
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),gridspec_kw={'width_ratios':[1.3,1,1]})
    image=axes[0].imshow(array,vmin=0,vmax=1,cmap='Blues')
    axes[0].set(xticks=[0,1],xticklabels=['BC from\nmin run','BC from\nmean run'],yticks=[0,1],
        yticklabels=['Min-backup critic','Mean-backup critic'],title='Last3 CPU50 success: actual parameter swap')
    for i in range(2):
        for j in range(2):axes[0].text(j,i,f'{array[i,j]:.2f}',ha='center',va='center',fontsize=15)
    axes[0].tick_params(length=0);axes[0].set_title(axes[0].get_title(),fontsize=10)
    labels=['Min backup','Mean backup'];colors=['#b96925','#2a6fbb']
    td=[snapshots[m]['actual_projected_finite_backup_snapshot_TD_RMS']for m in('min','mean')]
    axes[1].bar(labels,td,color=colors);axes[1].set(ylabel='Projected finite-target TD RMS',ylim=(0,5))
    axes[1].set_title('1M: fixed-train1024 snapshot',fontsize=10)
    for i,v in enumerate(td):axes[1].text(i,v+.08,f'{v:.3f}',ha='center')
    boundary=[goal[m]for m in('min','mean')]
    axes[2].bar(labels,boundary,color=colors);axes[2].set(yscale='log',ylim=(1,10000),ylabel='Both-head Q RMS to known zero')
    axes[2].set_title('1M: all heldout success-state transitions',fontsize=10)
    for i,v in enumerate(boundary):axes[2].text(i,v*1.2,f'{v:.1f}',ha='center')
    for axis in axes[1:]:axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    fig.suptitle('Cube-double seed0: critic-source gain, despite a larger fitted-data TD residual',fontsize=12)
    fig.text(.5,.025,'BC parameters and replay/agent RNGs are exactly equal at all 3 late checkpoints; all 50 crossed episode arrays match the same-critic diagonals.\nReturns average 800K/900K/1M. Other panels use 1M only. The TD panel is a frozen CPU snapshot, not an online batch loss.\nFinite proposal-root score is near zero in both methods. This one case does not establish oracle interior Q accuracy or a nine-task gain.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.16,1,.95));out=root/'analysis/CubeD_backup_controller_swap_v1'
    for suffix in('png','pdf'):fig.savefig(out/('mechanism_v1.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result=dict(time=time.time(),actual_BC_parameters_exact=True,crossed_vs_same_critic_diagonal_episode_arrays_exact=exact,
        critic_source_effect_at_fixed_BC=.32,actual_1M_projected_TD_RMS=dict(zip(('min','mean'),td)),
        heldout_1M_goal_Q_RMS=goal,inputs_sha256={str(p):sha(p)for p in inputs},source_sha256=sha(__file__),
        interpretation='Actualcritic-sourcechangeexplainsCubeDreturngainwithBCfixed;smallfitted-dataTDandfinite-rootresidualdonotcertifycorrectvalues.',
        limits='Oneenvironment/seed;knownboundaryerroronly;notafull9taskmeanorapopulationresidualmeasurement.',goal_complete=False)
    (out/'mechanism_review_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items()if k not in('inputs_sha256','crossed_vs_same_critic_diagonal_episode_arrays_exact')}))


if __name__=='__main__':main()
