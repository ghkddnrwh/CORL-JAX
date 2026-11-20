"""Audit and summarize the registered paired late-Q stabilization study."""
import argparse
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np

from utils.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, sha

LABELS={'baseline':'Original', 'q_lr_0p1':'Q LR x0.1', 'q_cosine':'Q cosine decay',
        'temporal_Q_average':'800K/current Q average'}


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def rank_stats(a,b):
    qa,qb=a.min(0),b.min(0)
    delta=qb-qa
    centered=delta-delta.mean(0,keepdims=True)
    return dict(argmax_flip_fraction=float(np.mean(qa.argmax(0)!=qb.argmax(0))),
        action_centered_Q_drift_rms=float(np.sqrt(np.mean(centered.astype(np.float64)**2))),
        initial_action_gap_median=float(np.median(np.sort(qa,axis=0)[-1]-np.sort(qa,axis=0)[-2])),
        final_action_gap_median=float(np.median(np.sort(qb,axis=0)[-1]-np.sort(qb,axis=0)[-2])))


def baseline_rankings(plan_path):
    from utils.flow_iql_research import sample_budget_core_v1 as core
    from utils.flow_iql_research.diagnose_sample_budget_components_v1 import read_source
    import jax
    import jax.numpy as jnp
    plan=load_plan(plan_path)
    out=plan_path.parent/'analysis/baseline_rankings'
    out.mkdir(exist_ok=True)
    for case in plan['cases']:
        job=next(j for j in plan['jobs'] if j['case']==case['id'])
        bankpath=Path(job['output'])/'ranking_bank.npz'
        assert bankpath.exists(),bankpath
        with np.load(bankpath) as bank:
            obs,actions=jnp.asarray(bank['observations']),jnp.asarray(bank['actions'])
        for step in (800000,900000,1000000):
            path=out/f'{case["id"]}_{step}.npz'
            if path.exists():continue
            parent=Path(case['parent_run'])
            source=dict(run=str(parent),checkpoint=str(parent/f'checkpoint_{step}.pkl'),step=step,
                checkpoint_sha256=case['parent_checkpoints'][str(step)],config_sha256=case['config_sha256'])
            _,state=read_source(source)
            net=state.base.agent.network
            heads=jax.jit(lambda p:net.replace(params=p).select('critic')(obs,actions=actions))(net.params)
            np.savez_compressed(path,heads=np.asarray(heads),step=step,bank_sha256=sha(bankpath))
            print(case['id'],step,flush=True)


def tree_delta(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return [v for k in a for v in tree_delta(a[k],b[k])]
    if isinstance(a,(list,tuple)):
        return [v for x,y in zip(a,b) for v in tree_delta(x,y)]
    return [np.asarray(a,np.float64)-np.asarray(b,np.float64)]


def analyze(plan_path,require_complete):
    plan=load_plan(plan_path)
    root=plan_path.parent
    rows,rankings,audits=[],[],[]
    paths={str(plan_path):sha(plan_path)}
    methods=['baseline','q_lr_0p1','q_cosine','temporal_Q_average']
    pending=[]
    for case in plan['cases']:
        baseline=Path(case['baseline_evaluation_directory'])
        for method in methods:
            job=next((j for j in plan['jobs'] if j['case']==case['id'] and j['method']==method),None)
            for mode in ('lean','actor'):
                if method=='temporal_Q_average' and mode=='actor':continue
                values=[]
                for step in plan['evaluation_steps']:
                    path=(baseline/f'{step}_{mode}.json' if method=='baseline' or step==800000
                        else root/'temporal_average_evaluations'/case['id']/f'{step}_{mode}.json' if method=='temporal_Q_average'
                        else root/'evaluations'/job['id']/f'{step}_{mode}.json')
                    if not path.exists():
                        pending.append(str(path));break
                    value=json.loads(path.read_text())
                    assert value['n_episodes']==50 and value['first_episode_seed']==42
                    assert value['success_rate']==np.mean(value['episode_successes'])
                    if method=='baseline' or step==800000:
                        assert sha(path)==case['baseline_evaluation_sha256'][f'{step}_{mode}']
                    values.append(value['success_rate']*100)
                    paths[str(path)]=sha(path)
                if len(values)!=3:continue
                rows.append(dict(case=case['id'],environment=case['environment'],N=case['bootstrap_samples'],
                    seed=case['seed'],method=method,deployment=mode,
                    success_800K=values[0],success_900K=values[1],success_1M=values[2],
                    late3_mean=float(np.mean(values)),change_800K_to_1M=values[2]-values[0],
                    change_900K_to_1M=values[2]-values[1],
                    collapse_20pp=int(values[2]-values[0]<=-20.)))
            if method=='temporal_Q_average':continue
            folder=root/'analysis/baseline_rankings' if method=='baseline' else Path(job['output'])
            name=lambda s:folder/(f'{case["id"]}_{s}.npz' if method=='baseline' else f'ranking_{s}.npz')
            if all(name(s).exists() for s in (800000,900000,1000000)):
                arrays={s:np.load(name(s)) for s in (800000,900000,1000000)}
                assert len({str(a['bank_sha256']) for a in arrays.values()})==1
                for start in (800000,900000):
                    rankings.append(dict(case=case['id'],environment=case['environment'],seed=case['seed'],
                        method=method,start=start,end=1000000,
                        **rank_stats(arrays[start]['heads'],arrays[1000000]['heads'])))
                for a in arrays.values():a.close()
            elif require_complete:raise ValueError('Missing rankings: '+case['id']+' '+method)
        for job in [j for j in plan['jobs'] if j['case']==case['id']]:
            run=Path(job['output'])
            cp=run/'checkpoint_1000000.pkl'
            if not cp.exists():continue
            source=Path(case['parent_run'])/'checkpoint_1000000.pkl'
            assert sha(source)==case['parent_checkpoints']['1000000']
            with cp.open('rb') as stream:new=pickle.load(stream)
            with source.open('rb') as stream:old=pickle.load(stream)
            aa=new['state']['base']['agent'];bb=old['state']['base']['agent']
            delta=tree_delta(aa['network']['params']['modules_actor_flow'],bb['network']['params']['modules_actor_flow'])
            replay_equal=all(np.array_equal(a,b) for a,b in zip(new['replay_rng'],old['replay_rng']))
            agent_equal=np.array_equal(aa['rng'],bb['rng'])
            assert replay_equal and agent_equal
            records=[json.loads(x) for x in (run/'diagnostics.jsonl').read_text().splitlines()]
            assert records[-1]['step']==1000000 and all(x['finite'] and x['v_updates']==0 for x in records)
            audits.append(dict(case=case['id'],method=job['method'],
                replay_rng_exact=replay_equal,agent_rng_exact=agent_equal,
                BC_flow_max_abs_difference=max(float(np.abs(d).max()) for d in delta),
                BC_flow_parameter_rms_difference=float(np.sqrt(sum(float((d*d).sum()) for d in delta)/sum(d.size for d in delta))),
                checkpoint_sha256=sha(cp),diagnostic_points=len(records),all_probes_finite=True,
                final_Q_rms=records[-1]['q_rms'],final_TD_rms=records[-1]['td_rms']))
    if require_complete:assert not pending,pending
    summary=[]
    for env in sorted({r['environment'] for r in rows}):
        for method in methods:
            for mode in ('lean','actor'):
                selected=[r for r in rows if (r['environment'],r['method'],r['deployment'])==(env,method,mode)]
                if not selected:continue
                if require_complete:assert len(selected)==3
                row=dict(environment=env,N=selected[0]['N'],method=method,deployment=mode,seeds=len(selected))
                for key in ('late3_mean','success_1M','change_800K_to_1M','change_900K_to_1M'):
                    v=np.array([r[key] for r in selected])
                    row[key+'_mean']=float(v.mean());row[key+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.
                row['worst_seed_1M']=min(r['success_1M'] for r in selected)
                row['collapse_20pp_count']=sum(r['collapse_20pp'] for r in selected)
                rank=[r for r in rankings if (r['environment'],r['method'],r['start'])==(env,method,900000)]
                row['rank_flip_900K_1M_mean']=float(np.mean([r['argmax_flip_fraction'] for r in rank])) if rank else None
                row['action_centered_Q_drift_rms_mean']=float(np.mean([r['action_centered_Q_drift_rms'] for r in rank])) if rank else None
                summary.append(row)
    if rows:write_csv(root/'analysis/per_seed_results_v1.csv',rows)
    if summary:write_csv(root/'analysis/task_summary_v1.csv',summary)
    if rankings:write_csv(root/'analysis/ranking_stability_v1.csv',rankings)
    if audits:write_csv(root/'analysis/continuation_audit_v1.csv',audits)
    result=dict(status='complete' if not pending else 'partial',timestamp_unix=time.time(),
        rows=len(rows),pending=pending,summary=summary,continuation_audits=audits,
        evaluation_evidence_sha256=paths,analysis_source_sha256=sha(__file__),
        limits=plan['limits'],training_seeds=3,episode_resets_per_checkpoint=50)
    atomic(root/'analysis/results_v1.json',result)
    if require_complete:plot(root,rows,summary)
    print(json.dumps(dict(status=result['status'],rows=len(rows),pending=len(pending),summary=summary)),flush=True)


def plot(root,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    envs=sorted({r['environment'] for r in rows})
    colors={'baseline':'#666666','q_lr_0p1':'#009e73','q_cosine':'#0072b2','temporal_Q_average':'#cc79a7'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ei,env in enumerate(envs):
        for mi,mode in enumerate(('lean','actor')):
            ax=axes[ei,mi]
            for method in LABELS:
                selected=[r for r in rows if (r['environment'],r['deployment'],r['method'])==(env,mode,method)]
                if not selected:continue
                data=np.array([[r[f'success_{s}'] for s in ('800K','900K','1M')] for r in selected])
                mean,sd=data.mean(0),data.std(0,ddof=1)
                ax.plot([800,900,1000],mean,marker='o',color=colors[method],label=LABELS[method])
                ax.fill_between([800,900,1000],np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=colors[method],alpha=.12)
            title='HM-M (N=18)' if env.startswith('humanoid') else 'Puzzle 4x4 (N=27)'
            ax.set_title(title+' / '+mode)
            ax.set_ylim(0,100);ax.set_xticks([800,900,1000]);ax.grid(alpha=.2)
            if mi==0:ax.set_ylabel('Success (%)')
            if ei==1:ax.set_xlabel('Training updates (thousands)')
            ax.legend(fontsize=8)
    fig.suptitle('800K checkpoint branches; mean +/- sample SD over 3 training seeds',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(directory/f'late_stabilization_learning_curves_v1.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,env in zip(axes,envs):
        methods=list(LABELS)
        for seed in range(3):
            selected=[next(r for r in rows if (r['environment'],r['deployment'],r['method'],r['seed'])==(env,'lean',m,seed)) for m in methods]
            ax.plot(range(4),[r['success_1M'] for r in selected],marker='o',alpha=.8,label=f'Seed {seed}')
        ax.set_xticks(range(4));ax.set_xticklabels(['Original','Q LR x0.1','Q cosine','Q average'],rotation=15)
        ax.set_title('HM-M N=18' if env.startswith('humanoid') else 'Puzzle 4x4 N=27')
        ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('1M Lean success (%)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(directory/f'paired_seed_1M_success_v1.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--baseline-rankings',action='store_true')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if args.baseline_rankings:baseline_rankings(args.plan.resolve())
    else:analyze(args.plan.resolve(),args.require_complete)
