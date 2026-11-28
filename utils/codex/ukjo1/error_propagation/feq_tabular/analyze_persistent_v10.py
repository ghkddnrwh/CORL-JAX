"""Complete aggregation, exact error-channel interventions and scientific figures."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .persistent_features_v10 import setup, references, evaluate
from .persistent_matched_features_v10 import setup as matched_setup, evaluate as matched_evaluate
from .mdp import chain, solve, roots, evaluate_policy
from .fair_population_v7 import expected_iql
from .theory_channels_v9 import score, score_root, secant_diagnostics
from .fair_study_v7 import write_json

LABELS = {'iql':'IQL', 'iql9':'IQL (9 V queries)', 'iql_batch9':'IQL (9 V queries)',
          'feq9':'FEQ-N9', 'iql_expanded':'IQL (realizable V)', 'repair_zero':'V-residual oracle'}
COLORS = {'iql':'#d55e00', 'iql9':'#e69f00', 'iql_batch9':'#e69f00',
          'feq9':'#009e73', 'iql_expanded':'#0072b2', 'repair_zero':'#0072b2'}


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        r = dict(zip(keys, key)); r['seeds'] = len(group)
        for metric in metrics:
            values = np.array([x[metric] for x in group])
            r[metric+'_mean'] = float(values.mean())
            r[metric+'_sd'] = float(values.std(ddof=1)) if len(group) > 1 else 0.
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    out = root/'analysis'; out.mkdir(exist_ok=True)
    rows = []; data = {}; metadata = {}
    for stage in ('features', 'tabular', 'matched_features', 'matched_tau'):
        for path in sorted((root/stage/'raw').glob('*/result.json')):
            meta = json.loads(path.read_text()); r = dict(meta['metrics']); r['stage'] = stage
            key = stage, r['case'], r['method'], r['seed']
            d = dict(np.load(path.with_name('data.npz'))); tail = d['step'] >= .9*r['steps']
            r['discounted_return'] = float(d['policy'][tail, 0].mean())
            r['deployed_return'] = float(d['model_policy'][tail, 0].mean())
            r['final_success'] = float(d['policy'][-1, 1])
            r['final_return'] = float(d['policy'][-1, 0])
            r['final_q_rmse'] = float(d['rmse'][-1])
            rows.append(r); data[key] = d; metadata[key] = meta
    csv_write(out/'per_seed.csv', rows)
    metrics = ('success','q_rmse','discounted_return','deployed_success','deployed_return',
               'final_success','final_q_rmse','final_return','model_tv')
    summary = aggregate(rows, ('stage','case','method'), metrics)
    csv_write(out/'summary.csv', summary)
    # Paired differences over dataset/training seeds, not over time checkpoints.
    paired = []; rng = np.random.default_rng(250011)
    groups = defaultdict(dict)
    for r in rows:
        groups[r['stage'], r['case']][r['method'], r['seed']] = r
    for (stage, case), group in sorted(groups.items()):
        seeds = sorted({s for m, s in group if m == 'feq9'})
        for baseline in ('iql','iql9','iql_batch9'):
            common = [s for s in seeds if (baseline, s) in group]
            if not common:
                continue
            indices = rng.integers(len(common), size=(20000, len(common)))
            for metric in ('success','discounted_return','q_rmse'):
                x = np.array([group['feq9',s][metric]-group[baseline,s][metric] for s in common])
                ci = np.quantile(x[indices].mean(-1), [.025,.975])
                paired.append(dict(stage=stage, case=case, baseline=baseline, metric=metric,
                    seeds=len(common), difference_mean=float(x.mean()),
                    difference_sd=float(x.std(ddof=1)), ci95_low=float(ci[0]), ci95_high=float(ci[1])))
    csv_write(out/'paired_differences.csv', paired)
    windows = []
    for (stage, case, method, seed), d in sorted(data.items()):
        if case not in ('main', 'five_million'):
            continue
        steps = d['step']
        for start in np.arange(0, steps[-1], 200000):
            mask = (steps > start) & (steps <= start+200000)
            if mask.any():
                windows.append(dict(stage=stage,case=case,method=method,seed=seed,
                    start=int(start),end=int(min(start+200000,steps[-1])),
                    success=float(d['policy'][mask,1].mean()),
                    discounted_return=float(d['policy'][mask,0].mean()),q_rmse=float(d['rmse'][mask].mean())))
    csv_write(out/'windows_per_seed.csv', windows)
    csv_write(out/'windows_summary.csv', aggregate(windows, ('stage','case','method','start','end'),
                                                   ('success','discounted_return','q_rmse')))

    # Natural persistent V fitting residual, then two controlled primitive-error channels.
    forcing = []; natural = []; exact = []
    levels = (0., .1, .25, .5, .75, 1., 1.25, 1.5)
    channel_cases=[('features',.9,range(1300,1316)),('features',.99,range(1300,1316)),
                   ('matched_features',.99,range(1500,1516))]
    for scope,tau,seeds in channel_cases:
        for seed in seeds:
            if scope=='features':
                m, phi, expanded, ds, da, occ, model, true_beta = setup(seed, tau)
                eval_fn=evaluate;deploy=m.behavior
            else:
                m, phi, expanded, ds, da, occ, model, true_beta, model_theta = matched_setup(seed)
                eval_fn=matched_evaluate;deploy=model
            ref = references(m, phi, occ, model); zeta = ref['leaf_score_residual']
            v0 = roots(m.reward, m.behavior, tau)
            for s in (1, 2, 3):
                natural.append(dict(stage=scope,tau=tau, seed=seed, state=s,
                    zeta=float(zeta[s-1]), value_gap=float(ref['leaf_value_gap'][s-1]),
                    kappa=float(ref['leaf_kappa'][s-1]), secant_kappa=float(ref['leaf_secant'][s-1]),
                    amplification=float(1/ref['leaf_secant'][s-1]),
                    true_v=float(v0[s]), fitted_v=float(ref['v_iql'][s]),
                    high_action_probability=float(m.behavior[s, [0,2]].sum())))
            for method, q in (('iql_exact_projected',ref['q_iql']),('feq9_exact_expected',ref['q_feq9']),
                              ('population',ref['q_truth'])):
                ret, suc = eval_fn(m, q, deploy)
                exact.append(dict(stage=scope, tau=tau, seed=seed, method=method,
                    success=suc, discounted_return=ret, q_rmse=float(np.sqrt(((q-ref['q_truth'])**2).mean()))))
            for level in levels:
                primitive = level*zeta
                # At each leaf, the same signed scalar pattern goes into either
                # G+zeta=0 or all Q outputs. Equal sup norm and statewise magnitude.
                vs = v0.copy()
                vs[1:] = [score_root(m.reward[s],m.behavior[s],tau,primitive[s-1]) for s in (1,2,3)]
                qs = m.reward.copy(); qs[0] += m.gamma*vs[m.successor[0]]
                qc = m.reward.copy(); qc[1:] += primitive[:,None]
                vc = roots(qc,m.behavior,tau); qc[0] += m.gamma*vc[m.successor[0]]
                np.testing.assert_allclose(vc[1:]-v0[1:],primitive,atol=1e-12)
                if level == 1:
                    np.testing.assert_allclose(qs,ref['q_iql'],atol=2e-12)
                for channel,q,v in (('score_residual',qs,vs),('critic_output',qc,vc)):
                    ret,suc = eval_fn(m,q,deploy)
                    forcing.append(dict(stage=scope,tau=tau,seed=seed,level=level,channel=channel,
                        primitive_max=float(abs(primitive).max()),primitive_rms=float(np.sqrt((primitive**2).mean())),
                        value_gap_rms=float(np.sqrt(((v[1:]-v0[1:])**2).mean())),
                        value_gap_max=float(abs(v[1:]-v0[1:]).max()),
                        q_rmse=float(np.sqrt(((q-ref['q_truth'])**2).mean())),
                        propagated_root_q_rmse=float(np.sqrt(((q[0]-ref['q_truth'][0])**2).mean())),
                        success=suc,discounted_return=ret))
                # Equal POST-expectile displacement must give exactly equal root targets.
                q_equal = m.reward.copy(); q_equal[1:] += (vs-v0)[1:,None]
                vequal = roots(q_equal,m.behavior,tau)
                np.testing.assert_allclose(vequal[1:],vs[1:],atol=2e-12)
    csv_write(out/'natural_score_residual.csv',natural)
    csv_write(out/'matched_primitive_channels.csv',forcing)
    channel_summary = aggregate(forcing,('stage','tau','level','channel'),
        ('primitive_max','primitive_rms','value_gap_rms','value_gap_max','q_rmse',
         'propagated_root_q_rmse','success','discounted_return'))
    csv_write(out/'matched_primitive_summary.csv',channel_summary)
    tabular_residual = []
    for seed in range(1100,1116):
        d = data['tabular','main','iql',seed]
        m=chain(4,.25,.999,.9,.8);m.reward[0,1]=0.;m.continuation[0,1]=1.;m.behavior=d['behavior']
        qi,vi=expected_iql(m,.002);qf,vf,_=solve(m,samples=9)
        diag=secant_diagnostics(qi,vi,m.behavior,m.tau)
        for method,q in (('iql_exact_expected_loss',qi),('feq9_exact_expected',qf),('population',d['q_truth'])):
            ret,suc=evaluate_policy(m,q)
            exact.append(dict(stage='tabular',tau=.9,seed=seed,method=method,success=suc,
                discounted_return=ret,q_rmse=float(np.sqrt(((q-d['q_truth'])**2).mean()))))
        for s in range(5):
            tabular_residual.append(dict(seed=seed,state=s,sigma_q=.002,zeta=float(diag['zeta'][s]),
                value_gap=float(diag['gap'][s]),local_kappa=float(diag['kappa'][s]),secant_kappa=float(diag['secant'][s])))
    csv_write(out/'tabular_expected_loss_residual.csv',tabular_residual)
    for (stage,case,method,seed),d in sorted(data.items()):
        if stage not in ('matched_features','matched_tau') or method!='iql':continue
        if stage=='matched_features' and case!='main':continue
        m,*_=matched_setup(seed)
        m.tau=metadata[stage,case,method,seed]['identity']['case'].get('tau',.99)
        for name,key in [('iql9_exact_projected','q_iql9')]+(
            [('iql_exact_projected','q_iql'),('feq9_exact_expected','q_feq9'),('population','q_truth')]
            if stage=='matched_tau' else []):
            ret,suc=matched_evaluate(m,d[key],d['model'])
            exact.append(dict(stage=stage,tau=m.tau,seed=seed,method=name,success=suc,
                discounted_return=ret,q_rmse=float(np.sqrt(((d[key]-d['q_truth'])**2).mean()))))
    csv_write(out/'exact_limits.csv',exact)
    csv_write(out/'exact_limits_summary.csv',aggregate(exact,('stage','tau','method'),('success','discounted_return','q_rmse')))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for stage in ('tabular','features','matched_features'):
        fig,axs=plt.subplots(2,3,figsize=(13.5,7.4))
        for r,case in enumerate(('main','five_million')):
            methods=['iql','iql_batch9' if stage=='tabular' else 'iql9','feq9']
            if case=='main':methods.append('repair_zero' if stage=='tabular' else 'iql_expanded')
            for method in methods:
                curves=[d for (st,ca,me,se),d in data.items() if (st,ca,me)==(stage,case,method)]
                time=curves[0]['step']/1e6
                for col,field,coordinate,scale in ((0,'policy',1,100.),(1,'policy',0,1.),(2,'rmse',None,1.)):
                    y=np.stack([d[field] if coordinate is None else d[field][:,coordinate] for d in curves])*scale
                    mean=y.mean(0);sd=y.std(0,ddof=1);ax=axs[r,col]
                    ax.plot(time,mean,label=LABELS[method],color=COLORS[method],lw=1.5)
                    ax.fill_between(time,mean-sd,mean+sd,color=COLORS[method],alpha=.13)
                    ax.set_xlabel('Training updates (million)');ax.grid(alpha=.15)
            axs[r,0].set_ylim(-3,103)
            for col,label in enumerate(('Goal/success probability (%)','Discounted return','Q RMSE')):
                axs[r,col].set_ylabel(label);axs[r,col].set_title(f'{case}: {len(curves)} paired seeds')
            axs[r,0].legend(fontsize=8)
        fig.suptitle('Noisy full-tabular model: identical Q error, no V noise' if stage=='tabular'
                     else ('Q, V and behavior share the same state features; V and behavior each have 3 parameters'
                           if stage=='matched_features' else 'Finite MDP with shared state features: no injected Q or V noise'))
        fig.tight_layout();fig.savefig(figures/f'{stage}_long_run.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for row,tau in enumerate((.9,.99)):
        for ch,color,label in (('score_residual','#d55e00','Expectile score residual'),('critic_output','#0072b2','Critic output error')):
            rs=[r for r in channel_summary if r['stage']=='features' and r['tau']==tau and r['channel']==ch]
            for col,metric,factor in ((0,'success',100.),(1,'discounted_return',1.),(2,'propagated_root_q_rmse',1.)):
                x=np.array([r['level'] for r in rs]);y=np.array([r[metric+'_mean'] for r in rs])*factor
                sd=np.array([r[metric+'_sd'] for r in rs])*factor
                axs[row,col].plot(x,y,'o-',color=color,label=label)
                axs[row,col].fill_between(x,y-sd,y+sd,color=color,alpha=.12)
                axs[row,col].set_xlabel('Multiplier of naturally occurring score residual')
                axs[row,col].grid(alpha=.15);axs[row,col].set_title(f'Expectile tau={tau}')
        for col,label in enumerate(('Success probability (%)','Discounted return','Propagated root Q RMSE')):
            axs[row,col].set_ylabel(label)
        axs[row,0].set_ylim(-3,103);axs[row,0].legend(fontsize=8)
    fig.suptitle('Equal primitive magnitudes and signed state patterns; 16 seeds; fixed score normalization')
    fig.tight_layout();fig.savefig(figures/'matched_error_channels.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for method in ('iql','iql9','feq9'):
        rs=sorted([r for r in summary if r['stage']=='matched_tau' and r['method']==method],key=lambda r:float(r['case'][3:]))
        xx=[float(r['case'][3:]) for r in rs]
        for ax,metric,factor,label in zip(axs,('success','discounted_return','q_rmse'),(100,1,1),
                                         ('Positive-reward success (%)','Discounted return','Q RMSE')):
            ax.errorbar(xx,[r[metric+'_mean']*factor for r in rs],
                        yerr=[r[metric+'_sd']*factor for r in rs],label=LABELS[method],color=COLORS[method],marker='o',capsize=3)
            ax.set_xlabel('Expectile level');ax.set_ylabel(label);ax.grid(alpha=.15)
    axs[1].legend(fontsize=8);fig.suptitle('Same features and parameter counts for V and behavior; 8 new paired seeds')
    fig.tight_layout();fig.savefig(figures/'shared_feature_tau_sweep.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for col,tau in enumerate((.9,.99)):
        m,phi,ex,ds,da,occ,model,tb=setup(1300,tau);ref=references(m,phi,occ,model)
        x=np.array([-1,0,1]);v=roots(m.reward,m.behavior,tau)[1:]
        axs[col].plot(x,m.reward[1:,0],'--',color='#777777',label='True Q (high action)')
        axs[col].plot(x,m.reward[1:,1],':',color='#777777',label='True Q (low action)')
        axs[col].plot(x,v,'o-',color='#009e73',label='Exact expectile of current Q')
        axs[col].plot(x,ref['v_iql'][1:],'s-',color='#d55e00',label='Globally optimal linear V fit')
        axs[col].set_title(f'tau={tau}, dataset seed 1300');axs[col].set_xlabel('Terminal-state feature x')
        axs[col].set_ylabel('Q or V');axs[col].set_xticks(x);axs[col].grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'persistent_v_projection.png',dpi=180);plt.close(fig)
    # One compact machine-readable completion record for downstream reporting.
    write_json(out/'analysis_metadata.json',dict(status='complete',training_runs=len(rows),
        exact_limits=len(exact),matched_primitive_cases=len(forcing),equal_post_expectile_controls=len(forcing)//2,
        bootstrap_resamples=20000,sd='sample SD across seed-level last-10%-window averages',
        no_test_set_optimization='Training settings were registered before these seeds were trained.',
        metric_caveat='Feature success means selecting an endpoint then its high-reward action; discounted return is primary corroboration. The uniform-behavior control can rationally prefer the central branch.',
        fixed_point_caveat='Exact expected-loss and expected-operator limits are distinct from finite constant-step SGD means.'))
    print(json.dumps(dict(status='complete',runs=len(rows),figures=str(figures))))


if __name__=='__main__':
    main()
