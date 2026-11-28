"""Recompute all v11 summaries, paired intervals and publication-export figures."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from .matched_error_v11 import save_json, save_csv
from .mdp import chain, roots, expectile, evaluate_policy
from .score_controls_v11 import forced_reference


def table(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    save_csv(path,[{k:r.get(k,'') for k in keys} for r in rows])


def mean_sd(x):
    return float(np.mean(x)),float(np.std(x,ddof=1)) if len(x)>1 else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root;out=root/'analysis';figdir=out/'figures'
    figdir.mkdir(parents=True,exist_ok=True)
    records=[];groups=defaultdict(list)
    for stage in ('discovery','ridge','channels','controls','score'):
        for path in sorted((root/stage/'raw').glob('*/result.json')):
            obj=json.loads(path.read_text());c=obj['config'];r=obj['metrics']
            rec={**r,'stage':stage,'artifact':str(path.parent)}
            records.append(rec)
            cfg={k:v for k,v in c.items() if k!='seed'}
            if stage!='discovery':groups[json.dumps(cfg,sort_keys=True)].append(rec)
    summaries=[]
    for key,rs in groups.items():
        r=json.loads(key);r['seeds']=' '.join(str(x['seed']) for x in sorted(rs,key=lambda x:x['seed']))
        r['n']=len(rs)
        for metric in ('success','return','q_rmse','noise_rms','model_tv','exact_success','exact_return'):
            if metric in rs[0]:
                r[metric+'_mean'],r[metric+'_sd']=mean_sd([x[metric] for x in rs])
        summaries.append(r)
    table(out/'all_runs.csv',records);table(out/'summary.csv',summaries)
    save_json(out/'summary.json',summaries)
    def select(**criteria):
        return [r for r in records if all(r.get(k)==v for k,v in criteria.items())]
    paired=[]
    def contrast(name,left,right):
        l={r['seed']:r for r in left};r={r['seed']:r for r in right}
        assert set(l)==set(r) and len(l)>1
        seeds=sorted(l);rng=np.random.default_rng(177)
        indices=rng.integers(len(seeds),size=(20000,len(seeds)))
        for metric in ('success','return','q_rmse'):
            d=np.array([r[s][metric]-l[s][metric] for s in seeds])
            ci=np.quantile(d[indices].mean(-1),[.025,.975])
            paired.append(dict(contrast=name,metric=metric,n=len(seeds),
                right_minus_left_mean=float(d.mean()),paired_sd=float(d.std(ddof=1)),
                ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                right_greater=int((d>1e-12).sum()),equal=int((abs(d)<=1e-12).sum())))
    for steps in (2000000,5000000):
        suffix='_5m' if steps==5000000 else ''
        for other in ('feq9','feq9_double_q_penalty','iql9_no_v_penalty'):
            contrast(f'ridge_L8_{steps}_iql9_to_{other}',
                select(stage='ridge',name='iql9'+suffix,steps=steps),select(stage='ridge',name=other+suffix,steps=steps))
    contrast('ridge_L16_iql9_to_feq9',select(stage='controls',name='longer_chain_iql9'),select(stage='controls',name='longer_chain_feq9'))
    for steps,rho in [(1000000,0.),(1000000,.9),(1000000,.99),(1000000,1.),(5000000,0.)]:
        stage='controls' if rho==1. else 'channels'
        contrast(f'output_Q_to_V_sigma0.1_rho{rho}_{steps}',
                 select(stage=stage,channel=1,sigma=.1,rho=rho,steps=steps),
                 select(stage=stage,channel=2,sigma=.1,rho=rho,steps=steps))
    for other in ('iql9_v_score','iql9_v_output','feq9_q_output'):
        contrast('constant_Q_output_to_'+other,select(stage='score',variant='iql9_q_output'),select(stage='score',variant=other))
    table(out/'paired_intervals.csv',paired)
    residuals=[]
    for r in select(suite='ridge'):
        with np.load(Path(r['artifact'])/'data.npz') as h:
            q=h['q_fixed'];v=h['v_fixed'];beta=h['beta'];tau=r['tau']
            ev=roots(q,beta,tau);d=q-v[:,None]
            g=(beta*np.where(d>=0,tau,1-tau)*d).sum(-1)
            kappa=(beta*np.where(q-ev[:,None]>=0,tau,1-tau)).sum(-1)
            tail=h['step']>=.9*r['steps']
            residuals.append(dict(name=r['name'],seed=r['seed'],length=r['length'],steps=r['steps'],
                score_residual_rms=float(np.sqrt(np.mean(g*g))),
                v_output_error_rms=float(np.sqrt(np.mean((v-ev)**2))),
                q_local_fit_error_rms=float(np.sqrt(np.mean((r['lq']*q)**2))),
                min_kappa=float(kappa.min()),max_kappa=float(kappa.max()),
                amplification_rms=float(np.sqrt(np.mean((v-ev)**2))/max(np.sqrt(np.mean(g*g)),1e-30)),
                learned_to_fixed_q_rmse=float(np.sqrt(((h['q'][tail]-q)**2).mean()))))
    table(out/'residual_decomposition.csv',residuals)
    exact=[]
    for length,safe in ((8,.9),(16,.8)):
        m=chain(length,.25,.999,.9,safe)
        for eta in (0.,.0005,.001,.002,.0025,.003,.005,.01):
            for channel in ('q_output','v_output','v_score'):
                q,v=forced_reference(m,channel,eta)
                ret,suc=evaluate_policy(m,q)
                exact.append(dict(length=length,safe=safe,eta=eta,channel=channel,
                                  success=suc,return_value=ret))
    table(out/'exact_forcing_sweep.csv',exact)
    local=[]
    for tau,prob in ((.9,.25),(.99,.01)):
        q=np.array([1.,0.]);b=np.array([prob,1-prob]);v=expectile(q,b,tau)
        eta=.001;k=float(np.dot(b,np.where(q-v>=0,tau,1-tau)))
        vv=brentq(lambda x:np.dot(b,np.where(q-x>=0,tau,1-tau)*(q-x))+eta,-10,10)
        local.append(dict(tau=tau,prob_good=prob,eta=eta,kappa=k,
                          critic_shift=expectile(q+eta,b,tau)-v,score_shift=vv-v))
    table(out/'local_conditioning.csv',local)
    # One seed-level time average is one observation in every statistical table.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ch,label,color in [(1,'Q-output error','#0072B2'),(2,'V-output error','#D55E00')]:
        xs=[0.,.02,.05,.1,.2];ys=[];sd=[];returns=[]
        for sigma in xs:
            rows=select(stage='channels',steps=1000000,rho=0.,sigma=sigma,channel=0 if sigma==0 else ch)
            m,s=mean_sd([r['success']*100 for r in rows]);ys.append(m);sd.append(s)
            returns.append(np.mean([r['return'] for r in rows]))
        axes[0].errorbar(xs,ys,yerr=sd,marker='o',color=color,label=label,capsize=3)
        axes[1].plot(xs,returns,marker='o',color=color,label=label)
        rr=[0.,.9,.99,1.];ys=[];sd=[]
        for rho in rr:
            rows=select(stage='controls' if rho==1. else 'channels',steps=1000000,rho=rho,sigma=.1,channel=ch)
            m,s=mean_sd([r['success']*100 for r in rows]);ys.append(m);sd.append(s)
        axes[2].errorbar(range(4),ys,yerr=sd,marker='o',color=color,label=label,capsize=3)
    axes[0].set(xlabel='Output-noise SD (IID)',ylabel='Better-branch selection (%)',ylim=(0,105))
    axes[1].set(xlabel='Output-noise SD (IID)',ylabel='Actual discounted return')
    axes[2].set(xlabel='AR(1) coefficient (SD = 0.1)',ylabel='Better-branch selection (%)',ylim=(0,115))
    axes[2].set_xticks(range(4),['0','.9','.99','1 (fixed)'])
    axes[0].legend();fig.suptitle('Matched output error: temporal structure matters (16 seeds, 1M updates)')
    fig.tight_layout();fig.savefig(figdir/'matched_output_error.png',dpi=200);fig.savefig(figdir/'matched_output_error.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    names=[('iql9','IQL-9, both penalties','#D55E00'),('feq9','FEQ-N9, Q penalty','#009E73'),
           ('iql9_no_v_penalty','IQL-9, Q penalty only','#0072B2')]
    for name,label,color in names:
        curves=[];qcurves=[]
        for r in select(stage='ridge',name=name,steps=2000000):
            with np.load(Path(r['artifact'])/'data.npz') as h:
                m=chain(8,.25,.999,.9,.9);m.behavior=h['beta'];steps=h['step']
                curves.append([evaluate_policy(m,qq) for qq in h['q']])
                qcurves.append(np.sqrt(((h['q']-h['q_truth'])**2).mean((1,2))))
        cc=np.array(curves);qc=np.array(qcurves)
        for ax,values in zip(axes,(cc[:,:,1]*100,cc[:,:,0],qc)):
            mu=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot(steps/1e6,mu,label=label,color=color)
            ax.fill_between(steps/1e6,mu-sd,mu+sd,color=color,alpha=.16)
            ax.set_xlabel('Updates (million)')
    axes[0].set_ylabel('Goal-reaching probability (%)');axes[1].set_ylabel('Actual discounted return')
    axes[2].set_ylabel('Q RMSE against population fixed point');axes[0].legend(fontsize=8)
    fig.suptitle('Fully tabular, no external noise: explicit regression penalty 0.003 (16 seeds)')
    fig.tight_layout();fig.savefig(figdir/'regularized_learning.png',dpi=200);fig.savefig(figdir/'regularized_learning.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for length,ax in zip((8,16),axes):
        for ch,label,color,style in [('q_output','Q output','#0072B2','-'),('v_output','V output','#009E73','--'),('v_score','V score residual','#D55E00','-')]:
            rs=[r for r in exact if r['length']==length and r['channel']==ch]
            ax.plot([r['eta'] for r in rs],[100*r['success'] for r in rs],style,marker='o',color=color,label=label)
        ax.set(xlabel='Magnitude eta (declared coordinates)',ylabel='Goal-reaching probability (%)',title=f'{length+1} states; exact fixed point')
    axes[0].legend(fontsize=9);fig.suptitle('Score residual and output error are different quantities')
    fig.tight_layout();fig.savefig(figdir/'score_vs_output.png',dpi=200);fig.savefig(figdir/'score_vs_output.pdf');plt.close(fig)
    save_json(out/'analysis_manifest.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=len(records),groups=len(summaries),seed_bootstrap_resamples=20000,
        channel_tail='last 50%; fixed before confirmation',other_tail='last 10%',
        caution='success means better-branch selection in channel suite; goal reaching in chain suites'))
    print(json.dumps(dict(status='complete',runs=len(records),groups=len(summaries))))


if __name__=='__main__':main()
