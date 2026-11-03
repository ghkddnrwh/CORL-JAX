"""Frozen common-proposal rank/scale diagnostics, without changing any policy.

No action-ranking correctness is assumed. Minority magnitude domination is a
descriptive disagreement event, not evidence that either selected action is right.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import policy_core,independent_critic_policy_v1 as selector
from utils.propagation_research.reward_bounds import return_bounds
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def ranks(q):
    own=q[:,:,None];other=q[:,None,:]
    return np.sum(other<own,axis=2)+.5*np.sum(other==own,axis=2)


def summarize(q,lower):
    assert q.ndim==3 and q.shape[:2]==(3,9)and np.isfinite(q).all()
    n=q.shape[-1];rr=ranks(q);rank_score=np.median(rr,axis=0)
    rank_choice=rank_score.argmax(0);mean_score=q.mean(0);mean_choice=mean_score.argmax(0)
    own_choice=q.argmax(1);unique=np.sort(q,axis=1)[:,-1]>np.sort(q,axis=1)[:,-2]
    counts=np.sum(own_choice[:,None,:]==np.arange(9)[None,:,None],axis=0)
    majority=counts.argmax(0)
    guaranteed=np.sum((own_choice==majority[None])&unique,axis=0)>=2
    assert np.all(rank_choice[guaranteed]==majority[guaranteed])
    index=np.arange(n);delta=q[:,mean_choice,index]-q[:,rank_choice,index]
    disagree=mean_choice!=rank_choice
    minority=(np.sum(delta>0,axis=0)==1)&(np.sum(delta<0,axis=0)==2)&disagree
    sd=q.std(1);positive=np.all(sd>0,axis=0)
    ratios=sd[:,positive].max(0)/sd[:,positive].min(0)
    raw_sum=q.sum(0);margin=np.sort(raw_sum,axis=0)[-1]-np.sort(raw_sum,axis=0)[-2]
    tolerance=64*np.finfo(np.float64).eps*np.maximum(1,np.max(np.abs(raw_sum),axis=0))
    strict=margin>tolerance
    centered=q-q[:,:1]
    offset_equal=centered.mean(0).argmax(0)==mean_choice
    assert np.all(offset_equal[strict]),'Mean offset invariance failed beyond rounding/ties'
    transforms=[]
    for scales in([.1,1.,10.],[10.,1.,.1],[1.,10.,.1]):
        transformed=q*np.asarray(scales)[:,None,None]+np.asarray([-1000.,0.,1000.])[:,None,None]
        transformed_ranks=ranks(transformed)
        assert np.array_equal(transformed_ranks,rr),'Positive affine transform altered actual midranks'
        assert np.array_equal(np.median(transformed_ranks,0).argmax(0),rank_choice)
        transforms.append(dict(scales=scales,offsets=[-1000,0,1000],rank_choice_unchanged=True,
            mean_choice_changed_fraction=float(np.mean(transformed.mean(0).argmax(0)!=mean_choice))))
    corr=[]
    for i,j in((0,1),(0,2),(1,2)):
        x=rr[i]-rr[i].mean(0,keepdims=True);y=rr[j]-rr[j].mean(0,keepdims=True)
        denom=np.sqrt(np.sum(x*x,0)*np.sum(y*y,0));ok=denom>0
        corr.append(dict(members=[i,j],defined_states=int(ok.sum()),
            mean_Spearman_midrank_correlation=float(np.mean(np.sum(x*y,0)[ok]/denom[ok]))if ok.any()else None))
    violation=np.maximum(q,0)+np.minimum(q-lower,0)
    return dict(query_states=n,all_three_argmax_agreement_fraction=float(np.mean(counts.max(0)==3)),
        all_three_argmax_distinct_fraction=float(np.mean(counts.max(0)==1)),
        two_unique_argmax_majority_guarantee_fraction=float(np.mean(guaranteed)),
        majority_guarantee_observed_exact=True,raw_mean_vs_rank_choice_difference_fraction=float(disagree.mean()),
        raw_mean_minority_magnitude_domination_fraction=float(minority.mean()),
        raw_mean_minority_magnitude_domination_among_differences=float(minority.sum()/disagree.sum())if disagree.any()else None,
        median_score_top_tie_fraction=float(np.mean(np.sum(rank_score==rank_score.max(0),axis=0)>1)),
        member_action_std_mean=sd.mean(1).tolist(),member_flat_action_fraction=np.mean(sd==0,axis=1).tolist(),
        action_std_scale_ratio_defined_states=int(positive.sum()),
        action_std_max_over_min_quantiles=dict(zip(('p50','p90','p99'),np.quantile(ratios,[.5,.9,.99]).tolist()))if ratios.size else None,
        member_minQ_mean=q.mean((1,2)).tolist(),member_minQ_RMS=np.sqrt(np.mean(q*q,(1,2))).tolist(),
        known_return_interval=[lower,0.],member_minQ_true_error_RMS_lower_bound=np.sqrt(np.mean(violation*violation,(1,2))).tolist(),
        raw_mean_non_tied_states=int(strict.sum()),raw_mean_offset_invariance_non_tied_exact=bool(offset_equal[strict].all()),
        rank_correlations=corr,positive_affine_counterfactuals=transforms)


def unit_checks():
    # Two critics uniquely prefer0, the third scale-dominates the raw mean for1.
    q=np.zeros((3,9,2));q[0,0]=2;q[0,1]=1;q[1,0]=3;q[1,1]=1;q[2,0]=-100;q[2,1]=100
    out=summarize(q,-1000.)
    assert out['two_unique_argmax_majority_guarantee_fraction']==1
    assert out['raw_mean_minority_magnitude_domination_fraction']==1
    assert out['raw_mean_vs_rank_choice_difference_fraction']==1
    constant=summarize(np.zeros((3,9,2)),-1000.)
    assert constant['member_flat_action_fraction']==[1.,1.,1.]and constant['action_std_max_over_min_quantiles']is None
    assert constant['median_score_top_tie_fraction']==1 and constant['raw_mean_vs_rank_choice_difference_fraction']==0
    return dict(two_unique_critics_vs_extreme_third=True,flat_tied_critics_defined=True,all_positive_affine_rank_checks=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not(out/'progress.json').exists(),'Inspect original process/handle before any recovery'
    units=unit_checks();plan=json.loads(args.plan.read_text())
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    rows=[];models=[];progress=dict(pid=__import__('os').getpid(),started_at=time.time(),source_sha256=sha(__file__),
        plan_sha256=sha(args.plan),query_rng_seed=91349,queries_per_split=512,unit_checks=units)
    for environment in plan['environments']:
        env,train,val=ogbench.make_env_and_datasets(environment)
        data_sets={};rng=np.random.RandomState(91349)
        for split,data in(('train',train),('validation',val)):
            indices=rng.choice(len(data['observations']),512,replace=False)
            data_sets[split]=(np.asarray(data['observations'][indices],np.float32),indices)
        job0=next(job for job in plan['jobs']if job['environment']==environment)
        conf0=json.loads((Path(job0['actor_run'])/'config.json').read_text());cfg=dict(conf0['identity']['config'])
        for field in('actor_hidden_dims','value_hidden_dims'):cfg[field]=tuple(cfg[field])
        template=policy_core.create(conf0['identity']['seed'],jnp.asarray(train['observations'][:1]),
            jnp.asarray(train['actions'][:1]),cfg,'ifql')
        compiled=selector.make_selector(template.agent,'median_rank',9)
        lower=return_bounds(environment,cfg['discount'],float(np.min(train['rewards'])),float(np.max(train['rewards'])),'ogbench')[0]
        raw={};configs={}
        for step in(800000,900000,1000000):
            job=next(job for job in plan['jobs']if job['environment']==environment and job['step']==step)
            critic_params=[];actor=None
            for run_name,pin in zip(job['critic_runs'],job['member_pins']):
                run=Path(run_name);conf=json.loads((run/'config.json').read_text());ident=conf['identity']
                assert ident['environment']==environment and ident['profile']=='bounded_v2'and ident['kind']=='ifql'
                assert ident['steps']==1000000 and ident['episodes']==50
                assert(run/'completed.json').exists()and not(run/'stopped.json').exists()
                assert sha(run/'config.json')==pin['config_sha256']
                for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
                cp=run/f'checkpoint_{step}.pkl';content=cp.read_bytes();digest=hashlib.sha256(content).hexdigest()
                assert digest==pin['checkpoint_sha256'];saved=pickle.loads(content);del content
                assert saved['step']==int(saved['state']['q_updates'])==step
                assert saved['source_sha256']==conf['source_sha256']and json.loads(json.dumps(saved['identity']))==ident
                assert int(saved['state']['agent']['network']['opt_state']['0']['count'])==step
                assert int(saved['state']['v_updates'])==int(saved['state']['v_opt']['0']['count'])==2*step
                assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(saved['state']))
                critic_params.append(jax.device_put(saved['state']['agent']['network']['params']['modules_critic'],jax.devices('cpu')[0]))
                if actor is None:actor=flax.serialization.from_state_dict(template,saved['state'])
                models.append(dict(run=run_name,step=step,checkpoint_sha256=digest,config_sha256=pin['config_sha256']))
                configs[run_name]=pin['config_sha256']
                del saved
            for split,(observations,indices)in data_sets.items():
                values=[]
                for start in range(0,len(observations),128):
                    key=jax.random.fold_in(jax.random.PRNGKey(19405 if split=='train' else 19406),start)
                    _,q,_=compiled(actor.agent.network.params,tuple(critic_params),jnp.asarray(observations[start:start+128]),key)
                    values.append(np.asarray(q,np.float64))
                q=np.concatenate(values,axis=2);raw[f'{split}_{step}_Q']=q;raw[f'{split}_indices']=indices
                rows.append(dict(environment=environment,split=split,step=step,index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),**summarize(q,lower)))
            atomic(out/'progress.json',progress|dict(updated_at=time.time(),completed_environment_steps=len(rows)//2,rows=rows,models=models,status='running'))
            del actor,critic_params
        short=Path(job0['actor_run']).name.split('_')[1]
        np.savez_compressed(out/(short+'_values.npz'),**raw)
        env.close();del compiled,template,train,val
    result=progress|dict(completed_at=time.time(),status='complete',rows=rows,models=models,models_audited=len(models),
        unique_environments=len(plan['environments']),primary_scores_unchanged=True,
        interpretation=['Rawmeanandranksbothignorecritic-wisestateoffsets;ranksaddmonotone/relativescaleinvariance.',
            'Minoritymagnitudedominationisamechanicaldisagreementevent,notevidencethattheminorityactioniswrong.',
            'Counterfactualscalesarenumericinvariancetests,notnewpolicyperformanceevaluations.'],
        limits=['Nooraclewithin-stateactionrank,conditionalresidualorinteriorQerrorlabel.',
            'Fixed512train/512validationquerystatesandoneBCdrawperquery;notallvisiteddeploymentstates.',
            'Constituentmodelsareseen0/1/2;thisdoesnotaddanindependentensemblegroup.',
            'Boundsyieldonlyerrorlowerbounds;nearlyzerodoesnotcertifyaccurateQ.'],goal_complete=False)
    atomic(out/'summary.json',result);atomic(out/'progress.json',result)
    print(json.dumps(dict(status='complete',models=len(models),rows=len(rows),primary_scores_unchanged=True)),flush=True)


if __name__=='__main__':main()
