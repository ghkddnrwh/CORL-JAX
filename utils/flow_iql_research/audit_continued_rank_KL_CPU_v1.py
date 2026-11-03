"""Verify actual200K continued-router artifacts and matched frozen controls."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research.watch_onestep_deploy_v8 import arrays
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import numpy as np


def equal(left,right):
    a,b=dict(arrays(left)),dict(arrays(right));assert a.keys()==b.keys()
    for k,v in a.items():assert v.dtype==b[k].dtype and v.shape==b[k].shape and v.tobytes()==b[k].tobytes(),k


def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+');a=p.parse_args();assert not a.output.exists()
    proposal=json.loads((a.study/'proposal_v1.json').read_text());cases=a.cases or list(proposal['cases'])
    assert len(cases)==len(set(cases))and set(cases)<=set(proposal['cases'])
    rows=[];groups={}
    for case in cases:
        entry=proposal['cases'][case];path=a.study/case/'result.json';r=json.loads(path.read_text())
        assert r['status']=='complete'and r['proposal_sha256']==sha(a.study/'proposal_v1.json')
        pp=Path(entry['parent_result']);assert sha(pp)==entry['parent_result_sha256']==r['parent_result_sha256']
        parent=json.loads(pp.read_text())
        for p,h in r['source_sha256'].items():assert sha(p)==h,p
        assert sha(entry['cache'])==r['cache_sha256']==parent['cache_sha256']
        assert sha(entry['teacher_checkpoint'])==r['teacher_checkpoint_sha256']==parent['teacher_checkpoint_sha256']
        assert sha(parent['private_artifact'])==r['baseline_actor_sha256']==parent['private_artifact_sha256']
        assert sha(parent['optimizer_artifact'])==r['baseline_optimizer_sha256']==parent['optimizer_sha256']
        assert r['full_optimizer_moments_resumed']and r['minibatch_RNG_advanced_updates']==100000
        assert r['objective_kind']==entry['objective']and r['forward_coefficient']==entry['forward_coefficient']
        assert r['baseline_router_Adam']==r['additional_router_Adam']==100000 and r['actual_router_Adam']==200000
        assert r['decoder_Adam_performed']==r['teacher_Adam_performed']==0
        assert sha(r['private_artifact'])==r['private_artifact_sha256']and sha(r['optimizer_artifact'])==r['optimizer_sha256']
        with Path(r['private_artifact']).open('rb')as f:new=pickle.load(f)
        with Path(r['optimizer_artifact']).open('rb')as f:optimizer=pickle.load(f)
        with Path(parent['private_artifact']).open('rb')as f:old=pickle.load(f)
        assert int(optimizer['opt_state']['0']['count'])==new['actual_router_Adam']==200000
        assert new['baseline_router_Adam']==new['additional_router_Adam']==100000 and new['decoder_Adam_performed']==0
        equal(new['actor_params']['router'],optimizer['params']);equal(old['actor_params']['decoder'],new['actor_params']['decoder'])
        assert all(np.isfinite(v).all()for _,v in arrays(optimizer))
        m=new['metadata'];assert m['latent_draws']==m['actor_forward_passes']==m['realized_actions']==1
        assert m['coupling_layers']==4 and m['MLP_blocks']==5 and m['dense_layers']==17
        assert m['generated_candidate_actions']==m['flow_integration_steps']==0
        assert not m['requires_critic']and not m['requires_BC_flow']
        assert [s['actual_router_Adam']for s in r['snapshots']]==list(range(100000,200001,5000))
        assert [s['additional_Adam']for s in r['snapshots']]==list(range(0,100001,5000))
        for s in r['snapshots']:assert s['training_objective_kind']==entry['objective']
        evaluations={}
        for mode,e in r['evaluations'].items():
            assert sha(e['evaluation_path'])==e['evaluation_sha256']
            v=json.loads(Path(e['evaluation_path']).read_text());assert v['n_episodes']==50 and v['first_episode_seed']==42
            assert v['actor_artifact_sha256']==r['private_artifact_sha256']and v['no_optimizer_updates']and not v['counts_toward_native_goal']
            assert len(v['episode_successes'])==50 and all(x in (0,1)for x in v['episode_successes'])
            score=sum(v['episode_successes'])/50;assert score==e['success'];evaluations[mode]=score
        assert set(evaluations)=={'composite','learned_router_exact_Flow'}
        assert r['reused_controls']==parent['reused_controls']and r['baseline_NLL_evaluations']==parent['evaluations']
        for e in [*r['reused_controls'].values(),*r['baseline_NLL_evaluations'].values()]:assert sha(e['evaluation_path'])==e['evaluation_sha256']
        groups.setdefault(str(pp),[]).append((case,r))
        rows.append(dict(case=case,objective=entry['objective'],forward_coefficient=entry['forward_coefficient'],
            actual_total_Adam=200000,actual_additional_Adam=100000,decoder_bytes_exact_parent=True,
            export_params_exact=True,full_optimizer_finite=True,evaluations=evaluations,
            baseline100K_evaluations={k:v['success']for k,v in parent['evaluations'].items()},result_sha256=sha(path)))
    matches=[]
    for parent,items in groups.items():
        reference=items[0][1]
        for case,r in items[1:]:
            assert r['baseline_optimizer_sha256']==reference['baseline_optimizer_sha256']
            for key in ['actor_model_Q','estimated_prior_KL','rank_CDF_mean','rank_CDF_min','sampling_latent_norm_squared']:
                assert abs(r['snapshots'][0][key]-reference['snapshots'][0][key])<1e-5,key
        matches.append(dict(parent=parent,reviewed_sibling_cases=[x[0]for x in items],identical_initial_optimizer_and_diagnostic_policy=True))
    result=dict(time=time.time(),status='pass'if len(cases)==len(proposal['cases'])else'reviewed_complete_subset',
        reviewed_cases=cases,planned_cases=list(proposal['cases']),rows=rows,matched_initial_states=matches,
        actual_new_evaluation_episodes=100*len(cases),reused_evaluations_not_recounted=True,
        no_native_or_final_goal_claim=True,source_sha256=sha(__file__),proposal_sha256=sha(a.study/'proposal_v1.json'))
    with a.output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(rows))


if __name__=='__main__':main()
