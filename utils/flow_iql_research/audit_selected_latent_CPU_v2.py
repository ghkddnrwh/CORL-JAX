"""Verify numerical optimizer subtrees; source-hash metadata checked separately."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research.watch_onestep_deploy_v8 import arrays
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    proposal=json.loads((a.study/'proposal_v1.json').read_text());rows=[]
    for case in ('puzzle3_v1','cube_single_v1'):
        path=a.study/case/'result.json';r=json.loads(path.read_text())
        assert r['status']=='complete'and r['proposal_sha256']==sha(a.study/'proposal_v1.json')
        for source,digest in r['source_sha256'].items():assert sha(source)==digest,source
        assert r['teacher_optimizer_updates']==0 and not r['evaluation_data_used_for_learning']
        assert sha(r['teacher_checkpoint'])==r['teacher_checkpoint_sha256']
        assert sha(r['private_artifact'])==r['private_artifact_sha256']
        assert sha(r['optimizer_state_artifact'])==r['optimizer_state_sha256']
        with Path(r['private_artifact']).open('rb')as f:actor=pickle.load(f)
        with Path(r['optimizer_state_artifact']).open('rb')as f:optimizer=pickle.load(f)
        for part in ('router','decoder'):
            state=optimizer[part+'_state'];assert int(state['opt_state']['0']['count'])==100000
            left=dict(arrays(state['params']));right=dict(arrays(actor['actor_params'][part]))
            assert left.keys()==right.keys()
            for k,v in left.items():assert v.dtype==right[k].dtype and v.shape==right[k].shape and v.tobytes()==right[k].tobytes()
        assert optimizer['source_sha256']==r['source_sha256']
        numeric_states={k:optimizer[k]for k in ('router_state','decoder_state')}
        assert all(np.isfinite(x).all()for _,x in arrays(numeric_states))
        assert actor['metadata']['MLP_blocks']==2 and actor['metadata']['actor_forward_passes']==1
        assert actor['metadata']['generated_candidate_actions']==0
        points={}
        for mode,entry in r['evaluations'].items():
            assert sha(entry['evaluation_path'])==entry['evaluation_sha256']
            evaluated=json.loads(Path(entry['evaluation_path']).read_text())
            assert evaluated['n_episodes']==50 and evaluated['first_episode_seed']==42
            assert evaluated['actor_artifact_sha256']==r['private_artifact_sha256']
            assert evaluated['no_optimizer_updates']and not evaluated['counts_toward_native_goal']
            assert len(evaluated['episode_successes'])==50 and all(x in (0,1)for x in evaluated['episode_successes'])
            successes=sum(evaluated['episode_successes']);assert successes/50==entry['success']
            points[mode]=dict(successes=successes,episodes=50,success=successes/50,
                hybrid_uses_teacher=evaluated['hybrid_uses_teacher'])
        assert set(points)=={'composite','learned_router_exact_Flow','exact_winner_fast_decoder','prior_fast_decoder'}
        rows.append(dict(case=case,result_sha256=sha(path),points=points,
            actual_router_Adam=100000,actual_decoder_Adam=100000,export_params_exact=True,
            full_saved_optimizer_state_finite=True,teacher_CPU50=r['stochastic_best9_teacher_CPU50']))
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        source_sha256=sha(__file__),proposal_sha256=sha(a.study/'proposal_v1.json'),
        total_CPU_evaluation_episodes=400,
        native_GPU_expansion_criterion_met=all(r['points']['composite']['success']>=.7 for r in rows),
        scope='Two fixed-teacher CPU cases with component interventions; no joint/full9/fresh/final-speed claim.'),f,indent=2,allow_nan=False)
    print(json.dumps(rows))


if __name__=='__main__':main()
