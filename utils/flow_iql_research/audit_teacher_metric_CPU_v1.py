"""Audit only completed prespecified coefficients; retain incomplete methods."""
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
    p.add_argument('--coefficients',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();coefficients=[int(x)for x in a.coefficients.split(',')]
    proposal=json.loads((a.study/'proposal_v1.json').read_text());assert set(coefficients).issubset({0,1,10})
    rows=[]
    for case in ('puzzle3_v1','cube_single_v1'):
        status=json.loads((a.study/case/'status.json').read_text())
        assert status['proposal_sha256']==sha(a.study/'proposal_v1.json')
        assert status['teacher_optimizer_updates']==0 and not status['evaluation_states_used_for_learning']
        for path,digest in status['source_sha256'].items():assert sha(path)==digest,path
        for coefficient in coefficients:
            name='metric_lambda'+str(coefficient);r=status['actors'][name];assert r['status']=='complete'
            assert sha(r['actor_path'])==r['actor_sha256']and sha(r['optimizer_path'])==r['optimizer_sha256']
            assert sha(r['evaluation_path'])==r['evaluation_sha256']
            with Path(r['actor_path']).open('rb')as f:actor=pickle.load(f)
            with Path(r['optimizer_path']).open('rb')as f:optimizer=pickle.load(f)
            assert int(optimizer['opt_state']['0']['count'])==actor['actual_Adam']==100000
            assert actor['coefficient']==coefficient and not actor['jointly_trained']
            left=dict(arrays(optimizer['params']));right=dict(arrays(actor['actor_params']))
            assert left.keys()==right.keys()
            for k,v in left.items():assert v.dtype==right[k].dtype and v.shape==right[k].shape and v.tobytes()==right[k].tobytes()
            assert all(x.dtype.kind in 'biufc'and np.isfinite(x).all()for _,x in arrays(optimizer))
            evaluated=json.loads(Path(r['evaluation_path']).read_text())
            assert evaluated['n_episodes']==50 and evaluated['first_episode_seed']==42
            assert evaluated['learned_model_trees_loaded']==['actor_params']
            assert evaluated['actor_forward_passes']==1 and evaluated['generated_candidate_actions']==0
            assert not evaluated['requires_critic']and not evaluated['requires_BC_flow']
            assert all(x in (0,1)for x in evaluated['episode_successes'])and len(evaluated['episode_successes'])==50
            successes=sum(evaluated['episode_successes']);assert successes/50==r['success']
            rows.append(dict(case=case,coefficient=coefficient,actual_Adam=100000,
                exported_params_exact=True,full_optimizer_finite=True,successes=successes,episodes=50,
                success=successes/50,actor_sha256=r['actor_sha256'],evaluation_sha256=r['evaluation_sha256'],
                recorded_heldout_diagnostics=r['snapshots'][-1],diagnostics_independently_recomputed=False))
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        source_sha256=sha(__file__),proposal_sha256=sha(a.study/'proposal_v1.json'),
        selected_completed_coefficients=coefficients,total_verified_evaluation_episodes=50*len(rows),
        incomplete_prespecified_coefficients_not_dropped=sorted({0,1,10}-set(coefficients)),
        scope='Fixed-teacher CPU results, not native joint last3/full9/fresh or a true-value oracle. Loss diagnostics read from the sealed completed method records, not independently recomputed.'),f,indent=2,allow_nan=False)
    print(json.dumps([{k:r[k]for k in ('case','coefficient','success','actual_Adam')}for r in rows]))


if __name__=='__main__':main()
