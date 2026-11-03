"""Check completed router-only studies and unchanged decoder/control lineage."""
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
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    proposal=json.loads((a.study/'proposal_v1.json').read_text());rows=[]
    for case,entry in proposal['cases'].items():
        path=a.study/case/'result.json';r=json.loads(path.read_text());assert r['status']=='complete'
        assert r['proposal_sha256']==sha(a.study/'proposal_v1.json')
        assert sha(entry['parent_result'])==entry['parent_result_sha256']==r['parent_result_sha256']
        parent=json.loads(Path(entry['parent_result']).read_text())
        assert sha(parent['private_artifact'])==parent['private_artifact_sha256']==r['frozen_decoder_parent_sha256']
        assert sha(r['private_artifact'])==r['private_artifact_sha256']
        assert sha(r['optimizer_artifact'])==r['optimizer_sha256']
        for source,digest in r['source_sha256'].items():assert sha(source)==digest
        with Path(parent['private_artifact']).open('rb')as f:old=pickle.load(f)
        with Path(r['private_artifact']).open('rb')as f:new=pickle.load(f)
        with Path(r['optimizer_artifact']).open('rb')as f:optimizer=pickle.load(f)
        assert int(optimizer['opt_state']['0']['count'])==new['actual_router_Adam']==r['actual_router_Adam']==100000
        assert r['decoder_Adam_performed']==new['decoder_Adam_performed']==0
        assert new['decoder_parent_Adam']==100000 and r['teacher_Adam_performed']==0
        equal(optimizer['params'],new['actor_params']['router'])
        equal(old['actor_params']['decoder'],new['actor_params']['decoder'])
        assert all(x.dtype.kind in 'biufc'and np.isfinite(x).all()for _,x in arrays(optimizer))
        if new['schema']=='private_affine_coupling_latent_actor_v1':
            m=new['metadata']
            assert m['MLP_blocks']==5 and m['dense_layers']==17 and m['coupling_layers']==4
            assert m['latent_draws']==1 and m['realized_latents']==1 and m['noise_dim']==m['action_dim']
            assert set(new['actor_params']['router'])=={f'coupling_{i}'for i in range(4)}
            assert m['router_hidden_dims']in ([128,128],[256,256])
        else:assert new['metadata']['MLP_blocks']==2
        assert new['metadata']['actor_forward_passes']==1
        assert new['metadata']['generated_candidate_actions']==0
        points={}
        for mode,e in r['evaluations'].items():
            assert sha(e['evaluation_path'])==e['evaluation_sha256']
            evaluated=json.loads(Path(e['evaluation_path']).read_text())
            assert evaluated['n_episodes']==50 and evaluated['first_episode_seed']==42
            assert evaluated['actor_artifact_sha256']==r['private_artifact_sha256']
            assert evaluated['no_optimizer_updates']and not evaluated['counts_toward_native_goal']
            assert all(x in (0,1)for x in evaluated['episode_successes'])and len(evaluated['episode_successes'])==50
            score=sum(evaluated['episode_successes'])/50;assert score==e['success']
            points[mode]=score
        assert set(points)=={'composite','learned_router_exact_Flow'}
        for mode,e in r['reused_controls'].items():
            assert e==parent['evaluations'][mode]and sha(e['evaluation_path'])==e['evaluation_sha256']
        rows.append(dict(case=case,new_evaluations=points,reused_controls={k:v['success']for k,v in r['reused_controls'].items()},
            actual_router_Adam=100000,decoder_updates_performed=0,decoder_bytes_exact_parent=True,
            router_export_bytes_exact=True,full_optimizer_finite=True,
            artifact_schema=new['schema'],result_sha256=sha(path)))
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        source_sha256=sha(__file__),proposal_sha256=sha(a.study/'proposal_v1.json'),
        actual_new_evaluation_episodes=100*len(rows),
        reused_controls_not_counted_as_new_episodes=True,
        scope='Fixed-teacher router ablation; no native joint9task/fresh/final-speed claim.'),f,indent=2,allow_nan=False)
    print(json.dumps(rows))


if __name__=='__main__':main()
