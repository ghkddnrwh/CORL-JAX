"""Aggregate real-shape implementation checks; no policy-quality qualification."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.watch_onestep_deploy_v8 import arrays
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    area=a.root/'analysis/latent_selector_joint_pipeline_v1';area.mkdir(exist_ok=True)
    target=area/'readiness_result_v1.json';assert not target.exists()
    rows=[];evidence=[];sources={str(Path(__file__).resolve()):sha(__file__)};training={};base_by_env={};comparisons=[]
    studies=[('latent','latent_selector_joint_actual_shapes_v1',8),
             ('latent_guided','latent_order_guided_actual_shapes_v1',5)]
    for tag,directory,count in studies:
        folder=a.root/'analysis'/directory;planpath=folder/'source_plan_v1.json'
        plan=json.loads(planpath.read_text());training[tag]=plan['training_source_sha256']
        for path,digest in plan['source_sha256'].items():
            assert sha(path)==digest,path
            assert path not in sources or sources[path]==digest
            sources[path]=digest
        evidence.append(dict(path=str(planpath),sha256=sha(planpath)))
        paths=sorted(folder.glob('*/*/result.json'));assert len(paths)==count,(directory,len(paths))
        for path in paths:
            r=json.loads(path.read_text());assert r['status']=='pass' and r['no_performance_claim']
            assert r['source_plan_sha256']==sha(planpath) and r['Q_Adam']==r['actor_Adam']==8 and r['V_Adam']==0
            assert r['full8_resume4plus4_all_saved_state_bitwise_equal'] and r['portable_actor_only_client_isolation_passed']
            run=Path(r['fixture_actor']).parent
            with(run/'checkpoint_8.pkl').open('rb')as f:full=pickle.load(f)
            with(run/'actor_8.pkl').open('rb')as f:actor=pickle.load(f)
            state=full['state'];base=state['base']
            if r['environment']in base_by_env:
                reference=base_by_env[r['environment']]
                first,second=dict(arrays(reference)),dict(arrays(base))
                assert first.keys()==second.keys()
                assert all(v.dtype==second[k].dtype and v.shape==second[k].shape and v.tobytes()==second[k].tobytes()for k,v in first.items())
                comparisons.append(dict(environment=r['environment'],recipe=r['recipe'],
                    base_array_leaves=len(first),all_base_arrays_dtype_shape_and_bytes_exact=True))
            else:base_by_env[r['environment']]=base
            assert int(base['q_updates'])==int(base['agent']['network']['opt_state']['0']['count'])==8
            assert int(state['actor_updates'])==int(state['actor']['opt_state']['0']['count'])==8
            assert int(base['v_updates'])==int(base['v_opt']['0']['count'])==0
            left,right=dict(arrays(state['actor']['params'])),dict(arrays(actor['actor_params']))
            assert left.keys()==right.keys() and all(v.dtype==right[k].dtype and v.shape==right[k].shape and v.tobytes()==right[k].tobytes()for k,v in left.items())
            assert all(np.isfinite(v).all() for _,v in arrays(state))
            conf=json.loads((run/'config.json').read_text());assert conf['source_sha256']==training[tag]
            assert sha(run/'checkpoint_8.pkl')==actor['checkpoint_sha256'] and sha(run/'actor_8.pkl')==r['fixture_actor_sha256']
            for e in r['evaluator_implementation_only']:
                assert sha(e['path'])==e['sha256']
                value=json.loads(Path(e['path']).read_text());assert value['no_performance_claim'] and value['no_optimizer_updates']
                assert value['checkpoint_sha256']==actor['checkpoint_sha256']
            rows.append(dict(implementation=tag,recipe=r['recipe'],environment=r['environment'],
                actual_hidden_width=r['actual_hidden_width'],run=str(run),status='pass',
                actual_Q_Adam=8,actual_actor_Adam=8,V_Adam=0,actor_export_bytes_exact=True,
                full_saved_state_finite=True,result_sha256=sha(path)))
            evidence.append(dict(path=str(path),sha256=sha(path)))
    for name in ('latent_selector_joint_synthetic_v1','latent_order_guided_joint_synthetic_v1'):
        path=a.root/'analysis'/name/'result.json';r=json.loads(path.read_text());assert r['status']=='pass'
        for source,digest in r['source_sha256'].items():assert sha(source)==digest
        sources.update(r['source_sha256']);evidence.append(dict(path=str(path),sha256=sha(path)))
    result=dict(time=time.time(),status='pass',rows=rows,evidence=evidence,source_sha256=sources,
        cross_architecture_base_comparisons=comparisons,
        implementation_training_source_sha256=training,no_performance_claim=True,
        scope='CPU real-dataset/default-width8updates/resume/API/portable integration and synthetic gradient isolation; no native1M efficacy qualification.')
    with target.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',rows=len(rows),output=str(target))))


if __name__=='__main__':main()
