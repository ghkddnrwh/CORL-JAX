"""Verify real bank_bc exports in a fresh actor-only process.

No learner, OGBench, optimizer, critic or behavior-flow module is imported.
This proves export/inference structure, not policy quality or isolated speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

from utils.flow_iql_research import standalone_onestep_actor_v1 as portable

import jax
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and all(d.platform=='cpu'for d in jax.devices())
    manifest=json.loads(args.manifest.read_text())
    assert sha(__file__)==manifest['source_sha256']
    assert sha(portable.__file__)==manifest['portable_client_sha256']
    args.output.mkdir(parents=True)
    forbidden=['ogbench','optax','utils.flow_iql_research.onestep_deploy_core_v1',
               'utils.flow_iql_research.efficient_mean_core_v1',
               'utils.flow_iql_research.mean_empirical_backup_core_v1']
    rows=[]
    for item in manifest['cases']:
        artifact=Path(item['artifact']);checkpoint=Path(item['checkpoint'])
        assert sha(artifact)==item['artifact_sha256'] and sha(checkpoint)==item['checkpoint_sha256']
        with artifact.open('rb')as f:export=pickle.load(f)
        with checkpoint.open('rb')as f:saved=pickle.load(f)
        assert export['checkpoint_sha256']==sha(checkpoint)
        actor=portable.load_actor(artifact,backend='cpu');m=actor.metadata
        assert m['input']=='bank'and m['noise_dim']==9*m['action_dim']
        assert m['actor_forward_passes']==1 and m['generated_candidate_actions']==0
        assert m['flow_integration_steps']==0 and not m['requires_critic']and not m['requires_BC_flow']
        params=export['actor_params'];original=saved['state']['actor']['params']
        assert set(params)=={'mlp'}and set(original)=={'mlp'}
        dimensions=[m['observation_dim']+m['noise_dim'],*m['hidden_dims'],m['action_dim']]
        count=0
        assert set(params['mlp'])=={f'Dense_{i}'for i in range(len(dimensions)-1)}
        for i,(before,after)in enumerate(zip(dimensions[:-1],dimensions[1:])):
            layer=params['mlp'][f'Dense_{i}'];reference=original['mlp'][f'Dense_{i}']
            assert set(layer)=={'kernel','bias'}
            assert layer['kernel'].shape==(before,after)and layer['bias'].shape==(after,)
            for key,array in layer.items():
                assert np.isfinite(array).all()and np.array_equal(array,reference[key])
                count+=array.size
        inputs=np.concatenate([np.zeros((1,m['observation_dim']),np.float32),
                              np.random.RandomState(49504).normal(size=(15,m['observation_dim'])).astype(np.float32)])
        actions=[]
        for i,x in enumerate(inputs):
            key=jax.random.PRNGKey(42+i);one=np.asarray(actor(x,key));two=np.asarray(actor(x,key))
            assert one.shape==(m['action_dim'],)and np.array_equal(one,two)
            assert np.isfinite(one).all()and np.max(np.abs(one))<=1
            actions.append(one)
        computation=jax.xla_computation(lambda p,x,k:actor.policy(p,x,k))(
            actor.params,inputs[0],jax.random.PRNGKey(42))
        hlo=computation.as_hlo_text()
        dots=len(re.findall(r'= [^\n]*\bdot\(',hlo))
        assert dots==len(m['hidden_dims'])+1,(dots,m['hidden_dims'])
        graph=args.output/(item['name']+'_CPU_HLO_v1.txt');graph.write_text(hlo)
        assert not any(name in sys.modules for name in forbidden)
        rows.append(dict(name=item['name'],step=saved['step'],metadata=m,parameter_count=count,
            exact_checkpoint_actor_parameters=True,MLP_matrix_multiplications=dots,
            HLO_sha256=sha(graph),artifact_sha256=sha(artifact),checkpoint_sha256=sha(checkpoint),
            input_fixtures=16,same_key_actions_identical=True,finite_bounded_actions=True,
            fresh_actor_only_imports=True,forbidden_modules_absent=forbidden))
    result=dict(time=time.time(),status='pass',manifest_sha256=sha(args.manifest),rows=rows,
        source_sha256=sha(__file__),portable_client_sha256=sha(portable.__file__),
        no_optimizer_updates=True,no_environment_rollouts=True,
        scope='Actual export integrity and one-MLP actor-only CPU graph, across three real models. '
              'Does not prove nine-task quality, GPU latency or final success.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
