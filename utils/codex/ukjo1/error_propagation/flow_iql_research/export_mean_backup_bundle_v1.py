"""Export one fixed1M mean-backup model, without training state or extra fitting."""
import argparse
import json
from pathlib import Path
import pickle
import time
from unittest import mock
from utils.flow_iql_research import core
from utils.flow_iql_research.mean_backup_bundle_v1 import load,sha,frameworks,inference_sources,RECIPE
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import jax
import numpy as np
import ogbench


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();assert not args.output.exists()
    conf=json.loads((run/'config.json').read_text());identity=conf['identity'];verifier=Verifier()
    assert identity['kind']=='ifql'and identity['profile']=='bounded_v2_mean_empirical_backup'
    assert (run/'completed.json').exists()and not(run/'stopped.json').exists()
    done=json.loads((run/'completed.json').read_text());assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
    for source,digest in conf['source_sha256'].items():assert verifier.digest(source)==digest,source
    checkpoint=run/'checkpoint_1000000.pkl';digest=verifier.checkpoint(checkpoint,conf,1000000)
    with checkpoint.open('rb')as stream:saved=pickle.load(stream)
    network=saved['state']['agent']['network']['params'];parameters={key:network[key]for key in('modules_actor_flow','modules_critic')}
    assert all(np.isfinite(np.asarray(a)).all()for a in jax.tree_util.tree_leaves(parameters))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True)
    observation_dim,action_dim=env.observation_space.shape[-1],env.action_space.shape[-1];env.close()
    bundle=dict(kind='mean_finite_expectile_IFQL_inference_v1',time=time.time(),resumable_training_checkpoint=False,step=1000000,
        environment=identity['environment'],training_seed=identity['seed'],identity=identity,recipe=RECIPE,
        observation_dim=observation_dim,action_dim=action_dim,network_params=parameters,
        inference_source_sha256=inference_sources(),framework_versions=frameworks(),
        provenance=dict(run=run.name,checkpoint=str(checkpoint),checkpoint_sha256=digest,config_sha256=verifier.digest(run/'config.json'),
            training_source_sha256=conf['source_sha256'],actual_source_Q_Adam=1000000,actual_source_V_Adam=2000000),
        note='One fixed1M deployment export. Source paths are provenance only. Not a new training replicate, checkpoint selection, or replacement of the last3 metric.')
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.tmp')
    with temporary.open('wb')as stream:pickle.dump(bundle,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    original=Path.open;blocked=[]
    def guarded(path,*positional,**keywords):
        if '/runs/'in str(path)or'/checkpoint_archive/'in str(path):blocked.append(str(path));raise AssertionError('Loader accessed source model')
        return original(path,*positional,**keywords)
    with mock.patch.object(Path,'open',guarded):_,state,_=load(args.output)
    assert not blocked
    for key in parameters:
        a,ta=jax.tree_util.tree_flatten(parameters[key]);b,tb=jax.tree_util.tree_flatten(state.agent.network.params[key])
        assert ta==tb and all(np.array_equal(np.asarray(x),np.asarray(y))for x,y in zip(a,b))
    assert state.agent.network.opt_state is None and set(state.agent.network.params)==set(parameters)
    proof=dict(time=time.time(),status='export_verified',bundle=str(args.output),bundle_sha256=sha(args.output),bundle_bytes=args.output.stat().st_size,
        source_checkpoint_sha256=digest,source_parameters_after_load_exact=True,no_source_directory_reads_during_load=True,
        omitted=['V','target_critic','Adam','replay','training_rng'],source_sha256=sha(__file__),goal_complete=False)
    sidecar=args.output.with_suffix('.export.json');assert not sidecar.exists();sidecar.write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
