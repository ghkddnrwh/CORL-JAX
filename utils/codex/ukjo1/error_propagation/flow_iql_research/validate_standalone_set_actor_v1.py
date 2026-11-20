"""Actor-only client isolation and network structure, not return performance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import standalone_set_onestep_actor_v1 as standalone


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert all(device.platform=='cpu' for device in jax.devices())
    loaded=standalone.load_actor(args.actor,'cpu');meta=loaded.metadata
    observation=np.linspace(-.1,.1,meta['observation_dim'],dtype=np.float32)
    batch=np.broadcast_to(observation,(16,len(observation)))
    action=np.asarray(loaded(observation,jax.random.PRNGKey(42)))
    actions=np.asarray(loaded(batch,jax.random.PRNGKey(43)))
    assert action.shape==(meta['action_dim'],) and actions.shape==(16,meta['action_dim'])
    assert np.isfinite(action).all() and np.isfinite(actions).all()
    assert np.abs(action).max()<=1 and np.abs(actions).max()<=1
    definition=standalone._SetActor(meta['action_dim'],meta['feature_width'],meta['decoder_width'],meta['layer_norm'])
    graph=str(jax.make_jaxpr(lambda params,obs,noise:definition.apply({'params':params},obs,noise))(
        loaded.params,jnp.asarray(observation),jnp.zeros(meta['noise_dim'])))
    assert graph.count('dot_general[')==6,graph
    assert 'argmax' not in graph and 'while[' not in graph and 'scan[' not in graph
    forbidden=[name for name in sys.modules if name.startswith('ogbench')
               or name.startswith('utils.flow_iql_research.vendor') or 'onestep_deploy_core' in name]
    assert not forbidden,forbidden
    result=dict(time=time.time(),status='pass',action=action.tolist(),batch_actions=actions.tolist(),
        actor_artifact_sha256=hashlib.sha256(args.actor.read_bytes()).hexdigest(),
        portable_source_sha256=hashlib.sha256(Path(standalone.__file__).read_bytes()).hexdigest(),
        learned_model_trees_loaded=['actor_params'],learner_or_environment_modules_loaded=forbidden,
        network_only_dense_matmuls=6,network_only_graph=graph,
        no_flow_integration=True,no_candidate_actions=True,no_critic=True,
        no_performance_claim=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
