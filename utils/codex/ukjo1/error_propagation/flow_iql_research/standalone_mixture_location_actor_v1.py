"""One actor: route by learned probabilities, evaluate one location head.

Deployment imports NumPy/JAX only. No optimizer, learner, Flax, OGBench,
flow integration, generated-action bank, or critic is needed. The neural
weights are exactly the native likelihood-trained MDN weights; this artifact
records the deployment rule separately from the unchanged training identity.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import jax
import numpy as np
from utils.flow_iql_research.lazy_mixture_location_actor_v1 import decode


def make_policy(metadata,backend='cpu'):
    assert metadata['architecture']=='routed_component_location_MLP'
    assert metadata['deployment_decoder']=='sampled_component_location'
    assert metadata['actor_forward_passes']==metadata['computed_action_heads']==1
    assert metadata['generated_candidate_actions']==metadata['flow_integration_steps']==0
    assert not metadata['requires_critic'] and not metadata['requires_BC_flow']
    return jax.jit(lambda params,obs,key:decode(
        params,obs,key,metadata,'sampled_component_location'),backend=backend)


@dataclass
class LoadedActor:
    params:object
    metadata:dict
    policy:object
    device:object

    def __call__(self,observations,key):
        return self.policy(self.params,
            jax.device_put(np.asarray(observations,np.float32),self.device),
            jax.device_put(key,self.device))


def load_actor(path,backend='cpu'):
    with Path(path).open('rb') as stream:
        artifact=pickle.load(stream)
    assert artifact['schema']=='routed_mixture_location_actor_v1'
    assert artifact['no_new_training_updates']
    assert artifact['training_identity']['recipe']=='mdn9_bc'
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(artifact['actor_params']))
    metadata=artifact['metadata'];device=jax.devices(backend)[0]
    return LoadedActor(jax.device_put(artifact['actor_params'],device),metadata,
                       make_policy(metadata,backend),device)
