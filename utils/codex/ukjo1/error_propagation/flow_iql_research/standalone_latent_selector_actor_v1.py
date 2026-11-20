"""Native latent-selector actor exports: two MLP blocks, one sampled action.

Only NumPy/JAX/Flax inference dependencies. No OGBench, optimizer, critic,
behavior-flow weights or action candidate bank is needed at deployment.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import latent_selected_actor_v1 as diagonal
from utils.flow_iql_research import orderstat_latent_actor_v1 as order


def definition(metadata):
    assert metadata['MLP_blocks']==2 and metadata['actor_forward_passes']==1
    hidden,d=tuple(metadata['hidden_dims']),metadata['action_dim']
    if metadata['selector_family']=='diagonal':
        return diagonal.LatentSelectedActor(hidden,d,metadata['components'],metadata['std_floor'])
    assert metadata['selector_family']=='orderstat'
    return order.OrderStatisticSelectedActor(hidden,d,metadata['directions'],metadata['best_of'])


def make_policy(metadata,backend='cpu'):
    assert metadata['noise_dim']==metadata['action_dim']+1
    assert metadata['generated_candidate_actions']==metadata['flow_integration_steps']==0
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']
    actor=definition(metadata)
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        noise=jax.random.normal(action_key,(*observations.shape[:-1],metadata['noise_dim']))
        return actor.apply({'params':params},observations,noise)
    return jax.jit(policy,backend=backend)


@dataclass
class LoadedActor:
    params: object
    metadata: dict
    policy: object
    device: object

    def __call__(self,observation,key):
        return self.policy(self.params,jax.device_put(np.asarray(observation,np.float32),self.device),jax.device_put(key,self.device))


def load_actor(path,backend='cpu'):
    with Path(path).open('rb')as f:artifact=pickle.load(f)
    assert artifact['schema']=='single_actor_deployment_v1'
    assert set(artifact['actor_params'])=={'router','decoder'}
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(artifact['actor_params']))
    device=jax.devices(backend)[0];metadata=artifact['metadata']
    return LoadedActor(jax.device_put(artifact['actor_params'],device),metadata,make_policy(metadata,backend),device)
