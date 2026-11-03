"""Permutation-invariant one-step actor candidate; original learner unchanged.

The teacher's best9 map is permutation-invariant except for exact Q ties.
Shared latent features and soft attention encode that symmetry. Feature width
is distinct from action dimension: no candidate actions or critic are generated
at deployment. This is an experimental candidate, not a validated result.
"""
from pathlib import Path
from utils.flow_iql_research import onestep_deploy_core_v1 as base
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.vendor.networks import MLP
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


ENVIRONMENTS=base.ENVIRONMENTS
RECIPES=('set_bc','set_q_a03','set_q_a1','set_q_a3','set_value_a1','set_regret_a1')
State=base.State


class SetActor(nn.Module):
    action_dim: int
    feature_width: int
    decoder_width: int
    layer_norm: bool=False

    def setup(self):
        self.features=MLP((self.feature_width,self.feature_width),activate_final=True,layer_norm=self.layer_norm)
        self.attention=nn.Dense(1)
        self.decoder=MLP((self.decoder_width,self.decoder_width,self.action_dim),layer_norm=self.layer_norm)

    def __call__(self,observations,packed_noises):
        noise=packed_noises.reshape((*observations.shape[:-1],9,self.action_dim))
        obs=jnp.broadcast_to(observations[...,None,:],(*noise.shape[:-1],observations.shape[-1]))
        features=self.features(jnp.concatenate([obs,noise],axis=-1))
        logits=self.attention(features)[...,0]
        weights=jax.nn.softmax(logits,axis=-1)
        pooled=(features*weights[...,None]).sum(axis=-2)
        maximum=features.max(axis=-2)
        return self.decoder(jnp.concatenate([observations,pooled,maximum],axis=-1))


def config_for(environment,recipe,width=None):
    assert recipe.startswith('set_')
    cfg=base.config_for(environment,'bank_'+recipe[4:],width)
    spec=cfg['onestep_specification']
    spec.update(name=recipe,architecture='permutation_invariant_attention',feature_depth=2,decoder_depth=2)
    return cfg


def actor_metadata(cfg,observation_dim,action_dim):
    spec=cfg['onestep_specification'];width=int(cfg['actor_hidden_dims'][0])*spec['actor_width_multiplier']
    return dict(schema='set_actor_deployment_v1',architecture='permutation_invariant_attention',
        observation_dim=int(observation_dim),action_dim=int(action_dim),feature_width=max(width//2,4),decoder_width=width,
        layer_norm=spec['actor_layer_norm'],input='bank',noise_dim=9*int(action_dim),latent_vectors=9,
        flow_integration_steps=0,generated_candidate_actions=0,requires_critic=False,requires_BC_flow=False,actor_forward_passes=1)


def actor_definition(metadata):
    return SetActor(action_dim=metadata['action_dim'],feature_width=metadata['feature_width'],decoder_width=metadata['decoder_width'],layer_norm=metadata['layer_norm'])


def create(seed,observations,actions,cfg,recipe):
    assert cfg['onestep_specification']['name']==recipe
    lean=base.lean.create(seed,observations,actions,cfg,'lean')
    metadata=actor_metadata(cfg,observations.shape[-1],actions.shape[-1]);definition=actor_definition(metadata)
    params=definition.init(jax.random.fold_in(jax.random.PRNGKey(seed),99231),observations,jnp.zeros((*observations.shape[:-1],metadata['noise_dim'])))['params']
    actor=TrainState.create(definition,params,optax.adam(cfg['lr']*cfg['onestep_specification']['actor_lr_multiplier']))
    return State(lean,actor,jnp.asarray(0))


# Identical base updates, teacher samples, losses, optimizer frequency and probes.
build_update=base.build_update
make_scan=base.make_scan
make_probe=base.make_probe
probe_summary=base.probe_summary
make_teacher_policy=base.make_teacher_policy


def make_actor_policy(metadata,backend='cpu'):
    assert metadata['schema']=='set_actor_deployment_v1'and metadata['actor_forward_passes']==1
    assert not metadata['requires_critic']and not metadata['requires_BC_flow']and metadata['generated_candidate_actions']==0
    definition=actor_definition(metadata)
    def policy(params,observations,key):
        action_key,_=jax.random.split(key)
        noise=jax.random.normal(action_key,(9,*observations.shape[:-1],metadata['action_dim']))
        packed=base.packed_input(noise,metadata)
        return jnp.clip(definition.apply({'params':params},observations,packed),-1,1)
    return jax.jit(policy,backend=backend)


def export_actor(state,metadata):
    exported=base.export_actor(state,metadata);exported['schema']='set_actor_deployment_v1'
    return exported


def source_paths():return [Path(__file__).resolve()]+base.source_paths()
