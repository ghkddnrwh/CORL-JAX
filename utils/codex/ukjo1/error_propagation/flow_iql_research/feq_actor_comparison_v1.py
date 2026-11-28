"""Deployment actors over the frozen gamma=.995, N9 FEQ learner.

FQL: same-noise BC-flow distillation + online mean-Q maximization.
CVAE: posterior reconstruction + .5 KL + TD3+BC-normalized prior-action Q1.
The CVAE combination is an adaptation, not an algorithm in the TD3+BC paper.
All actors update once per Q update and never supply bootstrap actions.
"""
from pathlib import Path

# Load the existing runtime compatibility shim before importing Optax.
from utils.codex.ukjo1.error_propagation.flow_iql_research import discount_0995_n9_core_v1 as reference
from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as budget
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.codex.ukjo1.error_propagation.flow_iql_research.vendor.flax_utils import TrainState
from utils.codex.ukjo1.error_propagation.flow_iql_research.vendor.networks import ActorVectorField, MLP

METHODS = ('fql', 'cvae')
config_for = reference.config_for
return_bounds = reference.return_bounds
make_scan = reference.make_scan


class CVAE(nn.Module):
    hidden_dims: tuple
    action_dim: int
    latent_dim: int

    def setup(self):
        self.encoder = MLP((*self.hidden_dims[:2], 2 * self.latent_dim))
        self.decoder = MLP((*self.hidden_dims, self.action_dim))

    def decode(self, observations, latent):
        return jnp.tanh(self.decoder(jnp.concatenate((observations, latent), axis=-1)))

    def __call__(self, observations, actions, posterior_noise, prior_noise):
        encoded = self.encoder(jnp.concatenate((observations, actions), axis=-1))
        mean, log_std = jnp.split(encoded, 2, axis=-1)
        log_std = jnp.clip(log_std, -4., 15.)
        latent = mean + jnp.exp(log_std) * posterior_noise
        return self.decode(observations, latent), mean, log_std, self.decode(observations, prior_noise)


@flax.struct.dataclass
class State:
    base: object
    actors: object
    actor_updates: object


def actor_metadata(cfg, observation_dim, action_dim, method, alpha):
    assert method in METHODS and alpha >= 0
    return dict(schema='feq_actor_comparison_v1', method=method, alpha=float(alpha),
                observation_dim=int(observation_dim), action_dim=int(action_dim),
                hidden_dims=list(cfg['actor_hidden_dims']), layer_norm=False,
                noise_dim=int(action_dim) * (2 if method == 'cvae' else 1),
                kl_coefficient=.5 if method == 'cvae' else 0.,
                q_aggregation='first' if method == 'cvae' else 'mean',
                q_normalization='detached_mean_abs_Q1_floor_1e-6' if method == 'cvae' else 'none',
                alpha_role='Q_term' if method == 'cvae' else 'distillation_MSE',
                latent_prior='standard_normal_untruncated',
                actor_updates_per_Q_update=1, actor_forward_passes=1,
                requires_critic=False, requires_BC_flow=False, deployment_candidates=1)


def definition(meta):
    if meta['method'] == 'fql':
        return ActorVectorField(tuple(meta['hidden_dims']), meta['action_dim'], False)
    return CVAE(tuple(meta['hidden_dims']), meta['action_dim'], meta['noise_dim'])


def create(seed, observations, actions, cfg, specifications):
    base = reference.lean.create(seed, observations, actions, cfg, 'lean')
    actors = {}
    for name, spec in specifications.items():
        meta = actor_metadata(cfg, observations.shape[-1], actions.shape[-1], **spec)
        model = definition(meta)
        # Same initialization and random inputs for every alpha of a method.
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 120801 if spec['method'] == 'fql' else 120802)
        noise = jnp.zeros((*observations.shape[:-1], meta['noise_dim']))
        inputs = (observations, noise) if spec['method'] == 'fql' else (observations, actions, noise, noise)
        params = model.init(key, *inputs)['params']
        actors[name] = TrainState.create(model, params, optax.adam(cfg['lr']))
    return State(base, actors, jnp.asarray(0))


def fql_loss(actor, params, critic, observations, noise, teacher, alpha):
    raw = actor(observations, noise, params=params)
    mse = jnp.mean((raw - jax.lax.stop_gradient(teacher)) ** 2)
    q = critic(observations, actions=jnp.clip(raw, -1, 1)).mean(0)
    loss = alpha * mse - q.mean()
    return loss, dict(loss=loss, distill_mse=mse, q_mean=q.mean(),
                      raw_saturation=jnp.mean((jnp.abs(raw) > 1).astype(jnp.float32)))


def cvae_loss(actor, params, critic, batch, posterior_noise, prior_noise, alpha):
    reconstructed, mean, log_std, prior_action = actor(
        batch['observations'], batch['actions'], posterior_noise, prior_noise, params=params)
    reconstruction = jnp.mean((reconstructed - batch['actions']) ** 2)
    kl = .5 * jnp.mean(mean ** 2 + jnp.exp(2 * log_std) - 1 - 2 * log_std)
    # Q guidance trains the same prior decoder that is deployed. The encoder
    # is trained only by reconstruction and KL, not by a privileged action label.
    q = critic(batch['observations'], actions=prior_action)[0]
    normalizer = jax.lax.stop_gradient(jnp.maximum(jnp.mean(jnp.abs(q)), 1e-6))
    q_loss = -alpha * q.mean() / normalizer
    loss = reconstruction + .5 * kl + q_loss
    return loss, dict(loss=loss, reconstruction_mse=reconstruction, kl=kl,
                      q_loss=q_loss, q_mean=q.mean(), q_normalizer=normalizer,
                      posterior_std_mean=jnp.exp(log_std).mean())


def build_update(specifications, bounds):
    update_base = budget.base_update('n9', bounds)
    def update(state, batch):
        base, logs = update_base(state.base, batch)
        agent = state.base.agent  # all actor losses use pre-update Q and Flow
        _, key = jax.random.split(agent.rng)
        critic = agent.network.select('critic')
        noises = {}
        if any(s['method'] == 'fql' for s in specifications.values()):
            z = jax.random.normal(jax.random.fold_in(key, 120811), batch['actions'].shape)
            teacher = reference.lean.flow(agent, batch['observations'], z, 10)
            noises['fql'] = (z, teacher)
        if any(s['method'] == 'cvae' for s in specifications.values()):
            shape = (*batch['actions'].shape[:-1], 2 * batch['actions'].shape[-1])
            noises['cvae'] = (jax.random.normal(jax.random.fold_in(key, 120812), shape),
                              jax.random.normal(jax.random.fold_in(key, 120813), shape))
        actors = {}
        for name, spec in specifications.items():
            actor = state.actors[name]
            if spec['method'] == 'fql':
                loss = lambda p: fql_loss(actor, p, critic, batch['observations'], *noises['fql'], spec['alpha'])
            else:
                loss = lambda p: cvae_loss(actor, p, critic, batch, *noises['cvae'], spec['alpha'])
            actors[name], info = actor.apply_loss_fn(loss)
            logs.update({f'deployment/{name}/{k}': v for k, v in info.items()})
        return state.replace(base=base, actors=actors, actor_updates=state.actor_updates+1), logs
    return update


def export_actor(state, name, meta):
    params = state.actors[name].params
    if meta['method'] == 'cvae':
        # Deployment needs no encoder and no learned Q/Flow parameters.
        params = {'decoder': params['decoder']}
    return dict(metadata=meta, actor_params=flax.serialization.to_state_dict(jax.device_get(params)),
                actor_updates=int(state.actor_updates), Q_updates=int(state.base.q_updates))


def make_actor_policy(meta, backend='cpu'):
    model = definition(meta)
    def policy(params, observations, key):
        key, _ = jax.random.split(key)
        z = jax.random.normal(key, (*observations.shape[:-1], meta['noise_dim']))
        if meta['method'] == 'fql':
            return jnp.clip(model.apply({'params': params}, observations, z), -1, 1)
        return model.apply({'params': params}, observations, z, method=model.decode)
    return jax.jit(policy, backend=backend)


def source_paths():
    return list(dict.fromkeys([Path(__file__).resolve(), *reference.source_paths()]))
