"""Amortize IFQL best-of-nine behavior selection with a second policy flow.

The BC teacher and IQL V2 critic/value learning remain separate from the policy
flow. Choose the highest old min-Q among nine old BC-flow actions, then flow-
match that selected-action distribution using independent Gaussian/time draws.
Deployment samples the policy flow directly with ten Euler steps.
"""
import copy
from utils.flow_iql_research import policy_core as reference
from utils.flow_iql_research.policy_core import ENVIRONMENTS, State, make_scan, probe, probe_summary
from utils.flow_iql_research.vendor.flax_utils import ModuleDict, TrainState
from utils.flow_iql_research.vendor.networks import ActorVectorField
import flax
import jax
import jax.numpy as jnp
import optax


PROFILES = {'bounded_v2_rank_flow': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    cfg = reference.config_for(environment, discount, expectile, width, alpha)
    cfg.update(agent_name='rank_flow', policy_type='rank_flow', num_samples=1,
               selector_teacher_samples=9, policy_objective='FM_of_best9_BC_action_distribution')
    return cfg


def select_best(actions, values):
    """Axis zero contains proposals; preserve every leading observation axis."""
    indices = jnp.argmax(values, axis=0)
    broadcast = jnp.broadcast_to(indices[None,...,None], (1, *actions.shape[1:]))
    return jnp.take_along_axis(actions, broadcast, axis=0)[0]


class RankFlowAgent(reference.FlowIQLAgent):
    def selected_teacher_targets(self, observations, key):
        n = self.config['selector_teacher_samples']
        repeated = jnp.broadcast_to(observations, (n, *observations.shape))
        noises = jax.random.normal(key, (n, *observations.shape[:-1], self.config['action_dim']))
        actions = self.compute_flow_actions(repeated, noises)
        values = self.network.select('critic')(repeated, actions=actions).min(0)
        return jax.lax.stop_gradient(select_best(actions, values)), values, actions

    def actor_loss(self, batch, grad_params, rng=None):
        net, cfg = self.network, self.config
        shape = batch['actions'].shape
        # Preserve the official IFQL teacher FM draws exactly. Extra policy
        # labels/noises use folded-in keys without changing the replay stream.
        _, xkey, tkey = jax.random.split(rng, 3)
        x0 = jax.random.normal(xkey, shape)
        t = jax.random.uniform(tkey, (*shape[:-1], 1))
        xt = (1-t)*x0+t*batch['actions']
        pred = net.select('actor_flow')(batch['observations'], xt, t, params=grad_params)
        bc_loss = jnp.mean((pred-(batch['actions']-x0))**2)
        selected, qs, candidates = self.selected_teacher_targets(
            batch['observations'], jax.random.fold_in(rng, 71927))
        p0 = jax.random.normal(jax.random.fold_in(rng, 71928), shape)
        pt = jax.random.uniform(jax.random.fold_in(rng, 71929), (*shape[:-1], 1))
        pxt = (1-pt)*p0+pt*selected
        prediction = net.select('actor_policy_flow')(batch['observations'], pxt, pt, params=grad_params)
        policy_loss = jnp.mean((prediction-(selected-p0))**2)
        ordered = jnp.sort(qs, axis=0)
        margin = ordered[-1]-ordered[-2]
        loss = bc_loss+policy_loss
        return loss, dict(actor_loss=loss, bc_flow_loss=bc_loss, policy_flow_loss=policy_loss,
            selected_teacher_Q_gain_over_mean=jnp.mean(qs.max(0)-qs.mean(0)),
            teacher_winner_margin_mean=margin.mean(),
            teacher_numeric_tie_fraction=jnp.mean((margin <= 1e-6).astype(jnp.float32)),
            teacher_candidate_Q_std_mean=jnp.std(qs, axis=0).mean(),
            teacher_candidate_action_std_mean=jnp.std(candidates, axis=0).mean(),
            selected_teacher_action_saturation=jnp.mean((jnp.abs(selected) >= .98).astype(jnp.float32)),
            policy_target_velocity_rms=jnp.sqrt(jnp.mean((selected-p0)**2)))

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.):
        assert self.config['num_samples'] == 1
        key, _ = jax.random.split(seed)
        action = jax.random.normal(key, (*observations.shape[:-1], self.config['action_dim']))
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i/self.config['flow_steps'])
            action = action+self.network.select('actor_policy_flow')(observations, action, t)/self.config['flow_steps']
        return jnp.clip(action, -1, 1)


def create(seed, observations, actions, config, kind='rank_flow'):
    assert kind == 'rank_flow' and config['num_samples'] == 1
    cfg = dict(config)
    cfg.update(policy_type='rank_flow', agent_name='rank_flow', action_dim=actions.shape[-1])
    # Reuse the official IFQL initial Q/V/BC parameters and RNG exactly.
    base = reference.create(seed, observations, actions, cfg, 'ifql')
    modules = dict(base.agent.network.model_def.modules)
    modules['actor_policy_flow'] = ActorVectorField(cfg['actor_hidden_dims'], cfg['action_dim'],
                                                   layer_norm=cfg['actor_layer_norm'])
    definition = ModuleDict(modules)
    inputs = dict(value=(observations,), critic=(observations, actions),
                  target_critic=(observations, actions),
                  actor_flow=(observations, actions, actions[...,:1]),
                  actor_policy_flow=(observations, actions, actions[...,:1]))
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 71930)
    initialized = flax.core.unfreeze(definition.init(key, **inputs)['params'])
    initialized.update(copy.deepcopy(base.agent.network.params))
    # A policy-improvement flow starts from the same behavior-flow parameters.
    initialized['modules_actor_policy_flow'] = copy.deepcopy(initialized['modules_actor_flow'])
    network = TrainState.create(definition, initialized, tx=optax.adam(cfg['lr']))
    agent = RankFlowAgent(base.agent.rng, network=network, config=flax.core.FrozenDict(**cfg))
    return State(agent, base.v_opt, jnp.asarray(0), jnp.asarray(0))


def build_update(profile_name, bounds, kind='rank_flow'):
    assert profile_name == 'bounded_v2_rank_flow' and kind == 'rank_flow'
    return reference.build_update('bounded_v2', bounds, kind)
