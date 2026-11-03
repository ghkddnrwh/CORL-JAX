"""Official FQL-repository IQL, with separately switchable value corrections.

The unmodified baseline calls the vendored official update verbatim. Target EMA
in that implementation uses the *old* critic, and actor advantages use old online
Q and old V. Corrections preserve those conventions. Extra V steps own a separate
Adam clock; Q/actor retain the official joint Adam clock and constant learning rate.
"""
import numpy as np
import scipy.linalg

if not hasattr(scipy.linalg, 'tril'):
    scipy.linalg.tril = np.tril
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

import flax
import jax
import jax.numpy as jnp
import optax

from utils.flow_iql_research.vendor.iql import IQLAgent, get_config
from utils.propagation_research.reward_bounds import return_bounds


ENVIRONMENTS = (
    'antmaze-large-navigate-singletask-v0',
    'humanoidmaze-medium-navigate-singletask-v0',
    'humanoidmaze-large-navigate-singletask-v0',
    'antsoccer-arena-navigate-singletask-v0',
    'cube-single-play-singletask-v0',
    'cube-double-play-singletask-v0',
    'puzzle-3x3-play-singletask-v0',
    'puzzle-4x4-play-singletask-v0',
    'scene-play-singletask-v0',
)

PROFILES = {
    'official': {},
    'bound': {'bound': True},
    'v2': {'v_steps': 2, 'fresh': True},
    'bounded_v2': {'v_steps': 2, 'fresh': True, 'bound': True},
    'fresh': {'fresh': True},
    'bounded_v2_old': {'v_steps': 2, 'bound': True},
}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    """Paper Table5/6 and AppendixE.2, not bare get_config() defaults."""
    config = get_config()
    manipulation = any(s in environment for s in ('cube-', 'puzzle-', 'scene-'))
    w = width or (256 if manipulation else 512)
    a = .3 if 'cube-double' in environment else 1. if any(s in environment for s in ('cube-single', 'antsoccer')) else 10.
    g = .995 if any(s in environment for s in ('humanoidmaze', 'antsoccer')) else .99
    config.update(dict(actor_hidden_dims=(w,)*4, value_hidden_dims=(w,)*4,
                       discount=g if discount is None else discount,
                       expectile=expectile, alpha=a if alpha is None else alpha,
                       encoder=None))
    return config.to_dict()


@flax.struct.dataclass
class State:
    agent: object
    v_opt: object
    v_updates: object
    q_updates: object


def create(seed, observations, actions, config, kind='iql'):
    if kind == 'iql':
        cls = IQLAgent
    elif kind == 'ifql':
        from utils.flow_iql_research.vendor.ifql import IFQLAgent
        cls = IFQLAgent
    elif kind == 'fql':
        from utils.flow_iql_research.vendor.fql import FQLAgent
        cls = FQLAgent
    else:
        raise ValueError(kind)
    agent = cls.create(seed, observations, actions, dict(config))
    v_params = agent.network.params.get('modules_value', {})
    return State(agent, optax.adam(config['lr']).init(v_params),
                 jnp.asarray(0), jnp.asarray(0))


def build_update(profile_name, bounds, kind='iql'):
    profile = PROFILES[profile_name]
    lower, upper = bounds
    if not profile:
        def update(state, batch):
            agent, info = state.agent.update(batch)
            return state.replace(agent=agent, v_updates=state.v_updates+int(kind != 'fql'),
                                 q_updates=state.q_updates+1), info
        return update

    def update(state, batch):
        agent = state.agent
        net = agent.network
        cfg = agent.config
        new_rng, rng = jax.random.split(agent.rng)
        tx = optax.adam(cfg['lr'])
        # Frozen same-minibatch target Q for all V steps.
        target_q = net.select('target_critic')(batch['observations'], actions=batch['actions']).min(0)
        def value_loss(params):
            values = net.select('value')(batch['observations'], params={**net.params, 'modules_value': params})
            diff = target_q-values
            return jnp.mean(jnp.where(diff >= 0, cfg['expectile'], 1-cfg['expectile'])*diff**2)
        vp, vo = net.params['modules_value'], state.v_opt
        initial_v_loss = value_loss(vp)
        vsteps = profile.get('v_steps', 1)
        for _ in range(vsteps):
            grads = jax.grad(value_loss)(vp)
            upd, vo = tx.update(grads, vo, vp)
            vp = optax.apply_updates(vp, upd)
        vparams = {**net.params, 'modules_value': vp} if profile.get('fresh') else net.params
        next_v = net.select('value')(batch['next_observations'], params=vparams)
        y = batch['rewards'] + cfg['discount'] * batch['masks'] * next_v
        bounded_y = jnp.clip(y, lower, upper) if profile.get('bound') else y
        # Actor sees original old online Q/V. No extra Q/actor updates are made.
        _, actor_rng = jax.random.split(rng)
        def loss_fn(grad_params):
            qs = net.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
            critic_loss = jnp.mean(jnp.sum((qs-bounded_y)**2, axis=0))
            actor_loss, actor_info = agent.actor_loss(batch, grad_params, actor_rng)
            return critic_loss + actor_loss, {
                'critic/critic_loss': critic_loss,
                'value/value_loss': initial_v_loss,
                'value/post_fit_loss': value_loss(vp),
                'value/steps': jnp.asarray(vsteps, jnp.float32),
                'backup/clip_fraction': jnp.mean((y != bounded_y).astype(jnp.float32)),
                'backup/clip_rms': jnp.sqrt(jnp.mean((y-bounded_y)**2)),
                'backup/fresh_v_shift_rms': jnp.sqrt(jnp.mean((next_v-net.select('value')(batch['next_observations']))**2)),
                **{'actor/'+k:v for k,v in actor_info.items()}}
        new_net, info = net.apply_loss_fn(loss_fn)
        # V's entries in the joint optimizer stay zero: their separate Adam owns
        # all V updates. Installing the fitted parameters is the only V mutation.
        new_net.params['modules_value'] = vp
        agent.target_update(new_net, 'critic')
        agent = agent.replace(network=new_net, rng=new_rng)
        return state.replace(agent=agent, v_opt=vo, v_updates=state.v_updates+vsteps,
                             q_updates=state.q_updates+1), info
    return update


def make_scan(update):
    return jax.jit(lambda state, batches: jax.lax.scan(update, state, batches))


@jax.jit
def probe(state, batch):
    net = state.agent.network
    q = net.select('critic')(batch['observations'], actions=batch['actions'])
    tq = net.select('target_critic')(batch['observations'], actions=batch['actions'])
    v = net.select('value')(batch['observations'])
    vn = net.select('value')(batch['next_observations'])
    y = batch['rewards']+state.agent.config['discount']*batch['masks']*vn
    all_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)]))
    return q, tq, v, y, all_finite


def probe_summary(values, expectile, bounds, alpha):
    q, tq, v, y = [np.asarray(a, np.float64) for a in values[:4]]
    lower, upper = bounds
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    adv = tq.min(0)-v
    score = np.where(adv >= 0, expectile, 1-expectile)*adv
    aw = np.exp(np.minimum(alpha*(q.min(0)-v), np.log(100)))
    return dict(q_mean=float(q.mean()), q_rms=rms(q), v_mean=float(v.mean()),
                v_rms=rms(v), q_centered_rms=rms(q-q.mean()),
                q_abs_p99=float(np.quantile(np.abs(q), .99)),
                q_disagreement_rms=rms(q[0]-q[1]), td_rms=rms(q-y[None]),
                score_mean=float(score.mean()), score_rms=rms(score),
                actor_weight_ess_fraction=float(aw.mean()**2/max(np.mean(aw**2), 1e-300)),
                actor_weight_clip_fraction=float(np.mean(alpha*(q.min(0)-v) >= np.log(100))),
                q_above_upper_rms=rms(np.maximum(q-upper,0)),
                q_below_lower_rms=rms(np.maximum(lower-q,0)),
                q_rms_over_return_bound=rms(q)/max(abs(lower),abs(upper),1),
                finite=bool(np.asarray(values[4])))
