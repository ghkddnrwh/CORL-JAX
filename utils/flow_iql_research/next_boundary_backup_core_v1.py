"""Finite expectile backup with an exact known next-success continuation.

Keep the original finite-backup learner and its replay/RNG/update clocks.
Only replace bootstrap(root) by0 when the next state is certified to have
state-based r0/mask0 for all actions. No new loss coefficient or sampling ratio.
"""
from utils.flow_iql_research import empirical_backup_core_v1 as base
from utils.flow_iql_research import policy_core as reference, selection_v2
from utils.flow_iql_research.policy_core import ENVIRONMENTS, State, create, make_scan
import jax
import jax.numpy as jnp


PROFILES = {'bounded_v2_next_boundary_backup': reference.PROFILES['bounded_v2']}


def config_for(environment, discount=None, expectile=.9, width=None, alpha=None):
    cfg = base.config_for(environment,discount,expectile,width,alpha)
    cfg.update(next_boundary_rule='zero_continuation_at_certified_adjacent_success_state')
    return cfg


def constrained_batch(batch):
    return {**batch,'masks':batch['masks']*(1.-batch['next_success'])}


def build_update(profile_name,bounds,kind='ifql'):
    assert profile_name == 'bounded_v2_next_boundary_backup' and kind == 'ifql'
    original = base.build_update('bounded_v2_empirical_backup',bounds,kind)
    def update(state,batch):
        state,info = original(state,constrained_batch(batch))
        return state,{**info,
            'boundary/next_success_fraction':jnp.mean(batch['next_success']),
            'boundary/active_next_success_fraction':jnp.mean(batch['masks']*batch['next_success'])}
    return update


@jax.jit
def probe(state,batch):
    q,tq,v,learned_y,finite = reference.probe(state,batch)
    cfg = state.agent.config
    actions,_ = selection_v2.proposals(state.agent,batch['next_observations'],
        jax.random.PRNGKey(19402),cfg['backup_samples'],'behavior')
    repeated = jnp.broadcast_to(batch['next_observations'],(cfg['backup_samples'],*batch['next_observations'].shape))
    root = selection_v2.empirical_expectile(state.agent.network.select('target_critic')(repeated,actions=actions).min(0),cfg['expectile'])
    raw_y = batch['rewards']+cfg['discount']*batch['masks']*root
    corrected_y = batch['rewards']+cfg['discount']*batch['masks']*(1.-batch['next_success'])*root
    finite = finite & jnp.all(jnp.isfinite(root)) & jnp.all(jnp.isfinite(corrected_y))
    return q,tq,v,corrected_y,finite,raw_y,root,batch['next_success'],batch['masks'],learned_y


def probe_summary(values,expectile,bounds,alpha):
    import numpy as np
    result = reference.probe_summary(values[:5],expectile,bounds,alpha)
    q,y,raw_y,root,next_success,masks,learned_y = [np.asarray(values[i],np.float64) for i in (0,3,5,6,7,8,9)]
    active = next_success*masks;count = active.sum();rms = lambda x:float(np.sqrt(np.mean(x*x)))
    result.update(snapshot_boundary_projected_actual_TD_RMS=rms(q-np.clip(y,*bounds)[None]),
        snapshot_unmodified_finite_backup_counterfactual_TD_RMS=rms(q-np.clip(raw_y,*bounds)[None]),
        snapshot_learned_V_only_counterfactual_TD_RMS=rms(q-learned_y[None]),
        snapshot_boundary_active_count=int(count),snapshot_boundary_active_fraction=float(active.mean()),
        snapshot_boundary_target_shift_RMS=rms(y-raw_y),
        snapshot_next_success_root_RMS_to_known_zero=float(np.sqrt(np.sum(root*root*active)/count)) if count else None,
        snapshot_next_success_root_mean=float(np.sum(root*active)/count) if count else None)
    return result
