"""Matched teacher/actor distributions on identical states and Gaussian banks.

Selected best9 teacher variance is distinguished from proposal-bank variance.
Cross-distance U-statistics exclude matched indices because the two samples
at the same index share their noise and are dependent.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def moments(actor,teacher):
    p,t = np.asarray(actor,np.float64),np.asarray(teacher,np.float64)
    assert p.shape == t.shape and p.ndim == 3
    k = len(p)
    assert k > 1
    pm,tm = p.mean(0),t.mean(0)
    pv,tv = p.var(0),t.var(0)
    cov = ((p-pm)*(t-tm)).mean(0)
    bias = float(((pm-tm)**2).mean())
    paired = float(((p-t)**2).mean())
    reconstructed = float(bias+pv.mean()+tv.mean()-2*cov.mean())
    assert abs(paired-reconstructed) < 1e-10
    def distance(x,y):
        d = np.linalg.norm(x[:,None]-y[None,:],axis=-1)
        # Sum over all distinct draw indices and all conditioning states.
        return float((d.sum()-np.trace(d,axis1=0,axis2=1).sum())/(k*(k-1)*p.shape[1]))
    energy = 2*distance(p,t)-distance(p,p)-distance(t,t)
    return dict(paired_action_MSE=paired,mean_action_bias_MSE=bias,
        actor_selected_action_variance=float(pv.mean()),
        teacher_selected_best9_variance=float(tv.mean()),
        actor_teacher_same_noise_covariance=float(cov.mean()),
        moment_identity_abs_error=abs(paired-reconstructed),
        independent_draw_MSE_moment_estimate=float(bias+pv.mean()+tv.mean()),
        coupled_noise_MSE_reduction=float(2*cov.mean()),
        actor_to_selected_teacher_variance_ratio=float(pv.mean()/max(tv.mean(),1e-12)),
        energy_distance_distinct_index_Ustat=energy,
        energy_convention='2E||P-T||-E||P-Pprime||-E||T-Tprime||; all terms use distinct draw indices',
        finite_sample_Ustat_can_be_negative=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift-case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    parent_path = args.drift_case/'result.json'
    parent = json.loads(parent_path.read_text())
    assert parent['status'] == 'complete' and 800000 in parent['steps']
    arrays = args.drift_case/'same_state_noise_drift_arrays_v1.npz'
    assert sha(arrays) == parent['arrays_sha256']
    with np.load(arrays) as z:
        obs = jnp.asarray(z['observations'][:64])
        dim = z['actor_800000'].shape[-1]
    run = Path(parent['run'])
    conf = json.loads((run/'config.json').read_text())
    for p,digest in conf['source_sha256'].items():
        assert sha(p) == digest,p
    cfg = dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):
        cfg[k] = tuple(cfg[k])
    template = flat.create(conf['identity']['seed'],obs[:1],jnp.zeros((1,dim)),cfg,'bank_bc')
    checkpoint = run/'checkpoint_800000.pkl'
    assert sha(checkpoint) == parent['checkpoint_sha256']['800000']
    with checkpoint.open('rb') as f:
        saved = pickle.load(f)
    assert saved['step'] == 800000 and saved['identity'] == conf['identity']
    assert saved['source_sha256'] == conf['source_sha256']
    state = flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.actor.opt_state[0].count) == int(state.base.agent.network.opt_state[0].count) == 800000
    assert int(state.base.v_updates) == 0

    @jax.jit
    def sample(state):
        agent = state.base.agent
        repeated = jnp.broadcast_to(obs,(32,*obs.shape))
        broad,noises = flat.lean.bank(agent,repeated,jax.random.PRNGKey(74193),9)
        bank = flat.lean.flow(agent,broad,noises,10)
        q = agent.network.select('critic')(broad,actions=bank).min(0)
        teacher = flat.gather(bank,q.argmax(0))
        actor = jnp.clip(state.actor(repeated,flat.packed_input(noises,cfg['onestep_specification'])),-1,1)
        aq = agent.network.select('critic')(repeated,actions=actor).min(0)
        return dict(actor=actor,teacher=teacher,actor_Q=aq,teacher_Q=q.max(0),
                    proposal_bank_variance=bank.var(0).mean())

    x = jax.device_get(sample(state))
    assert all(np.isfinite(a).all() for a in x.values())
    stats = moments(x['actor'],x['teacher'])
    regret = x['teacher_Q']-x['actor_Q']
    stats.update(critic_predicted_actor_regret_mean=float(regret.mean()),
                 critic_predicted_actor_regret_p95=float(np.quantile(regret,.95)),
                 proposal_bank_variance=float(x['proposal_bank_variance']))
    args.output.mkdir(parents=True)
    raw = args.output/'coupled_draws_v1.npz'
    np.savez_compressed(raw,observations=np.asarray(obs),**x)
    result = dict(time=time.time(),status='complete',environment=parent['environment'],
        parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        checkpoint_sha256=sha(checkpoint),step=800000,conditioning_replay_states=64,
        independent_draws_per_state=32,proposals_per_draw=9,noise_seed=74193,
        same_state_and_noise_coupling=True,optimizer_updates=0,evaluation_episodes=0,
        stats=stats,raw_arrays_sha256=sha(raw),
        source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__,*flat.source_paths()]},
        scope='Matched64-replay-state distribution diagnostic, not task success, coverage oracle, '
              'trueQerror, population guarantee or a causal variance intervention.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(stats),flush=True)


if __name__ == '__main__':
    main()
