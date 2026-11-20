"""Read-only, shared-forward-function comparison of two Ant critic trajectories.

One frozen baseline1M behavior Flow generates the same bank for all critics.
Reset observations, noises and evaluation program are fixed. This measures
function differences and preferred-action disagreement, not ground-truth
Q bias, Bellman error propagation, or an actor-loss causal effect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research import selection_v2
import jax
import jax.numpy as jnp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(tree, prefix=''):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from arrays(tree[key], prefix + '/' + str(key))
    elif isinstance(tree, (tuple, list)):
        for key, value in enumerate(tree):
            yield from arrays(value, prefix + '/' + str(key))
    elif tree is not None:
        yield prefix, np.asarray(tree)


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, np.float64) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    a = parser.parse_args()
    plan = json.loads(a.proposal.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert all(d.platform == 'cpu' for d in jax.devices())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    for path, digest in plan['checkpoint_sha256'].items():
        assert sha(path) == digest
    output = Path(plan['output'])
    assert not output.exists()
    root = Path(plan['root'])
    obs_path = Path(plan['observations'])
    assert sha(obs_path) == plan['observations_sha256']
    with np.load(obs_path, allow_pickle=False) as data:
        observations = data['observations_float32'].copy()
        assert np.array_equal(data['reset_seeds'], np.arange(42, 106))
    assert observations.shape == (64, 29)
    reference_run = root / 'runs' / plan['baseline_run']
    configuration = json.loads((reference_run / 'config.json').read_text())
    cfg = dict(configuration['identity']['config'])
    for key in ['actor_hidden_dims', 'value_hidden_dims']:
        cfg[key] = tuple(cfg[key])
    template = core.create(0, jnp.zeros((1, 29)), jnp.zeros((1, 8)), cfg, 'bank_bc')
    agent = template.base.agent

    # Both model evaluations use these very same compiled functions and the
    # same template configuration. Only the loaded parameter argument changes.
    @jax.jit
    def flow_forward(parameters, obs, noises):
        current = agent.replace(network=agent.network.replace(params=parameters))
        return core.lean.flow(current, obs, noises, 10)

    @jax.jit
    def critic_forward(parameters, obs, actions):
        network = agent.network.replace(params=parameters)
        return network.select('critic')(obs, actions=actions), network.select('target_critic')(obs, actions=actions)

    @jax.jit
    def finite_root(target_q):
        return selection_v2.empirical_expectile(jnp.swapaxes(target_q.mean(0), 0, 1), .9)

    def load(rid, step):
        path = root / 'runs' / rid / f'checkpoint_{step}.pkl'
        with path.open('rb') as stream:
            checkpoint = pickle.load(stream)
        assert checkpoint['step'] == step and checkpoint['identity']['seed'] == 0
        assert checkpoint['identity']['environment'] == 'antmaze-large-navigate-singletask-v0'
        params = checkpoint['state']['base']['agent']['network']['params']
        assert all(np.isfinite(value).all() for _, value in arrays(params))
        return checkpoint, params

    _, final_flow_params = load(plan['baseline_run'], 1000000)
    broadcast = jnp.broadcast_to(jnp.asarray(observations), (4, 9, *observations.shape))
    noises = jax.random.normal(jax.random.PRNGKey(plan['noise_seed']), (4, 9, 64, 8))
    actions = flow_forward(final_flow_params, broadcast, noises)
    actions.block_until_ready()
    assert np.isfinite(np.asarray(actions)).all()
    assert np.max(np.abs(np.asarray(actions))) <= 1
    print(json.dumps(dict(stage='fixed_final_Flow_bank_ready', unique_reset_states=64,
        noise_banks_per_state=4, actions_per_bank=9)), flush=True)
    raw = dict(observations=observations, noises=np.asarray(noises), actions=np.asarray(actions))
    rows = []
    for step in [800000, 900000, 1000000]:
        left, pleft = load(plan['baseline_run'], step)
        right, pright = load(plan['variant_run'], step)
        al, ar = dict(arrays(pleft)), dict(arrays(pright))
        flow_keys = [key for key in al if 'flow' in key]
        assert len(flow_keys) == 10
        assert all(al[key].tobytes() == ar[key].tobytes() for key in flow_keys)
        ql, tl = critic_forward(pleft, broadcast, actions)
        qr, tr = critic_forward(pright, broadcast, actions)
        rl, rr = finite_root(tl), finite_root(tr)
        ql, qr, tl, tr, rl, rr = [np.asarray(x) for x in [ql, qr, tl, tr, rl, rr]]
        assert ql.shape == qr.shape == tl.shape == tr.shape == (2, 4, 9, 64)
        assert all(np.isfinite(x).all() for x in [ql, qr, tl, tr, rl, rr])
        vl, vr = ql.min(0), qr.min(0)
        il, ir = vl.argmax(1), vr.argmax(1)
        target_l, target_r = np.swapaxes(tl.mean(0), 0, 1), np.swapaxes(tr.mean(0), 0, 1)
        score_l = np.mean(np.where(target_l >= rl[None], .9, .1) * (target_l - rl[None]), axis=0)
        score_r = np.mean(np.where(target_r >= rr[None], .9, .1) * (target_r - rr[None]), axis=0)
        # Preferences are internal to each critic and never called true regret.
        other_left = np.take_along_axis(vl, ir[:, None, :], axis=1)[:, 0, :]
        other_right = np.take_along_axis(vr, il[:, None, :], axis=1)[:, 0, :]
        row = dict(step=step, common_forward_program=True, same_Flow_at_matched_checkpoint=True,
            online_Q_difference_RMS=rms(qr - ql), target_Q_difference_RMS=rms(tr - tl),
            online_Q_difference_max_abs=float(np.max(np.abs(qr - ql))),
            baseline_Q_RMS=rms(ql), MDN_run_Q_RMS=rms(qr),
            online_minQ_best9_index_disagreement_fraction=float(np.mean(il != ir)),
            disagreement_banks=int(np.count_nonzero(il != ir)), bank_state_cases=4 * 64,
            baseline_internal_preference_gap_for_MDN_choice=float(np.mean(vl.max(1) - other_left)),
            MDN_internal_preference_gap_for_baseline_choice=float(np.mean(vr.max(1) - other_right)),
            target_finite_expectile_difference_RMS=rms(rr - rl),
            target_finite_expectile_difference_mean=float(np.mean(rr - rl)),
            baseline_finite_empirical_score_RMS=rms(score_l), MDN_finite_empirical_score_RMS=rms(score_r))
        rows.append(row)
        for name, value in dict(baseline_online=ql, MDN_online=qr, baseline_target=tl, MDN_target=tr,
            baseline_root=rl, MDN_root=rr, baseline_best9_index=il, MDN_best9_index=ir).items():
            raw[f'{step}_{name}'] = value
        print(json.dumps(row), flush=True)
    for path, digest in plan['checkpoint_sha256'].items():
        assert sha(path) == digest
    archive = Path(plan['raw_predictions'])
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **raw)
    result = dict(time=time.time(), status='pass', protocol='Ant_fixed_Flow_common_forward_critic_comparison_v1',
        proposal_sha256=sha(a.proposal), source_sha256=plan['source_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'], observations_sha256=sha(obs_path),
        raw_predictions=str(archive), raw_predictions_sha256=sha(archive), rows=rows,
        training_seed=0, unique_reset_states=64, independent_noise_banks_per_state=4,
        actions_per_bank=9, fixed_behavior_Flow_step=1000000,
        optimizer_updates_performed=0, performance_evaluation_episodes=0,
        model_rewards_primary_scores_or_GPU_queue_changed=False,
        scope='One Ant case, real reset states only. Shared critic program removes an evaluation-implementation difference from this forward comparison. Q/root differences and preferred-action disagreement are not ground-truth bias or measured propagation error. Fixed final-Flow proposals are not each earlier checkpoint training distribution.',
        interpretation='Read together with actual native parameter comparisons and paired teacher success. Do not ascribe native critic divergence to a particular GPU kernel, actor loss feedback, or dataset stream without further evidence.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
