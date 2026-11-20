"""Compare a learned policy-flow distribution with its frozen best9 teacher.

Unrelated policy/teacher latents are compared as distributions, never as paired
action-MSE labels. All Q-based quantities are model diagnostics, not true returns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import rank_flow_core_v1 as core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy(first, second):
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    cross = np.linalg.norm(first[:,None]-second[None,:], axis=-1).mean((0,1))
    within_first = np.linalg.norm(first[:,None]-first[None,:], axis=-1).mean((0,1))
    within_second = np.linalg.norm(second[:,None]-second[None,:], axis=-1).mean((0,1))
    return 2*cross-within_first-within_second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args(); assert not args.output.exists()
    config = json.loads((args.run/'config.json').read_text()); identity = config['identity']
    assert identity['kind'] == 'rank_flow'
    for path, digest in config['source_sha256'].items():
        assert sha(path) == digest, path
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    checkpoint = args.run/f'checkpoint_{args.step}.pkl'
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert json.loads(json.dumps(saved['identity'])) == identity
    assert saved['source_sha256'] == config['source_sha256']
    assert saved['step'] == int(saved['state']['q_updates']) == args.step
    assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2*args.step
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == args.step
    _, _, val = ogbench.make_env_and_datasets(identity['environment'])
    indices = np.random.RandomState(77231).randint(len(val['actions']), size=args.queries)
    obs = np.asarray(val['observations'][indices], np.float32)
    data_actions = np.clip(np.asarray(val['actions'][indices], np.float32), -1+1e-5, 1-1e-5)
    template = core.create(identity['seed'], jnp.zeros_like(obs[:1]), jnp.zeros_like(data_actions[:1]), cfg)
    state = flax.serialization.from_state_dict(template, saved['state']); agent = state.agent

    @jax.jit
    def inspect(observations, observed_actions):
        repeated = jnp.broadcast_to(observations, (args.samples, *observations.shape))
        teacher, scores, _ = agent.selected_teacher_targets(repeated, jax.random.PRNGKey(7171))
        teacher_second, _, _ = agent.selected_teacher_targets(repeated, jax.random.PRNGKey(7174))
        policy = agent.sample_actions(repeated, jax.random.PRNGKey(7172), 0.)
        noises = jax.random.normal(jax.random.PRNGKey(7173), policy.shape)
        behavior = agent.compute_flow_actions(repeated, noises)
        net = agent.network
        def q(actions):
            return net.select('critic')(repeated, actions=actions).min(0)
        q_policy, q_behavior = q(policy), q(behavior)
        p0 = jax.random.normal(jax.random.PRNGKey(7175), observed_actions.shape)
        time = jax.random.uniform(jax.random.PRNGKey(7176), (*observed_actions.shape[:-1], 1))
        x1 = teacher[0]
        xt = (1-time)*p0+time*x1
        velocity = x1-p0
        policy_prediction = net.select('actor_policy_flow')(observations, xt, time)
        bc_prediction = net.select('actor_flow')(observations, xt, time)
        observed_xt = (1-time)*p0+time*observed_actions
        observed_prediction = net.select('actor_flow')(observations, observed_xt, time)
        ordered = jnp.sort(scores, axis=0)
        return dict(policy_actions=policy, teacher_selected_actions=teacher,
            independent_teacher_selected_actions=teacher_second, behavior_actions=behavior,
            policy_Q=q_policy, teacher_Q=scores.max(0), behavior_Q=q_behavior,
            teacher_winner_margin=ordered[-1]-ordered[-2],
            teacher_Q_gain_over_mean=scores.max(0)-scores.mean(0),
            policy_selected_target_FM_loss=jnp.mean((policy_prediction-velocity)**2),
            BC_selected_target_FM_loss=jnp.mean((bc_prediction-velocity)**2),
            BC_observed_data_FM_loss=jnp.mean((observed_prediction-(observed_actions-p0))**2))

    result = jax.device_get(inspect(jnp.asarray(obs), jnp.asarray(data_actions)))
    assert all(np.isfinite(array).all() for array in result.values())
    energy_policy = energy(result['policy_actions'], result['teacher_selected_actions'])
    energy_behavior = energy(result['behavior_actions'], result['teacher_selected_actions'])
    energy_resolution = energy(result['independent_teacher_selected_actions'], result['teacher_selected_actions'])
    assert min(energy_policy.min(), energy_behavior.min(), energy_resolution.min()) >= -1e-7
    rms = lambda array:float(np.sqrt(np.mean(np.asarray(array, np.float64)**2)))
    summary = dict(policy_vs_teacher_energy_distance_mean=float(energy_policy.mean()),
        BC_vs_teacher_energy_distance_mean=float(energy_behavior.mean()),
        independent_teacher_vs_teacher_energy_distance_mean=float(energy_resolution.mean()),
        model_Q_teacher_minus_policy_mean=float(np.mean(result['teacher_Q']-result['policy_Q'])),
        model_Q_policy_minus_BC_mean=float(np.mean(result['policy_Q']-result['behavior_Q'])),
        teacher_model_Q_gain_over_pool_mean=float(result['teacher_Q_gain_over_mean'].mean()),
        teacher_winner_margin_mean=float(result['teacher_winner_margin'].mean()),
        teacher_numeric_tie_fraction=float(np.mean(result['teacher_winner_margin'] <= 1e-6)),
        policy_selected_target_FM_loss=float(result['policy_selected_target_FM_loss']),
        BC_selected_target_FM_loss=float(result['BC_selected_target_FM_loss']),
        BC_observed_data_FM_loss=float(result['BC_observed_data_FM_loss']),
        conditional_action_mean_shift_policy_to_teacher_rms=rms(result['policy_actions'].mean(0)-result['teacher_selected_actions'].mean(0)))
    for name, key in [('policy', 'policy_actions'), ('teacher', 'teacher_selected_actions'), ('BC', 'behavior_actions')]:
        summary[name+'_action_saturation_fraction'] = float(np.mean(np.abs(result[key]) >= .98))
        summary[name+'_within_state_action_std_mean'] = float(np.std(result[key], axis=0).mean())
    output = dict(time=time.time(), run=args.run.name, step=args.step, environment=identity['environment'],
        samples_per_state=args.samples, queries=args.queries, query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        checkpoint_sha256=sha(checkpoint), config_sha256=sha(args.run/'config.json'),
        diagnostic_source_sha256=sha(__file__), summary=summary,
        limits=['All Q quantities use the learned online min-Q, not oracle values or actual returns.',
                'Energy distances compare independent finite samples at heldout dataset states; the teacher-teacher value estimates Monte Carlo resolution, not a confidence interval.',
                'A flow-matching objective does not prescribe a pointwise coupling between independent policy and teacher latents.',
                'Dataset states differ from deployed-policy states. Actual performance requires full environment evaluations.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), indices=indices, **result,
                        energy_policy=energy_policy, energy_behavior=energy_behavior, energy_resolution=energy_resolution)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)); temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name, step=args.step, summary=summary)), flush=True)


if __name__ == '__main__':
    main()
