"""Trace observed-action vs generated-action backups on immutable snapshots.

Use an actual two-by-two critic/proposal source intervention at each step.
Only parameters are used for inference; no mixed optimizer state is assembled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import policy_core, selection_v2
from utils.flow_iql_research.next_action_anchors_v1 import augment
from utils.propagation_research.reward_bounds import return_bounds


def sha(path):
    path = Path(path)
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--environment', choices=('ant', 'cubed', 'pz3'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', default='100000')
    args = parser.parse_args()
    inspected_steps = [int(value) for value in args.steps.split(',')]
    assert all(value in (100000,300000,800000,900000,1000000) for value in inspected_steps)
    assert not args.output.exists()
    root = args.root.resolve()
    runs = dict(V2=root/'runs'/f'p1_{args.environment}_ifql_c0_bounded_v2_g999_t90_s0',
                finite=root/'runs'/f'eb1_{args.environment}_bounded_v2_empirical_backup_g999_t90_s0',
                mean=root/'runs'/f'meb1_{args.environment}_bounded_v2_mean_empirical_backup_g999_t90_s0')
    configs = {name:json.loads((run/'config.json').read_text()) for name,run in runs.items()}
    base = configs['V2']['identity']
    allowed = {'num_samples', 'backup_kind', 'backup_samples', 'backup_Q_aggregation'}
    for name, config in configs.items():
        identity = config['identity']
        assert identity['environment'] == base['environment'] and identity['seed'] == base['seed'] == 0
        assert identity['kind'] == 'ifql'
        assert {key:value for key,value in identity['config'].items() if key not in allowed} == {key:value for key,value in base['config'].items() if key not in allowed}
        for source, digest in config['source_sha256'].items():
            assert sha(source) == digest, source
    _, _, raw = ogbench.make_env_and_datasets(base['environment'])
    fields = ('observations', 'actions', 'next_observations', 'rewards', 'masks')
    data = {key:np.asarray(raw[key], np.float32) for key in fields}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    data = augment(raw, data)
    indices = np.random.RandomState(77231).randint(len(data['actions']), size=128)
    obs = jnp.asarray(data['next_observations'][indices])
    observed = jnp.asarray(data['next_actions'][indices])
    valid = data['next_action_valid'][indices].astype(bool)
    assert valid.any()
    bounds = return_bounds(base['environment'], base['config']['discount'],
                           float(data['rewards'].min()), float(data['rewards'].max()), 'ogbench')
    config = dict(base['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        config[key] = tuple(config[key])
    template = policy_core.create(0, obs[:1], observed[:1], config, 'ifql').agent

    @jax.jit
    def generate(params, observations):
        agent = template.replace(network=template.network.replace(params=params))
        actions, _ = selection_v2.proposals(agent, observations, jax.random.PRNGKey(99173), 9, 'behavior')
        return actions

    @jax.jit
    def inspect(params, observations, observed_actions, generated):
        net = template.network
        repeated = jnp.broadcast_to(observations, (9, *observations.shape))
        flow_heads = net.select('target_critic')(repeated, actions=generated, params=params)
        data_heads = net.select('target_critic')(observations, actions=observed_actions, params=params)
        online_heads = net.select('critic')(repeated, actions=generated, params=params)
        value = net.select('value')(observations, params=params)
        flow_q = flow_heads.min(0)
        data_q = data_heads.min(0)
        root = selection_v2.empirical_expectile(flow_q, .9)
        anchored_q = flow_q.at[0].set(jnp.where(jnp.asarray(valid), data_q, flow_q[0]))
        anchored = selection_v2.empirical_expectile(anchored_q, .9)
        mean_q = flow_heads.mean(0)
        mean_root = selection_v2.empirical_expectile(mean_q, .9)
        return dict(flow_Q=flow_q, flow_mean_Q=mean_q, online_flow_Q=online_heads.min(0),
                    online_mean_flow_Q=online_heads.mean(0), observed_Q=data_q, value=value, root=root, mean_root=mean_root,
                    anchored_root=anchored, flow_head_difference=flow_heads[0]-flow_heads[1],
                    observed_head_difference=data_heads[0]-data_heads[1])

    rows, evidence, arrays = [], [], {}
    rms = lambda array:float(np.sqrt(np.mean(np.asarray(array, np.float64)**2)))
    for step in inspected_steps:
        params = {}
        for name, run in runs.items():
            path = run/f'checkpoint_{step}.pkl'
            if not path.exists():
                path = root/'analysis/checkpoint_archive'/run.name/f'checkpoint_{step}.pkl'
            with path.open('rb') as stream:
                saved = pickle.load(stream)
            assert json.loads(json.dumps(saved['identity'])) == configs[name]['identity']
            assert saved['source_sha256'] == configs[name]['source_sha256']
            assert saved['step'] == int(saved['state']['q_updates']) == step
            assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2*step
            assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == step
            params[name] = jax.device_put(saved['state']['agent']['network']['params'])
            evidence.append(dict(source=name, step=step, path=str(path), sha256=sha(path)))
            del saved
        proposals = {name:generate(value, obs) for name,value in params.items()}
        for critic, critic_params in params.items():
            for actor, generated in proposals.items():
                result = jax.device_get(inspect(critic_params, obs, observed, generated))
                assert all(np.isfinite(array).all() for array in result.values())
                flow_q, data_q = result['flow_Q'], result['observed_Q']
                delta = (data_q-result['root'])[valid]
                anchor_shift = (result['anchored_root']-result['root'])[valid]
                row = dict(step=step, critic_source=critic, proposal_source=actor,
                    valid_observed_actions=int(valid.sum()),
                    actual_bootstrap_aggregation=configs[critic]['identity']['config'].get('backup_Q_aggregation', 'min_or_learned_V'),
                    online_action_Q_std_mean=float(result['online_flow_Q'][:,valid].std(0).mean()),
                    online_min_vs_mean_argmax_flip_fraction=float(np.mean(np.argmax(result['online_flow_Q'],0)!=np.argmax(result['online_mean_flow_Q'],0))),
                    online_vs_target_min_argmax_flip_fraction=float(np.mean(np.argmax(result['online_flow_Q'],0)!=np.argmax(flow_q,0))),
                    mean_minus_min_finite_root_mean=float(np.mean(result['mean_root']-result['root'])), observed_Q_mean=float(data_q[valid].mean()),
                    generated_Q_mean=float(flow_q[:,valid].mean()), generated_Q_within_state_std_mean=float(flow_q[:,valid].std(0).mean()),
                    observed_Q_minus_generated_root_mean=float(delta.mean()),
                    observed_Q_minus_generated_root_rms=rms(delta),
                    observed_Q_above_generated_root_fraction=float(np.mean(delta > 0)),
                    anchored_minus_unanchored_root_mean=float(anchor_shift.mean()),
                    anchored_minus_unanchored_root_rms=rms(anchor_shift),
                    learned_V_minus_generated_root_rms=rms((result['value']-result['root'])[valid]),
                    generated_Q_below_return_lower_plus_one_fraction=float(np.mean(flow_q[:,valid] < bounds[0]+1)),
                    observed_Q_below_return_lower_plus_one_fraction=float(np.mean(data_q[valid] < bounds[0]+1)),
                    generated_twin_head_disagreement_rms=rms(result['flow_head_difference'][:,valid]),
                    observed_twin_head_disagreement_rms=rms(result['observed_head_difference'][valid]))
                rows.append(row)
                for key, value in result.items():
                    arrays[f'{step}_{critic}_{actor}_{key}'] = value
    output = dict(time=time.time(), environment=base['environment'], rows=rows, checkpoints=evidence,
                  valid_next_action_fraction=float(data['next_action_valid'].mean()),
                  query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  query_observations_sha256=hashlib.sha256(np.asarray(obs).tobytes()).hexdigest(),
                  sources={str(Path(__file__)):sha(__file__), str(Path(selection_v2.__file__)):sha(selection_v2.__file__)},
                  limits=['Observed next actions occur at exactly the same heldout next states; they are single conditional samples, not expectile or Q oracles.',
                          'A two-by-two frozen parameter intervention separates critic and proposal sources here; no mixed optimizer state is assembled or trained.',
                          'Differences in predicted Q do not establish true action quality or identify the first harmful training update.',
                          'Twin-head disagreement is a proxy, not calibrated uncertainty. Training-return bounds are not oracle labels.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), indices=indices, valid=valid, **arrays)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)); temporary.replace(args.output)
    print(json.dumps(dict(environment=base['environment'], rows=len(rows), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
