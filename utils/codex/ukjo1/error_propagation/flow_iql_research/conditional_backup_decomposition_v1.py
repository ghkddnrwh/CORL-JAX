"""Separate finite proposal size, generator source and critic source on snapshots.

All root references use learned target Q and a learned generator; none is a
population oracle. No model, optimizer or benchmark score is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core, selection_v2
from utils.flow_iql_research.next_action_anchors_v1 import augment
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


ENVIRONMENTS = {
    'ant':'antmaze-large-navigate-singletask-v0',
    'hmmed':'humanoidmaze-medium-navigate-singletask-v0',
    'hmlarge':'humanoidmaze-large-navigate-singletask-v0',
    'cubed':'cube-double-play-singletask-v0',
    'pz3':'puzzle-3x3-play-singletask-v0',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def root64(values, tau):
    """Independent bisection in float64; sample axis is first."""
    values = np.asarray(values, np.float64)
    lower, upper = values.min(0), values.max(0)
    for _ in range(70):
        middle = (lower+upper)/2
        delta = values-middle
        score = np.mean(np.where(delta >= 0, tau, 1-tau)*delta, axis=0)
        lower, upper = np.where(score > 0, middle, lower), np.where(score > 0, upper, middle)
    return (lower+upper)/2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--environment', choices=ENVIRONMENTS, required=True)
    parser.add_argument('--steps', default='100000,300000,500000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    steps = [int(value) for value in args.steps.split(',')]
    assert all(step > 0 and step % 100000 == 0 for step in steps)
    root = args.root.resolve()
    runs = {'V2':root/'runs'/f'p1_{args.environment}_ifql_c0_bounded_v2_g999_t90_s0',
            'finite':root/'runs'/f'eb1_{args.environment}_bounded_v2_empirical_backup_g999_t90_s0'}
    configs = {name:json.loads((run/'config.json').read_text()) for name,run in runs.items()}
    base = configs['V2']['identity']
    ignore = {'num_samples', 'backup_kind', 'backup_samples'}
    for name, config in configs.items():
        ident = config['identity']
        assert ident['environment'] == ENVIRONMENTS[args.environment] == base['environment']
        assert ident['kind'] == 'ifql' and ident['seed'] == base['seed'] == 0
        assert {k:v for k,v in ident['config'].items() if k not in ignore} == {
            k:v for k,v in base['config'].items() if k not in ignore}
        for source, digest in config['source_sha256'].items(): assert sha(source) == digest, source
    _, _, raw = ogbench.make_env_and_datasets(base['environment'])
    fields = ('observations', 'actions', 'next_observations', 'rewards', 'masks')
    data = {key:np.asarray(raw[key], np.float32) for key in fields}
    data['actions'] = np.clip(data['actions'], -1+1e-5, 1-1e-5)
    data = augment(raw, data)
    indices = np.random.RandomState(78331).randint(len(data['actions']), size=64)
    batch = {key:jnp.asarray(value[indices]) for key,value in data.items()}
    valid = np.asarray(batch['next_action_valid'], bool)
    assert valid.any()
    cfg = dict(base['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'): cfg[key] = tuple(cfg[key])
    template = policy_core.create(0, batch['observations'][:1], batch['actions'][:1], cfg, 'ifql').agent
    tau, gamma = cfg['expectile'], cfg['discount']

    @jax.jit
    def generate(params, key):
        agent = template.replace(network=template.network.replace(params=params))
        return selection_v2.proposals(agent, batch['next_observations'], key, 288, 'behavior')[0]

    @jax.jit
    def inspect(params, generated):
        net = template.network
        ob = jnp.broadcast_to(batch['next_observations'], (288, *batch['next_observations'].shape))
        heads = net.select('target_critic')(ob, actions=generated, params=params)
        online = net.select('critic')(ob, actions=generated, params=params).min(0)
        observed = net.select('target_critic')(batch['next_observations'], actions=batch['next_actions'], params=params)
        data_q = net.select('critic')(batch['observations'], actions=batch['actions'], params=params)
        value = net.select('value')(batch['next_observations'], params=params)
        q = heads.min(0)
        gpu_formula_root9 = selection_v2.empirical_expectile(q[:9], tau)
        return dict(target_heads=heads, online_min=online, next_observed_heads=observed,
                    data_online_heads=data_q, value=value, formula_root9=gpu_formula_root9)

    rows, evidence, arrays = [], [], {'indices':indices, 'valid_next_action':valid}
    rms = lambda value:float(np.sqrt(np.mean(np.asarray(value, np.float64)**2)))
    for step in steps:
        parameters = {}
        for name,run in runs.items():
            checkpoint = run/f'checkpoint_{step}.pkl'
            if not checkpoint.exists(): checkpoint = root/'analysis/checkpoint_archive'/run.name/f'checkpoint_{step}.pkl'
            with checkpoint.open('rb') as stream: saved = pickle.load(stream)
            assert saved['step'] == step
            assert json.loads(json.dumps(saved['identity'])) == configs[name]['identity']
            assert saved['source_sha256'] == configs[name]['source_sha256']
            state = saved['state']
            assert int(state['q_updates']) == int(state['agent']['network']['opt_state']['0']['count']) == step
            assert int(state['v_updates']) == int(state['v_opt']['0']['count']) == 2*step
            parameters[name] = jax.device_put(state['agent']['network']['params'])
            assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(parameters[name]))
            evidence.append(dict(source=name, step=step, path=str(checkpoint), sha256=sha(checkpoint)))
            del saved
        bc_delta = max(float(np.max(np.abs(np.asarray(a,np.float64)-np.asarray(b,np.float64)))) for a,b in zip(
            jax.tree_util.tree_leaves(parameters['V2']['modules_actor_flow']),
            jax.tree_util.tree_leaves(parameters['finite']['modules_actor_flow'])))
        for proposal_source, actor_parameters in parameters.items():
            blocks = []
            for block in range(2):
                key = jax.random.fold_in(jax.random.PRNGKey(99183), block)
                blocks.append(generate(actor_parameters, key))
            inspected = {critic:[jax.device_get(inspect(params, actions)) for actions in blocks]
                         for critic,params in parameters.items()}
            for critic, results in inspected.items():
                assert all(np.isfinite(value).all() for result in results for value in result.values())
                heads = np.concatenate([result['target_heads'] for result in results], axis=1)
                q = heads.min(0).astype(np.float64)
                grouped = q.reshape(64, 9, 64).transpose(1, 0, 2)
                roots9 = root64(grouped, tau)
                max9 = grouped.max(0)
                roots288 = np.stack([root64(result['target_heads'].min(0), tau) for result in results])
                root576 = root64(q, tau)
                mean_root576 = root64(heads.mean(0), tau)
                value = results[0]['value'].astype(np.float64)
                observed_q = results[0]['next_observed_heads'].min(0).astype(np.float64)
                online_data_q = results[0]['data_online_heads'].min(0).astype(np.float64)
                expected9 = roots9.mean(0)
                td = np.asarray(batch['rewards'])+gamma*np.asarray(batch['masks'])*expected9-online_data_q
                td_large = np.asarray(batch['rewards'])+gamma*np.asarray(batch['masks'])*root576-online_data_q
                first9 = root64(q[:9], tau)
                finite_vs_large = root576-expected9
                root9_se = roots9.std(0, ddof=1)/np.sqrt(64)
                anchor = q[:9].copy()
                anchor[0] = np.where(valid, observed_q, anchor[0])
                anchor_delta = root64(anchor, tau)-first9
                prefix = f'{step}_{critic}_{proposal_source}_'
                arrays.update({prefix+'target_min_Q':q, prefix+'roots9':roots9, prefix+'root576':root576,
                               prefix+'independent_roots288':roots288, prefix+'learned_V':value,
                               prefix+'observed_next_Q':observed_q, prefix+'transition_backup_residual':td})
                rows.append(dict(step=step, critic_source=critic, proposal_source=proposal_source,
                    queries=64, independent_blocks=2, samples_per_block=288, disjoint_groups_of9=64,
                    BC_parameter_source_max_abs=bc_delta, target_generated_Q_mean=float(q.mean()),
                    target_generated_action_std_mean=float(q.std(0).mean()),
                    target_observed_next_Q_mean=float(observed_q[valid].mean()), learned_V_mean=float(value.mean()),
                    expected_root9_MC_mean=float(expected9.mean()), expected_max9_MC_mean=float(max9.mean()),
                    root576_mean=float(root576.mean()), root576_minus_expected_root9_mean=float(finite_vs_large.mean()),
                    root576_minus_expected_root9_RMS=rms(finite_vs_large),
                    expected_root9_MC_standard_error_RMS=rms(root9_se),
                    independent_root288_difference_RMS=rms(roots288[0]-roots288[1]),
                    root576_minus_mean_independent_root288_mean=float(np.mean(root576-roots288.mean(0))),
                    learned_V_minus_root576_RMS=rms(value-root576), learned_V_minus_root576_mean=float(np.mean(value-root576)),
                    observed_next_Q_minus_root576_mean=float(np.mean((observed_q-root576)[valid])),
                    observed_next_Q_above_root576_fraction=float(np.mean((observed_q-root576)[valid] > 0)),
                    twin_mean_minus_min_root576_mean=float(np.mean(mean_root576-root576)),
                    single_observed_anchor_root9_shift_mean=float(np.mean(anchor_delta[valid])),
                    root9_float32_formula_vs_float64_bisection_RMS=rms(results[0]['formula_root9']-first9),
                    heldout_transition_backup_residual_MC9_RMS=rms(td),
                    heldout_transition_backup_residual_root576_RMS=rms(td_large)))
    output = dict(time=time.time(), environment=base['environment'], rows=rows, checkpoints=evidence,
                  query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  query_observations_sha256=hashlib.sha256(np.asarray(batch['next_observations']).tobytes()).hexdigest(),
                  diagnostic_source_sha256=sha(__file__), root_solver='independent_float64_bisection',
                  limits=['All Q predictions and behavior generators are learned; a 576-action root is not a population expectile oracle.',
                          'Root9 MC standard errors describe conditional proposal randomness, not variation over states/training seeds or independent generator error.',
                          'Two 288-proposal blocks share a critic/generator; their difference describes finite-action sampling only.',
                          'Disjoint groups of9 and the576 reference share draws; their estimates are correlated.',
                          'Observed next actions are single heldout conditional samples, not optimal-action or Q oracles.',
                          'Transition residuals use one observed next transition per state/action; they are not exact expected Bellman errors.',
                          'Checkpoint sources are crossed for inference only. No optimizer state is mixed or trained, no primary metric is changed.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(output, indent=2, allow_nan=False))
    temp.replace(args.output)
    print(json.dumps(dict(environment=base['environment'], rows=len(rows), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
