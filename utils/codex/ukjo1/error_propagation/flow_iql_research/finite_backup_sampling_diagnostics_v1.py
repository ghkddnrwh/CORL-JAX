"""Separate finite-proposal sensitivity from learned-V fitting, on frozen models.

The 288-proposal reference is another model-based sample, not a true conditional
expectile. Its proposals also contain the small-sample groups being compared.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import policy_core, selection_v2
from utils.propagation_research.reward_bounds import return_bounds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=128)
    args = parser.parse_args()
    assert not args.output.exists()
    config = json.loads((args.run / 'config.json').read_text())
    identity = config['identity']
    assert identity['kind'] == 'ifql'
    for source, digest in config['source_sha256'].items():
        assert sha(source) == digest, source
    checkpoint = args.run / 'checkpoint_1000000.pkl'
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['source_sha256'] == config['source_sha256']
    assert json.loads(json.dumps(saved['identity'])) == identity
    assert saved['step'] == int(saved['state']['q_updates']) == 1000000
    assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2000000
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == 1000000
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    _, _, validation = ogbench.make_env_and_datasets(identity['environment'])
    indices = np.random.RandomState(77231).randint(len(validation['actions']), size=args.queries)
    observations = np.asarray(validation['next_observations'][indices], np.float32)
    actions = np.asarray(validation['actions'][indices[:1]], np.float32)
    state = policy_core.create(identity['seed'], jnp.zeros_like(observations[:1]),
                               jnp.zeros_like(actions), cfg, 'ifql')
    state = flax.serialization.from_state_dict(state, saved['state'])
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(state))
    agent = state.agent

    @jax.jit
    def probe(obs):
        actions, _ = selection_v2.proposals(agent, obs, jax.random.PRNGKey(99173), 288, 'behavior')
        repeated = jnp.broadcast_to(obs, (288, *obs.shape))
        target_q = agent.network.select('target_critic')(repeated, actions=actions).min(0)
        value = agent.network.select('value')(obs)
        return target_q, value, jnp.mean((jnp.abs(actions) >= .98).astype(jnp.float32))

    q, value, saturation = jax.device_get(probe(jnp.asarray(observations)))
    q = np.asarray(q)
    blocks = q.reshape(32, 9, args.queries).transpose(1, 0, 2)
    rms = lambda array: float(np.sqrt(np.mean(np.asarray(array, np.float64)**2)))
    summaries = {}
    roots = {}
    for tau in (.9, .99):
        large = np.asarray(selection_v2.empirical_expectile(jnp.asarray(q), tau))
        small = np.asarray(selection_v2.empirical_expectile(jnp.asarray(blocks), tau))
        prefix32 = np.asarray(selection_v2.empirical_expectile(jnp.asarray(q[:32]), tau))
        score = np.mean(np.where(blocks >= small, tau, 1-tau) * (blocks-small), axis=0)
        assert np.max(np.abs(score)) < .02
        roots[str(tau)] = dict(reference288=large, groups9=small, prefix32=prefix32)
        summaries[str(tau)] = dict(
            group9_minus_reference288_mean=float(np.mean(small-large)),
            group9_minus_reference288_rms=rms(small-large),
            within_state_group9_standard_deviation_mean=float(np.std(small, axis=0, ddof=1).mean()),
            prefix32_minus_reference288_mean=float(np.mean(prefix32-large)),
            prefix32_minus_reference288_rms=rms(prefix32-large),
            learned_V_minus_reference288_mean=float(np.mean(value-large)),
            learned_V_minus_reference288_rms=rms(value-large),
            finite_group_score_abs_max=float(np.max(np.abs(score))),
            reference288_gap_to_max_mean=float(np.mean(q.max(0)-large)))
    assert np.all(roots['0.99']['groups9'] >= roots['0.9']['groups9'])
    lower, upper = return_bounds(identity['environment'], cfg['discount'],
                                float(np.min(validation['rewards'])),
                                float(np.max(validation['rewards'])), source='ogbench')
    rewards = np.asarray(validation['rewards'][indices], np.float32)
    masks = np.asarray(validation['masks'][indices], np.float32)
    projected = {}
    for tau in (.9, .99):
        y = rewards + cfg['discount'] * masks * roots[str(tau)]['groups9']
        projected[str(tau)] = dict(clip_fraction=float(np.mean((y < lower) | (y > upper))),
                                   unclipped_target_mean=float(y.mean()))
    result = dict(time=time.time(), run=args.run.name, identity=identity, step=saved['step'],
                  queries=args.queries, generated_proposals=288, groups=32, group_size=9,
                  state_source='heldout validation next_observations',
                  query_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  query_observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
                  checkpoint_sha256=sha(checkpoint), config_sha256=sha(args.run/'config.json'),
                  diagnostic_source_sha256=sha(__file__), proposal_source_sha256=sha(selection_v2.__file__),
                  results=summaries, target_projection=projected,
                  proposal_action_saturation_fraction=float(saturation),
                  proposal_target_Q_mean=float(q.mean()), proposal_within_state_Q_std_mean=float(q.std(0).mean()),
                  high_tau_group9_minus_tau90_reference288_mean=float(np.mean(roots['0.99']['groups9']-roots['0.9']['reference288'])),
                  limits=['All expectiles are over this frozen learned behavior generator and critic, never oracle conditional residuals.',
                          'The 288 reference overlaps the small groups; these are sensitivity diagnostics, not independent training seeds or confidence intervals.',
                          'Counterfactual tau99 changes only the frozen scalar calculation; trained tau99 performance requires separate runs.',
                          'Heldout dataset next-states are not the deployed policy state distribution; numerical shifts do not prove action quality.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), target_q=q, learned_value=value,
                        query_indices=indices, **{f'root{tau}_{name}':array for tau, groups in roots.items() for name,array in groups.items()})
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)); temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name, results=summaries)), flush=True)


if __name__ == '__main__':
    main()
