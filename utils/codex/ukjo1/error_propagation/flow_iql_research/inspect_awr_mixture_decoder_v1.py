"""Fixed-query support/critic diagnostics for the frozen Ant decoder pilot.

Mixture likelihood is a fitted, pre-clipping density proxy, not true behavior
support or the density of the clipped action law. Model Q is never an oracle.
"""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research import awr_mixture_core_v1 as core
from utils.flow_iql_research.evaluate_awr_mixture_decoder_v1 import make_selector
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.rank_mean_bundle_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert (args.run / 'completed.json').exists()
    conf = json.loads((args.run / 'config.json').read_text())
    identity = conf['identity']
    assert identity['kind'] == 'awr_mixture' and identity['profile'] == 'bounded_v2'
    env, train, validation = ogbench.make_env_and_datasets(identity['environment'])
    cfg = dict(identity['config'])
    for name in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[name] = tuple(cfg[name])
    template = core.create(identity['seed'], jnp.asarray(train['observations'][:1]),
        jnp.asarray(train['actions'][:1]), cfg, 'awr_mixture')
    rng = np.random.RandomState(91350)
    samples = {}
    for split, data in (('train', train), ('validation', validation)):
        indices = rng.choice(len(data['observations']), 512, replace=False)
        samples[split] = (indices, np.asarray(data['observations'][indices], np.float32),
            np.clip(np.asarray(data['actions'][indices], np.float32), -1 + 1e-5, 1 - 1e-5))
    rows, models, arrays = [], [], {}
    for step in (800000, 900000, 1000000):
        actual_conf, saved, checkpoint = read(args.run, step)
        assert actual_conf == conf and int(saved['state']['v_updates']) == 2 * step
        state = flax.serialization.from_state_dict(template, saved['state'])
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        selectors = {rule: make_selector(state.agent, rule) for rule in ('component_Q', 'sample9_Q')}

        @jax.jit
        def inspect(params, observations, data_actions, key):
            model = state.agent.replace(network=state.agent.network.replace(params=params))
            logits, means, logstd = model.network.select('actor')(observations)
            probability = jax.nn.softmax(logits)
            component_actions = jnp.clip(jnp.moveaxis(means, -2, 0), -1, 1)
            repeated = jnp.broadcast_to(observations, (5, *observations.shape))
            component_heads = model.network.select('critic')(repeated, actions=component_actions)
            best_component = component_heads.min(0).argmax(0)
            selected_weight = jnp.take_along_axis(probability, best_component[..., None], axis=-1)[..., 0]
            greedy_component = probability.argmax(-1)
            greedy = jnp.clip(core.greedy_component_mean(logits, means), -1, 1)
            greedy_heads = model.network.select('critic')(observations, actions=greedy)
            greedy_density = core.mixture_log_prob(logits, means, logstd, greedy)[0]
            found = dict(component_probabilities=probability, component_Q_heads=component_heads,
                greedy_component=greedy_component, Q_selected_component=best_component,
                Q_selected_component_probability=selected_weight, greedy_Q_heads=greedy_heads,
                greedy_preclip_log_density=greedy_density)
            for rule, select in selectors.items():
                action, metrics = select(params, observations, key)
                heads = model.network.select('critic')(observations, actions=action)
                density = core.mixture_log_prob(logits, means, logstd, action)[0]
                found.update({rule + '/' + k: v for k, v in dict(
                    action=action, Q_heads=heads, preclip_log_density=density,
                    Q_minus_greedy=metrics['selected_Q_minus_greedy'],
                    action_RMS_from_greedy=metrics['action_RMS_from_greedy'],
                    preclip_log_density_minus_greedy=density - greedy_density).items()})
            return found

        for split, (indices, observations, actions) in samples.items():
            key = jax.random.PRNGKey(19407 if split == 'train' else 19408)
            found = {k: np.asarray(v) for k, v in inspect(state.agent.network.params, observations, actions, key).items()}
            assert all(np.isfinite(v).all() for v in found.values())
            for k, v in found.items():
                arrays[f'{split}_{step}_{k}'] = v
            arrays[f'{split}_indices'] = indices
            weight = found['Q_selected_component_probability']
            row = dict(split=split, step=step, query_states=512,
                Q_component_decoder_differs_from_most_probable_fraction=float(np.mean(found['Q_selected_component'] != found['greedy_component'])),
                Q_selected_component_probability_quantiles=np.quantile(weight, [.1, .5, .9]).tolist(),
                Q_selected_component_probability_mean=float(weight.mean()),
                Q_selected_component_weight_below_point1_fraction=float(np.mean(weight < .1)),
                fitted_max_component_probability_mean=float(found['component_probabilities'].max(-1).mean()),
                methods=[])
            for rule in selectors:
                prefix = rule + '/'
                heads = found[prefix + 'Q_heads']
                row['methods'].append(dict(rule=rule,
                    own_Q_minus_greedy_mean=float(found[prefix + 'Q_minus_greedy'].mean()),
                    mean_action_RMS_from_greedy=float(found[prefix + 'action_RMS_from_greedy'].mean()),
                    fitted_preclip_log_density_minus_greedy_mean=float(found[prefix + 'preclip_log_density_minus_greedy'].mean()),
                    fitted_preclip_density_lower_than_greedy_fraction=float(np.mean(found[prefix + 'preclip_log_density_minus_greedy'] < 0)),
                    selected_action_twin_head_gap_mean=float(np.mean(np.abs(heads[0] - heads[1]))),
                    greedy_action_twin_head_gap_mean=float(np.mean(np.abs(found['greedy_Q_heads'][0] - found['greedy_Q_heads'][1])))))
            rows.append(row)
        models.append(dict(step=step, checkpoint_sha256=sha(checkpoint), config_sha256=sha(args.run / 'config.json')))
        del saved, state, inspect, selectors
    env.close()
    np.savez_compressed(args.output / 'raw_fixed_queries.npz', **arrays)
    result = dict(time=time.time(), status='complete', rows=rows, models=models,
        source_sha256=sha(__file__), selector_source_sha256=sha('utils/flow_iql_research/evaluate_awr_mixture_decoder_v1.py'),
        config_sha256=sha(args.run / 'config.json'), query_rng=91350, queries_per_split=512,
        goal_complete=False, primary_scores_unchanged=True,
        limits=['Same fixed512 observations are reused at3checkpoints, not independent state replicates.',
            'Fixed dataset queries, not the actual policy rollout visitation.',
            'Fitted unclipped Gaussian-mixture density at a clipped action is a model proxy, not true support or clipped-policy density.',
            'Large own-Q gains, head gaps or lower fitted density cannot establish true rank error or residual/propagation causality.',
            'One Ant seed pilot only; not part of either locked fresh ensemble confirmation.'])
    (args.output / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status='complete', models=3, rows=6)), flush=True)


if __name__ == '__main__':
    main()
