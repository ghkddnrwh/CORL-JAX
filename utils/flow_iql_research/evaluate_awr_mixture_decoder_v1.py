"""Frozen AWR-mixture decoder comparison; learning and critic are unchanged."""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research import awr_mixture_core_v1 as core
from utils.flow_iql_research import train_awr_mixture_v1 as original
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.rank_mean_bundle_v1 import sha
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


RULES = ('highest_probability_mean', 'component_Q', 'sample9_Q')


def make_selector(agent, rule):
    assert rule in RULES[1:]

    def select(params, observations, key):
        model = agent.replace(network=agent.network.replace(params=params))
        logits, means, logstd = model.network.select('actor')(observations)
        if rule == 'component_Q':
            actions = jnp.moveaxis(means, -2, 0)
        else:
            component_key, noise_key = jax.random.split(key)
            shape = (9, *observations.shape[:-1])
            component = jax.random.categorical(component_key, logits, shape=shape)
            index = jnp.broadcast_to(component[..., None, None], (*shape, 1, means.shape[-1]))
            mu = jnp.take_along_axis(jnp.broadcast_to(means, (9, *means.shape)), index, axis=-2)[..., 0, :]
            sd = jnp.take_along_axis(jnp.broadcast_to(logstd, (9, *logstd.shape)), index, axis=-2)[..., 0, :]
            actions = mu + jnp.exp(sd) * jax.random.normal(noise_key, mu.shape)
        actions = jnp.clip(actions, -1, 1)
        obs = jnp.broadcast_to(observations, (actions.shape[0], *observations.shape))
        q = model.network.select('critic')(obs, actions=actions).min(0)
        choice = q.argmax(0)
        index = jnp.broadcast_to(choice[None, ..., None], (1, *actions.shape[1:]))
        action = jnp.take_along_axis(actions, index, axis=0)[0]
        greedy = jnp.clip(core.greedy_component_mean(logits, means), -1, 1)
        greedy_q = model.network.select('critic')(observations, actions=greedy).min(0)
        chosen_q = jnp.take_along_axis(q, choice[None], axis=0)[0]
        return action, dict(selected_Q_minus_greedy=chosen_q - greedy_q,
            action_RMS_from_greedy=jnp.sqrt(jnp.mean((action - greedy) ** 2, axis=-1)),
            chosen_candidate=choice, Q_candidate_spread=q.std(0),
            action_saturation_fraction=jnp.mean((jnp.abs(action) >= .98).astype(jnp.float32), axis=-1))

    return jax.jit(select, backend='cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, choices=(800000, 900000, 1000000), required=True)
    parser.add_argument('--rule', choices=RULES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-reference', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    conf, saved, checkpoint = read(args.run, args.step)
    identity = conf['identity']
    assert identity['kind'] == 'awr_mixture' and identity['profile'] == 'bounded_v2'
    assert identity['config']['mixture_components'] == 5 and identity['config']['discount'] == .999
    assert identity['config']['expectile'] == .9 and int(saved['state']['v_updates']) == 2 * args.step
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(saved['state']))
    cfg = dict(identity['config'])
    for name in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[name] = tuple(cfg[name])
    env = ogbench.make_env_and_datasets(identity['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = core.create(identity['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
        jnp.zeros((1, env.action_space.shape[-1])), cfg, 'awr_mixture')
    state = flax.serialization.from_state_dict(template, saved['state'])
    metrics = []
    if args.rule == 'highest_probability_mean':
        policy = original.make_policy(state.agent)
    else:
        compiled = make_selector(state.agent, args.rule)

        def policy(params, observation, key):
            action, values = compiled(params, observation, key)
            metrics.append({k: float(v) for k, v in values.items()})
            return action

    result = original.evaluate(state, env, policy, args.step, 50)
    env.close()
    exact = None
    if args.self_reference:
        assert args.rule == 'highest_probability_mean'
        references = [json.loads(line) for line in (args.run / 'evaluations.jsonl').read_text().splitlines()]
        reference = next(r for r in references if r['step'] == args.step)
        exact = {field: result[field] == reference[field] for field in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(exact.values()), exact
    summary = {key: float(np.mean([r[key] for r in metrics])) for key in metrics[0]} if metrics else None
    if args.rule == 'component_Q':
        # The original deterministic candidate belongs to this very proposal set.
        minimum = min(r['selected_Q_minus_greedy'] for r in metrics)
        assert minimum >= -1e-3, minimum
    result.update(time=time.time(), kind='frozen_AWR_mixture_decoder_diagnostic',
        environment=identity['environment'], learner_seed=identity['seed'], rule=args.rule,
        candidate_count=5 if args.rule == 'component_Q' else 9 if args.rule == 'sample9_Q' else 1,
        config_sha256=sha(args.run / 'config.json'), checkpoint_sha256=sha(checkpoint),
        actual_Q_Adam_count=args.step, actual_V_Adam_count=2 * args.step,
        source_sha256=sha(__file__), self_reference_episode_arrays_exact=exact,
        own_visitation_model_diagnostics=summary, reset_metadata=reset,
        primary_scores_modified=False, learning_state_modified=False,
        limits=['Same retained actor/critic; only extraction changes. Pilot discovery, not fresh training confirmation.',
            'Own-model Q gains do not certify true advantage, action ranking or better return.',
            'Candidate5 enumerates existing component means; sample9 draws from the fitted mixture at temperature1.',
            'This diagnostic is not part of the locked three-critic BC-flow rank-mean candidate.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(args.output)
    print(json.dumps(dict(rule=args.rule, step=args.step, success=result['success_rate'], exact=exact)), flush=True)


if __name__ == '__main__':
    main()
