"""Reactive flow actions with a selected Gaussian latent held for H steps.

This holds a latent, not a physical action. Every environment step uses the
current observation. The environment-step PRNG split sequence is unchanged;
unused intermediate keys are still consumed. H=1 recovers best-of-N flow.
"""
import random
import time
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import selection_v2


def make_functions(agent, samples):
    config = agent.config
    assert config.get('encoder') is None
    flow_name = 'actor_flow' if 'modules_actor_flow' in agent.network.params else 'actor_bc_flow'
    assert 'modules_'+flow_name in agent.network.params
    has_value = 'modules_value' in agent.network.params

    def boundary(params, observation, key):
        model = agent.replace(network=agent.network.replace(params=params))
        actions, q = selection_v2.proposals(model, observation, key, samples, 'behavior')
        index = jnp.argmax(q, axis=0)
        action_key, _ = jax.random.split(key)
        noises = jax.random.normal(action_key, (samples, config['action_dim']))
        v = model.network.select('value')(observation) if has_value else jnp.asarray(0.)
        return actions[index], noises[index], q[index], v

    def holding(params, observation, noise):
        model = agent.replace(network=agent.network.replace(params=params))
        observations = observation[None, ...]
        action = noise[None, ...]
        for i in range(config['flow_steps']):
            t = jnp.full((1, 1), i/config['flow_steps'])
            velocity = model.network.select(flow_name)(observations, action, t, is_encoded=True)
            action = action+velocity/config['flow_steps']
        action = jnp.clip(action, -1, 1)[0]
        q = model.network.select('critic')(observation, actions=action).min(0)
        v = model.network.select('value')(observation) if has_value else jnp.asarray(0.)
        return action, q, v

    return jax.jit(boundary, backend='cpu'), jax.jit(holding, backend='cpu'), has_value


def evaluate(state, env, step, episodes, samples=9, hold_steps=4, first_seed=42):
    assert samples > 0 and hold_steps > 0
    device = jax.devices('cpu')[0]
    params = jax.device_put(state.agent.network.params, device)
    boundary, holding, has_value = make_functions(state.agent, samples)
    rewards, successes, lengths = [], [], []
    changes, boundary_changes, internal_changes, slacks = [], [], [], []
    boundary_count = 0; total_steps = 0; start = time.monotonic()
    for episode in range(episodes):
        es = first_seed+episode
        np.random.seed(es); random.seed(es); env.action_space.seed(es)
        observation, _ = env.reset(seed=es)
        key = jax.device_put(jax.random.PRNGKey(es), device)
        noise = None; previous_action = None
        reward_sum = 0.; success = 0.; steps = 0
        while True:
            key, action_key = jax.random.split(key)
            is_boundary = steps % hold_steps == 0
            if is_boundary:
                action, noise, q, v = boundary(params, np.asarray(observation, np.float32), action_key)
                boundary_count += 1
            else:
                action, q, v = holding(params, np.asarray(observation, np.float32), noise)
            action = np.asarray(action).copy()
            if previous_action is not None:
                change = float(np.mean((action.astype(np.float64)-previous_action)**2))
                changes.append(change)
                (boundary_changes if is_boundary else internal_changes).append(change)
            if has_value:
                slacks.append(float(q-v))
            previous_action = action.astype(np.float64)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward; success = max(success, float(info.get('success', 0.)))
            steps += 1; total_steps += 1
            if terminated or truncated:
                break
        rewards.append(float(reward_sum)); successes.append(success); lengths.append(steps)
    rms = lambda x: float(np.sqrt(np.mean(x))) if x else None
    return dict(step=step, success_rate=float(np.mean(successes)), reward_mean=float(np.mean(rewards)),
        episode_successes=successes, episode_rewards=rewards, episode_lengths=lengths,
        n_episodes=episodes, eval_device='cpu', first_episode_seed=first_seed,
        protocol='last3_cpu50_independent_seeded_episodes_v1', seconds=time.monotonic()-start,
        selection_samples=samples, latent_hold_steps=hold_steps,
        total_environment_steps=total_steps, latent_reselection_decisions=boundary_count,
        flow_forward_proposals_per_environment_step=(samples*boundary_count+total_steps-boundary_count)/max(total_steps, 1),
        action_change_RMS=rms(changes), boundary_action_change_RMS=rms(boundary_changes),
        internal_action_change_RMS=rms(internal_changes),
        selected_Q_minus_V_mean=float(np.mean(slacks)) if slacks else None,
        selected_Q_below_V_fraction=float(np.mean(np.asarray(slacks) < 0)) if slacks else None)
