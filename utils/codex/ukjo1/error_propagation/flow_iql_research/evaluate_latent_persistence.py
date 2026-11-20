"""Frozen temporal-latent policy screen; never rewrites primary evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core, policy_core, benchmark_core, latent_persistence
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--hold-steps', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(); assert not args.output.exists()
    conf = json.loads((args.run/'config.json').read_text()); ident = conf['identity']
    for source, digest in conf['source_sha256'].items():
        assert sha(source) == digest, source
    cfg = dict(ident['config'])
    for field in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    module = benchmark_core if ident['kind'] == 'fql' else policy_core
    env = ogbench.make_env_and_datasets(ident['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = module.create(ident['seed'], jnp.zeros((1, env.observation_space.shape[-1])),
                             jnp.zeros((1, env.action_space.shape[-1])), cfg, ident['kind'])
    with args.checkpoint.open('rb') as file:
        saved = pickle.load(file)
    state = flax.serialization.from_state_dict(template, saved['state'])
    assert int(state.q_updates) == saved['step']
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    result = latent_persistence.evaluate(state, env, saved['step'], args.episodes,
                                         args.samples, args.hold_steps)
    comparison = None
    if args.reference is not None:
        assert args.hold_steps == 1
        reference = json.loads(args.reference.read_text())
        assert reference['checkpoint_sha256'] == sha(args.checkpoint)
        assert reference['selection_samples'] == args.samples
        comparison = {name: result[name] == reference[name]
                      for name in ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    result.update(kind='frozen_reactive_flow_latent_persistence_variant', time=time.time(),
        run=args.run.name, environment=ident['environment'], training_seed=ident['seed'],
        proposal_source='behavior', checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        config_sha256=sha(args.run/'config.json'), source_sha256=sha(__file__),
        policy_source_sha256=sha(latent_persistence.__file__),
        selection_source_sha256=sha(Path(latent_persistence.selection_v2.__file__)),
        hold1_episode_reference_comparison=comparison, reset_metadata=reset,
        primary_evaluations_modified=False,
        limits=['The chosen Gaussian latent is held; physical actions are recomputed from the current observation every step.',
                'H>1 changes the deployed policy and does not inherit a per-step iid-proposal dominance guarantee.',
                'Action-change and model-value metrics do not certify actual policy quality or a unique causal mechanism.',
                'Pilot results are not a nine-task or independent-training-seed conclusion.'])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2, allow_nan=False)); tmp.replace(args.output)
    print(json.dumps(dict(run=args.run.name, step=saved['step'], hold=args.hold_steps,
                         success=result['success_rate'], comparison=comparison)), flush=True)


if __name__ == '__main__':
    main()
