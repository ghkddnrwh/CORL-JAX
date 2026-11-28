"""Evaluate one exported actor with the established four-core CPU protocol."""
import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import ogbench

from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_actor_comparison_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import atomic, sha
from utils.codex.ukjo1.error_propagation.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.codex.ukjo1.error_propagation.propagation_research.seeded_reset import prepare_seeded_action_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(d.platform == 'cpu' for d in jax.devices())
    if not args.smoke:
        assert args.step in (800000, 900000, 1000000) and args.episodes == 50
    config = json.loads((args.run/'config.json').read_text())
    for path, expected in config['source_sha256'].items():
        assert sha(path) == expected, path
    artifact_path = args.run/f'actor_{args.actor}_{args.step}.pkl'
    with artifact_path.open('rb') as stream:
        artifact = pickle.load(stream)
    assert artifact['identity'] == config['identity'] and artifact['source_sha256'] == config['source_sha256']
    assert artifact['config_sha256'] == sha(args.run/'config.json')
    assert artifact['actor_updates'] == artifact['Q_updates'] == artifact['step'] == args.step
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(artifact['actor_params']))
    job = config['identity']['job']
    episode_seed = 10042 if job['stage'] == 'tuning' else 42
    env = ogbench.make_env_and_datasets(job['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, episode_seed)
    policy = core.make_actor_policy(artifact['metadata'])
    namespace = SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=artifact['actor_params'])))
    result = evaluate(namespace, env, policy, args.step, args.episodes, seed=episode_seed)
    env.close()
    result.update(environment=job['environment'], training_seed=job['seed'], stage=job['stage'],
        actor=args.actor, method=artifact['metadata']['method'], alpha=artifact['metadata']['alpha'],
        actor_artifact_sha256=sha(artifact_path), config_sha256=artifact['config_sha256'],
        checkpoint_sha256=artifact['checkpoint_sha256'], base_state_sha256=artifact['base_state_sha256'],
        evaluation_source_sha256=sha(__file__), actor_core_sha256=sha(core.__file__),
        no_optimizer_updates=True, reset_metadata=reset, deployment_metadata=artifact['metadata'],
        loaded_trees=['decoder_only' if artifact['metadata']['method'] == 'cvae' else 'actor_only'])
    if args.smoke:
        result.update(protocol='implementation_smoke_only', no_performance_claim=True)
    atomic(args.output, result)
    print(json.dumps(dict(actor=args.actor, step=args.step, success=result['success_rate'])), flush=True)


if __name__ == '__main__':
    main()
