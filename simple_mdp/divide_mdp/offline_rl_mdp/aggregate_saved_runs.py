from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .env import TwoPathBranchingMDP, evaluate_policy, optimal_policy_and_return
from .utils import save_json


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_saved_runs(root_path: str, seed_start: int, seed_end: int) -> Dict[str, Any]:
    if seed_end < seed_start:
        raise ValueError('seed_end must be >= seed_start')

    root = Path(root_path)
    per_seed: List[Dict[str, Any]] = []
    missing_seeds: List[int] = []

    for seed in range(seed_start, seed_end + 1):
        save_dir = root / str(seed)
        env_path = save_dir / 'env.json'
        policy_path = save_dir / 'greedy_policy.json'
        algo_cfg_path = save_dir / 'algorithm_config.json'

        if not (env_path.exists() and policy_path.exists() and algo_cfg_path.exists()):
            missing_seeds.append(seed)
            continue

        env_cfg = _load_json(env_path)
        policy_obj = _load_json(policy_path)
        algo_cfg = _load_json(algo_cfg_path)

        env = TwoPathBranchingMDP(
            short_horizon=int(env_cfg['short_horizon']),
            branch_num_actions=int(env_cfg['branch_num_actions']),
            reward_value=float(env_cfg.get('reward_value', 1.0)),
        )
        gamma = float(algo_cfg.get('gamma', 1.0))
        policy = [int(x) for x in policy_obj['policy']]
        eval_result = evaluate_policy(env, lambda s, p=policy: int(p[s]), gamma=gamma)
        optimal_policy, optimal_return, _ = optimal_policy_and_return(env, gamma=gamma)

        learned_return = float(eval_result['discounted_return'])
        seed_result = {
            'seed': seed,
            'gamma': gamma,
            'learned_return': learned_return,
            'optimal_return': float(optimal_return),
            'return_gap': float(optimal_return - learned_return),
            'chosen_branch': str(eval_result['chosen_branch']),
            'chose_short_path': bool(eval_result['chose_short_path']),
            'selected_optimal_path': bool(eval_result['selected_optimal_path']),
            'optimal_branch': str(eval_result['optimal_branch']),
            'short_path_return': float(eval_result['short_path_return']),
            'long_path_return': float(eval_result['long_path_return']),
            'policy': policy,
            'optimal_policy': [int(x) for x in optimal_policy.tolist()],
        }
        per_seed.append(seed_result)

    requested = seed_end - seed_start + 1
    loaded = len(per_seed)
    if loaded == 0:
        raise FileNotFoundError(
            f'No valid saved runs found under {root} for seeds {seed_start}..{seed_end}. '
            'Each seed directory must contain env.json, greedy_policy.json, and algorithm_config.json.'
        )

    mean_learned_return = sum(x['learned_return'] for x in per_seed) / loaded
    mean_optimal_return = sum(x['optimal_return'] for x in per_seed) / loaded
    mean_return_gap = sum(x['return_gap'] for x in per_seed) / loaded
    short_prob = sum(1.0 if x['chose_short_path'] else 0.0 for x in per_seed) / loaded
    optimal_prob = sum(1.0 if x['selected_optimal_path'] else 0.0 for x in per_seed) / loaded

    summary = {
        'requested_num_seeds': requested,
        'loaded_num_seeds': loaded,
        'seed_start': seed_start,
        'seed_end': seed_end,
        'missing_seeds': missing_seeds,
        'mean_learned_return': float(mean_learned_return),
        'mean_optimal_return': float(mean_optimal_return),
        'mean_return_gap': float(mean_return_gap),
        'short_path_selection_probability': float(short_prob),
        'optimal_path_selection_probability': float(optimal_prob),
    }

    out = {
        'root_path': str(root),
        'summary': summary,
        'per_seed': per_seed,
    }

    out_path = root / f'aggregate__seed_{seed_start}_to_{seed_end}.json'
    save_json(out_path, out)
    return {
        'summary': summary,
        'per_seed': per_seed,
        'artifacts': {
            'aggregate_summary_json': str(out_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Aggregate saved runs from root/{seed}/ directories.')
    parser.add_argument('--root', type=str, required=True, help='Root directory that contains per-seed subdirectories.')
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--seed-end', type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate_saved_runs(root_path=args.root, seed_start=args.seed_start, seed_end=args.seed_end)
    summary = result['summary']
    print('\n=== Aggregated Saved Runs ===')
    print(f"Loaded seeds                : {summary['loaded_num_seeds']}/{summary['requested_num_seeds']}")
    print(f"Seed range                  : {summary['seed_start']}..{summary['seed_end']}")
    print(f"Mean learned return         : {summary['mean_learned_return']:.6f}")
    print(f"Mean optimal return         : {summary['mean_optimal_return']:.6f}")
    print(f"Mean return gap             : {summary['mean_return_gap']:.6f}")
    print(f"Short path selection prob   : {summary['short_path_selection_probability']:.6f}")
    print(f"Optimal path selection prob : {summary['optimal_path_selection_probability']:.6f}")
    print(f"Saved to                    : {result['artifacts']['aggregate_summary_json']}")
    if summary['missing_seeds']:
        print(f"Missing seeds               : {summary['missing_seeds']}")


if __name__ == '__main__':
    main()
