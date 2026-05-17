from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .algorithms import build_algorithm
from .data import (
    build_branch_action_trajectory_buffer,
    buffer_state_action_coverage,
    format_action_table,
)
from .env import TwoPathBranchingMDP, evaluate_policy, optimal_policy_and_return
from .utils import plot_q_error_history, save_json


def run_experiment(
    *,
    short_horizon: int,
    branch_num_actions: int,
    algo_name: str,
    save_path: str,
    **algo_kwargs: Any,
) -> Dict[str, Any]:
    gamma = float(algo_kwargs.get("gamma", 1.0))
    reward_value = float(algo_kwargs.get("reward_value", 1.0))
    seed = int(algo_kwargs.get("seed", 0))

    env = TwoPathBranchingMDP(
        short_horizon=short_horizon,
        branch_num_actions=branch_num_actions,
        reward_value=reward_value,
    )
    replay_buffer = build_branch_action_trajectory_buffer(env)
    coverage = buffer_state_action_coverage(replay_buffer, env.num_states, env.num_actions)

    algo_builder_kwargs = dict(algo_kwargs)
    algo_builder_kwargs.pop("reward_value", None)
    optimal_policy, optimal_return, q_star = optimal_policy_and_return(env, gamma=gamma)

    algo = build_algorithm(algo_name=algo_name, env=env, **algo_builder_kwargs)
    algo.fit(replay_buffer, optimal_q=q_star)

    learned_policy = algo.greedy_policy()
    learned_eval = evaluate_policy(env, lambda s: int(learned_policy[s]), gamma=gamma)

    summary: Dict[str, Any] = {
        "env_name": "two_path_branching_mdp",
        "short_horizon": env.short_horizon,
        "branch_num_actions": env.branch_num_actions,
        "num_states": env.num_states,
        "num_actions": env.num_actions,
        "reward_value": env.reward_value,
        "coverage": coverage.tolist(),
        "optimal_policy": optimal_policy.tolist(),
        "optimal_return": float(optimal_return),
        "learned_policy": learned_policy.tolist(),
        "learned_return": float(learned_eval["discounted_return"]),
        "return_gap": float(optimal_return - float(learned_eval["discounted_return"])),
        "chosen_branch": str(learned_eval["chosen_branch"]),
        "chose_short_path": bool(learned_eval["chose_short_path"]),
        "selected_optimal_path": bool(learned_eval["selected_optimal_path"]),
        "optimal_branch": str(learned_eval["optimal_branch"]),
        "short_path_return": float(learned_eval["short_path_return"]),
        "long_path_return": float(learned_eval["long_path_return"]),
        "learned_actions": learned_eval["actions"],
        "learned_rewards": learned_eval["rewards"],
        "learned_states": learned_eval["states"],
        "learned_state_names": learned_eval["state_names"],
        "action_reward_table": format_action_table(env),
        "num_dataset_trajectories": len(replay_buffer.trajectories),
        "expected_num_dataset_trajectories": 2 * env.branch_num_actions,
        "seed": seed,
    }

    save_dir = Path(save_path) / str(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_json(save_dir / "env.json", env.as_dict())
    save_json(save_dir / "dataset.json", replay_buffer.to_dict())
    save_json(save_dir / "summary.json", summary)
    save_json(save_dir / "optimal_q.json", {"q_star": q_star.tolist()})
    algo.save(save_dir)
    plot_q_error_history(
        path=save_dir / "q_error_curve.png",
        history=algo.history,
        metric_key="q_rmse_valid",
    )

    return {
        "save_dir": str(save_dir),
        "summary": summary,
        "algorithm_config": algo.config_dict(),
        "artifacts": {
            "history_csv": str(save_dir / "train_history.csv"),
            "q_error_curve": str(save_dir / "q_error_curve.png"),
        },
    }
