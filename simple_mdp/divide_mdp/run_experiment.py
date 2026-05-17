from __future__ import annotations

import argparse

from offline_rl_mdp.experiment import run_experiment


ALGOS = ["clipped_q_learning", "monte_carlo", "frozen_ratio_value_learning"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RL on a two-path branching MDP")
    parser.add_argument("--algo", type=str, choices=ALGOS, required=True)
    parser.add_argument("--N", type=int, required=True, help="Short path horizon. Long path horizon becomes 2N.")
    parser.add_argument(
        "--A",
        "--branch-actions",
        dest="branch_actions",
        type=int,
        default=2,
        help="Number of equivalent actions available on branch states.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="./results")
    parser.add_argument(
        "--reward-value",
        type=float,
        default=1.0,
        help="Reward received when entering the shared terminal.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Use gamma < 1 if you want the short path to be uniquely optimal.",
    )
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--update-mode", type=str, default="sample", choices=["sample", "minibatch"])
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise injected into the entire online Q table after every mini-batch update.",
    )
    parser.add_argument("--target-update-freq", type=int, default=1)
    parser.add_argument("--polyak-tau", type=float, default=1.0)
    parser.add_argument("--init-std", type=float, default=1e-3)
    parser.add_argument("--policy-extraction", type=str, default="mean", choices=["mean", "min"])
    parser.add_argument("--fixed-update-freq", type=int, default=10)
    parser.add_argument("--fixed-polyak-tau", type=float, default=1.0)
    parser.add_argument("--value-learning-rate", type=float, default=None)
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--min-update-weight", type=float, default=0.0)
    parser.add_argument("--max-update-weight", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_kwargs = {
        "gamma": args.gamma,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "update_mode": args.update_mode,
        "seed": args.seed,
        "reward_value": args.reward_value,
        "init_std": args.init_std,
    }

    if args.algo == "clipped_q_learning":
        algo_kwargs = {
            **base_kwargs,
            "num_q": args.num_q,
            "policy_extraction": args.policy_extraction,
            "noise_std": args.noise_std,
            "target_update_freq": args.target_update_freq,
            "polyak_tau": args.polyak_tau,
        }
    elif args.algo == "monte_carlo":
        algo_kwargs = {
            **base_kwargs,
            "num_q": args.num_q,
            "policy_extraction": args.policy_extraction,
        }
    elif args.algo == "frozen_ratio_value_learning":
        algo_kwargs = {
            **base_kwargs,
            "noise_std": args.noise_std,
            "fixed_update_freq": args.fixed_update_freq,
            "fixed_polyak_tau": args.fixed_polyak_tau,
            "value_learning_rate": args.value_learning_rate,
            "weight_temperature": args.weight_temperature,
            "min_update_weight": args.min_update_weight,
            "max_update_weight": args.max_update_weight,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    result = run_experiment(
        short_horizon=args.N,
        branch_num_actions=args.branch_actions,
        algo_name=args.algo,
        save_path=args.save_path,
        **algo_kwargs,
    )
    summary = result["summary"]
    print("\n=== Final Evaluation ===")
    print(f"Chosen branch  : {summary['chosen_branch']}")
    print(f"Learned return : {summary['learned_return']:.6f}")
    print(f"Optimal return : {summary['optimal_return']:.6f}")
    print(f"Return gap     : {summary['return_gap']:.6f}")
    print(f"Saved to       : {result['save_dir']}")


if __name__ == "__main__":
    main()
