# check_ogbench_dataset_size.py
import argparse
import numpy as np
import ogbench


def dataset_len(dataset):
    """Return transition/sample count from an OGBench dataset."""
    if "actions" in dataset:
        return len(dataset["actions"])
    if "observations" in dataset:
        return len(dataset["observations"])
    first_key = next(iter(dataset.keys()))
    return len(dataset[first_key])


def print_dataset_info(name, dataset):
    n = dataset_len(dataset)

    print(f"\n[{name}]")
    print(f"num samples / transitions: {n:,}")
    print("keys and shapes:")

    for k, v in dataset.items():
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"  {k:18s} shape={shape}, dtype={dtype}")

    if "terminals" in dataset:
        num_terminals = int(np.asarray(dataset["terminals"]).sum())
        print(f"sum(terminals): {num_terminals:,}")

        if num_terminals > 0:
            approx_traj_len = n / num_terminals
            print(f"approx # trajectories: {num_terminals:,}")
            print(f"approx trajectory length: {approx_traj_len:.2f}")

    if "masks" in dataset:
        num_success_like = int((1 - np.asarray(dataset["masks"])).sum())
        print(f"sum(1 - masks): {num_success_like:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        # default="antmaze-large-navigate-singletask-v0",
        # default="cube-octuple-play-singletask-v0",
        # default="puzzle-4x5-play-singletask-v0",
        # default="puzzle-4x6-play-singletask-v0",
        default="humanoidmaze-giant-navigate-singletask-v0",
        help="OGBench environment/dataset name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="~/.ogbench/data",
        help="Directory where OGBench datasets are stored/downloaded",
    )
    parser.add_argument(
        "--compact_dataset",
        action="store_true",
        help="Use compact dataset format without next_observations",
    )
    args = parser.parse_args()

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        args.env_name,
        dataset_dir=args.dataset_dir,
        compact_dataset=args.compact_dataset,
    )

    print(f"env_name: {args.env_name}")
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    print_dataset_info("train_dataset", train_dataset)
    print_dataset_info("val_dataset", val_dataset)


if __name__ == "__main__":
    main()