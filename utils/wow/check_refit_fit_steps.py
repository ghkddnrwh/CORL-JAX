import os
from typing import List, Dict, Any, Tuple

import numpy as np


def build_trial_npz_path(
    root_path: str,
    train_prefix: str,
    env_id: str,
    dataset_name: str,
    trial_id: int,
    algo_id: str,
    first_arg,
    second_arg,
) -> str:
    train_dir = f"{train_prefix}-{env_id}-{dataset_name}"
    return os.path.join(
        root_path,
        train_dir,
        str(trial_id),
        algo_id,
        str(first_arg),
        str(second_arg),
        "fit_eval_logs.npz",
    )


def load_fit_step_info(npz_path: str) -> Dict[str, Any]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing npz file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "fit_step" not in data:
            raise KeyError(
                f"Missing key 'fit_step' in {npz_path}\n"
                f"Available keys: {list(data.keys())}"
            )
        fit_step = np.asarray(data["fit_step"])

    if fit_step.size == 0:
        raise ValueError(f"Empty fit_step in {npz_path}")

    fit_step = fit_step.reshape(-1)

    return {
        "path": npz_path,
        "fit_step": fit_step,
        "length": int(len(fit_step)),
        "last": int(fit_step[-1]),
    }


def collect_fit_step_infos(
    root_path: str,
    train_prefix: str,
    env_id_list: List[str],
    dataset_list: List[str],
    num_trials: int,
    algo_id: str,
    first_arg,
    second_arg,
) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []

    for env_id in env_id_list:
        for dataset_name in dataset_list:
            for i in range(num_trials):
                trial_id = i
                npz_path = build_trial_npz_path(
                    root_path=root_path,
                    train_prefix=train_prefix,
                    env_id=env_id,
                    dataset_name=dataset_name,
                    trial_id=trial_id,
                    algo_id=algo_id,
                    first_arg=first_arg,
                    second_arg=second_arg,
                )
                info = load_fit_step_info(npz_path)
                info["env"] = env_id
                info["dataset"] = dataset_name
                info["trial_id"] = trial_id
                infos.append(info)

    return infos


def check_same_length_and_last(infos: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    lengths = [info["length"] for info in infos]
    lasts = [info["last"] for info in infos]

    same_length = len(set(lengths)) == 1
    same_last = len(set(lasts)) == 1
    return same_length, same_last


def print_check_result(
    algo_id: str,
    first_arg,
    second_arg,
    infos: List[Dict[str, Any]],
) -> None:
    same_length, same_last = check_same_length_and_last(infos)

    print("=" * 120)
    print(f"{algo_id}, {first_arg}, {second_arg}")
    print(f"same fit_step length : {same_length}")
    print(f"same fit_step last   : {same_last}")

    if infos:
        ref_len = infos[0]["length"]
        ref_last = infos[0]["last"]
        print(f"reference length     : {ref_len}")
        print(f"reference last       : {ref_last}")

    print("-" * 120)
    for info in infos:
        print(
            f"{info['env']}-{info['dataset']}, "
            f"trial={info['trial_id']}, "
            f"len={info['length']}, "
            f"last={info['last']}"
        )

    if not same_length or not same_last:
        print("-" * 120)
        print("Mismatch details:")
        lengths = [info["length"] for info in infos]
        lasts = [info["last"] for info in infos]
        unique_lengths = sorted(set(lengths))
        unique_lasts = sorted(set(lasts))
        print(f"unique lengths: {unique_lengths}")
        print(f"unique lasts  : {unique_lasts}")

    print("=" * 120)
    print()


def main():
    train_prefix = "TD3-DW"

    env_id_list = ["walker2d"]
    dataset_list = [
        "medium-v2",
        "medium-replay-v2",
        "medium-expert-v2",
        "expert-v2",
        "full-replay-v2",
        "random-v2",
    ]

    num_trials = 1

    first_arg_list = [2.5]
    second_arg_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    algo_configs = [
        ("td3_bc_actor_refit", "logs/orl_bias/analysis/td3_dw_bc/0.0/0.0/100/0.0"),
    ]

    for algo_id, root_path in algo_configs:
        for first_arg in first_arg_list:
            for second_arg in second_arg_list:
                infos = collect_fit_step_infos(
                    root_path=root_path,
                    train_prefix=train_prefix,
                    env_id_list=env_id_list,
                    dataset_list=dataset_list,
                    num_trials=num_trials,
                    algo_id=algo_id,
                    first_arg=first_arg,
                    second_arg=second_arg,
                )
                print_check_result(
                    algo_id=algo_id,
                    first_arg=first_arg,
                    second_arg=second_arg,
                    infos=infos,
                )


if __name__ == "__main__":
    main()
