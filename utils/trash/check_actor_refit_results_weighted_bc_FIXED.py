import argparse
import os
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np


DATASET_ALIAS = {
    "medium-v2": "M",
    "medium-replay-v2": "MR",
    "medium-expert-v2": "ME",
    "expert-v2": "E",
    "full-replay-v2": "FR",
    "random-v2": "R",

    "umaze-v2": "U",
    "umaze-diverse-v2": "UD",
    "medium-play-v2": "MP",
    "medium-diverse-v2": "MD",
    "large-play-v2": "LP",
    "large-diverse-v2": "LD",
}


ArgGrid = List[Tuple[str, List[Any]]]
ArgSetting = Dict[str, Any]


def build_env_name(env_id: str, dataset_name: str) -> str:
    return f"{env_id}-{dataset_name}"


def normalize_metric_key(metric_name: str) -> str:
    if metric_name.endswith(".npy"):
        metric_name = metric_name[:-4]
    return metric_name


def resolve_metric_key(data: np.lib.npyio.NpzFile, metric_name: str) -> str:
    """
    fit_eval_logs.npz 안의 metric key를 robust하게 찾는다.

    예:
      metric_name = "best_d4rl_normalized_score_mean"

    실제 저장 key 후보:
      "best_d4rl_normalized_score_mean"
      "actor_refit/best_d4rl_normalized_score_mean"
      "fit_actor/best_d4rl_normalized_score_mean"
    """
    metric_name = normalize_metric_key(metric_name)
    available_keys = list(data.keys())

    candidates = [
        metric_name,
        f"actor_refit/{metric_name}",
        f"fit_actor/{metric_name}",
        f"post_training/fit_actor/{metric_name}",
    ]

    for key in candidates:
        if key in data:
            return key

    suffix_matches = [key for key in available_keys if key.endswith(metric_name)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise KeyError(
        f"Could not find metric '{metric_name}'.\n"
        f"Available keys:\n{available_keys}"
    )


def load_npz_metric(npz_path: str, metric_name: str) -> float:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        key = resolve_metric_key(data, metric_name)
        value = np.asarray(data[key])

    if value.size == 0:
        raise ValueError(f"Empty metric '{metric_name}' in {npz_path}")

    value = value.reshape(-1)

    if value.size == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
        value = np.asarray(value[0]).reshape(-1)

    if value.size == 1:
        return float(value[0])

    return float(np.mean(value))


def iter_arg_settings(arg_grid: ArgGrid) -> Iterable[ArgSetting]:
    names = [name for name, _ in arg_grid]
    value_lists = [values for _, values in arg_grid]

    for values in product(*value_lists):
        yield dict(zip(names, values))


def setting_to_path_args(setting: ArgSetting) -> List[Any]:
    """
    arg_grid에 넣은 순서대로 root_path 아래 경로를 만든다.

    현재 실행 커맨드 기준:
      --load_model logs/tuning/cdaf_jax_coverage_margin/increased_discount/${first_arg}/${third_arg}

    따라서 arg_grid는 보통 다음처럼 둔다.
      [("first", [...]), ("third", [...])]
    """
    return list(setting.values())


def format_setting_label(setting: ArgSetting, lr: Any, bc_coef: Any) -> str:
    setting_part = ", ".join(
        f"{name}={format_path_value(value)}" for name, value in setting.items()
    )
    lr_text = format_path_value(lr)
    bc_text = format_path_value(bc_coef)
    if setting_part:
        return f"{setting_part}, lr={lr_text}, bc={bc_text}"
    return f"lr={lr_text}, bc={bc_text}"


def format_path_value(value: Any) -> str:
    """
    Path component formatter.

    Python str(0.00005) becomes '5e-05', but your bash-created folders use
    decimal notation such as '0.00005'. Keep decimal notation for floats.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return np.format_float_positional(float(value), trim="0")
    return str(value)


def build_refit_npz_path(
    root_path: str,
    setting: ArgSetting,
    env_name: str,
    seed: int,
    lr: Any,
    bc_coef: Any,
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
    actor_refit_root_dir_name: str = "actor_refit",
) -> str:
    """
    현재 refit 커맨드가 만드는 경로:

      root_path/{first}/{third}/{algo_name}-{env}/{seed}/
      actor_refit/weighted_bc/{lr}/{bc_coef}/fit_eval_logs.npz
    """
    path_args = setting_to_path_args(setting)

    return os.path.join(
        root_path,
        *(format_path_value(arg) for arg in path_args),
        f"{algo_name}-{env_name}",
        str(seed),
        actor_refit_root_dir_name,
        actor_fit_method,
        format_path_value(lr),
        format_path_value(bc_coef),
        "fit_eval_logs.npz",
    )


def calculate_mean_std_for_setting(
    root_path: str,
    setting: ArgSetting,
    env_name: str,
    lr: Any,
    bc_coef: Any,
    seeds: List[int],
    metric_name: str,
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
    missing_ok: bool = True,
) -> Tuple[float, float, int]:
    values = []

    for seed in seeds:
        npz_path = build_refit_npz_path(
            root_path=root_path,
            setting=setting,
            env_name=env_name,
            seed=seed,
            lr=lr,
            bc_coef=bc_coef,
            algo_name=algo_name,
            actor_fit_method=actor_fit_method,
        )

        try:
            value = load_npz_metric(npz_path, metric_name)
            values.append(value)
        except Exception as e:
            if missing_ok:
                print(f"[Warning] Skip missing/invalid result: {npz_path}")
                print(f"          Reason: {e}")
            else:
                raise e

    if len(values) == 0:
        return np.nan, np.nan, 0

    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values)), len(values)


def generate_header(env_id_list: List[str], dataset_list: List[str]) -> str:
    header = "Setting & "
    for env_id in env_id_list:
        for dataset_name in dataset_list:
            alias = DATASET_ALIAS.get(dataset_name, dataset_name)
            header += f"{env_id}-{alias} & "
    header += "Avg \\\\" 
    return header


def generate_result_row(
    root_path: str,
    env_id_list: List[str],
    dataset_list: List[str],
    setting: ArgSetting,
    lr: Any,
    bc_coef: Any,
    seeds: List[int],
    metric_name: str,
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
) -> str:
    label = format_setting_label(setting, lr, bc_coef)
    row = f"{label} & "

    cell_means = []

    for env_id in env_id_list:
        for dataset_name in dataset_list:
            env_name = build_env_name(env_id, dataset_name)

            mean_value, std_value, n_found = calculate_mean_std_for_setting(
                root_path=root_path,
                setting=setting,
                env_name=env_name,
                lr=lr,
                bc_coef=bc_coef,
                seeds=seeds,
                metric_name=metric_name,
                algo_name=algo_name,
                actor_fit_method=actor_fit_method,
                missing_ok=True,
            )

            if n_found == 0:
                row += "NA & "
            else:
                row += f"{mean_value:.2f} ± {std_value:.2f} & "
                cell_means.append(mean_value)

    if len(cell_means) == 0:
        row += "NA \\\\"
    else:
        row += f"{np.mean(cell_means):.2f} \\\\"

    return row


def find_best_refit_per_cell(
    root_path: str,
    env_id_list: List[str],
    dataset_list: List[str],
    arg_grid: ArgGrid,
    lr_list: List[Any],
    bc_coef_list: List[Any],
    seeds: List[int],
    metric_name: str,
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    best_results = {}

    for env_id in env_id_list:
        for dataset_name in dataset_list:
            env_name = build_env_name(env_id, dataset_name)
            best_info: Optional[Dict[str, Any]] = None

            for setting in iter_arg_settings(arg_grid):
                for lr in lr_list:
                    for bc_coef in bc_coef_list:
                        mean_value, std_value, n_found = calculate_mean_std_for_setting(
                            root_path=root_path,
                            setting=setting,
                            env_name=env_name,
                            lr=lr,
                            bc_coef=bc_coef,
                            seeds=seeds,
                            metric_name=metric_name,
                            algo_name=algo_name,
                            actor_fit_method=actor_fit_method,
                            missing_ok=True,
                        )

                        if n_found == 0:
                            continue

                        if best_info is None or mean_value > best_info["mean"]:
                            best_info = {
                                "mean": mean_value,
                                "std": std_value,
                                "n_found": n_found,
                                "setting": setting.copy(),
                                "lr": lr,
                                "bc_coef": bc_coef,
                            }

            if best_info is None:
                best_info = {
                    "mean": np.nan,
                    "std": np.nan,
                    "n_found": 0,
                    "setting": {name: None for name, _ in arg_grid},
                    "lr": None,
                    "bc_coef": None,
                }

            best_results[(env_id, dataset_name)] = best_info

    return best_results


def generate_best_row(
    env_id_list: List[str],
    dataset_list: List[str],
    best_results: Dict[Tuple[str, str], Dict[str, Any]],
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
) -> str:
    row = f"{algo_name}-{actor_fit_method}-refit-oracle-best & "
    cell_means = []

    for env_id in env_id_list:
        for dataset_name in dataset_list:
            info = best_results[(env_id, dataset_name)]

            if info["n_found"] == 0:
                row += "NA & "
            else:
                row += f"{info['mean']:.2f} ± {info['std']:.2f} & "
                cell_means.append(info["mean"])

    if len(cell_means) == 0:
        row += "NA \\\\"
    else:
        row += f"{np.mean(cell_means):.2f} \\\\"

    return row


def print_best_param_summary(
    env_id_list: List[str],
    dataset_list: List[str],
    best_results: Dict[Tuple[str, str], Dict[str, Any]],
    algo_name: str = "CDAF-JAX",
    actor_fit_method: str = "weighted_bc",
) -> None:
    print(f"\n[Best actor-refit setting per env-dataset] algo_name={algo_name}, actor_fit_method={actor_fit_method}")
    for env_id in env_id_list:
        for dataset_name in dataset_list:
            info = best_results[(env_id, dataset_name)]
            setting_text = ", ".join(
                f"{name}={value}" for name, value in info["setting"].items()
            )
            print(
                f"{env_id}-{dataset_name}: "
                f"{setting_text}, "
                f"lr={info['lr']}, "
                f"bc_coef={info['bc_coef']}, "
                f"mean={info['mean']:.2f}, "
                f"std={info['std']:.2f}, "
                f"n={info['n_found']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="CDAF-JAX")
    parser.add_argument("--root_path", type=str, default="logs/tuning/cdaf_jax_coverage_margin/increased_discount")
    parser.add_argument("--actor_fit_method", type=str, default="weighted_bc")
    parser.add_argument("--metric_name", type=str, default="best_d4rl_normalized_score_mean")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    return parser.parse_args()


def main():
    args = parse_args()

    algo_name = args.algo_name
    root_path = args.root_path
    actor_fit_method = args.actor_fit_method
    metric_name = args.metric_name
    seeds = args.seeds

    env_id_list = ["antmaze"]
    dataset_list = [
        "large-play-v2",
    ]

    # 현재 실행 커맨드 기준 load_model 경로:
    # logs/tuning/cdaf_jax_coverage_margin/increased_discount/${first_arg}/${third_arg}
    # 따라서 second가 경로에 없다면 arg_grid에 넣으면 안 된다.
    arg_grid: ArgGrid = [
        ("first", [2.0]),
        ("third", [1000]),
    ]

    lr_list = [0.00001, 0.00005, 0.0001, 0.0003]
    bc_coef_list = [0.2, 0.4, 0.6, 0.8, 1.0]

    print(generate_header(env_id_list, dataset_list))

    for setting in iter_arg_settings(arg_grid):
        for lr in lr_list:
            for bc_coef in bc_coef_list:
                print(
                    generate_result_row(
                        root_path=root_path,
                        env_id_list=env_id_list,
                        dataset_list=dataset_list,
                        setting=setting,
                        lr=lr,
                        bc_coef=bc_coef,
                        seeds=seeds,
                        metric_name=metric_name,
                        algo_name=algo_name,
                        actor_fit_method=actor_fit_method,
                    )
                )

    best_results = find_best_refit_per_cell(
        root_path=root_path,
        env_id_list=env_id_list,
        dataset_list=dataset_list,
        arg_grid=arg_grid,
        lr_list=lr_list,
        bc_coef_list=bc_coef_list,
        seeds=seeds,
        metric_name=metric_name,
        algo_name=algo_name,
        actor_fit_method=actor_fit_method,
    )

    print(generate_best_row(env_id_list, dataset_list, best_results, algo_name=algo_name, actor_fit_method=actor_fit_method))
    print_best_param_summary(env_id_list, dataset_list, best_results, algo_name=algo_name, actor_fit_method=actor_fit_method)


if __name__ == "__main__":
    main()
