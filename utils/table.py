import argparse
import os
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


ArgGrid = List[Tuple[str, List[Any]]]
ArgSetting = Dict[str, Any]
EnvNameList = List[str]


# 출력용 짧은 이름.
# 실제 로그 경로에는 env_name 원본을 그대로 사용하고, table header에만 이 alias를 사용한다.
ENV_NAME_ALIAS: Dict[str, str] = {
    "antmaze-large-navigate-singletask-v0": "AM-L",
    "humanoidmaze-medium-navigate-singletask-v0": "HM-M",
    "humanoidmaze-large-navigate-singletask-v0": "HM-L",
    "antsoccer-arena-navigate-singletask-v0": "AS-A",
    "cube-single-play-singletask-v0": "Cube-S",
    "cube-double-play-singletask-v0": "Cube-D",
    "puzzle-3x3-play-singletask-v0": "Pz-3x3",
    "puzzle-4x4-play-singletask-v0": "Pz-4x4",
    "scene-play-singletask-v0": "Scene",
}


def abbreviate_env_name(env_name: str) -> str:
    """
    출력용 환경 이름을 짧게 만든다.

    - ENV_NAME_ALIAS에 등록된 환경은 지정된 축약형 사용
    - 등록되지 않은 환경은 최대한 일반 규칙으로 축약
    - 중요한 점: 이 함수는 출력용으로만 사용하고, 경로 생성에는 원본 env_name을 사용한다.
    """
    if env_name in ENV_NAME_ALIAS:
        return ENV_NAME_ALIAS[env_name]

    short_name = env_name
    replacements = [
        ("-navigate-singletask-v0", "-nav"),
        ("-play-singletask-v0", "-play"),
        ("-singletask-v0", ""),
        ("humanoidmaze", "hm"),
        ("antmaze", "am"),
        ("antsoccer", "as"),
        ("puzzle", "pz"),
        ("large", "L"),
        ("medium", "M"),
        ("arena", "A"),
        ("single", "S"),
        ("double", "D"),
    ]
    for old, new in replacements:
        short_name = short_name.replace(old, new)
    return short_name


def format_env_name_list_for_info(env_name_list: EnvNameList) -> str:
    return "[" + ", ".join(abbreviate_env_name(env_name) for env_name in env_name_list) + "]"


def print_env_name_mapping(env_name_list: EnvNameList) -> None:
    print("[Info] env_name_aliases:")
    for env_name in env_name_list:
        print(f"  {abbreviate_env_name(env_name)} = {env_name}")


def normalize_metric_key(metric_name: str) -> str:
    """
    npz key가 "eval/success_rate.npy" 또는 "eval/success_rate" 둘 중 하나로
    저장되어 있을 수 있어서 둘 다 찾을 수 있게 normalize한다.
    """
    if metric_name.endswith(".npy"):
        return metric_name[:-4]
    return metric_name


def resolve_metric_key(data: np.lib.npyio.NpzFile, metric_name: str) -> str:
    """
    eval_logs.npz 안에서 metric key를 찾는다.

    기본 target:
      eval/success_rate.npy

    실제 저장 key 후보:
      eval/success_rate.npy
      eval/success_rate
    """
    available_keys = list(data.keys())
    normalized_metric_name = normalize_metric_key(metric_name)

    candidates = [
        metric_name,
        normalized_metric_name,
    ]

    # 중복 제거하면서 순서 유지
    candidates = list(dict.fromkeys(candidates))

    for key in candidates:
        if key in data:
            return key

    suffix_matches = [
        key for key in available_keys
        if key.endswith(metric_name) or key.endswith(normalized_metric_name)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise KeyError(
        f"Could not find metric '{metric_name}'.\n"
        f"Available keys:\n{available_keys}"
    )


def _unwrap_object_array(value: np.ndarray) -> np.ndarray:
    """
    np.savez 저장 방식에 따라 object array 안에 list/array가 한 번 감싸져 있을 수 있어서
    가능한 경우 풀어준다.
    """
    value = np.asarray(value)

    if value.dtype == object and value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, (list, tuple, np.ndarray)):
            return np.asarray(item)

    return value


def select_last_evaluations(value: np.ndarray, last_n_evals: int) -> np.ndarray:
    """
    success_rate array에서 마지막 last_n_evals개 evaluation만 선택한다.

    - last_n_evals <= 0: 전체 evaluation 사용
    - scalar metric: 그대로 사용
    - 1D array: value[-last_n_evals:]
    - 2D 이상 array: 첫 번째 axis를 evaluation axis로 보고 value[-last_n_evals:, ...]
    """
    value = _unwrap_object_array(np.asarray(value))

    if value.size == 0:
        raise ValueError("Empty metric array")

    # scalar 또는 scalar처럼 생긴 값이면 선택할 evaluation 축이 없으므로 그대로 사용
    if value.ndim == 0:
        return value

    # object array가 남아 있으면 float array로 바꿀 수 있는지 시도
    if value.dtype == object:
        try:
            value = value.astype(np.float64)
        except (TypeError, ValueError):
            value = np.asarray(value.tolist())

    if last_n_evals <= 0:
        return value

    if value.shape[0] == 0:
        raise ValueError("Empty first axis in metric array")

    # 요청한 개수가 실제 evaluation 개수보다 크면 가능한 전체를 사용
    n_select = min(last_n_evals, value.shape[0])
    return value[-n_select:, ...]


def load_npz_metric(
    npz_path: str,
    metric_name: str,
    last_n_evals: int = 1,
) -> float:
    """
    eval_logs.npz에서 metric을 읽어서 seed별 float 하나로 만든다.

    핵심:
      1. eval/success_rate.npy 또는 eval/success_rate key를 찾는다.
      2. success_rate 배열에서 마지막 last_n_evals개 evaluation만 선택한다.
      3. 선택된 값들을 평균내어 해당 seed의 대표값으로 만든다.
      4. 이후 seed별 대표값들로 mean/std를 계산한다.

    last_n_evals:
      - 1: 마지막 evaluation 하나만 사용
      - 5: 마지막 5개 evaluation 평균 사용
      - 0 또는 음수: 전체 evaluation 평균 사용
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        key = resolve_metric_key(data, metric_name)
        value = np.asarray(data[key])

    selected_value = select_last_evaluations(value, last_n_evals)

    if selected_value.size == 0:
        raise ValueError(
            f"Empty metric '{metric_name}' after selecting last_n_evals={last_n_evals} "
            f"in {npz_path}"
        )

    return float(np.mean(selected_value.astype(np.float64)))


def iter_arg_settings(arg_grid: ArgGrid) -> Iterable[ArgSetting]:
    if len(arg_grid) == 0:
        yield {}
        return

    names = [name for name, _ in arg_grid]
    value_lists = [values for _, values in arg_grid]

    for values in product(*value_lists):
        yield dict(zip(names, values))


def setting_to_path_args(setting: ArgSetting) -> List[Any]:
    """
    arg_grid에 넣은 순서대로 root_path 아래 경로를 만든다.

    예:
      arg_grid = [("first", [1.0]), ("third", [1000])]
      -> root_path/1/1000/{algo_name}-{env_name}/{seed}/eval_logs.npz

    만약 root_path 바로 아래에 {algo_name}-{env_name}/{seed}/eval_logs.npz가 있다면
    arg_grid = [] 로 두면 된다.
    """
    return list(setting.values())


def format_path_value(value: Any) -> str:
    """
    Path component formatter.

    Python str(0.00005)는 '5e-05'가 될 수 있지만,
    bash로 만든 폴더가 '0.00005' 같은 decimal notation이면 이쪽을 유지한다.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return np.format_float_positional(float(value), trim="0")
    return str(value)


def format_setting_label(setting: ArgSetting) -> str:
    if not setting:
        return "eval"
    return ", ".join(
        f"{name}={format_path_value(value)}" for name, value in setting.items()
    )


def build_eval_npz_path(
    root_path: str,
    setting: ArgSetting,
    env_name: str,
    seed: int,
    algo_name: str = "CDAF-JAX",
    eval_log_filename: str = "eval_logs.npz",
) -> str:
    """
    일반 eval log 경로:

      root_path/{arg_grid values...}/{algo_name}-{env_name}/{seed}/eval_logs.npz

    예:
      root_path/1/1000/CDAF-JAX-antmaze-large-diverse-v2/0/eval_logs.npz
    """
    path_args = setting_to_path_args(setting)

    return os.path.join(
        root_path,
        *(format_path_value(arg) for arg in path_args),
        f"{algo_name}-{env_name}",
        str(seed),
        eval_log_filename,
    )


def calculate_mean_std_for_setting(
    root_path: str,
    setting: ArgSetting,
    env_name: str,
    seeds: List[int],
    metric_name: str,
    last_n_evals: int,
    algo_name: str = "CDAF-JAX",
    eval_log_filename: str = "eval_logs.npz",
    missing_ok: bool = True,
) -> Tuple[float, float, int]:
    values = []

    for seed in seeds:
        npz_path = build_eval_npz_path(
            root_path=root_path,
            setting=setting,
            env_name=env_name,
            seed=seed,
            algo_name=algo_name,
            eval_log_filename=eval_log_filename,
        )

        try:
            value = load_npz_metric(
                npz_path=npz_path,
                metric_name=metric_name,
                last_n_evals=last_n_evals,
            )
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


def generate_header(env_name_list: EnvNameList) -> str:
    """
    env_name_list에 명시된 환경들의 출력용 축약 이름을 header로 사용한다.

    실제 로그 경로 탐색에는 원본 env_name을 그대로 사용하고,
    table 출력에서만 abbreviate_env_name(env_name)을 사용한다.
    """
    short_env_names = [abbreviate_env_name(env_name) for env_name in env_name_list]
    return "Setting & " + " & ".join(short_env_names) + " & Avg \\\\"


def generate_result_row(
    root_path: str,
    env_name_list: EnvNameList,
    setting: ArgSetting,
    seeds: List[int],
    metric_name: str,
    last_n_evals: int,
    algo_name: str = "CDAF-JAX",
    eval_log_filename: str = "eval_logs.npz",
) -> str:
    """
    env_name_list에 들어 있는 모든 환경에 대해 결과를 계산해서 하나의 row로 반환한다.
    """
    label = format_setting_label(setting)
    row = f"{label} & "

    cell_values = []
    cell_means = []

    for env_name in env_name_list:
        mean_value, std_value, n_found = calculate_mean_std_for_setting(
            root_path=root_path,
            setting=setting,
            env_name=env_name,
            seeds=seeds,
            metric_name=metric_name,
            last_n_evals=last_n_evals,
            algo_name=algo_name,
            eval_log_filename=eval_log_filename,
            missing_ok=True,
        )

        if n_found == 0:
            cell_values.append("NA")
        else:
            cell_values.append(
                f"{mean_value:.2f} ± {std_value:.2f} ({n_found}/{len(seeds)})"
            )
            cell_means.append(mean_value)

    row += " & ".join(cell_values)

    if len(cell_means) == 0:
        row += " & NA \\\\"
    else:
        row += f" & {np.mean(cell_means):.2f} \\\\"

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="ReBRAC-JAX")
    parser.add_argument(
        "--root_path",
        type=str,
        # default="logs/basic/iql_jax/ogbench/original/",
        # default="logs/basic/iql_jax/ogbench/0.999/",
        default="logs/iclr2027/basic/rebrac_jax/ogbench/unnormalize/original/",
    )
    parser.add_argument("--metric_name", type=str, default="eval/success_rate.npy")
    parser.add_argument("--eval_log_filename", type=str, default="eval_logs.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--last_n_evals",
        type=int,
        default=3,
        help=(
            "success_rate에서 마지막 몇 개 evaluation을 seed별 평균 대상으로 쓸지 정한다. "
            "1이면 마지막 evaluation 하나만 사용하고, 5이면 마지막 5개 평균을 사용한다. "
            "0 또는 음수이면 전체 evaluation 평균을 사용한다."
        ),
    )
    parser.add_argument(
        "--env_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "평가할 전체 환경 이름 리스트. "
            "예: --env_names cube-single-play-singletask-v0 scene-play-singletask-v0"
        ),
    )
    parser.add_argument(
        "--show_env_mapping",
        action="store_true",
        help="출력용 축약 이름이 어떤 원본 env_name에 대응되는지 함께 출력한다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    algo_name = args.algo_name
    root_path = args.root_path
    metric_name = args.metric_name
    eval_log_filename = args.eval_log_filename
    seeds = args.seeds
    last_n_evals = args.last_n_evals

    # env_id와 dataset_name을 따로 관리하지 않고,
    # 실제 log folder에 들어가는 env_name 전체를 하나의 리스트에서 관리한다.
    #
    # 경로는 다음처럼 만들어진다:
    #   root_path/{arg_grid values...}/{algo_name}-{env_name}/{seed}/eval_logs.npz
    env_name_list: EnvNameList = [
        "antmaze-large-navigate-singletask-v0",
        "humanoidmaze-medium-navigate-singletask-v0",
        "humanoidmaze-large-navigate-singletask-v0",
        "antsoccer-arena-navigate-singletask-v0",
        "cube-single-play-singletask-v0",
        "cube-double-play-singletask-v0",
        "puzzle-3x3-play-singletask-v0",
        "puzzle-4x4-play-singletask-v0",
        "scene-play-singletask-v0",
    ]

    # CLI에서 리스트가 주어지면 그 리스트를 우선 사용한다.
    # 예:
    #   python script.py \
    #     --env_names cube-single-play-singletask-v0 scene-play-singletask-v0
    if args.env_names is not None:
        env_name_list = args.env_names

    # 현재 경로 기준:
    #   root_path/{first}/{third}/{algo_name}-{env_name}/{seed}/eval_logs.npz
    # 만약 root_path 바로 아래에 {algo_name}-{env_name}/{seed}/eval_logs.npz가 있으면
    #   arg_grid = []
    # 로 두면 된다.
    arg_grid: ArgGrid = [
        # ("first", [1.0]),
        # ("third", [1000]),
    ]

    print(f"[Info] metric_name={metric_name}")
    print(f"[Info] eval_log_filename={eval_log_filename}")
    print(f"[Info] seeds={seeds}")
    print(f"[Info] last_n_evals={last_n_evals}")
    print(f"[Info] env_name_list={format_env_name_list_for_info(env_name_list)}")
    if args.show_env_mapping:
        print_env_name_mapping(env_name_list)
    print(generate_header(env_name_list))

    for setting in iter_arg_settings(arg_grid):
        print(
            generate_result_row(
                root_path=root_path,
                env_name_list=env_name_list,
                setting=setting,
                seeds=seeds,
                metric_name=metric_name,
                last_n_evals=last_n_evals,
                algo_name=algo_name,
                eval_log_filename=eval_log_filename,
            )
        )


if __name__ == "__main__":
    main()
