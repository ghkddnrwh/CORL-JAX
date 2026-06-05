import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_STEP_KEYS = (
    "timestep",
    "timesteps",
    "step",
    "steps",
    "global_step",
    "total_it",
    "iteration",
    "iterations",
    "epoch",
)

CONFIG_KEYS = ("max_timesteps", "n_timesteps", "eval_freq")


@dataclass
class ExpectedEval:
    final_step: Optional[int] = None
    num_evals: Optional[int] = None
    source: str = ""


@dataclass
class EvalLogInfo:
    path: Path
    exists: bool
    load_error: Optional[str] = None
    last_step: Optional[int] = None
    num_evals: Optional[int] = None
    keys: Tuple[str, ...] = ()


@dataclass
class ScanItem:
    run_dir: Path
    eval_info: EvalLogInfo
    expected: ExpectedEval





def is_number_like(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_seed_run_dir(path: Path) -> bool:
    return path.name.isdigit() and "-" in path.parent.name and not is_number_like(path.parent.name)


def iter_candidate_dirs(root: Path, filename: str, mode: str) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__"} and not name.startswith(".")
        ]

        path = Path(current_root)
        if path == root:
            continue

        if mode == "all":
            yield path
            continue

        if mode == "leaf":
            if len(dirnames) == 0:
                yield path
            continue

        file_set = set(filenames)
        has_run_marker = bool(
            {filename, "config.yaml", "checkpoint.pkl", "refit_config.yaml"} & file_set
        )
        if has_run_marker or len(dirnames) == 0 or is_seed_run_dir(path):
            yield path


def strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False
    for i, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return value[:i]
    return value


def parse_config_number(raw_value: str) -> Optional[int]:
    value = strip_yaml_comment(raw_value).strip()
    value = re.sub(r"^![^\s]+\s+", "", value)
    value = value.strip("'\"")

    if value.lower() in {"", "null", "none"}:
        return None

    try:
        parsed = float(value.replace("_", ""))
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None
    return int(parsed)


def read_config_values(config_path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    if not config_path.is_file():
        return values

    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
    with config_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if match is None:
                continue

            key, raw_value = match.groups()
            if key not in CONFIG_KEYS:
                continue

            parsed = parse_config_number(raw_value)
            if parsed is not None:
                values[key] = parsed

    return values


def expected_from_config(run_dir: Path) -> ExpectedEval:
    config_values = read_config_values(run_dir / "config.yaml")
    max_timesteps = config_values.get("max_timesteps", config_values.get("n_timesteps"))
    eval_freq = config_values.get("eval_freq")

    if max_timesteps is None:
        return ExpectedEval()

    if eval_freq is not None and eval_freq > 0:
        num_evals = max_timesteps // eval_freq
        final_step = num_evals * eval_freq
        if final_step == 0 and max_timesteps > 0:
            final_step = max_timesteps
            num_evals = 1
        return ExpectedEval(
            final_step=int(final_step),
            num_evals=int(num_evals),
            source=str(run_dir / "config.yaml"),
        )

    return ExpectedEval(
        final_step=int(max_timesteps),
        num_evals=None,
        source=str(run_dir / "config.yaml"),
    )


def normalize_npz_key(key: str) -> str:
    if key.endswith(".npy"):
        return key[:-4]
    return key


def resolve_step_key(keys: Sequence[str], step_keys: Sequence[str]) -> Optional[str]:
    normalized_to_key = {normalize_npz_key(key): key for key in keys}
    for key in step_keys:
        normalized = normalize_npz_key(key)
        if key in keys:
            return key
        if normalized in normalized_to_key:
            return normalized_to_key[normalized]

    suffix_matches = [
        key
        for key in keys
        if any(normalize_npz_key(key).endswith(normalize_npz_key(candidate)) for candidate in step_keys)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def unwrap_object_array(value: np.ndarray) -> np.ndarray:
    while value.dtype == object and value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, np.ndarray):
            value = np.asarray(item)
            continue
        if isinstance(item, (list, tuple)):
            value = np.asarray(item)
            continue
        break
    return value


def numeric_vector(value: np.ndarray) -> Optional[np.ndarray]:
    value = unwrap_object_array(np.asarray(value))
    if value.size == 0:
        return np.asarray([], dtype=np.float64)

    try:
        numeric = value.astype(np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None

    numeric = numeric[np.isfinite(numeric)]
    return numeric


def extract_steps_from_arr0(value: np.ndarray, step_keys: Sequence[str]) -> Optional[np.ndarray]:
    value = unwrap_object_array(np.asarray(value))
    items = value.reshape(-1)
    steps: List[float] = []

    for item in items:
        if not isinstance(item, dict):
            return None

        key = resolve_step_key(tuple(str(k) for k in item.keys()), step_keys)
        if key is None:
            return None

        try:
            steps.append(float(item[key]))
        except (TypeError, ValueError):
            return None

    return np.asarray(steps, dtype=np.float64)


def load_eval_log_info(run_dir: Path, filename: str, step_keys: Sequence[str]) -> EvalLogInfo:
    eval_path = run_dir / filename
    if not eval_path.is_file():
        return EvalLogInfo(path=eval_path, exists=False)

    try:
        with np.load(eval_path, allow_pickle=True) as data:
            keys = tuple(data.keys())
            step_key = resolve_step_key(keys, step_keys)

            if step_key is not None:
                steps = numeric_vector(np.asarray(data[step_key]))
            elif "arr_0" in data:
                steps = extract_steps_from_arr0(np.asarray(data["arr_0"]), step_keys)
            else:
                steps = None

            if steps is None:
                return EvalLogInfo(
                    path=eval_path,
                    exists=True,
                    load_error=f"missing step key; available keys={list(keys)}",
                    keys=keys,
                )
            if steps.size == 0:
                return EvalLogInfo(
                    path=eval_path,
                    exists=True,
                    load_error="empty step array",
                    keys=keys,
                )

            return EvalLogInfo(
                path=eval_path,
                exists=True,
                last_step=int(steps[-1]),
                num_evals=int(steps.size),
                keys=keys,
            )
    except Exception as exc:
        return EvalLogInfo(
            path=eval_path,
            exists=True,
            load_error=f"{type(exc).__name__}: {exc}",
        )


def build_scan_items(args: argparse.Namespace, step_keys: Sequence[str]) -> List[ScanItem]:
    root = args.root
    items: List[ScanItem] = []

    for run_dir in iter_candidate_dirs(root, args.filename, args.candidate_mode):
        expected = expected_from_config(run_dir)
        if args.expected_final_timestep is not None:
            expected.final_step = args.expected_final_timestep
            expected.source = "CLI --expected-final-timestep"
        if args.expected_num_evals is not None:
            expected.num_evals = args.expected_num_evals

        eval_info = load_eval_log_info(run_dir, args.filename, step_keys)
        items.append(ScanItem(run_dir=run_dir, eval_info=eval_info, expected=expected))

    return items


def fill_global_expected(items: List[ScanItem]) -> None:
    configured_final_steps = {
        item.expected.final_step
        for item in items
        if item.expected.final_step is not None and item.expected.source
    }

    fallback_final_step: Optional[int] = None
    source = ""
    if len(configured_final_steps) == 1:
        fallback_final_step = configured_final_steps.pop()
        source = "unique config.yaml value under root"
    else:
        observed_steps = [
            item.eval_info.last_step
            for item in items
            if item.eval_info.last_step is not None
        ]
        if observed_steps:
            fallback_final_step = max(observed_steps)
            source = "maximum observed timestep under root"

    if fallback_final_step is None:
        return

    for item in items:
        if item.expected.final_step is None:
            item.expected.final_step = fallback_final_step
            item.expected.source = source


def incomplete_reason(item: ScanItem) -> Optional[str]:
    eval_info = item.eval_info
    expected = item.expected

    if not eval_info.exists:
        return "missing eval_logs.npz"
    if eval_info.load_error is not None:
        return eval_info.load_error

    if (
        expected.final_step is not None
        and eval_info.last_step is not None
        and eval_info.last_step < expected.final_step
    ):
        return f"last_step={eval_info.last_step} < expected_final_step={expected.final_step}"

    if (
        expected.num_evals is not None
        and eval_info.num_evals is not None
        and eval_info.num_evals < expected.num_evals
    ):
        return f"num_evals={eval_info.num_evals} < expected_num_evals={expected.num_evals}"

    return None


def format_path(path: Path, root: Path, relative: bool) -> str:
    if not relative:
        return str(path)

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively print run directories whose eval_logs.npz is missing, "
            "empty/corrupt, or does not reach the expected final evaluation step."
        )
    )
    parser.add_argument(
        "--root",
        default="logs/tuning",
        type=Path,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="eval_logs.npz",
        help="Evaluation log filename to check.",
    )
    parser.add_argument(
        "--expected-final-timestep",
        type=int,
        default=None,
        help=(
            "Expected final timestep. If omitted, each config.yaml is used first; "
            "otherwise the maximum observed timestep is used as a fallback."
        ),
    )
    parser.add_argument(
        "--expected-num-evals",
        type=int,
        default=None,
        help="Expected number of evaluation rows. Optional extra strictness.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("auto", "leaf", "all"),
        default="auto",
        help=(
            "Which directories should be treated as run directories. "
            "'auto' checks leaf dirs and dirs with run markers such as config.yaml."
        ),
    )
    parser.add_argument(
        "--step-key",
        action="append",
        default=[],
        help=(
            "NPZ key to use as the training step. Can be passed multiple times. "
            "Defaults include timestep/step/global_step/etc."
        ),
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Print paths relative to the scan root.",
    )
    parser.add_argument(
        "--show-reason",
        action="store_true",
        help="Print the reason after each path.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = args.root

    if not root.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root}")

    step_keys = tuple(dict.fromkeys([*args.step_key, *DEFAULT_STEP_KEYS]))
    items = build_scan_items(args, step_keys)

    if args.expected_final_timestep is None:
        fill_global_expected(items)

    for item in sorted(items, key=lambda scan_item: str(scan_item.run_dir)):
        reason = incomplete_reason(item)
        if reason is None:
            continue

        path_text = format_path(item.run_dir, root, args.relative)
        if args.show_reason:
            print(f"{path_text}\t{reason}")
        else:
            print(path_text)


if __name__ == "__main__":
    main()
