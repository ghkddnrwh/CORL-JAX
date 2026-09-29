from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import orjson as _fast_json
except ImportError:  # pragma: no cover - stdlib fallback for minimal environments
    _fast_json = None


# ============================================================================
# Default paths for the 2026-09-25 diagnostic reruns
# ============================================================================

DEFAULT_IQL_LOG_ROOT = Path("logs/wandb_logs_2026_09_25_orl_smooth_144")
DEFAULT_DDIQL_LOG_ROOT = Path("logs/iclr2027/iql_based/ogbench/diagnostics_dd_iql")
DEFAULT_OUT_ROOT = Path("logs/iclr2027_dd_iql/paper_diagnostic_tables_20260925")


# Only these W&B history metrics are needed by analyses 01--05.  Restricting
# the parsed DataFrame to this set keeps the downloaded W&B histories compact
# and avoids carrying unrelated training scalars through pandas.
TABLE_HISTORY_METRICS = {
    "diag/filter_receiver_td_corr",
    "diag/filter_receiver_td_cov",
    "diag/filter_source_loo_corr",
    "diag/filter_receiver_loo_corr",
    "diag/filter_source_adv_corr",
    "diag/filter_receiver_adv_corr",
    "diag/source_receiver_error_corr_abs",
    "diag/filter_weight_mean",
    "diag/td_residual_rmse",
    "eval/success_rate",
    "eval/q_return_gap_mean",
    "eval/q_return_gap_rmse",
    "eval/q_over_return_fraction",
    "eval/v_return_gap_mean",
    "eval/v_return_gap_rmse",
}


# ============================================================================
# Environment ordering / display names
# ============================================================================

# Same order as the nine-environment main result table.
ENV_ORDER = [
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


def display_env(env: str) -> str:
    env_lower = env.lower()
    if "antmaze" in env_lower and "large" in env_lower:
        return "AntMaze Large"
    if "humanoidmaze" in env_lower and "medium" in env_lower:
        return "HumanoidMaze Medium"
    if "humanoidmaze" in env_lower and "large" in env_lower:
        return "HumanoidMaze Large"
    if "antsoccer" in env_lower and "arena" in env_lower:
        return "AntSoccer Arena"
    if "cube-double" in env_lower:
        return "Cube Double Play"
    if "cube-single" in env_lower:
        return "Cube Single Play"
    if "puzzle" in env_lower and "3x3" in env_lower:
        return "Puzzle 3x3"
    if "puzzle" in env_lower and "4x4" in env_lower:
        return "Puzzle 4x4"
    if "scene" in env_lower:
        return "Scene Play"
    return env.replace("_", r"\_")


def env_sort_key(env: str) -> Tuple[int, str]:
    try:
        return ENV_ORDER.index(env), env
    except ValueError:
        return len(ENV_ORDER), env


# ============================================================================
# Run specification
# ============================================================================

@dataclass(frozen=True)
class RunSpec:
    algo: str
    env: str
    discount: float
    expectile: float
    seed: int

    # data_format is either "wandb_jsonl" or "local_npz".
    data_format: str
    diag_path: Path
    eval_path: Optional[Path]
    config_path: Path
    delay_period: Optional[int]

    # Optional local-DD-IQL files / metadata.
    delayed_diag_path: Optional[Path] = None
    status_path: Optional[Path] = None
    status: str = "unknown"
    max_timesteps: Optional[int] = None
    latest_timestep: Optional[int] = None

    project: str = ""
    group: str = ""
    run_id: str = ""

    @property
    def setting(self) -> str:
        return "stress" if self.discount >= 0.999 - 1e-12 else "original"

    @property
    def condition(self) -> Tuple[str, float, float]:
        return self.env, self.discount, self.expectile

    @property
    def identity(self) -> Tuple[str, str, float, float, int]:
        return self.algo, self.env, self.discount, self.expectile, self.seed


# ============================================================================
# Config / scalar helpers
# ============================================================================


def _unwrap_wandb_config_value(value):
    if isinstance(value, dict) and "value" in value and len(value) <= 3:
        return value["value"]
    return value


def load_config(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        return {}

    if (
        isinstance(payload.get("config"), dict)
        and not any(k in payload for k in ("env", "seed", "discount", "iql_tau"))
    ):
        payload = payload["config"]

    return {k: _unwrap_wandb_config_value(v) for k, v in payload.items()}


def _to_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=-1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _special_string_to_number(value):
    if not isinstance(value, str):
        return value
    x = value.strip().lower()
    if x in {"nan", "+nan", "-nan"}:
        return np.nan
    if x in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if x in {"-inf", "-infinity"}:
        return -np.inf
    return value


# ============================================================================
# Run discovery
# ============================================================================


def _run_config_path(run_dir: Path) -> Optional[Path]:
    for name in ("config.json", "config.yaml", "config.yml"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _read_local_status(run_dir: Path) -> Tuple[str, Optional[int], Optional[int], Optional[Path]]:
    status_path = run_dir / "training_status.yaml"
    if not status_path.exists():
        return "unknown", None, None, None
    try:
        payload = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "unknown", None, None, status_path
    status = str(payload.get("status", "unknown"))
    max_timesteps = _to_int(payload.get("max_timesteps"), default=-1)
    timestep = _to_int(payload.get("timestep"), default=-1)
    return (
        status,
        max_timesteps if max_timesteps >= 0 else None,
        timestep if timestep >= 0 else None,
        status_path,
    )


def _discover_iql_wandb_runs(log_root: Path) -> List[RunSpec]:
    log_root = Path(log_root)
    if not log_root.exists():
        warnings.warn(f"IQL W&B log root does not exist: {log_root}")
        return []

    runs: List[RunSpec] = []
    for history_path in sorted(log_root.rglob("history.jsonl")):
        run_dir = history_path.parent
        config_path = _run_config_path(run_dir)
        if config_path is None:
            warnings.warn(f"Skipping IQL history without config: {run_dir}")
            continue
        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.warn(f"Could not read {config_path}: {exc}")
            continue

        env = str(config.get("env", "")).strip()
        discount = _to_float(config.get("discount"))
        expectile = _to_float(config.get("iql_tau"))
        seed = _to_int(config.get("seed"))
        if not env or not np.isfinite(discount) or not np.isfinite(expectile) or seed < 0:
            warnings.warn(f"Skipping IQL run with incomplete metadata: {run_dir}")
            continue

        delay_period = None
        candidate = _to_int(config.get("delayed_update_period"), default=-1)
        if candidate > 0:
            delay_period = candidate

        # New export layout is GROUP/RUN_ID, but use config values whenever possible.
        project = str(config.get("project", ""))
        group = str(config.get("group", run_dir.parent.name))
        run_id = run_dir.name

        runs.append(
            RunSpec(
                algo="IQL",
                env=env,
                discount=discount,
                expectile=expectile,
                seed=seed,
                data_format="wandb_jsonl",
                diag_path=history_path,
                eval_path=history_path,
                config_path=config_path,
                delay_period=delay_period,
                project=project,
                group=group,
                run_id=run_id,
                status="exported",
                max_timesteps=_to_int(config.get("max_timesteps"), default=-1),
            )
        )
    return runs


def _discover_ddiql_local_runs(log_root: Path) -> List[RunSpec]:
    log_root = Path(log_root)
    if not log_root.exists():
        warnings.warn(f"DD-IQL local log root does not exist: {log_root}")
        return []

    # A local run is identified by config.{yaml,json}; diagnostics can be absent
    # for interrupted/failed runs, which we still retain for coverage reporting.
    config_paths = sorted(list(log_root.rglob("config.yaml")) + list(log_root.rglob("config.json")))
    runs: List[RunSpec] = []
    seen_dirs = set()

    for config_path in config_paths:
        run_dir = config_path.parent
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)

        try:
            config = load_config(config_path)
        except Exception as exc:
            warnings.warn(f"Could not read {config_path}: {exc}")
            continue

        env = str(config.get("env", "")).strip()
        discount = _to_float(config.get("discount"))
        expectile = _to_float(config.get("iql_tau"))
        seed = _to_int(config.get("seed"))
        if not env or not np.isfinite(discount) or not np.isfinite(expectile) or seed < 0:
            continue

        # Avoid accidentally treating unrelated configs under the root as DD-IQL.
        text = " ".join(
            [
                str(config.get("name", "")),
                str(config.get("group", "")),
                str(config.get("project", "")),
                str(run_dir),
            ]
        ).lower().replace("-", "").replace("_", "")
        if "ddiql" not in text and "decoupleddelayediql" not in text:
            continue

        bias_path = run_dir / "bias_diagnostics.npz"
        delayed_path = run_dir / "delayed_diagnostics.npz"
        eval_path = run_dir / "eval_logs.npz"
        status, max_timesteps, latest_timestep, status_path = _read_local_status(run_dir)

        delay_period = _to_int(config.get("delayed_update_period"), default=-1)
        if delay_period <= 0:
            delay_period = None

        # diag_path may not exist for failed runs; load_diag() handles that.
        runs.append(
            RunSpec(
                algo="DD-IQL",
                env=env,
                discount=discount,
                expectile=expectile,
                seed=seed,
                data_format="local_npz",
                diag_path=bias_path,
                delayed_diag_path=delayed_path if delayed_path.exists() else None,
                eval_path=eval_path if eval_path.exists() else None,
                config_path=config_path,
                delay_period=delay_period,
                status_path=status_path,
                status=status,
                max_timesteps=max_timesteps if max_timesteps is not None else _to_int(config.get("max_timesteps"), default=-1),
                latest_timestep=latest_timestep,
                project=str(config.get("project", "")),
                group=str(config.get("group", "")),
                run_id=run_dir.name + f"/seed{seed}",
            )
        )
    return runs


def _artifact_size(run: RunSpec) -> int:
    total = 0
    for p in (run.diag_path, run.delayed_diag_path, run.eval_path):
        if p is None:
            continue
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _deduplicate_runs(runs: Sequence[RunSpec]) -> List[RunSpec]:
    dedup: Dict[Tuple[str, str, float, float, int], RunSpec] = {}
    for run in runs:
        old = dedup.get(run.identity)
        if old is None:
            dedup[run.identity] = run
            continue
        # Prefer a completed local run, otherwise the run with more stored data.
        old_completed = old.status == "completed"
        new_completed = run.status == "completed"
        if new_completed and not old_completed:
            dedup[run.identity] = run
        elif new_completed == old_completed and _artifact_size(run) > _artifact_size(old):
            dedup[run.identity] = run
    return list(dedup.values())


def discover_runs(
    iql_log_root: Optional[Path] = None,
    ddiql_log_root: Optional[Path] = None,
    legacy_log_root: Optional[Path] = None,
) -> List[RunSpec]:
    """Discover mixed-format IQL/DD-IQL runs.

    IQL is read from the W&B download format (history.jsonl + config.json).
    DD-IQL is read from local NPZ checkpoints (bias_diagnostics.npz,
    delayed_diagnostics.npz, eval_logs.npz + config.yaml).

    ``legacy_log_root`` is retained only for backward compatibility.  If it is
    supplied, both W&B histories and local DD-IQL NPZ directories below that root
    are scanned.
    """
    if legacy_log_root is not None:
        root = Path(legacy_log_root)
        candidates = _discover_iql_wandb_runs(root) + _discover_ddiql_local_runs(root)
    else:
        iql_root = Path(iql_log_root or DEFAULT_IQL_LOG_ROOT)
        dd_root = Path(ddiql_log_root or DEFAULT_DDIQL_LOG_ROOT)
        candidates = _discover_iql_wandb_runs(iql_root) + _discover_ddiql_local_runs(dd_root)

    runs = _deduplicate_runs(candidates)
    runs.sort(
        key=lambda r: (
            env_sort_key(r.env),
            r.algo,
            r.discount,
            r.expectile,
            r.seed,
        )
    )

    print(f"[Layout] Found {len(runs)} unique mixed-format runs.")
    for algo in ("IQL", "DD-IQL"):
        subset = [r for r in runs if r.algo == algo]
        print(f"[Layout] {algo}: {len(subset)} runs")
        envs = sorted({r.env for r in subset}, key=env_sort_key)
        print(f"[Layout] {algo} environments ({len(envs)}): {', '.join(display_env(e) for e in envs)}")

    dd = [r for r in runs if r.algo == "DD-IQL"]
    if dd:
        counts: Dict[str, int] = {}
        for r in dd:
            counts[r.status] = counts.get(r.status, 0) + 1
        print("[Layout] DD-IQL status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # This experiment design has 9 envs x 4 settings x 4 seeds = 144 runs/algo.
    for algo in ("IQL", "DD-IQL"):
        n = sum(r.algo == algo for r in runs)
        if n != 144:
            print(f"[Layout warning] Expected 144 {algo} runs for the full 9-env grid; found {n}.")

    return runs


def discover_runs_from_args(args) -> List[RunSpec]:
    legacy = getattr(args, "log_root", None)
    if legacy is not None:
        return discover_runs(legacy_log_root=legacy)
    return discover_runs(
        iql_log_root=getattr(args, "iql_log_root", DEFAULT_IQL_LOG_ROOT),
        ddiql_log_root=getattr(args, "ddiql_log_root", DEFAULT_DDIQL_LOG_ROOT),
    )


# ============================================================================
# W&B history loading
# ============================================================================


def _normalize_history_row(row: Dict) -> Dict:
    for key in ("history", "data", "metrics"):
        nested = row.get(key)
        if isinstance(nested, dict):
            outer = {k: v for k, v in row.items() if k != key}
            return {**outer, **nested}
    return row


def _first_numeric(row: Dict, keys: Sequence[str]) -> float:
    for key in keys:
        if key not in row:
            continue
        value = _special_string_to_number(row[key])
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return np.nan


@lru_cache(maxsize=16)
def _load_history_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["timestep"])

    config_path = _run_config_path(path.parent)
    try:
        config = load_config(config_path) if config_path is not None else {}
    except Exception:
        config = {}
    diagnostic_freq = _to_int(config.get("diagnostic_freq"), default=1)
    if diagnostic_freq <= 0:
        diagnostic_freq = 1

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"Malformed JSON in {path}:{line_number}: {exc}")
                continue
            if not isinstance(raw, dict):
                continue
            row = _normalize_history_row(raw)

            # IMPORTANT for the 2026-09-25 export: training/timestep is explicit
            # and must be preferred over W&B _step.  The latter is a history index
            # and different metric families have different logging cadences.
            step = _first_numeric(
                row,
                ("training/timestep", "timestep", "global_step", "step"),
            )
            if not np.isfinite(step):
                wandb_step = _first_numeric(row, ("_step",))
                if np.isfinite(wandb_step):
                    step = wandb_step * diagnostic_freq
            if not np.isfinite(step):
                continue

            clean = {
                k: _special_string_to_number(row[k])
                for k in TABLE_HISTORY_METRICS
                if k in row
            }
            if not clean:
                continue
            clean["timestep"] = int(round(step))
            rows.append(clean)

    return _merge_sparse_rows(rows)


@lru_cache(maxsize=None)
def _load_wandb_series_cached(path_str: str):
    """Parse one W&B history once into compact metric-wise NumPy arrays.

    The downloaded histories are sparse JSONL streams and can be large.  The
    old analysis repeatedly reparsed the same file for different tables.  This
    cache keeps only the metrics used by analyses 01--05, indexed by the actual
    training timestep.  It therefore handles mixed logging cadences without
    using W&B's history-row ``_step`` as a training step.
    """
    path = Path(path_str)
    if not path.exists():
        return {}

    config_path = _run_config_path(path.parent)
    try:
        config = load_config(config_path) if config_path is not None else {}
    except Exception:
        config = {}
    diagnostic_freq = _to_int(config.get("diagnostic_freq"), default=1)
    if diagnostic_freq <= 0:
        diagnostic_freq = 1

    # Store the last logged value at each training timestep, matching the
    # GroupBy.last() behavior of the DataFrame loader while using far less RAM.
    by_metric = {metric: {} for metric in TABLE_HISTORY_METRICS}

    # Binary mode + orjson is substantially faster for the ~1 GB W&B export.
    # Fall back to stdlib json when orjson is unavailable on the user's machine.
    with path.open("rb") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = _fast_json.loads(line) if _fast_json is not None else json.loads(line)
            except Exception as exc:
                warnings.warn(f"Malformed JSON in {path}:{line_number}: {exc}")
                continue
            if not isinstance(raw, dict):
                continue
            row = _normalize_history_row(raw)

            # Prefer the explicit optimizer/training step.  In the current W&B
            # export ``_step`` is merely the history-row index and is not
            # comparable when logging frequencies differ across runs.
            step = _first_numeric(
                row,
                ("training/timestep", "timestep", "global_step", "step"),
            )
            if not np.isfinite(step):
                wandb_step = _first_numeric(row, ("_step",))
                if np.isfinite(wandb_step):
                    step = wandb_step * diagnostic_freq
            if not np.isfinite(step):
                continue
            step = int(round(step))

            for metric in TABLE_HISTORY_METRICS.intersection(row):
                value = _special_string_to_number(row[metric])
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                # Keep NaN out of the compact series.  It carries no information
                # for the seed-level summaries and can overwrite a finite value
                # when sparse W&B rows share a timestep.
                if not np.isfinite(value):
                    continue
                by_metric[metric][step] = value

    result = {}
    for metric, step_values in by_metric.items():
        if not step_values:
            continue
        steps = np.fromiter(sorted(step_values), dtype=np.int64)
        values = np.asarray([step_values[int(step)] for step in steps], dtype=np.float64)
        result[metric] = (steps, values)
    return result


def _wandb_metric_series(run: RunSpec, metric: str):
    series = _load_wandb_series_cached(str(run.diag_path.resolve()))
    return series.get(metric, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))


# ============================================================================
# Local NPZ loading
# ============================================================================


def _merge_sparse_rows(rows: Sequence[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestep"])
    frame = pd.DataFrame(rows)
    if "timestep" not in frame.columns:
        return pd.DataFrame(columns=["timestep"])
    frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce")
    frame = frame.dropna(subset=["timestep"])
    frame["timestep"] = frame["timestep"].astype(int)
    if frame["timestep"].is_unique:
        return frame.sort_values("timestep").reset_index(drop=True)
    return (
        frame.groupby("timestep", sort=True, as_index=False)
        .last()
        .sort_values("timestep")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=192)
def _load_npz_metric_cached(path_str: str, metric: str, include_delay_age: bool = False) -> pd.DataFrame:
    """Load only one scalar metric (plus timestep / optional delay age) from NPZ.

    ``delayed_diagnostics.npz`` is much denser than the coarse bias log.  NumPy's
    NPZ reader decompresses members lazily, so selecting only the required arrays
    avoids loading all diagnostic columns for every table.
    """
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["timestep", metric])
    with np.load(path, allow_pickle=True) as data:
        if "timestep" not in data.files or metric not in data.files:
            return pd.DataFrame(columns=["timestep", metric])
        timestep = np.asarray(data["timestep"]).reshape(-1)
        values = np.asarray(data[metric]).reshape(-1)
        arrays = {"timestep": timestep, metric: values}
        if include_delay_age and "diag/delay_age" in data.files:
            arrays["diag/delay_age"] = np.asarray(data["diag/delay_age"]).reshape(-1)
        n = min(len(v) for v in arrays.values())
        frame = pd.DataFrame({k: v[:n] for k, v in arrays.items()})
    frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce")
    frame = frame.dropna(subset=["timestep"])
    frame["timestep"] = frame["timestep"].astype(int)
    if frame["timestep"].is_unique:
        return frame.sort_values("timestep").reset_index(drop=True)
    return _merge_sparse_rows(frame.to_dict(orient="records"))


@lru_cache(maxsize=32)
def _load_npz_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["timestep"])
    with np.load(path, allow_pickle=True) as data:
        if not data.files:
            return pd.DataFrame(columns=["timestep"])
        columns = {}
        lengths = []
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim != 1:
                # Current diagnostic logs are scalar-per-checkpoint.  Preserve
                # unusual values as objects rather than silently reshaping them.
                arr = np.asarray(list(arr), dtype=object)
            columns[key] = arr
            lengths.append(len(arr))
        if not lengths:
            return pd.DataFrame(columns=["timestep"])
        n = min(lengths)
        if len(set(lengths)) != 1:
            warnings.warn(f"NPZ arrays have inconsistent lengths in {path}; truncating to {n}.")
        frame = pd.DataFrame({k: v[:n] for k, v in columns.items()})
    if "timestep" not in frame.columns:
        warnings.warn(f"NPZ has no timestep column: {path}")
        return pd.DataFrame(columns=["timestep"])
    return _merge_sparse_rows(frame.to_dict(orient="records"))


def _merge_frames_on_timestep(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=["timestep"])
    # Concatenating sparse frames and GroupBy.last() merges non-null columns at
    # duplicate timesteps.  In the supplied DD-IQL logs overlapping temporal
    # entries in bias_diagnostics.npz and delayed_diagnostics.npz are identical.
    combo = pd.concat(usable, ignore_index=True, sort=False)
    records = combo.to_dict(orient="records")
    return _merge_sparse_rows(records)


# ============================================================================
# Public loaders
# ============================================================================


def load_diag(run: RunSpec) -> pd.DataFrame:
    if run.data_format == "wandb_jsonl":
        return _load_history_cached(str(run.diag_path.resolve()))
    if run.data_format == "local_npz":
        frames = []
        if run.diag_path.exists():
            frames.append(_load_npz_cached(str(run.diag_path.resolve())))
        if run.delayed_diag_path is not None and run.delayed_diag_path.exists():
            frames.append(_load_npz_cached(str(run.delayed_diag_path.resolve())))
        return _merge_frames_on_timestep(frames)
    raise ValueError(f"Unknown data format: {run.data_format}")


def load_eval(run: RunSpec) -> pd.DataFrame:
    if run.eval_path is None or not run.eval_path.exists():
        return pd.DataFrame(columns=["timestep"])
    if run.data_format == "wandb_jsonl":
        return _load_history_cached(str(run.eval_path.resolve()))
    if run.data_format == "local_npz":
        return _load_npz_cached(str(run.eval_path.resolve()))
    raise ValueError(f"Unknown data format: {run.data_format}")


def load_metric_frame(run: RunSpec, metric: str, prefer_delayed: bool = False) -> pd.DataFrame:
    """Load only the stream needed for one diagnostic metric.

    This matters for the new local DD-IQL layout: bias_diagnostics.npz has
    ~400 coarse checkpoints for a completed run, whereas delayed_diagnostics.npz
    has ~40,000 high-frequency checkpoints.  Most tables should not materialize
    the high-frequency stream unless they actually need it.
    """
    if run.data_format == "wandb_jsonl":
        timesteps, values = _wandb_metric_series(run, metric)
        if len(values) == 0:
            return pd.DataFrame(columns=["timestep", metric])
        return pd.DataFrame({"timestep": timesteps, metric: values})

    if run.data_format != "local_npz":
        raise ValueError(f"Unknown data format: {run.data_format}")

    # Prefer the high-frequency delayed stream only for metrics that need it
    # (analysis 01).  Include delay_age in that case because the table groups
    # observations by phase within the hard-delay interval.
    if prefer_delayed and run.delayed_diag_path is not None and run.delayed_diag_path.exists():
        delayed = _load_npz_metric_cached(
            str(run.delayed_diag_path.resolve()), metric, include_delay_age=True
        )
        if metric in delayed.columns and not delayed.empty:
            return delayed

    if run.diag_path.exists():
        bias = _load_npz_metric_cached(str(run.diag_path.resolve()), metric, include_delay_age=False)
        if metric in bias.columns and not bias.empty:
            return bias

    if run.delayed_diag_path is not None and run.delayed_diag_path.exists():
        delayed = _load_npz_metric_cached(
            str(run.delayed_diag_path.resolve()), metric, include_delay_age=prefer_delayed
        )
        if metric in delayed.columns and not delayed.empty:
            return delayed

    return pd.DataFrame(columns=["timestep", metric])


# ============================================================================
# Windows / subsampling / summaries
# ============================================================================


def window_frame(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if frame.empty or "timestep" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[(frame["timestep"] >= start) & (frame["timestep"] <= end)]


def evenly_spaced_mask(timesteps: np.ndarray, max_points: int) -> np.ndarray:
    timesteps = np.asarray(timesteps, dtype=np.int64)
    n = len(timesteps)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if max_points is None or int(max_points) <= 0 or n <= int(max_points):
        return np.ones(n, dtype=bool)
    max_points = int(max_points)
    positions = (np.arange(max_points, dtype=float) + 0.5) * n / max_points
    indices = np.unique(np.clip(np.floor(positions).astype(int), 0, n - 1))
    keep = np.zeros(n, dtype=bool)
    keep[indices] = True
    return keep


def seed_metric_mean(
    run: RunSpec,
    metric: str,
    start: int,
    end: int,
    absolute: bool = False,
    max_points: int = 0,
) -> float:
    if run.data_format == "wandb_jsonl":
        timesteps, values = _wandb_metric_series(run, metric)
        if len(values) == 0:
            return np.nan
        in_window = (timesteps >= int(start)) & (timesteps <= int(end))
        timesteps = timesteps[in_window]
        values = values[in_window]
    else:
        frame = window_frame(load_metric_frame(run, metric), start, end)
        if frame.empty or metric not in frame.columns:
            return np.nan
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        timesteps = frame["timestep"].to_numpy(dtype=np.int64)
        valid = np.isfinite(values)
        values = values[valid]
        timesteps = timesteps[valid]

    if len(values) == 0:
        return np.nan
    keep = evenly_spaced_mask(timesteps, max_points)
    values = values[keep]
    if absolute:
        values = np.abs(values)
    return float(np.mean(values)) if len(values) else np.nan


def final_eval_seed_value(
    run: RunSpec,
    metric: str,
    last_evals: int = 1,
    eval_step: Optional[int] = 1_000_000,
    allow_earlier_eval: bool = False,
) -> float:
    """Extract a seed-level evaluation value at a comparable training horizon.

    By default the run must contain the requested ``eval_step`` (1M).  This is
    important for the current local DD-IQL archive, where interrupted runs have
    only a 100k/200k/300k evaluation; treating those as "final" would create an
    invalid comparison against IQL's 1M evaluation.
    """
    if run.data_format == "wandb_jsonl":
        timesteps, metric_values = _wandb_metric_series(run, metric)
        if len(metric_values) == 0:
            return np.nan
        selected = pd.DataFrame({"timestep": timesteps.astype(int), "value": metric_values})
    else:
        frame = load_eval(run)
        if frame.empty or metric not in frame.columns:
            return np.nan

        metric_values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        timesteps = pd.to_numeric(frame["timestep"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(metric_values) & np.isfinite(timesteps)
        if not np.any(valid):
            return np.nan
        selected = pd.DataFrame({"timestep": timesteps[valid].astype(int), "value": metric_values[valid]})

    selected = selected.sort_values("timestep")

    if eval_step is not None and int(eval_step) > 0:
        eval_step = int(eval_step)
        selected = selected[selected["timestep"] <= eval_step]
        if selected.empty:
            return np.nan
        latest = int(selected["timestep"].max())
        if latest < eval_step and not allow_earlier_eval:
            return np.nan

    values = selected["value"].to_numpy(dtype=float)
    if last_evals > 0:
        values = values[-int(last_evals):]
    return float(np.mean(values)) if len(values) else np.nan


def summarize_seed_values(values: Iterable[float]) -> Tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(values)), float(np.std(values)), int(len(values))


def group_by_condition(runs: Sequence[RunSpec]) -> Dict[Tuple[str, float, float], List[RunSpec]]:
    groups: Dict[Tuple[str, float, float], List[RunSpec]] = {}
    for run in runs:
        groups.setdefault(run.condition, []).append(run)
    for key in groups:
        groups[key].sort(key=lambda r: r.seed)
    return groups


# ============================================================================
# Setting labels / ordering
# ============================================================================


def original_discount_for_env(env: str) -> float:
    env_lower = env.lower()
    # In the attached 9-env logs, both HumanoidMaze variants and AntSoccer use
    # gamma=.995 as the reference setting.  The remaining six use gamma=.99.
    if "humanoidmaze" in env_lower or "antsoccer" in env_lower:
        return 0.995
    return 0.99


def condition_setting_label(condition: Tuple[str, float, float]) -> str:
    env, gamma, _ = condition
    original_gamma = original_discount_for_env(env)
    if np.isclose(gamma, original_gamma):
        return "Original"
    if np.isclose(gamma, 0.999):
        return r"High-$\gamma$"
    return rf"$\gamma={gamma:g}$"


def condition_setting_order(condition: Tuple[str, float, float]) -> int:
    env, gamma, tau = condition
    original_gamma = original_discount_for_env(env)
    gamma_rank = 0 if np.isclose(gamma, original_gamma) else (1 if np.isclose(gamma, 0.999) else 2)
    tau_rank = 0 if np.isclose(tau, 0.9) else (1 if np.isclose(tau, 0.99) else 2)
    return 2 * gamma_rank + tau_rank


def condition_sort_key(condition: Tuple[str, float, float]):
    env, discount, expectile = condition
    return env_sort_key(env), condition_setting_order(condition), discount, expectile


# ============================================================================
# Formatting / output
# ============================================================================


def format_pm(mean: float, std: float, digits: int = 3) -> str:
    if not (np.isfinite(mean) and np.isfinite(std)):
        return r"$\mathrm{nan}$"
    scale = max(abs(mean), abs(std))
    if scale >= 1e5 or (scale > 0 and scale < 10 ** (-(digits + 1))):
        def sci(value: float) -> str:
            if value == 0:
                return f"{0:.{digits}f}"
            exponent = int(math.floor(math.log10(abs(value))))
            mantissa = value / (10.0 ** exponent)
            return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"
        return rf"${sci(mean)} \pm {sci(std)}$"
    return rf"${mean:.{digits}f} \pm {std:.{digits}f}$"


def format_percent(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return r"$\mathrm{nan}$"
    return rf"${value:.{digits}f}\%$"


def float_label(value: float) -> str:
    return f"{value:g}"


def make_latex_row(cells: Sequence[str]) -> str:
    return " & ".join(cells) + r" \\"


def write_latex_rows(
    path: Path,
    title: str,
    columns: Sequence[str],
    rows: Sequence[str],
    comments: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"% {title}", "% Columns: " + " | ".join(columns)]
    if comments:
        lines.extend(f"% {comment}" for comment in comments)
    lines.extend(rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[LaTeX rows: {title}]")
    for line in lines:
        print(line)


def write_csv(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def write_coverage_report(runs: Sequence[RunSpec], path: Path) -> None:
    records = []
    for r in runs:
        diag = load_diag(r)
        ev = load_eval(r)
        records.append(
            {
                "algorithm": r.algo,
                "environment": r.env,
                "setting": condition_setting_label(r.condition),
                "discount": r.discount,
                "expectile": r.expectile,
                "seed": r.seed,
                "format": r.data_format,
                "status": r.status,
                "configured_max_timesteps": r.max_timesteps,
                "status_latest_timestep": r.latest_timestep,
                "diag_max_timestep": int(diag["timestep"].max()) if not diag.empty else np.nan,
                "eval_max_timestep": int(ev["timestep"].max()) if not ev.empty else np.nan,
                "has_bias_diag": bool(r.diag_path.exists()),
                "has_delayed_diag": bool(r.delayed_diag_path is not None and r.delayed_diag_path.exists()),
                "has_eval": bool(r.eval_path is not None and r.eval_path.exists()),
                "run_id": r.run_id,
                "config_path": str(r.config_path),
            }
        )
    write_csv(path, records)


# ============================================================================
# CLI
# ============================================================================


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--iql-log-root",
        type=Path,
        default=DEFAULT_IQL_LOG_ROOT,
        help="W&B-download root for IQL (history.jsonl + config.json).",
    )
    parser.add_argument(
        "--ddiql-log-root",
        type=Path,
        default=DEFAULT_DDIQL_LOG_ROOT,
        help="Local DD-IQL root (bias_diagnostics.npz / delayed_diagnostics.npz / eval_logs.npz).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Legacy single-root mode. If supplied, overrides the two roots above and scans both formats below this root.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--early-start", type=int, default=5_000)
    parser.add_argument("--early-end", type=int, default=100_000)
    parser.add_argument(
        "--diag-points",
        type=int,
        default=20,
        help="Max diagnostic checkpoints per seed in the early window; 0 uses every point.",
    )
    parser.add_argument("--primary-discount", type=float, default=0.999)
    parser.add_argument("--primary-tau", type=float, default=0.99)
    parser.add_argument(
        "--last-evals",
        type=int,
        default=1,
        help="Number of evaluations ending at --eval-step to average within a seed.",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=1_000_000,
        help="Required final evaluation training step. Default 1,000,000.",
    )
    parser.add_argument(
        "--allow-earlier-eval",
        action="store_true",
        help="Allow a seed whose last evaluation is earlier than --eval-step. Off by default to avoid comparing partial DD-IQL runs to 1M-step IQL runs.",
    )
    return parser
