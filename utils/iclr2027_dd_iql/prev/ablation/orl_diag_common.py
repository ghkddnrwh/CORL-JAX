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


# ============================================================================
# Default paths
# ============================================================================

DEFAULT_LOG_ROOT = Path(
    "logs/wandb_logs_2026_09_22_orl_160"
)

DEFAULT_OUT_ROOT = Path(
    "paper_diagnostic_tables"
)


# ============================================================================
# Environment ordering / display names
# ============================================================================

ENV_ORDER = [
    "antmaze-large-navigate-singletask-v0",
    "cube-single-play-singletask-v0",
    "humanoidmaze-medium-navigate-singletask-v0",
    "puzzle-4x4-play-singletask-v0",
    "scene-play-singletask-v0",
]


def display_env(env: str) -> str:
    env_lower = env.lower()

    if (
        "antmaze" in env_lower
        and "large" in env_lower
    ):
        return "AntMaze Large"

    if "cube" in env_lower:
        return "Cube Single Play"

    if (
        "humanoidmaze" in env_lower
        and "medium" in env_lower
    ):
        return "HumanoidMaze Medium"

    if (
        "puzzle" in env_lower
        and "4x4" in env_lower
    ):
        return "Puzzle 4x4"

    if "scene" in env_lower:
        return "Scene Play"

    return env.replace("_", r"\_")


def env_sort_key(
    env: str,
) -> Tuple[int, str]:
    try:
        return (
            ENV_ORDER.index(env),
            env,
        )

    except ValueError:
        return (
            len(ENV_ORDER),
            env,
        )


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

    # For compatibility with the analysis scripts.
    #
    # In this W&B export both diagnostic and evaluation metrics
    # are stored inside history.jsonl, so diag_path and eval_path
    # will usually point to the same file.
    diag_path: Path
    eval_path: Optional[Path]

    config_path: Path
    delay_period: Optional[int]

    project: str = ""
    group: str = ""
    run_id: str = ""

    @property
    def setting(self) -> str:
        return (
            "stress"
            if self.discount >= 0.999 - 1e-12
            else "original"
        )

    @property
    def condition(
        self,
    ) -> Tuple[str, float, float]:
        return (
            self.env,
            self.discount,
            self.expectile,
        )

    @property
    def identity(
        self,
    ) -> Tuple[
        str,
        str,
        float,
        float,
        int,
    ]:
        return (
            self.algo,
            self.env,
            self.discount,
            self.expectile,
            self.seed,
        )


# ============================================================================
# Config loading
# ============================================================================

def _unwrap_wandb_config_value(
    value,
):
    """
    Some W&B-exported configs can store entries as

        "seed": {"value": 0}

    while others directly store

        "seed": 0

    Support both.
    """
    if (
        isinstance(value, dict)
        and "value" in value
        and len(value) <= 3
    ):
        return value["value"]

    return value


def load_config(
    path: Path,
) -> Dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {path}"
        )

    if path.suffix.lower() == ".json":
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

    else:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = yaml.safe_load(f)

    if payload is None:
        return {}

    if not isinstance(
        payload,
        dict,
    ):
        return {}

    # Some export formats wrap the actual config
    # below a top-level "config" field.
    if (
        isinstance(
            payload.get("config"),
            dict,
        )
        and not any(
            key in payload
            for key in (
                "env",
                "seed",
                "discount",
                "iql_tau",
            )
        )
    ):
        payload = payload["config"]

    result = {}

    for key, value in payload.items():
        result[key] = (
            _unwrap_wandb_config_value(
                value
            )
        )

    return result


# ============================================================================
# Scalar conversion helpers
# ============================================================================

def _to_float(
    value,
    default=np.nan,
) -> float:
    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return float(default)


def _to_int(
    value,
    default=-1,
) -> int:
    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return int(default)


# ============================================================================
# Algorithm inference
# ============================================================================

def infer_algo(
    config: Dict,
    history_path: Path,
) -> str:
    """
    Infer IQL vs DD-IQL from both config and exported directory path.

    Expected project layout:

        ORL-SMOOTH/... -> IQL
        ORL-BIAS/...   -> DD-IQL
    """

    text = " ".join(
        [
            str(
                config.get(
                    "name",
                    "",
                )
            ),
            str(
                config.get(
                    "group",
                    "",
                )
            ),
            str(
                config.get(
                    "project",
                    "",
                )
            ),
            str(history_path),
        ]
    ).lower()

    compact = (
        text
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    # Check DD-IQL first because "DDIQL" also contains "IQL".
    if (
        "orl-bias" in text
        or "ddiql" in compact
        or "decoupleddelayed" in compact
        or "decoupleddelayediql" in compact
    ):
        return "DD-IQL"

    if (
        "orl-smooth" in text
        or "iql" in compact
    ):
        return "IQL"

    return "UNKNOWN"


# ============================================================================
# Run discovery
# ============================================================================

def discover_runs(
    log_root: Path,
) -> List[RunSpec]:
    """
    Discover runs from the exported W&B structure:

        log_root/
          PROJECT/
            GROUP/
              RUN_ID/
                history.jsonl
                config.json
                summary.json
                metadata.json

    Example:

        logs/wandb_logs_2026_09_22_orl_160/
          ORL-SMOOTH/
            hm_iql/
              <run_id>/
                history.jsonl
                config.json
                ...

          ORL-BIAS/
            hm_dd/
              <run_id>/
                history.jsonl
                config.json
                ...
    """

    log_root = Path(log_root)

    if not log_root.exists():
        raise FileNotFoundError(
            "Log root does not exist:\n"
            f"  {log_root}"
        )

    history_paths = sorted(
        log_root.rglob(
            "history.jsonl"
        )
    )

    if not history_paths:
        raise FileNotFoundError(
            "No history.jsonl files were found under:\n"
            f"  {log_root}\n\n"
            "Expected structure:\n"
            "  PROJECT/GROUP/RUN_ID/history.jsonl"
        )

    discovered: List[
        RunSpec
    ] = []

    for history_path in history_paths:
        run_dir = (
            history_path.parent
        )

        config_json = (
            run_dir
            / "config.json"
        )

        config_yaml = (
            run_dir
            / "config.yaml"
        )

        if config_json.exists():
            config_path = (
                config_json
            )

        elif config_yaml.exists():
            config_path = (
                config_yaml
            )

        else:
            warnings.warn(
                "Skipping run without "
                f"config file: {run_dir}"
            )
            continue

        try:
            config = load_config(
                config_path
            )

        except Exception as exc:
            warnings.warn(
                "Could not load config "
                f"{config_path}: {exc}"
            )
            continue

        # ------------------------------------------------------------
        # Extract project / group / run ID from path
        # ------------------------------------------------------------

        try:
            relative = (
                history_path
                .resolve()
                .relative_to(
                    log_root.resolve()
                )
            )

            parts = (
                relative.parts
            )

            project = (
                parts[0]
                if len(parts) >= 1
                else ""
            )

            group = (
                parts[1]
                if len(parts) >= 2
                else ""
            )

            run_id = (
                parts[2]
                if len(parts) >= 3
                else run_dir.name
            )

        except Exception:
            project = ""
            group = ""
            run_id = (
                run_dir.name
            )

        # ------------------------------------------------------------
        # Infer run metadata
        # ------------------------------------------------------------

        algo = infer_algo(
            config,
            history_path,
        )

        env = str(
            config.get(
                "env",
                "",
            )
        ).strip()

        discount = _to_float(
            config.get(
                "discount"
            )
        )

        expectile = _to_float(
            config.get(
                "iql_tau"
            )
        )

        seed = _to_int(
            config.get(
                "seed"
            )
        )

        delay_period = None

        if (
            config.get(
                "delayed_update_period"
            )
            is not None
        ):
            candidate = _to_int(
                config.get(
                    "delayed_update_period"
                )
            )

            if candidate > 0:
                delay_period = (
                    candidate
                )

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------

        if algo == "UNKNOWN":
            warnings.warn(
                "Could not infer algorithm: "
                f"{run_dir}"
            )
            continue

        if not env:
            warnings.warn(
                "Missing environment in "
                f"{config_path}"
            )
            continue

        if not np.isfinite(
            discount
        ):
            warnings.warn(
                "Missing discount in "
                f"{config_path}"
            )
            continue

        if not np.isfinite(
            expectile
        ):
            warnings.warn(
                "Missing iql_tau in "
                f"{config_path}"
            )
            continue

        if seed < 0:
            warnings.warn(
                "Missing seed in "
                f"{config_path}"
            )
            continue

        # Diagnostics and evaluation metrics
        # are both present in W&B history.
        discovered.append(
            RunSpec(
                algo=algo,
                env=env,
                discount=discount,
                expectile=expectile,
                seed=seed,
                diag_path=history_path,
                eval_path=history_path,
                config_path=config_path,
                delay_period=delay_period,
                project=project,
                group=group,
                run_id=run_id,
            )
        )

    # ------------------------------------------------------------
    # Deduplicate accidentally duplicated exports.
    #
    # Expected identity:
    # algorithm / env / discount / expectile / seed
    #
    # If duplicates exist, retain the larger history file because
    # it is generally the most complete export.
    # ------------------------------------------------------------

    deduplicated: Dict[
        Tuple[
            str,
            str,
            float,
            float,
            int,
        ],
        RunSpec,
    ] = {}

    for run in discovered:
        key = (
            run.identity
        )

        if (
            key
            not in deduplicated
        ):
            deduplicated[
                key
            ] = run

            continue

        old = (
            deduplicated[
                key
            ]
        )

        try:
            new_size = (
                run.diag_path
                .stat()
                .st_size
            )

        except OSError:
            new_size = 0

        try:
            old_size = (
                old.diag_path
                .stat()
                .st_size
            )

        except OSError:
            old_size = 0

        if (
            new_size
            > old_size
        ):
            deduplicated[
                key
            ] = run

    runs = list(
        deduplicated.values()
    )

    runs.sort(
        key=lambda run: (
            env_sort_key(
                run.env
            ),
            run.algo,
            run.discount,
            run.expectile,
            run.seed,
        )
    )

    # ------------------------------------------------------------
    # Layout diagnostics
    # ------------------------------------------------------------

    print(
        "[Layout] "
        f"Found {len(runs)} unique runs "
        f"from {len(history_paths)} "
        "history files."
    )

    for algo in (
        "IQL",
        "DD-IQL",
    ):
        count = sum(
            run.algo == algo
            for run in runs
        )

        print(
            "[Layout] "
            f"{algo}: "
            f"{count} runs"
        )

    if (
        len(runs)
        != 160
    ):
        print(
            "[Layout warning] "
            "Expected 160 unique runs, "
            f"found {len(runs)}."
        )

    return runs


# ============================================================================
# W&B history loading
# ============================================================================

def _normalize_history_row(
    row: Dict,
) -> Dict:
    """
    Support possible export wrappers such as

        {"data": {...}, "_step": ...}

    or

        {"history": {...}}

    while preserving ordinary flat W&B rows.
    """

    for key in (
        "history",
        "data",
        "metrics",
    ):
        nested = (
            row.get(key)
        )

        if isinstance(
            nested,
            dict,
        ):
            outer = {
                k: v
                for k, v
                in row.items()
                if k != key
            }

            return {
                **outer,
                **nested,
            }

    return row


def _first_numeric(
    row: Dict,
    keys: Sequence[str],
) -> float:
    for key in keys:
        if (
            key
            not in row
        ):
            continue

        try:
            value = float(
                row[key]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if np.isfinite(
            value
        ):
            return value

    return np.nan


def _last_non_null(
    series: pd.Series,
):
    """
    Aggregate sparse W&B records that share the same training step.

    Example at step 100000:

        row 1: train/loss
        row 2: diag/filter_receiver_td_corr
        row 3: eval/success_rate

    We want one merged row containing all three metrics.
    """

    values = (
        series.dropna()
    )

    if values.empty:
        return np.nan

    return values.iloc[-1]


@lru_cache(
    maxsize=None
)
def _load_history_cached(
    path_str: str,
) -> pd.DataFrame:
    path = Path(
        path_str
    )

    if not path.exists():
        raise FileNotFoundError(
            f"History does not exist: {path}"
        )

    rows: List[
        Dict
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for (
            line_number,
            line,
        ) in enumerate(
            f,
            start=1,
        ):
            line = (
                line.strip()
            )

            if not line:
                continue

            try:
                raw = (
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:
                warnings.warn(
                    "Malformed JSON in "
                    f"{path}:"
                    f"{line_number}: "
                    f"{exc}"
                )
                continue

            if not isinstance(
                raw,
                dict,
            ):
                continue

            row = (
                _normalize_history_row(
                    raw
                )
            )

            # --------------------------------------------------------
            # Determine the training step.
            #
            # Evaluation rows explicitly contain "timestep".
            #
            # Diagnostic rows were logged with:
            #
            #     wandb.log(metrics, step=...)
            #
            # so their exported W&B history normally contains "_step".
            # --------------------------------------------------------

            step = (
                _first_numeric(
                    row,
                    (
                        "timestep",
                        "_step",
                        "step",
                        "global_step",
                    ),
                )
            )

            if not np.isfinite(
                step
            ):
                continue

            clean = dict(
                row
            )

            clean[
                "timestep"
            ] = int(
                round(step)
            )

            rows.append(
                clean
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestep"
            ]
        )

    frame = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------
    # CRITICAL:
    #
    # W&B can have multiple history records at the same explicit step:
    #
    #     training metrics
    #     bias diagnostics
    #     evaluation metrics
    #
    # We MUST merge them column-wise.
    #
    # A simple
    #
    #     drop_duplicates("timestep")
    #
    # would silently discard diagnostic/evaluation values.
    # ------------------------------------------------------------

    merged = (
        frame
        .groupby(
            "timestep",
            sort=True,
            as_index=False,
        )
        .agg(
            _last_non_null
        )
        .sort_values(
            "timestep"
        )
        .reset_index(
            drop=True
        )
    )

    # Force timestep back to numeric integer form.
    merged[
        "timestep"
    ] = pd.to_numeric(
        merged[
            "timestep"
        ],
        errors="coerce",
    )

    merged = (
        merged
        .dropna(
            subset=[
                "timestep"
            ]
        )
        .copy()
    )

    merged[
        "timestep"
    ] = (
        merged[
            "timestep"
        ]
        .astype(int)
    )

    return merged


# ============================================================================
# Public history loaders
# ============================================================================

def load_diag(
    run: RunSpec,
) -> pd.DataFrame:
    """
    Return the merged W&B history for diagnostic analysis.
    """

    return (
        _load_history_cached(
            str(
                run.diag_path.resolve()
            )
        )
        .copy()
    )


def load_eval(
    run: RunSpec,
) -> pd.DataFrame:
    """
    Evaluation metrics live in the same history.jsonl file.
    """

    if (
        run.eval_path
        is None
    ):
        return (
            pd.DataFrame()
        )

    return (
        _load_history_cached(
            str(
                run.eval_path.resolve()
            )
        )
        .copy()
    )


# ============================================================================
# Window helpers
# ============================================================================

def window_frame(
    frame: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    if (
        frame.empty
        or "timestep"
        not in frame.columns
    ):
        return pd.DataFrame()

    return (
        frame[
            (
                frame[
                    "timestep"
                ]
                >= start
            )
            & (
                frame[
                    "timestep"
                ]
                <= end
            )
        ]
        .copy()
    )


# ============================================================================
# Seed-level diagnostic summary
# ============================================================================

def seed_metric_mean(
    run: RunSpec,
    metric: str,
    start: int,
    end: int,
    absolute: bool = False,
) -> float:
    """
    Compute a time average inside one training seed.

    This is intentionally done before averaging across seeds so
    training timesteps are not treated as independent experimental
    replicates.
    """

    frame = window_frame(
        load_diag(run),
        start,
        end,
    )

    if (
        frame.empty
        or metric
        not in frame.columns
    ):
        return np.nan

    values = pd.to_numeric(
        frame[
            metric
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if absolute:
        values = np.abs(
            values
        )

    values = values[
        np.isfinite(
            values
        )
    ]

    if (
        len(values)
        == 0
    ):
        return np.nan

    return float(
        np.mean(
            values
        )
    )


# ============================================================================
# Evaluation summary
# ============================================================================

def final_eval_seed_value(
    run: RunSpec,
    metric: str,
    last_evals: int = 1,
) -> float:
    """
    Extract the last evaluation value(s) for one training seed.

    Only rows where this evaluation metric is actually present are used.
    """

    frame = (
        load_eval(run)
    )

    if (
        frame.empty
        or metric
        not in frame.columns
    ):
        return np.nan

    metric_values = (
        pd.to_numeric(
            frame[
                metric
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    valid = np.isfinite(
        metric_values
    )

    if not np.any(
        valid
    ):
        return np.nan

    selected = (
        pd.DataFrame(
            {
                "timestep": (
                    frame.loc[
                        valid,
                        "timestep",
                    ]
                    .to_numpy(
                        dtype=int
                    )
                ),
                "value": (
                    metric_values[
                        valid
                    ]
                ),
            }
        )
        .sort_values(
            "timestep"
        )
    )

    values = (
        selected[
            "value"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if (
        last_evals
        > 0
    ):
        values = values[
            -last_evals:
        ]

    if (
        len(values)
        == 0
    ):
        return np.nan

    return float(
        np.mean(
            values
        )
    )


# ============================================================================
# Seed aggregation
# ============================================================================

def summarize_seed_values(
    values: Iterable[float],
) -> Tuple[
    float,
    float,
    int,
]:
    values = np.asarray(
        list(values),
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if (
        len(values)
        == 0
    ):
        return (
            np.nan,
            np.nan,
            0,
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.std(
                values
            )
        ),
        int(
            len(values)
        ),
    )


# ============================================================================
# Grouping helpers
# ============================================================================

def group_by_condition(
    runs: Sequence[
        RunSpec
    ],
) -> Dict[
    Tuple[
        str,
        float,
        float,
    ],
    List[
        RunSpec
    ],
]:
    groups: Dict[
        Tuple[
            str,
            float,
            float,
        ],
        List[
            RunSpec
        ],
    ] = {}

    for run in runs:
        groups.setdefault(
            run.condition,
            [],
        ).append(
            run
        )

    for key in groups:
        groups[
            key
        ].sort(
            key=lambda run: (
                run.seed
            )
        )

    return groups


def condition_sort_key(
    condition: Tuple[
        str,
        float,
        float,
    ],
):
    (
        env,
        discount,
        expectile,
    ) = condition

    return (
        env_sort_key(
            env
        ),
        discount,
        expectile,
    )


def is_primary_condition(
    condition: Tuple[
        str,
        float,
        float,
    ],
    discount: float,
    expectile: float,
) -> bool:
    (
        _,
        gamma,
        tau,
    ) = condition

    return (
        np.isclose(
            gamma,
            discount,
        )
        and np.isclose(
            tau,
            expectile,
        )
    )


# ============================================================================
# Formatting
# ============================================================================

def format_number(
    value: float,
    digits: int = 3,
) -> str:
    if not np.isfinite(
        value
    ):
        return (
            r"$\mathrm{nan}$"
        )

    abs_value = abs(
        value
    )

    if (
        abs_value >= 1e5
        or (
            abs_value > 0
            and abs_value
            < 10 ** (
                -(digits + 1)
            )
        )
    ):
        exponent = int(
            math.floor(
                math.log10(
                    abs_value
                )
            )
        )

        mantissa = (
            value
            / (
                10.0
                ** exponent
            )
        )

        return (
            rf"${mantissa:.{digits}f}"
            rf"\times 10^{{{exponent}}}$"
        )

    return (
        rf"${value:.{digits}f}$"
    )


def format_pm(
    mean: float,
    std: float,
    digits: int = 3,
) -> str:
    if not (
        np.isfinite(
            mean
        )
        and np.isfinite(
            std
        )
    ):
        return (
            r"$\mathrm{nan}$"
        )

    scale = max(
        abs(mean),
        abs(std),
    )

    if (
        scale >= 1e5
        or (
            scale > 0
            and scale
            < 10 ** (
                -(digits + 1)
            )
        )
    ):

        def sci(
            value: float,
        ) -> str:
            if (
                value == 0
            ):
                return (
                    f"{0:.{digits}f}"
                )

            exponent = int(
                math.floor(
                    math.log10(
                        abs(value)
                    )
                )
            )

            mantissa = (
                value
                / (
                    10.0
                    ** exponent
                )
            )

            return (
                rf"{mantissa:.{digits}f}"
                rf"\times 10^{{{exponent}}}"
            )

        return (
            rf"${sci(mean)}"
            rf" \pm {sci(std)}$"
        )

    return (
        rf"${mean:.{digits}f}"
        rf" \pm {std:.{digits}f}$"
    )


def format_percent(
    value: float,
    digits: int = 1,
) -> str:
    if not np.isfinite(
        value
    ):
        return (
            r"$\mathrm{nan}$"
        )

    return (
        rf"${value:.{digits}f}\%$"
    )


def float_label(
    value: float,
) -> str:
    return (
        f"{value:g}"
    )


def make_latex_row(
    cells: Sequence[str],
) -> str:
    return (
        " & ".join(
            cells
        )
        + r" \\"
    )


# ============================================================================
# Output helpers
# ============================================================================

def write_latex_rows(
    path: Path,
    title: str,
    columns: Sequence[str],
    rows: Sequence[str],
    comments: Optional[
        Sequence[str]
    ] = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        f"% {title}",
        (
            "% Columns: "
            + " | ".join(
                columns
            )
        ),
    ]

    if comments:
        for comment in comments:
            lines.append(
                f"% {comment}"
            )

    lines.extend(
        rows
    )

    text = (
        "\n".join(
            lines
        )
        + "\n"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "\n"
        f"[LaTeX rows: {title}]"
    )

    for line in lines:
        print(
            line
        )


def write_csv(
    path: Path,
    records: Sequence[Dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        records
    )

    frame.to_csv(
        path,
        index=False,
    )


# ============================================================================
# CLI
# ============================================================================

def common_parser(
    description: str,
) -> argparse.ArgumentParser:
    parser = (
        argparse.ArgumentParser(
            description=description
        )
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        default=(
            DEFAULT_LOG_ROOT
        ),
        help=(
            "Root directory of the "
            "exported W&B runs."
        ),
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=(
            DEFAULT_OUT_ROOT
        ),
        help=(
            "Directory where CSV and "
            "LaTeX-row outputs are written."
        ),
    )

    parser.add_argument(
        "--early-start",
        type=int,
        default=5_000,
        help=(
            "Start training step for "
            "mechanism diagnostics."
        ),
    )

    parser.add_argument(
        "--early-end",
        type=int,
        default=100_000,
        help=(
            "End training step for "
            "mechanism diagnostics."
        ),
    )

    parser.add_argument(
        "--primary-discount",
        type=float,
        default=0.999,
        help=(
            "Discount used for the "
            "primary bias-stress table."
        ),
    )

    parser.add_argument(
        "--primary-tau",
        type=float,
        default=0.99,
        help=(
            "Expectile used for the "
            "primary bias-stress table."
        ),
    )

    parser.add_argument(
        "--last-evals",
        type=int,
        default=1,
        help=(
            "Number of final evaluations "
            "averaged within each seed. "
            "Default=1 means final evaluation."
        ),
    )

    return parser