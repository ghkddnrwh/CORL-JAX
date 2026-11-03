"""Crash-resistant, W&B-independent diagnostics for DDIQL and LDDIQL.

Install this module in ``algorithms/uk_offline/common/diagnostics.py``.
It deliberately depends only on NumPy and the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize diagnostic value of type {type(value).__name__}")


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_npz(logs: list, path: str) -> None:
    """Atomically replace the existing evaluation NPZ; never leave a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for log in logs for key in log))
    payload = {}
    for key in keys:
        values = [log.get(key, np.nan) for log in logs]
        try:
            payload[key] = np.asarray(values)
        except ValueError:
            payload[key] = np.asarray(values, dtype=object)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.npz")
    try:
        np.savez(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class LocalMetricStore:
    """Append and fsync every JSONL record; reconcile after checkpoint restore.

    Resume keeps only records at or before the restored checkpoint step.
    Records from steps that will be replayed are ARCHIVED, never destroyed.
    This prevents diagnostics from mixing two diverging training trajectories.
    """

    def __init__(self, directory: Path, filename: str = "metrics.jsonl") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename

    def append(self, timestep: int, event: str, metrics: Mapping[str, Any]) -> None:
        record = {
            "timestep": int(timestep),
            "event": str(event),
            "metrics": _jsonable(metrics),
        }
        line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def reconcile(self, restored_timestep: int) -> Dict[str, int]:
        """Archive uncommitted future records and recover an incomplete tail."""
        if not self.path.exists():
            return {"kept": 0, "archived_future": 0, "archived_invalid": 0}
        retained, future, invalid = [], [], []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    invalid.append(line)
                    continue
                try:
                    record = json.loads(line)
                    step = record["timestep"]
                    if not isinstance(step, int) or isinstance(step, bool):
                        raise ValueError("Invalid timestep")
                    if step > int(restored_timestep):
                        future.append(line)
                    else:
                        retained.append(line)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    invalid.append(line)

        if future or invalid:
            archive_dir = self.directory / "abandoned_segments"
            marker = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            nonce = uuid.uuid4().hex[:8]
            if future:
                _atomic_write_text(
                    archive_dir / f"future_after_{restored_timestep}_{marker}_{nonce}.jsonl",
                    "".join(future),
                )
            if invalid:
                _atomic_write_text(
                    archive_dir / f"incomplete_{marker}_{nonce}.jsonl",
                    "".join(invalid),
                )
            _atomic_write_text(self.path, "".join(retained))
        return {
            "kept": len(retained),
            "archived_future": len(future),
            "archived_invalid": len(invalid),
        }

    def save_manifest_once(self, content: Mapping[str, Any]) -> None:
        manifest = self.directory / "diagnostics_manifest.json"
        if not manifest.exists():
            _atomic_write_text(manifest, json.dumps(_jsonable(content), indent=2, ensure_ascii=False))


def choose_local_log_dir(
    *,
    configured_dir: Optional[str],
    checkpoints_path: Optional[str],
    run_name: str,
    seed: int,
) -> Path:
    """Share the checkpoint directory when available, otherwise isolate runs."""
    if configured_dir is not None:
        return Path(configured_dir)
    if checkpoints_path is not None:
        return Path(checkpoints_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("local_logs") / run_name / str(seed) / f"{stamp}_{uuid.uuid4().hex[:8]}"


def make_fixed_probe_numpy(
    replay_buffer: Any,
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Sample once, without disturbing NumPy's global training RNG."""
    size = int(replay_buffer._size)
    if size == 0:
        raise ValueError("Cannot build a diagnostic probe from an empty replay buffer")
    count = min(int(batch_size), size)
    probe_seed = (int(seed) + 104729) & 0xFFFFFFFF
    indices = np.random.default_rng(probe_seed).choice(size, size=count, replace=False)
    batch = {
        "observations": replay_buffer._states[indices].copy(),
        "actions": replay_buffer._actions[indices].copy(),
        "rewards": replay_buffer._rewards[indices].copy(),
        "next_observations": replay_buffer._next_states[indices].copy(),
        "dones": replay_buffer._dones[indices].copy(),
    }
    metadata = {
        "probe_size": count,
        "dataset_size": size,
        "probe_seed": probe_seed,
        "probe_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    return batch, metadata


def _safe_mean(array: np.ndarray) -> float:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _add_pair_statistics(
    name: str,
    values: np.ndarray,
    scalars: Dict[str, float],
    matrices: Dict[str, Any],
) -> None:
    """Pearson correlations across probe transitions, with constant-vector guards."""
    values = np.asarray(values, dtype=np.float64)
    n, batch_size = values.shape
    centered = values - np.mean(values, axis=1, keepdims=True)
    dot = centered @ centered.T
    lengths = np.sqrt(np.maximum(np.diag(dot), 0.0))
    denom = np.outer(lengths, lengths)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    np.divide(dot, denom, out=corr, where=denom > 1e-12)
    corr = np.clip(corr, -1.0, 1.0)
    rmses, pair_corrs = [], []
    for i in range(n):
        scalars[f"{name}/member_{i}_mean"] = float(np.mean(values[i]))
        scalars[f"{name}/member_{i}_std"] = float(np.std(values[i]))
        for j in range(i + 1, n):
            corr_value = float(corr[i, j])
            scalars[f"{name}/corr_{i}_{j}"] = corr_value
            rmse = float(np.sqrt(np.mean((values[i] - values[j]) ** 2)))
            scalars[f"{name}/pair_rmse_{i}_{j}"] = rmse
            rmses.append(rmse)
            pair_corrs.append(corr_value)
    scalars[f"{name}/corr_mean"] = _safe_mean(np.asarray(pair_corrs))
    scalars[f"{name}/corr_min"] = (
        float(np.min(np.asarray(pair_corrs)[np.isfinite(pair_corrs)]))
        if np.any(np.isfinite(pair_corrs)) else float("nan")
    )
    scalars[f"{name}/valid_corr_pairs"] = int(np.sum(np.isfinite(pair_corrs)))
    scalars[f"{name}/pair_rmse_mean"] = _safe_mean(np.asarray(rmses))
    scalars[f"{name}/ensemble_std_mean"] = float(np.mean(np.std(values, axis=0)))
    matrices[f"{name}/corr"] = corr.tolist()


def _add_sign_statistics(name: str, values: np.ndarray, scalars: Dict[str, float]) -> None:
    signs = np.asarray(values) >= 0.0
    n = signs.shape[0]
    scalars[f"{name}/positive_fraction"] = float(np.mean(signs))
    scalars[f"{name}/all_sign_agree_fraction"] = float(np.mean(np.all(signs == signs[0:1], axis=0)))
    agreements = []
    for i in range(n):
        scalars[f"{name}/member_{i}_positive_fraction"] = float(np.mean(signs[i]))
        for j in range(i + 1, n):
            agreements.append(float(np.mean(signs[i] == signs[j])))
    scalars[f"{name}/pair_sign_agreement_mean"] = _safe_mean(np.asarray(agreements))


def compute_ensemble_diagnostics(
    predictions: Mapping[str, np.ndarray],
    *,
    algorithm: str,
    timestep: int,
    delayed_update_period: int,
    iql_tau: float,
    beta: float,
) -> Dict[str, Any]:
    """Compute interpretable ensemble/advantage/filter/TD metrics.

    Each predictions[key] is [N,B], except filter_indices=[N]. The observation
    batch is fixed across steps and across DDIQL/LDDIQL for matched datasets.
    """
    data = {key: np.asarray(value) for key, value in predictions.items()}
    q = data["q"]
    target_q = data["q_target"]
    delayed_q = data["q_delayed"]
    v = data["v"]
    delayed_v = data["v_delayed"]
    next_v = data["next_v"]
    rewards = data["rewards"].reshape(-1)
    dones = data["dones"].reshape(-1)
    discount = float(data["discount"])
    n, batch_size = q.shape
    indices = np.asarray(data["filter_indices"], dtype=np.int64)
    if indices.shape != (n,) or np.any((indices < 0) | (indices >= n)):
        raise ValueError("Invalid filter_indices in diagnostic model state")

    delayed_adv = delayed_q - delayed_v
    loo_filter = (
        (np.sum(delayed_adv, axis=0, keepdims=True) - delayed_adv) / (n - 1)
        if n > 1 else delayed_adv.copy()
    )
    if n > 1:
        delayed_round = int(timestep) // int(delayed_update_period)
        cross_shift = 1 + delayed_round % (n - 1)
        cross_indices = (np.arange(n) + cross_shift) % n
    else:
        cross_indices = np.array([0], dtype=np.int64)
    cross_filter = delayed_adv[cross_indices, :]
    active_filter = loo_filter if algorithm.upper() == "LDDIQL" else delayed_adv[indices, :]
    td_residual = q - (rewards[None, :] + discount * (1.0 - dones[None, :]) * next_v)
    value_residual = target_q - v
    advantage_scale = max(float(np.std(delayed_adv)), 1e-8)

    scalars: Dict[str, float] = {
        "probe/batch_size": int(batch_size),
        "probe/ensemble_size": int(n),
        "filter/active_vs_loo_sign_disagree_fraction": float(
            np.mean((active_filter < 0) != (loo_filter < 0))
        ),
        "filter/cross_vs_loo_sign_disagree_fraction": float(
            np.mean((cross_filter < 0) != (loo_filter < 0))
        ),
        "filter/active_vs_self_sign_disagree_fraction": float(
            np.mean((active_filter < 0) != (delayed_adv < 0))
        ),
        "filter/active_vs_value_residual_sign_disagree_fraction": float(
            np.mean((active_filter < 0) != (value_residual < 0))
        ),
        "filter/active_weight_mean": float(np.mean(np.where(active_filter < 0, 1.0 - iql_tau, iql_tau))),
        "filter/loo_weight_mean": float(np.mean(np.where(loo_filter < 0, 1.0 - iql_tau, iql_tau))),
        "filter/active_abs_margin_mean": float(np.mean(np.abs(active_filter))),
        "filter/loo_abs_margin_mean": float(np.mean(np.abs(loo_filter))),
        "filter/active_small_margin_fraction": float(
            np.mean(np.abs(active_filter) <= 0.1 * advantage_scale)
        ),
        "filter/loo_small_margin_fraction": float(
            np.mean(np.abs(loo_filter) <= 0.1 * advantage_scale)
        ),
        "filter/active_normalized_margin_mean": float(
            np.mean(np.abs(active_filter)) / advantage_scale
        ),
        "filter/loo_normalized_margin_mean": float(
            np.mean(np.abs(loo_filter)) / advantage_scale
        ),
        "filter/active_is_self_fraction": float(np.mean(indices == np.arange(n)))
        if algorithm.upper() != "LDDIQL" else float(n == 1),
        "critic/delayed_q_target_mae": float(np.mean(np.abs(target_q - delayed_q))),
        "critic/delayed_v_online_mae": float(np.mean(np.abs(v - delayed_v))),
        "critic/td_residual_rmse": float(np.sqrt(np.mean(td_residual ** 2))),
        "critic/value_residual_rmse": float(np.sqrt(np.mean(value_residual ** 2))),
    }
    matrices: Dict[str, Any] = {}
    for name, values in (
        ("q", q), ("q_target", target_q), ("v", v),
        ("delayed_adv", delayed_adv), ("td_residual", td_residual),
        ("value_residual", value_residual), ("active_filter", active_filter),
        ("loo_filter", loo_filter),
    ):
        _add_pair_statistics(name, values, scalars, matrices)

    for name, values in (
        ("delayed_adv", delayed_adv),
        ("active_filter", active_filter),
        ("loo_filter", loo_filter),
        ("cross_filter", cross_filter),
    ):
        _add_sign_statistics(name, values, scalars)

    actor_adv = np.mean(target_q, axis=0) - np.mean(v, axis=0)
    scalars["actor/adv_mean"] = float(np.mean(actor_adv))
    scalars["actor/adv_std"] = float(np.std(actor_adv))
    scalars["actor/exp_weight_clipped_fraction"] = float(
        np.mean(beta * actor_adv >= np.log(100.0))
    )
    for i in range(n):
        scalars[f"critic/member_{i}_td_rmse"] = float(np.sqrt(np.mean(td_residual[i] ** 2)))
        scalars[f"critic/member_{i}_value_rmse"] = float(np.sqrt(np.mean(value_residual[i] ** 2)))
        scalars[f"filter/member_{i}_active_weight_mean"] = float(
            np.mean(np.where(active_filter[i] < 0, 1.0 - iql_tau, iql_tau))
        )

    return {
        "scalars": scalars,
        "matrices": matrices,
        "active_filter_indices": indices.tolist() if algorithm.upper() != "LDDIQL" else None,
        "counterfactual_cross_indices": cross_indices.tolist(),
    }
