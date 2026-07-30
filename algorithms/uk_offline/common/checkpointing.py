"""Reusable automatic-resume utilities for offline-RL training scripts.

The manager is framework-agnostic. Algorithms provide a serializable training
state and a callback that restores it. The module owns run-directory state,
atomic checkpoint writes, hyperparameter identity checks, evaluation logs,
Python/NumPy/PyTorch RNG state, and checkpoint-synchronised W&B delivery.

W&B metrics are never uploaded ahead of a recoverable local model. Training
metrics are buffered in memory, embedded in ``latest_checkpoint.pkl``, and only
flushed after that checkpoint has been atomically committed. W&B is resumed
with the public ``id=...`` + ``resume="allow"`` API. A custom
``training/timestep`` metric is used as the chart x-axis, so W&B's internal
history step may continue monotonically even when local training restarts from
a checkpoint.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

LATEST_CHECKPOINT_NAME = "latest_checkpoint.pkl"
FINAL_CHECKPOINT_NAME = "checkpoint.pkl"
TRAINING_STATUS_NAME = "training_status.yaml"
DEFAULT_CHECKPOINT_VERSION = 2

PathLike = Union[str, Path]

DEFAULT_IDENTITY_IGNORED_FIELDS = frozenset(
    {
        "device",
        "checkpoints_path",
        "load_model",
        "mode",
        "hyperparams_path",
        "use_hyperparams",
        "log_wandb",
        "log_every",
        "checkpoint_freq",
        "save_final_model",
        "save_best_model",
        "project",
        "group",
        "name",
        "wandb_entity",
        "actor_refit_dir_name",
    }
)


@dataclass(frozen=True)
class RunPreparation:
    mode: str  # new, resume, completed
    message: str

    @property
    def is_resuming(self) -> bool:
        return self.mode == "resume"

    @property
    def is_completed(self) -> bool:
        return self.mode == "completed"


@dataclass(frozen=True)
class WandbResumeState:
    run_id: Optional[str]
    entity: Optional[str]
    project: Optional[str]
    last_flushed_sequence: int = 0
    last_flushed_timestep: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "entity": self.entity,
            "project": self.project,
            "last_flushed_sequence": int(self.last_flushed_sequence),
            "last_flushed_timestep": self.last_flushed_timestep,
        }


@dataclass
class PendingWandbLog:
    sequence: int
    timestep: int
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "timestep": int(self.timestep),
            "metrics": copy.deepcopy(self.metrics),
        }


def _fsync_parent_directory(path: PathLike) -> None:
    directory = Path(path).parent
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def save_pickle(path: PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: PathLike) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_atomic(path: PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_yaml_atomic(path: PathLike, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w") as f:
            yaml.safe_dump(dict(data), f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_logs_npz(logs: List[Dict[str, Any]], path: PathLike) -> None:
    if not logs:
        return
    all_keys = sorted({key for log in logs for key in log})
    data: Dict[str, np.ndarray] = {}
    for key in all_keys:
        values = [log.get(key, np.nan) for log in logs]
        try:
            data[key] = np.asarray(values)
        except (TypeError, ValueError):
            data[key] = np.asarray(values, dtype=object)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "wb") as f:
            np.savez(f, **data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def remove_stale_atomic_files(directory: PathLike) -> None:
    directory = Path(directory)
    for name in (
        "config.yaml",
        TRAINING_STATUS_NAME,
        LATEST_CHECKPOINT_NAME,
        FINAL_CHECKPOINT_NAME,
        "eval_logs.npz",
    ):
        path = directory / f".{name}.tmp"
        if path.exists():
            path.unlink()
            _fsync_parent_directory(path)


def evaluation_is_due(timestep: int, eval_freq: int) -> bool:
    return int(timestep) > 0 and int(timestep) % int(eval_freq) == 0


def find_eval_log(eval_logs: List[Dict[str, Any]], timestep: int) -> Optional[Dict[str, Any]]:
    for log in reversed(eval_logs):
        if int(log.get("timestep", -1)) == int(timestep):
            return log
    return None


def upsert_eval_log(eval_logs: List[Dict[str, Any]], eval_log: Dict[str, Any]) -> None:
    timestep = int(eval_log["timestep"])
    eval_logs[:] = [
        log for log in eval_logs if int(log.get("timestep", -1)) != timestep
    ]
    eval_logs.append(dict(eval_log))
    eval_logs.sort(key=lambda log: int(log.get("timestep", -1)))


def best_eval_metric(eval_logs: List[Dict[str, Any]]) -> float:
    best = -np.inf
    for log in eval_logs:
        candidates = (
            log.get("eval/success_rate", np.nan),
            log.get("eval/d4rl_normalized_score_mean", np.nan),
            log.get("eval/normalized_score_mean", np.nan),
            log.get("eval/reward_mean", np.nan),
        )
        metric = next((float(value) for value in candidates if np.isfinite(value)), np.nan)
        if np.isfinite(metric):
            best = max(best, metric)
    return float(best)


def _capture_framework_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            state["torch_cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_framework_rng_state(state: Mapping[str, Any]) -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


def resolve_checkpoint_path(
    load_model: PathLike,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
    checkpoint_names: Sequence[str] = (FINAL_CHECKPOINT_NAME, "best_checkpoint.pkl"),
) -> Tuple[Path, Path]:
    load_path = Path(load_model)
    if load_path.is_file():
        return load_path.parent, load_path
    if not load_path.exists():
        raise FileNotFoundError(f"load_model path does not exist: {load_path}")

    candidates: List[Path] = []
    for name in checkpoint_names:
        candidates.append(load_path / name)
    if run_name is not None and seed is not None:
        for name in checkpoint_names:
            candidates.append(load_path / run_name / str(seed) / name)
    if run_name is not None:
        run_dir = load_path / run_name
        for name in checkpoint_names:
            candidates.extend(sorted(run_dir.glob(f"*/{name}")))
    for name in checkpoint_names:
        candidates.extend(sorted(load_path.glob(f"*/{name}")))
        candidates.extend(sorted(load_path.glob(f"*/*/{name}")))

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.exists():
            seen.add(candidate)
            unique.append(candidate)
    if not unique:
        raise FileNotFoundError(f"No checkpoint found under: {load_path}")
    if len(unique) > 1:
        found = "\n".join(str(path) for path in unique)
        raise FileNotFoundError(
            f"Multiple checkpoint files found under {load_path}. "
            f"Provide a more specific --load_model path.\n{found}"
        )
    return unique[0].parent, unique[0]


class TrainingCheckpointManager:
    """Manage one replaceable progress checkpoint for a training run."""

    def __init__(
        self,
        run_dir: PathLike,
        current_config: Mapping[str, Any],
        default_config: Mapping[str, Any],
        max_timesteps: int,
        checkpoint_type: str,
        identity_ignored_fields: Iterable[str] = DEFAULT_IDENTITY_IGNORED_FIELDS,
        critical_config_fields: Optional[Sequence[str]] = None,
        checkpoint_version: int = DEFAULT_CHECKPOINT_VERSION,
        accepted_checkpoint_versions: Optional[Iterable[int]] = None,
        wandb_enabled: bool = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        final_checkpoint_name: str = FINAL_CHECKPOINT_NAME,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.current_config = dict(current_config)
        self.default_config = dict(default_config)
        self.max_timesteps = int(max_timesteps)
        self.checkpoint_type = str(checkpoint_type)
        self.identity_ignored_fields = frozenset(identity_ignored_fields)
        self.critical_config_fields = tuple(
            critical_config_fields
            if critical_config_fields is not None
            else sorted(set(self.current_config) - self.identity_ignored_fields)
        )
        self.checkpoint_version = int(checkpoint_version)
        self.accepted_checkpoint_versions = frozenset(
            accepted_checkpoint_versions or (self.checkpoint_version,)
        )

        self.config_path = self.run_dir / "config.yaml"
        self.status_path = self.run_dir / TRAINING_STATUS_NAME
        self.latest_checkpoint_path = self.run_dir / LATEST_CHECKPOINT_NAME
        self.final_checkpoint_name = str(final_checkpoint_name)
        self.final_checkpoint_path = self.run_dir / self.final_checkpoint_name
        self.eval_logs_path = self.run_dir / "eval_logs.npz"
        self.preparation_mode: Optional[str] = None

        self.wandb_enabled = bool(wandb_enabled)
        self.wandb_state = WandbResumeState(
            run_id=uuid.uuid4().hex if self.wandb_enabled else None,
            entity=wandb_entity,
            project=wandb_project,
        )
        self._wandb_module: Any = None
        self._wandb_run: Any = None
        self._pending_wandb_logs: List[PendingWandbLog] = []
        self._next_wandb_sequence = 1
        self._remote_wandb_sequence = 0

    @property
    def critical_config(self) -> Dict[str, Any]:
        return {key: self.current_config.get(key) for key in self.critical_config_fields}

    @property
    def pending_wandb_log_count(self) -> int:
        return len(self._pending_wandb_logs)

    def _normalized_config(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(self.default_config)
        for key, value in raw.items():
            if key not in normalized:
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            normalized[key] = value
        return normalized

    def _identity(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = self._normalized_config(raw)
        return {
            key: normalized[key]
            for key in sorted(normalized)
            if key not in self.identity_ignored_fields
        }

    def _assert_config_matches(self) -> None:
        with open(self.config_path, "r") as f:
            saved = yaml.safe_load(f) or {}
        if not isinstance(saved, dict):
            raise ValueError(f"Invalid saved config: {self.config_path}")
        old, new = self._identity(saved), self._identity(self.current_config)
        mismatches = {
            key: (old.get(key), new.get(key))
            for key in sorted(set(old) | set(new))
            if old.get(key) != new.get(key)
        }
        if mismatches:
            lines = "\n".join(
                f"  - {key}: saved={saved_value!r}, current={current_value!r}"
                for key, (saved_value, current_value) in mismatches.items()
            )
            raise ValueError(
                "The output directory belongs to a different training configuration.\n"
                f"{lines}"
            )

    def _load_status(self) -> Dict[str, Any]:
        if not self.status_path.exists():
            return {}
        with open(self.status_path, "r") as f:
            status = yaml.safe_load(f) or {}
        if not isinstance(status, dict):
            raise ValueError(f"Invalid status file: {self.status_path}")
        return status

    def _load_wandb_state(self, raw: Any) -> WandbResumeState:
        if not isinstance(raw, Mapping) or not raw.get("run_id"):
            return WandbResumeState(
                run_id=uuid.uuid4().hex if self.wandb_enabled else None,
                entity=self.wandb_state.entity,
                project=self.wandb_state.project,
            )
        # ``resume_timestep`` is accepted for backward compatibility with the
        # earlier private-preview rewind implementation.
        old_resume_timestep = raw.get("resume_timestep")
        last_timestep = raw.get("last_flushed_timestep", old_resume_timestep)
        return WandbResumeState(
            run_id=str(raw["run_id"]),
            entity=raw.get("entity", self.wandb_state.entity),
            project=raw.get("project", self.wandb_state.project),
            last_flushed_sequence=int(raw.get("last_flushed_sequence", 0) or 0),
            last_flushed_timestep=None if last_timestep is None else int(last_timestep),
        )

    @staticmethod
    def _load_pending_wandb_logs(raw: Any) -> List[PendingWandbLog]:
        if not isinstance(raw, list):
            return []
        result: List[PendingWandbLog] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            metrics = item.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            result.append(
                PendingWandbLog(
                    sequence=int(item.get("sequence", len(result) + 1)),
                    timestep=int(item.get("timestep", 0)),
                    metrics=dict(metrics),
                )
            )
        result.sort(key=lambda item: item.sequence)
        return result

    def _update_status(self, status: str, timestep: int, final_checkpoint: Optional[str] = None) -> None:
        save_yaml_atomic(
            self.status_path,
            {
                "status": status,
                "timestep": int(timestep),
                "max_timesteps": self.max_timesteps,
                "latest_checkpoint": LATEST_CHECKPOINT_NAME if status != "completed" else None,
                "final_checkpoint": final_checkpoint,
                "wandb": self.wandb_state.to_dict() if self.wandb_state.run_id else None,
                "pending_wandb_logs": len(self._pending_wandb_logs),
            },
        )

    def prepare(self) -> RunPreparation:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_atomic_files(self.run_dir)
        custom_final_temporary = self.run_dir / f".{self.final_checkpoint_name}.tmp"
        if custom_final_temporary.exists():
            custom_final_temporary.unlink()
            _fsync_parent_directory(custom_final_temporary)
        if not any(self.run_dir.iterdir()):
            save_yaml_atomic(self.config_path, self.current_config)
            self._update_status("running", 0)
            self.preparation_mode = "new"
            return RunPreparation("new", "No saved run found. Starting from timestep 0.")

        if not self.config_path.exists():
            raise FileExistsError(
                f"Output directory is not empty but has no config.yaml: {self.run_dir}"
            )
        self._assert_config_matches()
        status = self._load_status()
        self.wandb_state = self._load_wandb_state(status.get("wandb"))

        if status.get("status") == "completed":
            if self.latest_checkpoint_path.exists():
                self.latest_checkpoint_path.unlink()
                _fsync_parent_directory(self.latest_checkpoint_path)
            self.preparation_mode = "completed"
            return RunPreparation("completed", "This matching training run is already complete.")

        if self.latest_checkpoint_path.exists():
            self.preparation_mode = "resume"
            return RunPreparation("resume", "Found an incomplete checkpoint. Resuming automatically.")

        if self.final_checkpoint_path.exists():
            self._update_status("completed", self.max_timesteps, self.final_checkpoint_name)
            self.preparation_mode = "completed"
            return RunPreparation("completed", "A final checkpoint already exists. Nothing to do.")

        if status.get("status") in (None, "running", "interrupted") and int(status.get("timestep", 0)) == 0:
            self.preparation_mode = "new"
            return RunPreparation(
                "new",
                "The previous process stopped during initialization. Restarting from timestep 0.",
            )

        raise RuntimeError(
            "The matching run directory has no recoverable model checkpoint: "
            f"{self.run_dir}"
        )

    def _persist_wandb_identity_to_progress_checkpoint(self) -> None:
        if not self.latest_checkpoint_path.exists():
            return
        payload = load_pickle(self.latest_checkpoint_path)
        if not isinstance(payload, dict) or payload.get("checkpoint_type") != self.checkpoint_type:
            return
        payload["wandb"] = self.wandb_state.to_dict() if self.wandb_state.run_id else None
        save_pickle_atomic(self.latest_checkpoint_path, payload)

    @staticmethod
    def _summary_get(summary: Any, key: str, default: Any) -> Any:
        if summary is None:
            return default
        try:
            return summary.get(key, default)
        except Exception:
            try:
                return summary[key]
            except Exception:
                return default

    def initialize_wandb(self, wandb_module: Any, config: Mapping[str, Any], code_root: PathLike = ".") -> Any:
        """Start or resume the same W&B run using the public resume API.

        No run rewind is required. W&B's internal history step is allowed to
        continue from its latest row, while ``training/timestep`` is registered
        as the x-axis for all training metrics.
        """
        if not self.wandb_enabled:
            return None
        self._wandb_module = wandb_module
        state = self.wandb_state
        if not state.run_id:
            state = WandbResumeState(
                uuid.uuid4().hex,
                state.entity,
                state.project or config.get("project"),
            )
            self.wandb_state = state

        run = wandb_module.init(
            config=dict(config),
            entity=state.entity,
            project=state.project or config.get("project"),
            group=config.get("group"),
            name=config.get("name"),
            id=state.run_id,
            resume="allow",
        )
        self._wandb_run = run
        summary = getattr(run, "summary", None)
        remote_sequence = int(
            self._summary_get(summary, "resume/last_sequence", state.last_flushed_sequence) or 0
        )
        remote_timestep_raw = self._summary_get(
            summary,
            "resume/last_checkpoint_timestep",
            state.last_flushed_timestep,
        )
        remote_timestep = None if remote_timestep_raw is None else int(remote_timestep_raw)
        self._remote_wandb_sequence = max(remote_sequence, int(state.last_flushed_sequence))
        self.wandb_state = WandbResumeState(
            run_id=str(getattr(run, "id", None) or state.run_id),
            entity=getattr(run, "entity", None) or state.entity,
            project=getattr(run, "project", None) or state.project or config.get("project"),
            last_flushed_sequence=self._remote_wandb_sequence,
            last_flushed_timestep=remote_timestep,
        )

        if hasattr(run, "define_metric"):
            try:
                run.define_metric("training/timestep")
                run.define_metric("*", step_metric="training/timestep")
            except Exception as exc:
                print(f"Warning: failed to configure W&B training/timestep axis: {exc}")
        if hasattr(run, "log_code"):
            run.log_code(str(code_root))

        existing = self._load_status()
        self._update_status(
            existing.get("status", "running"),
            int(existing.get("timestep", 0)),
            existing.get("final_checkpoint"),
        )
        self._persist_wandb_identity_to_progress_checkpoint()

        # Any records present here were already protected by the local
        # checkpoint from which this process resumed.
        self.flush_wandb_logs()
        return run

    def initialize_fresh_wandb(self, wandb_module: Any, config: Mapping[str, Any], code_root: PathLike = ".") -> Any:
        """Fallback to a replacement W&B run while preserving local training."""
        if not self.wandb_enabled:
            return None
        self.wandb_state = WandbResumeState(
            uuid.uuid4().hex,
            self.wandb_state.entity,
            self.wandb_state.project or config.get("project"),
        )
        self._remote_wandb_sequence = 0
        return self.initialize_wandb(wandb_module, config, code_root)

    def log_wandb(self, metrics: Mapping[str, Any], step: int) -> None:
        """Buffer a metric record; do not contact W&B until a checkpoint exists."""
        if not self.wandb_enabled:
            return
        step = int(step)
        clean_metrics = dict(metrics)
        if self._pending_wandb_logs and self._pending_wandb_logs[-1].timestep == step:
            self._pending_wandb_logs[-1].metrics.update(clean_metrics)
            return
        self._pending_wandb_logs.append(
            PendingWandbLog(
                sequence=self._next_wandb_sequence,
                timestep=step,
                metrics=clean_metrics,
            )
        )
        self._next_wandb_sequence += 1

    def _buffer_checkpoint_marker(self, timestep: int) -> None:
        self.log_wandb(
            {
                "checkpoint/saved": 1,
                "resume/last_checkpoint_timestep": int(timestep),
            },
            int(timestep),
        )

    def flush_wandb_logs(self) -> int:
        """Upload checkpoint-protected records to W&B in sequence order.

        Calls use W&B's implicit monotonically increasing history step. The
        original optimizer step is carried by ``training/timestep``. Each row
        also records a durable sequence number, allowing a resumed process to
        skip rows that were already accepted before an interruption.
        """
        if not self.wandb_enabled or self._wandb_run is None:
            return 0

        sent = 0
        for item in list(self._pending_wandb_logs):
            if item.sequence <= self._remote_wandb_sequence:
                continue
            payload = dict(item.metrics)
            payload["training/timestep"] = int(item.timestep)
            payload["resume/log_sequence"] = int(item.sequence)
            payload["resume/last_sequence"] = int(item.sequence)
            self._wandb_run.log(payload)
            self._remote_wandb_sequence = int(item.sequence)
            self.wandb_state = WandbResumeState(
                self.wandb_state.run_id,
                self.wandb_state.entity,
                self.wandb_state.project,
                last_flushed_sequence=self._remote_wandb_sequence,
                last_flushed_timestep=(
                    int(item.timestep)
                    if payload.get("checkpoint/saved") == 1
                    else self.wandb_state.last_flushed_timestep
                ),
            )
            sent += 1

        self._pending_wandb_logs = [
            item
            for item in self._pending_wandb_logs
            if item.sequence > self._remote_wandb_sequence
        ]
        existing = self._load_status()
        if existing:
            self._update_status(
                existing.get("status", "running"),
                int(existing.get("timestep", 0)),
                existing.get("final_checkpoint"),
            )
        return sent

    # Backward-compatible alias used by older integrations.
    def flush_wandb(self, checkpoint_timestep: Optional[int] = None) -> None:
        if checkpoint_timestep is not None:
            self._buffer_checkpoint_marker(int(checkpoint_timestep))
        self.flush_wandb_logs()

    def _payload(self, timestep: int, trainer_state: Any, eval_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "checkpoint_type": self.checkpoint_type,
            "version": self.checkpoint_version,
            "timestep": int(timestep),
            "max_timesteps": self.max_timesteps,
            "critical_config": self.critical_config,
            "trainer_state": trainer_state,
            "eval_logs": copy.deepcopy(eval_logs),
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "framework_random_state": _capture_framework_rng_state(),
            "wandb": self.wandb_state.to_dict() if self.wandb_state.run_id else None,
            "pending_wandb_logs": [item.to_dict() for item in self._pending_wandb_logs],
            "next_wandb_sequence": int(self._next_wandb_sequence),
        }

    def save_progress(self, timestep: int, trainer_state: Any, eval_logs: List[Dict[str, Any]], status: str = "running") -> None:
        """Commit model first, then deliver only checkpoint-protected W&B rows."""
        if status not in ("running", "interrupted"):
            raise ValueError(f"Invalid progress status: {status}")
        timestep = int(timestep)
        if self.wandb_enabled:
            self._buffer_checkpoint_marker(timestep)

        # The local checkpoint containing all pending W&B rows is the commit
        # point. Nothing is uploaded before this atomic replace succeeds.
        save_pickle_atomic(
            self.latest_checkpoint_path,
            self._payload(timestep, trainer_state, eval_logs),
        )
        self._update_status(status, timestep)
        save_logs_npz(eval_logs, self.eval_logs_path)

        if self.wandb_enabled and self._wandb_run is not None:
            try:
                sent = self.flush_wandb_logs()
                if sent:
                    print(
                        f"Flushed {sent} checkpoint-protected W&B log record(s) "
                        f"through training timestep {timestep}."
                    )
            except Exception as exc:
                # The checkpoint still contains every unsent record. A later
                # process will resume the same run and retry them.
                print(f"Warning: failed to flush checkpoint-protected W&B logs: {exc}")
                print("Local training state is safe; W&B delivery will be retried on resume.")

    def restore(
        self,
        load_trainer_state: Callable[[Any], None],
        get_restored_timestep: Callable[[], int],
    ) -> Tuple[int, List[Dict[str, Any]], int]:
        payload = load_pickle(self.latest_checkpoint_path)
        if not isinstance(payload, dict) or payload.get("checkpoint_type") != self.checkpoint_type:
            raise ValueError(f"Unsupported progress checkpoint: {self.latest_checkpoint_path}")
        version = int(payload.get("version", -1))
        if version not in self.accepted_checkpoint_versions:
            raise ValueError(f"Unsupported checkpoint version: {version}")
        saved_critical = payload.get("critical_config", {})
        mismatch = {
            key: (saved_critical.get(key), value)
            for key, value in self.critical_config.items()
            if saved_critical.get(key) != value
        }
        if mismatch:
            raise ValueError(f"Checkpoint-critical config mismatch: {mismatch}")

        load_trainer_state(payload["trainer_state"])
        restored = int(get_restored_timestep())
        declared = int(payload.get("timestep", restored))
        if restored != declared:
            raise ValueError(
                f"Checkpoint timestep mismatch: metadata={declared}, trainer={restored}"
            )
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        if "python_random_state" in payload:
            random.setstate(payload["python_random_state"])
        _restore_framework_rng_state(payload.get("framework_random_state", {}))
        logs = payload.get("eval_logs", [])
        if not isinstance(logs, list):
            raise ValueError("eval_logs in checkpoint must be a list")

        self.wandb_state = self._load_wandb_state(payload.get("wandb"))
        self._pending_wandb_logs = self._load_pending_wandb_logs(
            payload.get("pending_wandb_logs", [])
        )
        inferred_next = (
            max((item.sequence for item in self._pending_wandb_logs), default=0) + 1
        )
        self._next_wandb_sequence = max(
            int(payload.get("next_wandb_sequence", inferred_next)),
            inferred_next,
            int(self.wandb_state.last_flushed_sequence) + 1,
        )
        self._remote_wandb_sequence = int(self.wandb_state.last_flushed_sequence)
        self._update_status("running", restored)
        return restored, logs, version

    def complete(
        self,
        timestep: int,
        final_state: Any,
        save_final_model: bool,
        eval_logs: Optional[List[Dict[str, Any]]] = None,
        final_saver: Optional[Callable[[Path, Any], None]] = None,
    ) -> Optional[Path]:
        timestep = int(timestep)
        logs = [] if eval_logs is None else eval_logs

        # Commit a recoverable final progress state before sending the final
        # buffered W&B records. If the process stops afterwards, the next run
        # simply restores this final timestep and completes cleanup.
        self.save_progress(
            timestep=timestep,
            trainer_state=final_state,
            eval_logs=logs,
            status="running",
        )

        final_name: Optional[str] = None
        final_path: Optional[Path] = None
        if save_final_model:
            if final_saver is None:
                save_pickle_atomic(self.final_checkpoint_path, final_state)
            else:
                temporary = _temporary_path(self.final_checkpoint_path)
                try:
                    final_saver(temporary, final_state)
                    os.replace(temporary, self.final_checkpoint_path)
                    _fsync_parent_directory(self.final_checkpoint_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            final_name = self.final_checkpoint_name
            final_path = self.final_checkpoint_path
        if logs:
            save_logs_npz(logs, self.eval_logs_path)
        self._update_status("completed", timestep, final_name)
        if self.latest_checkpoint_path.exists():
            self.latest_checkpoint_path.unlink()
            _fsync_parent_directory(self.latest_checkpoint_path)
        return final_path

    def close_wandb(self) -> None:
        """Do not flush unsaved metrics during generic shutdown.

        Only ``save_progress``/``complete`` may upload metrics, because those
        methods first establish a matching recoverable model checkpoint.
        """
        return None
