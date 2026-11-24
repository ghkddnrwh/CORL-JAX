"""Hand the preserved batch-size study to its supervisor after verified EMA completion.

This process uses no accelerator. It never edits either study's learner sources,
and keeps the old batch STOP in place until the dependency has fully finished.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.run_stability_study_v6 import atomic, birth


SUPERVISOR = "utils.codex.ukjo1.error_propagation.flow_iql_research.supervise_stability_study_v6"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    if not path.exists():
        return None
    return dict(sha256=sha(path), mtime_ns=path.stat().st_mtime_ns)


def live(record):
    return record.get("birth") is not None and birth(record.get("pid", -1)) == record["birth"]


def dependency_gate(plan):
    """Require successful final analysis, all evaluations, and an exited controller."""
    root = Path(plan["dependency_root"])
    if (root / "STOP").exists():
        return "cancelled", "EMA STOP exists"
    status = read(root / "controller_status.json")
    if status.get("phase") in ("cancelled", "stopping", "draining"):
        return "cancelled", "EMA was stopped"
    if status.get("phase") in ("incomplete", "verification_failed"):
        return "blocked", "EMA did not finish successfully"
    if status.get("phase") != "complete":
        return "waiting_for_ema", "EMA training or evaluation is still running"
    if status.get("analysis_exit_code") is None:
        return "waiting_for_ema", "EMA final analysis has not committed"
    if status["analysis_exit_code"] != 0:
        return "blocked", "EMA final analysis failed"
    expected = plan["dependency_expected"]
    for field in ("training_complete", "expected_training", "evaluations_complete", "expected_evaluations"):
        if status.get(field) != expected[field]:
            return "blocked", "EMA completion count mismatch: " + field
    if status.get("registration") != "registration_v2.json":
        return "blocked", "EMA registration changed"
    if status.get("running") or status.get("evaluating") or status.get("pending") or live(status):
        return "waiting_for_ema", "EMA controller or workers are still draining"
    summary = read(root / "analysis/summary_final_v2.json")
    if not summary.get("complete") or summary.get("failures"):
        return "blocked", "EMA final summary did not pass verification"
    if (summary.get("training_complete"), summary.get("total_cells"), summary.get("primary_evaluations")) != (
        expected["total_cells"], expected["total_cells"], expected["primary_evaluations"]
    ):
        return "blocked", "EMA final summary count mismatch"
    return "ready", "EMA training, evaluation and final analysis completed"


def verify_files(plan):
    """Refuse source, registration, reused-result or saved-checkpoint drift."""
    for name, digest in plan["required_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("Queued artifact changed: " + name)
    registered = read(Path(plan["batch_root"]) / "registration_v6.json")
    assert registered["methods"] == ["control", "critic_batch1024"]
    assert registered["training_seeds"] == [0, 1]
    assert registered["environments"] == plan["environments"]
    assert registered["discount"] == .999 and registered["sample_counts"] == [9]
    assert len(registered["jobs"]) == 8 and len(registered["reused_jobs"]) == 8
    assert read(Path(plan["preflight_path"]))["status"] == "PASS"


def prepare_activation(queue, plan):
    """Idempotently archive cancellation metadata and requeue only authorized jobs."""
    root = Path(plan["batch_root"])
    if (queue / "STOP").exists():
        raise RuntimeError("Successor queue was stopped")
    prepared = queue / "activation_prepared.json"
    current_stop = fingerprint(root / "STOP")
    if current_stop is not None and current_stop != plan["original_batch_stop"]:
        raise RuntimeError("Batch STOP was changed after this successor was authorized")
    if prepared.exists():
        return
    if current_stop != plan["original_batch_stop"]:
        raise RuntimeError("Original batch STOP is missing before activation")
    previous = read(root / "controller_status.json")
    if live(previous) or previous.get("running") or previous.get("evaluating"):
        raise RuntimeError("Batch controller or workers already active")
    supervisor = read(root / "supervisor_status.json")
    if supervisor.get("pid") and birth(supervisor["pid"]) is not None:
        raise RuntimeError("Batch supervisor already active")
    archive = queue / "activation_archive"
    archive.mkdir(exist_ok=True)
    for filename in ("STOP", "controller_status.json", "supervisor_status.json"):
        source, target = root / filename, archive / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    registered = read(root / "registration_v6.json")
    for job in registered["jobs"]:
        marker = Path(job["output"]) / "cancelled.json"
        if marker.exists():
            target = archive / (job["id"] + "_cancelled.json")
            if not target.exists():
                shutil.copy2(marker, target)
            marker.unlink()
        if (Path(job["output"]) / "stopped.json").exists():
            raise RuntimeError("Non-finite batch run needs review: " + job["id"])
    atomic(root / "controller_status.json", dict(
        phase="queued", time=time.time(), running={}, evaluating={}, finished={},
        evaluation_finished={}, attempts={}, evaluation_attempts={}, retry_history=[],
        pending=[job["id"] for job in registered["jobs"]],
        registration="registration_v6.json", successor_queue=str(queue),
        authorization="Resume batch256/batch1024 study for seeds 0/1 only after the current EMA study."
    ))
    atomic(root / "supervisor_status.json", dict(phase="queued", time=time.time(), successor_queue=str(queue)))
    atomic(prepared, dict(time=time.time(), archive=str(archive), jobs=8,
                          preserved_checkpoints=8, reused_controls=8))


def start_supervisor(queue, plan):
    """Support recovery if the queue process exits during the handoff."""
    root = Path(plan["batch_root"])
    record = read(queue / "launched_supervisor.json")
    if live(record):
        command = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b"\0")
        assert SUPERVISOR.encode() in command and str(root).encode() in command
        return record
    if (queue / "STOP").exists():
        raise RuntimeError("Successor queue was stopped")
    stop = root / "STOP"
    if stop.exists():
        if fingerprint(stop) != plan["original_batch_stop"]:
            raise RuntimeError("A new batch STOP prevents automatic restart")
        stop.unlink()
    elif not (queue / "activation_prepared.json").exists():
        raise RuntimeError("Missing activation transaction")
    with (queue / "batch_supervisor_stdout.log").open("a") as stream:
        worker = subprocess.Popen(
            [plan["python"], "-u", "-m", SUPERVISOR, "--root", str(root)],
            cwd=plan["repository"], stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = dict(pid=worker.pid, birth=birth(worker.pid), time=time.time(), module=SUPERVISOR)
    atomic(queue / "launched_supervisor.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Read completion gate only; never launch")
    args = parser.parse_args()
    queue = args.queue.resolve()
    plan = read(queue / "queue_plan.json")
    if args.check:
        phase, reason = dependency_gate(plan)
        print(json.dumps(dict(phase=phase, reason=reason)))
        return
    lease = (queue / "queue.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, **fields):
        value = dict(pid=os.getpid(), birth=birth(os.getpid()), time=time.time(), phase=phase,
                     dependency_root=plan["dependency_root"], batch_root=plan["batch_root"], **fields)
        atomic(queue / "status.json", value)
        return value

    def stop(signum, frame):
        (queue / "STOP").write_text("Successor queue received a stop signal.\n")
        status("cancelled", reason="Queue received signal " + str(signum))
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        verify_files(plan)
        while True:
            if (queue / "STOP").exists():
                status("cancelled", reason="Successor STOP exists")
                return
            if fingerprint(Path(plan["batch_root"]) / "STOP") not in (None, plan["original_batch_stop"]):
                status("cancelled", reason="Batch STOP changed after queue creation")
                return
            phase, reason = dependency_gate(plan)
            status(phase, reason=reason)
            if phase in ("cancelled", "blocked"):
                return
            if phase == "ready":
                break
            time.sleep(15)
        verify_files(plan)
        prepare_activation(queue, plan)
        if dependency_gate(plan)[0] != "ready":
            raise RuntimeError("EMA completion gate changed before launch")
        supervisor = start_supervisor(queue, plan)
        status("starting_batch", supervisor=supervisor)
        # The existing supervisor owns recovery and automatic slot refill thereafter.
        for _ in range(24):
            current = read(Path(plan["batch_root"]) / "controller_status.json")
            if current.get("phase") == "running" and live(current) and current.get("running"):
                status("handed_off", supervisor=supervisor, controller_pid=current["pid"],
                       running=len(current["running"]), pending=len(current.get("pending", [])))
                return
            if current.get("phase") in ("incomplete", "verification_failed", "cancelled"):
                raise RuntimeError("Batch controller did not start: " + current["phase"])
            if (queue / "STOP").exists() or (Path(plan["batch_root"]) / "STOP").exists():
                status("cancelled", reason="Stop requested during handoff")
                return
            if not live(supervisor):
                raise RuntimeError("Batch supervisor exited during handoff")
            time.sleep(5)
        raise RuntimeError("Batch controller did not confirm startup within two minutes")
    except Exception as exc:
        status("blocked", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
