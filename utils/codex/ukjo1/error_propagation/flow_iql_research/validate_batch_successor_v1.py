"""Exercise dependency gates, cancellation preservation and 16-slot refill without GPUs."""
import argparse
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from utils.codex.ukjo1.error_propagation.flow_iql_research import resume_batch_after_ema_v1 as successor
from utils.codex.ukjo1.error_propagation.flow_iql_research import run_stability_study_v5 as controller


def validate(real, queue):
    checked = []
    with tempfile.TemporaryDirectory(prefix="batch_successor_") as temporary:
        base = Path(temporary)
        dependency, batch, pending = (base / name for name in ("ema", "batch", "queue"))
        for directory in (dependency / "analysis", batch / "analysis", pending):
            directory.mkdir(parents=True)
        (batch / "STOP").write_text("Original cancellation\n")
        plan = dict(dependency_root=str(dependency), batch_root=str(batch),
                    dependency_expected=dict(training_complete=14, expected_training=14,
                        evaluations_complete=84, expected_evaluations=84, total_cells=24, primary_evaluations=144),
                    original_batch_stop=successor.fingerprint(batch / "STOP"),
                    python=sys.executable, repository=str(Path.cwd()))
        status = dict(phase="running")

        def gate():
            successor.atomic(dependency / "controller_status.json", status)
            return successor.dependency_gate(plan)[0]

        assert gate() == "waiting_for_ema"
        checked.append("active_EMA_never_launches_batch")
        status.update(phase="complete", **{k:v for k,v in plan["dependency_expected"].items() if k not in ("total_cells", "primary_evaluations")},
                      registration="registration_v2.json", running={}, evaluating={}, pending=[])
        assert gate() == "waiting_for_ema"
        status["analysis_exit_code"] = 1
        assert gate() == "blocked"
        checked.append("missing_or_failed_final_analysis_prevents_launch")
        status["analysis_exit_code"] = 0
        successor.atomic(dependency / "analysis/summary_final_v2.json", dict(complete=True,
                         failures=[], training_complete=24, total_cells=24, primary_evaluations=144))
        assert gate() == "ready"
        status.update(pid=777, birth="test")
        with patch.object(successor, "birth", return_value="test"):
            assert gate() == "waiting_for_ema"
        status.pop("pid"); status.pop("birth")
        checked.append("wait_for_controller_exit_even_after_final_summary")
        status["evaluations_complete"] = 83
        assert gate() == "blocked"
        status["evaluations_complete"] = 84
        (dependency / "STOP").touch()
        assert gate() == "cancelled"
        (dependency / "STOP").unlink()
        checked.append("incomplete_counts_and_STOP_prevent_launch")

        registration = json.loads((real / "registration_v5.json").read_text())
        for job in registration["jobs"]:
            job["output"] = str(batch / "runs" / job["id"])
            job["evaluation_directory"] = str(batch / "evaluations" / job["id"])
        registration["source_sha256"] = {}
        successor.atomic(batch / "registration_v5.json", registration)
        original = Path(registration["jobs"][0]["output"])
        original.mkdir(parents=True)
        (original / "latest.pkl").write_bytes(b"preserved_checkpoint")
        (original / "cancelled.json").write_text('{"reason":"old cancellation"}')
        successor.atomic(batch / "controller_status.json", dict(phase="cancelled", running={}, evaluating={},
            finished={j["id"]:dict(result="cancelled") for j in registration["jobs"]},
            attempts={j["id"]:2 for j in registration["jobs"]}))
        successor.prepare_activation(pending, plan)
        assert (batch / "STOP").exists()
        assert successor.read(batch / "controller_status.json")["finished"] == {}
        assert len(successor.read(batch / "controller_status.json")["pending"]) == 29
        assert (original / "latest.pkl").read_bytes() == b"preserved_checkpoint"
        assert not (original / "cancelled.json").exists()
        assert (pending / "activation_archive" / (original.name + "_cancelled.json")).exists()
        snapshot = (batch / "controller_status.json").read_bytes()
        successor.prepare_activation(pending, plan)
        assert snapshot == (batch / "controller_status.json").read_bytes()
        checked.append("idempotent_requeue_preserves_checkpoints_and_cancellation_history")
        (batch / "STOP").write_text("New cancellation\n")
        with patch.object(successor.subprocess, "Popen") as launch:
            try:
                successor.start_supervisor(pending, plan)
                raise AssertionError("A new STOP must block handoff")
            except RuntimeError:
                pass
            launch.assert_not_called()
        checked.append("new_cancellation_cannot_be_overridden_by_old_authorization")
        (batch / "STOP").unlink()

        # Run the actual unchanged batch controller against fake workers.
        successor.atomic(batch / "analysis/numerical_validation_v1.json", dict(status="PASS"))
        clock = [1e6]
        workers, attempts = {}, {}
        active = dict(train=0, eval=0)
        peaks = dict(active)

        class Worker:
            def __init__(self, command, **kwargs):
                self.pid = 10000 + len(workers)
                self.returncode = None
                self.training = "--env" in command
                self.role = "train" if self.training else "eval"
                self.output = Path(command[command.index("--output") + 1])
                token = str(self.output)
                attempts[token] = attempts.get(token, 0) + 1
                self.fail = attempts[token] == 1 and (
                    self.training and self.output.name == registration["jobs"][0]["id"] or
                    not self.training and self.output.name == "800000_lean.json")
                workers[self.pid] = self
                active[self.role] += 1
                peaks[self.role] = max(peaks[self.role], active[self.role])

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                self.returncode = 1 if self.fail else 0
                active[self.role] -= 1
                if not self.fail:
                    self.output.parent.mkdir(parents=True, exist_ok=True)
                    if self.training:
                        self.output.mkdir(exist_ok=True)
                        (self.output / "completed.json").write_text("{}")
                        for step in registration["evaluation_steps"]:
                            for kind in ("checkpoint", "actor"):
                                (self.output / f"{kind}_{step}.pkl").write_bytes(b"stub")
                    else:
                        self.output.write_text("{}")
                return self.returncode

        def now():
            clock[0] += 1
            return clock[0]

        fake_process = SimpleNamespace(Popen=Worker, STDOUT=-2, run=lambda *a, **kw: SimpleNamespace(returncode=0))
        fake_time = SimpleNamespace(time=now, sleep=lambda n: clock.__setitem__(0, clock[0] + n))
        fake_birth = lambda pid: str(pid) if pid in workers and workers[pid].returncode is None else None
        with patch.object(controller, "subprocess", fake_process), patch.object(controller, "time", fake_time), \
             patch.object(controller, "birth", fake_birth), patch.object(sys, "argv", ["controller", "--root", str(batch)]):
            with (queue / "controller_simulation_stdout.log").open("w") as stream, redirect_stdout(stream):
                controller.main()
        end = successor.read(batch / "controller_status.json")
        assert end["phase"] == "complete" and end["analysis_exit_code"] == 0
        assert end["training_complete"] == 29 and end["evaluations_complete"] == 348
        assert peaks == dict(train=16, eval=8)
        assert {r["kind"] for r in end["retry_history"]} == {"training", "evaluation"}
        checked.append("real_controller_refills_16_train_and_8_eval_slots_and_retries_failures")
    result = dict(status="PASS", checks=checked, actual_GPU_work=False,
                  simulated_training=29, simulated_evaluations=348, peak_slots=peaks)
    successor.atomic(queue / "queue_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve(), args.queue.resolve()), indent=2))


if __name__ == "__main__":
    main()
