"""Exercise finish/retry controller paths with synthetic, resource-free workers."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from utils.flow_iql_research import run_sample_budget_v2 as controller


def check_loop(retry):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        output = root / "runs" / "synthetic"
        (root / "analysis").mkdir()
        (root / "analysis/numerical_validation_v1.json").write_text('{"status":"PASS"}')
        job = dict(id="synthetic", output=str(output), environment="synthetic",
                   recipe="n8", seed=0, train_module="synthetic.train",
                   eval_module="synthetic.evaluate")
        plan = dict(training_seed=0, source_sha256={}, jobs=[job], reused_jobs=[],
                    evaluation_steps=[800000, 900000, 1000000],
                    deployment_modes=["lean", "actor"], total_cells=1)
        (root / "registration_v1.json").write_text(json.dumps(plan))
        launches = []
        train_attempts = 0
        clock = 0

        def tick():
            nonlocal clock
            clock += 31
            return float(clock)

        class SyntheticWorker:
            def __init__(self, command, **kwargs):
                nonlocal train_attempts
                launches.append(command)
                self.pid = 100000 + len(launches)
                self.returncode = 0
                if "--env" in command:
                    train_attempts += 1
                    if retry and train_attempts == 1:
                        self.returncode = 1
                        return
                    (output / "completed.json").write_text("{}")
                    for step in plan["evaluation_steps"]:
                        for prefix in ("checkpoint", "actor"):
                            (output / f"{prefix}_{step}.pkl").write_bytes(b"synthetic")
                else:
                    result = Path(command[command.index("--output") + 1])
                    result.write_text("{}")

            def poll(self):
                return self.returncode

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(controller.sys, "argv", ["test", "--root", str(root)]), \
                patch.object(controller, "birth", return_value="synthetic-birth"), \
                patch.object(controller.os, "sched_getaffinity", return_value=set(range(96))), \
                patch.object(controller.time, "time", side_effect=tick), \
                patch.object(controller.time, "sleep"), \
                patch.object(controller.subprocess, "Popen", SyntheticWorker), \
                patch.object(controller.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            controller.main()
        status = json.loads((root / "controller_status.json").read_text())
        events = [json.loads(line) for line in (root / "controller_events.jsonl").read_text().splitlines()]
        assert status["phase"] == "complete" and status["analysis_exit_code"] == 0
        assert status["training_complete"] == 1 and status["evaluations_complete"] == 6
        assert train_attempts == (2 if retry else 1)
        assert sum(e["kind"] == "TRAIN_FINISH" for e in events) == 1
        assert sum(e["kind"] == "EVAL_FINISH" for e in events) == 6
        assert sum(e["kind"] == "TRAIN_RETRY_QUEUED" for e in events) == int(retry)
        assert all("event_time" in e and "time" in e for e in events)
        assert all(len(c[c.index("-c") + 1].split(",")) == 4 for c in launches)
        return dict(retry=retry, learner_attempts=train_attempts,
                    evaluation_finishes=6, controller_phase=status["phase"])


def main():
    stream = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        controller.record_event(stream, "EVAL_FINISH", time=123.0, result="complete")
    value = json.loads(stream.getvalue())
    assert value["time"] == 123.0 and value["kind"] == "EVAL_FINISH"
    cases = [check_loop(False), check_loop(True)]
    print(json.dumps(dict(status="PASS", timestamp_preserved=True, cases=cases,
                          scope="Controller only; no learners or evaluators executed."), indent=2))


if __name__ == "__main__":
    main()
