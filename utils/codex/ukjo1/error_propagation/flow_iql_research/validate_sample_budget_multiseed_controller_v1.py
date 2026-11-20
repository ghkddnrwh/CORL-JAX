"""Resource-free tests for multi-seed orchestration and paired aggregation."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from utils.codex.ukjo1.error_propagation.flow_iql_research import run_sample_budget_multiseed_v1 as controller
from utils.codex.ukjo1.error_propagation.flow_iql_research import analyze_sample_budget_multiseed_v1 as analyzer
import csv


def check_loop(retry, seed):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        output = root / "runs" / "synthetic"
        (root / "analysis").mkdir()
        (root / "analysis/numerical_validation_v1.json").write_text('{"status":"PASS"}')
        job = dict(id="synthetic", output=str(output), environment="synthetic",
                   recipe="n8", seed=seed, train_module="synthetic.train",
                   eval_module="synthetic.evaluate")
        plan = dict(training_seeds=[0, 1, 2], new_training_seeds=[1, 2], source_sha256={}, jobs=[job], reused_jobs=[],
                    environments=["synthetic"], sample_counts=[8],
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
                    assert command[command.index("--seed") + 1] == str(seed)
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
        return dict(retry=retry, training_seed=seed, learner_attempts=train_attempts,
                    evaluation_finishes=6, controller_phase=status["phase"])



def check_analysis():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "analysis").mkdir()
        envs = ["antmaze-large-navigate-singletask-v0",
                "humanoidmaze-medium-navigate-singletask-v0",
                "puzzle-4x4-play-singletask-v0",
                "antsoccer-arena-navigate-singletask-v0"]
        plan = dict(training_seeds=[0, 1, 2], environments=envs, sample_counts=[8, 9],
                    deployment_modes=["lean", "actor"], evaluation_steps=[800000, 900000, 1000000],
                    jobs=[], reused_jobs=[], source_sha256={}, preserved_reference_sha256={},
                    reused_evidence_sha256={})
        paths = []
        for eidx, env in enumerate(envs):
            for n in plan["sample_counts"]:
                for seed in plan["training_seeds"]:
                    token = f"fixture_{eidx}_n{n}_s{seed}"
                    run = root / "runs" / token
                    run.mkdir(parents=True)
                    directory = root / "evaluations" / token
                    directory.mkdir(parents=True)
                    job = dict(id=token, output=str(run), environment=env, seed=seed, recipe=f"n{n}",
                               bootstrap_samples=n, head_aggregation="min" if env.startswith("antmaze-") else "mean",
                               reused=seed == 0, evaluation_directory=str(directory))
                    plan["reused_jobs" if seed == 0 else "jobs"].append(job)
                    cfg = dict(discount=.999, expectile=.9, target_clipping=True,
                               backup_samples=n, num_samples=9, backup_Q_aggregation=job["head_aggregation"])
                    config = dict(identity=dict(environment=env, seed=seed, recipe=job["recipe"], config=cfg),
                                  source_sha256={})
                    config_path = run / "config.json"
                    config_path.write_text(json.dumps(config))
                    (run / "completed.json").write_text(json.dumps(dict(step=1000000, q_updates=1000000,
                                                                       actor_updates=1000000, v_updates=0)))
                    (run / "diagnostics.jsonl").write_text("".join(json.dumps(dict(
                        step=step, q_updates=step, actor_updates=step, v_updates=0, finite=True,
                        diagnostic_actual_proposal_count=n)) + "\n" for step in range(1000, 1000001, 1000)))
                    successes = [1.0] * (10 + 5 * seed + eidx + (5 if n == 9 else 0))
                    successes += [0.0] * (50 - len(successes))
                    for step in plan["evaluation_steps"]:
                        checkpoint, actor = run / f"checkpoint_{step}.pkl", run / f"actor_{step}.pkl"
                        checkpoint.write_bytes(b"synthetic checkpoint")
                        actor.write_bytes(b"synthetic actor")
                        for mode in plan["deployment_modes"]:
                            result = dict(training_seed=seed, environment=env, recipe=job["recipe"],
                                run=run.name, step=step, policy_mode=mode, n_episodes=50,
                                first_episode_seed=42, episode_successes=successes,
                                success_rate=sum(successes)/50, no_optimizer_updates=True,
                                protocol="last3_cpu50_independent_seeded_episodes_v1",
                                config_sha256=analyzer.sha(config_path), checkpoint_sha256=analyzer.sha(checkpoint),
                                actor_artifact_sha256=analyzer.sha(actor))
                            path = directory / f"{step}_{mode}.json"
                            path.write_text(json.dumps(result))
                            paths.append(path)
        plan["expected_new_evaluations"] = len(plan["jobs"]) * 6
        plan["expected_reused_evaluations"] = len(plan["reused_jobs"]) * 6
        (root / "registration_v1.json").write_text(json.dumps(plan))
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                analyzer.sys if hasattr(analyzer, "sys") else __import__("sys"),
                "argv", ["test", "--root", str(root), "--final"]):
            analyzer.main()
        summary = json.loads((root / "analysis/summary_final_v1.json").read_text())
        assert summary["complete"] and summary["late_evaluation_episode_records"] == 7200
        macros = list(csv.DictReader((root / "analysis/macro_results_mean_sd_final_v1.csv").open()))
        row = next(r for r in macros if r["bootstrap_samples"] == "8" and r["deployment"] == "lean")
        assert abs(float(row["three_seed_mean"]) - .33) < 1e-12
        assert abs(float(row["three_seed_sample_sd"]) - .1) < 1e-12
        assert abs(float(row["paired_change_from_N9_mean"]) + .1) < 1e-12
        assert abs(float(row["paired_change_from_N9_sample_sd"])) < 1e-12
        missing = root / "evaluations/fixture_0_n8_s2/1000000_actor.json"
        saved = missing.read_text()
        missing.unlink()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                __import__("sys"), "argv", ["test", "--root", str(root)]):
            analyzer.main()
        summary = json.loads((root / "analysis/summary_progress_v1.json").read_text())
        assert not summary["complete"]
        row = next(r for r in summary["macros"] if r["bootstrap_samples"] == 8 and r["deployment"] == "actor")
        assert row["three_seed_mean"] is None and row["available_complete_seeds"] == 2
        missing.write_text(saved)
        (root / "runs/fixture_0_n8_s0/checkpoint_800000.pkl").write_bytes(b"corruption")
        caught = False
        try:
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                    __import__("sys"), "argv", ["test", "--root", str(root), "--final"]):
                analyzer.main()
        except AssertionError:
            caught = True
        assert caught
        return dict(status="PASS", synthetic_training_cells=24, synthetic_evaluation_points=144,
                    paired_mean_and_sample_SD=True, missing_evaluation_excluded=True,
                    model_hash_corruption_detected=True)


def main():
    stream = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        controller.record_event(stream, "EVAL_FINISH", time=123.0, result="complete")
    value = json.loads(stream.getvalue())
    assert value["time"] == 123.0 and value["kind"] == "EVAL_FINISH"
    cases = [check_loop(False, 1), check_loop(True, 2)]
    analysis_case = check_analysis()
    print(json.dumps(dict(status="PASS", timestamp_preserved=True, cases=cases, analysis=analysis_case,
                          scope="Synthetic controller/aggregation regression; no learners or evaluators executed."), indent=2))


if __name__ == "__main__":
    main()
