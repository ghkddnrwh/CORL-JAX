"""Check seed-1 updates/resume, paired aggregation and adoption of 16 existing workers."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def numerical(root):
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import numpy as np
    from utils.codex.ukjo1.error_propagation.flow_iql_research import nondefault_n9_core_v1 as core
    from utils.codex.ukjo1.error_propagation.flow_iql_research import train_nondefault_n9_v1 as old
    from utils.codex.ukjo1.error_propagation.flow_iql_research import train_nondefault_n9_v2 as new

    assert all(d.platform == "cpu" for d in jax.devices())
    before, after = ast.parse(Path(old.__file__).read_text()), ast.parse(Path(new.__file__).read_text())
    before.body[0] = copy.deepcopy(after.body[0])  # Only the module description and seed guard may differ.
    old_main = next(v for v in before.body if isinstance(v, ast.FunctionDef) and v.name == "main")
    new_main = next(v for v in after.body if isinstance(v, ast.FunctionDef) and v.name == "main")
    guard = next(v for v in new_main.body if isinstance(v, ast.Assert))
    assert "args.seed in (0, 1)" in ast.unparse(guard)
    new_main.body[new_main.body.index(guard)] = copy.deepcopy(next(v for v in old_main.body if isinstance(v, ast.Assert)))
    assert ast.dump(before) == ast.dump(after)

    def equal(a, b):
        left, right = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
        return len(left) == len(right) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(left, right))

    rng = np.random.RandomState(7364)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0, 1, 1, 0, 1, 1, 1, 1], jnp.float32))
    checks = []
    for code in ("aml", "puzzle4"):
        env = next(t["environment"] for t in core.TASKS if t["code"] == code)
        assert new.configuration(env, "g995_n9", 1) == old.configuration(env, "g995_n9", 1)
        cfg = core.config_for(env, "g995_n9", width=16)
        bounds = core.return_bounds(env, .995, -1., 0.)
        initial = core.create(1, batch["observations"][:1], batch["actions"][:1], cfg, "g995_n9")
        repeated = core.create(1, batch["observations"][:1], batch["actions"][:1], cfg, "g995_n9")
        seed0 = core.create(0, batch["observations"][:1], batch["actions"][:1], cfg, "g995_n9")
        assert equal(initial, repeated) and not equal(initial, seed0)
        update = jax.jit(core.build_update("g995_n9", bounds))
        state = initial
        for _ in range(3):
            state, _ = update(state, batch)
        restored = flax.serialization.from_state_dict(initial, flax.serialization.to_state_dict(state))
        continued, _ = update(state, batch)
        resumed, _ = update(restored, batch)
        assert equal(continued, resumed)
        assert int(continued.base.q_updates) == int(continued.actor_updates) == 4
        assert int(continued.base.v_updates) == 0
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(continued))
        checks.append(dict(environment=env, seed1_deterministic=True, differs_from_seed0=True,
                           resume_bitwise_equal=True, finite=True))
    value = dict(status="PASS", trainer_AST_equal_except_seed_guard_and_docstring=True,
                 real_CPU_updates=checks, actual_GPU_work=False)
    atomic(root / "validation/numerical_v2.json", value)
    return value


def aggregation(root, plan):
    from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_nondefault_n9_v2 import aggregate
    import math
    rows = []
    for job in plan["jobs"]:
        tasks = [j["environment"] for j in plan["jobs"] if j["seed"] == 0 and j["family"] == job["family"]]
        score = 10. + 20 * tasks.index(job["environment"]) + 10 * job["seed"]
        for mode in plan["deployment_modes"]:
            rows.append(dict(environment=job["environment"], family=job["family"], seed=job["seed"],
                             deployment=mode, last3_success_pct=score))
    tasks, per_seed, families = aggregate(rows, plan)
    assert len(tasks) == 80 and len(per_seed) == 40 and len(families) == 20
    assert abs(tasks[0]["last3_mean_pct"] - 15) < 1e-12
    assert abs(tasks[0]["sample_sd_pp"] - math.sqrt(50)) < 1e-12
    assert families[0]["last3_mean_pct"] == 45.
    assert abs(families[0]["sample_sd_pp"] - math.sqrt(50)) < 1e-12
    missing = next(r for r in rows if r["seed"] == 1 and r["deployment"] == plan["deployment_modes"][0])
    missing["last3_success_pct"] = None
    partial_tasks, _, partial_family = aggregate(rows, plan)
    assert partial_tasks[0]["last3_mean_pct"] is None and partial_family[0]["last3_mean_pct"] is None
    value = dict(status="PASS", seed_level_family_SD=True, missing_seed_never_averaged_as_complete=True)
    atomic(root / "validation/aggregation_v2.json", value)
    return value


def controller(root, plan):
    from utils.codex.ukjo1.error_propagation.flow_iql_research import run_nondefault_n9_v2 as module
    with tempfile.TemporaryDirectory(prefix="nondefault_seed01_controller_") as temporary:
        folder = Path(temporary)
        (folder / "validation").mkdir()
        for name in ("numerical_v2", "controller_v2"):
            atomic(folder / f"validation/{name}.json", dict(status="PASS"))
        plan = copy.deepcopy(plan)
        plan["source_sha256"] = {}
        for job in plan["jobs"]:
            job["output"] = str(folder / "runs" / job["id"])
            job["evaluation_directory"] = str(folder / "evaluations" / job["id"])
        atomic(folder / "registration_v2.json", plan)
        clock, workers, attempts = [1e6], {}, {}
        counts, peaks = dict(train=0, eval=0), dict(train=0, eval=0)
        finished, evaluated, adopted = {}, {}, {}

        def artifacts(output):
            output.mkdir(parents=True, exist_ok=True)
            (output / "completed.json").write_text("{}")
            for step in plan["evaluation_steps"]:
                for stem in ("checkpoint", "actor"):
                    (output / f"{stem}_{step}.pkl").write_bytes(b"stub")

        class Worker:
            def __init__(self, command, existing=False, **kwargs):
                self.command, self.pid, self.returncode = command, 10000 + len(workers), None
                self.existing = existing
                self.training = "--env" in command
                self.role = "train" if self.training else "eval"
                self.output = Path(command[command.index("--output") + 1])
                token = str(self.output)
                attempts[token] = attempts.get(token, 0) + 1
                self.fail = not existing and attempts[token] == 1 and (
                    self.training and self.output.name == plan["jobs"][40]["id"] or
                    not self.training and self.output.name == "800000_lean.json")
                self.ready = clock[0] + 200 if existing else clock[0]
                workers[self.pid] = self
                counts[self.role] += 1
                peaks[self.role] = max(peaks[self.role], counts[self.role])

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                if clock[0] < self.ready:
                    return None
                self.returncode = 1 if self.fail else 0
                counts[self.role] -= 1
                if not self.fail:
                    if self.training:
                        artifacts(self.output)
                    else:
                        self.output.parent.mkdir(parents=True, exist_ok=True)
                        self.output.write_text("{}")
                return self.returncode

        for job in plan["jobs"][:16]:
            artifacts(Path(job["output"]))
            finished[job["id"]] = dict(result="complete", exit_code=0)
            for step in plan["evaluation_steps"]:
                for mode in plan["deployment_modes"]:
                    path = folder / "evaluations" / job["id"] / f"{step}_{mode}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}")
                    evaluated[f"{job['id']}:{step}:{mode}"] = dict(result="complete", exit_code=0)
        for index, job in enumerate(plan["jobs"][16:32]):
            command = [sys.executable, "-m", job["train_module"], "--env", job["environment"], "--output", job["output"]]
            worker = Worker(command, existing=True)
            slot = plan["resources"]["train_slots"][index]
            adopted[job["id"]] = dict(pid=worker.pid, birth=str(worker.pid), gpu=slot[0], slot=slot, started=clock[0])
        atomic(folder / "controller_status.json", dict(running=adopted, evaluating={}, finished=finished,
                                                       evaluation_finished=evaluated, attempts={}, evaluation_attempts={}))

        def now():
            clock[0] += 1
            return clock[0]

        def birth(pid):
            worker = workers.get(pid)
            if not worker:
                return None
            if worker.existing:
                worker.poll()
            return str(pid) if worker.returncode is None else None

        original_read = Path.read_bytes

        def command_bytes(path):
            if str(path).startswith("/proc/") and path.name == "cmdline":
                worker = workers[int(path.parts[2])]
                return b"\0".join(word.encode() for word in worker.command) + b"\0"
            return original_read(path)

        fake_process = SimpleNamespace(Popen=Worker, STDOUT=-2, run=lambda *a, **kw: SimpleNamespace(returncode=0))
        fake_time = SimpleNamespace(time=now, sleep=lambda n: clock.__setitem__(0, clock[0] + n))
        registered_cpus = {cpu for group in plan["resources"]["train_cpus"] + plan["resources"]["eval_cpus"] for cpu in group}
        with patch.object(module, "subprocess", fake_process), patch.object(module, "time", fake_time), \
             patch.object(module, "birth", birth), patch.object(Path, "read_bytes", command_bytes), \
             patch.object(module.os, "sched_getaffinity", return_value=registered_cpus), \
             patch.object(sys, "argv", ["controller", "--root", str(folder)]):
            with (root / "validation/controller_simulation_v2.log").open("w") as stream, redirect_stdout(stream):
                module.main()
        status = json.loads((folder / "controller_status.json").read_text())
        assert status["phase"] == "complete" and status["training_complete"] == 80 and status["evaluations_complete"] == 480
        assert peaks == dict(train=16, eval=8), peaks
        assert {r["kind"] for r in status["retry_history"]} == {"training", "evaluation"}
        events = [json.loads(line) for line in (folder / "controller_events.jsonl").read_text().splitlines()]
        assert sum(v["kind"] == "ADOPT" for v in events) == 16
        starts = [v["identity"] for v in events if v["kind"] == "TRAIN_START"]
        assert not any(j["id"] in starts for j in plan["jobs"][:32])
        assert starts[0] == plan["jobs"][32]["id"]
        value = dict(status="PASS", training_completed=80, evaluations_completed=480,
                     adopted_existing_training=16, completed_training_not_restarted=16,
                     original_workers_not_restarted=True, seed0_pending_precedes_seed1=True,
                     maximum_concurrency=peaks, automatic_refill=True, training_and_evaluation_retry=True,
                     actual_GPU_work=False)
        atomic(root / "validation/controller_v2.json", value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "registration_v2.json").read_text())
    assert plan["training_seeds"] == [0, 1] and len(plan["jobs"]) == 80
    results = dict(numerical=numerical(root), aggregation=aggregation(root, plan), controller=controller(root, plan))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
