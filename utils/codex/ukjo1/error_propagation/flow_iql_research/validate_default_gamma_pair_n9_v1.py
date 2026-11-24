"""Check frozen baseline updates, checkpoint resume, matrix aggregation and scheduler refill."""
import argparse
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import atomic, assert_matrix


def numerical(root, plan):
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import numpy as np
    from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as base
    from utils.codex.ukjo1.error_propagation.flow_iql_research import discount_0995_n9_core_v1 as low
    from utils.codex.ukjo1.error_propagation.flow_iql_research import stability_study_core_v1 as control

    assert all(d.platform == 'cpu' for d in jax.devices())
    rng = np.random.RandomState(7364)
    batch = dict(observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8, 5), jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8, 2), -.9, .9), jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0, 1, 1, 0, 1, 1, 1, 1], jnp.float32))

    def equal(a, b):
        x, y = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
        assert len(x) == len(y)
        assert all(np.array_equal(np.asarray(i), np.asarray(j)) for i, j in zip(x, y))

    checks = []
    for gamma, seed, env in ((.995, 2, plan['environments'][0]),
                             (.999, 3, 'cube-double-play-singletask-v0')):
        core, recipe = (low, 'g995_n9') if gamma == .995 else (control, 'control')
        cfg = core.config_for(env, recipe, width=16)
        expected = base.config_for(env, 'n9', width=16)
        expected['discount'] = gamma
        bounds = core.return_bounds(env, gamma, -1., 0.)
        state = core.create(seed, batch['observations'][:1], batch['actions'][:1], cfg, recipe)
        reference = base.create(seed, batch['observations'][:1], batch['actions'][:1], expected, 'n9')
        update, old_update = jax.jit(core.build_update(recipe, bounds)), jax.jit(base.build_update('n9', bounds))
        equal(state, reference)
        for _ in range(3):
            state, _ = update(state, batch)
            reference, _ = old_update(reference, batch)
            equal(state, reference)
        restored = flax.serialization.from_state_dict(state, flax.serialization.to_state_dict(state))
        advanced, _ = update(state, batch)
        resumed, _ = update(restored, batch)
        equal(advanced, resumed)
        assert int(advanced.base.q_updates) == int(advanced.actor_updates) == 4
        assert int(advanced.base.v_updates) == 0
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(advanced))
        checks.append(dict(gamma=gamma, seed=seed, environment=env, three_updates_bitwise_equal=True,
                           serialization_resume_bitwise_equal=True, finite=True))
    result = dict(status='PASS', frozen_existing_trainers=True, checks=checks)
    atomic(root / 'validation/numerical_v1.json', result)
    return result


def aggregation(root, plan):
    from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_default_gamma_pair_n9_v1 import aggregate
    import statistics
    rows = []
    for j in plan['jobs'] + plan['reused_jobs']:
        for mode in plan['deployment_modes']:
            rows.append(dict(environment=j['environment'], gamma=j['gamma'], seed=j['seed'], deployment=mode,
                             last3_success_pct=10. * j['seed'] + (5. if j['gamma'] == .995 else 0.)))
    groups, paired = aggregate(rows, plan)
    assert len(groups) == 40 and len(paired) == 20
    assert groups[0]['mean_pct'] == 20.
    assert groups[0]['sample_sd_pp'] == statistics.stdev([5., 15., 25., 35.])
    assert all(r['gamma0995_minus0999_pp'] == 5. for r in paired)
    target = next(r for r in rows if r['environment'] == plan['environments'][0] and r['gamma'] == .995 and r['seed'] == 3 and r['deployment'] == 'lean')
    target['last3_success_pct'] = None
    groups, paired = aggregate(rows, plan)
    assert groups[0]['mean_pct'] is None and groups[0]['complete_seeds'] == 3
    assert paired[0]['gamma0995_minus0999_pp'] is None and paired[0]['complete_pairs'] == 3
    result = dict(status='PASS', complete_four_seed_mean_and_sample_SD=True,
                  paired_gamma_differences=True, missing_seeds_never_pooled_as_complete=True)
    atomic(root / 'validation/aggregation_v1.json', result)
    return result

def controller(root, plan):
    from utils.codex.ukjo1.error_propagation.flow_iql_research import run_default_gamma_pair_n9_v1 as module
    with tempfile.TemporaryDirectory(prefix="nondefault_seed01_controller_") as temporary:
        folder = Path(temporary)
        (folder / "validation").mkdir()
        for name in ("numerical_v1", "controller_v1", "aggregation_v1", "reuse_audit_v1"):
            atomic(folder / f"validation/{name}.json", dict(status="PASS"))
        plan = copy.deepcopy(plan)
        plan["source_sha256"] = {}
        for job in plan["jobs"]:
            job["output"] = str(folder / "runs" / job["id"])
            job["evaluation_directory"] = str(folder / "evaluations" / job["id"])
        atomic(folder / "registration_v1.json", plan)
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
                    self.training and self.output.name == plan["jobs"][32]["id"] or
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
            with (root / "validation/controller_simulation_v1.log").open("w") as stream, redirect_stdout(stream):
                module.main()
        status = json.loads((folder / "controller_status.json").read_text())
        assert status["phase"] == "complete" and status["training_complete"] == 36 and status["evaluations_complete"] == 216
        assert peaks == dict(train=16, eval=8), peaks
        assert {r["kind"] for r in status["retry_history"]} == {"training", "evaluation"}
        events = [json.loads(line) for line in (folder / "controller_events.jsonl").read_text().splitlines()]
        assert sum(v["kind"] == "ADOPT" for v in events) == 16
        starts = [v["identity"] for v in events if v["kind"] == "TRAIN_START"]
        assert not any(j["id"] in starts for j in plan["jobs"][:32])
        assert starts[0] == plan["jobs"][32]["id"]
        value = dict(status="PASS", training_completed=36, evaluations_completed=216, reused_completed=44,
                     adopted_existing_training=16, completed_training_not_restarted=16,
                     original_workers_not_restarted=True, registered_queue_order_preserved=True,
                     maximum_concurrency=peaks, automatic_refill=True, training_and_evaluation_retry=True,
                     actual_GPU_work=False)
        atomic(root / "validation/controller_v1.json", value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'registration_v1.json').read_text())
    assert_matrix(plan)
    result = dict(numerical=numerical(root, plan), aggregation=aggregation(root, plan), controller=controller(root, plan))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

