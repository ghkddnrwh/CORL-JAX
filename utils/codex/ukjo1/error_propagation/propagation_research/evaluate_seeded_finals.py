"""CPU-only secondary evaluation of the fixed final-model cohorts.

All original scores remain primary. Wait for audited final checkpoints, verify
equal legacy/corrected Puzzle4 actors with full50 rollouts, then evaluate the36
planned candidate/original models with the cached-space seed protocol. No training.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.propagation_research.seeded_reset import PROTOCOL


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def valid_result(path, job, checkpoint_hash, sources):
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    successes, rewards = d.get("episode_successes", []), d.get("episode_rewards", [])
    return (d.get("evaluation_protocol") == PROTOCOL and d.get("step") == 1000000
            and d.get("episodes") == 50 and d.get("device") == "cpu"
            and d.get("seed") == job["seed"] and d.get("checkpoint_sha256") == checkpoint_hash
            and d.get("evaluator_source_sha256") == sources["evaluate_checkpoint.py"]
            and d.get("seeded_action_space", {}).get("helper_source_sha256") == sources["seeded_reset.py"]
            and len(successes) == len(rewards) == 50 and all(x in (0., 1.) for x in successes)
            and all(math.isfinite(x) for x in rewards)
            and abs(sum(successes)/50 - d["success_rate"]) < 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "analysis/seeded_reset_cpu50_v1"
    out.mkdir(exist_ok=True)
    status = out / "status.json"
    if status.exists():
        pid = json.loads(status.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Secondary evaluator is already live: {pid}")
    plan = root / "seeded_evaluation_plan.json"
    assert json.loads(plan.read_text())["required_distinct_final_models"] == 36
    audit_path = root / "analysis/selected_benchmark_audit.json"
    audit = json.loads(audit_path.read_text())
    rows = [r for r in audit["rows"] if r["family"] in ("candidate", "original_baseline")]
    assert len(rows) == 36 and all(r["run"] for r in rows)
    jobs = [{"id": r["run"], "run": r["run"], "family": r["family"],
             "environment": r["environment"], "seed": r["seed"]} for r in rows]
    known = next(j["id"] for j in jobs if j["family"] == "candidate" and j["seed"] == 0
                 and j["environment"] == "puzzle-4x4-play-singletask-v0")
    legacy = {"id": "validation_legacy_puzzle4_seed0", "family": "validation_only",
              "run": "full9_puzzle4_bounded_v2_s2_seed0", "seed": 0,
              "environment": "puzzle-4x4-play-singletask-v0"}
    jobs.append(legacy)
    sources = {name: digest(Path(__file__).with_name(name)) for name in
               ("evaluate_checkpoint.py", "seeded_reset.py", "evaluate_seeded_finals.py")}
    manifest = {"protocol": PROTOCOL, "registered_plan_sha256": digest(plan),
                "source_sha256": sources, "jobs": jobs, "validation_pair": [known, legacy["id"]]}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Secondary evaluation inputs changed; preserve existing results")
    else:
        atomic(manifest_path, manifest)
        (out / "sources").mkdir()
        for name in sources:
            (out / "sources" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    running, completed = {}, {}
    # These short-lived CPU slots follow the earlier pilot evaluator; no GPU context.
    slots = ("4,5", "10,11", "16,17", "22,23")
    while True:
        for name, checksum in sources.items():
            if digest(Path(__file__).with_name(name)) != checksum:
                raise ValueError("Pinned frozen-evaluator source changed during the run")
        current = {r["run"]: r for r in json.loads(audit_path.read_text())["rows"] if r["run"]}
        ready = {}
        for job in jobs:
            row = current.get(job["run"])
            if job["family"] == "validation_only":
                checksum = digest(root / "runs" / job["run"] / "checkpoint_1000000.pkl")
            elif row and row["status"] == "verified_final":
                checksum = row["model_and_episode_audit"]["checkpoint_sha256"]
                if (row["family"], row["environment"], row["seed"]) != (
                        job["family"], job["environment"], job["seed"]):
                    raise ValueError("Audited scientific identity changed")
            else:
                continue
            ready[job["id"]] = checksum
            output = out / (job["id"] + ".json")
            if job["id"] not in running and output.exists():
                if not valid_result(output, job, checksum, sources):
                    raise ValueError(f"Invalid existing secondary result: {job['id']}")
                completed[job["id"]] = json.loads(output.read_text())
        for name, item in list(running.items()):
            code = item["process"].poll()
            if code is not None:
                item["log"].close()
                if code or not valid_result(out/(name+".json"), item["job"], ready[name], sources):
                    raise RuntimeError(f"Secondary evaluation failed: {name}, exit={code}")
                completed[name] = json.loads((out/(name+".json")).read_text())
                del running[name]
                print("FINISH", name, completed[name]["success_rate"], flush=True)
        pair = manifest["validation_pair"]
        validated = all(name in completed for name in pair)
        if validated:
            a, b = (completed[name] for name in pair)
            equal = (a["actor_parameter_sha256"] == b["actor_parameter_sha256"]
                     and a["episode_successes"] == b["episode_successes"]
                     and a["episode_rewards"] == b["episode_rewards"])
            atomic(out/"equal_actor_validation.json", {"runs": pair, "passed": equal,
                "all50_success_and_reward_arrays_equal": equal,
                "actor_parameter_sha256": a["actor_parameter_sha256"],
                "not_an_additional_training_seed": True})
            if not equal:
                raise ValueError("Equal actors did not reproduce complete50 episode arrays; inspect without retrying")
        eligible = jobs if validated else [j for j in jobs if j["id"] in pair]
        for slot in slots:
            if slot in {v["slot"] for v in running.values()}:
                continue
            job = next((j for j in eligible if j["id"] in ready and j["id"] not in completed
                        and j["id"] not in running), None)
            if job is None:
                break
            name = job["id"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", MUJOCO_GL="egl",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = ["taskset", "-c", slot, sys.executable, "-u", "-m",
                "utils.propagation_research.evaluate_checkpoint", str(root/"runs"/job["run"]),
                str(out/(name+".json")), "--checkpoint", "checkpoint_1000000.pkl",
                "--device", "cpu", "--episodes", "50", "--seeded-action-space",
                "--expected-checkpoint-sha256", ready[name]]
            log = (out/(name+".log")).open("a")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            running[name] = {"process": process, "log": log, "slot": slot, "job": job}
            print("START", name, process.pid, flush=True)
        table = [{**j, "status": "verified_secondary" if j["id"] in completed else
                  "running" if j["id"] in running else "waiting_for_audited_model",
                  "secondary_success": completed.get(j["id"], {}).get("success_rate"),
                  "primary_success": current.get(j["run"], {}).get("final_success")}
                 for j in jobs if j["family"] != "validation_only"]
        means = {}
        for family in ("candidate", "original_baseline"):
            seeds = sorted({j["seed"] for j in table if j["family"] == family})
            per_seed = {str(seed): sum(j["secondary_success"] for j in table
                        if j["family"] == family and j["seed"] == seed)/9
                        for seed in seeds if sum(j["status"] == "verified_secondary" for j in table
                        if j["family"] == family and j["seed"] == seed) == 9}
            means[family] = {"per_seed_complete9_mean": per_seed,
                "all_planned_seed_mean": sum(per_seed.values())/len(seeds)
                    if len(per_seed) == len(seeds) else None}
        summary = {"time": time.time(), "protocol": PROTOCOL, "rows": table,
                   "macros": means, "equal_actor_validation_passed": validated,
                   "primary_scores_unchanged": True, "secondary_scores_never_replace_primary": True}
        atomic(out/"summary.json", summary)
        atomic(status, {"pid": os.getpid(), "time": time.time(), "complete": len(completed),
            "required_including_validation": 37, "equal_actor_validation_passed": validated,
            "running": {k: v["process"].pid for k, v in running.items()}})
        if len(completed) == len(jobs):
            atomic(out/"completed.json", summary)
            print("All36 secondary final-model evaluations and equal-actor control verified", flush=True)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
