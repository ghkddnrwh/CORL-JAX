"""Match six existing pilot methods at 300k and CPU50 without more training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def valid_result(path, digest):
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    return (d["checkpoint_sha256"] == digest and d["step"] == 300000 and
            d["episodes"] == 50 and d["device"] == "cpu" and d["seed"] == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "analysis/pilot300k_cpu50"
    out.mkdir(exist_ok=True)
    status = out / "status.json"
    if status.exists():
        pid = json.loads(status.read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError(f"Pilot evaluator is already live: {pid}")
    templates = {"baseline": "w6_{env}_baseline_s2_seed0",
        "bounded": "w6_{env}_bounded_s2_seed0",
        "v2_fresh_backup": "ablation_{env}_v2_fresh_s2_seed0",
        "bounded_v2": "w7_{env}_bounded_v2_s2_seed0",
        "bounded_v2_independent": "refine_{env}_independent_v_s2_seed0",
        "negative_v2": "constraint_{env}_negative_v2_s2_seed0"}
    jobs = []
    for env in ("cube", "scene"):
        for profile, template in templates.items():
            run = root / "runs" / template.format(env=env)
            cfg = json.loads((run / "config.json").read_text())
            assert cfg["arguments"]["profile"] == profile
            assert cfg["identity"]["config"]["discount"] == .999
            assert cfg["identity"]["config"]["iql_tau"] == .9
            checkpoint = run / "checkpoint_300000.pkl"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = out / f"{env}_{profile}.json"
            # Reuse only previously verified evaluations of the same bytes.
            previous = root / "analysis" / ({"baseline": "reference_scene_s2_300k_cpu.json",
                "bounded_v2": "bounded_v2_scene_s2_300k_cpu.json"}.get(profile, "unused"))
            if env == "scene" and not output.exists() and valid_result(previous, digest):
                output.write_bytes(previous.read_bytes())
            jobs.append({"id": f"{env}_{profile}", "profile": profile, "run": str(run),
                         "environment": cfg["identity"]["config"]["env"],
                         "checkpoint_sha256": digest, "output": str(output)})
    evaluator = Path(__file__).with_name("evaluate_checkpoint.py")
    atomic_json(out / "manifest.json", {"scope": "Pilot comparison only; excluded from final1M benchmark means",
        "Q_updates": 300000, "episodes": 50, "device": "cpu", "training_seed": 0,
        "evaluator_source_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(), "jobs": jobs})
    (out / "evaluator_source.py").write_bytes(evaluator.read_bytes())
    running, done = {}, {}
    slots = ("4,5", "10,11", "16,17", "22,23")
    for job in jobs:
        if valid_result(Path(job["output"]), job["checkpoint_sha256"]):
            done[job["id"]] = 0
    while len(done) < len(jobs):
        for key, item in list(running.items()):
            code = item["process"].poll()
            if code is not None:
                item["log"].close()
                done[key] = code
                del running[key]
                print("FINISH", key, code, flush=True)
        for slot in slots:
            if slot in {item["slot"] for item in running.values()}:
                continue
            job = next((j for j in jobs if j["id"] not in done and j["id"] not in running), None)
            if job is None:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl")
            cmd = ["taskset", "-c", slot, sys.executable, "-u", "-m",
                "utils.propagation_research.evaluate_checkpoint", job["run"], job["output"],
                "--checkpoint", "checkpoint_300000.pkl", "--device", "cpu", "--episodes", "50"]
            log = (out / (job["id"]+".log")).open("a")
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            running[job["id"]] = {"process": process, "log": log, "slot": slot}
            print("START", job["id"], process.pid, flush=True)
        atomic_json(status, {"pid": os.getpid(), "updated_at": time.time(), "finished": done,
            "running": {key: item["process"].pid for key, item in running.items()}})
        if running:
            time.sleep(2)
    rows = []
    for job in jobs:
        if done[job["id"]] or not valid_result(Path(job["output"]), job["checkpoint_sha256"]):
            raise ValueError(f"Pilot evaluation did not complete correctly: {job['id']}")
        d = json.loads(Path(job["output"]).read_text())
        rows.append({"environment": job["environment"], "profile": job["profile"],
            "step": d["step"], "seed": 0, "episodes": 50, "evaluation_device": "cpu",
            "success_rate": d["success_rate"], "checkpoint_sha256": job["checkpoint_sha256"]})
    with (out / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / "completed.json", {"evaluated_pilots": len(rows), "rows": rows,
                                        "not_a_final_9_environment_mean": True})
    print("Completed", len(rows), "matched pilot evaluations", flush=True)


if __name__ == "__main__":
    main()
