"""Persistent two-GPU work queue. Re-read queue.json whenever a slot opens."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--per-gpu", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    running = {}
    finished_path = root / "finished.json"
    finished = json.loads(finished_path.read_text()) if finished_path.exists() else {}
    cpu_ids = sorted(os.sched_getaffinity(0))
    slots = [(gpu, slot) for slot in range(args.per_gpu) for gpu in range(2)]
    while not (root / "STOP").exists():
        for key, record in list(running.items()):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            finished[key] = {"exit_code": code, "finished_at": time.time(),
                             "output": record["output"]}
            atomic_json(finished_path, finished)
            del running[key]
            print(f"FINISH {key}: exit={code}", flush=True)
        queue = json.loads((root / "queue.json").read_text())
        used = {v["slot"] for v in running.values()}
        for slot in slots:
            if slot in used:
                continue
            job = next((j for j in queue if j["id"] not in running
                        and j["id"] not in finished), None)
            if job is None:
                break
            output = root / "runs" / job["id"]
            output.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]),
                       XLA_PYTHON_CLIENT_MEM_FRACTION="0.1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="egl",
                       PYTHONUNBUFFERED="1", TF_NUM_INTRAOP_THREADS="1",
                       TF_NUM_INTEROP_THREADS="1")
            idx = slots.index(slot)
            per_process = max(1, len(cpu_ids)//len(slots))
            cpus = cpu_ids[idx*per_process:(idx+1)*per_process]
            cmd = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-u", "-m",
                   "algorithms.uk_offline.basic.iql_residual_control", "--output", str(output)]
            for k, value in job["args"].items():
                cmd.extend(["--"+k, str(value)])
            log = (output / "stdout.log").open("a")
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            running[job["id"]] = {"process": p, "log": log, "slot": slot,
                                   "output": str(output), "started_at": time.time()}
            print(f"START {job['id']}: GPU={slot[0]} PID={p.pid}", flush=True)
        atomic_json(root / "scheduler_status.json", {
            "pid": os.getpid(), "updated_at": time.time(),
            "running": {k: {"pid": v["process"].pid, "gpu": v["slot"][0],
                             "output": v["output"], "started_at": v["started_at"]}
                        for k, v in running.items()},
            "finished": len(finished),
            "pending": sum(j["id"] not in running and j["id"] not in finished for j in queue)})
        time.sleep(5)
    # STOP means drain without launching further work. No foreign processes are touched.
    for record in running.values():
        record["process"].wait()


if __name__ == "__main__":
    main()
