"""Preserve key checkpoints and run independent CPU diagnostics in spare time."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.propagation_research.diagnose_checkpoint import CheckpointReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    running = {}
    finished = set()
    status_path = root / "diagnostic_watch.json"
    if status_path.exists():
        finished = set(json.loads(status_path.read_text()).get("finished", []))
    slots = ("0-3", "90-93")
    while not (root / "STOP_DIAGNOSTICS").exists():
        for key, item in list(running.items()):
            code = item["process"].poll()
            if code is not None:
                item["log"].close()
                finished.add(key)
                del running[key]
                print("DIAG FINISH", key, code, flush=True)
        candidates = []
        for run in sorted((root / "runs").iterdir()):
            latest = run / "latest.pkl"
            if latest.exists():
                try:
                    with latest.open("rb") as f:
                        saved = CheckpointReader(f).load()
                    step = saved["step"]
                    snapshot = run / f"checkpoint_{step}.pkl"
                    if step in (100000, 300000, 1000000) and not snapshot.exists():
                        os.link(latest, snapshot)
                except (OSError, EOFError):
                    continue
            for step in (100000, 300000, 1000000):
                snapshot = run / f"checkpoint_{step}.pkl"
                key = f"{run.name}:{step}"
                done = run / f"deep_diagnostics_{step}_cpu_default.json"
                legacy = run / f"deep_diagnostics_{step}.json"
                if done.exists() or legacy.exists():
                    finished.add(key)
                if snapshot.exists() and key not in finished and key not in running:
                    candidates.append((step, key, run, snapshot))
        used = {item["slot"] for item in running.values()}
        candidates.sort(key=lambda x: (-x[0], x[1]))
        for slot in slots:
            if slot in used or not candidates:
                continue
            step, key, run, snapshot = candidates.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MUJOCO_GL="egl")
            cmd = ["taskset", "-c", slot, sys.executable, "-u", "-m",
                   "utils.propagation_research.diagnose_checkpoint", str(run),
                   "--checkpoint", snapshot.name, "--queries", "128", "--bank", "4096"]
            log = (run / f"deep_{step}.log").open("a")
            process = subprocess.Popen(cmd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            running[key] = {"process": process, "slot": slot, "log": log}
            print("DIAG START", key, process.pid, flush=True)
        payload = {"pid": os.getpid(), "updated_at": time.time(), "finished": sorted(finished),
                   "running": {k: v["process"].pid for k, v in running.items()}}
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2))
        os.replace(temp, status_path)
        time.sleep(15)


if __name__ == "__main__":
    main()
