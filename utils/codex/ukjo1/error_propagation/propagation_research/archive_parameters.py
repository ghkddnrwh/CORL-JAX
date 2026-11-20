"""Retain three recent parameter snapshots for late-collapse counterfactuals.

These are analysis-only: optimizer states are deliberately omitted. Permanent
100k/300k/final training checkpoints remain untouched. Source bytes are hashed
and decoded together so an atomic latest-checkpoint update cannot mix versions.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import time

from utils.propagation_research.diagnose_checkpoint import CheckpointReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--keep", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.root)
    seen, archived = {}, 0
    while not (root / "STOP_PARAMETER_ARCHIVE").exists():
        for run in sorted((root / "runs").iterdir()):
            source = run / "latest.pkl"
            if not source.exists():
                continue
            stat = source.stat()
            token = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if seen.get(str(run)) == token:
                continue
            try:
                raw = source.read_bytes()
                saved = CheckpointReader(io.BytesIO(raw)).load()
                seen[str(run)] = token
                step = int(saved["step"])
                if step in (100000, 300000, 1000000):
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                destination = run / f"parameters_{step:07d}_{digest[:12]}.pkl"
                if destination.exists():
                    continue
                core = saved["state"].core.replace(q_opt_state=None, v_opt_state=None,
                                                    actor_opt_state=None)
                state = saved["state"].replace(core=core, kappa_opt=None)
                payload = {"analysis_only": True, "optimizer_states_retained": False,
                           "step": step, "state": state, "source_checkpoint_sha256": digest,
                           "captured_at": time.time()}
                temp = destination.with_suffix(".pkl.tmp")
                with temp.open("wb") as f:
                    pickle.dump(payload, f, protocol=4)
                os.replace(temp, destination)
                archived += 1
                retained = sorted(run.glob("parameters_*.pkl"), key=lambda p: p.stat().st_mtime_ns)
                for old in retained[:-args.keep]:
                    old.unlink()
                print("ARCHIVED", run.name, step, destination.name, flush=True)
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                print("ARCHIVE RETRY", run.name, str(exc), flush=True)
        state = {"pid": os.getpid(), "updated_at": time.time(), "archived": archived,
                 "keep_per_run": args.keep}
        temp = root / "parameter_archive_status.tmp"
        temp.write_text(json.dumps(state, indent=2))
        os.replace(temp, root / "parameter_archive_status.json")
        time.sleep(20)


if __name__ == "__main__":
    main()
