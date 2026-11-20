"""Continue an existing scientific run in a separate, provenance-linked directory.

Only the evaluation episode count and screening endpoint change. Model parameters,
all optimizer states, training RNG and the original 1M actor schedule are retained.
The parent remains untouched. This does not create an independent training seed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from utils.propagation_research.diagnose_checkpoint import CheckpointReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent")
    parser.add_argument("output")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", default="latest.pkl")
    parser.add_argument("--allow-live", action="store_true",
                        help="Read an explicitly named immutable checkpoint of a live parent.")
    parser.add_argument("--replay-failure", action="store_true",
                        help="Replay a terminated run from its last full checkpoint for analysis only.")
    parser.add_argument("--evaluation-device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Destination is not empty: {output}")
    if (parent / "invalidated.json").exists():
        raise ValueError("An invalidated parent cannot be promoted or replayed")
    failed = (parent / "diverged.json").exists()
    if failed and not args.replay_failure:
        raise ValueError("A failed or invalidated parent cannot be promoted as a successful pilot")
    status_path = parent / "status.json"
    if args.replay_failure:
        if args.allow_live or not failed or not status_path.exists():
            raise ValueError("Failure replay requires a terminated failed parent")
        parent_status = json.loads(status_path.read_text())
        if parent_status.get("status") != "diverged":
            raise ValueError("Parent has not terminated as a failed run")
    elif args.allow_live:
        if args.checkpoint == "latest.pkl" or Path(args.checkpoint).name != args.checkpoint:
            raise ValueError("Live continuation requires an immutable checkpoint filename")
    elif not status_path.exists() or json.loads(status_path.read_text()).get("status") != "screened":
        raise ValueError("Parent must have finished screening before its checkpoint is continued")
    checkpoint = parent / args.checkpoint
    with checkpoint.open("rb") as f:
        saved = CheckpointReader(f).load()
    if saved.get("analysis_only"):
        raise ValueError("Analysis-only snapshots cannot resume training")
    step = int(saved["step"])
    if args.replay_failure and step >= int(parent_status["step"]):
        raise ValueError("Replay must begin before the recorded failure")
    if not args.allow_live and not args.replay_failure and int(json.loads(status_path.read_text())["step"]) != step:
        raise ValueError("Screening status and checkpoint step disagree")
    config = json.loads((parent / "config.json").read_text())
    budget = int(config["identity"]["config"]["max_timesteps"])
    if int(saved["state"].core.total_it) != step or not 0 < step < budget:
        raise ValueError("Parent must have a coherent, resumable partial training checkpoint")
    if "rng" not in saved:
        raise ValueError("Parent lacks the training RNG needed to preserve minibatches")
    original_protocol = dict(config["identity"]["protocol"])
    config["identity"]["protocol"]["episodes"] = args.episodes
    if args.evaluation_device != "gpu":
        config["identity"]["protocol"]["evaluation_device"] = args.evaluation_device
    config["arguments"].update({"output": str(output), "episodes": args.episodes,
                                 "stop_at": budget, "promote": False,
                                 "eval_device": args.evaluation_device})
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config["parent_continuation"] = {
        "parent": str(parent), "step": step, "checkpoint_sha256": digest,
        "parent_checkpoint": args.checkpoint,
        "parent_protocol": original_protocol, "time": time.time(),
        "independent_seed": False,
        "preserved": ["Q/V/actor parameters", "all optimizer states", "target networks",
                      "training RNG", "actor learning-rate schedule", "controller state"]}
    if args.replay_failure:
        config["analysis_replay_of_failure"] = {"parent_failure_step": parent_status["step"],
            "resume_step": step, "independent_seed": False,
            "benchmark_eligible": False, "purpose": "Preserve and inspect the terminal state"}
    output.mkdir(parents=True, exist_ok=True)
    # Parent checkpoints are atomically replaced, never mutated in place.
    try:
        os.link(checkpoint, output / "latest.pkl")
    except OSError:
        shutil.copy2(checkpoint, output / "latest.pkl")
    for name in ("diagnostics.jsonl", "evaluations.jsonl", "controller.jsonl"):
        source = parent / name
        if source.exists():
            records = []
            for line in source.read_text().splitlines():
                record = json.loads(line)
                if int(record["step"]) <= step:
                    if name == "evaluations.jsonl":
                        record.setdefault("eval_device", original_protocol.get("evaluation_device", "gpu"))
                    records.append(json.dumps(record)+"\n")
            (output / name).write_text("".join(records))
    for name in ("algorithm_source.py", "dataset.json"):
        if (parent / name).exists():
            shutil.copy2(parent / name, output / name)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"parent": str(parent), "output": str(output), "resume_step": step,
                      "remaining_updates": budget-step, "episodes_from_now": args.episodes,
                      "checkpoint_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
