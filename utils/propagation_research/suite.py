"""Prepare and verify a reproducible OGBench suite for the existing scheduler.

Example (creates files; launch run_suite.sh when the GPUs are available):
  python -m utils.propagation_research.suite logs/my_iql_suite \
      --profile bounded_v2 --discount .999 --seeds 0 1 2
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from utils.propagation_research.summarize import ENVIRONMENTS


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCES = ("algorithms/uk_offline/basic/iql_residual_control.py",
           "algorithms/uk_offline/basic/iql_jax.py", "hyperparams/ogbench/iql_jax.yml",
           "utils/propagation_research/reward_bounds.py")


def profiles():
    # Queue preparation does not initialize JAX or reserve GPU memory.
    tree = ast.parse((REPOSITORY / SOURCES[0]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROFILES"
                                               for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Cannot locate the algorithm profile definitions")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root / "suite_manifest.json").read_text())
    problems = []
    for name, digest in manifest["source_files_sha256"].items():
        path = REPOSITORY / name
        if not path.exists() or file_hash(path) != digest:
            problems.append(name)
    if file_hash(root / "queue.json") != manifest["queue_sha256"]:
        problems.append("queue.json")
    if problems:
        raise ValueError("Reproduction inputs have changed: " + ", ".join(problems))
    print(f"Verified {manifest['number_of_runs']} queued runs and their source/config hashes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true", help="Check an existing suite without launching jobs")
    parser.add_argument("--profile", choices=profiles(), default="bounded_v2")
    parser.add_argument("--expectile", type=float, default=.9)
    parser.add_argument("--discount", default=".999", help="Discount factor or 'original'")
    parser.add_argument("--bound-source", choices=("empirical", "ogbench"), default="ogbench")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS, default=ENVIRONMENTS)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100000)
    parser.add_argument("--chunk", type=int, default=100)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.verify:
        verify(root)
        return
    if not 0 < args.expectile < 1:
        parser.error("Expectile must be strictly between 0 and 1")
    discount = None if args.discount == "original" else float(args.discount)
    if discount is not None and not 0 < discount < 1:
        parser.error("Discount must be strictly between 0 and 1")
    if min(args.steps, args.episodes, args.eval_every, args.chunk) <= 0:
        parser.error("Step, episode and chunk counts must be positive")
    endpoint = args.stop_at or args.steps
    if not 0 < endpoint <= args.steps or endpoint % args.chunk or args.eval_every % args.chunk:
        parser.error("Screening endpoint/evaluation interval must align with chunks and fit within the schedule")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.environments)) != len(args.environments):
        parser.error("Seeds and environments must not contain duplicates")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Use an empty destination; existing experiments are preserved: {root}")
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for env in args.environments:
            arguments = {"env": env, "profile": args.profile, "expectile": args.expectile,
                         "seed": seed, "steps": args.steps, "stop-at": endpoint,
                         "eval-every": args.eval_every, "episodes": args.episodes,
                         "chunk": args.chunk, "eval-device": "cpu"}
            if discount is not None:
                arguments["discount"] = discount
            if profiles()[args.profile].get("bound_backup"):
                arguments["bound-source"] = args.bound_source
            jobs.append({"id": f"{args.profile}_{env}_seed{seed}", "args": arguments})
    queue_path = root / "queue.json"
    queue_path.write_text(json.dumps(jobs, indent=2))
    source_hashes = {name: file_hash(REPOSITORY / name) for name in SOURCES}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {"profile": args.profile, "profile_parameters": profiles()[args.profile],
        "expectile": args.expectile, "discount": discount if discount is not None else "original",
        "environments": args.environments, "training_seeds": args.seeds,
        "number_of_runs": len(jobs), "Q_updates": endpoint,
        "V_updates": endpoint*profiles()[args.profile].get("v_steps", 1),
        "actor_schedule": args.steps, "final_evaluation_episodes": args.episodes,
        "evaluation_device": "cpu", "source_files_sha256": source_hashes,
        "return_bound_source": args.bound_source if profiles()[args.profile].get("bound_backup") else "unused",
        "queue_sha256": file_hash(queue_path),
        "git_head": revision,
        "scheduler": {"GPUs": 2, "maximum_runs_per_GPU": 8, "XLA_memory_fraction": .1},
        "selection": "Final checkpoint; no best-checkpoint selection"}
    (root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2))
    snapshots = root / "source_inputs"
    snapshots.mkdir()
    for name in SOURCES:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / name).read_bytes())
    command = [sys.executable, "-u", "-m", "utils.propagation_research.scheduler", str(root)]
    script = ("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPOSITORY)) + "\n" +
              shlex.join([sys.executable, "-m", "utils.propagation_research.suite", str(root), "--verify"]) +
              "\nexec " + shlex.join(command) + "\n")
    launch = root / "run_suite.sh"
    launch.write_text(script)
    launch.chmod(0o755)
    verify(root)
    print(f"Prepared {len(jobs)} runs. Launch {launch} when two GPUs are available.")


if __name__ == "__main__":
    main()
