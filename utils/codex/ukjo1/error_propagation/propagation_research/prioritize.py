"""Keep long evaluations and short mechanism experiments moving together.

Run between research iterations. This reorders pending work only; the live
scheduler and training processes keep their existing identities and states.
"""

import argparse
import json
import os
from pathlib import Path


def is_full(job):
    return int(job["args"].get("stop-at", job["args"].get("steps", 1000000))) >= 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--full-slots", type=int, default=12)
    args = parser.parse_args()
    root = Path(args.root)
    status = json.loads((root / "scheduler_status.json").read_text())
    os.kill(int(status["pid"]), 0)
    path = root / "queue.json"
    jobs = json.loads(path.read_text())
    assert len({j["id"] for j in jobs}) == len(jobs), "Duplicate job IDs"
    lookup = {j["id"]: j for j in jobs}
    finished = json.loads((root / "finished.json").read_text())
    live = {}
    for key, record in status["running"].items():
        try:
            os.kill(int(record["pid"]), 0)
            live[key] = record
        except ProcessLookupError:
            pass
    occupied = set(status["running"]) | set(finished)
    pending = [j for j in jobs if j["id"] not in occupied]
    long = [j for j in pending if is_full(j)]
    short = [j for j in pending if not is_full(j)]
    full_live = sum(is_full(lookup[key]) for key in live)
    quota = max(0, args.full_slots-full_live)
    # In the full-evaluation phase every future vacancy should take a full run.
    # Otherwise a currently full pool would have quota=0 and launch a pilot
    # immediately after a long run finishes.
    pending_order = long+short if args.full_slots >= 16 else long[:quota]+short+long[quota:]
    reordered = pending_order+[j for j in jobs if j["id"] in occupied]
    assert {j["id"] for j in reordered} == set(lookup)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(reordered, indent=2))
    os.replace(temporary, path)
    print(json.dumps({"live": len(live), "full_live": full_live,
                      "next_full_quota": quota, "pending_full": len(long),
                      "pending_pilots": len(short),
                      "next_jobs": [j["id"] for j in reordered[:min(6, len(pending))]]}, indent=2))


if __name__ == "__main__":
    main()
