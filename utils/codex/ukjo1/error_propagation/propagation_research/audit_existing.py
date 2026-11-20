"""Summarize historical causal interventions and their actual activation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob("causal_branches/*.jsonl")):
        steps, evals, start = [], [], {}
        with path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "branch_start":
                    start = record
                elif record.get("event") == "branch_step":
                    steps.append(record)
                elif "eval" in record.get("event", ""):
                    evals.append(record)
        if not steps:
            continue
        mean = lambda key: float(np.mean([x[key] for x in steps if key in x])) if any(key in x for x in steps) else None
        fraction = lambda key, fun: float(np.mean([fun(x[key]) for x in steps if key in x])) if any(key in x for x in steps) else None
        probe = steps[-1].get("probe", {})
        row = {"path": str(path), "kind": start.get("kind"), "start": start.get("timestep"),
               "last": steps[-1].get("timestep"), "n_records": len(steps),
               "q_damping_fraction": fraction("q_update_scale", lambda x: x < .999),
               "extra_v_fraction": fraction("residual_extra_v_fit_trigger", lambda x: x > .5),
               "relax_fraction": fraction("target_relax_alpha", lambda x: x < .999),
               "residual_target_fraction": fraction("residual_target_alpha", lambda x: x > 0),
               "mean_residual_alpha": mean("residual_target_alpha"),
               "mean_target_displacement": mean("stage2_applied_target_displacement_rms"),
               "last_q_rms": probe.get("q_common_rms"), "last_v_rms": probe.get("v_rms"),
               "last_q_ratio_baseline": steps[-1].get("q_rms_over_baseline"),
               "n_evals": len(evals), "last_eval": json.dumps(evals[-1]) if evals else None}
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        print(kind, "n=", len(group), {k: round(float(np.mean([r[k] for r in group if r[k] is not None])), 5)
              for k in ("q_damping_fraction", "extra_v_fraction", "relax_fraction",
                        "residual_target_fraction", "mean_target_displacement")
              if any(r[k] is not None for r in group)})


if __name__ == "__main__":
    main()
