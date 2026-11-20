"""Compare paired stress-case trajectories, retaining censored final scores.

Native IQL, the shared arithmetic control and the fixed candidate each have
three preassigned training seeds. This summary never pools a last observed
pre-failure policy score into a final1M mean.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from utils.propagation_research.summarize import records


FAMILIES = ("stress_baseline", "numerical_control", "candidate")
LABELS = {"stress_baseline": "Native IQL", "numerical_control": "Custom arithmetic control",
          "candidate": "Bounded + V2 + fresh"}
COLORS = {"stress_baseline": "#63656b", "numerical_control": "#bb792c", "candidate": "#277d85"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/"replication_plan.json").read_text())
    environments = plan["matched_stress_case_baselines"]["environments"]
    audit_path = root/"analysis/selected_benchmark_audit.json"
    audit = json.loads(audit_path.read_text())
    rows, histories = [], {}
    for row in audit["rows"]:
        if row["family"] not in FAMILIES or row["environment"] not in environments:
            continue
        result = {k: row.get(k) for k in ("family", "environment", "seed", "run", "status",
            "final_success", "last_training_step", "last_evaluation_step", "last_observed_success",
            "last_q_mean", "last_v_mean", "last_q_rms", "max_q_rms_over_return_bound")}
        result.update(declared_training_failure=row["status"] == "stopped_censored",
                      scale_stop=row["status"] == "stopped_censored" and
                          row.get("max_q_rms_over_return_bound", 0) > 100,
                      actual_evaluation_count=0, excluded_policy_points_with_other_protocols=0,
                      maximum_observed_policy_drawdown=None,
                      positive_mean_Q_fraction_after100k=None, max_centered_Q_RMS=None)
        if row["run"]:
            run = root/"runs"/row["run"]
            ds, es = records(run/"diagnostics.jsonl"), records(run/"evaluations.jsonl")
            ds = [d for d in ds if d["step"] <= row["last_training_step"]]
            observed = [e for e in es if e["step"] <= row["last_evaluation_step"]]
            # Seed0 CPU continuations inherited GPU and/or20-episode pilot
            # history. Preserve the logs, but do not label those points CPU50.
            es = [e for e in observed if e.get("eval_device") == "cpu" and e.get("n_episodes") == 50]
            result["excluded_policy_points_with_other_protocols"] = len(observed)-len(es)
            histories[row["run"]] = (ds, es)
            scores = np.asarray([e["success_rate"] for e in es], dtype=float)
            if len(scores):
                result["maximum_observed_policy_drawdown"] = float(np.max(np.maximum.accumulate(scores)-scores))
                result["actual_evaluation_count"] = len(scores)
            late = [d["q_mean"] for d in ds if d["step"] >= 100000 and d.get("q_mean") is not None]
            centered = [d["q_centered_rms"] for d in ds if d.get("q_centered_rms") is not None]
            if late:
                result["positive_mean_Q_fraction_after100k"] = float(np.mean(np.asarray(late)>0))
            if centered:
                result["max_centered_Q_RMS"] = float(max(centered))
            result["diagnostic_prefix_sha256"] = hashlib.sha256(json.dumps(ds, sort_keys=True).encode()).hexdigest()
            result["evaluation_prefix_sha256"] = hashlib.sha256(json.dumps(es, sort_keys=True).encode()).hexdigest()
        rows.append(result)
    assert len({(r["family"], r["environment"], r["seed"]) for r in rows}) == len(rows)
    assert all(r["final_success"] is None for r in rows if r["status"] != "verified_final")
    groups = []
    for env in environments:
        for family in FAMILIES:
            selected = [r for r in rows if r["environment"] == env and r["family"] == family]
            if not selected:
                continue
            scores = {str(r["seed"]): r["final_success"] for r in selected}
            all_final = all(r["status"] == "verified_final" for r in selected)
            groups.append({"environment": env, "family": family, "planned_seeds": sorted(r["seed"] for r in selected),
                "verified_final_count": sum(r["status"] == "verified_final" for r in selected),
                "scale_stop_count": sum(r["scale_stop"] for r in selected),
                "declared_training_failure_count": sum(r["declared_training_failure"] for r in selected),
                "pending_count": sum(r["status"] not in ("verified_final", "stopped_censored") for r in selected),
                "final1M_success_by_seed": scores,
                "all_planned_seeds_final1M_mean": float(np.mean(list(scores.values()))) if all_final else None,
                "all_planned_seeds_final1M_standard_deviation": float(np.std(list(scores.values()), ddof=1))
                    if all_final and len(selected)>1 else None})
    output = root/"analysis"
    report = {"time": time.time(), "rows": rows, "groups": groups,
        "audit_input_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caveats": ["Scale stop is the existing runner stopping rule, not a complete definition of stability.",
            "Positive mean Q violates the physical upper bound, but a common Q/V shift alone does not change actor advantage.",
            "Policy drawdown describes actual compatible checkpoint evaluations and includes episode sampling variation; no best-checkpoint selection.",
            "Policy curves and drawdowns use only CPU50 points. Inherited GPU/short-episode pilot points remain in the original logs but are excluded here.",
            "A finite completed model may still have severely distorted values or poor policy performance.",
            "Final group means require all planned seeds to reach verified1M evaluation. Censored final outcomes stay unobserved."]}
    (output/"paired_stress_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = [k for k in rows[0] if not k.endswith("_sha256")]
    with (output/"paired_stress_table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(groups, indent=2))
    if not args.plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(environments), 3, figsize=(14.5, 7.7), squeeze=False)
    styles = {0: "-", 1: "--", 2: ":"}
    for i, env in enumerate(environments):
        for row in rows:
            if row["environment"] != env or not row["run"]:
                continue
            ds, es = histories[row["run"]]
            color, style = COLORS[row["family"]], styles[row["seed"]]
            kw = dict(color=color, linestyle=style, linewidth=1.3, alpha=.9)
            axes[i, 0].plot([d["step"]/1000 for d in ds],
                            [d.get("q_rms_over_return_bound") for d in ds], **kw)
            axes[i, 1].plot([d["step"]/1000 for d in ds],
                            [d.get("q_centered_rms") for d in ds], **kw)
            axes[i, 2].plot([e["step"]/1000 for e in es], [e["success_rate"] for e in es], marker=".", **kw)
            if row["declared_training_failure"] and ds:
                axes[i, 0].plot(ds[-1]["step"]/1000, ds[-1]["q_rms_over_return_bound"],
                                marker="x", color=color, markersize=7)
        for j, title in enumerate(("Q RMS / recorded return scale", "Centered Q RMS (probe)", "Actual policy evaluations")):
            ax = axes[i, j]
            ax.set_title(("Cube" if env.startswith("cube-") else "Scene")+": "+title)
            ax.set_xlabel("Q updates (thousands)")
            ax.set_xlim(0, 1000)
            ax.grid(alpha=.2)
            if j < 2:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-.02, 1.02)
    handles = [Line2D([0], [0], color=COLORS[f], label=LABELS[f]) for f in FAMILIES]
    handles += [Line2D([0], [0], color="k", linestyle=styles[s], label=f"Seed {s}") for s in styles]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(.5, .036), frameon=False)
    fig.suptitle("Paired stress cases / expectile .9 / discount .999 / CPU50 / fixed1M budget")
    fig.text(.5, .012, "Missing 1M scores stay unobserved. x = training scale stop. V2 uses twice the V updates; finite/completed does not certify stable values.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .96))
    fig.savefig(output/"paired_stress_trajectories.png", dpi=170)
    fig.savefig(output/"paired_stress_trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
