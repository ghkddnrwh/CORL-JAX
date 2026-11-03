"""Generate final tables and a standalone figure from audited real outcomes.

Requires all48 model outcomes and all36 frozen secondary evaluations. Censored
stress cases stay unobserved; they never enter the nine-task performance means.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from utils.propagation_research.evaluate_seeded_finals import valid_result
from utils.propagation_research.summarize import CRITERION, ENVIRONMENTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    audit_path = output / "selected_benchmark_audit.json"
    audit = json.loads(audit_path.read_text())
    assert len(audit["rows"]) == 48
    assert all(r["status"] in ("verified_final", "stopped_censored") for r in audit["rows"])
    by_run = {r["run"]: r for r in audit["rows"]}
    secondary_dir = output / "seeded_reset_cpu50_v1"
    secondary = json.loads((secondary_dir / "completed.json").read_text())
    manifest = json.loads((secondary_dir / "manifest.json").read_text())
    assert len(secondary["rows"]) == 36
    for name, expected in manifest["source_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == expected
    for job in manifest["jobs"]:
        row = by_run.get(job["run"])
        checkpoint_hash = (row["model_and_episode_audit"]["checkpoint_sha256"]
                           if job["family"] != "validation_only" else
                           sha(root / "runs" / job["run"] / "checkpoint_1000000.pkl"))
        assert valid_result(secondary_dir / (job["id"] + ".json"), job,
                            checkpoint_hash, manifest["source_sha256"])
    assert json.loads((secondary_dir / "equal_actor_validation.json").read_text())["passed"]

    def primary(environment, family):
        rows = sorted([r for r in audit["rows"] if r["environment"] == environment
                       and r["family"] == family], key=lambda r: r["seed"])
        assert all(r["status"] == "verified_final" for r in rows)
        return [r["final_success"] for r in rows]

    def frozen(environment, family):
        rows = sorted([r for r in secondary["rows"] if r["environment"] == environment
                       and r["family"] == family], key=lambda r: r["seed"])
        assert all(r["status"] == "verified_secondary" for r in rows)
        return [r["secondary_success"] for r in rows]

    table = []
    for environment in ENVIRONMENTS:
        c, n = primary(environment, "candidate"), primary(environment, "original_baseline")
        cs, ns = frozen(environment, "candidate"), frozen(environment, "original_baseline")
        assert len(c) == len(cs) == 3 and len(n) == len(ns) == 1
        table.append({"environment": environment, "provided_IQL_mean": CRITERION[environment],
            "native_original_seed0_primary": n[0], "candidate_primary_seed0": c[0],
            "candidate_primary_seed1": c[1], "candidate_primary_seed2": c[2],
            "candidate_primary_mean": float(np.mean(c)), "candidate_primary_sample_SD": float(np.std(c, ddof=1)),
            "native_original_seed0_secondary": ns[0], "candidate_secondary_seed0": cs[0],
            "candidate_secondary_seed1": cs[1], "candidate_secondary_seed2": cs[2],
            "candidate_secondary_mean": float(np.mean(cs)),
            "candidate_secondary_sample_SD": float(np.std(cs, ddof=1))})
    candidate_macros = np.mean([[r[f"candidate_primary_seed{s}"] for r in table] for s in range(3)], axis=1)
    secondary_macros = np.mean([[r[f"candidate_secondary_seed{s}"] for r in table] for s in range(3)], axis=1)
    native = float(np.mean([r["native_original_seed0_primary"] for r in table]))
    native_secondary = float(np.mean([r["native_original_seed0_secondary"] for r in table]))
    assert np.isclose(candidate_macros.mean(), audit["nine_environment_macros"]["candidate"]["mean_over_all_planned_seeds"])
    assert np.isclose(native, audit["nine_environment_macros"]["original_baseline"]["mean_over_all_planned_seeds"])
    assert np.isclose(secondary_macros.mean(), secondary["macros"]["candidate"]["all_planned_seed_mean"])
    assert np.isclose(native_secondary, secondary["macros"]["original_baseline"]["all_planned_seed_mean"])
    assert (candidate_macros > .19).all() and (secondary_macros > .19).all()
    average = {"environment": "ALL9_MACRO", "provided_IQL_mean": .19,
        "native_original_seed0_primary": native, "candidate_primary_mean": float(candidate_macros.mean()),
        "candidate_primary_sample_SD": float(np.std(candidate_macros, ddof=1)),
        "native_original_seed0_secondary": native_secondary,
        "candidate_secondary_mean": float(secondary_macros.mean()),
        "candidate_secondary_sample_SD": float(np.std(secondary_macros, ddof=1))}
    for seed in range(3):
        average[f"candidate_primary_seed{seed}"] = float(candidate_macros[seed])
        average[f"candidate_secondary_seed{seed}"] = float(secondary_macros[seed])
    candidate = [r for r in audit["rows"] if r["family"] == "candidate"]
    result = {"time": time.time(), "kind": "final_audited_primary_and_secondary_comparison",
        "input_sha256": {"primary_audit": sha(audit_path), "secondary_completed": sha(secondary_dir/"completed.json"),
                         "secondary_manifest": sha(secondary_dir/"manifest.json"), "report_source": sha(Path(__file__))},
        "table": table, "macro": average,
        "independent_followup_primary_mean": float(candidate_macros[1:].mean()),
        "independent_followup_secondary_mean": float(secondary_macros[1:].mean()),
        "primary_gain_over_fresh_native_seed0": float(candidate_macros.mean()-native),
        "secondary_gain_over_fresh_native_seed0": float(secondary_macros.mean()-native_secondary),
        "primary_candidate_actual_successes": int(round(1350*candidate_macros.mean())),
        "secondary_candidate_actual_successes": int(round(1350*secondary_macros.mean())),
        "primary_native_actual_successes": int(round(450*native)),
        "secondary_native_actual_successes": int(round(450*native_secondary)),
        "candidate_max_observed_probe_Q_RMS_over_recorded_return_bound": max(r["max_q_rms_over_return_bound"] for r in candidate),
        "censored_model_cells": [dict(family=r["family"], environment=r["environment"], seed=r["seed"],
            run=r["run"], last_training_step=r["last_training_step"], final_success=None)
            for r in audit["rows"] if r["status"] == "stopped_censored"],
        "caveats": ["Candidate uses2M Vupdates; native uses1M. Both use1M Q/actor and1M actor schedule.",
            "Seed0 selected the method; seeds1/2 are fixed followups.",
            "Fresh native original-discount reference has one training seed; no statistical significance is claimed.",
            "SampleSD is across per-training-seed scores and includes finite evaluation noise.",
            "Secondary includes fresh environment/global RNG and cached action-space seeding; no primary scores are replaced.",
            "Provided IQL macro.19 is quoted from criterion.txt, not recomputed from its rounded per-task cells.",
            "Finite state and negative mean alone do not prove stable/accurate values. Inspect the stress and tail diagnostics."]}
    (output/"final_benchmark_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (output/"final_benchmark_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table + [average])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(9)
    nvals = [r["native_original_seed0_primary"] for r in table]
    cvals = [r["candidate_primary_mean"] for r in table]
    sd = [r["candidate_primary_sample_SD"] for r in table]
    axes[0].bar(x-.18, nvals, .34, color="#8b969f", label="Native IQL, original discount (seed0)")
    axes[0].bar(x+.18, cvals, .34, yerr=sd, capsize=3, color="#26758b", alpha=.85,
                label="bounded_v2, discount = 0.999 (3 seeds)")
    for seed, jitter in enumerate([-.055, 0, .055]):
        axes[0].scatter(x+.18+jitter, [r[f"candidate_primary_seed{seed}"] for r in table],
                        s=13, color="black", alpha=.65, zorder=4)
    axes[0].set_xticks(x, ["AM-L", "HM-M", "HM-L", "AS-A", "Cube-S", "Cube-D", "Pz3", "Pz4", "Scene"], rotation=30)
    axes[0].set_ylim(-.025, 1.065)
    axes[0].set_ylabel("Final success rate")
    axes[0].set_title("Primary evaluation: final 1M / CPU 50")
    axes[0].legend(fontsize=8, loc="upper left")
    positions = [0, 1, 3, 4]
    values = [native, candidate_macros.mean(), native_secondary, secondary_macros.mean()]
    axes[1].bar(positions, values, .7, color=["#8b969f", "#26758b"]*2, alpha=.85)
    axes[1].errorbar([1, 4], [candidate_macros.mean(), secondary_macros.mean()],
                    yerr=[np.std(candidate_macros, ddof=1), np.std(secondary_macros, ddof=1)],
                    fmt="none", color="black", capsize=4, lw=1)
    for pos, points in [(1, candidate_macros), (4, secondary_macros)]:
        axes[1].scatter(pos+np.array([-.07, 0, .07]), points, color="black", s=17, zorder=4)
    for pos, value in zip(positions, values):
        axes[1].text(pos, value+.007, f"{value:.3f}", ha="center", fontsize=9)
    axes[1].axhline(.19, color="#af5738", ls="--", lw=1, label="Provided IQL reference: 0.19")
    axes[1].set_xticks(positions, ["IQL\nprimary", "Candidate\nprimary", "IQL\nfrozen", "Candidate\nfrozen"])
    axes[1].set_ylim(0, .37)
    axes[1].set_title("Nine-task macro, equal task weights")
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("IQL residual / propagation study: audited final models", fontsize=13)
    fig.text(.5, .015, "Q/actor: 1M updates; candidate V: 2M vs native V: 1M. Candidate seed 0 used for selection; seeds 1/2 are fixed followups.\n"
             "Points: per-training-seed scores. Error bars: sample SD (3 seeds); native reference uses 1 seed. Frozen evaluation is separate.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(output/("final_benchmark_comparison."+extension), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"primary_candidate": float(candidate_macros.mean()), "primary_native": native,
        "secondary_candidate": float(secondary_macros.mean()), "secondary_native": native_secondary,
        "independent_followup_primary": result["independent_followup_primary_mean"],
        "candidate_peak_Q_norm_ratio": result["candidate_max_observed_probe_Q_RMS_over_recorded_return_bound"]}, indent=2))


if __name__ == "__main__":
    main()
