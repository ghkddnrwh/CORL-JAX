"""Honest live tables: final evaluations, coverage, and separate pilot results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
if __package__:
    from utils.propagation_research.reward_bounds import effective_backup_rule
else:
    from reward_bounds import effective_backup_rule


ENVIRONMENTS = ["antmaze-large-navigate-singletask-v0",
                "humanoidmaze-medium-navigate-singletask-v0",
                "humanoidmaze-large-navigate-singletask-v0",
                "antsoccer-arena-navigate-singletask-v0",
                "cube-single-play-singletask-v0", "cube-double-play-singletask-v0",
                "puzzle-3x3-play-singletask-v0", "puzzle-4x4-play-singletask-v0",
                "scene-play-singletask-v0"]
CRITERION = dict(zip(ENVIRONMENTS, [.14, .44, .05, .06, .84, .00, .01, .04, .11]))


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def collect(root):
    rows, histories = [], {}
    for run in sorted((root / "runs").iterdir()):
        path = run / "config.json"
        if not path.exists():
            continue
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg, arguments = config["identity"]["config"], config["arguments"]
        dataset_path = run / "dataset.json"
        bound_rule = (effective_backup_rule(config, json.loads(dataset_path.read_text()))
                      if dataset_path.exists() else "not_initialized")
        diagnostics = records(run / "diagnostics.jsonl")
        evals = records(run / "evaluations.jsonl")
        last_diag = diagnostics[-1] if diagnostics else {}
        last_eval = evals[-1] if evals else {}
        parent = config.get("parent_continuation", {})
        previous_backend = parent.get("parent_protocol", {}).get("evaluation_device", "gpu")
        backend = last_eval.get("eval_device", previous_backend if last_eval.get("step", 0) <= parent.get("step", -1)
                                else config["identity"]["protocol"].get("evaluation_device", "gpu"))
        row = {"id": run.name, "environment": cfg["env"], "profile": arguments["profile"],
               "seed": cfg["seed"], "expectile": cfg["iql_tau"], "discount": cfg["discount"],
               "discount_setting": arguments.get("discount") or "original",
               "initialization": arguments.get("initialization", "iql"),
               "evaluation_device": backend,
               "backup_bound_rule": bound_rule,
               "superseded": (run / "superseded.json").exists(),
               "analysis_replay": bool(config.get("analysis_replay_of_failure")),
               "inherited_updates": parent.get("step", 0),
               "last_step": last_diag.get("step", 0), "eval_step": last_eval.get("step", 0),
               "success_rate": last_eval.get("success_rate"),
               "n_episodes": last_eval.get("n_episodes"), "reference_success": CRITERION.get(cfg["env"]),
               "last_q_rms": last_diag.get("q_rms"), "last_v_rms": last_diag.get("v_rms"),
               "max_q_rms_over_bound": max((d["q_rms_over_return_bound"] for d in diagnostics
                   if isinstance(d.get("q_rms_over_return_bound"), (int, float))
                   and np.isfinite(d["q_rms_over_return_bound"])), default=0),
               "last_td_rms": last_diag.get("td_rms"),
               "last_score_over_kappa": last_diag.get("global_score_over_kappa"),
               "updates_per_second": last_diag.get("updates_per_second"),
               "complete_1m": (run / "completed.json").exists(),
               "diverged": (run / "diverged.json").exists() and not (run / "invalidated.json").exists(),
               "invalid_instrumentation": (run / "invalidated.json").exists()}
        rows.append(row)
        histories[run.name] = (diagnostics, evals)
    return rows, histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    rows, histories = collect(root)
    with (output / "live_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['id']:42s} step={row['last_step']:7d} eval={row['eval_step']:7d} "
              f"success={row['success_rate']} Q={row['last_q_rms']} "
              f"max_Q/bound={row['max_q_rms_over_bound']:.3g}")
    groups = {}
    for row in rows:
        if row["invalid_instrumentation"] or row["analysis_replay"] or row["superseded"]:
            continue
        if not row["complete_1m"] or row["eval_step"] < 1000000 or row["n_episodes"] < 50:
            continue
        key = (row["profile"], row["expectile"], row["discount_setting"], row["initialization"],
               row["evaluation_device"], row["backup_bound_rule"])
        groups.setdefault(key, {}).setdefault(row["environment"], []).append(row)
    full_summaries = []
    for key, environments in groups.items():
        coverage = len(environments)
        print("FULL EVALUATION COVERAGE", key, coverage, "/ 9")
        item = {"profile": key[0], "expectile": key[1], "discount_setting": key[2],
                "initialization": key[3], "evaluation_device": key[4],
                "backup_bound_rule": key[5],
                "environment_coverage": coverage, "required_environments": ENVIRONMENTS,
                "final_step": 1000000, "minimum_evaluation_episodes": 50,
                "threshold": .19, "nine_environment_mean": None,
                "threshold_exceeded": False, "environments": {}}
        for env, rs in environments.items():
            seeds = sorted({r["seed"] for r in rs})
            values = [float(np.mean([r["success_rate"] for r in rs if r["seed"] == s]))
                      for s in seeds]
            item["environments"][env] = {
                "independent_training_seeds": seeds, "per_seed_success": dict(zip(seeds, values)),
                "mean": float(np.mean(values)),
                "seed_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "run_ids": [r["id"] for r in rs],
                "any_declared_divergence": any(r["diverged"] for r in rs),
                "max_observed_q_rms_over_return_bound": max(r["max_q_rms_over_bound"] for r in rs)}
        if set(environments) == set(ENVIRONMENTS):
            by_seed = {e: {seed: float(np.mean([r["success_rate"] for r in environments[e]
                                              if r["seed"] == seed]))
                           for seed in {r["seed"] for r in environments[e]}} for e in ENVIRONMENTS}
            common_seeds = sorted(set.intersection(*(set(by_seed[e]) for e in ENVIRONMENTS)))
            available = float(np.mean([np.mean(list(by_seed[e].values())) for e in ENVIRONMENTS]))
            # Asynchronous replications finish at different times per task.
            # A nine-task mean uses complete seed cohorts, not a changing mix
            # of1/2/3 seeds across environments.
            avg = (float(np.mean([by_seed[e][s] for e in ENVIRONMENTS for s in common_seeds]))
                   if common_seeds else None)
            print("9-environment final mean", key, avg,
                  "seed coverage", {e: len(by_seed[e]) for e in ENVIRONMENTS},
                  "per-seed 9-environment means", {s: float(np.mean([by_seed[e][s] for e in ENVIRONMENTS]))
                                                    for s in common_seeds})
            item.update({"nine_environment_mean": avg,
                         "threshold_exceeded": avg is not None and avg > .19,
                         "available_seed_environment_mean": available,
                         "nine_environment_mean_uses_complete_seed_cohorts": True,
                         "common_training_seeds": common_seeds,
                         "per_seed_nine_environment_mean": {
                             s: float(np.mean([by_seed[e][s] for e in ENVIRONMENTS]))
                             for s in common_seeds}})
        full_summaries.append(item)
    (output / "full_evaluation_summary.json").write_text(json.dumps(full_summaries, indent=2))
    if not args.plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for env in sorted({r["environment"] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for row in rows:
            if row["environment"] != env:
                continue
            ds, es = histories[row["id"]]
            label = f"{row['profile']} tau={row['expectile']} g={row['discount']} s={row['seed']}"
            if row["backup_bound_rule"] != "none":
                label += " bounds="+row["backup_bound_rule"]
            if row["analysis_replay"]:
                label += " (failure replay; analysis only)"
            if ds:
                axes[0].plot([d["step"] for d in ds], [d["q_rms_over_return_bound"]
                    if d.get("q_rms_over_return_bound") is not None else np.nan for d in ds], label=label)
                axes[1].plot([d["step"] for d in ds], [abs(d["global_score_over_kappa"])
                    if d.get("global_score_over_kappa") is not None else np.nan for d in ds], label=label)
            if es:
                axes[2].plot([e["step"] for e in es], [e["success_rate"] for e in es], marker="o", label=label)
        axes[0].set_yscale("log")
        axes[1].set_yscale("symlog", linthresh=.001)
        for ax, title in zip(axes, ["Q RMS / recorded return bound",
                                    "Absolute global G/kappa (proxy)", "Evaluation success"]):
            ax.set_title(title)
            ax.set_xlabel("Q updates")
            ax.grid(alpha=.2)
        axes[2].axhline(CRITERION[env], ls="--", color="k", lw=1)
        axes[2].set_ylim(-.02, 1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
        fig.suptitle(env)
        fig.subplots_adjust(bottom=.35, top=.85)
        fig.savefig(output / (env+".png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
