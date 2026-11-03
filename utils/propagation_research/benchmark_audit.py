"""Audit final models and episodes without filling in unfinished environments.

This is a read-only CPU audit of the fixed replication plan. Training continuations
remain one seed, failure replays never enter the benchmark, and a missing or
stopped 1M evaluation stays unobserved. Run after new final checkpoints arrive.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from utils.propagation_research.summarize import ENVIRONMENTS, collect, records


def optimizer_counts(tree):
    """Return actual Adam/schedule counters, including nested Optax chains."""
    counts = []
    if hasattr(tree, "_fields") and "count" in tree._fields:
        counts.append(int(np.asarray(tree.count)))
    elif isinstance(tree, (tuple, list)):
        for part in tree:
            counts.extend(optimizer_counts(part))
    elif isinstance(tree, dict):
        for part in tree.values():
            counts.extend(optimizer_counts(part))
    return counts


def audit_lineage(run, config):
    """Verify that resumes preserve the scientific identity and the actual seed."""
    from utils.propagation_research.diagnose_checkpoint import CheckpointReader
    failures, links, seen = [], [], set()
    expected = config["identity"]
    critical = ("env", "seed", "discount", "iql_tau", "tau", "beta", "normalize",
                "normalize_reward", "hidden_dim", "n_hidden", "num_critics", "batch_size",
                "max_timesteps", "qf_lr", "vf_lr", "actor_lr", "actor_dropout", "iql_deterministic")
    current = run.resolve()
    while config.get("parent_continuation"):
        if current in seen:
            failures.append("Continuation lineage contains a cycle")
            break
        seen.add(current)
        link = config["parent_continuation"]
        parent = Path(link["parent"])
        parent_config = parent/"config.json"
        if not parent_config.exists():
            failures.append("Continuation parent identity is unavailable")
            break
        ancestor = json.loads(parent_config.read_text())
        identity = ancestor["identity"]
        for key in critical:
            if identity["config"][key] != expected["config"][key]:
                failures.append(f"Continuation ancestor changed {key}")
        if identity["profile"] != expected["profile"]:
            failures.append("Continuation ancestor used a different update profile")
        if ancestor["arguments"].get("initialization", "iql") != "iql":
            failures.append("Continuation ancestor did not use native IQL initialization")
        if link.get("independent_seed") is not False:
            failures.append("Continuation is incorrectly labeled as an independent seed")
        if ancestor.get("analysis_replay_of_failure"):
            failures.append("Continuation descended from an analysis-only failure replay")
        # Older screened continuations predate the explicit filename field.
        checkpoint = parent/link.get("parent_checkpoint", "latest.pkl")
        verified = checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == link["checkpoint_sha256"]
        if not verified:
            failures.append("Inherited full-checkpoint hash cannot be verified")
        else:
            with checkpoint.open("rb") as stream:
                saved = CheckpointReader(stream).load()
            inherited_step = int(link["step"])
            if (saved.get("analysis_only") or int(saved["step"]) != inherited_step
                    or int(saved["state"].core.total_it) != inherited_step or "rng" not in saved):
                failures.append("Inherited full state or training RNG is inconsistent")
            for name, multiplier in (("q_opt_state", 1), ("v_opt_state", expected["profile"].get("v_steps", 1)),
                                     ("actor_opt_state", 1)):
                counts = optimizer_counts(getattr(saved["state"].core, name))
                if not counts or any(c != inherited_step*multiplier for c in counts):
                    failures.append("Inherited optimizer counters are inconsistent")
        links.append({"parent": str(parent), "step": link["step"],
                      "inherited_checkpoint_sha256_verified": bool(verified),
                      "same_training_seed": identity["config"]["seed"] == expected["config"]["seed"]})
        current, config = parent.resolve(), ancestor
    return {"passed": not failures, "failures": failures, "links": links,
            "root_run": str(current), "root_training_seed": config["identity"]["config"]["seed"],
            "continuations_are_not_additional_independent_seeds": True}


def audit_final(run, row, plan, candidate):
    # Lazy imports keep the table/coverage utilities free of a JAX initialization.
    from utils.propagation_research.diagnose_checkpoint import CheckpointReader
    import jax

    failures = []
    config = json.loads((run / "config.json").read_text())
    dataset = json.loads((run / "dataset.json").read_text())
    cfg = config["identity"]["config"]
    completed = json.loads((run / "completed.json").read_text())
    evaluations = [e for e in records(run / "evaluations.jsonl")
                   if e["step"] == plan["Q_updates"]]
    if len(evaluations) != 1:
        return {"passed": False, "failures": ["Expected exactly one final evaluation"]}
    evaluation = evaluations[0]
    successes = np.asarray(evaluation.get("episode_successes", []), dtype=float)
    rewards = np.asarray(evaluation.get("episode_rewards", []), dtype=float)
    expected_episodes = plan["final_evaluation"]["episodes"]
    if (successes.shape != (expected_episodes,) or rewards.shape != (expected_episodes,)
            or not np.isin(successes, [0., 1.]).all() or not np.isfinite(rewards).all()):
        failures.append("Final episode arrays must contain 50 finite binary outcomes and rewards")
    if (not len(successes) or not np.isclose(successes.mean(), evaluation["success_rate"],
                                           rtol=0, atol=1e-12)):
        failures.append("Final success rate disagrees with episode outcomes")
    if evaluation.get("eval_device") != plan["final_evaluation"]["device"]:
        failures.append("Final evaluation backend differs from the fixed protocol")
    if evaluation.get("n_episodes") != expected_episodes:
        failures.append("Final evaluation episode count differs from the fixed protocol")
    if completed.get("step") != plan["Q_updates"] or completed.get("status") != "complete":
        failures.append("Completion record is not the scheduled final update")
    if cfg["max_timesteps"] != plan["actor_schedule"]:
        failures.append("Actor learning-rate schedule differs from the fixed protocol")
    checkpoint = run / f"checkpoint_{plan['Q_updates']}.pkl"
    if not checkpoint.exists():
        return {"passed": False, "failures": failures + ["Final full checkpoint is missing"]}
    payload = checkpoint.read_bytes()
    import io
    with io.BytesIO(payload) as stream:
        saved = CheckpointReader(stream).load()
    if saved.get("analysis_only"):
        failures.append("Analysis-only checkpoint is ineligible for the final benchmark")
    if int(saved["step"]) != plan["Q_updates"]:
        failures.append("Saved model step differs from the scheduled final update")
    core = saved["state"].core
    if int(core.total_it) != plan["Q_updates"]:
        failures.append("Actual critic update count differs from the scheduled final update")
    counts = {name: optimizer_counts(getattr(core, attr)) for name, attr in
              [("Q", "q_opt_state"), ("V", "v_opt_state"), ("actor", "actor_opt_state")]}
    expected_counts = {"Q": plan["Q_updates"], "V": plan["V_updates"] if candidate
                       else plan["Q_updates"], "actor": plan["Q_updates"]}
    for name, values in counts.items():
        if not values or any(v != expected_counts[name] for v in values):
            failures.append(f"Actual {name} optimizer counters disagree with the protocol")
    all_finite = all(np.isfinite(np.asarray(x)).all()
                     for x in jax.tree_util.tree_leaves(saved["state"]))
    if not all_finite:
        failures.append("Final parameters or optimizer state contain nonfinite values")
    expected_profile = (plan["profile_parameters"] if candidate else
                        {"numerical_control": True} if row["profile"] == "numerical_control" else {})
    if config["identity"]["profile"] != expected_profile:
        failures.append("Update profile differs from the specified comparison method")
    lineage = audit_lineage(run, config)
    if not lineage["passed"]:
        failures.extend(lineage["failures"])
    return {"passed": not failures, "failures": failures,
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "evaluation_sha256": hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest(),
            "optimizer_counters": counts, "all_state_finite": bool(all_finite),
            "return_bounds": [dataset["return_lower"], dataset["return_upper"]],
            "return_bound_source_label": dataset.get("return_bound_source", "empirical"),
            "effective_bound_rule": row["backup_bound_rule"],
            "parent_continuation": config.get("parent_continuation"),
            "verified_training_lineage": lineage,
            "success_rate": evaluation["success_rate"],
            "episodes": expected_episodes, "evaluation_device": evaluation["eval_device"]}


def family_rows(root, collected, plan, family):
    expected_seeds = plan["training_seeds"] if family == "candidate" else [0]
    environments = ENVIRONMENTS
    if family == "stress_baseline":
        expected_seeds = plan["matched_stress_case_baselines"]["training_seeds"]
        environments = plan["matched_stress_case_baselines"]["environments"]
    elif family == "numerical_control":
        expected_seeds = plan["numerical_implementation_control_replication"]["training_seeds"]
        environments = plan["numerical_implementation_control_replication"]["environments"]
    output = []
    for seed in expected_seeds:
        for environment in environments:
            matches = []
            for row in collected:
                if (row["seed"] != seed or row["environment"] != environment
                        or row["initialization"] != "iql" or row["superseded"]
                        or row["analysis_replay"] or row["invalid_instrumentation"]
                        or row["evaluation_device"] != "cpu" or row["expectile"] != plan["expectile"]):
                    continue
                if family == "candidate":
                    match = (row["profile"] == plan["profile"] and row["discount"] == plan["discount"]
                             and row["backup_bound_rule"] == "known_reward_bounds")
                elif family == "numerical_control":
                    match = row["profile"] == "numerical_control" and row["discount"] == plan["discount"]
                else:
                    match = row["profile"] == "baseline" and (
                        row["discount_setting"] == "original" if family == "original_baseline"
                        else row["discount"] == plan["discount"])
                if match:
                    matches.append(row)
            # Do not average repeated continuations or choose the best run of a seed.
            completed = [r for r in matches if r["complete_1m"] and r["eval_step"] == plan["Q_updates"]]
            record = {"family": family, "environment": environment, "seed": seed,
                      "run": None, "status": "not_started_or_no_observation", "final_success": None,
                      "last_evaluation_step": None, "last_observed_success": None,
                      "model_and_episode_audit": None}
            if len(completed) > 1:
                record.update(status="ambiguous_multiple_finals", matching_runs=[r["id"] for r in completed])
            elif completed:
                row = completed[0]
                audit = audit_final(root / "runs" / row["id"], row, plan, family == "candidate")
                record.update(run=row["id"], status="verified_final" if audit["passed"] else "audit_failed",
                              final_success=audit["success_rate"] if audit["passed"] else None,
                              model_and_episode_audit=audit)
            elif matches:
                # Multiple unfinished continuations are displayed explicitly; none is a final score.
                row = max(matches, key=lambda r: r["last_step"])
                record.update(run=row["id"], status="stopped_censored" if row["diverged"] else "in_progress",
                              matching_runs=[r["id"] for r in matches])
            else:
                row = None
            if matches and len(completed) <= 1:
                ds = records(root / "runs" / row["id"] / "diagnostics.jsonl")
                last = ds[-1] if ds else {}
                record.update(last_evaluation_step=row["eval_step"], last_observed_success=row["success_rate"],
                    last_training_step=row["last_step"], declared_divergence=row["diverged"],
                    max_q_rms_over_return_bound=row["max_q_rms_over_bound"],
                    last_q_mean=last.get("q_mean"), last_v_mean=last.get("v_mean"),
                    last_q_rms=row["last_q_rms"],
                    last_q_upper_violation_rms=last.get("q_min_above_return_upper_rms"),
                    last_v_upper_violation_rms=last.get("v_above_return_upper_rms"))
            output.append(record)
    return output


def complete_macro(rows, environments):
    """Only complete, audited environment sets have a macro average."""
    seeds = sorted({r["seed"] for r in rows})
    means = {}
    for seed in seeds:
        final = {r["environment"]: r["final_success"] for r in rows
                 if r["seed"] == seed and r["status"] == "verified_final"}
        if set(final) == set(environments):
            means[str(seed)] = float(np.mean([final[e] for e in environments]))
    complete = len(means) == len(seeds)
    return {"expected_seed_count": len(seeds), "complete_seed_count": len(means),
            "per_seed_complete_environment_mean": means,
            "all_seeds_complete": complete,
            "mean_over_all_planned_seeds": float(np.mean(list(means.values()))) if complete else None,
            "training_seed_standard_deviation": float(np.std(list(means.values()), ddof=1))
                if complete and len(means) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    # No training GPU context is needed for inspecting NumPy checkpoints.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    root = args.root
    plan = json.loads((root / "replication_plan.json").read_text())
    collected, _ = collect(root)
    families = ["candidate", "original_baseline", "stress_baseline"]
    if "numerical_implementation_control_replication" in plan:
        families.append("numerical_control")
    rows = [row for family in families
            for row in family_rows(root, collected, plan, family)]
    macros = {family: complete_macro([r for r in rows if r["family"] == family], ENVIRONMENTS)
              for family in ("candidate", "original_baseline")}
    candidate = macros["candidate"]["mean_over_all_planned_seeds"]
    result = {"time": time.time(), "kind": "fixed_plan_model_and_episode_audit",
        "plan_sha256": hashlib.sha256((root / "replication_plan.json").read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows, "nine_environment_macros": macros,
        "candidate_3seed_mean_exceeds_criterion": candidate > .19 if candidate is not None else None,
        "unobserved_final_scores_are_never_zero_filled": True,
        "stability_caveat": "Finite checkpoint state is necessary but does not establish stable or accurate values. Inspect physical-bound violations, trajectories, failures, and paired stress baselines.",
        "compute_caveat": "Same 1M Q/actor updates, but candidate uses 2M V updates versus baseline 1M; this is not equal total optimization compute.",
        "selection_caveat": "Seed0 selected the method; seeds1/2 are fixed follow-up replications. Episode uncertainty is distinct from training-seed variation."}
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    target = output / "selected_benchmark_audit.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    os.replace(temporary, target)
    fields = ["family", "environment", "seed", "run", "status", "final_success",
              "last_training_step", "last_evaluation_step", "last_observed_success",
              "declared_divergence", "max_q_rms_over_return_bound", "last_q_mean",
              "last_v_mean", "last_q_rms", "last_q_upper_violation_rms", "last_v_upper_violation_rms"]
    with (output / "selected_benchmark_table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"verified_finals": {family: sum(r["status"] == "verified_final" for r in rows
            if r["family"] == family) for family in families},
        "nine_environment_macros": macros,
        "audit_failures": [r["run"] for r in rows if r["status"] in
                           ("audit_failed", "ambiguous_multiple_finals")]}, indent=2))


if __name__ == "__main__":
    main()
