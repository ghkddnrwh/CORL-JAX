"""Two-seed non-default results: checkpoint means within seed, then mean/SD across seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import sha


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_summary(selected, expected):
    available = [r for r in selected if r["last3_success_pct"] is not None]
    full = len(available) == expected
    values = [r["last3_success_pct"] for r in available]
    return dict(complete_seeds=len(available), expected_seeds=expected,
                last3_mean_pct=st.mean(values) if full else None,
                sample_sd_pp=st.stdev(values) if full and len(values) > 1 else None,
                seed0_pct=next((r["last3_success_pct"] for r in available if r["seed"] == 0), None),
                seed1_pct=next((r["last3_success_pct"] for r in available if r["seed"] == 1), None))


def aggregate(rows, plan):
    """Tasks are averaged inside a seed before estimating family-level seed SD."""
    tasks, family_seeds, families = [], [], []
    for environment in plan["environments"]:
        job = next(j for j in plan["jobs"] if j["environment"] == environment)
        for mode in plan["deployment_modes"]:
            selected = [r for r in rows if r["environment"] == environment and r["deployment"] == mode]
            assert len(selected) == len(plan["training_seeds"])
            tasks.append(dict(family=job["family"], task=job["task_id"], environment=environment,
                              deployment=mode, **paired_summary(selected, len(plan["training_seeds"]))))
    for code, family, default in plan["families"]:
        for mode in plan["deployment_modes"]:
            selected_seeds = []
            for seed in plan["training_seeds"]:
                selected = [r for r in rows if r["family"] == family and r["deployment"] == mode and r["seed"] == seed]
                assert len(selected) == 4
                ready = [r["last3_success_pct"] for r in selected if r["last3_success_pct"] is not None]
                row = dict(family=family, deployment=mode, seed=seed, complete_tasks=len(ready), expected_tasks=4,
                           last3_success_pct=st.mean(ready) if len(ready) == 4 else None)
                family_seeds.append(row)
                selected_seeds.append(row)
            families.append(dict(code=code, family=family, deployment=mode, excluded_default_task=default,
                                 **paired_summary(selected_seeds, len(plan["training_seeds"])) ))
    return tasks, family_seeds, families


def collect(root, plan, final=False):
    rows, problems, trained, evaluated = [], [], 0, 0
    training_by_seed = {seed: 0 for seed in plan["training_seeds"]}
    evaluation_by_seed = dict(training_by_seed)
    evidence = {}
    for job in plan["jobs"]:
        run = Path(job["output"])
        conf = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else None
        if conf:
            ident, cfg = conf["identity"], conf["identity"]["config"]
            assert ident["environment"] == job["environment"] and ident["seed"] == job["seed"]
            assert ident["recipe"] == job["recipe"]
            assert cfg["discount"] == plan["discount"] and cfg["batch_size"] == 256 and cfg["tau"] == .005
            assert cfg["backup_samples"] == cfg["num_samples"] == 9 and cfg["target_clipping"]
            assert cfg["backup_Q_aggregation"] == job["head_aggregation"]
        complete = (run / "completed.json").exists()
        if complete:
            done = json.loads((run / "completed.json").read_text())
            assert done["step"] == done["q_updates"] == done["actor_updates"] == plan["steps"]
            assert done["v_updates"] == 0
            trained += 1
            training_by_seed[job["seed"]] += 1
        if (run / "stopped.json").exists():
            problems.append(dict(run=job["id"], reason="nonfinite_state"))
        for mode in plan["deployment_modes"]:
            scores = {}
            for step in plan["evaluation_steps"]:
                path = root / "evaluations" / job["id"] / f"{step}_{mode}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                assert conf and result["config_sha256"] == sha(run / "config.json")
                assert result["environment"] == job["environment"] and result["training_seed"] == job["seed"]
                assert result["discount"] == plan["discount"] and result["task"]["task_id"] == job["task_id"]
                assert result["step"] == step and result["policy_mode"] == mode
                assert result["first_episode_seed"] == 42 and result["n_episodes"] == len(result["episode_successes"]) == 50
                assert result["protocol"] == "last3_cpu50_independent_seeded_episodes_v1" and result["no_optimizer_updates"]
                assert abs(result["success_rate"] - st.mean(result["episode_successes"])) < 1e-12
                if final:
                    for artifact, digest in ((run / f"checkpoint_{step}.pkl", result["checkpoint_sha256"]),
                                             (run / f"actor_{step}.pkl", result["actor_artifact_sha256"])):
                        if str(artifact) not in evidence:
                            assert sha(artifact) == digest, artifact
                            evidence[str(artifact)] = digest
                        assert evidence[str(artifact)] == digest
                scores[step] = 100 * result["success_rate"]
                evaluated += 1
                evaluation_by_seed[job["seed"]] += 1
            rows.append(dict(family=job["family"], task=job["task_id"], environment=job["environment"],
                             gamma=plan["discount"], seed=job["seed"], deployment=mode, training_complete=complete,
                             evaluation_points=len(scores), success_800K=scores.get(800000),
                             success_900K=scores.get(900000), success_1M=scores.get(1000000),
                             last3_success_pct=st.mean(scores.values()) if len(scores) == 3 else None))
    expected = len(plan["jobs"]) * len(plan["evaluation_steps"]) * len(plan["deployment_modes"])
    status = dict(time=time.time(), complete=trained == len(plan["jobs"]) and evaluated == expected and not problems,
                  training_complete=trained, total_cells=len(plan["jobs"]), evaluations_complete=evaluated,
                  expected_evaluations=expected, failures=problems, discount=plan["discount"],
                  training_seeds=plan["training_seeds"], training_by_seed=training_by_seed,
                  evaluations_by_seed=evaluation_by_seed,
                  uncertainty="Two training seeds; sample SD is descriptive. Family SD is computed across seed-level four-task means.")
    if final:
        assert status["complete"]
        for name, digest in plan["source_sha256"].items():
            assert sha(name) == digest, name
    return rows, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "registration_v2.json").read_text())
    assert plan["training_seeds"] == [0, 1] and plan["total_cells"] == 80
    rows, status = collect(root, plan, args.final)
    tasks, family_seeds, families = aggregate(rows, plan)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    suffix = "final_v2" if args.final else "progress_v2"
    for prefix, values in (("per_seed_task", rows), ("per_task", tasks), ("per_seed_family", family_seeds), ("family_summary", families)):
        write_csv(out / f"{prefix}_{suffix}.csv", values)
    temporary = out / f"summary_{suffix}.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(out / f"summary_{suffix}.json")
    lines = [f"OGBench non-default 40 task / N=9 / gamma={plan['discount']} / seed0/1", "",
             f"완료 학습 {status['training_complete']}/80, 완료 평가 {status['evaluations_complete']}/480.",
             "각 시드의 800K/900K/1M CPU50 성공률을 평균한 뒤 두 학습 시드의 평균±sample SD(%).",
             "두 시드이므로 SD는 기술 통계이며, 두 시드가 모두 완료된 경우에만 평균을 표시한다.", "",
             "환경 | task | Lean 평균±SD | 단일 actor 평균±SD"]
    for environment in plan["environments"]:
        job = next(j for j in plan["jobs"] if j["environment"] == environment)
        formatted = []
        for mode in plan["deployment_modes"]:
            row = next(r for r in tasks if r["environment"] == environment and r["deployment"] == mode)
            formatted.append("대기" if row["last3_mean_pct"] is None else f"{row['last3_mean_pct']:.2f}±{row['sample_sd_pp']:.2f}")
        lines.append(f"{job['code']} | {job['task_id']} | " + " | ".join(formatted))
    lines += ["", "family 평균은 각 시드에서 non-default 네 task를 먼저 평균하고, 그 두 값의 평균/SD를 집계한다."]
    (out / f"RESULTS_KO_{suffix}.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
