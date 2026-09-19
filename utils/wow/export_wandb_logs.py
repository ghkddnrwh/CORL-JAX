# export_wandb_runs_2026_09_19_orl_hardcoded_old_style.py
#
# 2026-09-19에 제공된 2개 W&B CSV의 run ID를 직접 하드코딩한 exporter.
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# IMPORTANT:
#   이번 데이터는 group마다 W&B project가 다릅니다.
#
#   - DDIQL-JAX-cube-single-play-singletask-v0 -> ORL-BIAS
#   - IQL-JAX-cube-single-play-singletask-v0   -> ORL-SMOOTH
#
# CSV rows: 8
# Unique run IDs: 8
# State counts: finished 8
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_19_orl_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


# CSV에는 entity/team 정보가 없어서 기존 사용 계정 기준으로 둡니다.
# 다른 entity라면 이 한 줄만 수정하면 됩니다.
ENTITY = "ukjo19"

OUT_ROOT = "logs/wandb_logs_2026_09_19_orl"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 하드코딩된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'wandb_export_2026-09-19T17_40_52.407+09_00.csv': {'group': 'ddiql_jax_cube_single_play_singletask_v0',
                                                    'project': 'ORL-BIAS',
                                                    'run_name': 'DDIQL-JAX-cube-single-play-singletask-v0',
                                                    'env': 'cube-single-play-singletask-v0',
                                                    'rows': 4,
                                                    'run_ids': 4,
                                                    'unique_run_ids': 4,
                                                    'seeds': [0, 1, 2, 3],
                                                    'state_counts': {'finished': 4}},
 'wandb_export_2026-09-19T17_41_21.118+09_00.csv': {'group': 'iql_jax_cube_single_play_singletask_v0',
                                                    'project': 'ORL-SMOOTH',
                                                    'run_name': 'IQL-JAX-cube-single-play-singletask-v0',
                                                    'env': 'cube-single-play-singletask-v0',
                                                    'rows': 4,
                                                    'run_ids': 4,
                                                    'unique_run_ids': 4,
                                                    'seeds': [0, 1, 2, 3],
                                                    'state_counts': {'finished': 4}}}


PROJECT_BY_GROUP = {'ddiql_jax_cube_single_play_singletask_v0': 'ORL-BIAS',
 'iql_jax_cube_single_play_singletask_v0': 'ORL-SMOOTH'}


RUN_IDS_BY_GROUP = {'ddiql_jax_cube_single_play_singletask_v0': ['ed216177b9424b1e954cf2850dda0f0e',
                                              '3efce9cf13014a6d989bbe058f126c2c',
                                              '8bc2e26ce36c4359bf9bf699f8b1877a',
                                              '32f6c720efbc461793a1c3c834684649'],
 'iql_jax_cube_single_play_singletask_v0': ['76bebb50f25b4200b135616ad5e68997',
                                            'c159f4781d4a40fb929aaa7a31047d1b',
                                            'dae53d4238b74935b795eb4394a54a7f',
                                            '7857358b74d2453e846b17ef81541f1a']}


RUN_IDS = list(dict.fromkeys(
    run_id
    for ids in RUN_IDS_BY_GROUP.values()
    for run_id in ids
))


def json_safe(x):
    if x is None:
        return None

    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x

    if isinstance(x, (str, int, bool)):
        return x

    if isinstance(x, datetime):
        return x.isoformat()

    try:
        import numpy as np

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            x = float(x)
            return None if math.isnan(x) or math.isinf(x) else x

        if isinstance(x, np.ndarray):
            return x.tolist()

    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    return str(x)


def dump_json(path: Path, obj):
    path.write_text(
        json.dumps(
            json_safe(obj),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def group_for_run(run_id: str):
    for group, ids in RUN_IDS_BY_GROUP.items():
        if run_id in ids:
            return group

    return "unknown"


def project_for_run(run_id: str):
    group = group_for_run(run_id)

    if group not in PROJECT_BY_GROUP:
        raise KeyError(
            f"No project configured for run_id={run_id}, group={group}"
        )

    return PROJECT_BY_GROUP[group]


def export_run(run, out_root: Path, project: str):
    run_id = str(run.id)
    group = group_for_run(run_id)

    # project도 폴더에 명시해서 서로 다른 project 로그가 섞이지 않게 함
    out = (
        out_root
        / project
        / group
        / run_id
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. config
    config = {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }

    dump_json(
        out / "config.json",
        config,
    )

    # 2. summary
    try:
        summary = dict(run.summary)

    except Exception:
        try:
            summary = run.summary._json_dict

        except Exception:
            summary = {}

    dump_json(
        out / "summary.json",
        summary,
    )

    # 3. metadata
    try:
        api_path = "/".join(run.path)

    except Exception:
        api_path = (
            f"{ENTITY}/"
            f"{project}/"
            f"{run_id}"
        )

    metadata = {
        "experiment_group": group,
        "configured_project": project,
        "api_path": api_path,
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "created_at": getattr(run, "created_at", None),
        "url": getattr(run, "url", None),
        "tags": getattr(run, "tags", None),
        "notes": getattr(run, "notes", None),
    }

    dump_json(
        out / "metadata.json",
        metadata,
    )

    # 4. full scalar history
    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    history_path = (
        out
        / "history.jsonl"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in run.scan_history(
            page_size=PAGE_SIZE
        ):
            row = json_safe(row)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            n_rows += 1
            all_keys.update(row.keys())

            if len(first_rows) < 5:
                first_rows.append(row)

            last_rows.append(row)

            if len(last_rows) > 5:
                last_rows.pop(0)

    # 5. README
    readme = [
        "# W&B Run Export",
        "",
        f"- Experiment group: `{group}`",
        f"- Configured project: `{project}`",
        f"- API path: `{api_path}`",
        f"- Run name: `{getattr(run, 'name', None)}`",
        f"- Run id: `{run_id}`",
        f"- State: `{getattr(run, 'state', None)}`",
        f"- History rows: `{n_rows}`",
        "",
        "## Logged keys",
        ", ".join(
            f"`{k}`"
            for k in sorted(all_keys)
        ),
        "",
        "## Summary",
        "```json",
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Config",
        "```json",
        json.dumps(
            json_safe(config),
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## First 5 history rows",
        "```json",
        json.dumps(
            first_rows,
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Last 5 history rows",
        "```json",
        json.dumps(
            last_rows,
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]

    (
        out
        / "README_for_GPT.md"
    ).write_text(
        "\n".join(readme),
        encoding="utf-8",
    )

    return {
        "id": run_id,
        "group": group,
        "configured_project": project,
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "history_rows": n_rows,
        "out_dir": str(out),
    }


def main():
    api = wandb.Api()

    out_root = Path(
        OUT_ROOT
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[INFO] unique runs: {len(RUN_IDS)}"
    )

    for group, ids in RUN_IDS_BY_GROUP.items():
        project = PROJECT_BY_GROUP[group]

        print(
            f"  - {group}: "
            f"{len(ids)} "
            f"(project={project})"
        )

    print()

    exported = []
    skipped = []
    errors = []

    for i, run_id in enumerate(
        RUN_IDS,
        start=1,
    ):
        group = group_for_run(
            run_id
        )

        project = project_for_run(
            run_id
        )

        api_path = (
            f"{ENTITY}/"
            f"{project}/"
            f"{run_id}"
        )

        print(
            f"[{i:03d}/"
            f"{len(RUN_IDS):03d}] "
            f"{project} / "
            f"{group} / "
            f"{run_id}"
        )

        try:
            run = api.run(
                api_path
            )

            state = str(
                getattr(
                    run,
                    "state",
                    "",
                )
            ).lower()

            if (
                ONLY_FINISHED
                and state != "finished"
            ):
                print(
                    f"    SKIP: state={state}"
                )

                skipped.append({
                    "id": run_id,
                    "group": group,
                    "project": project,
                    "state": state,
                })

                continue

            result = export_run(
                run,
                out_root,
                project,
            )

            exported.append(
                result
            )

            print(
                f"    DONE: "
                f"{result['history_rows']} "
                f"history rows"
            )

        except Exception as e:
            print(
                f"    ERROR: {e}"
            )

            errors.append({
                "id": run_id,
                "group": group,
                "project": project,
                "api_path": api_path,
                "error": str(e),
            })

    dump_json(
        out_root
        / "_run_ids_by_group.json",
        RUN_IDS_BY_GROUP,
    )

    dump_json(
        out_root
        / "_project_by_group.json",
        PROJECT_BY_GROUP,
    )

    dump_json(
        out_root
        / "_source_file_metadata.json",
        SOURCE_FILE_METADATA,
    )

    dump_json(
        out_root
        / "_export_summary.json",
        {
            "entity": ENTITY,
            "unique_run_count": len(
                RUN_IDS
            ),
            "only_finished": ONLY_FINISHED,
            "projects": sorted(
                set(
                    PROJECT_BY_GROUP.values()
                )
            ),
            "expected_by_group": {
                group: len(ids)
                for group, ids
                in RUN_IDS_BY_GROUP.items()
            },
            "exported_count": len(exported),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "exported": exported,
            "skipped": skipped,
            "errors": errors,
        },
    )

    print()
    print("=" * 72)

    print(
        f"Expected : "
        f"{len(RUN_IDS)}"
    )

    print(
        f"Exported : "
        f"{len(exported)}"
    )

    print(
        f"Skipped  : "
        f"{len(skipped)}"
    )

    print(
        f"Errors   : "
        f"{len(errors)}"
    )

    print(
        f"Output   : "
        f"{out_root}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
