# export_wandb_runs_2026_09_25_orl_smooth_144_hardcoded_old_style.py
#
# 2026-09-25에 제공된 9개 W&B CSV의 run ID를 직접 하드코딩한 exporter.
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# Project:
#   ORL-SMOOTH
#
# Groups:
#   am_l_iql
#   as_iql
#   cd_iql
#   cs_iql
#   hl_iql
#   hm_iql
#   p3_iql
#   p4_iql
#   scene_iql
#
# Total CSV rows: 144
# Unique run IDs: 144
# State counts: finished 144
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_25_orl_smooth_144_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


ENTITY = "ukjo19"
PROJECT = "ORL-SMOOTH"
OUT_ROOT = "logs/wandb_logs_2026_09_25_orl_smooth_144"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 하드코딩된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'AM_L_IQL_wandb_export_2026-09-25T18_21_17.347+09_00.csv': {'group': 'am_l_iql',
                                                             'project': 'ORL-SMOOTH',
                                                             'run_name': 'IQL-JAX-antmaze-large-navigate-singletask-v0',
                                                             'env': 'antmaze-large-navigate-singletask-v0',
                                                             'rows': 16,
                                                             'run_ids': 16,
                                                             'unique_run_ids': 16,
                                                             'seeds': [0, 1, 2, 3],
                                                             'state_counts': {'finished': 16}},
 'AS_IQL_wandb_export_2026-09-25T21_20_08.789+09_00.csv': {'group': 'as_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-antsoccer-arena-navigate-singletask-v0',
                                                           'env': 'antsoccer-arena-navigate-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'CD_IQL_wandb_export_2026-09-25T18_22_10.867+09_00.csv': {'group': 'cd_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-cube-double-play-singletask-v0',
                                                           'env': 'cube-double-play-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'CS_IQL_wandb_export_2026-09-25T18_21_59.150+09_00.csv': {'group': 'cs_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-cube-single-play-singletask-v0',
                                                           'env': 'cube-single-play-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'HL_IQL_wandb_export_2026-09-25T18_21_01.116+09_00.csv': {'group': 'hl_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-humanoidmaze-large-navigate-singletask-v0',
                                                           'env': 'humanoidmaze-large-navigate-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'HM_IQL_wandb_export_2026-09-25T18_20_31.421+09_00.csv': {'group': 'hm_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-humanoidmaze-medium-navigate-singletask-v0',
                                                           'env': 'humanoidmaze-medium-navigate-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'P3_IQL_wandb_export_2026-09-25T18_23_17.903+09_00.csv': {'group': 'p3_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-puzzle-3x3-play-singletask-v0',
                                                           'env': 'puzzle-3x3-play-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'P4_IQL_wandb_export_2026-09-25T18_23_06.143+09_00.csv': {'group': 'p4_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-puzzle-4x4-play-singletask-v0',
                                                           'env': 'puzzle-4x4-play-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'SCENE_IQLwandb_export_2026-09-25T18_22_38.968+09_00.csv': {'group': 'scene_iql',
                                                             'project': 'ORL-SMOOTH',
                                                             'run_name': 'IQL-JAX-scene-play-singletask-v0',
                                                             'env': 'scene-play-singletask-v0',
                                                             'rows': 16,
                                                             'run_ids': 16,
                                                             'unique_run_ids': 16,
                                                             'seeds': [0, 1, 2, 3],
                                                             'state_counts': {'finished': 16}}}


RUN_IDS_BY_GROUP = {'am_l_iql': ['79a1d4cf497c48d9a5217c99ee3398f8',
              '8c81126e6a134257be687bcbada1e5a3',
              '46f51e52cbcf46549f17029ebd8222f7',
              '0dc2f3ddba27423f80c1f07d96bece3e',
              '8671b9dbc3354143b3ce91ff83ad89f6',
              '25d053fe24104bb19268c14e5f07d820',
              'ae24b1a6c54c43cabacf5e618048b9b9',
              '82134068a06d481782142de91a0e30a6',
              '8846ba20a5204fe189f07dcf731c1c5f',
              '488bdd26ed98495fab9a218cbaa86795',
              '6fbe2704c33d4e028d47b2262f6fa313',
              '28044c17b2134b43a53093913e8067f4',
              '23a8c48e529544439664e20188e785b6',
              'a55e3b9ecc6d4a05b31998cb30161878',
              '0c3bb5287a314fe8ba2d3d5dfe46def0',
              'b48cf69133ee445fa6908af6dbc364f3'],
 'as_iql': ['5430879a8e3f4c1689856236bf31fc18',
            '44ca2166a19742b289bbf659557f8132',
            'e5cf87c20d6e464580afb48f81f8c222',
            'bfe846a686844657ac03308459725100',
            '41f37d472831439ab3a9d7765ce0fe01',
            'c00b9bbe203e4a4ba7af9d5d8326630a',
            'b3233f018c85454daa6f470650cdebb4',
            '7f57d06608ae49778a9d1e364950537d',
            '2a68b365b5fa451a9f9103f60e7ede07',
            '3a3f88c77ad64cdbbbf7fc524fe78c37',
            '6e594525a6e146cd90dcd251cd913e04',
            '103b9b818ff24626b70ed34ff9d7fedf',
            'dc550919d0314aed8055f570997521d6',
            'fc742d23e1614397a6fb38bd7175d054',
            '640a0396e8bf46e2827b43344704b72f',
            'af49d436814247798f7d34544728cc09'],
 'cd_iql': ['fd5b35bf1b3b45b6bf14b8f01603a1df',
            'f2199257a154481faf2aebe3f42e8b63',
            'c67be99858614a67bd4b2d5468ba29a1',
            '5a4b00828a974425acac8594b5478501',
            'a5a637b4f21f471b9116cae219cf4d2e',
            '985031df92c6430aade9110bfb6b4c38',
            '5c1297e9cec44dfb8f1b5a575a545ba7',
            '097e90e3f2eb4219b66b00e863be7ea1',
            '055d44e3b1ec46e0b6e5a00d51995dc7',
            '2b576ae2551246eba3624b4a3fd1ebba',
            'e05ac04c521e4f6497e02ef804b30c11',
            'afdeed2dd1144e96bf047bc362924628',
            '9db09a5d69844e25b9953ef3091879a6',
            '19927dce8de046c88f25b2d391721072',
            'e7c5656c07a541929d345e005812cdb2',
            '661d5298a7ae434696c33bdad97cf183'],
 'cs_iql': ['9ed1a9945cef41beb8b26a5f69d6bcc7',
            '9bd1619edfac445996311f0386fb2b9c',
            '96dff7260e4a4297b6f33a6947991868',
            'a343a33bc8744b52ac59519e20605e02',
            '6f8a492ff3a0479394c26af43628cb4b',
            'dfa1b5cdae4e494c80b3706977bd779a',
            '2e04f371a8ea4290907ec4b1d5a91958',
            '532d921d02ea4d17beb8b535f98764f2',
            '2014398fe85a4307854d6285129853dd',
            'f9ba78f4962048869aff8de2538bab2a',
            '782ecfc010494aae92d73d89d0d51ab9',
            'f92e54bf576f4026a93d93b4ac4ca0fb',
            '88e7c43c9c094602a1697f12a43a1312',
            '4e35842a05a241778da9e5935aedd89f',
            '72257685b04149cb9beb67df9a65dd6c',
            '99541d5b6b9e438e9fac373ccfedbe2c'],
 'hl_iql': ['7fd5130b51e643e58d941d0ce2c77be9',
            'ad953b6ef39b4f97bded3ea8296a2b9c',
            '4203171f2ee347fb95b462131498e5f0',
            '3bb470a0a2d647fdb9b00c1aa0d41b66',
            '79443e05283f42ed8c945034e4435b02',
            'c8f8b0df1bf448dfaf894c7d7d6cb39f',
            'c914a3f6358f42f7a9e82c34be406f5e',
            'c0b900bf62ef44059fced807a22d9263',
            '21fde21dfb2a422eb3faf64167789db3',
            'ae07e3ac50434b6fadf2196db3899517',
            '51471d51c5594732ae204bb458c0e899',
            '22cbc565cd354037a31d62ca7cde3623',
            '90ffc7581b9645efab75dcd4c67b69c8',
            '907398ae72064b798e9d385ecc1d74dd',
            'd819e405d7e24b2ab005d79ec98a4ad7',
            '04ecdc65f44741038a5bfe6ff352dfa5'],
 'hm_iql': ['76eaa0baaae345e797e30e0cd5f5d880',
            '915b9fe6375242009df2f19ed7fe0ab7',
            'cc39fa342ff44f458d0e8d1bc8dd2eba',
            'ed2fbf8b26ec4a038d76b9831dcc889e',
            '55db7c397b2241fabf553f5109534759',
            '4428bf3f55da4a24a516c627edb451e2',
            '223db09bf17647c3aaa58b3c98248ee3',
            'c412281ee999406680173a9ed1c2f0ed',
            '092fa02a360545e98224a7890938867d',
            '41edc16cddaf4e59a509e84db16c9131',
            'a198990ad5b0426d808f00242a2874a6',
            '6c6700b9e7af49daab2b9fa6cdf6a2d3',
            '5f669824f5c14544b0b74b9ae4fea037',
            'd0ffe2705ee7493d8ef8353237876b70',
            'b85b8d7420df44dbb93e0471e659e5ae',
            '05b9daf5b27c4e08aebf1534aab3ab89'],
 'p3_iql': ['b61b898fd321401da994c1f264e96b39',
            '2bb81d48975d420da9e26a27df3842ea',
            '5a563dd8c2aa4839968f2c1ee18b7d3a',
            '592c0dbbb3294a44a62d92c5e2f78bd0',
            'ddc2dd445908498c897ede49571c8405',
            'da35a241486547af8a3731b03bd318d4',
            '59af7580827d4c779db1095e8d1d6dec',
            '96d5220b923244368033e21a38714f22',
            'd2a029cea934406aad5fa32cc51858bc',
            '880b26f2004e422b8a4eac0a64498de6',
            '007a85c60aaf4db484d8a6cd2e4d0934',
            '00ce3173c076449a84822b672e34d476',
            'c938ebc8d1664c8a882b22593a6d1ad6',
            '33c779da50af43689cf3935b890f05ec',
            '3d2aebf6e8d1447796a3440eb35e5b8c',
            'ff80293f46f34ba1adaec1c83b866c7a'],
 'p4_iql': ['d249224ad4254c82b46adb7a3f2ded28',
            'ed327c60332c405e9d4984929ef72d2f',
            '75e438e8923a421faba4ae3358c26e9d',
            '0c682bf6ac474c7ea78475fa80575198',
            '344142a2f1ae4380bdf1fe36639dd466',
            'f8ddd5bd7281434b8b95cce2f3dc2286',
            '0e56cf5d45b04d008663e540c6500f55',
            '7c39726d57764c5a9a31451efe714926',
            'bbe30ed94a1644dcb06e95fbc7ee4812',
            'fe42e9d7080e4d82bbac63c38b284618',
            '30929fddb9c74b788248b08d3c0f00cc',
            'd070e4f474ce4e078ebcae3a6b15ab4b',
            '408b47ccb2174ff5a757a782a7636f24',
            '95ab535d0c274d8c913bf7792c950e46',
            '0e0cd30c7d6446f397d2ca03fdc72432',
            '83818f3b70b0477fbe8ecd158d9792f9'],
 'scene_iql': ['a63c6f8a6c9048f198ef097b9d4923f5',
               'ea0669ffdb81483fa9f1e71e465653eb',
               'e5a71ee5a7be40bd82f5598a07b91ceb',
               '6bac392a62f640beb4a25e0d4912cd31',
               'cac6d9796a1d4635bdb83e1bee680b51',
               '3844f5d4219c46b9a4bb7b5acecb343a',
               '1bbb33f7088947ee86eb9b855d3dd4b9',
               '50fce8a1fb19483e8d90b5846ab519f1',
               '9dec2d72ac3344eab9bb74ec38e7a957',
               '4c2f70d942344b4f8e643e0d96d000ea',
               'e7ce473d928f4380a88e30b18c82c913',
               '9a0b495de4104f6487c764993e81b0b4',
               'bd25a60ce8164c7d9d81d8145a83ae5b',
               '4284177c5f6546fa9b0ed57251c4e4d2',
               '348f49a4828549818663b6a2bc95981f',
               '4e5c751488374827a3c39e48493ad5be']}


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
        return {
            str(k): json_safe(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            json_safe(v)
            for v in x
        ]

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


def export_run(run, out_root: Path):
    run_id = str(run.id)
    group = group_for_run(run_id)

    out = (
        out_root
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
            f"{PROJECT}/"
            f"{run_id}"
        )

    metadata = {
        "experiment_group": group,
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
        f"- Project: `{PROJECT}`",
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
        print(
            f"  - {group}: "
            f"{len(ids)}"
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

        api_path = (
            f"{ENTITY}/"
            f"{PROJECT}/"
            f"{run_id}"
        )

        print(
            f"[{i:03d}/"
            f"{len(RUN_IDS):03d}] "
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
                    "state": state,
                })

                continue

            result = export_run(
                run,
                out_root,
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
        / "_source_file_metadata.json",
        SOURCE_FILE_METADATA,
    )

    dump_json(
        out_root
        / "_export_summary.json",
        {
            "entity": ENTITY,
            "project": PROJECT,
            "unique_run_count": len(
                RUN_IDS
            ),
            "only_finished": ONLY_FINISHED,
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
