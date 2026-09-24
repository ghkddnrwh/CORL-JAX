# export_wandb_runs_2026_09_22_orl_160_hardcoded_old_style.py
#
# 2026-09-20 ~ 2026-09-22에 제공된 10개 W&B CSV의
# run ID를 직접 하드코딩한 exporter.
#
# 실행 시 CSV 파일은 필요하지 않습니다.
#
# Projects:
#   ORL-BIAS   : DD / DDIQL 계열 80 runs
#   ORL-SMOOTH : IQL 계열 80 runs
#
# Groups:
#   am_l_dd, am_l_iql
#   cs_dd, cs_iql
#   hm_dd, hm_iql
#   p4_dd, p4_iql
#   scene_dd, scene_iql
#
# Total CSV rows: 160
# Unique run IDs: 160
# State counts: finished 160
#
# Usage:
#   pip install wandb
#   wandb login
#   python export_wandb_runs_2026_09_22_orl_160_hardcoded_old_style.py

import json
import math
from datetime import datetime
from pathlib import Path

import wandb


# CSV의 User 컬럼에는 entity 정보가 비어 있어서
# 기존 사용 계정 기준으로 둡니다.
# 다른 W&B entity/team이면 이 한 줄만 수정하세요.
ENTITY = "ukjo19"

OUT_ROOT = "logs/wandb_logs_2026_09_22_orl_160"
PAGE_SIZE = 500

# True면 API 조회 시점에 finished인 run만 export
# False면 하드코딩된 모든 run의 현재까지 로그를 export
ONLY_FINISHED = False


SOURCE_FILE_METADATA = {'AM_L_DD_wandb_export_2026-09-21T13_36_04.075+09_00.csv': {'group': 'am_l_dd',
                                                            'project': 'ORL-BIAS',
                                                            'run_name': 'DDIQL-JAX-antmaze-large-navigate-singletask-v0',
                                                            'env': 'antmaze-large-navigate-singletask-v0',
                                                            'rows': 16,
                                                            'run_ids': 16,
                                                            'unique_run_ids': 16,
                                                            'seeds': [0, 1, 2, 3],
                                                            'state_counts': {'finished': 16}},
 'AM_L_IQL_wandb_export_2026-09-21T13_36_47.454+09_00.csv': {'group': 'am_l_iql',
                                                             'project': 'ORL-SMOOTH',
                                                             'run_name': 'IQL-JAX-antmaze-large-navigate-singletask-v0',
                                                             'env': 'antmaze-large-navigate-singletask-v0',
                                                             'rows': 16,
                                                             'run_ids': 16,
                                                             'unique_run_ids': 16,
                                                             'seeds': [0, 1, 2, 3],
                                                             'state_counts': {'finished': 16}},
 'CS_wandb_export_2026-09-20T22_15_35.869+09_00.csv': {'group': 'cs_dd',
                                                       'project': 'ORL-BIAS',
                                                       'run_name': 'DDIQL-JAX-cube-single-play-singletask-v0',
                                                       'env': 'cube-single-play-singletask-v0',
                                                       'rows': 16,
                                                       'run_ids': 16,
                                                       'unique_run_ids': 16,
                                                       'seeds': [0, 1, 2, 3],
                                                       'state_counts': {'finished': 16}},
 'CS_wandb_export_2026-09-20T22_15_49.425+09_00.csv': {'group': 'cs_iql',
                                                       'project': 'ORL-SMOOTH',
                                                       'run_name': 'IQL-JAX-cube-single-play-singletask-v0',
                                                       'env': 'cube-single-play-singletask-v0',
                                                       'rows': 16,
                                                       'run_ids': 16,
                                                       'unique_run_ids': 16,
                                                       'seeds': [0, 1, 2, 3],
                                                       'state_counts': {'finished': 16}},
 'HM_DD_wandb_export_2026-09-22T20_51_16.691+09_00.csv': {'group': 'hm_dd',
                                                          'project': 'ORL-BIAS',
                                                          'run_name': 'DDIQL-JAX-humanoidmaze-medium-navigate-singletask-v0',
                                                          'env': 'humanoidmaze-medium-navigate-singletask-v0',
                                                          'rows': 16,
                                                          'run_ids': 16,
                                                          'unique_run_ids': 16,
                                                          'seeds': [0, 1, 2, 3],
                                                          'state_counts': {'finished': 16}},
 'HM_IQL_wandb_export_2026-09-22T20_50_40.315+09_00.csv': {'group': 'hm_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-humanoidmaze-medium-navigate-singletask-v0',
                                                           'env': 'humanoidmaze-medium-navigate-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'P4_DD_wandb_export_2026-09-22T17_22_12.687+09_00.csv': {'group': 'p4_dd',
                                                          'project': 'ORL-BIAS',
                                                          'run_name': 'DDIQL-JAX-puzzle-4x4-play-singletask-v0',
                                                          'env': 'puzzle-4x4-play-singletask-v0',
                                                          'rows': 16,
                                                          'run_ids': 16,
                                                          'unique_run_ids': 16,
                                                          'seeds': [0, 1, 2, 3],
                                                          'state_counts': {'finished': 16}},
 'P4_IQL_wandb_export_2026-09-22T17_21_52.857+09_00.csv': {'group': 'p4_iql',
                                                           'project': 'ORL-SMOOTH',
                                                           'run_name': 'IQL-JAX-puzzle-4x4-play-singletask-v0',
                                                           'env': 'puzzle-4x4-play-singletask-v0',
                                                           'rows': 16,
                                                           'run_ids': 16,
                                                           'unique_run_ids': 16,
                                                           'seeds': [0, 1, 2, 3],
                                                           'state_counts': {'finished': 16}},
 'SCENE_DD_wandb_export_2026-09-22T17_22_57.030+09_00.csv': {'group': 'scene_dd',
                                                             'project': 'ORL-BIAS',
                                                             'run_name': 'DDIQL-JAX-scene-play-singletask-v0',
                                                             'env': 'scene-play-singletask-v0',
                                                             'rows': 16,
                                                             'run_ids': 16,
                                                             'unique_run_ids': 16,
                                                             'seeds': [0, 1, 2, 3],
                                                             'state_counts': {'finished': 16}},
 'SCENE_IQL_wandb_export_2026-09-22T17_23_14.734+09_00.csv': {'group': 'scene_iql',
                                                              'project': 'ORL-SMOOTH',
                                                              'run_name': 'IQL-JAX-scene-play-singletask-v0',
                                                              'env': 'scene-play-singletask-v0',
                                                              'rows': 16,
                                                              'run_ids': 16,
                                                              'unique_run_ids': 16,
                                                              'seeds': [0, 1, 2, 3],
                                                              'state_counts': {'finished': 16}}}


PROJECT_BY_GROUP = {'am_l_dd': 'ORL-BIAS',
 'am_l_iql': 'ORL-SMOOTH',
 'cs_dd': 'ORL-BIAS',
 'cs_iql': 'ORL-SMOOTH',
 'hm_dd': 'ORL-BIAS',
 'hm_iql': 'ORL-SMOOTH',
 'p4_dd': 'ORL-BIAS',
 'p4_iql': 'ORL-SMOOTH',
 'scene_dd': 'ORL-BIAS',
 'scene_iql': 'ORL-SMOOTH'}


RUN_IDS_BY_GROUP = {'am_l_dd': ['407b080cfcaf4ca7922b262e73fc222e',
             '1b462b7ef7d14d1ca84f420050f51b40',
             '55f76b904ea0430fa9f03feb56e6cc48',
             'e406bea137e745a19919a17c28f75d4c',
             '313176c790654c6881ad447f0052fd3d',
             '46640baa4a554f9c90d5a7d9a5eac8ba',
             'a76e0688516c4312863841d3bb2e4393',
             '9c7a80aa381c4d6594486746a1c36016',
             'eec55dcb0651468a9ce24fba36aba266',
             'f8d47bc01c384b43b6dec6fe1000834b',
             '067393ec8d174380ad39990092c9ae25',
             '8b5d944fa8f54c4798abee700538cf0b',
             'd971943cd23145eeb87fd2930a597f54',
             '2e4e57b6757b4ec3bb1142af84cb240c',
             'e2a0404a68f441148372d8ff2df631e7',
             '8d2d1538a55645b5a1aea9fd9d7a393b'],
 'am_l_iql': ['79a1d4cf497c48d9a5217c99ee3398f8',
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
 'cs_dd': ['10c2e17911a64eb9898b76a8c373fca3',
           'a6c3563d7e034f57855dc433ac0ac0a6',
           '41f6a528ac05454b988d1caf302512da',
           'e810bbb1cb294ab89066fdc9519907dc',
           '838f1b6bf22341fd82c59b3e9b9cf37c',
           '04c2709ddd4243109d9307faf6659ab7',
           '1b17050c6ae34e0e846b5dec6147cfca',
           'a11b7961ba7c430fb01319e3c428698d',
           '46252958d0664eef9829865f763193cd',
           'ff991a2d13ac43edadd52bb65610ffa9',
           '6641ec05139d4f9486a3ccb3ffda5a90',
           '0bc720a68ef14d0bb45760c4bc69084e',
           '946ae012a48844808c054446152322ef',
           '0ee89a3c6a8a42e0ad3d7716708ade82',
           '76c30aa9419e4652aac9a051508d770f',
           'c71abbe5fb1146e1abff9628d6de92d5'],
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
 'hm_dd': ['7184adc085c744fbb5b3f959854db2f5',
           'a48447344a59420da72ba998d430af19',
           'c2fdf0b717924271b540b0c6f0b0a91a',
           'a52532678e824d4db8e2fe3acd8a8d3c',
           'f4b21acf33b045e28c23195d0dbc7039',
           'b057bfef24f7414b93a0245768d0664f',
           'b77ddcb176794c6aaf62606622babf6d',
           '12f9d2ca129c42c59c65ecc56a880144',
           '120a74253d8249f982d1e080bba9e16b',
           'bb3dcb170ecc4ff084b0c762d819c27e',
           'fa7de0fa9990490da82403764501945b',
           'eaae1b0db1554826bb3b148824458c15',
           'bd4663c6e5ce4f9ba6f6dacd23dbc21a',
           'afb4ca6aefb54002a49754e51f6df84b',
           'cd66f4c7c5544204a741ca4a06d1020f',
           'd852ede724c64ebe80578fd1c88636f5'],
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
 'p4_dd': ['d002037943bc4368905c6f6781a66b31',
           '6b53766d37c846569b74c7892b792189',
           '26d9e75a50ba4c8cbfc7730a1b4db053',
           '224dd9baadec47b5a9e3342200a0e451',
           'e8d1165c00804fab9e4ff3aace2c2d66',
           '4ee397d788114d8e87e11f59e5957c7b',
           'ac5b8a7a5d4a475ba4782c0083246804',
           '2f7ee5015faa4d889fa109bea797bda8',
           '42080ba932d24bc3a7b37352e014ac78',
           '0f9b85e906d146e998e3809f9e2244a2',
           'db6d3825932f4dbe9e58ac3b14a00da9',
           '81e6c49dc37641d5a87503a1c2938480',
           '041c1e37d4c747aea7b0d6ed3eb2ff60',
           '6ca3e0430d3d4b7cb1e56223780b4e5e',
           '5c224f80553b410780e7a38f293a91a2',
           '5373072e0ee6423aa5aeea5f4070af9d'],
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
 'scene_dd': ['b04a6c44b5c74ad88fe13bf516f365b3',
              '74cac2aba2bb436099f79aca4f97f274',
              'e2895fa3dcde4e6fa5373dce6633799d',
              '3f142f175ccb4393bfe6cf8335eb80a8',
              'a017b95752b74cafa6d8d8763864a585',
              '9eaa3ae631424e31a8060019bd0579f5',
              'c8d8173cafe34d229b8a70ce619efdc6',
              'a199ae648faa4079b1880fe9d3b61a04',
              '4bb497c4a3284847a2fc62680a9a1108',
              '4b6f9afda7084c9ea71a8c3daa367e2a',
              '691a8033c4944e5db0e9bbd03809db47',
              '6ebf9bb4ca554a1c969e71297c2558b9',
              '4c1e4221cb5549df845e15cc066d0cde',
              'a518469a0bb641739ad118c81395dd14',
              'fae0d967311f47128bb6063574f87e2e',
              '823cc3e235ac49b0940c08cd2f3c1d61'],
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
            "unique_run_count": len(RUN_IDS),
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
