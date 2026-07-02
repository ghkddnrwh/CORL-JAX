# export_wandb_run_for_gpt.py

import json
import math
from pathlib import Path
from datetime import datetime

import wandb


def normalize_run_path(path: str) -> str:
    """
    W&B UI URL 스타일:
        /ukjo19/sb3/runs/o5v89to4
    W&B API 스타일:
        ukjo19/sb3/o5v89to4
    로 변환.
    """
    path = path.strip().strip("/")
    parts = path.split("/")

    if len(parts) == 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"

    return path


def json_safe(x):
    """NaN, numpy scalar, datetime 등을 GPT가 읽기 쉬운 JSON 값으로 변환."""
    if x is None:
        return None

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x

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
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_wandb_run_for_gpt(
    run_path: str,
    out_dir: str = "wandb_export_for_gpt",
    page_size: int = 1000,
    # max_rows: int | None = None,
):
    print("[1] Creating wandb.Api...", flush=True)
    api = wandb.Api(timeout=60)

    api_path = normalize_run_path(run_path)
    print(f"[2] Loading run: {api_path}", flush=True)
    run = api.run(api_path)
    print("[3] Run loaded", flush=True)

    out = Path(out_dir) / api_path.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)

    print("[4] Saving config...", flush=True)
    config = {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }
    dump_json(out / "config.json", config)

    print("[5] Saving summary...", flush=True)
    try:
        summary = run.summary._json_dict
    except Exception:
        summary = dict(run.summary)
    dump_json(out / "summary.json", summary)

    print("[6] Saving metadata...", flush=True)
    metadata = {
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
    dump_json(out / "metadata.json", metadata)

    print("[7] Start scanning history...", flush=True)

    history_path = out / "history.jsonl"
    n_rows = 0
    all_keys = set()
    first_rows = []
    last_rows = []

    with history_path.open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=page_size):
            row = json_safe(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_rows += 1
            all_keys.update(row.keys())

            if len(first_rows) < 5:
                first_rows.append(row)

            last_rows.append(row)
            if len(last_rows) > 5:
                last_rows.pop(0)

            if n_rows % 1000 == 0:
                print(f"[7] scanned rows: {n_rows}", flush=True)

            # if max_rows is not None and n_rows >= max_rows:
            #     print(f"[7] max_rows={max_rows} reached. Stop early.", flush=True)
            #     break

    print(f"[8] Finished scanning history. rows={n_rows}", flush=True)

    # README 생성부는 기존 코드 그대로

    # 5. 사람이 보기 쉬운 Markdown 요약 저장
    md = []
    md.append("# W&B Run Export\n")
    md.append(f"- API path: `{api_path}`")
    md.append(f"- Run name: `{getattr(run, 'name', None)}`")
    md.append(f"- Run id: `{getattr(run, 'id', None)}`")
    md.append(f"- State: `{getattr(run, 'state', None)}`")
    md.append(f"- Created at: `{getattr(run, 'created_at', None)}`")
    md.append(f"- URL: {getattr(run, 'url', None)}")
    md.append("")
    md.append("## Files")
    md.append("- `metadata.json`: run 기본 정보")
    md.append("- `config.json`: 하이퍼파라미터")
    md.append("- `summary.json`: 최종 metric / summary")
    md.append("- `history.jsonl`: 전체 step별 로그")
    md.append("")
    md.append("## History")
    md.append(f"- Number of rows: `{n_rows}`")
    md.append(f"- Number of keys: `{len(all_keys)}`")
    md.append("")
    md.append("### Logged keys")
    md.append(", ".join(f"`{k}`" for k in sorted(all_keys)))
    md.append("")
    md.append("## Summary")
    md.append("```json")
    md.append(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Config")
    md.append("```json")
    md.append(json.dumps(json_safe(config), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## First 5 history rows")
    md.append("```json")
    md.append(json.dumps(json_safe(first_rows), ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Last 5 history rows")
    md.append("```json")
    md.append(json.dumps(json_safe(last_rows), ensure_ascii=False, indent=2))
    md.append("```")

    (out / "README_for_GPT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[DONE] Exported to: {out}")
    print(f"  - {out / 'README_for_GPT.md'}")
    print(f"  - {out / 'history.jsonl'}")
    print(f"  - {out / 'config.json'}")
    print(f"  - {out / 'summary.json'}")
    print(f"  - {out / 'metadata.json'}")


if __name__ == "__main__":
    export_wandb_run_for_gpt(
        run_path="/ukjo19/ORL-BIAS/runs/2715d0d6-96b9-4823-908d-f6a677eded71",
        out_dir="logs/wandb_logs/2715d0d6-96b9-4823-908d-f6a677eded71",
        page_size=1000,
        # max_rows=None,
        # max_rows=1000,
    )