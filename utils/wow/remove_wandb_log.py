import os
import time
import wandb

ENTITY = "ukjo19"
PROJECT = "ORL-BIAS"
TARGET = "checkpoint.pkl"

DRY_RUN = False

api = wandb.Api()

matched_count = 0
deleted_count = 0
failed_runs = []
failed_files = []


def safe_get_files(run):
    try:
        files = []

        # 현재 wandb 버전에서는 pattern 인자를 쓰면 안 됨
        for f in run.files(per_page=100):
            files.append(f)

        return files

    except Exception as e:
        print(f"[RUN FILE LIST FAILED] run={run.id} name={run.name}")
        print(f"  error: {repr(e)}")
        failed_runs.append((run.id, run.name, repr(e)))
        return []


runs = api.runs(f"{ENTITY}/{PROJECT}", per_page=100)

for run_idx, run in enumerate(runs, start=1):
    print(f"\n[RUN {run_idx}] id={run.id} name={run.name}")

    files = safe_get_files(run)

    for f in files:
        if os.path.basename(f.name) != TARGET:
            continue

        size = getattr(f, "size", None)
        print(f"[FOUND] run={run.id} file={f.name} size={size}")
        matched_count += 1

        if not DRY_RUN:
            try:
                f.delete()
                deleted_count += 1
                print(f"[DELETED] {f.name}")
                time.sleep(0.2)
            except Exception as e:
                print(f"[DELETE FAILED] run={run.id} file={f.name}")
                print(f"  error: {repr(e)}")
                failed_files.append((run.id, f.name, repr(e)))

print("\n==============================")
print(f"matched files: {matched_count}")
print(f"deleted files: {deleted_count}")
print(f"failed runs: {len(failed_runs)}")
print(f"failed files: {len(failed_files)}")

if failed_runs:
    print("\n===== Failed runs =====")
    for run_id, run_name, err in failed_runs:
        print(f"run={run_id} name={run_name} error={err}")

if failed_files:
    print("\n===== Failed files =====")
    for run_id, file_name, err in failed_files:
        print(f"run={run_id} file={file_name} error={err}")