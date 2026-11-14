#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synchronize directories with selected exact names from a remote host.

Performance:
- Discover ALL requested depths with ONE find on each host (not one find/depth).
- Filter folder basenames within find to reduce SSH output.
- Compare remote fingerprints in batches (instead of 1 SSH call/leaf).

Optional --delete-source removes selected remote leaf directories only after
every copy and a checksum-based rsync dry-run verification have succeeded.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ====== 환경 설정(필요하면 여기만 바꿔도 됨) ======
SRC_USER = "ukjo3"
SRC_HOST = "166.104.9.137"
# SRC_ROOT = "/home/ukjo1/work/fed_rl/moderate-rl/logs/action_robust/sam/exclude_critic/ver2/single"

# DST_ROOT = "/home/ukjo2/work/fed_rl/expectile-rl/logs/action_robust/sam/exclude_critic/ver2/single"
SRC_ROOT = "/home/ukjo3/work/offline_rl/CORL-JAX/logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long"


DST_ROOT = "/home/ukjo1/research/ssd1/offline_rl/insampling/logs/iclr2027/basic/iql_jax/ogbench/soft_target_ablation/long"

# SRC_ROOT 상대 경로 기준 깊이. 예) SRC_ROOT/Ant-v4_1 이면 depth=1.
DEFAULT_LEAF_DEPTHS = [5]

# 실제 폴더 이름이 아래 값(예: "0", "1", "2", "3")과 정확히 일치해야 함.
DEFAULT_TARGET_FOLDERS = [
    "0", "1", "2", "3",
]

# 한 번의 SSH 호출에서 fingerprint를 계산할 최대 폴더 수.
FINGERPRINT_BATCH_SIZE = 200

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_sync")
SSH_OPTS = [
    "-i", SSH_KEY,
    "-o", "IdentitiesOnly=yes",
    "-o", "BatchMode=yes",
    "-o", "PreferredAuthentications=publickey",
    "-o", "PasswordAuthentication=no",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-p", "2203",
]
# ==============================================


@dataclass
class ItemStatus:
    rel_dir: str
    status: str  # missing | already | conflict
    src_fp: Optional[str] = None
    dst_fp: Optional[str] = None


def run_cmd(cmd: List[str], *, capture_stdout: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=None,
        check=True,
    )


def ssh(remote_cmd: str) -> bytes:
    cmd = ["ssh", *SSH_OPTS, f"{SRC_USER}@{SRC_HOST}", remote_cmd]
    return run_cmd(cmd).stdout or b""


def normalize_depths(depths: Sequence[int]) -> List[int]:
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("leaf depths must contain one or more positive integers")
    return sorted(set(depths))


def make_target_regex(target_folders: Sequence[str]) -> re.Pattern:
    if not target_folders or any(not name or "/" in name for name in target_folders):
        raise ValueError("target folders must be nonempty directory basenames")
    names = "|".join(re.escape(name) for name in target_folders)
    return re.compile(rf"(?:{names})\Z")


def find_command(root: str, depths: Sequence[int], target_folders: Sequence[str]) -> List[str]:
    """Use one bounded traversal, with a name predicate evaluated by find."""
    cmd = [
        "find", root,
        "-mindepth", str(min(depths)),
        "-maxdepth", str(max(depths)),
        "-type", "d",
        "(",
    ]
    for i, name in enumerate(target_folders):
        if i:
            cmd.append("-o")
        # Match only the directory basename itself (e.g. "0", not "0_1").
        cmd.extend(["-name", name])
    cmd.extend([")", "-print0"])
    return cmd


def parse_find_results(
    output: bytes, root: str, depths: Sequence[int], target_folders: Sequence[str]
) -> List[str]:
    valid_depths = set(depths)
    pattern = make_target_regex(target_folders)
    results = set()
    for raw in output.split(b"\0"):
        if not raw:
            continue
        absolute = os.fsdecode(raw)
        rel = os.path.relpath(absolute, root)
        if (
            rel != "."
            and len(Path(rel).parts) in valid_depths
            and pattern.fullmatch(os.path.basename(absolute))
        ):
            results.add(rel)
    return sorted(results)


def list_remote_leaf_dirs(depths: Sequence[int], target_folders: Sequence[str]) -> List[str]:
    # The entire discovery requires exactly one SSH request, regardless of depth count.
    output = ssh(shlex.join(find_command(SRC_ROOT, depths, target_folders)))
    return parse_find_results(output, SRC_ROOT, depths, target_folders)


def list_local_leaf_dirs(depths: Sequence[int], target_folders: Sequence[str]) -> List[str]:
    root = Path(DST_ROOT)
    if not root.exists():
        return []
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    output = run_cmd(find_command(str(root), depths, target_folders)).stdout or b""
    return parse_find_results(output, str(root), depths, target_folders)


def local_dir_fingerprint(directory: Path) -> str:
    """Fast fingerprint based on relative path, size and integer mtime."""
    h = hashlib.sha256()
    for current, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()
        for filename in files:
            filepath = Path(current) / filename
            try:
                stat = filepath.stat()
            except FileNotFoundError:
                continue
            rel = filepath.relative_to(directory).as_posix()
            h.update(f"{rel}\t{stat.st_size}\t{int(stat.st_mtime)}\n".encode("utf-8"))
    return h.hexdigest()


def remote_dir_fingerprints(rel_dirs: Sequence[str]) -> Dict[str, Optional[str]]:
    """Calculate many fingerprints with one remote Python process / SSH call."""
    if not rel_dirs:
        return {}
    remote_py = f'''import hashlib
import json
import os

root = {SRC_ROOT!r}
rel_dirs = {list(rel_dirs)!r}
results = {{}}
for rel_dir in rel_dirs:
    directory = os.path.join(root, rel_dir)
    try:
        if not os.path.isdir(directory):
            results[rel_dir] = None
            continue
        h = hashlib.sha256()
        for current, dirs, files in os.walk(directory):
            dirs.sort()
            files.sort()
            for filename in files:
                filepath = os.path.join(current, filename)
                try:
                    stat = os.stat(filepath)
                except FileNotFoundError:
                    continue
                rel = os.path.relpath(filepath, directory).replace(os.sep, "/")
                h.update((rel + "\\t" + str(stat.st_size) + "\\t" +
                          str(int(stat.st_mtime)) + "\\n").encode("utf-8"))
        results[rel_dir] = h.hexdigest()
    except OSError:
        results[rel_dir] = None
print(json.dumps(results, ensure_ascii=True))
'''
    remote_cmd = "python3 - <<'PY_SYNC_LEAF'\n" + remote_py + "PY_SYNC_LEAF\n"
    return json.loads(ssh(remote_cmd).decode("utf-8"))


def rsync_copy(rel_dir: str, dry_run: bool = False) -> None:
    src = f"{SRC_USER}@{SRC_HOST}:{os.path.join(SRC_ROOT, rel_dir)}/"
    dst_path = Path(DST_ROOT) / rel_dir

    if dry_run:
        # In dry-run, do not create destination folders or contact rsync again.
        print(f"[DRY-RUN RSYNC] {src} -> {dst_path}/")
        return

    dst_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-a", "--info=progress2", "--partial",
        "-e", "ssh " + shlex.join(SSH_OPTS),
        src, str(dst_path) + "/",
    ]
    print(f"\n[RSYNC] {rel_dir}")
    run_cmd(cmd, capture_stdout=False)


def rsync_verify_identical(rel_dir: str) -> bool:
    """Check actual file contents and directory entries before source deletion.

    --checksum checks file contents even if size and mtime happen to match.
    --delete is used ONLY with --dry-run here to detect extra destination entries.
    """
    src = f"{SRC_USER}@{SRC_HOST}:{os.path.join(SRC_ROOT, rel_dir)}/"
    dst_path = Path(DST_ROOT) / rel_dir
    if not dst_path.is_dir():
        print(f"[VERIFY FAILED] Destination missing: {dst_path}", file=sys.stderr)
        return False

    cmd = [
        "rsync", "-a", "--checksum", "--dry-run", "--itemize-changes",
        "--delete",  # dry-run only: detect destination differences; never delete there
        "-e", "ssh " + shlex.join(SSH_OPTS),
        src, str(dst_path) + "/",
    ]
    output = run_cmd(cmd).stdout or b""
    if output.strip():
        lines = output.decode("utf-8", errors="replace").splitlines()
        print(f"[VERIFY FAILED] {rel_dir}: {len(lines)} differences", file=sys.stderr)
        for line in lines[:5]:
            print(f"  {line}", file=sys.stderr)
        return False
    print(f"[VERIFIED] {rel_dir}")
    return True


def delete_remote_leaf_dir(rel_dir: str) -> None:
    """Remove only one explicitly selected source leaf, never SRC_ROOT itself."""
    remote_py = '''import os
import shutil
import sys

root = os.path.realpath(sys.argv[1])
rel = sys.argv[2]
if (not rel or os.path.isabs(rel) or
        any(part in ("", ".", "..") for part in rel.split("/"))):
    raise ValueError("Unsafe relative source directory")
target = os.path.join(root, rel)
if (os.path.realpath(target) == root or
        os.path.commonpath([root, os.path.realpath(target)]) != root):
    raise ValueError("Source directory escapes SRC_ROOT")
if os.path.islink(target) or not os.path.isdir(target):
    raise ValueError("Source directory missing or is a symlink")
shutil.rmtree(target)
'''
    ssh(shlex.join(["python3", "-c", remote_py, SRC_ROOT, rel_dir]))


def save_report(report: Dict[str, Any]) -> Path:
    timestamp = report["generated_at"]
    path = Path(f"logs/sync_report/sync_report_{timestamp}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync directories with exact target names at one or more depths."
    )
    parser.add_argument(
        "--leaf-depths", nargs="+", type=int, default=DEFAULT_LEAF_DEPTHS,
        metavar="N", help=f"SRC_ROOT-relative directory depths (default: {DEFAULT_LEAF_DEPTHS})",
    )
    parser.add_argument(
        "--target-folders", nargs="+", default=DEFAULT_TARGET_FOLDERS,
        metavar="NAME", help="Exact directory names (default: 0 1 2 3)",
    )
    # 안전한 기본 동작: --apply가 명시되지 않으면 검증만 수행합니다.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--apply", action="store_true",
        help="실제 rsync 복사 및 충돌 처리 수행 (기본값은 검증만)",
    )
    mode_group.add_argument(
        "--verify-only", action="store_true",
        help="복사 없이 검증만 수행 (기본 동작; 기존 옵션 호환)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="--apply와 함께 사용 시 복사/이동/삭제 없이 실행 계획만 출력",
    )
    parser.add_argument(
        "--delete-source", action="store_true",
        help="모든 대상 폴더의 복사 및 체크섬 검증에 성공한 후 원격 원본 폴더 삭제 (SRC_ROOT는 유지)",
    )
    parser.add_argument(
        "--conflict", choices=["skip", "rename", "overwrite"], default="skip",
        help="같은 경로에 내용이 다른 경우 (기본 skip)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="폴더 존재 여부만 확인; fingerprint 비교 생략",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="앞에서 N개 폴더만 처리 (0=전체)",
    )
    args = parser.parse_args()

    if args.delete_source and not args.apply:
        parser.error("--delete-source requires --apply (optionally with --dry-run)")
    if args.delete_source and args.fast:
        parser.error("--delete-source cannot be combined with --fast")

    try:
        depths = normalize_depths(args.leaf_depths)
        target_folders = list(dict.fromkeys(args.target_folders))
        make_target_regex(target_folders)
    except ValueError as exc:
        parser.error(str(exc))

    dst_root = Path(DST_ROOT)
    if dst_root.exists() and not dst_root.is_dir():
        parser.error(f"DST_ROOT is not a directory: {DST_ROOT}")

    try:
        ssh("echo SSH_OK")
    except subprocess.CalledProcessError:
        print(f"[ERROR] SSH connection failed: {SRC_USER}@{SRC_HOST}", file=sys.stderr)
        sys.exit(3)

    print(f"[CONFIG] depths={depths} target_folders={target_folders}")
    print("[1/3] Listing remote target directories (single find)...")
    remote_leaf = list_remote_leaf_dirs(depths, target_folders)
    if args.limit > 0:
        remote_leaf = remote_leaf[:args.limit]
    print(f"  Found {len(remote_leaf)} target dirs on {SRC_HOST}.")

    print("[1b/3] Listing local target directories (single find)...")
    local_leaf = set(list_local_leaf_dirs(depths, target_folders))
    print(f"  Found {len(local_leaf)} target dirs locally.")

    missing_rels = [rel for rel in remote_leaf if rel not in local_leaf]
    common_rels = [rel for rel in remote_leaf if rel in local_leaf]
    statuses: List[ItemStatus] = [ItemStatus(rel, "missing") for rel in missing_rels]
    already: List[str] = []
    conflicts: List[str] = []

    print(f"[2/3] Verifying... missing={len(missing_rels)} common={len(common_rels)}")
    if args.fast:
        for rel in common_rels:
            statuses.append(ItemStatus(rel, "already"))
            already.append(rel)
    else:
        # Remote fingerprints are batched to avoid one SSH handshake per directory.
        for start in range(0, len(common_rels), FINGERPRINT_BATCH_SIZE):
            batch = common_rels[start:start + FINGERPRINT_BATCH_SIZE]
            try:
                fingerprints = remote_dir_fingerprints(batch)
            except (subprocess.CalledProcessError, ValueError) as exc:
                print(f"[WARN] Remote fingerprint batch failed: {exc}", file=sys.stderr)
                fingerprints = {}

            for rel in batch:
                src_fp = fingerprints.get(rel)
                try:
                    dst_fp = local_dir_fingerprint(dst_root / rel)
                except OSError as exc:
                    print(f"[WARN] Local fingerprint failed for {rel}: {exc}", file=sys.stderr)
                    dst_fp = None

                if src_fp is not None and src_fp == dst_fp:
                    statuses.append(ItemStatus(rel, "already", src_fp, dst_fp))
                    already.append(rel)
                else:
                    statuses.append(ItemStatus(rel, "conflict", src_fp, dst_fp))
                    conflicts.append(rel)
            print(f"  ...checked {min(start + len(batch), len(common_rels))}/{len(common_rels)} common")

    statuses.sort(key=lambda status: status.rel_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Summary ===")
    print(f"missing : {len(missing_rels)}")
    print(f"already : {len(already)}")
    print(f"conflict: {len(conflicts)}")
    if args.fast:
        print("  (fast mode: contents not checked)")

    report = {
        "src": {"host": SRC_HOST, "user": SRC_USER, "root": SRC_ROOT},
        "dst": {"root": DST_ROOT},
        "leaf_depths": depths,
        "target_folders": target_folders,
        "summary": {
            "missing": len(missing_rels), "already": len(already), "conflict": len(conflicts)
        },
        "items": [asdict(status) for status in statuses],
        "conflict_policy": args.conflict,
        "dry_run": bool(args.dry_run),
        "verify_only": not args.apply,
        "apply": bool(args.apply),
        "delete_source": bool(args.delete_source),
        "fast": bool(args.fast),
        "generated_at": timestamp,
    }
    print(f"\nReport saved: {save_report(report)}")
    if not args.apply:
        print("Verification only (default). No files copied. Done.")
        return

    if args.dry_run:
        print("[3/3] Dry-run: showing copy/conflict plan only...")
    else:
        print("[3/3] Copying with rsync...")
    to_copy = list(missing_rels)
    if args.conflict in ("rename", "overwrite"):
        for rel in conflicts:
            dst_path = dst_root / rel
            if args.dry_run:
                print(f"[DRY-RUN {args.conflict.upper()}] {dst_path}")
            elif dst_path.exists():
                if args.conflict == "rename":
                    new_path = dst_path.with_name(dst_path.name + f".conflict_{timestamp}")
                    print(f"[CONFLICT rename] {dst_path} -> {new_path}")
                    shutil.move(str(dst_path), str(new_path))
                else:
                    print(f"[CONFLICT overwrite] deleting {dst_path}")
                    shutil.rmtree(dst_path)
        to_copy = sorted(set(to_copy + conflicts))

    copy_failures: List[str] = []
    for rel in to_copy:
        try:
            rsync_copy(rel, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] rsync failed for {rel}: {exc}", file=sys.stderr)
            copy_failures.append(rel)

    report["copy_failures"] = copy_failures
    if args.delete_source:
        if args.dry_run:
            if conflicts and args.conflict == "skip":
                print("[DRY-RUN] Source deletion blocked: unresolved conflicts (policy=skip).")
            else:
                for rel in remote_leaf:
                    print(f"[DRY-RUN DELETE SOURCE, after verification] {SRC_ROOT.rstrip('/')}/{rel}")
        elif copy_failures or (conflicts and args.conflict == "skip"):
            print("[DELETE SOURCE SKIPPED] Copy errors or unresolved conflicts; no source directories removed.")
            report["source_deletion_blocked"] = True
        else:
            # Check ALL selected folders first. If any differs, delete NONE.
            verify_failures: List[str] = []
            print(f"[DELETE SOURCE] Verifying all {len(remote_leaf)} directories with rsync checksums...")
            for rel in remote_leaf:
                try:
                    if not rsync_verify_identical(rel):
                        verify_failures.append(rel)
                except subprocess.CalledProcessError as exc:
                    print(f"[VERIFY ERROR] {rel}: {exc}", file=sys.stderr)
                    verify_failures.append(rel)
            report["source_verify_failures"] = verify_failures

            if verify_failures:
                print("[DELETE SOURCE SKIPPED] Verification failed; no source directories removed.")
                report["source_deletion_blocked"] = True
            else:
                deleted: List[str] = []
                deletion_failures: List[str] = []
                # Descendants before ancestors if multiple depths overlap.
                ordered = sorted(remote_leaf, key=lambda rel: (len(Path(rel).parts), rel), reverse=True)
                for rel in ordered:
                    try:
                        delete_remote_leaf_dir(rel)
                        deleted.append(rel)
                        print(f"[DELETE SOURCE] {SRC_ROOT.rstrip('/')}/{rel}")
                    except subprocess.CalledProcessError as exc:
                        print(f"[DELETE SOURCE ERROR] {rel}: {exc}", file=sys.stderr)
                        deletion_failures.append(rel)
                report["source_deleted"] = deleted
                report["source_deletion_failures"] = deletion_failures

    # Update the initial report with actual operation results.
    save_report(report)
    print("\nDone.")
    if args.delete_source and not args.dry_run and (
        report.get("source_deletion_blocked") or report.get("source_deletion_failures")
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()


