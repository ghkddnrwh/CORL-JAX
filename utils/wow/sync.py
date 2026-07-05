#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# ====== 환경 설정(필요하면 여기만 바꿔도 됨) ======
SRC_USER = "ukjo2"
SRC_HOST = "166.104.35.89"
# SRC_ROOT = "/home/ukjo1/work/fed_rl/moderate-rl/logs/action_robust/sam/exclude_critic/ver2/single"

# DST_ROOT = "/home/ukjo2/work/fed_rl/expectile-rl/logs/action_robust/sam/exclude_critic/ver2/single"
SRC_ROOT = "/home/ukjo2/work/fed_rl/expectile-rl/logs/action_robust/ga_sac_total"


DST_ROOT = "/home/ukjo1/work/fed_rl/moderate-rl/logs/tnnls_revision/meso/ga_sac"

LEAF_DEPTH = 1

# SSH 옵션
# - accept-new 같은 옵션 호환성 문제를 피하고,
# - known_hosts 충돌/프롬프트 때문에 멈추는 일을 줄이기 위해 아래를 기본으로 둠.
# 보안이 신경 쓰이면 아래 옵션을 바꾸고 ssh-keyscan으로 known_hosts를 채워 사용하세요.

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_sync")

SSH_OPTS = [
    "-i", SSH_KEY,                     # ★ 이게 핵심: 이 키를 사용
    "-o", "IdentitiesOnly=yes",         # 다른 키들 시도하지 않게
    "-o", "BatchMode=yes",              # 비밀번호 프롬프트 금지(키 안되면 바로 실패)
    "-o", "PreferredAuthentications=publickey",
    "-o", "PasswordAuthentication=no",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
]

# ==============================================


@dataclass
class ItemStatus:
    rel_dir: str                 # SRC_ROOT 기준 상대 경로 (param1/param2/algo/env_trial)
    status: str                  # "missing" | "already" | "conflict"
    src_fp: Optional[str] = None
    dst_fp: Optional[str] = None


def run_cmd(cmd: List[str], *, capture_stdout: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """
    stderr는 그대로 터미널에 흘려보내서 SSH/rsync 에러를 바로 볼 수 있게 함.
    """
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=None,      # 중요: stderr는 화면에 출력
        check=check
    )


def ssh(remote_cmd: str) -> bytes:
    """
    원격(88)에서 remote_cmd 실행하고 stdout(bytes) 반환.
    """
    cmd = ["ssh", *SSH_OPTS, f"{SRC_USER}@{SRC_HOST}", remote_cmd]
    cp = run_cmd(cmd, capture_stdout=True, check=True)
    return cp.stdout or b""


# def list_remote_leaf_dirs() -> List[str]:
#     """
#     88번 서버의 SRC_ROOT 아래에서 깊이 4 디렉터리 목록을 찾아
#     SRC_ROOT 기준 상대경로 리스트로 반환.
#     """
#     # find 결과를 '\0'로 구분해서 안전하게 전달
#     remote_py = r"""
# import os, subprocess, sys
# root = os.path.normpath(r'''{root}''')
# cmd = ['find', root, '-mindepth', 'depth_variable', '-maxdepth', 'depth_variable', '-type', 'd', '-print0']
# out = subprocess.check_output(cmd)
# paths = [p for p in out.split(b'\0') if p]
# rels = []
# for p in paths:
#     s = p.decode('utf-8', errors='replace')
#     rels.append(os.path.relpath(s, root))
# sys.stdout.write('\0'.join(sorted(rels)))
# """.format(root=SRC_ROOT)

#     remote_cmd = "python3 - <<'PY'\n" + remote_py + "\nPY\n"
#     out = ssh(remote_cmd).decode("utf-8", errors="replace")
#     if not out:
#         return []
#     rels = [r for r in out.split("\0") if r and r != "."]
#     rels.sort()
#     return rels

def list_remote_leaf_dirs() -> List[str]:
    remote_py = r"""
import os, subprocess, sys
root = os.path.normpath(r'''{root}''')
depth = int({depth})
cmd = ['find', root, '-mindepth', str(depth), '-maxdepth', str(depth), '-type', 'd', '-print0']
out = subprocess.check_output(cmd)
paths = [p for p in out.split(b'\0') if p]
rels = []
for p in paths:
    s = p.decode('utf-8', errors='replace')
    rels.append(os.path.relpath(s, root))
sys.stdout.write('\0'.join(sorted(rels)))
""".format(root=SRC_ROOT, depth=LEAF_DEPTH)

    remote_cmd = "python3 - <<'PY'\n" + remote_py + "\nPY\n"
    out = ssh(remote_cmd).decode("utf-8", errors="replace")
    if not out:
        return []
    rels = [r for r in out.split("\0") if r and r != "."]
    rels.sort()
    return rels

# def list_local_leaf_dirs() -> List[str]:
#     """
#     89번 서버의 DST_ROOT 아래에서 깊이 4 디렉터리 목록을 찾아
#     DST_ROOT 기준 상대경로 리스트로 반환.
#     """
#     root = Path(DST_ROOT)
#     if not root.exists():
#         return []
#     rels: List[str] = []
#     # os.walk로 전체를 돈 뒤 depth=4만 골라도 되지만,
#     # find가 더 빠르고 간단하므로 로컬도 find 사용
#     cmd = ["find", str(root), "-mindepth", "depth_variable", "-maxdepth", "depth_variable", "-type", "d", "-print0"]
#     cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, check=True)
#     out = cp.stdout or b""
#     for p in out.split(b"\0"):
#         if not p:
#             continue
#         s = p.decode("utf-8", errors="replace")
#         rels.append(os.path.relpath(s, str(root)))
#     rels.sort()
#     return rels


def list_local_leaf_dirs() -> List[str]:
    root = Path(DST_ROOT)
    if not root.exists():
        return []
    rels: List[str] = []

    depth = str(LEAF_DEPTH)
    cmd = ["find", str(root), "-mindepth", depth, "-maxdepth", depth, "-type", "d", "-print0"]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, check=True)

    out = cp.stdout or b""
    for p in out.split(b"\0"):
        if not p:
            continue
        s = p.decode("utf-8", errors="replace")
        rels.append(os.path.relpath(s, str(root)))
    rels.sort()
    return rels

def local_dir_fingerprint(dir_path: Path) -> str:
    """
    빠른 동일성 검사용 fingerprint:
    파일 상대경로 + size + mtime 목록을 sha256으로 요약.
    """
    h = hashlib.sha256()
    root = dir_path

    for cur, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        for fn in files:
            fp = Path(cur) / fn
            try:
                st = fp.stat()
            except FileNotFoundError:
                continue
            rel = fp.relative_to(root).as_posix()
            line = f"{rel}\t{st.st_size}\t{int(st.st_mtime)}\n"
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def remote_dir_fingerprint(remote_abs_dir: str) -> str:
    remote_py = f"""
import os, hashlib, sys
root = os.path.normpath({remote_abs_dir!r})
h = hashlib.sha256()

for cur, dirs, files in os.walk(root):
    dirs.sort(); files.sort()
    for fn in files:
        fp = os.path.join(cur, fn)
        try:
            st = os.stat(fp)
        except FileNotFoundError:
            continue

        rel = os.path.relpath(fp, root).replace('\\\\', '/')
        line = rel + "\\t" + str(st.st_size) + "\\t" + str(int(st.st_mtime)) + "\\n"
        h.update(line.encode("utf-8"))

sys.stdout.write(h.hexdigest())
"""
    remote_cmd = "python3 - <<'PY'\n" + remote_py + "\nPY\n"
    out = ssh(remote_cmd).decode("utf-8", errors="replace").strip()
    return out


def rsync_copy(rel_dir: str, dry_run: bool = False) -> None:
    """
    rel_dir(leaf 디렉터리)를 88 -> 89로 복사.
    """
    src = f"{SRC_USER}@{SRC_HOST}:{os.path.join(SRC_ROOT, rel_dir)}/"
    dst = os.path.join(DST_ROOT, rel_dir)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",
        "-e", "ssh " + " ".join(SSH_OPTS),
    ]
    if dry_run:
        cmd.append("--dry-run")

    cmd.extend([src, str(dst_path) + "/"])

    print(f"\n[RSYNC] {rel_dir}")
    run_cmd(cmd, capture_stdout=False, check=True)


def save_report(report: Dict[str, Any]) -> Path:
    ts = report["generated_at"]
    p = Path(f"logs/sync_report/sync_report_{ts}.json")
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return p


def main():
    ap = argparse.ArgumentParser(
        description="Verify duplicates and sync leaf dirs (depth=4) from 88 -> 89 safely."
    )
    ap.add_argument("--verify-only", action="store_true", help="검증만 하고 복사는 하지 않음")
    ap.add_argument("--dry-run", action="store_true", help="rsync --dry-run 사용(파일 변경 없음)")
    ap.add_argument(
        "--conflict",
        choices=["skip", "rename", "overwrite"],
        default="skip",
        help="conflict(경로 같고 내용 다름) 처리: skip|rename|overwrite (기본 skip)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="빠른 모드: leaf 경로 중복 여부만 체크하고 fingerprint 비교는 하지 않음(충돌 판단 X)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="테스트용: 앞에서 N개 leaf만 처리 (0이면 전체)",
    )
    args = ap.parse_args()

    dst_root = Path(DST_ROOT)
    if not dst_root.exists():
        print(f"[ERROR] DST_ROOT does not exist: {DST_ROOT}", file=sys.stderr)
        sys.exit(2)

    # 0) ssh 사전 체크(실패하면 여기서 바로 에러)
    try:
        _ = ssh("echo SSH_OK")
    except subprocess.CalledProcessError:
        print("\n[ERROR] SSH connection to 88 failed.", file=sys.stderr)
        print("Try manually:", file=sys.stderr)
        print(f"  ssh {SRC_USER}@{SRC_HOST} \"echo OK\"", file=sys.stderr)
        sys.exit(3)

    print("[1/3] Listing remote leaf directories (88)...")
    remote_leaf = list_remote_leaf_dirs()
    if args.limit and args.limit > 0:
        remote_leaf = remote_leaf[: args.limit]
    print(f"  Found {len(remote_leaf)} leaf dirs on 88.")

    print("[1b/3] Listing local leaf directories (89)...")
    local_leaf = set(list_local_leaf_dirs())
    print(f"  Found {len(local_leaf)} leaf dirs on 89.")

    statuses: List[ItemStatus] = []
    to_copy: List[str] = []
    already: List[str] = []
    conflicts: List[str] = []

    print("[2/3] Verifying...")
    for i, rel in enumerate(remote_leaf, 1):
        dst_path = dst_root / rel

        if rel not in local_leaf:
            statuses.append(ItemStatus(rel, "missing"))
            to_copy.append(rel)
            continue

        # 존재하면 fast 모드에서는 conflict 판단 없이 "already"로만 기록(복사 안 함)
        if args.fast:
            statuses.append(ItemStatus(rel, "already"))
            already.append(rel)
            continue

        # fingerprint 비교
        src_abs = os.path.join(SRC_ROOT, rel)
        try:
            src_fp = remote_dir_fingerprint(src_abs)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] remote fingerprint failed for {rel} (treat as conflict): {e}", file=sys.stderr)
            statuses.append(ItemStatus(rel, "conflict"))
            conflicts.append(rel)
            continue

        dst_fp = local_dir_fingerprint(dst_path)

        if src_fp == dst_fp:
            statuses.append(ItemStatus(rel, "already", src_fp=src_fp, dst_fp=dst_fp))
            already.append(rel)
        else:
            statuses.append(ItemStatus(rel, "conflict", src_fp=src_fp, dst_fp=dst_fp))
            conflicts.append(rel)

        if i % 50 == 0:
            print(f"  ...checked {i}/{len(remote_leaf)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n=== Summary ===")
    print(f"missing : {len(to_copy)}")
    print(f"already : {len(already)}")
    print(f"conflict: {len(conflicts)}")
    if args.fast:
        print("  (fast mode: conflict 판단을 하지 않았습니다)")

    report = {
        "src": {"host": SRC_HOST, "user": SRC_USER, "root": SRC_ROOT},
        "dst": {"root": DST_ROOT},
        "summary": {"missing": len(to_copy), "already": len(already), "conflict": len(conflicts)},
        "items": [s.__dict__ for s in statuses],
        "conflict_policy": args.conflict,
        "dry_run": bool(args.dry_run),
        "verify_only": bool(args.verify_only),
        "fast": bool(args.fast),
        "generated_at": ts,
    }
    report_path = save_report(report)
    print(f"\nReport saved: {report_path}")

    if args.verify_only:
        print("\nVerification only. Done.")
        return

    print("[3/3] Copying with rsync...")

    # conflict 처리 정책 적용
    if conflicts and args.conflict != "skip":
        for rel in conflicts:
            dst_path = Path(DST_ROOT) / rel
            if not dst_path.exists():
                continue

            if args.conflict == "rename":
                new_path = dst_path.with_name(dst_path.name + f".conflict_{ts}")
                print(f"[CONFLICT rename] {dst_path} -> {new_path}")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst_path), str(new_path))

            elif args.conflict == "overwrite":
                print(f"[CONFLICT overwrite] deleting {dst_path}")
                shutil.rmtree(dst_path)

        # conflict 처리 후 복사 대상으로 포함
        if args.conflict in ("rename", "overwrite"):
            to_copy = sorted(set(to_copy + conflicts))

    # missing(및 rename/overwrite된 conflict)만 복사
    for rel in to_copy:
        try:
            rsync_copy(rel, dry_run=args.dry_run)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] rsync failed for {rel}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
