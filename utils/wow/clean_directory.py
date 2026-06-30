#!/usr/bin/env python3
from pathlib import Path
import argparse


TARGET_NAME = "checkpoint.pkl"


def format_size(num_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def find_checkpoint_files(root: Path):
    """Find all checkpoint.pkl files recursively under root."""
    checkpoint_files = []

    for path in root.rglob(TARGET_NAME):
        if path.is_file():
            checkpoint_files.append(path)

    return checkpoint_files


def main():
    parser = argparse.ArgumentParser(
        description="Find and optionally delete all checkpoint.pkl files under the current directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="logs",
        help="Root directory to search. Default: current directory.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete found checkpoint.pkl files. Without this option, only dry-run is performed.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    files = find_checkpoint_files(root)

    total_size = 0
    file_infos = []

    for file_path in files:
        try:
            size = file_path.stat().st_size
            total_size += size
            file_infos.append((file_path, size))
        except OSError as e:
            print(f"[WARN] Could not access: {file_path} ({e})")

    print("=" * 80)
    print(f"Search root        : {root}")
    print(f"Target filename    : {TARGET_NAME}")
    print(f"Matched files      : {len(file_infos)}")
    print(f"Total size         : {format_size(total_size)}")
    print("=" * 80)

    for file_path, size in sorted(file_infos, key=lambda x: x[1], reverse=True):
        print(f"{format_size(size):>12}  {file_path}")

    print("=" * 80)

    if not args.delete:
        print("[DRY RUN] No files were deleted.")
        print("To delete them, run again with:")
        print()
        print(f"python {Path(__file__).name} --root {root} --delete")
        return

    print("[DELETE MODE] Deleting files...")

    deleted_count = 0
    deleted_size = 0

    for file_path, size in file_infos:
        try:
            file_path.unlink()
            deleted_count += 1
            deleted_size += size
            print(f"[DELETED] {file_path}")
        except OSError as e:
            print(f"[FAILED] {file_path} ({e})")

    print("=" * 80)
    print(f"Deleted files      : {deleted_count}")
    print(f"Deleted total size : {format_size(deleted_size)}")
    print("=" * 80)


if __name__ == "__main__":
    main()