"""Losslessly relocate immutable completed-run checkpoints, retaining file paths.

The current/latest inode and every active training output remain on the fast
filesystem. Each other inode is copied and SHA256-verified before any original
hardlink is atomically replaced with a symlink. No training data is removed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def active_outputs():
    outputs = set()
    for command in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            parts = command.read_bytes().decode().split('\0')
            if '-m' in parts and parts[parts.index('-m') + 1].startswith('utils.flow_iql_research.train') and '--output' in parts:
                outputs.add(Path(parts[parts.index('--output') + 1]).resolve())
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return outputs


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def inventory(root):
    active = active_outputs()
    groups = {}
    for run in sorted((root / 'runs').iterdir()):
        completed = run / 'completed.json'
        if run.resolve() in active or not completed.is_file():
            continue
        if json.loads(completed.read_text())['step'] != 1000000:
            continue
        latest = run / 'latest.pkl'
        if not latest.is_file() or latest.is_symlink():
            continue
        protected = identity(latest)[:2]
        candidates = list(run.glob('*.pkl')) + list((root / 'analysis/checkpoint_archive' / run.name).glob('*.pkl'))
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            signature = identity(path)
            if signature[:2] == protected:
                continue
            group = groups.setdefault(signature[:2], dict(signature=signature, paths=[], run=str(run)))
            assert tuple(group['signature']) == signature
            group['paths'].append(str(path))
    return sorted(groups.values(), key=lambda group: (group['signature'][2], group['paths'][0]), reverse=True)


def relocate(group, destination, journal):
    paths = [Path(path) for path in group['paths']]
    signature = tuple(group['signature'])
    assert Path(group['run']).resolve() not in active_outputs()
    assert all(not path.is_symlink() and identity(path) == signature for path in paths)
    temporary = destination / f'.copy-{os.getpid()}-{signature[1]}.tmp'
    hasher = hashlib.sha256()
    try:
        with paths[0].open('rb') as source, temporary.open('xb') as target:
            for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                hasher.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        assert identity(paths[0]) == signature, 'Source changed during copy'
        sha = hasher.hexdigest()
        assert temporary.stat().st_size == signature[2]
        assert digest(temporary) == sha, 'Destination verification failed'
        target = destination / (sha + '.pkl')
        if target.exists():
            assert target.stat().st_size == signature[2] and digest(target) == sha
            temporary.unlink()
        else:
            temporary.replace(target)
        receipt = dict(**group, sha256=sha, destination=str(target), verified_at=time.time(),
                       path_count=len(paths), size_bytes=signature[2], status='verified_copy_before_relink')
        receipt_path = journal / f'{signature[0]}-{signature[1]}.json'
        atomic(receipt_path, receipt)
        assert Path(group['run']).resolve() not in active_outputs()
        for path in paths:
            assert not path.is_symlink() and identity(path) == signature
            link = path.with_name(path.name + f'.offload-{os.getpid()}')
            link.symlink_to(target)
            os.replace(link, path)
            assert path.is_file() and path.stat().st_size == signature[2]
        assert digest(paths[0]) == sha
        receipt.update(status='verified_relocated', relinked_at=time.time())
        atomic(receipt_path, receipt)
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--target-free-gib', type=float, default=50)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'analysis/checkpoint_offload_v1'
    output.mkdir(exist_ok=True)
    lock = (output / 'offload.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = inventory(root)
    manifest_path = output / 'manifest.json'
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(relocated=[])
    record = dict(pid=os.getpid(), started_at=time.time(), destination=str(args.destination.resolve()),
                  target_free_gib=args.target_free_gib, apply=args.apply,
                  inventory_groups=len(groups), inventory_bytes=sum(g['signature'][2] for g in groups),
                  free_bytes_before=shutil.disk_usage(root).free, relocated=prior['relocated'])
    atomic(output / ('manifest.json' if args.apply else 'preview.json'), record)
    if not args.apply:
        print(json.dumps({k:v for k,v in record.items() if k != 'relocated'}), flush=True)
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    assert args.destination.stat().st_dev != root.stat().st_dev, 'Offload must use another filesystem'
    journal = output / 'transactions'
    journal.mkdir(exist_ok=True)
    for group in groups:
        if shutil.disk_usage(root).free >= args.target_free_gib * 1024**3:
            break
        moved = relocate(group, args.destination.resolve(), journal)
        record['relocated'].append(moved)
        record.update(updated_at=time.time(), free_bytes_now=shutil.disk_usage(root).free)
        atomic(manifest_path, record)
        print(json.dumps(dict(run=Path(group['run']).name, bytes=moved['size_bytes'],
                              links=moved['path_count'], free_gib=record['free_bytes_now']/1024**3)), flush=True)
    record.update(completed_at=time.time(), free_bytes_now=shutil.disk_usage(root).free,
                  status='target_reached' if shutil.disk_usage(root).free >= args.target_free_gib * 1024**3 else 'inventory_exhausted')
    atomic(manifest_path, record)


if __name__ == '__main__':
    main()
