"""One-time relocation; preserve file bytes/inodes and legacy path aliases."""
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path('/home/ukjo1/research/ssd1/offline_rl/insampling')
CODE = REPO / 'utils/codex/ukjo1/error_propagation'
LOGS = REPO / 'logs/codex/ukjo1/error_propagation'
DISK = Path('/data/sda2/ukjo1/codex/error_propagation')
AUDIT = CODE / 'migration_20261120'
events = []


def move(old, new):
    old, new = Path(old), Path(new)
    assert old.exists() and not old.is_symlink(), old
    assert not new.exists() and not new.is_symlink(), new
    new.parent.mkdir(parents=True, exist_ok=True)
    assert old.stat().st_dev == new.parent.stat().st_dev, (old, new)
    old.rename(new)
    old.symlink_to(new, target_is_directory=new.is_dir())
    events.append(dict(old=str(old), new=str(new)))


def unique_roots():
    roots = [REPO / p for p in ('utils/flow_iql_research', 'utils/propagation_research',
        'utils/iclr2027_dd_iql', 'algorithms/uk_offline/basic/propagation',
        'logs/diagnosis', 'logs/iclr2027_dd_iql', 'logs/iclr2027')]
    roots += [p.resolve() for p in (REPO/'logs/diagnosis').iterdir() if p.is_symlink()]
    roots += [Path('/data/sda2/ukjo1/codex_research/insampling/flow_iql_research_20261026')]
    return list(dict.fromkeys(p.resolve() for p in roots))


def inventory(roots):
    entries = []
    for base in roots:
        for root, ds, fs in os.walk(base, followlinks=False):
            for name in fs:
                p = Path(root)/name
                if not p.exists() or p.is_symlink():
                    continue
                s = p.stat()
                row = dict(path=str(p), device=s.st_dev, inode=s.st_ino, size=s.st_size,
                           mtime_ns=s.st_mtime_ns)
                if p.suffix in ('.py', '.sh', '.md', '.txt', '.tex') and s.st_size < 20_000_000:
                    row['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                entries.append(row)
    return entries


def main():
    before = inventory(unique_roots())
    (AUDIT/'files_before.json').write_text(json.dumps(before, indent=2)+'\n')
    print('Inventoried', len(before), 'files', flush=True)
    # External volumes stay on their current volume; all are reachable below LOGS.
    external = list(dict.fromkeys(p.resolve() for p in (REPO/'logs/diagnosis').iterdir() if p.is_symlink()))
    for old in external:
        move(old, DISK/'logs/diagnosis'/old.name)
    move('/data/sda2/ukjo1/codex_research/insampling/flow_iql_research_20261026',
         DISK/'logs/checkpoint_objects/flow_iql_research_20261026')
    (LOGS/'checkpoint_objects').symlink_to(DISK/'logs/checkpoint_objects', target_is_directory=True)
    for name in ('flow_iql_research', 'propagation_research', 'iclr2027_dd_iql'):
        move(REPO/'utils'/name, CODE/name)
    move(REPO/'algorithms/uk_offline/basic/propagation', CODE/'references')
    for name in ('iql_residual_control.py', 'iql_propagation_diagnostics.py'):
        move(REPO/'algorithms/uk_offline/diagnostics'/name, CODE/'diagnostics'/name)
        legacy = REPO/'algorithms/uk_offline/basic'/name
        if not legacy.exists() and not legacy.is_symlink():
            legacy.symlink_to(CODE/'diagnostics'/name)
    move(REPO/'codex.txt', CODE/'notes/codex.txt')
    for name in ('diagnosis', 'iclr2027_dd_iql', 'iclr2027'):
        move(REPO/'logs'/name, LOGS/name)
    # Normalize experiment symlinks to their current storage locations.
    for p in (LOGS/'diagnosis').iterdir():
        if p.is_symlink():
            target = p.resolve()
            p.unlink()
            p.symlink_to(target, target_is_directory=True)
    log_roots = [LOGS/'diagnosis'/p.name for p in (LOGS/'diagnosis').iterdir()]
    log_roots += [LOGS/'iclr2027_dd_iql', LOGS/'iclr2027']
    seen = set()
    report_directories = []
    snapshot_directories = []
    for base in log_roots:
        physical = base.resolve()
        if physical in seen:
            continue
        seen.add(physical)
        for root, ds, fs in os.walk(physical, followlinks=False):
            for name in ds[:]:
                path = Path(root)/name
                if path.is_symlink():
                    continue
                logical = base/path.relative_to(physical)
                if name in ('analysis', 'reports', 'figures', 'tables'):
                    report_directories.append((path, logical.relative_to(LOGS)))
                    ds.remove(name)
                elif name in ('source', 'sources'):
                    snapshot_directories.append((path, logical.relative_to(LOGS)))
                    ds.remove(name)
    raw_suffixes = {'.pkl', '.npz', '.npy', '.h5', '.hdf5', '.msgpack', '.pt', '.pth',
                    '.log', '.jsonl', '.tmp', '.lock'}
    for old, rel in report_directories:
        target = CODE/'reports'/rel
        if old.stat().st_dev != CODE.stat().st_dev:
            storage = DISK/'reports'/rel
            move(old, storage)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(storage, target_is_directory=True)
        else:
            storage = target
            move(old, storage)
        # Preserve machine-readable training artifacts under logs, including
        # large checkpoint archives that historically lived inside analysis/.
        for root, ds, fs in os.walk(storage, followlinks=False):
            for name in fs:
                path = Path(root)/name
                if path.is_symlink() or path.suffix not in raw_suffixes:
                    continue
                artifact = old.parent/(old.name+'_artifacts')/path.relative_to(storage)
                move(path, artifact)
    for old, rel in snapshot_directories:
        target = CODE/'source_snapshots'/rel
        if old.stat().st_dev == CODE.stat().st_dev:
            move(old, target)
        else:
            storage = DISK/'source_snapshots'/rel
            move(old, storage)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(storage, target_is_directory=True)
    # Experiment-level prose also belongs alongside reports.
    for base in log_roots:
        for p in list(base.glob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.txt', '.md', '.tex', '.pdf'):
                rel = base.relative_to(LOGS)/'documents'/p.name
                target = CODE/'reports'/rel
                if p.stat().st_dev == CODE.stat().st_dev:
                    move(p, target)
                else:
                    storage = DISK/'reports'/rel
                    move(p, storage)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(storage)
    (AUDIT/'path_moves.json').write_text(json.dumps(events, indent=2)+'\n')
    problems = []
    for row in before:
        path = Path(row['path'])
        try:
            s = path.stat()
            for key, actual in (('device',s.st_dev),('inode',s.st_ino),('size',s.st_size),('mtime_ns',s.st_mtime_ns)):
                if row[key] != actual:
                    problems.append(dict(path=str(path), field=key, before=row[key], after=actual))
            if 'sha256' in row and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                problems.append(dict(path=str(path), field='sha256'))
        except OSError as exc:
            problems.append(dict(path=str(path), error=str(exc)))
    result = dict(time=time.time(), files_checked=len(before), moves=len(events),
                  report_directories=len(report_directories), source_snapshot_directories=len(snapshot_directories),
                  status='PASS' if not problems else 'FAIL', problems=problems)
    (AUDIT/'relocation_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    assert not problems


if __name__ == '__main__':
    main()
