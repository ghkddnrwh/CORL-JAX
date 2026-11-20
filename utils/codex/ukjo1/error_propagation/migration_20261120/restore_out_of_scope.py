"""Undo the over-broad migration without reverting unrelated user changes."""
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path('/home/ukjo1/research/ssd1/offline_rl/insampling')
CODE = REPO/'utils/codex/ukjo1/error_propagation'
LOGS = REPO/'logs/codex/ukjo1/error_propagation'
AUDIT = CODE/'migration_20261120'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(event):
    old, new = Path(event['old']), Path(event['new'])
    if 'iclr2027_dd_iql' in old.parts or 'iclr2027_dd_iql' in new.parts:
        return True
    return old in (
        REPO/'logs/iclr2027', REPO/'algorithms/uk_offline/basic/propagation',
        REPO/'algorithms/uk_offline/diagnostics/iql_propagation_diagnostics.py',
        REPO/'codex.txt',
    )


def main():
    marker = AUDIT/'scope_correction.json'
    assert not marker.exists(), 'Already restored; inspect audit instead of repeating.'
    events = [e for e in json.loads((AUDIT/'path_moves.json').read_text()) if selected(e)]
    fixes = [e for e in json.loads((AUDIT/'repository_path_fixes.json').read_text())
             if 'iclr2027_dd_iql' in Path(e['path']).parts]
    # Preflight everything before touching any paths.
    for event in events:
        old, new = Path(event['old']), Path(event['new'])
        assert old.is_symlink() and old.resolve() == new.resolve(), event
        assert new.exists() and not new.is_symlink(), event
    for fix in fixes:
        path = Path(fix['path'])
        assert sha(path) == fix['relocated_sha256'], ('Concurrent edit; preserve it', path)
        backup = AUDIT/'original_path_sensitive_sources'/path.relative_to(CODE)
        assert sha(backup) == fix['original_sha256']
    for fix in fixes:
        path = Path(fix['path'])
        backup = AUDIT/'original_path_sensitive_sources'/path.relative_to(CODE)
        path.write_bytes(backup.read_bytes())
        assert sha(path) == fix['original_sha256']
    restored = []
    for event in reversed(events):
        old, new = Path(event['old']), Path(event['new'])
        assert old.is_symlink() and old.resolve() == new.resolve(), event
        old.unlink()
        new.rename(old)
        restored.append(event)
    # The pre-existing diagnostic module was moved by another project change.
    # Keep its historical basic-module alias working at the original location.
    legacy = REPO/'algorithms/uk_offline/basic/iql_propagation_diagnostics.py'
    assert legacy.is_symlink()
    legacy.unlink()
    legacy.symlink_to(REPO/'algorithms/uk_offline/diagnostics/iql_propagation_diagnostics.py')
    # logs/diagnosis contains both our experiments and an older shared baseline.
    # Turn the old aggregate link into a directory of per-study compatibility
    # links, and restore the shared baseline directory there.
    old_parent = REPO/'logs/diagnosis'
    assert old_parent.is_symlink() and old_parent.resolve() == (LOGS/'diagnosis').resolve()
    temporary = REPO/'logs/.diagnosis_scope_correction'
    temporary.mkdir()
    for entry in (LOGS/'diagnosis').iterdir():
        (temporary/entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    old_parent.unlink()
    temporary.rename(old_parent)
    shared = LOGS/'diagnosis/basic'
    restored_shared = old_parent/'basic'
    assert restored_shared.is_symlink() and restored_shared.resolve() == shared.resolve()
    restored_shared.unlink()
    shared.rename(restored_shared)
    # Prune only now-empty skeleton directories from the mistaken separation.
    for base in (CODE/'reports/iclr2027_dd_iql', CODE/'source_snapshots/iclr2027_dd_iql',
                 CODE/'notes'):
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base, topdown=False, followlinks=False):
            path = Path(root)
            if not any(path.iterdir()):
                path.rmdir()
    # Raw compile-log separation created an empty tables_artifacts skeleton.
    for path in (REPO/'logs/iclr2027_dd_iql').rglob('*_artifacts'):
        if not path.is_dir() or path.is_symlink():
            continue
        for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
            item = Path(root)
            if not any(item.iterdir()):
                item.rmdir()
    result = dict(time=time.time(), reason='User clarified: only this Error propagation research belongs in the new namespace.',
                  restored_moves=restored, reverted_path_only_edits=fixes,
                  shared_baseline_restored=str(restored_shared),
                  learner_processes_stopped=False)
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(restored_moves=len(restored), reverted_path_only_edits=len(fixes),
                          shared_baseline_restored=True)), flush=True)


if __name__ == '__main__':
    main()
