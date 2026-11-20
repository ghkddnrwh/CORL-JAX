"""One-time, audited import/path migration; never change numerical run payloads."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time

REPO = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
CODE = REPO / 'utils/codex/ukjo1/error_propagation'
AUDIT = CODE / 'migration_20261120/remove_code_aliases'
ROOT = (REPO / 'logs/codex/ukjo1/error_propagation/diagnosis/finite_mean_stability_4task_20261120_v1').resolve()
BACKUP = ROOT / 'namespace_migration_backup_20261120'
REPLACEMENTS = tuple((f'utils{sep}{name}',
                      f'utils{sep}codex{sep}ukjo1{sep}error_propagation{sep}{name}')
                     for sep in ('.', '/') for name in ('flow_iql_research', 'propagation_research'))
OLD_ROOT = 'Path(__file__).resolve().parents[2]'
NEW_ROOT = "next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())"
changes = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.namespace.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare():
    assert not (AUDIT / 'code_changes.json').exists()
    records = []
    for folder in ('flow_iql_research', 'propagation_research', 'diagnostics'):
        for path in sorted((CODE / folder).rglob('*')):
            if path.is_symlink() or not path.is_file() or path.suffix not in ('.py', '.txt', '.md', '.sh'):
                continue
            old = path.read_text()
            new = old
            for source, target in REPLACEMENTS:
                new = new.replace(source, target)
            root_fix = OLD_ROOT in new
            if root_fix:
                new = new.replace(OLD_ROOT, NEW_ROOT)
            if old == new:
                continue
            if path.suffix == '.py':
                compile(new, str(path), 'exec')
                normalized = new.replace(NEW_ROOT, OLD_ROOT) if root_fix else new
                expected = old
                for source, target in REPLACEMENTS:
                    expected = expected.replace(source, target)
                assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(normalized)), path
            relative = path.relative_to(CODE)
            original = AUDIT / 'original_sources' / relative
            staged = AUDIT / 'staged_sources' / relative
            original.parent.mkdir(parents=True, exist_ok=True)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original)
            staged.write_text(new)
            records.append(dict(path=str(path), relative=str(relative), before=sha(path),
                                after=sha(staged), root_discovery_fixed=root_fix,
                                changes='namespace/path substitutions only'))
    atomic_json(AUDIT / 'code_changes.json', records)
    print(json.dumps(dict(staged_files=len(records), root_fixes=sum(r['root_discovery_fixed'] for r in records))), flush=True)


def backup(path):
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    os.link(path, target)
    return sha(path)


def numerical_digest(value):
    fields = {'source_sha256', 'checkpoint_sha256', 'config_sha256'}
    return hashlib.sha256(pickle.dumps({k: v for k, v in value.items() if k not in fields},
                                     protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def write_model(path, value, numerical, before):
    assert numerical_digest(value) == numerical
    temporary = path.with_suffix('.namespace.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with temporary.open('rb') as stream:
        assert numerical_digest(pickle.load(stream)) == numerical
    temporary.replace(path)
    changes.append(dict(path=str(path), before=before, after=sha(path),
                        numerical_payload_sha256=numerical, numerically_identical=True))


def apply():
    assert (AUDIT / 'stopped_processes.json').exists(), 'Stop owned writers before applying'
    assert not BACKUP.exists(), 'This migration is not repeatable'
    records = json.loads((AUDIT / 'code_changes.json').read_text())
    for r in records:
        assert sha(r['path']) == r['before'], r['path']
    plan_path = ROOT / 'registration_v3.json'
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in plan['reused_evidence_sha256'].items():
        assert sha(path) == digest, path
    for r in records:
        path = Path(r['path'])
        staged = AUDIT / 'staged_sources' / r['relative']
        path.write_bytes(staged.read_bytes())
        assert sha(path) == r['after']
    # Each runtime pin still validates the exact file bytes after relocation.
    old_plan_sources = dict(plan['source_sha256'])
    plan['source_sha256'] = {path: sha(path) for path in old_plan_sources}
    backup(plan_path)
    plan['namespace_migration'] = dict(time=time.time(), audit=str(AUDIT),
        scope='Canonical imports; both old code-directory symlinks removed',
        algorithm_and_experiment_settings_unchanged=True)
    atomic_json(plan_path, plan)
    resumes = []
    for job in plan['jobs']:
        run = Path(job['output']).resolve()
        config_path = run / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        old_sources = dict(config['source_sha256'])
        assert all(old_plan_sources[k] == v for k, v in old_sources.items())
        sources = {k: sha(k) for k in old_sources}
        before = backup(config_path)
        config['source_sha256'] = sources
        atomic_json(config_path, config)
        changes.append(dict(path=str(config_path), before=before, after=sha(config_path)))
        models = sorted(run.glob('checkpoint_*.pkl'))
        if (run / 'latest.pkl').exists():
            models.append(run / 'latest.pkl')
        for path in models:
            before = backup(path)
            with path.open('rb') as stream:
                value = pickle.load(stream)
            assert value['identity'] == config['identity'] and value['source_sha256'] == old_sources
            numerical = numerical_digest(value)
            value['source_sha256'] = sources
            write_model(path, value, numerical, before)
            if path.name == 'latest.pkl':
                resumes.append(dict(id=job['id'], step=value['step'], numerical_payload_sha256=numerical))
        for path in sorted(run.glob('actor_*.pkl')):
            before = backup(path)
            with path.open('rb') as stream:
                value = pickle.load(stream)
            assert value['identity'] == config['identity'] and value['source_sha256'] == old_sources
            numerical = numerical_digest(value)
            value['source_sha256'] = sources
            value['config_sha256'] = sha(config_path)
            value['checkpoint_sha256'] = sha(run / f"checkpoint_{value['step']}.pkl")
            write_model(path, value, numerical, before)
        for path in sorted(Path(job['evaluation_directory']).glob('*.json')):
            value = json.loads(path.read_text())
            if 'checkpoint_sha256' not in value:
                continue
            before = backup(path)
            unchanged = copy.deepcopy(value)
            value['config_sha256'] = sha(config_path)
            value['checkpoint_sha256'] = sha(run / f"checkpoint_{value['step']}.pkl")
            value['actor_artifact_sha256'] = sha(run / f"actor_{value['step']}.pkl")
            for key in value.keys() - {'config_sha256', 'checkpoint_sha256', 'actor_artifact_sha256'}:
                assert unchanged[key] == value[key], key
            atomic_json(path, value)
            changes.append(dict(path=str(path), before=before, after=sha(path), evaluation_values_unchanged=True))
    # Keep original upstream hashes; record the actual new import namespace hashes.
    path = CODE / 'flow_iql_research/vendor/provenance.json'
    shutil.copy2(path, AUDIT / 'vendor_provenance_before.json')
    provenance = json.loads(path.read_text())
    for relative, record in provenance.items():
        record['vendored_sha256'] = sha(path.parent / Path(relative).name)
        record['changes'].append('2026-11-20 canonical Error propagation package namespace')
    atomic_json(path, provenance)
    removed = []
    for name in ('flow_iql_research', 'propagation_research'):
        link = REPO / 'utils' / name
        assert link.is_symlink() and link.resolve() == CODE / name
        link.unlink()
        assert not link.exists() and (CODE / name).is_dir()
        removed.append(str(link))
    result = dict(time=time.time(), status='PASS', code_files=len(records),
                  resume_checkpoints=resumes, artifact_changes=changes, removed_links=removed,
                  backup=str(BACKUP), algorithm_changes=False)
    atomic_json(AUDIT / 'applied.json', result)
    print(json.dumps(dict(status='PASS', code_files=len(records), artifacts=len(changes),
                         resumes={r['id']: r['step'] for r in resumes}, removed_links=removed)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'apply'])
    arguments = parser.parse_args()
    prepare() if arguments.stage == 'prepare' else apply()
