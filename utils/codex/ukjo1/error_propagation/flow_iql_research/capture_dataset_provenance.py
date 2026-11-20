"""Freeze hashes of the actual nine-task local OGBench input archives.

This reads existing files only. It documents current inputs; it cannot recover
unrecorded historical in-memory reward arrays or prove a past loader state.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time
import zipfile
import ogbench


ENVS = (
    'antmaze-large-navigate-singletask-v0',
    'humanoidmaze-medium-navigate-singletask-v0',
    'humanoidmaze-large-navigate-singletask-v0',
    'antsoccer-arena-navigate-singletask-v0',
    'cube-single-play-singletask-v0',
    'cube-double-play-singletask-v0',
    'puzzle-3x3-play-singletask-v0',
    'puzzle-4x4-play-singletask-v0',
    'scene-play-singletask-v0',
)


def fingerprint(job):
    env, split, path = job
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as file:
        while True:
            data = file.read(8*1024*1024)
            if not data:
                break
            digest.update(data)
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns), f'Input changed while hashing: {path}'
    with zipfile.ZipFile(path) as archive:
        fields = sorted(archive.namelist())
    assert {'observations.npy', 'actions.npy', 'terminals.npy'} <= set(fields)
    print(env, split, before.st_size, digest.hexdigest()[:16], flush=True)
    return dict(environment=env, split=split, path=str(path), size_bytes=before.st_size,
                inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                sha256=digest.hexdigest(), archive_fields=fields)


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    dataset_dir = Path(ogbench.make_env_and_datasets.__globals__['DEFAULT_DATASET_DIR']).expanduser()
    jobs = []
    for environment in ENVS:
        parts = environment.split('-'); index = parts.index('singletask')
        stem = '-'.join(parts[:index]+parts[-1:])
        for split, suffix in (('train', ''), ('validation', '-val')):
            path = dataset_dir/f'{stem}{suffix}.npz'
            assert path.exists(), path
            jobs.append((environment, split, path))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fingerprint, jobs))
    source_paths = {Path(inspect.getsourcefile(ogbench.make_env_and_datasets))}
    for name in ('load_dataset', 'relabel_dataset'):
        source_paths.add(Path(inspect.getsourcefile(ogbench.make_env_and_datasets.__globals__[name])))
    versions = {}
    for name in ('ogbench', 'mujoco', 'numpy', 'jax', 'jaxlib', 'flax', 'optax', 'ml-collections'):
        versions[name] = importlib.metadata.version(name)
    result = dict(time=time.time(), dataset_directory=str(dataset_dir), environments=list(ENVS),
        files=rows, total_archive_bytes=sum(r['size_bytes'] for r in rows), versions=versions,
        loader_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Current local archive hashes, not a retrospective in-memory fingerprint for old runs.',
                'Single-task rewards and masks are deterministically relabeled by the installed loader and environment goal specification; archive bytes alone do not encode those computed arrays.',
                'No archives, installed packages, training states or replay streams are modified.'])
    path = root/'analysis/dataset_provenance_v1.json'
    assert not path.exists()
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2)); temp.replace(path)
    print('CAPTURED', len(rows), 'archives', result['total_archive_bytes'], 'bytes', flush=True)


if __name__ == '__main__':
    main()
