"""Record the actual runtime, provided-code translation and current data bytes.

No packages are installed/upgraded. Dataset hashes describe current files, not
retrospective proof of the bytes consumed by every earlier training process.
"""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from utils.flow_iql_research import core
import ogbench.utils


def signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def digest(path):
    before = signature(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    assert before == signature(path), f'File changed during hashing: {path}'
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    supplied = Path('algorithms/uk_offline/basic/propagation/fql').resolve()
    vendor = Path('utils/flow_iql_research/vendor').resolve()
    assert (supplied / '.git').exists()
    assert Path(subprocess.check_output(['git', '-C', str(supplied), 'rev-parse', '--show-toplevel'], text=True).strip()) == supplied
    commit = subprocess.check_output(['git', '-C', str(supplied), 'rev-parse', 'HEAD'], text=True).strip()
    rows = []
    for name in ('iql', 'ifql', 'fql', 'rebrac', 'flax_utils', 'networks', 'encoders'):
        original = supplied / ('agents' if name in ('iql', 'ifql', 'fql', 'rebrac') else 'utils') / (name + '.py')
        destination = vendor / (name + '.py')
        expected = original.read_text().replace('from utils.', 'from utils.flow_iql_research.vendor.')
        unfreeze = name in ('iql', 'ifql', 'fql', 'rebrac')
        if unfreeze:
            before = "network_params = network_def.init(init_rng, **network_args)['params']"
            after = "network_params = flax.core.unfreeze(network_def.init(init_rng, **network_args)['params'])"
            assert expected.count(before) == 1
            expected = expected.replace(before, after)
        assert expected == destination.read_text(), f'Unregistered translation in {name}'
        assert ast.dump(ast.parse(expected), include_attributes=False) == ast.dump(ast.parse(destination.read_text()), include_attributes=False)
        rows.append(dict(module=name, provided_source=str(original), provided_sha256=digest(original),
            vendored_source=str(destination), vendored_sha256=digest(destination),
            exact_text_after_explicit_import_and_container_translation=True,
            unfreeze_parameter_container_only=unfreeze))
    translation = dict(provided_repository=str(supplied), provided_commit=commit, modules=rows,
        provided_tracked_diff_names=subprocess.check_output(['git', '-C', str(supplied), 'diff', '--name-only'], text=True).splitlines(),
        license_path=str(supplied / 'LICENSE'), license_sha256=digest(supplied / 'LICENSE'),
        scope='Current supplied working-tree source, with explicit import/container-only adaptation. Does not claim bitwise equivalence across library versions.')
    write(output / 'official_vendor_translation.json', translation)
    names = ('jax', 'jaxlib', 'flax', 'optax', 'chex', 'distrax', 'ml-collections', 'numpy', 'scipy',
             'ogbench', 'mujoco', 'gymnasium', 'gym', 'matplotlib')
    key_versions = {name: metadata.version(name) for name in names}
    installed = sorted([dict(name=d.metadata.get('Name', 'UNKNOWN'), version=d.version)
        for d in metadata.distributions()], key=lambda d: (d['name'].lower(), d['version']))
    # Whitelist numeric/runtime settings only; never dump the process environment.
    keys = ('XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_FLAGS',
        'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_ENABLE_X64', 'NVIDIA_TF32_OVERRIDE',
        'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'MUJOCO_GL')
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    worker_environment = []
    for name, row in scheduler['running'].items():
        proc = Path('/proc') / str(row['pid'])
        assert (proc / 'stat').read_text().split(') ', 1)[1].split()[19] == row['process_identity']
        environment = dict(part.split(b'=', 1) for part in (proc / 'environ').read_bytes().split(b'\0') if b'=' in part)
        worker_environment.append(dict(run=name, pid=row['pid'], process_identity=row['process_identity'],
            gpu=row['gpu'], settings={k: environment[k.encode()].decode() if k.encode() in environment else None for k in keys}))
    runtime = dict(time=time.time(), python=sys.version, python_executable=sys.executable,
        platform=platform.platform(), machine=platform.machine(), CPU_count=os.cpu_count(),
        important_versions=key_versions, installed_distribution_versions=installed,
        active_GPU_worker_settings=worker_environment,
        GPU_driver=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'], text=True),
        provided_requirements_path=str(supplied / 'requirements.txt'),
        provided_requirements_sha256=digest(supplied / 'requirements.txt'),
        provided_requirements_text=(supplied / 'requirements.txt').read_text(),
        limitations=['The working campaign uses the existing ogbench environment, not the exact software versions requested by the provided FQL requirements.',
            'Official algorithm losses/architecture/configuration alignment does not imply a bitwise reproduction of the paper environment.',
            'Observed package versions are provenance, not a newly resolved or tested installation lock.',
            'Only runtime-setting names from an explicit whitelist were read; no credentials or arbitrary environment contents are recorded.'])
    write(output / 'runtime.json', runtime)
    (output / 'runtime_observed_versions.txt').write_text('# Observed working Python3.10 environment; no installation or resolver change was made.\n'
        '# Full provenance and provided-requirement differences are in runtime.json.\n'
        + ''.join(f'{name}=={version}\n' for name, version in key_versions.items()))
    package = Path(ogbench.utils.__file__).resolve().parent
    suffixes = {'.py', '.xml', '.stl', '.obj', '.png', '.jpg', '.jpeg', '.json', '.yaml', '.yml'}
    sources = []
    for file in sorted(package.rglob('*')):
        if file.is_file() and file.suffix.lower() in suffixes and '__pycache__' not in file.parts:
            sources.append(dict(path=str(file.relative_to(package.parent)), bytes=file.stat().st_size, sha256=digest(file)))
    write(output / 'installed_ogbench_sources_assets.json', dict(package_root=str(package), files=sources,
        scope='Current installed OGBench Python sources/assets. Package version alone does not identify locally changed source bytes.'))
    plan = json.loads((root / 'independent_critic_fresh_confirmation_registration_v1.json').read_text())
    data_dir = Path(ogbench.utils.DEFAULT_DATASET_DIR).expanduser()
    datasets = []
    for environment in plan['environments']:
        parts = environment.split('-')
        pos = parts.index('singletask')
        dataset = '-'.join(parts[:pos] + parts[-1:])
        for split, filename in (('train', dataset + '.npz'), ('validation', dataset + '-val.npz')):
            path = data_dir / filename
            assert path.is_file(), f'Missing existing dataset; no download permitted here: {path}'
            stat = path.stat()
            row = dict(environment=environment, split=split, path=str(path), resolved_path=str(path.resolve()),
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path))
            datasets.append(row)
            write(output / 'dataset_hash_progress.json', dict(pid=os.getpid(), time=time.time(), files=len(datasets), rows=datasets))
            print(json.dumps(dict(dataset=filename, bytes=stat.st_size, files_complete=len(datasets))), flush=True)
    assert len(datasets) == 18 and len({r['path'] for r in datasets}) == 18
    write(output / 'datasets.json', dict(time=time.time(), datasets=datasets,
        loader_source_sha256=digest(ogbench.utils.__file__),
        semantics='Current raw train/validation bytes, compact_datasetFalse, single-task loader adds state information and relabels rewards/masks with installed default task.',
        limitation='Current-file provenance is not retrospective proof of all historical run inputs; no training output or dataset file was changed.'))
    files = [p for p in output.iterdir() if p.is_file()]
    write(output / 'manifest.json', dict(time=time.time(), status='complete', pid=os.getpid(), source_sha256=digest(__file__),
        files={p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in files},
        provided_commit=commit, translated_modules=7, current_raw_dataset_files=18,
        total_raw_dataset_bytes=sum(r['bytes'] for r in datasets), installed_ogbench_source_assets=len(sources),
        no_package_install_or_upgrade=True, no_training_or_primary_evaluation_changes=True, goal_complete=False))
    print(json.dumps(dict(status='complete', modules=7, raw_dataset_files=18,
        total_raw_dataset_bytes=sum(r['bytes'] for r in datasets), important_versions=key_versions)), flush=True)


if __name__ == '__main__':
    main()
