"""Preserve source bytes, installed versions and the18 actual dataset hashes.

No imports of JAX or OGBench and no training GPU allocations are needed. The
archive contains repository source, not datasets, model weights or credentials.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

from utils.propagation_research.summarize import ENVIRONMENTS


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-directory", type=Path, default=Path.home()/".ogbench/data")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty snapshot directory")
    repository = Path(__file__).resolve().parents[2]
    paths = [repository/"algorithms/uk_offline/basic/iql_residual_control.py",
             repository/"algorithms/uk_offline/basic/iql_jax.py",
             repository/"algorithms/uk_offline/common/checkpointing.py",
             repository/"hyperparams/ogbench/iql_jax.yml"]
    criterion = repository/"algorithms/uk_offline/basic/propagation/criterion.txt"
    if criterion.exists():
        paths.append(criterion)
    paths += sorted((repository/"utils/propagation_research").glob("*.py"))
    paths += sorted((repository/"utils/propagation_research").glob("*.md"))
    for name in ("algorithms/__init__.py", "algorithms/uk_offline/__init__.py",
                 "algorithms/uk_offline/basic/__init__.py", "algorithms/uk_offline/common/__init__.py",
                 "utils/__init__.py", "utils/propagation_research/__init__.py", "hyperparams/iql_jax.yml"):
        if (repository/name).exists():
            paths.append(repository/name)
    # Capture each source once, so the hash and exported bytes always agree.
    sources = {str(p.relative_to(repository)): p.read_bytes() for p in sorted(set(paths))}
    package_spec = importlib.util.find_spec("ogbench")
    if package_spec is None or package_spec.origin is None:
        raise ValueError("OGBench must be installed for an environment snapshot")
    package = Path(package_spec.origin).parent
    package_files = sorted(set(package.rglob("*.py")) | set(package.rglob("*.xml")))
    dependencies = {dist.metadata["Name"]: dist.version
                    for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    datasets = []
    for environment in ENVIRONMENTS:
        # All requested defaults omit an explicit task number; OGBench removes
        # 'singletask' and retains the same train/validation dataset family.
        name = environment.replace("-singletask", "")
        for suffix, split in ((".npz", "train"), ("-val.npz", "validation")):
            path = args.dataset_directory/(name+suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            datasets.append({"environment": environment, "split": split,
                "filename": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
            print("HASHED", path.name, flush=True)
    plan_bytes = (args.campaign/"replication_plan.json").read_bytes()
    extra_inputs = {name: (args.campaign/name).read_bytes() for name in
                    ("factorial_plan.json", "seeded_evaluation_plan.json", "queue.json")
                    if (args.campaign/name).exists()}
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=10).splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu = []
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result = {"captured_at": time.time(), "python_version": platform.python_version(),
        "python_executable": sys.executable, "platform": platform.platform(), "GPU": gpu,
        "git_head": revision,
        "installed_package_versions": dict(sorted(dependencies.items())),
        "repository_source_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()},
        "ogbench_source_sha256": {str(p.relative_to(package)): digest(p) for p in package_files},
        "dataset_files": datasets,
        "replication_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "additional_experiment_input_sha256": {name: hashlib.sha256(data).hexdigest()
                                               for name, data in extra_inputs.items()},
        "scope": "Repository code and fixed-plan export plus local environment and data identity. Dataset/model binaries and installed packages are not bundled. Regenerate suite paths in the restored repository."}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps(result, indent=2).encode()
    (args.output/"environment.json").write_bytes(manifest)
    archive = args.output/"source_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        items = {"repository/"+name: payload for name, payload in sources.items()}
        items.update({"inputs/environment.json": manifest, "inputs/replication_plan.json": plan_bytes})
        items.update({"inputs/"+name: data for name, data in extra_inputs.items()})
        for name, payload in items.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            tar.addfile(info, io.BytesIO(payload))
    (args.output/"source_bundle.sha256").write_text(digest(archive)+"  source_bundle.tar.gz\n")
    print(json.dumps({"datasets_hashed": len(datasets), "source_files": len(sources),
                      "ogbench_code_files": len(package_files), "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
