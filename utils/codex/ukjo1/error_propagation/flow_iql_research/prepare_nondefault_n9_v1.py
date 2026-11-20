"""Register 40 seed-zero tasks and a portable CPU/GPU allocation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research import nondefault_n9_core_v1 as core


HERE = Path(__file__).resolve().parent
PREFIX = 'utils.codex.ukjo1.error_propagation.flow_iql_research.'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def resources(gpus, per_gpu, eval_workers):
    cpus = sorted(os.sched_getaffinity(0))
    slots = [(gpu, slot) for slot in range(per_gpu) for gpu in gpus]
    # Evaluation always has four dedicated cores, matching the existing scores.
    evaluations = min(eval_workers, (len(cpus) - len(slots)) // 4)
    if evaluations < 1:
        raise ValueError('Need at least one CPU per training slot plus four CPU evaluation cores; reduce --per-gpu.')
    training_cores = min(4, (len(cpus) - evaluations * 4) // len(slots))
    train_cpus = [cpus[i*training_cores:(i+1)*training_cores] for i in range(len(slots))]
    start = len(slots) * training_cores
    eval_cpus = [cpus[start+4*i:start+4*(i+1)] for i in range(evaluations)]
    return dict(gpus=gpus, per_gpu=per_gpu, train_slots=slots, train_cpus=train_cpus,
                eval_cpus=eval_cpus, memory_fraction=.1, available_cpu_count=len(cpus))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--discount', type=float, choices=(.995, .999), required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--per-gpu', type=int, default=8)
    parser.add_argument('--eval-workers', type=int, default=8)
    parser.add_argument('--dataset-dir', type=Path, default=Path('~/.ogbench/data'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.per_gpu < 1 or args.eval_workers < 1 or len(set(args.gpus)) != len(args.gpus):
        parser.error('GPU IDs must be unique; worker counts must be positive.')
    root = args.root.expanduser().resolve()
    recipe = 'g995_n9' if args.discount == .995 else 'g999_n9'
    jobs = [dict(id=f"nd1_{t['code']}_t{t['task_id']}_s0", **t, recipe=recipe, seed=0,
                 method='finite_mean_n9', output=str(root/'runs'/f"nd1_{t['code']}_t{t['task_id']}_s0"),
                 train_module=PREFIX+'train_nondefault_n9_v1', eval_module=PREFIX+'evaluate_nondefault_n9_v1',
                 head_aggregation='min' if t['family'].startswith('antmaze-') else 'mean', reused=False)
            for t in core.TASKS]
    for job in jobs:
        job['evaluation_directory'] = str(root/'evaluations'/job['id'])
    allocation = resources(args.gpus, args.per_gpu, args.eval_workers)
    if args.dry_run:
        print(json.dumps(dict(discount=args.discount, seed=0, cells=40, jobs=jobs, resources=allocation), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    registration = root/'registration_v1.json'
    if registration.exists():
        previous = json.loads(registration.read_text())
        assert previous['discount'] == args.discount and previous['resources'] == json.loads(json.dumps(allocation))
        assert previous['dataset_dir'] == str(args.dataset_dir.expanduser().resolve())
        for path, expected in previous['source_sha256'].items():
            assert sha(path) == expected, path
        print('Existing identical registration retained:', registration)
        return
    sources = core.source_paths() + [HERE/name for name in (
        'train_nondefault_n9_v1.py', 'evaluate_nondefault_n9_v1.py', 'prepare_nondefault_n9_v1.py',
        'validate_nondefault_n9_v1.py', 'validate_nondefault_controller_v1.py',
        'run_nondefault_n9_v1.py', 'supervise_nondefault_n9_v1.py',
        'analyze_nondefault_n9_v1.py', 'train_mean_empirical_backup_v1.py')]
    sources.append(HERE.parent/'propagation_research/seeded_reset.py')
    packages = {}
    for name in ('ogbench', 'jax', 'jaxlib', 'flax', 'optax', 'numpy', 'scipy',
                 'gymnasium', 'mujoco', 'dm-control', 'ml-collections', 'distrax'):
        packages[name] = importlib.metadata.version(name)
    plan = dict(created=time.time(), purpose='N9 non-default OGBench task expansion', discount=args.discount,
                training_seeds=[0], sample_counts=[9], methods=['finite_mean_n9'], jobs=jobs, reused_jobs=[],
                total_cells=40, families=core.FAMILIES, environments=list(core.ENVIRONMENTS),
                steps=1000000, batch_size=256,
                expectile=.9, target_EMA=.005, lr=.0003, target_clipping=True,
                head_rule='AntMaze min; other families mean', actor_label_samples=9,
                lean_deployment_samples=9, actor_objective='best9 likelihood distillation, no Q maximization',
                actor_decoder='sampled_component_location', flow_steps=10, critic_heads=2,
                learned_V_updates=0, evaluation_steps=[800000, 900000, 1000000],
                deployment_modes=['lean', 'actor'], episodes_per_point=50,
                evaluation_reset_seeds=list(range(42, 92)), evaluation_cpu_cores=4,
                aggregation='Mean of 800K/900K/1M per task and deployment; one training seed; no seed SD',
                dataset_dir=str(args.dataset_dir.expanduser().resolve()), resources=allocation,
                source_sha256={str(p.resolve()): sha(p) for p in sources}, packages=packages,
                python=platform.python_version(), python_executable=sys.executable,
                prior_cancelled_studies_restarted=False)
    registration.write_text(json.dumps(plan, indent=2)+'\n')
    (root/'validation').mkdir(exist_ok=True)
    (root/'analysis').mkdir(exist_ok=True)
    print(json.dumps(dict(registration=str(registration), discount=args.discount, jobs=len(jobs),
                          concurrent_training=len(allocation['train_slots']),
                          concurrent_evaluations=len(allocation['eval_cpus'])), indent=2))


if __name__ == '__main__':
    main()
