"""Bounded read-only common-decoder CPU study; never changes GPU queues."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def birth(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]


def worker(proposal, case):
    plan = json.loads(proposal.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    area = proposal.parent
    jobs = [x for x in plan['evaluations'] if x['case'] == case]
    save(area / f'{case}_worker_state_v1.json', dict(
        time=time.time(), pid=os.getpid(), birth=birth(os.getpid()),
        proposal_sha256=sha(proposal), status='running', completed=0))
    for index, job in enumerate(jobs):
        assert sha(job['actor']) == job['actor_sha256']
        assert sha(Path(job['run']) / 'config.json') == job['config_sha256']
        output = Path(job['output'])
        assert not output.exists(), output
        argv = [sys.executable, '-u', '-m',
                'utils.flow_iql_research.evaluate_native_mixture_locations_v1',
                '--run', job['run'], '--step', str(job['step']),
                '--decoder', job['decoder'], '--output', str(output)]
        started = time.time()
        result = subprocess.run(argv, check=False, close_fds=True)
        if result.returncode:
            save(area / f'{case}_worker_state_v1.json', dict(
                time=time.time(), pid=os.getpid(), birth=birth(os.getpid()),
                status='failed', completed=index, job=job,
                returncode=result.returncode))
            return result.returncode
        row = json.loads(output.read_text())
        assert row['actor_artifact_sha256'] == job['actor_sha256']
        assert row['n_episodes'] == 50 and row['first_episode_seed'] == 42
        assert row['optimizer_updates_performed'] == 0
        assert row['decoder'] == job['decoder']
        assert sha(job['actor']) == job['actor_sha256']
        save(area / f'{case}_worker_state_v1.json', dict(
            time=time.time(), pid=os.getpid(), birth=birth(os.getpid()),
            status='running' if index + 1 < len(jobs) else 'complete',
            completed=index + 1, expected=len(jobs), last_output=str(output),
            last_evaluation_seconds=time.time() - started))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.case:
        assert args.proposal
        raise SystemExit(worker(args.proposal, args.case))
    root = args.root.resolve()
    area = root / 'analysis/native_mixture_locations_v1'
    area.mkdir(parents=True, exist_ok=True)
    proposal = area / 'prospective_common_decoder_study_v1.json'
    assert not proposal.exists(), 'Existing study is never relaunched.'
    sources = [Path(__file__), Path(__file__).with_name('lazy_mixture_location_actor_v1.py'),
               Path(__file__).with_name('evaluate_native_mixture_locations_v1.py')]
    preflight = area / 'implementation_preflight_v1.json'
    assert json.loads(preflight.read_text())['status'] == 'pass'
    jobs = []
    cases = ('cube', 'pz3', 'cubed', 'ant')
    for case in cases:
        run = root / 'runs' / f'osm1_{case}_mdn9_bc_s0'
        for step in (800000, 900000, 1000000):
            for decoder in ('sampled_component_location', 'highest_weight_location'):
                actor = run / f'actor_{step}.pkl'
                jobs.append(dict(case=case, run=str(run), step=step, decoder=decoder,
                    actor=str(actor), actor_sha256=sha(actor),
                    config_sha256=sha(run / 'config.json'),
                    output=str(area / f'{case}_{step}_{decoder}_CPU50_v1.json')))
    # Reproduce one sealed primary result before the Puzzle decoder comparisons.
    control = dict(jobs[6], step=1000000, decoder='original_sample',
                   actor=str(root / 'runs/osm1_pz3_mdn9_bc_s0/actor_1000000.pkl'),
                   output=str(area / 'pz3_1000000_original_sample_replay_CPU50_v1.json'))
    control['actor_sha256'] = sha(control['actor'])
    jobs.insert(6, control)
    save(proposal, dict(time=time.time(), schema='private_common_decoder_comparison_v1',
        purpose='Test two globally fixed decoder rules on the same native jointly trained MDN actors.',
        no_optimizer_or_primary_result_changes=True, no_GPU_jobs_or_queue_mutations=True,
        not_full_nine_or_fresh_seed_confirmation=True, no_per_environment_decoder_selection=True,
        cases=list(cases), training_seed=0, steps=[800000, 900000, 1000000],
        episodes_per_point=50, first_episode_seed=42, evaluation_points=len(jobs),
        original_sample_replay_points=1, new_decoder_evaluation_points=24,
        preflight=str(preflight), preflight_sha256=sha(preflight),
        source_sha256={str(p.resolve()): sha(p) for p in sources}, evaluations=jobs))
    handles = []
    for index, case in enumerate(cases):
        cpus = f'{48 + index * 6}-{53 + index * 6}'
        command = ['nice', '-n', '19', 'taskset', '-c', cpus, sys.executable,
            '-u', '-m', 'utils.flow_iql_research.run_native_mixture_locations_CPU_v1',
            '--proposal', str(proposal), '--case', case]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORM_NAME='cpu',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        with open(area / f'{case}_worker_stdout_v1.log', 'xb') as log:
            child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        handles.append(dict(case=case, pid=child.pid, birth=birth(child.pid), command=command))
        save(area / 'worker_handles_v1.json', dict(time=time.time(),
            proposal_sha256=sha(proposal), workers=handles))
    print(json.dumps(dict(proposal=str(proposal), workers=handles)))


if __name__ == '__main__':
    main()
