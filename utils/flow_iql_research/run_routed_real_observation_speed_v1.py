"""Register and run a supplemental real-input timing after primary audits.

No producer is stopped. Both GPUs and all owned evaluators must already be
drained. The original benchmark and gates are kept intact and reported.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from utils.flow_iql_research.run_routed_strict_GPU_speed_v1 import readiness


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def birth(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    assert fields[0] != 'Z'
    return fields[19]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    area = root / 'analysis/routed_real_observation_speed_v1'
    area.mkdir(exist_ok=True)
    proposal = area / 'prospective_real_observation_speed_plan_v1.json'
    folder = Path(__file__).resolve().parent
    if a.register:
        assert not proposal.exists()
        reg = root / 'registration_v19.json'
        assert json.loads(reg.read_text())['routed_deployment_extension']['policy_key'] == 'mdn9_bc__component_location'
        paths = [Path(__file__).resolve(), folder / 'prepare_routed_latency_observations_v1.py',
            folder / 'benchmark_routed_real_observations_v1.py', folder / 'run_routed_strict_GPU_speed_v1.py']
        atomic(proposal, dict(time=time.time(), source_sha256={str(path): sha(path) for path in paths},
            registration_sha256=sha(reg), original_gates_and_speed_protocol_unchanged=True,
            benchmark_module='utils.flow_iql_research.benchmark_routed_real_observations_v1',
            environment_count=9, observations_per_environment=64, reset_seeds=[42, 105],
            model_step=1000000, model_training_seed=0, NVIDIA_TF32_OVERRIDE='0',
            common_precision_for_teacher_and_actor=True, modes=['primary', 'teacher'],
            input_placements=['resident', 'host_to_host'], repetitions=3, calls_per_repetition=100,
            warmup_calls=10, raw_per_call_times_saved=True,
            wrapper_requires_all64_actions_bitwise_equal_to_frozen_policy_before_timing=True,
            input_paths=[str(root / f'analysis/onestep_pipeline_v1/benchmark_attempt4/GPU{gpu}_input.json') for gpu in [0, 1]],
            admission='Originalseed0+fresh3performanceanddefaultspeedpass,strictFP32supplementcomplete,STOP+PAUSE,allGPUlearners/processesandownedCPUevaluatorsdrained.',
            scope='Additional realistic reset-observation and host-to-host timing. Original all-zero resident-input speed gates unchanged; report all scopes, no environment-specific runtime choice.',
            why='Original frozen benchmark uses zero observations and resident inputs. A deployed controller receives real host observations and consumes host actions; measure that cost explicitly. No new policy tuning.'))
        print(json.dumps(dict(status='registered_no_GPU_workers_started', proposal=str(proposal))))
        return
    plan = json.loads(proposal.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    checks = readiness(root)
    obs = area / 'real_reset_observations_manifest_v1.json'
    validation = area / 'actual_four_CPU_wrapper_validation_v1.json'
    strict = root / 'analysis/routed_strict_GPU_speed_v1/root_precision_speed_summary_v1.json'
    checks.update(real_observations_prepared=obs.exists(), CPU_wrapper_validation_passed=False,
                  strict_FP32_supplement_complete=False)
    if validation.exists():
        data = json.loads(validation.read_text())
        checks['CPU_wrapper_validation_passed'] = data['status'] == 'complete' and data['validation_only'] and len(data['rows']) == 4
    if strict.exists():
        checks['strict_FP32_supplement_complete'] = json.loads(strict.read_text())['status'] == 'complete'
    if a.check:
        print(json.dumps(dict(ready=all(checks.values()), checks=checks)))
        return
    assert all(checks.values()), checks
    observation_digest = sha(obs)
    invocation = area / 'invocation_before_workers_v1.json'
    assert not invocation.exists()
    jobs = []
    for gpu in [0, 1]:
        input_path = Path(plan['input_paths'][gpu])
        output = area / f'GPU{gpu}_real_observation_timings_v1.json'
        assert not output.exists()
        command = ['taskset', '-c', f'{6 * gpu}-{6 * gpu + 5}', sys.executable, '-u', '-m',
            plan['benchmark_module'], '--input', str(input_path), '--observations', str(obs), '--output', str(output)]
        jobs.append(dict(gpu=gpu, input=str(input_path), input_sha256=sha(input_path),
                         output=str(output), command=command))
    atomic(invocation, dict(time=time.time(), plan_sha256=sha(proposal), checks=checks,
        observations_manifest_sha256=observation_digest, CPU_validation_sha256=sha(validation), jobs=jobs))
    children, handles = [], []
    for job in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), NVIDIA_TF32_OVERRIDE='0',
            XLA_PYTHON_CLIENT_MEM_FRACTION='0.1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('JAX_PLATFORM_NAME', None)
        env.pop('JAX_PLATFORMS', None)
        with (area / f'GPU{job["gpu"]}_stdout_v1.log').open('xb') as log:
            child = subprocess.Popen(job['command'], env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        children.append(child)
        handles.append(dict(pid=child.pid, birth=birth(child.pid), **job))
        atomic(area / 'worker_handles_v1.json', dict(time=time.time(), workers=handles))
    done = set()
    while len(done) != 2:
        for index, child in enumerate(children):
            if index in done or child.poll() is None:
                continue
            assert child.returncode == 0, 'Inspect sealed handle/log; do not restart on timeout.'
            done.add(index)
        if len(done) != 2:
            time.sleep(.2)
    rows = []
    for job in jobs:
        d = json.loads(Path(job['output']).read_text())
        assert d['status'] == 'complete' and not d['validation_only']
        assert d['source_sha256'] == plan['source_sha256'][str(folder / 'benchmark_routed_real_observations_v1.py')]
        assert d['input_sha256'] == job['input_sha256'] and d['observations_manifest_sha256'] == observation_digest
        assert d['NVIDIA_TF32_OVERRIDE'] == '0'
        rows.extend(d['rows'])
    assert len(rows) == len({row['environment'] for row in rows}) == 9
    original = json.loads(strict.read_text())
    original_by_run = {row['run']: row for row in original['rows']}
    for row in rows:
        ref = original_by_run[row['run']]
        assert row['checkpoint_sha256'] == ref['checkpoint_sha256']
        assert row['actor_artifact_sha256'] == ref['actor_artifact_sha256']
    ratios = {f'{backend}_{placement}': math.exp(statistics.mean(math.log(
        row[f'{backend}_{placement}_latency_ratio']) for row in rows))
        for backend in ['cpu', 'gpu'] for placement in ['resident', 'host_to_host']}
    atomic(area / 'root_real_observation_speed_summary_v1.json', dict(time=time.time(), status='complete',
        rows=rows, geometric_latency_ratios=ratios, proposal_sha256=sha(proposal),
        invocation_sha256=sha(invocation), original_zero_input_primary_gates_unchanged=True,
        scope='Report resident-real and host-to-host-real ratios alongside all original benchmark results. No selective omission.'))
    print(json.dumps(dict(status='complete', ratios=ratios)))


if __name__ == '__main__':
    main()
