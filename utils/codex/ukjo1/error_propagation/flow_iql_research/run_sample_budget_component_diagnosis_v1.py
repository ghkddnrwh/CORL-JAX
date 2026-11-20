"""CPU-only bounded scheduler for frozen component crossover evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    root = args.plan.parent
    slots = [list(range(6 * i, 6 * i + 6)) for i in range(args.workers)]
    assert set(sum(slots, [])) <= os.sched_getaffinity(0)
    queue, running, completed, failed = [], {}, {}, []
    for job in plan['jobs']:
        output = root / 'evaluations' / f'{job["id"]}.json'
        if output.exists():
            result = json.loads(output.read_text())
            assert result['job'] == job and result['n_episodes'] == 50
            completed[job['id']] = result['success_rate']
        else:
            queue.append(job)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', MUJOCO_GL='egl')
    (root / 'worker_logs').mkdir(exist_ok=True)
    while queue or running:
        for label, worker in list(running.items()):
            process = worker['process']
            if process.poll() is None:
                continue
            worker['log'].close()
            if process.returncode:
                failed.append(dict(job=label, exit_code=process.returncode))
            else:
                result = json.loads((root / 'evaluations' / f'{label}.json').read_text())
                assert result['source_sha256'] == plan['source_sha256']
                completed[label] = result['success_rate']
                print(json.dumps(dict(job=label, success=result['success_rate'], completed=len(completed))), flush=True)
            del running[label]
        if not failed:
            for slot, cores in enumerate(slots):
                if not queue or any(w['slot'] == slot for w in running.values()):
                    continue
                job = queue.pop(0)
                log = (root / 'worker_logs' / f'{job["id"]}.log').open('w')
                command = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, '-u', '-m',
                    'utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1', 'evaluate',
                    '--plan', str(args.plan.resolve()), '--job', job['id']]
                process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['id']] = dict(process=process, log=log, slot=slot)
        atomic(root / 'controller_status_v1.json', dict(timestamp_unix=time.time(), pid=os.getpid(),
            phase='failed' if failed else 'running' if queue or running else 'complete',
            completed=completed, failures=failed, pending=len(queue),
            running={k: dict(pid=w['process'].pid, slot=w['slot']) for k, w in running.items()}))
        if failed and not running:
            raise RuntimeError(failed)
        if queue or running:
            time.sleep(3)
    print(json.dumps(dict(status='complete', evaluations=len(completed))), flush=True)


if __name__ == '__main__':
    main()
