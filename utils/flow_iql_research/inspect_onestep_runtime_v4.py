"""Read-only process-identity and learner-counter inspection; no GPU imports."""
import json
import os
from pathlib import Path
import sys
import time


def process(pid, birth, module):
    directory = Path('/proc')/str(pid)
    stat = (directory/'stat').read_text().rsplit(') ', 1)[1].split()
    command = (directory/'cmdline').read_bytes().decode().split('\0')
    assert stat[0] != 'Z' and stat[19] == birth
    allowed=(module,) if isinstance(module,str) else module
    assert '-m' in command and command[command.index('-m')+1] in allowed
    return directory


def inspect(root):
    scheduler = json.loads((root/'scheduler_status.json').read_text())
    analysis = json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    continuation = json.loads((root/'CONTINUATION.json').read_text())
    counts, handles, controllers = {}, [], []
    for name, item in scheduler['running'].items():
        directory = process(item['pid'], item['process_identity'],
                            ('utils.flow_iql_research.train_onestep_deploy_v1',
                             'utils.flow_iql_research.train_set_onestep_deploy_v1',
                             'utils.flow_iql_research.train_mean_teacher_onestep_v1',
                             'utils.flow_iql_research.train_energy_onestep_v1'))
        env = dict(x.split('=', 1) for x in (directory/'environ').read_text().split('\0') if '=' in x)
        assert env['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.1'
        assert env['CUDA_VISIBLE_DEVICES'] == str(item['gpu'])
        assert len(os.sched_getaffinity(item['pid'])) == 6
        gpu = str(item['gpu'])
        counts[gpu] = counts.get(gpu, 0)+1
        handles.append(dict(run=name, pid=item['pid'], birth=item['process_identity'], gpu=item['gpu']))
    for role in ('scheduler', 'analysis_controller', 'gradient_controller'):
        item = continuation[role]
        command = item['command']
        module = command[command.index('-m')+1]
        process(item['pid'], item['process_identity'], module)
        controllers.append(dict(role=role, pid=item['pid'], birth=item['process_identity'], module=module))
    rows = [row for row in analysis['rows'] if row['run'] in scheduler['running']]
    probes = [row['last_probe'] for row in rows if row.get('last_probe')]
    assert all(probe['q_updates'] == probe['actor_updates'] == probe['step']
               and probe['v_updates'] == 0 for probe in probes)
    steps = [probe['step'] for probe in probes]
    return dict(time=time.time(), live=counts, handles=handles, controllers=controllers,
                pending=scheduler['pending'], complete=analysis['complete_primary_and_teacher_runs'],
                min_step=min(steps, default=0), max_step=max(steps, default=0),
                finite=all(probe['finite'] for probe in probes),
                probed_running_count=len(probes), running_count=len(handles),
                aux={row['run']: row['auxiliary_cpu10'] for row in rows if row.get('auxiliary_cpu10')},
                evaluation_failures=analysis['evaluation_failures'])


if __name__ == '__main__':
    print(json.dumps(inspect(Path(sys.argv[1]))), flush=True)
