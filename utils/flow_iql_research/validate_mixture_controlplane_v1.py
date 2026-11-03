"""Private, weightless integration of the fifth implementation. No GPU use."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research import onestep_implementation_registry_v3 as old_registry
from utils.flow_iql_research import onestep_implementation_registry_v4 as registry
from utils.flow_iql_research import scheduler_onestep_deploy_v5 as old_manager
from utils.flow_iql_research import scheduler_onestep_deploy_v6 as new_manager
from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending
from utils.flow_iql_research.register_mixture_onestep_v1 import build_plan
from utils.flow_iql_research.scheduler_v4 import process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    root,out = args.root.resolve(),args.output_dir.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out.mkdir(parents=True,exist_ok=False)
    (out / 'analysis').mkdir()
    plan,_ = build_plan(root)
    state = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert len(plan['jobs']) == 100 and len(state['jobs']) == 95
    for job in state['jobs']:
        x = job['args']
        assert registry.implementation(x['recipe']) == old_registry.implementation(x['recipe'])
        assert registry.configuration(x['env'],x['recipe'],x['seed']) == plan['expected_identities'][job['id']]
    assert new_manager.allowed_modules() == old_manager.allowed_modules() | {'utils.flow_iql_research.train_mixture_onestep_v1'}
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    assert len(scheduler['running']) == 16
    for item in scheduler['running'].values():
        assert process_identity(item['pid']) == item['process_identity']
        command = Path(f'/proc/{item["pid"]}/cmdline').read_bytes().decode().split('\0')
        assert command[command.index('-m')+1] in new_manager.allowed_modules()
    controller = Path(__file__).with_name('watch_onestep_deploy_v8.py').resolve()
    main_node = next(x for x in ast.parse(controller.read_text()).body if isinstance(x,ast.FunctionDef) and x.name == 'main')
    nodes = [next(x for x in main_node.body if isinstance(x,ast.FunctionDef) and x.name == name)
             for name in ('add_configuration','promote','begin_fresh')]
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(controller),'exec')
    captured = []
    bindings = dict(core=SimpleNamespace(ENVIRONMENTS=registry.MIXTURE.core.ENVIRONMENTS),
        configuration=registry.configuration,implementation=registry.implementation,
        append_jobs=lambda new,expected,label,details:captured.append(copy.deepcopy(dict(new=new,expected=expected,label=label))))
    bindings['state'] = dict(jobs=copy.deepcopy(plan['jobs']),attempts=[],phase='pilot',active_attempt=None)
    exec(code,bindings)
    bindings['promote'](dict(recipe='mdn9_bc',private_routing_only=True))
    assert len(captured) == 1 and len(captured[0]['new']) == 4
    missing = set(registry.MIXTURE.core.ENVIRONMENTS)-{j['args']['env'] for j in plan['mixture_implementation_extension']['new_jobs']}
    assert {j['args']['env'] for j in captured[0]['new']} == missing
    bindings['state']['jobs'] += captured[0]['new']
    captured.clear()
    bindings['begin_fresh'](bindings['state']['attempts'][0])
    assert len(captured) == 1 and len(captured[0]['new']) == 27
    assert len({(j['args']['env'],j['args']['seed']) for j in captured[0]['new']}) == 27
    queue = json.loads((root / 'queue.json').read_text())
    started = set(scheduler['running']) | set(json.loads((root / 'finished.json').read_text()))
    started |= {j['id'] for j in state['jobs'] if (root / 'runs' / j['id'] / 'config.json').exists()}
    extended = copy.deepcopy(state)
    extended['jobs'] = plan['jobs']
    combined = queue + plan['mixture_implementation_extension']['new_jobs']
    ordered = order_pending(plan,extended,combined,started)
    assert all(a == b for a,b in zip(combined,ordered) if a['id'] in started)
    assert order_pending(plan,extended,ordered,started) == ordered
    assert {j['id']:j for j in ordered} == {j['id']:j for j in combined}
    # Boot the actual controller with no model files and no real worker handles.
    # The validator's live PID prevents the private root from spawning a manager.
    private = copy.deepcopy(plan)
    private['root'] = str(out)
    registration = out / 'private_registration.json'
    registration.write_text(json.dumps(private,indent=2))
    contract = out / 'private_contract.json'
    contract.write_text(json.dumps(dict(registration=str(registration),registration_sha256=sha(registration),
        source_sha256=private['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),
        GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v6'),indent=2))
    private_state = {k:copy.deepcopy(state[k]) for k in
        ('round','round_recipes','all_recipes','phase','tested_candidates','attempts','active_attempt')}
    private_state['round_recipes'].append('mdn9_bc')
    private_state['all_recipes'].append('mdn9_bc')
    private_state.update(jobs=plan['jobs'],identities=plan['expected_identities'],evaluation_workers={},
        evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={},events=[])
    directory = out / 'analysis/onestep_pipeline_v1'
    directory.mkdir()
    (directory / 'research_state.json').write_text(json.dumps(private_state,indent=2))
    (out / 'queue.json').write_text(json.dumps(plan['jobs'],indent=2))
    (out / 'finished.json').write_text('{}')
    (out / 'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=100,finished=0)))
    for job in state['jobs']:
        if job['id'] in list(scheduler['running'])[:2]:
            target = out / 'runs' / job['id']
            target.mkdir(parents=True)
            shutil.copyfile(root / 'runs' / job['id'] / 'config.json',target / 'config.json')
    command = [sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v8',str(contract)]
    with (out / 'controller.log').open('a') as log:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,close_fds=True)
    birth = process_identity(child.pid)
    (out / 'controller_launch.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command),indent=2))
    status_path = directory / 'status.json'
    end = time.monotonic()+25
    while time.monotonic() < end:
        if status_path.exists():
            break
        assert child.poll() is None,(out / 'controller.log').read_text()
        time.sleep(.1)
    assert status_path.exists(),'Observation deadline: inspect the same private PID before recovery.'
    status = json.loads(status_path.read_text())
    after = json.loads((directory / 'research_state.json').read_text())
    assert len(status['rows']) == 100 and status['phase'] == 'confirmation' and after['active_attempt'] == 1
    assert after['attempts'] == state['attempts']
    assert not status['CPU_running'] and not status['benchmark_running'] and not status['evaluation_failures']
    assert all(row['primary_mean'] is None and row['teacher_mean'] is None for row in status['rows'])
    assert not status['pilot_candidates']
    assert not any(e['label'] == 'restart_owned_GPU_scheduler' for e in after['events'])
    assert process_identity(child.pid) == birth
    os.kill(child.pid,signal.SIGTERM)
    child.wait(timeout=5)
    sources = [Path(__file__).resolve(),Path(__file__).with_name('register_mixture_onestep_v1.py').resolve(),
        controller,Path(__file__).with_name('onestep_implementation_registry_v4.py').resolve(),
        Path(__file__).with_name('scheduler_onestep_deploy_v6.py').resolve(),
        Path(__file__).with_name('diagnose_onestep_gradients_v5.py').resolve(),
        Path(__file__).with_name('watch_onestep_gradients_v5.py').resolve()]
    result = dict(time=time.time(),status='pass',no_performance_claim=True,
        checks=['All95legacy jobs/configurations/implementations unchanged;100unique jobs',
                'New manager only adds mixture module and accepts all16current process identities',
                'Native promotion reuses5mixture screens and adds4missing seed0tasks; fresh routing adds27',
                'Pending-only order preserves started positions and remains idempotent',
                'Actual V8 weightless controller retains bankBC attempt1 and100jobs without fake scores',
                'Private controller launches zero learners/evaluators/benchmarks/GPU managers'],
        source_sha256={str(p):sha(p) for p in sources},private_controller_terminal=True,
        no_production_queue_or_model_changes=True)
    (out / 'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='pass',output=str(out / 'result.json'))),flush=True)


if __name__ == '__main__':
    main()
