"""Private CPU routing checks; zero production learners/models/evaluations."""
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
from utils.flow_iql_research.onestep_implementation_registry_v2 import configuration, implementation
from utils.flow_iql_research.onestep_scheduling_policy_v1 import order_pending
from utils.flow_iql_research.register_set_full9_prefetch_v1 import build_plan
from utils.flow_iql_research.scheduler_v4 import process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out.mkdir(parents=True,exist_ok=False)
    (out / 'analysis').mkdir()
    plan, _ = build_plan(root)
    state = json.loads((root / 'analysis/onestep_pipeline_v1/research_state.json').read_text())
    environments = plan['scheduling_amendment']['all_nine_environments']
    selected = [j for j in plan['jobs'] if j['args']['recipe'] == 'set_bc']
    assert len(plan['jobs']) == 90 and len(selected) == 9
    assert {j['args']['env'] for j in selected} == set(environments)
    assert all(j['args']['seed'] == 0 for j in plan['jobs'])
    assert plan['initial_recipes'] == json.loads((root / 'registration_v8.json').read_text())['initial_recipes']
    controller = Path(__file__).with_name('watch_onestep_deploy_v6.py').resolve()
    tree = ast.parse(controller.read_text())
    main_node = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name == 'main')
    nodes = [next(x for x in main_node.body if isinstance(x,ast.FunctionDef) and x.name == name)
             for name in ('add_configuration','promote')]
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(controller),'exec')
    captured = []
    bindings = dict(core=SimpleNamespace(ENVIRONMENTS=environments),configuration=configuration,
                    implementation=implementation,append_jobs=lambda new,expected,label,details:
                    captured.append(dict(new=copy.deepcopy(new),expected=copy.deepcopy(expected),label=label)))
    def simulate(jobs):
        bindings['state'] = dict(jobs=copy.deepcopy(jobs),attempts=[],phase='pilot',active_attempt=None)
        captured.clear()
        exec(code,bindings)
        bindings['promote'](dict(recipe='set_bc',private_routing_only=True))
        assert len(captured) == 1
        return copy.deepcopy(captured[0])
    duplicate = simulate(plan['jobs'])
    assert duplicate['new'] == [] and duplicate['expected'] == {}
    removed = plan['set_full9_prefetch_extension']['new_jobs'][0]
    control = simulate([j for j in plan['jobs'] if j['id'] != removed['id']])
    assert len(control['new']) == 1 and control['new'][0]['args'] == removed['args']
    assert control['new'][0]['module'] == implementation('set_bc').runner_module
    assert control['expected'][control['new'][0]['id']] == configuration(removed['args']['env'],'set_bc',0)
    # Validate the pending-only order on the real queue snapshot, without I/O.
    queue = json.loads((root / 'queue.json').read_text())
    scheduler = json.loads((root / 'scheduler_status.json').read_text())
    started = set(scheduler['running']) | set(json.loads((root / 'finished.json').read_text()))
    started |= {j['id'] for j in state['jobs'] if (root / 'runs' / j['id'] / 'config.json').exists()}
    extended_state = copy.deepcopy(state)
    extended_state['jobs'] = plan['jobs']
    new = plan['set_full9_prefetch_extension']['new_jobs']
    combined = queue + new
    ordered = order_pending(plan,extended_state,combined,started)
    assert all(left == right for left,right in zip(combined,ordered) if left['id'] in started)
    assert {j['id']:j for j in ordered} == {j['id']:j for j in combined}
    pending = [j for j in ordered if j['id'] not in started]
    current = [j for j in pending if j['args']['recipe'] == 'bank_bc']
    assert pending[:len(current)] == current
    assert order_pending(plan,extended_state,ordered,started) == ordered
    # No weights and no actual worker handles enter the isolated root. Preserve
    # actual discovery phase/attempt definitions, but do not invent any scores.
    private = copy.deepcopy(plan)
    private['root'] = str(out)
    registration = out / 'private_registration.json'
    registration.write_text(json.dumps(private,indent=2))
    contract = out / 'private_contract.json'
    contract.write_text(json.dumps(dict(registration=str(registration),registration_sha256=sha(registration),
        source_sha256=private['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),
        GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v4'),indent=2))
    private_state = {k:copy.deepcopy(state[k]) for k in
        ('round','round_recipes','all_recipes','phase','tested_candidates','attempts','active_attempt')}
    private_state.update(jobs=plan['jobs'],identities=plan['expected_identities'],evaluation_workers={},
        evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={},events=[])
    state_dir = out / 'analysis/onestep_pipeline_v1'
    state_dir.mkdir()
    (state_dir / 'research_state.json').write_text(json.dumps(private_state,indent=2))
    (out / 'queue.json').write_text(json.dumps(plan['jobs'],indent=2))
    (out / 'finished.json').write_text('{}')
    (out / 'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=90,finished=0)))
    for job in selected:
        src = root / 'runs' / job['id'] / 'config.json'
        if src.exists():
            dest = out / 'runs' / job['id']
            dest.mkdir(parents=True)
            shutil.copyfile(src,dest / 'config.json')
    command = [sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v6',str(contract)]
    with (out / 'controller.log').open('a') as log:
        child = subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth = process_identity(child.pid)
    (out / 'controller_launch.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command),indent=2))
    path = state_dir / 'status.json'
    end = time.monotonic() + 25
    while time.monotonic() < end:
        if path.exists():
            break
        assert child.poll() is None, (out / 'controller.log').read_text()
        time.sleep(.1)
    assert path.exists(), 'Observation deadline: inspect the same private child before recovery.'
    status = json.loads(path.read_text())
    after = json.loads((state_dir / 'research_state.json').read_text())
    assert status['pid'] == child.pid and status['phase'] == 'confirmation'
    assert len(status['rows']) == 90 and after['active_attempt'] == 1
    assert after['attempts'] == state['attempts']
    assert not status['CPU_running'] and not status['benchmark_running'] and not status['evaluation_failures']
    assert all(row['primary_mean'] is None and row['teacher_mean'] is None for row in status['rows'])
    assert not status['pilot_candidates']
    assert not any(e['label'] == 'restart_owned_GPU_scheduler' for e in after['events'])
    assert process_identity(child.pid) == birth
    os.kill(child.pid,signal.SIGTERM)
    child.wait(timeout=5)
    sources = [Path(__file__).resolve(),Path(__file__).with_name('register_set_full9_prefetch_v1.py').resolve(),
               controller,Path(__file__).with_name('onestep_scheduling_policy_v1.py').resolve()]
    result = dict(time=time.time(),status='pass',no_performance_claim=True,
        checks=['All9same-recipe Set jobs unique, seed0only, existing84jobs preserved',
                'Native promotion reuses all9; removing one schedules only the missing configuration',
                'Pending order retains started slots and prioritizes active bankBC discovery; idempotent',
                'Unchanged V6 controller starts with90jobs and preserves actual active discovery/attempt definitions',
                'Weightless private root starts zero learners, evaluators, benchmarks or GPU managers',
                'No fabricated performance, no formal Set promotion, no fresh seed allocation'],
        source_sha256={str(p):sha(p) for p in sources},private_controller_terminal=True,
        no_production_queue_or_model_changes=True)
    (out / 'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='pass',output=str(out / 'result.json'))),flush=True)


if __name__ == '__main__':
    main()
