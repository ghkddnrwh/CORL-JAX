"""Audit an exact empirical upper bound and privately test failure-state routing."""
import argparse
import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.watch_onestep_deploy_v6 import qualified_pick
from utils.flow_iql_research.scheduler_v4 import process_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/'analysis/onestep_pipeline_v1').mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    proof_path=root/'analysis/discovery_bound_attempt0_v1/raw_success_upper_bound_proof_v1.json';proof=json.loads(proof_path.read_text())
    plan=json.loads((root/'registration_v7.json').read_text());state=json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text());status=json.loads((root/'analysis/onestep_pipeline_v1/status.json').read_text())
    assert state['phase']=='confirmation'and state['active_attempt']==0 and state['attempts'][0]['stage']=='discovery'
    observed=0
    for row in proof['completed_environments']:
        count=0
        for step in proof['registered_checkpoint_steps']:
            path=root/'analysis/evaluations_v1'/f'{row["run"]}_{step}_primary_n50.json';raw=json.loads(path.read_text());assert sha(path)==proof['raw_evidence_sha256'][str(path)]
            assert raw['environment']==row['environment']and raw['recipe']==proof['recipe']and raw['training_seed']==0
            assert raw['step']==step and raw['n_episodes']==50 and raw['first_episode_seed']==42
            assert raw['protocol']=='last3_cpu50_independent_seeded_episodes_v1'and raw['no_optimizer_updates']
            assert len(raw['episode_successes'])==50 and all(x in(0,1)for x in raw['episode_successes'])
            value=sum(raw['episode_successes']);assert value==int(value);count+=int(value)
        assert count==row['successes_sum'];observed+=count
    assert observed==proof['observed_successes']and len(proof['completed_environments'])+len(proof['remaining_environments'])==9
    upper=Fraction(observed+150*len(proof['remaining_environments']),1350)
    floor=plan['thresholds']['historical_lean_seed0_nine_task_reference']-plan['thresholds']['mean_loss_tolerance']
    assert str(upper)==proof['nine_task_mean_upper_bound_fraction']and float(upper)<floor-1e-12
    assert proof['unchanged_historical_gate_minimum']==floor
    recipe=proof['recipe'];assert qualified_pick(status['pilot_candidates'],[])['recipe']==recipe
    tested=list(state['tested_candidates']);tested.append(recipe)
    picked=qualified_pick(status['pilot_candidates'],tested);assert picked is None or picked['recipe']!=recipe
    # Restore the actual coordinator in a private, weightless root with the
    # proposed failed-attempt state. The live test parent owns its manager slot.
    private=copy.deepcopy(plan);private['root']=str(out);registration=out/'private_registration.json';registration.write_text(json.dumps(private,indent=2))
    contract=out/'private_contract.json';contract.write_text(json.dumps(dict(registration=str(registration),registration_sha256=sha(registration),source_sha256=private['all_source_sha256'],GPU_manager_CPU_affinity=list(range(96)),GPU_scheduler_module='utils.flow_iql_research.scheduler_onestep_deploy_v4'),indent=2))
    state['phase']='pilot';state['tested_candidates']=tested
    state['attempts'][0].update(status='discovery_historical_gate_impossible',early_failure_bound=dict(proof=str(proof_path),sha256=sha(proof_path),upper_bound=float(upper),historical_minimum=floor))
    state['evaluation_workers']={};state['benchmark_workers']={};state['benchmark_results']={}
    (out/'analysis/onestep_pipeline_v1/research_state.json').write_text(json.dumps(state,indent=2))
    (out/'queue.json').write_text(json.dumps(state['jobs'],indent=2));(out/'finished.json').write_text('{}');(out/'scheduler_status.json').write_text(json.dumps(dict(pid=os.getpid(),running={},pending=len(state['jobs']),finished=0)))
    for rid in ('os1_ant_bank_bc_s0','os1_cubed_bank_bc_s0'):
        target=out/'runs'/rid;target.mkdir(parents=True);shutil.copyfile(root/'runs'/rid/'config.json',target/'config.json')
    command=[sys.executable,'-u','-m','utils.flow_iql_research.watch_onestep_deploy_v6',str(contract)]
    with(out/'controller.log').open('a')as log:child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=process_identity(child.pid);(out/'controller_launch.json').write_text(json.dumps(dict(pid=child.pid,birth=birth,command=command),indent=2))
    path=out/'analysis/onestep_pipeline_v1/status.json';end=time.monotonic()+25
    while time.monotonic()<end:
        if path.exists():break
        assert child.poll()is None,(out/'controller.log').read_text();time.sleep(.1)
    assert path.exists(),'Observe the same private PID before recovery; timeout is not terminal'
    result=json.loads(path.read_text());research=json.loads((out/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    assert result['pid']==child.pid and result['phase']=='pilot'and len(result['rows'])==78
    assert research['tested_candidates']==tested and len(research['attempts'])==1
    assert research['attempts'][0]['status']=='discovery_historical_gate_impossible'
    assert not result['CPU_running']and not result['benchmark_running']and not result['evaluation_failures']
    assert all(row['primary_mean']is None and row['teacher_mean']is None for row in result['rows'])
    assert not any(e['label']=='restart_owned_GPU_scheduler'for e in research['events'])
    assert process_identity(child.pid)==birth;os.kill(child.pid,signal.SIGTERM);child.wait(timeout=5)
    sources=[Path(__file__).resolve(),Path(__file__).resolve().with_name('watch_onestep_deploy_v6.py')]
    result=dict(time=time.time(),status='pass',no_performance_claim=True,proof_sha256=sha(proof_path),
        checks=['Recomputed exact716/1350 optimistic9taskbound from21rawCPU50episodefiles',
            'Same historical gate and .03 tolerance, no per-task veto',
            'Actual candidate picker excludes the proven failed recipe',
            'Actual unchanged controller accepts failed discovery state and remains in pilot',
            'Zero private GPU, learner, evaluator, benchmark or fabricated score'],
        source_sha256={str(p):sha(p)for p in sources},private_controller_terminal=True,no_production_state_queue_or_model_changes=True)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='pass',output=str(out/'result.json'))),flush=True)


if __name__=='__main__':main()
