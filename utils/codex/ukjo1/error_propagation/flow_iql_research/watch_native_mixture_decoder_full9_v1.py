"""Read-only nine-task deployment-decoder comparison over existing MDN jobs.

This private observer never registers or changes training jobs, checkpoints,
production evaluations, selection attempts, or final validation criteria.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def birth(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def score(path, job):
    row = json.loads(path.read_text())
    assert row['run'] == job['run_id'] and row['decoder'] == job['decoder']
    assert row['step'] == job['step'] and row['environment'] == job['environment']
    assert row['training_seed'] == 0 and row['recipe'] == 'mdn9_bc'
    assert row['n_episodes'] == 50 and row['first_episode_seed'] == 42
    assert row['optimizer_updates_performed'] == 0
    assert row['protocol'] == 'posthoc_native_mixture_common_decoder_CPU50_v1'
    assert row['learned_model_trees_loaded'] == ['actor_params']
    assert not row['candidate_bank'] and not row['critic_loaded'] and not row['Flow_loaded']
    assert row['action_heads_evaluated'] == 1 and not row['counts_toward_native_primary']
    assert row['actor_artifact_sha256'] == sha(job['actor'])
    assert row['config_sha256'] == sha(Path(job['run'])/'config.json')
    for key in ('episode_successes','episode_rewards','episode_lengths'):
        assert len(row[key]) == 50 and all(math.isfinite(float(v)) for v in row[key])
    assert all(v in (0,1) for v in row['episode_successes'])
    assert abs(sum(row['episode_successes'])/50-row['success_rate']) < 1e-12
    return dict(successes=int(sum(row['episode_successes'])), episodes=50,
                success_rate=row['success_rate'], path=str(path), sha256=sha(path))


def create(root):
    area = root/'analysis/native_mixture_decoder_full9_v1'
    area.mkdir(parents=True, exist_ok=True)
    proposal = area/'prospective_extension_v1.json'
    assert not proposal.exists(), 'Never relaunch an existing study.'
    old_area = root/'analysis/native_mixture_locations_v1'
    old_proposal = old_area/'prospective_common_decoder_study_v1.json'
    old_plan = json.loads(old_proposal.read_text())
    production = json.loads((root/'analysis/onestep_pipeline_v1/research_state.json').read_text())
    jobs = [j for j in production['jobs'] if j['args']['recipe']=='mdn9_bc' and j['args']['seed']==0]
    assert len(jobs) == len({j['args']['env'] for j in jobs}) == 9
    reused = {(x['case'],x['step'],x['decoder']): x['output'] for x in old_plan['evaluations']}
    evaluations = []
    for job in jobs:
        run = root/'runs'/job['id']
        case = job['id'].split('_')[1]
        for step in (800000,900000,1000000):
            for decoder in ('sampled_component_location','highest_weight_location'):
                previous = reused.get((case,step,decoder))
                output = previous or str(area/f'{job["id"]}_{step}_{decoder}_CPU50_v1.json')
                evaluations.append(dict(run=str(run),run_id=job['id'],case=case,
                    environment=job['args']['env'],step=step,decoder=decoder,
                    actor=str(run/f'actor_{step}.pkl'),output=output,reuses_first_study=bool(previous)))
    sources = [Path(__file__),Path(__file__).with_name('evaluate_native_mixture_locations_v1.py'),
               Path(__file__).with_name('lazy_mixture_location_actor_v1.py')]
    plan = dict(time=time.time(), schema='nine_task_common_decoder_readonly_extension_v1',
        root=str(root), first_study=str(old_proposal), first_study_sha256=sha(old_proposal),
        first_study_workers=str(old_area/'worker_handles_v1.json'),
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        cases=sorted({j['case'] for j in evaluations}), evaluations=evaluations,
        total_decoder_points=54, reused_decoder_points=24, new_decoder_points=30,
        fixed_global_decoders=['sampled_component_location','highest_weight_location'],
        no_per_environment_or_checkpoint_decoder_selection=True,
        discovered_on_training_seed_zero=True, fresh_training_confirmation_in_this_study=False,
        no_GPU_jobs_or_production_state_changes=True, optimizer_updates_performed=0,
        no_primary_evaluation_replacement=True,
        scope='All nine seed0 tasks, every800K/900K/1MCPU50. Exploratory decoder comparison, not new joint training or final fresh/latency confirmation.')
    with proposal.open('x') as f:
        json.dump(plan,f,indent=2,allow_nan=False)
    command=['nice','-n','19','taskset','-c','72-95',sys.executable,'-u','-m',
             'utils.flow_iql_research.watch_native_mixture_decoder_full9_v1','--proposal',str(proposal)]
    with (area/'observer_stdout_v1.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True)
    atomic(area/'observer_handle_v1.json',dict(time=time.time(),pid=child.pid,birth=birth(child.pid),
        command=command,proposal_sha256=sha(proposal)))
    print(json.dumps(dict(proposal=str(proposal),pid=child.pid,birth=birth(child.pid))))


def observe(proposal):
    plan=json.loads(proposal.read_text());root=Path(plan['root']);area=proposal.parent
    for path,digest in plan['source_sha256'].items():
        assert sha(path)==digest,path
    state_path=area/'observer_state_v1.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(running={},failures={})
    for info in state['running'].values():
        assert birth(info['pid'])==info['birth'] or Path(info['output']).exists()
    children={}
    while True:
        # The bounded four-task workers own their files until they exit.
        old_handles=json.loads(Path(plan['first_study_workers']).read_text())['workers']
        old_live=any(birth(x['pid'])==x['birth'] for x in old_handles)
        complete={}
        for job in plan['evaluations']:
            output=Path(job['output'])
            if output.exists():
                complete[job['output']]=score(output,job)
        for key,info in list(state['running'].items()):
            child=children.get(key)
            live=child.poll() is None if child is not None else birth(info['pid'])==info['birth']
            if not live:
                if key not in complete:
                    state['failures'][key]=dict(time=time.time(),**info)
                del state['running'][key]
                children.pop(key,None)
        if not old_live and not state['failures']:
            for job in plan['evaluations']:
                key=job['output']
                if key in complete or key in state['running']:
                    continue
                if len(state['running'])>=4:
                    break
                if not Path(job['actor']).exists():
                    continue
                occupied={x['slot'] for x in state['running'].values()}
                slot=next(x for x in range(4) if x not in occupied)
                argv=['nice','-n','19','taskset','-c',f'{72+slot*6}-{77+slot*6}',sys.executable,
                      '-u','-m','utils.flow_iql_research.evaluate_native_mixture_locations_v1',
                      '--run',job['run'],'--step',str(job['step']),'--decoder',job['decoder'],
                      '--output',key]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORM_NAME='cpu',OMP_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
                log_path=area/f'{job["run_id"]}_{job["step"]}_{job["decoder"]}_stdout_v1.log'
                with log_path.open('xb') as log:
                    child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                children[key]=child
                state['running'][key]=dict(pid=child.pid,birth=birth(child.pid),slot=slot,
                    output=key,command=argv,actor_sha256=sha(job['actor']))
                atomic(state_path,state)
        rows=[]
        for case in plan['cases']:
            for decoder in plan['fixed_global_decoders']:
                matches=[j for j in plan['evaluations'] if j['case']==case and j['decoder']==decoder]
                values=[complete[j['output']] for j in matches if j['output'] in complete]
                row=dict(case=case,decoder=decoder,complete=len(values)==3,observed_points=len(values))
                if len(values)==3:
                    row.update(last3_mean=sum(x['successes'] for x in values)/150,points=values)
                rows.append(row)
        state.update(time=time.time(),pid=os.getpid(),birth=birth(os.getpid()),
            proposal_sha256=sha(proposal),complete_points=len(complete),expected_points=54,
            old_four_task_workers_live=old_live,rows=rows,status='waiting_or_evaluating')
        if state['failures']:
            state['status']='failed_requires_root_review'
        if len(complete)==54 and not state['running']:
            state['status']='decoder_evaluations_complete'
            state['nine_task_means']={decoder:statistics.mean(x['last3_mean'] for x in rows if x['decoder']==decoder)
                                     for decoder in plan['fixed_global_decoders']}
        atomic(state_path,state)
        if state['status'] in ('failed_requires_root_review','decoder_evaluations_complete'):
            return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--proposal',type=Path);a=p.parse_args()
    if a.proposal:
        observe(a.proposal)
    else:
        create(a.root.resolve())


if __name__=='__main__':
    main()
