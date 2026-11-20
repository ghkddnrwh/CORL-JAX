"""Persistent CPU50/model audit, isolated timings, and preregistered promotion."""
import argparse
import csv
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
import sys
import time
import numpy as np
from utils.flow_iql_research import train_efficient_v1 as runner
from utils.flow_iql_research import efficient_mean_core_v1 as core


STEPS=(800000,900000,1000000)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def records(path,complete=False):
    if not path.exists():return []
    content=path.read_text()
    if content and not content.endswith('\n'):
        assert not complete
        content=content.rsplit('\n',1)[0] if '\n' in content else ''
    return [json.loads(line) for line in content.splitlines()]


def finite(tree):
    if tree is None:return True
    if isinstance(tree,dict):return all(finite(value) for value in tree.values())
    if isinstance(tree,(tuple,list)):return all(finite(value) for value in tree)
    return bool(np.isfinite(np.asarray(tree)).all())


def fingerprint(tree):
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,dict):
            for key in sorted(value):digest.update(str(key).encode()+b'\0');visit(value[key])
        else:
            array=np.ascontiguousarray(value)
            digest.update(str(array.shape).encode()+array.dtype.str.encode()+array.tobytes())
    visit(tree)
    return digest.hexdigest()


class Verifier:
    def __init__(self,plan):
        self.plan=plan
        self.pins={}
        self.contracts={}

    def config(self,run,identity):
        conf=json.loads((run/'config.json').read_text())
        assert conf['identity']==identity and conf['source_sha256']==self.plan['training_source_sha256']
        if run.name not in self.contracts:
            paths=[]
            for path,digest in conf['source_sha256'].items():
                p=Path(path)
                target=run/'source'/('vendor_'+p.name if p.parent.name=='vendor' else p.name)
                if not target.exists():return None
                assert sha(target)==digest
                paths.append(str(target))
            if not (run/'dataset.json').exists():return None
            dataset=json.loads((run/'dataset.json').read_text())
            assert dataset['normalize_states'] is False and dataset['action_clip_eps']==1e-5
            self.contracts[run.name]=dict(config_sha256=sha(run/'config.json'),frozen_source_copies=len(paths))
        assert sha(run/'config.json')==self.contracts[run.name]['config_sha256']
        return conf

    def model(self,run,conf,step):
        key=f'{run.name}:{step}'
        if key in self.pins:return self.pins[key]
        path=run/f'checkpoint_{step}.pkl'
        with path.open('rb') as stream:saved=pickle.load(stream)
        identity=conf['identity'];variant=identity['variant'];state=saved['state']
        assert saved['step']==int(state['q_updates'])==step
        assert saved['identity']==identity and saved['source_sha256']==conf['source_sha256']
        net=state['agent']['network']
        assert int(net['opt_state']['0']['count'])==step
        assert int(state['v_updates'])==int(state['v_opt']['0']['count'])==(2*step if variant=='reference' else 0)
        for name,counter,enabled in [('student','student_updates',variant in core.BC_STUDENTS),
                                     ('policy_student','policy_updates',variant in core.GUIDED_POLICIES)]:
            assert int(state[counter])==(step if enabled else 0)
            if enabled:assert int(state[name]['opt_state']['0']['count'])==step
            else:assert state[name] is None
        assert finite(state)
        pin=dict(path=str(path),sha256=sha(path),Q_Adam=step,V_Adam=int(state['v_updates']),full_saved_state_finite=True)
        if 'modules_actor_flow' in net['params']:
            pin['BC_params_sha256']=fingerprint(net['params']['modules_actor_flow'])
            pin['BC_Adam_mu_sha256']=fingerprint(net['opt_state']['0']['mu']['modules_actor_flow'])
            pin['BC_Adam_nu_sha256']=fingerprint(net['opt_state']['0']['nu']['modules_actor_flow'])
        self.pins[key]=pin
        return pin

    def evaluation(self,path,job,mode,step,conf,pin):
        row=json.loads(path.read_text())
        assert row['run']==job['id'] and row['variant']==job['args']['variant'] and row['policy_mode']==mode
        assert row['environment']==job['args']['env'] and row['training_seed']==job['args']['seed']
        assert row['step']==step and row['n_episodes']==50 and row['first_episode_seed']==42
        assert row['eval_device']=='cpu' and row['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
        assert row['source_sha256']==self.plan['all_source_sha256'][str(Path(runner.__file__).with_name('evaluate_efficient_v1.py'))]
        assert row['core_source_sha256']==self.plan['training_source_sha256'][str(Path(core.__file__))]
        assert row['config_sha256']==self.contracts[job['id']]['config_sha256']
        assert row['checkpoint_sha256']==pin['sha256'] and row['no_optimizer_updates'] is True
        assert all(len(row[name])==50 for name in ('episode_successes','episode_rewards','episode_lengths'))
        assert all(value in (0,1) for value in row['episode_successes'])
        assert all(math.isfinite(float(value)) for name in ('episode_rewards','episode_lengths') for value in row[name])
        assert all(float(value).is_integer() and value>0 for value in row['episode_lengths'])
        successes=sum(int(value) for value in row['episode_successes'])
        assert abs(row['success_rate']-successes/50)<1e-12
        assert abs(row['reward_mean']-statistics.mean(row['episode_rewards']))<1e-8
        return dict(success_rate=successes/50,success_count=successes,sha256=sha(path))


def select_candidate(groups,timing,pilots,criteria):
    candidates=[]
    for variant in core.VARIANTS:
        if variant in ('reference','fql'):continue
        current=[next((g for g in groups if g['variant']==variant and g['environment']==env and g['seed']==0),None) for env in pilots]
        reference=[next((g for g in groups if g['variant']=='reference' and g['environment']==env and g['seed']==0),None) for env in pilots]
        if not all(g and g['complete'] for g in current+reference):continue
        try:
            speeds=[timing[(variant,env)] for env in pilots]
            bases=[timing[('reference',env)] for env in pilots]
        except KeyError:continue
        if not all(row['benchmark_state_finite'] for row in speeds+bases):continue
        assert all(a['measurement_GPU']==b['measurement_GPU'] and a['measurement_CPUs']==b['measurement_CPUs'] for a,b in zip(speeds,bases))
        losses=[b['last3_mean']-g['last3_mean'] for g,b in zip(current,reference)]
        training=[g['training_ms_per_update_median']/b['training_ms_per_update_median'] for g,b in zip(speeds,bases)]
        inference=[g['cpu_action_ms_median']/b['cpu_action_ms_median'] for g,b in zip(speeds,bases)]
        train_ratio=math.exp(statistics.mean(math.log(value) for value in training))
        inference_ratio=math.exp(statistics.mean(math.log(value) for value in inference))
        eligible=statistics.mean(losses)<=criteria['pilot_average_loss_tolerance'] and max(losses)<=criteria['pilot_max_task_loss']
        eligible=eligible and train_ratio<=criteria['training_latency_ratio_max'] and inference_ratio<=criteria['CPU_deployment_latency_ratio_max']
        candidates.append(dict(variant=variant,eligible=eligible,pilot_mean_loss=statistics.mean(losses),pilot_task_losses=losses,
            training_latency_ratio_geomean=train_ratio,CPU_deployment_latency_ratio_geomean=inference_ratio,
            combined_speedup=1/(train_ratio*inference_ratio)))
    selected=sorted([c for c in candidates if c['eligible']],key=lambda c:(-c['combined_speedup'],c['variant']))
    return (selected[0] if selected else None),candidates


def GPU_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('contract',type=Path);args=parser.parse_args()
    contract=json.loads(args.contract.read_text());registration=Path(contract['registration'])
    assert sha(registration)==contract['registration_sha256']
    plan=json.loads(registration.read_text());root=Path(plan['root'])
    output=root/'analysis/efficiency_pipeline_v1';output.mkdir(exist_ok=False)
    eval_folder=root/'analysis/evaluations_v1';eval_folder.mkdir()
    bench_folder=root/'analysis/trained_isolated_timing_v1';bench_folder.mkdir()
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=list(plan['jobs']);identities=dict(plan['expected_identities'])
    verifier=Verifier(plan);running={};failed={};benchmarks={};benchmark_done={};promotion_decided=False;selected=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',MUJOCO_GL='egl')
    def result_path(rid,step,mode):return eval_folder/f'{rid}_{step}_{mode}.json'
    def launch_evaluation(job,step,mode,slot):
        key=f'{job["id"]}:{step}:{mode}'
        log=(eval_folder/(key.replace(':','_')+'.log')).open('a')
        command=['nice','-n','10','taskset','-c',','.join(map(str,plan['CPUslots'][slot])),sys.executable,'-u','-m',
            'utils.flow_iql_research.evaluate_efficient_v1','--run',str(root/'runs'/job['id']),'--step',str(step),
            '--mode',mode,'--output',str(result_path(job['id'],step,mode))]
        p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        running[key]=dict(process=p,log=log,slot=slot,process_identity=Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()[19])
    def promote(candidate,evidence):
        nonlocal jobs,identities
        # Only immutable completed models from THIS experiment may be relocated.
        # Every byte is verified and all public paths remain available via links.
        if shutil.disk_usage(root).free<18*1024**3:
            destination=Path(contract['completed_checkpoint_offload_destination'])
            command=[sys.executable,'-u','-m','utils.flow_iql_research.offload_completed_checkpoints_v2',
                str(root),'--destination',str(destination),'--target-free-gib','18','--apply']
            with (output/'completed_checkpoint_offload.log').open('a') as log:
                subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert shutil.disk_usage(root).free>=14*1024**3,'Insufficient free space for conditional confirmation; original data preserved'
        variant=candidate['variant'];new=[];expected={}
        for seed in (0,*plan['future_confirmation_seeds']):
            for index,environment in enumerate(core.ENVIRONMENTS):
                if any(j['args']['env']==environment and j['args']['variant']==variant and j['args']['seed']==seed for j in jobs):continue
                short=['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene'][index]
                rid=f'eff2_{short}_{variant}_s{seed}'
                job=dict(id=rid,module='utils.flow_iql_research.train_efficient_v1',args=dict(env=environment,variant=variant,seed=seed,steps=1000000))
                new.append(job);expected[rid]=runner.configuration(environment,variant,seed)
        receipt=dict(time=time.time(),selected_candidate=candidate,selection_evidence=evidence,new_jobs=new,expected_identities=expected,
            conditional_rule=plan['promotion_rule'],fixed_seeds=plan['future_confirmation_seeds'],
            scope='Same globallyselected recipe;missing6seed0tasks and27freshlearners launched together tofill16slots. Fresh results judged independently; no per-taskwinning recipe.',
            source_contract_sha256=sha(args.contract))
        queue_lock=(root/'queue_mutation.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_EX)
        queue=json.loads((root/'queue.json').read_text());ids={j['id'] for j in queue}
        assert not any(j['id'] in ids for j in new)
        atomic(output/'queue_before_promotion.json',queue)
        atomic(output/'selected_candidate_registration_v1.json',receipt)
        atomic(root/'queue.json',queue+new)
        fcntl.flock(queue_lock,fcntl.LOCK_UN);queue_lock.close()
        jobs+=new;identities.update(expected)
    while True:
        try:
            for path,digest in {**plan['all_source_sha256'],**contract['source_sha256']}.items():assert sha(path)==digest,path
            assert sha(registration)==contract['registration_sha256']
            for key,row in list(running.items()):
                code=row['process'].poll()
                if code is None:continue
                row['log'].close();del running[key]
                if code:failed[key]=dict(exit_code=code)
            for gpu,row in list(benchmarks.items()):
                code=row['process'].poll()
                if code is None:continue
                row['log'].close();del benchmarks[gpu]
                if code:failed[f'benchmark_GPU{gpu}']=dict(exit_code=code)
                else:
                    result=json.loads(row['output'].read_text());assert result['status']=='complete'
                    assert result['source_sha256']==contract['source_sha256'][str(Path(core.__file__).with_name('benchmark_efficient_models_v1.py'))]
                    benchmark_done[gpu]=result
            rows=[];ready_primary=[];ready_teacher=[];evaluations={}
            for job in jobs:
                rid=job['id'];run=root/'runs'/rid;variant=job['args']['variant']
                base=dict(run=rid,variant=variant,environment=job['args']['env'],seed=job['args']['seed'])
                if not (run/'config.json').exists():
                    rows.append(dict(**base,complete=False,training_complete=False,status='pending',last3_mean=None));continue
                conf=verifier.config(run,identities[rid])
                if conf is None:
                    rows.append(dict(**base,complete=False,training_complete=False,status='initializing',last3_mean=None));continue
                done=(run/'completed.json').exists();stopped=(run/'stopped.json').exists()
                probes=records(run/'diagnostics.jsonl',done)
                assert all(p['finite'] and p['q_updates']==p['step'] and p['v_updates']==(2*p['step'] if variant=='reference' else 0) for p in probes)
                if done:assert len(probes)==1000 and probes[-1]['step']==1000000
                if stopped:failed[rid]=dict(reason='nonfinite_state',path=str(run/'stopped.json'))
                per_step={};teacher={};counts=[]
                modes=['primary']+(['teacher'] if variant not in ('reference','fql','lean') else [])
                for step in STEPS:
                    for mode in modes:
                        key=f'{rid}:{step}:{mode}';path=result_path(rid,step,mode)
                        if not (run/f'checkpoint_{step}.pkl').exists():
                            if mode=='primary':per_step[str(step)]=None
                            continue
                        pin=verifier.model(run,conf,step)
                        if path.exists():
                            ev=verifier.evaluation(path,job,mode,step,conf,pin);evaluations[key]=ev
                            if mode=='primary':per_step[str(step)]=ev['success_rate'];counts.append(ev['success_count'])
                            else:teacher[str(step)]=ev['success_rate']
                        elif key not in running and key not in failed:
                            (ready_primary if mode=='primary' else ready_teacher).append((job,step,mode))
                complete=done and len(counts)==3
                rows.append(dict(**base,complete=complete,training_complete=done,status='complete' if complete else 'stopped' if stopped else 'evaluating' if done else 'training',
                    last3_mean=float(Fraction(sum(counts),150)) if complete else None,per_step=per_step,
                    auxiliary_teacher_last3=statistics.mean(teacher.values()) if len(teacher)==3 else None,
                    last_probe=probes[-1] if probes else None,observed_probe_count=len(probes)))
            ready=ready_primary+ready_teacher;occupied={row['slot'] for row in running.values()}
            if not failed:
                for slot in range(len(plan['CPUslots'])):
                    if slot not in occupied and ready:launch_evaluation(*ready.pop(0),slot)
            scheduler=json.loads((root/'scheduler_status.json').read_text())
            finished=json.loads((root/'finished.json').read_text()) if (root/'finished.json').exists() else {}
            for rid,record in finished.items():
                if record['result'] not in ('complete',):failed[rid]=dict(reason=record['result'],exit_code=record['exit_code'])
            initial_training_done=all(next(r for r in rows if r['run']==j['id'])['training_complete'] for j in plan['jobs'])
            external_GPU_processes=GPU_processes() if initial_training_done and not scheduler['running'] and not benchmarks else []
            if initial_training_done and not scheduler['running'] and not external_GPU_processes and not benchmark_done and not benchmarks and not failed:
                for gpu,task_indices in [(0,[0,2]),(1,[1])]:
                    environments=[plan['pilot_environments'][i] for i in task_indices]
                    cases=[dict(run=str(root/'runs'/j['id']),variant=j['args']['variant'],environment=j['args']['env'])
                        for j in plan['jobs'] if j['args']['env'] in environments]
                    np.random.RandomState(703+gpu).shuffle(cases)
                    input_path=bench_folder/f'GPU{gpu}_input.json';atomic(input_path,dict(jobs=cases))
                    out=bench_folder/f'GPU{gpu}_timings.json';log=(bench_folder/f'GPU{gpu}.log').open('a')
                    cpu_ids=list(range(6*gpu,6*gpu+6))
                    command=['taskset','-c',','.join(map(str,cpu_ids)),sys.executable,'-u','-m',
                        'utils.flow_iql_research.benchmark_efficient_models_v1','--input',str(input_path),'--output',str(out)]
                    p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    benchmarks[gpu]=dict(process=p,log=log,output=out)
            timing={}
            for data in benchmark_done.values():
                for row in data['rows']:
                    pin=verifier.pins[f'{row["run"]}:1000000'];assert row['checkpoint_sha256']==pin['sha256']
                    timing[(row['variant'],row['environment'])]=dict(row,measurement_GPU=data['visible_GPU'],measurement_CPUs=data['CPU_affinity'])
            initial_primary_complete=all(next(r for r in rows if r['run']==j['id'])['complete'] for j in plan['jobs'])
            if initial_primary_complete and len(benchmark_done)==2 and not promotion_decided and not failed:
                selected,evidence=select_candidate(rows,timing,plan['pilot_environments'],plan['promotion_rule'])
                atomic(output/'pilot_selection_audit_v1.json',dict(time=time.time(),selected=selected,candidates=evidence,rule=plan['promotion_rule']))
                promotion_decided=True
                if selected:promote(selected,evidence)
            task_groups=[]
            for variant in core.VARIANTS:
                selected_rows=[r for r in rows if r['variant']==variant and r['seed']==0 and r['environment'] in plan['pilot_environments']]
                ready=len(selected_rows)==3 and all(r['complete'] for r in selected_rows)
                task_groups.append(dict(variant=variant,completed_tasks=sum(r['complete'] for r in selected_rows),
                    pilot_three_task_mean=statistics.mean(r['last3_mean'] for r in selected_rows) if ready else None))
            full9=[]
            for variant in sorted({r['variant'] for r in rows}):
                for seed in sorted({r['seed'] for r in rows if r['variant']==variant}):
                    chosen=[r for r in rows if r['variant']==variant and r['seed']==seed]
                    if len(chosen)!=9:continue
                    assert {r['environment'] for r in chosen}==set(core.ENVIRONMENTS)
                    full9.append(dict(variant=variant,seed=seed,completed_tasks=sum(r['complete'] for r in chosen),
                        nine_task_mean=statistics.mean(r['last3_mean'] for r in chosen) if all(r['complete'] for r in chosen) else None))
            all_primary=len(rows)==len(jobs) and all(r['complete'] for r in rows)
            all_aux=all(result_path(j['id'],s,m).exists() for j in jobs for s in STEPS
                for m in (['primary','teacher'] if j['args']['variant'] not in ('reference','fql','lean') else ['primary']))
            complete=promotion_decided and all_primary and all_aux and not running and not benchmarks and not failed
            result=dict(time=time.time(),pid=os.getpid(),status='failure_requires_inspection' if failed else 'complete' if complete else 'confirmation_running' if selected else 'pilot_running',
                planned_GPU_jobs=len(jobs),completed_primary_runs=sum(r['complete'] for r in rows),rows=rows,pilot_groups=task_groups,
                full9=full9,selected_candidate=selected,promotion_decided=promotion_decided,
                isolated_timing_rows=list(timing.values()),external_GPU_processes_before_isolation=external_GPU_processes,
                filesystem_free_GiB=shutil.disk_usage(root).free/1024**3,
                actual_finite_models=len(verifier.pins),actual_model_pins=verifier.pins,
                valid_CPU_evaluations=len(evaluations),evaluation_receipts=evaluations,
                CPU_running={k:dict(pid=v['process'].pid,slot=v['slot'],process_identity=v['process_identity']) for k,v in running.items()},
                benchmark_running={str(g):v['process'].pid for g,v in benchmarks.items()},benchmark_completed_GPUs=list(benchmark_done),
                failures=failed,metric=plan['metric'],registration_sha256=sha(registration),execution_contract_sha256=sha(args.contract),
                limits=['Timing-only improvement does not establish preserved performance.',
                    'Primary policy and same-Q teacher controls are separate. No peak or task-wise winning mixtures.',
                    'Pilotmeans are three selected tasks, not nine tasks. Full9/freshmeans are withheld until complete.',
                    'Conditional3freshseedreference comparison is descriptive,not a matched statistical noninferiority proof.'])
            if selected:
                discovery=next((r['nine_task_mean'] for r in full9 if r['variant']==selected['variant'] and r['seed']==0),None)
                if discovery is not None:
                    result['discovery_practical_performance_threshold']=plan['promotion_rule']['full9_discovery_reference']-plan['promotion_rule']['full9_discovery_loss_tolerance']
                    result['discovery_performance_preserved_descriptively']=discovery>=result['discovery_practical_performance_threshold']
                fresh=[r['nine_task_mean'] for r in full9 if r['variant']==selected['variant'] and r['seed'] in plan['future_confirmation_seeds']]
                if len(fresh)==3 and all(value is not None for value in fresh):
                    result['fresh_three_seed_nine_task_mean']=statistics.mean(fresh)
                    result['fresh_seed_sample_SD']=statistics.stdev(fresh)
                    result['fresh_practical_performance_threshold']=plan['promotion_rule']['fresh_reference']-plan['promotion_rule']['fresh_mean_loss_tolerance']
                    result['fresh_performance_preserved_descriptively']=statistics.mean(fresh)>=result['fresh_practical_performance_threshold']
            atomic(output/'status.json',result)
            buffer=io.StringIO();writer=csv.writer(buffer)
            writer.writerow(['variant','environment','seed','status','800K','900K','1M','last3','teacher_last3'])
            for row in rows:
                writer.writerow([row['variant'],row['environment'],row['seed'],row['status'],
                    *[row.get('per_step',{}).get(str(step)) for step in STEPS],row['last3_mean'],row.get('auxiliary_teacher_last3')])
            temp=output/'results.csv.tmp';temp.write_text(buffer.getvalue());temp.replace(output/'results.csv')
            if complete:
                for pin in verifier.pins.values():assert sha(pin['path'])==pin['sha256']
                atomic(output/'final_audit.json',result)
                (root/'STOP').write_text('Allregistered efficiency learners,CPU50evaluations andisolatedtimings complete.\n')
                return
        except Exception as error:
            atomic(output/'controller_failure.json',dict(time=time.time(),error=repr(error),existing_results_preserved=True))
            raise
        time.sleep(20)


if __name__=='__main__':main()
