"""Implementation-aware actor search: preserve old jobs and support set actors.

This controller never updates a model. Every newly queued configuration is
registered before training; failed recipes do not end the explicitly active goal.
"""
import argparse
import csv
import fcntl
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time
import numpy as np
from utils.flow_iql_research import onestep_deploy_core_v1 as core
from utils.flow_iql_research.routed_policy_registry_v1 import (
    implementation, specification, configuration, config_for, training_sources, SET_RECIPES,
    order_pending,training_recipe,POLICY,NATIVE,MODE)
from utils.flow_iql_research.onestep_scheduling_policy_v1 import discovery_confirmation
from utils.flow_iql_research.routed_policy_verifier_v1 import PolicyVerifier,derived_rows,actor_path
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha


from utils.flow_iql_research.discovery_bound_policy_v1 import empirical_bound
from utils.flow_iql_research.routed_policy_registry_v1 import preferred_candidate

STEPS=(800000,900000,1000000)


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except (FileNotFoundError,ProcessLookupError):return None


def arrays(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):yield from arrays(tree[key],prefix+'/'+str(key))
    elif isinstance(tree,(tuple,list)):
        for key,value in enumerate(tree):yield from arrays(value,prefix+'/'+str(key))
    elif tree is not None:yield prefix,np.asarray(tree)


def partial_records(path,complete=False):
    if not path.exists():return []
    data=path.read_text()
    if data and not data.endswith('\n'):
        assert not complete
        data=data.rsplit('\n',1)[0] if '\n' in data else ''
    return [json.loads(line)for line in data.splitlines()]


class Verifier:
    def __init__(self,plan):self.plan=plan;self.pins={};self.configs={}

    def config(self,run,identity):
        conf=json.loads((run/'config.json').read_text())
        assert conf['identity']==identity and conf['source_sha256']==training_sources(self.plan,identity['recipe'])
        if run.name not in self.configs:
            for source,digest in conf['source_sha256'].items():
                p=Path(source);copy=run/'source'/('vendor_'+p.name if p.parent.name=='vendor' else p.name)
                if not copy.exists():return None
                assert sha(copy)==digest
            if not(run/'dataset.json').exists():return None
            dataset=json.loads((run/'dataset.json').read_text())
            assert dataset['normalize_states'] is False and dataset['action_clip_eps']==1e-5
            self.configs[run.name]=dict(config_sha256=sha(run/'config.json'),dataset=dataset)
        assert sha(run/'config.json')==self.configs[run.name]['config_sha256']
        return conf

    def model(self,run,conf,step):
        key=f'{run.name}:{step}'
        if key in self.pins:return self.pins[key]
        path=run/f'checkpoint_{step}.pkl';actor_path=run/f'actor_{step}.pkl'
        with path.open('rb')as stream:saved=pickle.load(stream)
        with actor_path.open('rb')as stream:actor=pickle.load(stream)
        assert saved['step']==actor['step']==actor['actor_updates']==actor['Q_updates']==step
        assert saved['identity']==actor['identity']==conf['identity']
        assert saved['source_sha256']==actor['source_sha256']==conf['source_sha256']
        state=saved['state'];base=state['base'];net=base['agent']['network']
        assert int(base['q_updates'])==int(net['opt_state']['0']['count'])==step
        assert int(state['actor_updates'])==int(state['actor']['opt_state']['0']['count'])==step
        assert int(base['v_updates'])==int(base['v_opt']['0']['count'])==0
        assert base['student'] is None and base['policy_student'] is None
        assert all(np.isfinite(value).all()for _,value in arrays(state))
        left=dict(arrays(state['actor']['params']));right=dict(arrays(actor['actor_params']))
        assert left.keys()==right.keys() and all(np.array_equal(left[k],right[k])for k in left)
        dataset=self.configs[run.name]['dataset']
        expected=implementation(conf['identity']['recipe']).core.actor_metadata(conf['identity']['config'],dataset['observation_dim'],dataset['action_dim'])
        assert actor['metadata']==expected and actor['config_sha256']==self.configs[run.name]['config_sha256']
        digest=sha(path);assert actor['checkpoint_sha256']==digest
        pin=dict(path=str(path),checkpoint_sha256=digest,actor_path=str(actor_path),actor_artifact_sha256=sha(actor_path),
            actual_Q_Adam=step,actual_actor_Adam=step,V_Adam=0,finite_full_saved_state=True,
            exported_actor_params_exact=True,actor_parameter_count=sum(value.size for value in right.values()),metadata=expected)
        self.pins[key]=pin;return pin

    def evaluation(self,path,job,step,mode,episodes,pin):
        row=json.loads(path.read_text());args=job['args'];rid=job['id']
        assert row['run']==rid and row['recipe']==args['recipe'] and row['training_seed']==args['seed']
        assert row['environment']==args['env'] and row['step']==step and row['policy_mode']==mode
        assert row['n_episodes']==episodes and row['first_episode_seed']==42 and row['eval_device']=='cpu' and row['no_optimizer_updates'] is True
        protocol='last3_cpu50_independent_seeded_episodes_v1' if episodes==50 else 'auxiliary_cpu10_100K300K_screening_v1'
        assert row['protocol']==protocol
        assert row['checkpoint_sha256']==pin['checkpoint_sha256'] and row['actor_artifact_sha256']==pin['actor_artifact_sha256']
        assert row['config_sha256']==self.configs[rid]['config_sha256']
        assert row['source_sha256']==self.plan['all_source_sha256'][str(implementation(args['recipe']).evaluator_path)]
        assert row['core_source_sha256']==training_sources(self.plan,args['recipe'])[str(implementation(args['recipe']).core_path)]
        if mode=='primary':assert row['learned_model_trees_loaded']==['actor_params'] and row['deployment_metadata']==pin['metadata']
        assert all(len(row[name])==episodes for name in ('episode_successes','episode_rewards','episode_lengths'))
        assert all(value in (0,1)for value in row['episode_successes'])
        assert all(math.isfinite(float(value))for name in ('episode_rewards','episode_lengths')for value in row[name])
        assert all(float(value).is_integer()and value>0 for value in row['episode_lengths'])
        successes=sum(int(value)for value in row['episode_successes']);score=Fraction(successes,episodes)
        assert abs(float(score)-row['success_rate'])<1e-12
        assert abs(statistics.mean(row['episode_rewards'])-row['reward_mean'])<1e-8
        return dict(successes=successes,episodes=episodes,success_rate=float(score),sha256=sha(path),
            shared_machine_CPU_action_ms=row['diagnostic_shared_machine_CPU_action_ms_median'])


def pilot_candidates(rows,recipes,plan):
    candidates=[];limits=plan['thresholds']
    for recipe in recipes:
        chosen=[next((r for r in rows if r['recipe']==recipe and r['environment']==env and r['seed']==0),None)for env in plan['pilot_environments']]
        if not all(r and r['complete']for r in chosen):continue
        score=statistics.mean(r['primary_mean']for r in chosen);teacher=statistics.mean(r['teacher_mean']for r in chosen)
        paired_losses=[r['teacher_mean']-r['primary_mean']for r in chosen]
        historical_losses=[plan['pilot_reference'][r['environment']]-r['primary_mean']for r in chosen]
        eligible=teacher-score<=limits['mean_loss_tolerance']+1e-12
        eligible=eligible and score>=limits['pilot_historical_mean_reference']-limits['mean_loss_tolerance']-1e-12
        candidates.append(dict(recipe=recipe,eligible=eligible,pilot_primary_mean=score,pilot_teacher_mean=teacher,
            paired_task_losses=paired_losses,historical_task_losses=historical_losses,
            actor_parameter_count=chosen[0]['actor_parameter_count'],objective=specification(recipe)['objective'],
            eligibility_scope='Macro averages govern selection; per-task losses are diagnostic.'))
    return candidates


def qualified_pick(candidates,tested):
    allowed=[r for r in candidates if r['eligible'] and r['recipe']not in tested]
    if not allowed:return None
    best=max(r['pilot_primary_mean']for r in allowed)
    close=[r for r in allowed if r['pilot_primary_mean']>=best-.01-1e-12]
    return sorted(close,key=lambda r:(r['actor_parameter_count'],r['objective']!='none',-r['pilot_primary_mean'],r['recipe']))[0]


def family(recipe):
    stem=recipe.split('__')[0]
    return 'fqlraw' if stem.startswith('fqlraw_x') else stem.rsplit('_a',1)[0]


def pick_candidate(candidates,tested):
    qualified=qualified_pick(candidates,tested)
    if qualified is not None:return dict(qualified,selection_kind='passed_pilot_macro_screen')
    explored=[row for row in candidates if row['recipe'] in tested and not row['eligible']]
    if len(explored)>=2:return None
    remaining=[row for row in candidates if row['recipe'] not in tested]
    if not remaining:return None
    used={family(row['recipe']) for row in explored}
    pool=[row for row in remaining if family(row['recipe']) not in used] or remaining
    picked=qualified_pick([dict(row,eligible=True) for row in pool],tested)
    original=next(row for row in pool if row['recipe']==picked['recipe'])
    return dict(original,selection_kind='exploratory_full9_despite_failed_pilot_screen',
        exploratory_promotion=True,reason='The full nine-task goal cannot be refuted by three pilot tasks alone.')


def select_available(candidates,tested,round_complete):
    qualified=qualified_pick(candidates,tested)
    if qualified:return dict(qualified,selection_kind='passed_complete_pilot_macro_as_available')
    return pick_candidate(candidates,tested)if round_complete else None


def full_confirmation(rows,attempt,plan):
    means=[];task_losses=[]
    for seed in [0,*attempt['fresh_seeds']]:
        chosen=[r for r in rows if r['recipe']==attempt['recipe']and r['seed']==seed]
        if len(chosen)!=9 or not all(r['complete']for r in chosen):return None
        assert {r['environment']for r in chosen}==set(core.ENVIRONMENTS)
        means.append(dict(seed=seed,primary=statistics.mean(r['primary_mean']for r in chosen),teacher=statistics.mean(r['teacher_mean']for r in chosen)))
    limits=plan['thresholds'];discovery=means[0];fresh=means[1:]
    fresh_mean=statistics.mean(r['primary']for r in fresh);fresh_teacher=statistics.mean(r['teacher']for r in fresh)
    fresh_rows=[r for r in rows if r['recipe']==attempt['recipe']and r['seed']in attempt['fresh_seeds']]
    for env in core.ENVIRONMENTS:
        group=[r for r in fresh_rows if r['environment']==env];assert len(group)==3
        task_losses.append(dict(environment=env,mean_loss=statistics.mean(r['teacher_mean']-r['primary_mean']for r in group)))
    checks=dict(seed0_paired_mean=discovery['primary']>=discovery['teacher']-limits['mean_loss_tolerance']-1e-12,
        seed0_historical_mean=discovery['primary']>=limits['historical_lean_seed0_nine_task_reference']-limits['mean_loss_tolerance']-1e-12,
        fresh_paired_mean=fresh_mean>=fresh_teacher-limits['mean_loss_tolerance']-1e-12,
        fresh_historical_mean=fresh_mean>=limits['historical_original_fresh_mean_reference']-limits['mean_loss_tolerance']-1e-12)
    return dict(recipe=attempt['recipe'],seed_means=means,fresh_mean=fresh_mean,fresh_teacher_mean=fresh_teacher,
        fresh_seed_sample_SD=statistics.stdev(r['primary']for r in fresh),fresh_task_losses=task_losses,
        checks=checks,performance_passed=all(checks.values()),
        diagnostic_task_loss_cap_satisfied=max(r['mean_loss']for r in task_losses)<=limits['max_task_loss']+1e-12,
        eligibility_scope='Nine-task means and fresh-seed macro confirmation; task regressions remain reported.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('contract',type=Path);args=parser.parse_args()
    contract=json.loads(args.contract.read_text());registration=Path(contract['registration'])
    assert sha(registration)==contract['registration_sha256'];plan=json.loads(registration.read_text());root=Path(plan['root'])
    assert plan['primary_steps']==list(STEPS) and plan['episodes']==50 and plan['first_episode_seed']==42
    assert plan['auxiliary_steps']==[100000,300000] and plan['auxiliary_episodes']==10
    assert plan['GPU_slots_per_device']==8 and float(plan['memory_fraction'])==.1
    assert set(plan['implementation_training_source_sha256'])=={'flat','set','mean_teacher','energy','mixture','latent','latent_guided'}
    assert plan['scheduling_amendment']['performance_gates_unchanged'] is True
    assert contract['GPU_scheduler_module']=='utils.flow_iql_research.scheduler_onestep_deploy_v7'
    output=root/'analysis/onestep_pipeline_v1';output.mkdir(exist_ok=True)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evdir=root/'analysis/evaluations_v1';evdir.mkdir(exist_ok=True);receipts=output/'registrations';receipts.mkdir(exist_ok=True)
    state_path=output/'research_state.json'
    if state_path.exists():state=json.loads(state_path.read_text())
    else:
        state=dict(jobs=plan['jobs'],identities=plan['expected_identities'],round=0,round_recipes=plan['initial_recipes'],
            all_recipes=plan['initial_recipes'],phase='pilot',tested_candidates=[],attempts=[],active_attempt=None,
            evaluation_workers={},evaluation_tries={},evaluation_failures={},benchmark_workers={},benchmark_results={},events=[])
        atomic(state_path,state)
    for job in state['jobs']:
        assert job['module']==implementation(job['args']['recipe']).runner_module
    assert all(attempt.get('stage')in ('discovery','fresh')for attempt in state['attempts'])
    verifier=Verifier(plan);policy_verifier=PolicyVerifier(plan,verifier,sha);workers={};benchmarks={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.1',MUJOCO_GL='egl')
    def persist():atomic(state_path,state)
    def result_path(rid,step,mode,episodes):return evdir/f'{rid}_{step}_{mode}_n{episodes}.json'
    def evaluate_key(rid,step,mode,episodes):return f'{rid}:{step}:{mode}:{episodes}'
    # Ownership-checked adoption makes restarting analysis safe.
    for key,row in list(state['evaluation_workers'].items()):
        if process_identity(row['pid'])==row['process_identity']:
            command=Path(f'/proc/{row["pid"]}/cmdline').read_bytes().decode().split('\0')
            job=next(j for j in state['jobs']if j['id']==row['run'])
            evaluator=implementation(POLICY if row['mode']==MODE else job['args']['recipe']).evaluator_module
            assert evaluator in command and str(root/'runs'/row['run'])in command
            assert set(os.sched_getaffinity(row['pid']))==set(plan['CPUslots'][row['slot']])
            workers[key]=dict(process=None,log=None,**row)
        else:del state['evaluation_workers'][key]
    for gpu,row in list(state['benchmark_workers'].items()):
        if process_identity(row['pid'])==row['process_identity']:
            command=Path(f'/proc/{row["pid"]}/cmdline').read_bytes().decode().split('\0')
            assert implementation(state['attempts'][state['active_attempt']]['recipe']).benchmark_module in command
            benchmarks[gpu]=dict(process=None,log=None,**row)
        else:
            path=Path(row['output'])
            if path.exists():
                data=json.loads(path.read_text())
                assert data['status']=='complete','Interrupted benchmark requires a versioned recovery, not an inferred timing'
                state['benchmark_results'][gpu]=data
            del state['benchmark_workers'][gpu]
    persist()

    def launch_evaluation(job,step,mode,episodes,slot):
        rid=job['id'];key=evaluate_key(rid,step,mode,episodes)
        trial=state['evaluation_tries'].get(key,0)+1;state['evaluation_tries'][key]=trial
        log=(evdir/f'{rid}_{step}_{mode}_n{episodes}_try{trial}.log').open('a')
        command=['nice','-n','10','taskset','-c',','.join(map(str,plan['CPUslots'][slot])),sys.executable,'-u','-m']
        if mode==MODE:
            assert episodes==50
            command += [implementation(POLICY).evaluator_module,'--run',str(root/'runs'/rid),'--step',str(step),
                '--actor-output',str(actor_path(root,rid,step)),'--output',str(result_path(rid,step,mode,episodes))]
        else:
            command += [implementation(job['args']['recipe']).evaluator_module,'--run',str(root/'runs'/rid),'--step',str(step),
                '--mode',mode,'--episodes',str(episodes),'--output',str(result_path(rid,step,mode,episodes))]
        p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        record=dict(pid=p.pid,process_identity=process_identity(p.pid),run=rid,step=step,mode=mode,episodes=episodes,slot=slot,trial=trial)
        workers[key]=dict(process=p,log=log,**record);state['evaluation_workers'][key]=record;persist()

    def append_jobs(new,expected,label,details):
        receipt=receipts/f'{label}.json';assert not receipt.exists()
        with (root/'queue_mutation.lock').open('a') as queue_lock:
            fcntl.flock(queue_lock,fcntl.LOCK_EX);queue=json.loads((root/'queue.json').read_text());old_ids={j['id']for j in queue}
            assert not any(j['id']in old_ids for j in new)
            atomic(receipt,dict(time=time.time(),source_contract_sha256=sha(args.contract),new_jobs=new,expected_identities=expected,details=details))
            state['jobs']+=new;state['identities'].update(expected);state['events'].append(dict(time=time.time(),label=label,jobs=len(new)));persist()
            atomic(root/'queue.json',queue+new)
        prioritize_pending()

    def started_jobs():
        scheduler=json.loads((root/'scheduler_status.json').read_text())
        finished=json.loads((root/'finished.json').read_text())if(root/'finished.json').exists()else{}
        return set(scheduler['running'])|set(finished)|{j['id']for j in state['jobs']if(root/'runs'/j['id']/'config.json').exists()}

    def prioritize_pending():
        with(root/'queue_mutation.lock').open('a')as queue_lock:
            fcntl.flock(queue_lock,fcntl.LOCK_EX);queue=json.loads((root/'queue.json').read_text())
            started=started_jobs();ordered=order_pending(plan,state,queue,started)
            if ordered==queue:return
            number=state.get('queue_reorder_count',0);receipt=receipts/f'pending_order{number}.json';assert not receipt.exists()
            atomic(receipt,dict(time=time.time(),source_contract_sha256=sha(args.contract),before=[j['id']for j in queue],after=[j['id']for j in ordered],started=sorted(started),rule=plan['scheduling_amendment']))
            state['queue_reorder_count']=number+1;state['events'].append(dict(time=time.time(),label='registered_pending_only_order',receipt=str(receipt)));persist()
            atomic(root/'queue.json',ordered)

    pause_path=root/'PAUSE_NEW_LAUNCHES.json'
    def set_benchmark_pause(attempt):
        value=dict(attempt=attempt['number'],execution_contract_sha256=sha(args.contract),reason='Drain existing learners for isolated nine-task CPU/GPU benchmark; goal not yet complete.')
        if pause_path.exists():assert json.loads(pause_path.read_text())==value
        else:atomic(pause_path,value)

    def clear_benchmark_pause(attempt):
        if pause_path.exists():
            value=json.loads(pause_path.read_text());assert value['attempt']==attempt['number']and value['execution_contract_sha256']==sha(args.contract)
            pause_path.unlink()

    def add_configuration(new,expected,recipe,environment,seed,prefix):
        recipe=training_recipe(recipe)
        if any(j['args']['recipe']==recipe and j['args']['env']==environment and j['args']['seed']==seed for j in state['jobs']+new):return
        tag=['ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene'][core.ENVIRONMENTS.index(environment)]
        rid=f'{prefix}_{tag}_{recipe}_s{seed}'
        new.append(dict(id=rid,module=implementation(recipe).runner_module,args=dict(env=environment,recipe=recipe,seed=seed,steps=1000000)))
        expected[rid]=configuration(environment,recipe,seed)

    def promote(candidate):
        number=len(state['attempts']);fresh=[12+3*number+i for i in range(3)]
        attempt=dict(number=number,recipe=candidate['recipe'],fresh_seeds=fresh,pilot_selection=candidate,status='running')
        if candidate['recipe']==POLICY:
            attempt.update(native_training_recipe=NATIVE,deployment_rule='sampled_component_location',
                training_identity_unchanged=True)
        state['attempts'].append(attempt);state['active_attempt']=number;state['phase']='confirmation';state['benchmark_results']={};state['benchmark_workers']={}
        attempt['stage']='discovery';attempt['status']='discovery_running'
        new=[];expected={}
        for seed in [0]:
            for environment in core.ENVIRONMENTS:add_configuration(new,expected,candidate['recipe'],environment,seed,f'osc{number}')
        append_jobs(new,expected,f'confirmation_attempt{number}',attempt)

    def begin_fresh(attempt):
        attempt['stage']='fresh';attempt['status']='fresh_running';new=[];expected={}
        for seed in attempt['fresh_seeds']:
            for environment in core.ENVIRONMENTS:add_configuration(new,expected,attempt['recipe'],environment,seed,f'osc{attempt["number"]}')
        append_jobs(new,expected,f'confirmation_attempt{attempt["number"]}_fresh',attempt)

    def expand(candidates):
        state['round']+=1;number=state['round']
        if number==1:recipes=list(plan['second_round_recipes'])
        else:
            ranked=sorted(candidates,key=lambda r:-r['pilot_primary_mean'])
            stems=[]
            for row in ranked:
                stem=row['recipe'].split('__')[0]
                family=stem.rsplit('_a',1)[0] if '_a'in stem else stem
                if family not in stems:stems.append(family)
                if len(stems)==4:break
            proposed=[]
            for family in stems:
                if family.startswith(('set_','energy_','mdn9_')):continue  # Only explicitly validated recipes; no inferred architecture suffixes.
                if family.endswith('_bc'):
                    proposed.extend(family+suffix for suffix in ('__w2','__d6','__ln','__current','__w2__ln','__w2__d6'))
                elif family in ('fqln','bank_q','state_q','bank_value','bank_regret'):
                    proposed.extend(family+'_a'+alpha for alpha in ('0003','001','003','01','03','1','3','10','30','100','300','1000'))
                    proposed.extend(family+'_a1'+suffix for suffix in ('__ln','__w2','__current','__d6','__spreadQ'))
            proposed.extend(['bank_bc__w2__d6','bank_bc__w3','bank_bc__d8','state_bc__w2','state_bc__ln'])
            recipes=[]
            for name in proposed:
                if name not in state['all_recipes'] and name not in recipes:
                    config_for(plan['pilot_environments'][0],name);recipes.append(name)
                if len(recipes)==16:break
        recipes=[name for name in recipes if name not in state['all_recipes']]
        if not recipes:
            state['phase']='research_decision_needed'
            atomic(output/'NEXT_RESEARCH_DECISION_NEEDED.json',dict(time=time.time(),reason='Registered scalar/architecture variants exhausted; new loss/density actor requires a versioned implementation.',active_goal_not_complete=True,candidates=candidates))
            persist();return
        state['round_recipes']=recipes;state['all_recipes']+=recipes;state['phase']='pilot'
        new=[];expected={}
        for recipe in recipes:
            for environment in plan['pilot_environments']:add_configuration(new,expected,recipe,environment,0,f'osa{number}')
        append_jobs(new,expected,f'adaptive_round{number}',dict(recipes=recipes,basis=candidates,rule=plan['adaptive_search']))

    extension=plan['routed_deployment_extension']
    if 'routed_deployment_activation' not in state:
        # This is a registered change in candidate selection, never a failed-
        # performance claim for the retained default-MDN control.
        assert state['phase']=='confirmation' and state['active_attempt']==extension['superseded_attempt']
        old=state['attempts'][state['active_attempt']]
        assert old['recipe']==NATIVE and old['stage']=='discovery'
        assert not any(j['args']['seed'] in old['fresh_seeds'] for j in state['jobs'])
        receipt=receipts/'routed_deployment_candidate_activation_v1.json'
        with (root/'queue_mutation.lock').open('a') as queue_lock:
            fcntl.flock(queue_lock,fcntl.LOCK_EX)
            if receipt.exists():
                transition=json.loads(receipt.read_text())
                assert transition['source_contract_sha256']==sha(args.contract)
            else:
                transition=dict(time=time.time(),source_contract_sha256=sha(args.contract),
                    previous_attempt=dict(old),new_policy_candidate=POLICY,native_training_recipe=NATIVE,
                    no_job_identity_or_queue_changes=True,no_fresh_allocation=True,
                    reason='Prefer the globally fixed component-location decoder after four-task same-model discovery. Default-MDN full-nine results remain reported controls; no failed-performance claim.')
                atomic(receipt,transition)
            old.update(status='superseded_by_registered_decoder_candidate',
                no_fresh_training_run_allocated=True,full9_results_remain_control=True,
                supersession_receipt=str(receipt))
            if NATIVE not in state['tested_candidates']:state['tested_candidates'].append(NATIVE)
            if POLICY not in state['all_recipes']:state['all_recipes'].append(POLICY)
            state['phase']='pilot';state['routed_deployment_activation']=transition
            persist()

    while True:
        try:
            assert sha(registration)==contract['registration_sha256']
            for path,digest in {**plan['all_source_sha256'],**contract['source_sha256']}.items():assert sha(path)==digest,path
            # Repair only an interrupted atomic registration-before-queue transaction.
            queue=json.loads((root/'queue.json').read_text());known={j['id']for j in queue}
            registered={job['id']:job for job in state['jobs']}
            assert all(job['id'] in registered and job==registered[job['id']] for job in queue)
            missing=[j for j in state['jobs']if j['id']not in known]
            if missing:
                with (root/'queue_mutation.lock').open('a')as queue_lock:
                    fcntl.flock(queue_lock,fcntl.LOCK_EX);queue=json.loads((root/'queue.json').read_text());known={j['id']for j in queue}
                    atomic(root/'queue.json',queue+[j for j in missing if j['id']not in known])
            for key,row in list(workers.items()):
                alive=row['process'].poll()is None if row['process']is not None else process_identity(row['pid'])==row['process_identity']
                if alive:continue
                if row['log']:row['log'].close()
                path=result_path(row['run'],row['step'],row['mode'],row['episodes'])
                if not path.exists() and row['trial']>=2:state['evaluation_failures'][key]=dict(pid=row['pid'],trial=row['trial'],reason='worker_failed_no_result')
                del workers[key];state['evaluation_workers'].pop(key,None);persist()
            for gpu,row in list(benchmarks.items()):
                alive=row['process'].poll()is None if row['process']is not None else process_identity(row['pid'])==row['process_identity']
                if alive:continue
                if row['log']:row['log'].close()
                data=json.loads(Path(row['output']).read_text());assert data['status']=='complete'
                assert data['source_sha256']==plan['all_source_sha256'][str(implementation(state['attempts'][state['active_attempt']]['recipe']).benchmark_path)]
                state['benchmark_results'][gpu]=data;state['benchmark_workers'].pop(gpu,None);del benchmarks[gpu];persist()
            prioritize_pending()
            finished=json.loads((root/'finished.json').read_text())if(root/'finished.json').exists()else{}
            scheduler_status=json.loads((root/'scheduler_status.json').read_text())
            manager_pid=scheduler_status['pid']
            if process_identity(manager_pid) is None and not(root/'STOP').exists():
                log=(output/'GPU_scheduler_restarts.log').open('a')
                command=['taskset','-c',','.join(map(str,contract['GPU_manager_CPU_affinity'])),sys.executable,'-u','-m',
                    contract['GPU_scheduler_module'],str(root),'--per-gpu','8']
                manager=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                log.close();state['events'].append(dict(time=time.time(),label='restart_owned_GPU_scheduler',pid=manager.pid,process_identity=process_identity(manager.pid)));persist()
            rows=[];ready_late=[];ready_early=[];evaluations={}
            for job in state['jobs']:
                rid=job['id'];run=root/'runs'/rid;row=dict(run=rid,recipe=job['args']['recipe'],environment=job['args']['env'],seed=job['args']['seed'],
                    complete=False,terminal=False,training_complete=False,primary_mean=None,teacher_mean=None,status='pending')
                failure=finished.get(rid,{}).get('result')
                if failure and failure!='complete':row.update(status='training_failed',terminal=True,failure=finished[rid]);rows.append(row);continue
                if not(run/'config.json').exists():rows.append(row);continue
                conf=verifier.config(run,state['identities'][rid])
                if conf is None:row['status']='initializing';rows.append(row);continue
                done=(run/'completed.json').exists();stopped=(run/'stopped.json').exists()
                probes=partial_records(run/'diagnostics.jsonl',done)
                assert all(p['finite']and p['q_updates']==p['actor_updates']==p['step']and p['v_updates']==0 for p in probes)
                if done:assert len(probes)==1000 and probes[-1]['step']==1000000
                row.update(training_complete=done,last_probe=probes[-1]if probes else None,observed_probe_count=len(probes),status='evaluating'if done else 'training')
                if stopped:row.update(status='training_nonfinite',terminal=True);rows.append(row);continue
                counts={'primary':[],'teacher':[]};scores={'primary':{},'teacher':{}};early={}
                for step in (*STEPS,*plan['auxiliary_steps']):
                    if not(run/f'checkpoint_{step}.pkl').exists()or not(run/f'actor_{step}.pkl').exists():continue
                    pin=verifier.model(run,conf,step);row['actor_parameter_count']=pin['actor_parameter_count']
                    episodes=50 if step in STEPS else 10
                    for mode in (('primary','teacher')if episodes==50 else ('primary',)):
                        key=evaluate_key(rid,step,mode,episodes);path=result_path(rid,step,mode,episodes)
                        if path.exists():
                            value=verifier.evaluation(path,job,step,mode,episodes,pin);evaluations[key]=value
                            if episodes==50:counts[mode].append(value['successes']);scores[mode][str(step)]=value['success_rate']
                            else:early[str(step)]=value['success_rate']
                        elif key not in workers and key not in state['evaluation_failures']:
                            (ready_late if episodes==50 else ready_early).append((job,step,mode,episodes))
                row.update(per_step=scores,auxiliary_cpu10=early)
                if any(k.startswith(rid+':')and k.endswith(':50')and f':{MODE}:'not in k for k in state['evaluation_failures']):row.update(status='evaluation_failed',terminal=True)
                elif done and all(len(counts[m])==3 for m in ('primary','teacher')):
                    row.update(complete=True,terminal=True,status='complete',primary_mean=float(Fraction(sum(counts['primary']),150)),teacher_mean=float(Fraction(sum(counts['teacher']),150)))
                rows.append(row)
            policy_rows,ready_policy,policy_evaluations=derived_rows(root,state['jobs'],rows,state,
                verifier,policy_verifier,result_path,evaluate_key)
            rows+=policy_rows;evaluations.update(policy_evaluations)
            ready=ready_late+ready_policy+ready_early;occupied={r['slot']for r in workers.values()}
            for slot in range(len(plan['CPUslots'])):
                if slot not in occupied and ready:launch_evaluation(*ready.pop(0),slot)
            candidates=pilot_candidates(rows,state['all_recipes'],plan)
            round_rows=[r for r in rows if r['recipe']in state['round_recipes']and r['seed']==0 and r['environment']in plan['pilot_environments']]
            round_complete=len(round_rows)==3*len(state['round_recipes'])and all(r['terminal']for r in round_rows)
            if state['phase']=='confirmation':
                active=state['attempts'][state['active_attempt']]
                if active['stage']=='discovery':
                    # Routed policy has separate result/actor identities. Its
                    # full nine-task gate is mandatory; native futility logic
                    # remains unchanged for all original candidates.
                    bound=None if active['recipe']==POLICY else empirical_bound(rows,active['recipe'],plan)
                    if bound and bound['impossible_even_if_all_missing_perfect']:
                        decision_path=output/f'bounded_discovery_attempt{active["number"]}_v1.json'
                        if decision_path.exists():
                            decision=json.loads(decision_path.read_text())
                            assert decision['attempt']==active['number']and decision['recipe']==active['recipe']
                            assert decision['bound']['unchanged_historical_minimum']==bound['unchanged_historical_minimum']
                            assert decision['bound']['impossible_even_if_all_missing_perfect']
                            for evidence in decision['raw_evidence']:assert sha(evidence['path'])==evidence['sha256']
                        else:
                            evidence=[]
                            for point in bound['known_points']:
                                path=result_path(point['run'],point['step'],'primary',50)
                                evidence.append(dict(path=str(path),sha256=sha(path)))
                            decision=dict(time=time.time(),attempt=active['number'],recipe=active['recipe'],
                                bound=bound,raw_evidence=evidence,execution_contract_sha256=sha(args.contract),
                                no_learner_or_evaluator_signals=True,no_gate_or_job_changes=True,
                                reason='Even perfect remaining fixed empirical points cannot meet the unchanged historical macro gate. All learners and evaluations continue.')
                            atomic(decision_path,decision)
                        active.update(status='discovery_historical_gate_impossible',early_failure_bound=decision,
                                      no_fresh_training_run_allocated=True)
                        if active['recipe']not in state['tested_candidates']:state['tested_candidates'].append(active['recipe'])
                        state['phase']='pilot'
                        state['events'].append(dict(time=time.time(),label='verified_fixed_metric_discovery_futility',receipt=str(decision_path)))
                        persist()
            if state['phase']=='pilot':
                selected=preferred_candidate(state,plan,candidates)or select_available(candidates,state['tested_candidates'],round_complete)
                if selected:
                    atomic(output/f'available_selection_attempt{len(state["attempts"])}.json',dict(time=time.time(),round_complete=round_complete,candidates=candidates,selected=selected))
                    promote(selected)
                elif round_complete:expand(candidates)
            confirmation=None;speed=None
            if state['phase']in ('confirmation','benchmark'):
                attempt=state['attempts'][state['active_attempt']]
                failed=[r for r in rows if r['recipe']==attempt['recipe']and r['seed']in [0,*attempt['fresh_seeds']]and r['terminal']and not r['complete']]
                if failed:
                    attempt.update(status='confirmation_worker_failed',failed_runs=[r['run']for r in failed])
                    state['tested_candidates'].append(attempt['recipe']);state['phase']='pilot';persist()
                    atomic(output/f'confirmation_attempt{attempt["number"]}_worker_failure.json',dict(time=time.time(),rows=failed))
                    continue
                if attempt['stage']=='discovery':
                    discovery=discovery_confirmation(rows,attempt,plan,core.ENVIRONMENTS)
                    if discovery is not None:
                        attempt['discovery']=discovery
                        if not discovery['discovery_passed']:
                            attempt.update(status='discovery_performance_failed');state['tested_candidates'].append(attempt['recipe']);state['phase']='pilot';persist()
                        else:begin_fresh(attempt)
                confirmation=full_confirmation(rows,attempt,plan)if attempt['stage']=='fresh'else None
                if confirmation is not None and not confirmation['performance_passed']:
                    attempt.update(status='performance_failed',confirmation=confirmation);state['tested_candidates'].append(attempt['recipe']);state['phase']='pilot';persist()
                elif confirmation is not None:
                    state['phase']='benchmark';set_benchmark_pause(attempt);persist()
                    scheduler=json.loads((root/'scheduler_status.json').read_text())
                    GPU_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                    if not scheduler['running']and not GPU_pids and not benchmarks and not state['benchmark_results']:
                        folder=output/f'benchmark_attempt{attempt["number"]}';folder.mkdir(exist_ok=False)
                        chosen=[r for r in rows if r['recipe']==attempt['recipe']and r['seed']==0];assert len(chosen)==9
                        for gpu in (0,1):
                            input_path=folder/f'GPU{gpu}_input.json';atomic(input_path,dict(jobs=[dict(run=str(root/'runs'/r['run']))for r in chosen[gpu::2]]))
                            path=folder/f'GPU{gpu}_timings.json';log=(folder/f'GPU{gpu}.log').open('a')
                            command=['taskset','-c',f'{6*gpu}-{6*gpu+5}',sys.executable,'-u','-m',implementation(attempt['recipe']).benchmark_module,'--input',str(input_path),'--output',str(path)]
                            p=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                            record=dict(pid=p.pid,process_identity=process_identity(p.pid),output=str(path));benchmarks[str(gpu)]=dict(process=p,log=log,**record);state['benchmark_workers'][str(gpu)]=record;persist()
                    if len(state['benchmark_results'])==2:
                        speed_rows=[r for d in state['benchmark_results'].values()for r in d['rows']];assert len(speed_rows)==9
                        assert {r['environment']for r in speed_rows}==set(core.ENVIRONMENTS)
                        for r in speed_rows:
                            source_pins=policy_verifier.pins if attempt['recipe']==POLICY else verifier.pins
                            pin=source_pins[f'{r["run"]}:1000000'];assert r['checkpoint_sha256']==pin['checkpoint_sha256']and r['actor_artifact_sha256']==pin['actor_artifact_sha256']
                        speed={backend+'_latency_ratio':math.exp(statistics.mean(math.log(r[backend+'_latency_ratio'])for r in speed_rows))for backend in ('cpu','gpu')}
                        speed['passed']=speed['cpu_latency_ratio']<=plan['thresholds']['CPU_latency_ratio_max']and speed['gpu_latency_ratio']<=plan['thresholds']['GPU_latency_ratio_max']
                        attempt.update(confirmation=confirmation,speed=speed,status='success_pending_root_review'if speed['passed']else 'speed_failed')
                        if speed['passed']:
                            state['phase']='success_pending_root_review'
                            atomic(output/'GOAL_ACHIEVED_AUDIT_READY.json',dict(time=time.time(),attempt=attempt,registration_sha256=sha(registration),execution_contract_sha256=sha(args.contract)))
                        else:
                            clear_benchmark_pause(attempt);state['tested_candidates'].append(attempt['recipe']);state['phase']='pilot'
                        persist()
            result=dict(time=time.time(),pid=os.getpid(),phase=state['phase'],round=state['round'],planned_GPU_jobs=len(state['jobs']),
                complete_primary_and_teacher_runs=sum(r['complete'] for r in rows if r['recipe']!=POLICY),
                complete_routed_policy_runs=sum(r['complete'] for r in policy_rows),rows=rows,pilot_candidates=candidates,attempts=state['attempts'],
                verified_models=verifier.pins,valid_evaluations=evaluations,CPU_running=state['evaluation_workers'],benchmark_running=state['benchmark_workers'],
                verified_routed_policy_models=policy_verifier.pins,
                evaluation_failures=state['evaluation_failures'],events=state['events'],thresholds=plan['thresholds'],metric=plan['metric'],
                registration_sha256=sha(registration),execution_contract_sha256=sha(args.contract),active_goal_not_marked_complete=True)
            atomic(output/'status.json',result)
            buffer=io.StringIO();writer=csv.writer(buffer);writer.writerow(['recipe','environment','seed','status','800K','900K','1M','primary_last3','teacher_last3','primary_minus_teacher'])
            for row in rows:
                writer.writerow([row['recipe'],row['environment'],row['seed'],row['status'],*[row.get('per_step',{}).get('primary',{}).get(str(s))for s in STEPS],
                    row['primary_mean'],row['teacher_mean'],row['primary_mean']-row['teacher_mean']if row['complete']else None])
            temp=output/'results.csv.tmp';temp.write_text(buffer.getvalue());temp.replace(output/'results.csv')
            if state['phase']=='success_pending_root_review'and not workers and not benchmarks:
                for pin in [*verifier.pins.values(),*policy_verifier.pins.values()]:assert sha(pin['path'])==pin['checkpoint_sha256']and sha(pin['actor_path'])==pin['actor_artifact_sha256']
                atomic(output/'final_audit.json',result);(root/'STOP').write_text('Registeredone-stepgoalpracticalcriteriapassed;rootindependentreviewrequired.\n');return
        except Exception as error:
            atomic(output/'controller_failure.json',dict(time=time.time(),error=repr(error),models_and_results_preserved=True));raise
        time.sleep(20)


if __name__=='__main__':main()
