"""Audit unit-return scoring pilots against the paired raw-return control."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)is type(b)and len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    args=parser.parse_args();plan=json.loads(args.registration.read_text())
    output=Path(plan['analysis_output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path(plan['root']);cache={};pins={}
    def model(run_id,step):
        key=f'{run_id}:{step}';path=root/'runs'/run_id/f'checkpoint_{step}.pkl'
        stat=path.stat();stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
        if key not in cache or pins[key]['stamp']!=stamp:
            conf,saved,actual,sha=helper.load_parent(plan,run_id,step)
            if key in pins:assert pins[key]['sha256']==sha
            pins[key]=dict(path=str(actual),sha256=sha,stamp=stamp);cache[key]=saved
        return cache[key]
    while True:
        helper.check_sources(plan['analysis_source_sha256']);rows=[]
        for run_id in plan['all_run_ids']:
            run=root/'runs'/run_id;identity=plan['expected_identities'][run_id]
            probes=[json.loads(x)for x in(run/'diagnostics.jsonl').read_text().splitlines()]if(run/'diagnostics.jsonl').exists()else[]
            assert all(p['finite']for p in probes)
            per_step={}
            for step in helper.STEPS:
                record=helper.primary_record(run,step)
                if record is not None:model(run_id,step)
                per_step[str(step)]=None if record is None else record['success_rate']
            complete=False
            if(run/'completed.json').exists():
                done=json.loads((run/'completed.json').read_text())
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
                assert probes[-1]['step']==1000000
                complete=all(v is not None for v in per_step.values())
            late=[p for p in probes if p['step']>800000]
            selected_metrics={}
            for metric in plan['diagnostic_metrics']:
                available=[p for p in late if metric in p]
                if available:
                    total=sum(p['metric_window_updates']for p in available)
                    selected_metrics[metric]=sum(p[metric]*p['metric_window_updates']for p in available)/total
            rows.append(dict(run_id=run_id,environment=identity['environment'],profile=identity['profile'],
                complete=complete,per_step=per_step,last3_mean=statistics.mean(per_step.values())if complete else None,
                stopped=json.loads((run/'stopped.json').read_text())if(run/'stopped.json').exists()else None,
                last_training_step=probes[-1]['step']if probes else None,
                late_training_metrics=selected_metrics))
        byid={r['run_id']:r for r in rows};paired=[]
        for pair in plan['pairs']:
            base=byid[pair['control']];items={mode:byid[pair[mode]]for mode in ('mse','kl')};checks={}
            for step in helper.STEPS:
                if all(f'{rid}:{step}'in cache for rid in (pair['control'],pair['mse'],pair['kl'])):
                    states={name:cache[f'{rid}:{step}']for name,rid in [('control',pair['control']),('mse',pair['mse']),('kl',pair['kl'])]}
                    ref=states['control'];ref_net=ref['state']['agent']['network']
                    checks[str(step)]={name:dict(
                        replay_rng_exact=equal(saved['replay_rng'],ref['replay_rng']),
                        agent_rng_exact=equal(saved['state']['agent']['rng'],ref['state']['agent']['rng']),
                        BC_params_exact=equal(saved['state']['agent']['network']['params']['modules_actor_flow'],ref_net['params']['modules_actor_flow']),
                        BC_Adam_mu_exact=equal(saved['state']['agent']['network']['opt_state']['0']['mu']['modules_actor_flow'],ref_net['opt_state']['0']['mu']['modules_actor_flow']),
                        BC_Adam_nu_exact=equal(saved['state']['agent']['network']['opt_state']['0']['nu']['modules_actor_flow'],ref_net['opt_state']['0']['nu']['modules_actor_flow']))
                        for name,saved in states.items()if name!='control'}
            delta=lambda a,b:None if a['last3_mean']is None or b['last3_mean']is None else a['last3_mean']-b['last3_mean']
            paired.append(dict(environment=pair['environment'],MSE_minus_raw=delta(items['mse'],base),
                KL_minus_raw=delta(items['kl'],base),KL_minus_MSE=delta(items['kl'],items['mse']),
                actual_BC_and_RNG_checks=checks))
        methods=[]
        for profile in ('bounded_v2_repeated_batches','bounded_v2_unit_mse','bounded_v2_unit_kl'):
            chosen=[r for r in rows if r['profile']==profile];assert len(chosen)==3
            complete=all(r['complete']for r in chosen)
            methods.append(dict(profile=profile,completed_tasks=sum(r['complete']for r in chosen),
                three_task_mean=statistics.mean(r['last3_mean']for r in chosen)if complete else None))
        complete=all(r['complete']for r in rows)
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),status='complete'if complete else 'running',
            rows=rows,methods=methods,paired_contrasts=paired,actual_checkpoint_pins=pins,
            registration_sha256=helper.digest(args.registration),additional_GPU_jobs=0,additional_policy_evaluations=0,
            goal_complete=False,limits=plan['analysis_limits']))
        if complete:return
        time.sleep(30)


if __name__=='__main__':main()
