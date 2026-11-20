"""Finish the existing three expectile-weighted-flow pilots without new knobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def tree_equal(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def array_differences(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        values=[array_differences(a[k],b[k]) for k in a]
        return dict(arrays=sum(v['arrays'] for v in values),
            max_abs=max(v['max_abs'] for v in values),exact=all(v['exact'] for v in values))
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape
    return dict(arrays=1,max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
        exact=bool(np.array_equal(a,b)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    args=parser.parse_args();plan=json.loads(args.registration.read_text())
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache={};pins={};root=Path(plan['root'])

    def model(run_id,step):
        key=f'{run_id}:{step}'
        if key not in cache:
            conf,saved,path,sha=helper.load_parent(plan,run_id,step)
            cache[key]=(conf,saved)
            stat=path.stat();pins[key]=dict(path=str(path),sha256=sha,
                stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns])
        pin=pins[key];stat=Path(pin['path']).stat()
        assert pin['stamp']==[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
        return cache[key]

    while True:
        helper.check_sources(plan['source_sha256']);rows=[]
        for pair in plan['pairs']:
            weighted=root/'runs'/pair['weighted_run'];control=root/'runs'/pair['control_run']
            per_step={};comparisons={}
            for step in helper.STEPS:
                cp=Path(pair['control_evaluations'][str(step)])
                record=json.loads(cp.read_text());helper.check_episodes(record,step)
                assert helper.digest(cp)==plan['control_evaluation_sha256'][str(cp)]
                assert record['selection_samples']==9 and record['selection_rule']=='max'
                assert record['policy_source_sha256']==plan['source_sha256']['utils/flow_iql_research/selection_v2.py']
                conf,saved_control=model(pair['control_run'],step)
                assert record['run']==pair['control_run'] and record['config_sha256']==helper.digest(control/'config.json')
                assert record['checkpoint_sha256']==pins[f"{pair['control_run']}:{step}"]['sha256']
                new=helper.primary_record(weighted,step)
                if new is not None:
                    newconf,saved_weighted=model(pair['weighted_run'],step)
                    net=saved_weighted['state']['agent']['network'];base=saved_control['state']['agent']['network']
                    comparisons[str(step)]=dict(
                        replay_rng_exact=tree_equal(saved_weighted['replay_rng'],saved_control['replay_rng']),
                        agent_rng_exact=tree_equal(saved_weighted['state']['agent']['rng'],saved_control['state']['agent']['rng']),
                        modules={module:array_differences(net['params'][module],base['params'][module])
                            for module in ('modules_critic','modules_target_critic','modules_value')})
                per_step[str(step)]=dict(unweighted_BC_N9=record['success_rate'],
                    expectile_flow_N9=None if new is None else new['success_rate'])
            complete=False
            if (weighted/'completed.json').exists():
                done=json.loads((weighted/'completed.json').read_text())
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
                complete=all(v['expectile_flow_N9'] is not None for v in per_step.values())
            baseline=statistics.mean(v['unweighted_BC_N9'] for v in per_step.values())
            proposed=statistics.mean(v['expectile_flow_N9'] for v in per_step.values()) if complete else None
            rows.append(dict(environment=pair['environment'],weighted_run=pair['weighted_run'],
                control_run=pair['control_run'],complete=complete,per_step=per_step,
                control_last3_mean=baseline,weighted_last3_mean=proposed,
                delta=None if proposed is None else proposed-baseline,Q_V_trajectory_comparisons=comparisons))
        complete=all(r['complete'] for r in rows)
        helper.atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),status='complete' if complete else 'running',
            rows=rows,completed_tasks=sum(r['complete'] for r in rows),
            three_task_control_mean=statistics.mean(r['control_last3_mean'] for r in rows),
            three_task_weighted_mean=statistics.mean(r['weighted_last3_mean'] for r in rows) if complete else None,
            actual_checkpoint_pins=pins,registration_sha256=helper.digest(args.registration),
            no_new_CPU_evaluations=True,no_new_GPU_jobs=True,goal_complete=False,limits=plan['limits']))
        if complete:return
        time.sleep(30)


if __name__=='__main__':main()
