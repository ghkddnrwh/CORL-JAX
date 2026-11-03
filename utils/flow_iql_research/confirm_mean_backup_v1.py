"""Conditionally confirm a uniform full9 mean-backup winner on fresh seeds.

Never launch from a three-task mean or partial/peak outcomes. Discovery and
fresh actual-model audits stay separate from the learned-V ensemble candidate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def qualifies(audit,threshold=.47):
    return (audit.get('status')=='complete' and audit.get('completed_tasks')==9
        and audit.get('actual_unique_late_checkpoints_verified')==27
        and audit.get('nine_task_mean') is not None and audit['nine_task_mean']>threshold)


def gate(discovery,means,threshold=.47):
    if len(means)!=3 or any(v is None for v in means):return None
    return (statistics.mean(means)>threshold and statistics.mean([discovery,*means])>threshold
        and sum(v>threshold for v in means)>=2)


def enqueue(plan,output):
    root=Path(plan['root']);lock=(root/'queue_mutation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=json.loads((root/'queue.json').read_text());byid={j['id']:j for j in queue}
    fresh=plan['confirmation_jobs'];existing=[j['id']in byid for j in fresh]
    assert all(existing) or not any(existing)
    if all(existing):
        assert all(byid[j['id']]==j for j in fresh)
        fcntl.flock(lock,fcntl.LOCK_UN);return
    finished=json.loads((root/'finished.json').read_text());status=json.loads((root/'scheduler_status.json').read_text())
    locked=set(finished)|set(status['running'])
    before=[j for j in queue if j['id']in locked]
    pending=[j for j in queue if j['id']not in locked]
    priority=lambda key:key.startswith(('vs1_','ht1_','ieconfirm_','meb1_','resume_phase3_ef1_'))
    next_queue=before+[j for j in pending if priority(j['id'])]+fresh+[j for j in pending if not priority(j['id'])]
    assert len(next_queue)==len(queue)+27 and len({j['id']for j in next_queue})==len(next_queue)
    assert all(next(j for j in next_queue if j['id']==old['id'])==old for old in queue)
    backup=output/'queue_before_promotion.json';assert not backup.exists();backup.write_bytes((root/'queue.json').read_bytes())
    helper.atomic(root/'queue.json',next_queue)
    helper.atomic(output/'queue_promotion_receipt.json',dict(time=time.time(),jobs_added=27,
        current_learners_preserved=status['running'],old_jobs_exact=True,
        queue_before_sha256=helper.digest(backup),queue_after_sha256=helper.digest(root/'queue.json'),
        scheduling='After pending locked research jobs, before older unstarted exploratory jobs.',
        no_training_source_edits=True,no_scheduler_restart=True))
    fcntl.flock(lock,fcntl.LOCK_UN)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    plan=json.loads(args.registration.read_text());helper.check_sources(plan['source_sha256'])
    if args.validate:
        assert len(plan['confirmation_jobs'])==len({j['id']for j in plan['confirmation_jobs']})==27
        good=dict(status='complete',completed_tasks=9,actual_unique_late_checkpoints_verified=27,nine_task_mean=.48)
        assert qualifies(good) and not qualifies({**good,'completed_tasks':8})
        assert not qualifies({**good,'nine_task_mean':.47}) and not qualifies({**good,'status':'incomplete'})
        assert gate(.5,[.5,None,.5]) is None and not gate(.5,[.4,.4,.6])
        assert gate(.5,[.48,.49,.5]) and not gate(.5,[.46,.46,.6])
        print(json.dumps(dict(status='passed',queue_mutations=0,new_training=0)),flush=True);return
    root=Path(plan['root']);output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None;discovery=None;index=0;pins={};promoted=False
    while True:
        helper.check_sources(plan['source_sha256'])
        assert helper.digest(plan['discovery_registration'])==plan['discovery_registration_sha256']
        discovery_plan=json.loads(Path(plan['discovery_registration']).read_text())
        fingerprint=[(j['id'],(root/'runs'/j['id']/'completed.json').exists())for j in discovery_plan['all9jobs']]
        if previous!=fingerprint:
            path=output/f'discovery_audit_{index}.json';index+=1
            command=[sys.executable,'-u','-m','utils.flow_iql_research.audit_mean_backup_full9_v1',
                plan['discovery_registration'],'--output',str(path)]
            with(output/f'discovery_audit_{index-1}.log').open('a')as log:
                code=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            assert code==0,'Discovery audit failed; inspect the preserved log.'
            discovery=json.loads(path.read_text());previous=fingerprint
        if qualifies(discovery,plan['threshold']) and not promoted:
            enqueue(plan,output);promoted=True
        rows=[]
        for job in plan['confirmation_jobs']:
            run_id=job['id'];run=root/'runs'/run_id;per_step={};stopped=None
            if(run/'stopped.json').exists():stopped=json.loads((run/'stopped.json').read_text())
            for step in helper.STEPS:
                record=helper.primary_record(run,step)
                if record is not None:
                    key=f'{run_id}:{step}';path=run/f'checkpoint_{step}.pkl';stat=path.stat()
                    stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
                    if key not in pins or pins[key]['stamp']!=stamp:
                        _,_,actual,sha=helper.load_parent(plan,run_id,step)
                        if key in pins:assert pins[key]['sha256']==sha
                        pins[key]=dict(path=str(actual),sha256=sha,stamp=stamp)
                per_step[str(step)]=None if record is None else record['success_rate']
            complete=False
            if(run/'completed.json').exists():
                done=json.loads((run/'completed.json').read_text())
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000 and not stopped
                complete=all(v is not None for v in per_step.values())
            rows.append(dict(run=run_id,seed=job['args']['seed'],environment=job['args']['env'],stopped=stopped,
                complete=complete,per_step=per_step,last3_mean=statistics.mean(per_step.values())if complete else None))
        seeds=[]
        for seed in (9,10,11):
            selected=[r for r in rows if r['seed']==seed];assert len(selected)==9
            seeds.append(dict(seed=seed,completed_tasks=sum(r['complete']for r in selected),
                nine_task_mean=statistics.mean(r['last3_mean']for r in selected)if all(r['complete']for r in selected)else None))
        means=[r['nine_task_mean']for r in seeds];complete=all(v is not None for v in means)
        passed=gate(discovery['nine_task_mean'],means)if promoted else None
        rejected=discovery['status']=='complete' and not qualifies(discovery,plan['threshold'])
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='discovery_did_not_qualify'if rejected else 'fresh_complete'if complete else 'fresh_running'if promoted else 'waiting_for_full9_discovery',
            discovery=discovery,promoted=promoted,confirmation_seeds=seeds,rows=rows,
            fresh_three_seed_mean=statistics.mean(means)if complete else None,
            fresh_seed_sample_SD=statistics.stdev(means)if complete else None,
            all_four_seed_mean=statistics.mean([discovery['nine_task_mean'],*means])if complete else None,
            independent_confirmation_criteria_passed=passed,actual_fresh_checkpoint_pins=pins,
            registration_sha256=helper.digest(args.registration),goal_complete=False,
            note='A passing flag still requires final root raw/source/model review; this controller never completes the tracked goal.'))
        if rejected or complete:return
        time.sleep(30)


if __name__=='__main__':main()
