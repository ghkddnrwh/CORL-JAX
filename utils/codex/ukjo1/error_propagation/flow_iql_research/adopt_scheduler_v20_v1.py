"""Add the registered6unit-return pilots while preserving live learner handles."""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.flow_iql_research import scheduler_v20, scheduler_v4
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z'else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None


def learners(root,queue,finished,previous):
    rows={}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        pid=int(proc.name)
        try:
            cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if '-m'not in cmd or '--output'not in cmd:continue
            module=cmd[cmd.index('-m')+1]
            if module not in scheduler_v4.MODULES:continue
            output=Path(cmd[cmd.index('--output')+1]).resolve()
            if output.parent!=root/'runs':continue
            start=identity(pid)
            if start is None:continue
            env=dict(x.split('=',1)for x in proc.joinpath('environ').read_bytes().decode().split('\0')if '='in x)
            assert env['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'
            candidates=[j for j in queue if j['id']not in finished and root/'runs'/j.get('output_run',j['id'])==output]
            assert len(candidates)==1,(pid,candidates)
            key=candidates[0]['id'];assert key not in rows
            old=previous.get(key,{})
            if old:assert old['pid']==pid and old['process_identity']==start
            rows[key]=dict(pid=pid,output=str(output),started_at=old.get('started_at',time.time()),
                process_identity=start,gpu=int(env['CUDA_VISIBLE_DEVICES']))
        except (FileNotFoundError,ProcessLookupError):continue
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('registration',type=Path);args=parser.parse_args();root=args.root.resolve()
    plan=json.loads(args.registration.read_text());assert Path(plan['root'])==root
    assert len(plan['jobs'])==6 and helper.digest('utils/flow_iql_research/scheduler_v20.py')==plan['scheduler_source_sha256']
    for source,digest in plan['analysis_source_sha256'].items():assert helper.digest(source)==digest
    for source,digest in next(iter(plan['expected_training_sources'].values())).items():assert helper.digest(source)==digest
    receipt_path=root/'analysis/scheduler_v20_adoption_v1.json';assert not receipt_path.exists()
    lock=(root/'queue_mutation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    before=json.loads((root/'scheduler_status.json').read_text());old=before['pid'];old_start=identity(old)
    assert old_start is not None
    cmd=Path(f'/proc/{old}/cmdline').read_bytes().decode().split('\0')
    assert cmd[cmd.index('-m')+1]=='utils.flow_iql_research.scheduler_v19' and str(root)in cmd
    queue=json.loads((root/'queue.json').read_text());finished=json.loads((root/'finished.json').read_text())
    actual=learners(root,queue,finished,before['running'])
    assert len(actual)==16 and collections.Counter(r['gpu']for r in actual.values())=={0:8,1:8}
    assert {r['pid']for r in actual.values()}=={r['pid']for r in before['running'].values()}
    intent=dict(time=time.time(),old_scheduler_pid=old,old_scheduler_identity=old_start,
        actual_learners_before=actual,registration_sha256=helper.digest(args.registration),
        action='SIGTERM scheduler PID only; never signal learner PIDs or process groups.')
    helper.atomic(root/'analysis/scheduler_v20_adoption_intent_v1.json',intent)
    assert identity(old)==old_start
    os.kill(old,signal.SIGTERM)
    for _ in range(150):
        if identity(old)!=old_start:break
        time.sleep(.02)
    else:raise AssertionError('Old controller still live; inspect, do not launch a competing manager.')
    # Reconcile natural exits and any just-started process before adopting. Never
    # equate a missing observation to process death or restart a live learner.
    queue=json.loads((root/'queue.json').read_text());finished=json.loads((root/'finished.json').read_text())
    for key,row in before['running'].items():
        if identity(row['pid'])==row['process_identity']:continue
        output=Path(row['output'])
        result=next((name for file,name in [('completed.json','complete'),('stopped.json','stopped'),
            ('screened.json','screened'),('staging_request.json','staged')]if(output/file).exists()),'failed')
        finished.setdefault(key,dict(exit_code=None,result=result,finished_at=time.time(),output=str(output),
            evidence='ActualPIDidentityterminalduringcontrollermigration;notanobservationtimeout.'))
    actual=learners(root,queue,finished,before['running'])
    assert all(identity(row['pid'])==row['process_identity']for row in actual.values())
    oldmap={j['id']:j for j in queue};assert not any(j['id']in oldmap for j in plan['jobs'])
    backup=root/'queue_before_unit_return_pilots_v1.json';assert not backup.exists();backup.write_bytes((root/'queue.json').read_bytes())
    combined=queue+plan['jobs'];done_or_live=set(finished)|set(actual)
    buckets=collections.OrderedDict(sampling_tau9=[],tau99=[],unit_return=[],existing_expectile_flow=[],confirmation=[],mean_backup=[])
    fixed=[];rest=[]
    for job in combined:
        key=job['id']
        if key in done_or_live:fixed.append(job);continue
        bucket='sampling_tau9'if key.startswith('vs1_')else'tau99'if key.startswith('ht1_')else'unit_return'if key.startswith('unit1_')else'existing_expectile_flow'if key.startswith('resume_phase3_ef1_')else'confirmation'if key.startswith('ieconfirm_')else'mean_backup'if key.startswith('meb1_')else None
        (rest if bucket is None else buckets[bucket]).append(job)
    original_buckets={k:[j['id']for j in v]for k,v in buckets.items()};pending=[]
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:pending.append(bucket.pop(0))
    updated=fixed+pending+rest
    assert len(updated)==len(queue)+6 and len({j['id']for j in updated})==len(updated)
    assert all(next(j for j in updated if j['id']==old['id'])==old for old in queue)
    helper.atomic(root/'finished.json',finished)
    helper.atomic(root/'queue.json',updated)
    helper.atomic(root/'scheduler_status.json',dict(pid=old,updated_at=time.time(),phase='controller_migration',
        running=actual,finished=len(finished),pending=len(updated)-len(finished)-len(actual)))
    amendment=dict(time=time.time(),scope='Add6registeredunitpilotsandORDERONLYforoldjobs',
        supersedes_order_only='scheduling_amendment_20261028_v2.json',buckets=original_buckets,
        all_previous_job_dicts_exact=True,new_jobs=plan['jobs'],learners_preserved=actual,
        no_learner_restarts=True,queue_before_sha256=helper.digest(backup),queue_after_sha256=helper.digest(root/'queue.json'))
    helper.atomic(root/'scheduling_amendment_20261028_v3.json',amendment)
    fcntl.flock(lock,fcntl.LOCK_UN)
    logpath=root/'scheduler_v20.log'
    with logpath.open('a')as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','utils.flow_iql_research.scheduler_v20',str(root),'--per-gpu','8'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    for _ in range(150):
        current=json.loads((root/'scheduler_status.json').read_text())
        if current['pid']==proc.pid and current['phase']=='running':break
        assert proc.poll()is None,'New controller exited; inspect preserved scheduler log.'
        time.sleep(.1)
    else:raise AssertionError('Adoption not yet observed; poll same handle, do not restart it.')
    for key,row in actual.items():
        if identity(row['pid'])==row['process_identity']:
            assert current['running'][key]['pid']==row['pid'] and current['running'][key]['process_identity']==row['process_identity']
    assert len(current['running'])==16 and collections.Counter(r['gpu']for r in current['running'].values())=={0:8,1:8}
    receipt=dict(time=time.time(),old_scheduler_pid=old,new_scheduler_pid=proc.pid,
        new_scheduler_identity=identity(proc.pid),new_scheduler_module='utils.flow_iql_research.scheduler_v20',
        current_learners=current['running'],preserved_original_live_handles=actual,
        old_training_jobs_unchanged=True,queue_size=len(updated),new_GPU_jobs=6,no_learner_restarts=True,
        registration_sha256=helper.digest(args.registration),source_sha256=helper.digest(__file__),log=str(logpath))
    helper.atomic(receipt_path,receipt)
    print(json.dumps(dict(status='adopted',new_scheduler_pid=proc.pid,actual_workers=16,new_jobs=6,queue_size=len(updated))),flush=True)


if __name__=='__main__':main()
