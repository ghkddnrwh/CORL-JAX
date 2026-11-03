"""Queue three isolated teacher controls after all locked confirmation jobs."""
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
from utils.flow_iql_research import scheduler_v21,scheduler_v4
from utils.flow_iql_research.adopt_scheduler_v20_v1 import identity,learners
from utils.flow_iql_research import tau_matched_extraction_v1 as helper


def insert(queue,jobs):
    positions=[i for i,j in enumerate(queue)if j['id'].startswith('meconfirm_')];assert positions
    index=max(positions)+1;new=queue[:index]+jobs+queue[index:]
    assert len(new)==len(queue)+len(jobs)and len({j['id']for j in new})==len(new)
    assert all(next(j for j in new if j['id']==old['id'])==old for old in queue)
    return new,index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('registration',type=Path)
    parser.add_argument('--validate',action='store_true');args=parser.parse_args();root=args.root.resolve()
    plan=json.loads(args.registration.read_text());assert Path(plan['root'])==root and len(plan['jobs'])==3
    helper.check_sources(plan['analysis_source_sha256'])
    assert helper.digest('utils/flow_iql_research/scheduler_v21.py')==plan['scheduler_source_sha256']
    queue=json.loads((root/'queue.json').read_text());updated,index=insert(queue,plan['jobs'])
    if args.validate:
        assert all(j['module']=='utils.flow_iql_research.train_mean_value_teacher_v1'for j in plan['jobs'])
        assert all(j['args']['profile']=='bounded_v2_mean_value_teacher'and j['args']['seed']==0 for j in plan['jobs'])
        assert all(j['args']['discount']==.999 and j['args']['expectile']==.9 and j['args']['samples']==9 for j in plan['jobs'])
        print(json.dumps(dict(status='passed',queue_mutations=0,manager_signals=0,new_training=0,insertion_index=index)),flush=True);return
    receipt=root/'analysis/scheduler_v21_adoption_v1.json';assert not receipt.exists()and not(root/'STOP').exists()
    lock=(root/'queue_mutation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    before=json.loads((root/'scheduler_status.json').read_text());old=before['pid'];old_start=identity(old);assert old_start is not None
    cmd=Path(f'/proc/{old}/cmdline').read_bytes().decode().split('\0')
    assert cmd[cmd.index('-m')+1]=='utils.flow_iql_research.scheduler_v20'and str(root)in cmd
    queue=json.loads((root/'queue.json').read_text());finished=json.loads((root/'finished.json').read_text())
    actual=learners(root,queue,finished,before['running'])
    assert len(actual)==16 and collections.Counter(h['gpu']for h in actual.values())=={0:8,1:8}
    helper.atomic(root/'analysis/scheduler_v21_adoption_intent_v1.json',dict(time=time.time(),
        old_pid=old,old_identity=old_start,learners_before=actual,registration_sha256=helper.digest(args.registration),
        action='Signal only the verified scheduler PID. Never signal learner PIDs or process groups.'))
    assert identity(old)==old_start;os.kill(old,signal.SIGTERM)
    for _ in range(150):
        if identity(old)!=old_start:break
        time.sleep(.02)
    else:raise AssertionError('Old manager still live; do not launch another manager.')
    queue=json.loads((root/'queue.json').read_text());finished=json.loads((root/'finished.json').read_text())
    for key,row in before['running'].items():
        if identity(row['pid'])==row['process_identity']:continue
        output=Path(row['output']);result=next((kind for file,kind in [('completed.json','complete'),('stopped.json','stopped'),
            ('screened.json','screened'),('staging_request.json','staged')]if(output/file).exists()),'failed')
        finished.setdefault(key,dict(exit_code=None,result=result,finished_at=time.time(),output=str(output),
            evidence='Verified PID identity became terminal during manager adoption, not an observation timeout.'))
    actual=learners(root,queue,finished,before['running']);assert all(identity(h['pid'])==h['process_identity']for h in actual.values())
    updated,index=insert(queue,plan['jobs']);backup=root/'queue_before_mean_value_teacher_pilots_v1.json';assert not backup.exists()
    backup.write_bytes((root/'queue.json').read_bytes());helper.atomic(root/'finished.json',finished);helper.atomic(root/'queue.json',updated)
    helper.atomic(root/'scheduler_status.json',dict(pid=old,updated_at=time.time(),phase='controller_migration',running=actual,
        finished=len(finished),pending=len(updated)-len(finished)-len(actual)))
    fcntl.flock(lock,fcntl.LOCK_UN);logpath=root/'scheduler_v21.log'
    with logpath.open('a')as log:process=subprocess.Popen([sys.executable,'-u','-m','utils.flow_iql_research.scheduler_v21',str(root),'--per-gpu','8'],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    for _ in range(150):
        current=json.loads((root/'scheduler_status.json').read_text())
        if current['pid']==process.pid and current['phase']=='running':break
        assert process.poll()is None,'New manager exited; inspect the preserved log.'
        time.sleep(.1)
    else:raise AssertionError('New manager adoption observation expired; poll the same handle, never restart solely for timeout.')
    for key,h in actual.items():
        if identity(h['pid'])==h['process_identity']:
            assert current['running'][key]['pid']==h['pid']and current['running'][key]['process_identity']==h['process_identity']
    assert len(current['running'])==16 and collections.Counter(h['gpu']for h in current['running'].values())=={0:8,1:8}
    result=dict(time=time.time(),old_scheduler_pid=old,new_scheduler_pid=process.pid,new_scheduler_identity=identity(process.pid),
        module='utils.flow_iql_research.scheduler_v21',current_learners=current['running'],preserved_live_handles=actual,
        old_jobs_exact=True,queue_size=len(updated),jobs_added=3,insertion_index=index,after_all_mean_confirmation=True,
        no_learner_restarts=True,queue_before_sha256=helper.digest(backup),queue_after_sha256=helper.digest(root/'queue.json'),
        registration_sha256=helper.digest(args.registration),source_sha256=helper.digest(__file__),log=str(logpath))
    helper.atomic(receipt,result)
    print(json.dumps(dict(status='adopted',scheduler_pid=process.pid,workers=16,new_jobs=3,queue_size=len(updated))),flush=True)


if __name__=='__main__':main()
