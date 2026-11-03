"""Private CPU-only ownership/adoption/refill test; no learner or GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from utils.flow_iql_research import scheduler_onestep_deploy_v3 as wrapper
from utils.flow_iql_research.scheduler_v4 import atomic_json,process_identity


MODULE='utils.flow_iql_research.validate_onestep_scheduler_adoption_v2'


def wait_for(test, timeout=25):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        answer=test()
        if answer:return answer
        time.sleep(.1)
    raise AssertionError('Private adoption fixture timed out; inspect actual PIDs before recovery')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-worker',action='store_true');parser.add_argument('--duration',type=float,default=1)
    args=parser.parse_args()
    if args.fixture_worker:
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output/'started.json',dict(pid=os.getpid(),birth=process_identity(os.getpid()),
            affinity=sorted(os.sched_getaffinity(0)),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            XLA_PYTHON_CLIENT_MEM_FRACTION=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),validation_fixture_only=True))
        time.sleep(args.duration)
        atomic_json(args.output/'completed.json',dict(validation_fixture_only=True,no_performance_claim=True))
        return
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    assert len(os.sched_getaffinity(0))==12
    assert {'utils.flow_iql_research.train_onestep_deploy_v1',
            'utils.flow_iql_research.train_set_onestep_deploy_v1'}<=wrapper.allowed_modules()
    cpus=sorted(os.sched_getaffinity(0));jobs=[];running={};workers=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',XLA_PYTHON_CLIENT_MEM_FRACTION='0.1')
    for index,(rid,duration) in enumerate((('fixture_old_a',4),('fixture_old_b',8),('fixture_pending',1))):
        output=root/'runs'/rid
        jobs.append(dict(id=rid,module=MODULE,args={'fixture-worker':'', 'duration':duration}))
        if index==2:continue
        output.mkdir(parents=True);log=(output/'stdout.log').open('a')
        command=['taskset','-c',','.join(map(str,cpus[index*6:(index+1)*6])),sys.executable,'-u','-m',MODULE,
                 '--output',str(output),'--fixture-worker','--duration',str(duration)]
        worker=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(index)),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        log.close();workers.append(worker)
        running[rid]=dict(pid=worker.pid,process_identity=process_identity(worker.pid),gpu=index,
            output=str(output),started_at=time.time())
    # Empty string is not a flag encoding supported by the scheduler; use a
    # numeric option accepted by the fixture worker instead.
    for job in jobs:job['args']={'duration':job['args']['duration']}
    # The pending worker is selected through a dedicated environment marker.
    env['ONESTEP_ADOPTION_FIXTURE_WORKER']='1'
    atomic_json(root/'queue.json',jobs)
    atomic_json(root/'scheduler_status.json',dict(pid=999999999,running=running,pending=1,finished=0))
    log=(root/'manager.log').open('a')
    code=('import sys; from utils.flow_iql_research import scheduler_onestep_deploy_v3 as w; '
          'w.scheduler_v4.MODULES=w.allowed_modules()|{'+repr(MODULE)+'}; '
          'w.scheduler_v4.main()')
    manager=subprocess.Popen([sys.executable,'-u','-c',code,str(root),'--per-gpu','1'],env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    log.close()
    status=wait_for(lambda: (value if (value:=json.loads((root/'scheduler_status.json').read_text()))['pid']==manager.pid else None))
    assert all(status['running'][rid]['pid']==record['pid']for rid,record in running.items())
    (root/'PAUSE_NEW_LAUNCHES.json').write_text(json.dumps(dict(validation_fixture_only=True)))
    paused=wait_for(lambda: (value if (value:=json.loads((root/'scheduler_status.json').read_text())).get('phase')=='paused_new_launches' and not value['running'] else None))
    assert paused['pending']==1 and process_identity(manager.pid) and not(root/'runs/fixture_pending/started.json').exists()
    assert len(json.loads((root/'finished.json').read_text()))==2
    (root/'PAUSE_NEW_LAUNCHES.json').unlink()
    finished=wait_for(lambda: (value if len(value:=json.loads((root/'finished.json').read_text()) if (root/'finished.json').exists() else {})==3 else None))
    assert all(row['result']=='complete'for row in finished.values())
    first=json.loads((root/'runs/fixture_old_a/started.json').read_text())
    refill=json.loads((root/'runs/fixture_pending/started.json').read_text())
    assert first['affinity']==refill['affinity'] and first['CUDA_VISIBLE_DEVICES']==refill['CUDA_VISIBLE_DEVICES']=='0'
    assert refill['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.1'
    text=(root/'manager.log').read_text()
    assert text.count('ADOPT ')==2 and text.count('START ')==1 and text.count('FINISH ')==3
    (root/'STOP').write_text('Private validation fixture only.\n')
    manager.wait(timeout=8)
    for worker in workers:worker.wait(timeout=1)
    assert manager.returncode==0
    recovery=root/'terminal_recovery';recovery.mkdir();output=recovery/'runs/fixture_already_complete';output.mkdir(parents=True)
    atomic_json(output/'completed.json',dict(validation_fixture_only=True))
    atomic_json(recovery/'queue.json',[dict(id='fixture_already_complete',module=MODULE,args=dict(duration=1))])
    atomic_json(recovery/'scheduler_status.json',dict(pid=999999999,running={'fixture_already_complete':dict(
        pid=999999998,process_identity='missing_fixture',gpu=0,output=str(output),started_at=time.time())}))
    with(recovery/'manager.log').open('a')as log:
        recovered=subprocess.Popen([sys.executable,'-u','-c',code,str(recovery),'--per-gpu','1'],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    status=wait_for(lambda: (value if (value:=json.loads((recovery/'scheduler_status.json').read_text()))['pid']==recovered.pid else None))
    assert status['pending']==0 and status['finished']==1 and not status['running']
    terminal=json.loads((recovery/'finished.json').read_text())['fixture_already_complete']
    assert terminal['result']=='complete'and terminal['adoption_recovered_terminal']
    assert 'START 'not in(recovery/'manager.log').read_text()and not(output/'started.json').exists()
    (recovery/'STOP').write_text('Private terminal-recovery fixture only.\n');recovered.wait(timeout=8)
    assert recovered.returncode==0
    atomic_json(root/'result.json',dict(time=time.time(),status='pass',checks=[
        'V3 wrapper allowlists both frozen learner implementations',
        'Adopt actual PID/birth/output/CPU-slot ownership without duplicate launch',
        'Refill the first freed slot once, preserving CPU affinity, CUDA index and memory fraction',
        'Pause new launches while both existing fixtures finish; keep manager live, resume without duplicates',
        'Classify both adopted completions and the replacement; STOP drains manager',
        'Recover an already-dead completed owner from stale running status without re-launching it'],
        no_performance_claim=True,scope='Three private stdlib CPU fixtures; zero learner, model, evaluator or GPU operations.'))
    print(json.dumps(dict(status='pass',output=str(root/'result.json'))),flush=True)


if __name__=='__main__':
    if os.environ.get('ONESTEP_ADOPTION_FIXTURE_WORKER')=='1'and '--fixture-worker'not in sys.argv:
        sys.argv.append('--fixture-worker')
    main()
