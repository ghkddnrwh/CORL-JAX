"""Persistent controller supervisor; orphan learners are adopted after restart."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from utils.flow_iql_research.run_stability_study_v1 import atomic, birth


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    lease=(root/'supervisor.lock').open('a'); fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    retries=0
    while True:
        old=json.loads((root/'controller_status.json').read_text()) if (root/'controller_status.json').exists() else {}
        if old.get('phase')=='complete' and old.get('analysis_exit_code')==0: return
        if (root/'STOP').exists(): return
        worker=None
        if birth(old.get('pid',-1))!=old.get('birth') or old.get('birth') is None:
            with (root/'controller_stdout.log').open('a') as stream:
                worker=subprocess.Popen([sys.executable,'-u','-m','utils.flow_iql_research.run_stability_study_v1',
                    '--root',str(root)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            pid=worker.pid; started=time.time()
        else:
            pid=old['pid']; started=time.time()
        stamp=birth(pid)
        while birth(pid)==stamp and stamp is not None:
            status=json.loads((root/'controller_status.json').read_text()) if (root/'controller_status.json').exists() else {}
            atomic(root/'supervisor_status.json',dict(pid=os.getpid(),time=time.time(),controller_pid=pid,
                controller_restarts=retries,phase='watching',controller_heartbeat=status.get('time'),
                training_complete=status.get('training_complete',0),running=len(status.get('running',{}))))
            # A long missing heartbeat is surfaced without killing active learners.
            if time.time()-max(started,status.get('time',0))>1800:
                atomic(root/'ATTENTION.json',dict(reason='controller_heartbeat_stale',pid=pid,time=time.time()))
                os.kill(pid,15)
            time.sleep(15)
            if worker is not None: worker.poll()
        if worker is not None: worker.wait()
        status=json.loads((root/'controller_status.json').read_text()) if (root/'controller_status.json').exists() else {}
        if status.get('phase') in ('complete','incomplete','verification_failed'):
            atomic(root/'supervisor_status.json',dict(pid=os.getpid(),time=time.time(),phase=status['phase'],
                controller_restarts=retries,analysis_exit_code=status.get('analysis_exit_code')))
            return
        if (root/'STOP').exists(): return
        retries+=1
        if retries>3:
            atomic(root/'ATTENTION.json',dict(reason='controller_restart_limit',time=time.time(),restarts=retries))
            raise RuntimeError('Controller repeatedly failed; see logs. Active learners remain intact.')
        time.sleep(10)


if __name__=='__main__': main()
