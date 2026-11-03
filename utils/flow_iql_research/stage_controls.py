"""Stage selected ablations at >=300K, preserving full state for later resume.

The eight principal official/corrected trajectories continue to1M. Ownership is
verified before terminating a worker; no checkpoint, score or source is deleted.
The queue contains explicit resume jobs pointing at the same scientific run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time
from utils.flow_iql_research.scheduler_v3 import process_identity


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def read_saved(path):
    with path.open('rb') as f:return pickle.load(f)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();plan=json.loads((root/'staged_value_controls_plan.json').read_text())
    targets=set(plan['stage_runs']);done={};requested={};seen={}
    while targets-set(done):
        status=json.loads((root/'scheduler_status.json').read_text())
        for name in sorted(targets-set(done)):
            run=root/'runs'/name
            if (run/'completed.json').exists():
                done[name]=dict(status='naturally_complete');continue
            if name in requested:
                pid,identity=requested[name]
                if process_identity(pid)==identity:continue
                latest=run/'latest.pkl';saved=read_saved(latest);step=saved['step']
                target=run/f'staged_checkpoint_{step}.pkl'
                if not target.exists():os.link(latest,target)
                assert int(saved['state']['q_updates'])==step
                count=int(saved['state']['v_updates'])
                cfg=json.loads((run/'config.json').read_text())
                assert count==step*(2 if 'v2' in cfg['identity']['profile'] else 1)
                record=dict(status='staged_for_resume',time=time.time(),pid=pid,resume_step=step,
                            q_updates=step,v_updates=count,checkpoint=target.name,
                            checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            note='Any unsaved updates after this snapshot were discarded; final retained optimizer trajectory resumes from this full state and replay RNG. This is not a final performance result.')
                atomic(run/'staged_interruption.json',record);done[name]=record
                print('STAGED',name,step,flush=True);continue
            record=status['running'].get(name)
            if record is None:continue
            latest=run/'latest.pkl'
            if not latest.exists():continue
            inode=latest.stat().st_ino
            if seen.get(name)==inode:continue
            seen[name]=inode
            saved=read_saved(latest)
            if saved['step']<plan['threshold']:continue
            pid=record['pid'];identity=process_identity(pid)
            if identity is None:continue
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert 'utils.flow_iql_research.train' in cmd
            assert Path(cmd[cmd.index('--output')+1]).resolve()==run
            atomic(run/'staging_request.json',dict(time=time.time(),pid=pid,threshold=plan['threshold'],
                last_saved_step=saved['step'],reason=plan['reason'],resume_job='resume_'+name))
            os.kill(pid,signal.SIGTERM);requested[name]=(pid,identity)
            print('REQUEST',name,pid,saved['step'],flush=True)
        atomic(root/'staged_value_controls_status.json',dict(pid=os.getpid(),time=time.time(),done=done,
              requested=list(requested),pending=sorted(targets-set(done))))
        time.sleep(5)
    print('ALL STAGED OR COMPLETE',flush=True)


if __name__=='__main__':main()
