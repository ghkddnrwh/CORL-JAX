"""Retain each actual latest checkpoint before later atomic replacement.

Hardlinks are placed in a separate auxiliary archive, so the trainer's own
checkpoint creation cannot race with this process. No training files are changed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();archive=root/'analysis/checkpoint_archive';archive.mkdir(parents=True,exist_ok=True)
    lock=(archive/'archive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seen={};errors=[]
    while not (root/'STOP_ARCHIVE').exists():
        low_disk=shutil.disk_usage(root).free < 20*1024**3
        if not low_disk:
            for latest in sorted((root/'runs').glob('*/latest.pkl')):
                stat=latest.stat();identity=(stat.st_ino,stat.st_size)
                if seen.get(str(latest))==identity:continue
                destdir=archive/latest.parent.name;destdir.mkdir(exist_ok=True)
                temp=destdir/'captured.pkl'
                try:
                    if temp.exists():temp.unlink()
                    os.link(latest,temp)
                    with temp.open('rb') as f:saved=pickle.load(f)
                    step=saved['step'];assert int(saved['state']['q_updates'])==step
                    target=destdir/f'checkpoint_{step}.pkl'
                    if target.exists():
                        # Keep the first captured object. A repeated step after a
                        # restart is retained separately if its inode differs.
                        if target.stat().st_ino!=temp.stat().st_ino:
                            target=destdir/f'checkpoint_{step}_inode{temp.stat().st_ino}.pkl'
                    temp.replace(target)
                    seen[str(latest)]=identity
                    print('ARCHIVE',latest.parent.name,step,flush=True)
                except (OSError,ValueError,AssertionError,EOFError,pickle.UnpicklingError) as error:
                    errors.append(dict(path=str(latest),error=repr(error),time=time.time()))
                    if temp.exists():temp.unlink()
        status=dict(pid=os.getpid(),updated_at=time.time(),low_disk_optional_archiving_paused=low_disk,
                    checkpoint_files=len(list(archive.glob('*/checkpoint_*.pkl'))),errors=errors[-20:])
        temp=archive/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(archive/'status.json')
        time.sleep(20)


if __name__=='__main__':main()
