"""New collision-safe offload journal, protecting all fresh and mean models."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
from utils.flow_iql_research import offload_completed_checkpoints_v2 as previous

PROTECTED_PREFIXES=('ieconfirm_','meconfirm_','meb1_','ef1_')


def readers(root):
    protected=set();prefix=str(root/'runs')+'/'
    for cmd in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            for arg in cmd.read_bytes().decode().split('\0'):
                if arg.startswith(prefix):protected.add(arg[len(prefix):].split('/')[0])
        except (OSError,UnicodeError):continue
    return protected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-free-gib',type=float,default=32)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=args.root.resolve();args.output.mkdir(parents=True,exist_ok=False)
    lock=(args.output/'offload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_pins={str(Path(__file__).resolve()):previous.digest(Path(__file__)),
        str(Path(previous.__file__).resolve()):previous.digest(Path(previous.__file__))}
    reading=readers(root)
    groups=[g for g in previous.inventory(root)
        if not Path(g['run']).name.startswith(PROTECTED_PREFIXES)and Path(g['run']).name not in reading]
    record=dict(pid=os.getpid(),started_at=time.time(),source_sha256=source_pins,
        apply=args.apply,target_free_GiB=args.target_free_gib,destination=str(args.destination.resolve()),
        protected_prefixes=PROTECTED_PREFIXES,protected_reader_runs=sorted(reading),
        active_outputs_and_all_latest_inodes_protected=True,
        free_bytes_before=shutil.disk_usage(root).free,inventory_groups=len(groups),
        inventory_bytes=sum(g['signature'][2]for g in groups),relocated=[])
    previous.atomic(args.output/'manifest.json',record)
    if not args.apply:
        print({k:v for k,v in record.items()if k!='relocated'},flush=True);return
    args.destination.mkdir(parents=True,exist_ok=True)
    assert args.destination.stat().st_dev!=root.stat().st_dev
    journal=args.output/'transactions';journal.mkdir(exist_ok=False)
    for group in groups:
        if shutil.disk_usage(root).free>=args.target_free_gib*1024**3:break
        name=Path(group['run']).name
        if name in readers(root):continue
        assert not name.startswith(PROTECTED_PREFIXES)
        dev,inode=group['signature'][:2];assert not(journal/f'{dev}-{inode}.json').exists()
        moved=previous.relocate(group,args.destination.resolve(),journal)
        record['relocated'].append(moved);record.update(updated_at=time.time(),free_bytes_now=shutil.disk_usage(root).free)
        previous.atomic(args.output/'manifest.json',record)
        if len(record['relocated'])%25==0:print(dict(groups=len(record['relocated']),free_GiB=record['free_bytes_now']/1024**3),flush=True)
    for source,sha in source_pins.items():assert previous.digest(Path(source))==sha
    record.update(completed_at=time.time(),free_bytes_now=shutil.disk_usage(root).free,
        status='target_reached'if shutil.disk_usage(root).free>=args.target_free_gib*1024**3 else'inventory_exhausted')
    previous.atomic(args.output/'manifest.json',record)
    print(dict(status=record['status'],groups=len(record['relocated']),free_GiB=record['free_bytes_now']/1024**3),flush=True)


if __name__=='__main__':main()
