"""Reallocate selected auxiliary controls at fresh, resumable checkpoints.

This is an explicit research-priority change, never an observation-timeout
restart. Verify worker ownership, all frozen sources, real optimizer clocks,
and a full checkpoint before requesting termination. Preserve every file.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time
from utils.flow_iql_research.scheduler_v4 import process_identity


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def checkpoint(run):
    with (run/'latest.pkl').open('rb') as file:
        saved = pickle.load(file)
    step = saved['step']; state = saved['state']
    config = json.loads((run/'config.json').read_text())
    assert json.loads(json.dumps(saved['identity'])) == config['identity']
    assert saved['source_sha256'] == config['source_sha256']
    for source, digest in config['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    assert int(state['q_updates']) == step
    own_value_optimizer = 'v2' in config['identity']['profile']
    assert config['identity']['profile'] in ('official', 'bounded_v2')
    expected_value_updates = (2 if own_value_optimizer else 1)*step
    assert int(state['v_updates']) == expected_value_updates
    assert int(state['agent']['network']['opt_state']['0']['count']) == step
    assert int(state['v_opt']['0']['count']) == (2*step if own_value_optimizer else 0)
    return saved


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--plan', type=Path, required=True); args = parser.parse_args()
    root = args.root.resolve(); plan = json.loads(args.plan.read_text())
    lock = (root/(args.plan.stem+'_staging.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    targets = set(plan['stage_runs']); requested = {}; done = {}; seen = {}
    while targets-set(done):
        status = json.loads((root/'scheduler_status.json').read_text())
        for name in sorted(targets-set(done)):
            run = root/'runs'/name
            if (run/'completed.json').exists() or (run/'stopped.json').exists():
                done[name] = dict(status='naturally_terminal'); continue
            if name in requested:
                pid, identity = requested[name]
                if process_identity(pid) == identity:
                    continue
                saved = checkpoint(run); latest = run/'latest.pkl'
                target = run/f"staged_checkpoint_{saved['step']}_inode{latest.stat().st_ino}.pkl"
                if not target.exists():
                    os.link(latest, target)
                record = dict(time=time.time(), status='staged_for_resume', pid=pid,
                    resume_step=saved['step'], Q_updates=int(saved['state']['q_updates']),
                    V_updates=int(saved['state']['v_updates']), checkpoint=str(target),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    resume_job=plan['resume_prefix']+name,
                    note='Full state and replay RNG are retained. Unsaved post-checkpoint updates can be replayed; long GPU numerical trajectories need not be bitwise identical. This is not a final score.')
                atomic(run/'staged_interruption.json', record); done[name] = record
                print('STAGED', name, saved['step'], flush=True); continue
            record = status['running'].get(name)
            if record is None or not (run/'latest.pkl').exists():
                continue
            config = json.loads((run/'config.json').read_text())
            assert config['identity']['profile'] == plan['expected_profiles'][name]
            assert config['identity']['kind'] == plan['expected_kinds'][name]
            inode = (run/'latest.pkl').stat().st_ino
            if seen.get(name) == inode:
                continue
            seen[name] = inode; saved = checkpoint(run)
            if saved['step'] < plan['minimum_saved_steps'][name] or saved['step'] == 1000000:
                continue
            pid = record['pid']; identity = process_identity(pid)
            if identity is None:
                continue
            assert identity == record['process_identity']
            command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert command[command.index('-m')+1] == 'utils.flow_iql_research.train_policy_v2'
            assert Path(command[command.index('--output')+1]).resolve() == run
            queue = json.loads((root/'queue.json').read_text())
            resume = next(j for j in queue if j['id'] == plan['resume_prefix']+name)
            assert resume['output_run'] == name and resume['args']['steps'] == 1000000
            atomic(run/'staging_request.json', dict(time=time.time(), pid=pid,
                process_identity=identity, last_saved_step=saved['step'],
                reason=plan['early_stage_not_final_rejection'], resume_job=resume['id']))
            os.kill(pid, signal.SIGTERM); requested[name] = (pid, identity)
            print('REQUEST', name, pid, saved['step'], flush=True)
        atomic(root/(args.plan.stem+'_staging_status.json'), dict(pid=os.getpid(), time=time.time(),
            done=done, requested=list(requested), pending=sorted(targets-set(done))))
        time.sleep(5)


if __name__ == '__main__':
    main()
