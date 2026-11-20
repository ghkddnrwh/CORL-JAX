"""Audit complete finite-expectile task cells; withhold a macro until all nine."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from utils.flow_iql_research.audit_gaussian_factorial_v1 import finite


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();root=args.root.resolve();rows=[]
    shorts=('ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene')
    for short in shorts:
        run=root/'runs'/f'eb1_{short}_bounded_v2_empirical_backup_g999_t90_s0'
        conf=json.loads((run/'config.json').read_text());ident=conf['identity'];cfg=ident['config']
        assert ident['seed']==0 and ident['profile']=='bounded_v2_empirical_backup'and ident['kind']=='ifql'
        assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']==cfg['backup_samples']==9
        assert cfg['backup_kind']=='finite_behavior_flow_target_Q_expectile'
        assert all(sha(path)==digest for path,digest in conf['source_sha256'].items())
        if not(run/'completed.json').exists():
            last=json.loads((run/'diagnostics.jsonl').read_text().splitlines()[-1])
            rows.append(dict(environment=ident['environment'],run=run.name,complete=False,last_step=last['step'],last3_mean=None))
            continue
        completion=json.loads((run/'completed.json').read_text());assert completion['kind']=='complete'and completion['step']==1000000
        evaluations=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()];cells=[]
        for step in(800000,900000,1000000):
            found=[entry for entry in evaluations if entry['step']==step];assert len(found)==1
            ev=found[0];assert ev['n_episodes']==50 and ev['first_episode_seed']==42 and ev['eval_device']=='cpu'
            assert ev['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
            for field in('episode_successes','episode_rewards','episode_lengths'):assert len(ev[field])==50 and np.isfinite(ev[field]).all()
            assert abs(np.mean(ev['episode_successes'])-ev['success_rate'])<1e-12
            path=run/f'checkpoint_{step}.pkl'
            with path.open('rb')as stream:saved=pickle.load(stream)
            assert saved['step']==step and json.loads(json.dumps(saved['identity']))==ident
            assert saved['source_sha256']==conf['source_sha256']and finite(saved['state'])
            state=saved['state'];assert int(state['q_updates'])==int(state['agent']['network']['opt_state']['0']['count'])==step
            assert int(state['v_updates'])==int(state['v_opt']['0']['count'])==2*step
            cells.append(dict(step=step,CPU50_success=ev['success_rate'],checkpoint_sha256=sha(path),
                full_state_finite=True,actual_Q_Adam_count=step,actual_V_Adam_count=2*step))
        rows.append(dict(environment=ident['environment'],run=run.name,complete=True,cells=cells,
            last3_mean=float(np.mean([cell['CPU50_success']for cell in cells])),config_sha256=sha(run/'config.json'),
            evaluation_file_sha256=sha(run/'evaluations.jsonl')))
    complete=sum(row['complete']for row in rows)
    result=dict(time=time.time(),kind='finite_backup_complete_task_audit',rows=rows,completed_tasks=complete,
        nine_task_mean=float(np.mean([row['last3_mean']for row in rows]))if complete==9 else None,
        pending_environments=[row['environment']for row in rows if not row['complete']],
        metric='Equal800K/900K/1MCPU50andnine-taskmeans;missingcellsarenotfilledzero.',source_sha256=sha(__file__),goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(dict(completed_tasks=complete,nine_task_mean=result['nine_task_mean'],
        values={row['environment']:row['last3_mean']for row in rows})),flush=True)


if __name__=='__main__':main()
