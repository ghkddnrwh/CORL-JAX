"""Audit all24Giant learners, write equal-last3/four-seed tables, then stop."""
import argparse
import csv
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import time
import numpy as np


STEPS=(800000,900000,1000000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temp,path)


def records(path,complete):
    if not path.exists():
        return []
    content=path.read_text()
    if content and not content.endswith('\n'):
        assert not complete
        content=content.rsplit('\n',1)[0] if '\n' in content else ''
    return [json.loads(line) for line in content.splitlines()]


def finite(tree):
    if isinstance(tree,dict):
        return all(finite(v) for v in tree.values())
    if isinstance(tree,(tuple,list)):
        return all(finite(v) for v in tree)
    return bool(np.isfinite(np.asarray(tree)).all())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('registration',type=Path)
    args=parser.parse_args()
    plan=json.loads(args.registration.read_text())
    root=Path(plan['root'])
    output=root/'analysis/audit_v1'
    output.mkdir(exist_ok=False)
    lock=(output/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pins={}
    while True:
        try:
            for path,digest in plan['all_source_sha256'].items():
                assert sha(path)==digest,path
            rows=[]
            for job in plan['jobs']:
                rid=job['id']
                run=root/'runs'/rid
                base=dict(run=rid,kind=job['args']['kind'],discount=job['args']['discount'],seed=job['args']['seed'])
                if not (run/'config.json').exists():
                    rows.append(dict(**base,status='pending',complete=False,last3_mean=None))
                    continue
                conf=json.loads((run/'config.json').read_text())
                assert conf['identity']==plan['expected_identities'][rid]
                assert conf['source_sha256']==plan['training_source_sha256']
                for path,digest in conf['source_sha256'].items():
                    p=Path(path)
                    target=run/'source'/('vendor_'+p.name if p.parent.name=='vendor' else p.name)
                    if target.exists():
                        assert sha(target)==digest
                complete=(run/'completed.json').exists()
                evals=records(run/'evaluations.jsonl',complete)
                assert all(row['step'] in STEPS for row in evals)
                per_step={}
                counts=[]
                for step in STEPS:
                    found=[row for row in evals if row['step']==step]
                    assert len(found)<=1
                    if not found:
                        per_step[str(step)]=None
                        continue
                    row=found[0]
                    assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
                    assert row['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
                    assert all(len(row[k])==50 for k in ('episode_successes','episode_rewards','episode_lengths'))
                    assert all(s in (0,1) for s in row['episode_successes'])
                    assert all(math.isfinite(float(x)) for k in ('episode_rewards','episode_lengths') for x in row[k])
                    assert all(float(n).is_integer() and n>0 for n in row['episode_lengths'])
                    successes=sum(int(s) for s in row['episode_successes'])
                    assert abs(row['success_rate']-successes/50)<1e-12
                    counts.append(successes)
                    per_step[str(step)]=successes/50
                    key=f'{rid}:{step}'
                    if key not in pins:
                        checkpoint=run/f'checkpoint_{step}.pkl'
                        with checkpoint.open('rb') as stream:
                            saved=pickle.load(stream)
                        state=saved['state']
                        ratio=plan['V_updates_per_Q'][base['kind']]
                        assert saved['step']==int(state['q_updates'])==step
                        assert int(state['v_updates'])==ratio*step
                        assert int(state['agent']['network']['opt_state']['0']['count'])==step
                        assert int(state['v_opt']['0']['count'])==(2*step if base['kind']=='mean_backup' else 0)
                        assert saved['identity']==conf['identity'] and saved['source_sha256']==conf['source_sha256']
                        assert finite(state)
                        pins[key]=dict(path=str(checkpoint),sha256=sha(checkpoint),Q_Adam=step,V_total=ratio*step,finite=True)
                probes=records(run/'diagnostics.jsonl',complete)
                ratio=plan['V_updates_per_Q'][base['kind']]
                assert all(p['finite'] and p['q_updates']==p['step'] and p['v_updates']==ratio*p['step'] for p in probes)
                stopped=json.loads((run/'stopped.json').read_text()) if (run/'stopped.json').exists() else None
                if complete:
                    done=json.loads((run/'completed.json').read_text())
                    assert done['kind']=='complete' and done['step']==done['q_updates']==1000000 and done['v_updates']==ratio*1000000
                    assert len(counts)==3 and not stopped and len(probes)==1000
                    for path,digest in conf['source_sha256'].items():
                        p=Path(path)
                        assert sha(run/'source'/('vendor_'+p.name if p.parent.name=='vendor' else p.name))==digest
                rows.append(dict(**base,status='complete' if complete else 'stopped' if stopped else 'running',
                    complete=complete,last_training_step=probes[-1]['step'] if probes else 0,stopped=stopped,
                    per_step=per_step,last3_mean=float(Fraction(sum(counts),150)) if complete else None,
                    evaluations_sha256=sha(run/'evaluations.jsonl') if evals else None))
            groups=[]
            for discount in (.99,.999):
                for kind in ('iql','fql','mean_backup'):
                    chosen=[r for r in rows if r['discount']==discount and r['kind']==kind]
                    assert len(chosen)==4 and {r['seed'] for r in chosen}=={0,1,2,3}
                    ready=all(r['complete'] for r in chosen)
                    groups.append(dict(kind=kind,discount=discount,completed_seeds=sum(r['complete'] for r in chosen),
                        seed_last3={str(r['seed']):r['last3_mean'] for r in chosen},
                        four_seed_mean=statistics.mean(r['last3_mean'] for r in chosen) if ready else None,
                        four_seed_sample_SD=statistics.stdev(r['last3_mean'] for r in chosen) if ready else None))
            complete=all(row['complete'] for row in rows)
            result=dict(time=time.time(),pid=os.getpid(),status='complete' if complete else 'running',
                completed_runs=sum(r['complete'] for r in rows),expected_runs=24,rows=rows,groups=groups,
                actual_finite_late_models=len(pins),expected_late_models=72,actual_model_pins=pins,
                registration_sha256=sha(args.registration),metric=plan['metric'],
                limits=['No bestcheckpoint/seed selection; incomplete or failed runs are not zero.',
                    'Comparison is one new environment and four training seeds; no redefinition of previous nine-task averages.',
                    'FQL uses the provided antmaze public configuration:alpha10,targetQmin. Proposed algorithm has9samples and2unusedVfits. No equal-compute claim.'])
            atomic(output/'status.json',result)
            buffer=io.StringIO()
            writer=csv.writer(buffer)
            writer.writerow(['kind','discount','completed_seeds','seed0_last3','seed1_last3','seed2_last3','seed3_last3','mean','sample_SD'])
            for group in groups:
                writer.writerow([group['kind'],group['discount'],group['completed_seeds'],
                    *[group['seed_last3'][str(s)] for s in range(4)],group['four_seed_mean'],group['four_seed_sample_SD']])
            temp=output/'results.csv.tmp'
            temp.write_text(buffer.getvalue())
            os.replace(temp,output/'results.csv')
            if complete:
                assert len(pins)==72
                atomic(output/'final_audit.json',result)
                (root/'STOP').write_text('All24Giant runs and72actual late models audited; no further jobs.\n')
                print(json.dumps({'status':'complete','groups':groups}),flush=True)
                return
        except Exception as error:
            atomic(output/'failure.json',dict(time=time.time(),error=repr(error),existing_results_preserved=True))
            raise
        time.sleep(20)


if __name__=='__main__':main()
