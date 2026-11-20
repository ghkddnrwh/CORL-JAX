"""Two-tau/two-V-batch comparison with count-aware native tau99 controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper

METRICS=('q_rms','v_rms','q_centered_rms','q_rms_over_return_bound','td_rms','score_rms',
    'value/value_loss','value/post_fit_loss','critic/critic_loss',
    'backup/clip_fraction','backup/clip_rms','backup/fresh_v_shift_rms',
    'backup/lower_clip_fraction','backup/upper_clip_fraction','grad_norm')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    args=parser.parse_args();plan=json.loads(args.registration.read_text())
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path(plan['root']);pins={}
    while True:
        helper.check_sources(plan['source_sha256']);rows=[]
        for run_id in plan['run_ids']:
            run=root/'runs'/run_id;identity=plan['expected_identities'][run_id];cfg=identity['config']
            stopped=json.loads((run/'stopped.json').read_text()) if (run/'stopped.json').exists() else None
            probes=[json.loads(line) for line in (run/'diagnostics.jsonl').read_text().splitlines()] if (run/'diagnostics.jsonl').exists() else []
            if probes:
                assert all(p['finite'] for p in probes)
            per_step={}
            for step in helper.STEPS:
                record=helper.primary_record(run,step)
                if record is not None:
                    key=f'{run_id}:{step}';path=run/f'checkpoint_{step}.pkl'
                    if not path.exists():path=root/'analysis/checkpoint_archive'/run_id/f'checkpoint_{step}.pkl'
                    stat=path.stat();stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
                    if key not in pins or pins[key]['stamp']!=stamp:
                        _,_,actual,sha=helper.load_parent(plan,run_id,step)
                        if key in pins:assert pins[key]['sha256']==sha
                        pins[key]=dict(path=str(actual),sha256=sha,stamp=stamp)
                per_step[str(step)]=None if record is None else record['success_rate']
            parent_complete=False
            if (run/'completed.json').exists():
                done=json.loads((run/'completed.json').read_text());counts=plan['expected_final_clocks'][run_id]
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==counts['V_total']
                assert not stopped and probes[-1]['step']==1000000
                parent_complete=True
            complete=parent_complete and all(v is not None for v in per_step.values())
            native=identity['profile']=='official'
            mode='native' if native else 'fresh' if 'fresh_batches' in identity['profile'] else 'repeated'
            late=[p for p in probes if p['step']>800000]
            metrics={}
            for metric in METRICS:
                available=[p for p in late if metric in p]
                if available:
                    weights=[p['metric_window_updates'] for p in available]
                    metrics[metric]=sum(p[metric]*w for p,w in zip(available,weights))/sum(weights)
            rows.append(dict(run_id=run_id,environment=identity['environment'],expectile=cfg['expectile'],
                mode=mode,last_training_step=probes[-1]['step'] if probes else None,stopped=stopped,
                parent_complete=parent_complete,complete=complete,per_step=per_step,
                last3_mean=statistics.mean(per_step.values()) if complete else None,
                late_training_window_means=metrics,
                late_window_semantics='Updates800001..1M; loss/grad logs averaged over their actual window counts. Probe RMS values are sampled only at window endpoints.',
                target_semantics='old_V_unprojected' if native else 'fresh_V_projected',
                joint_gradient_semantics='Q_plus_V_plus_actor' if native else 'Q_plus_actor_separate_V_Adam'))
        methods=[]
        for tau,mode in ((.9,'repeated'),(.9,'fresh'),(.99,'native'),(.99,'repeated'),(.99,'fresh')):
            selected=[r for r in rows if r['expectile']==tau and r['mode']==mode];assert len(selected)==3
            complete=all(r['complete'] for r in selected)
            methods.append(dict(expectile=tau,mode=mode,completed_tasks=sum(r['complete'] for r in selected),
                three_task_mean=statistics.mean(r['last3_mean'] for r in selected) if complete else None))
        contrasts=[]
        for env in sorted({r['environment'] for r in rows}):
            lookup={(r['expectile'],r['mode']):r['last3_mean'] for r in rows if r['environment']==env}
            delta=lambda a,b:None if lookup[a] is None or lookup[b] is None else lookup[a]-lookup[b]
            a=delta((.9,'fresh'),(.9,'repeated'));b=delta((.99,'fresh'),(.99,'repeated'))
            contrasts.append(dict(environment=env,fresh_minus_repeated_tau9=a,fresh_minus_repeated_tau99=b,
                sampling_by_tau_interaction=None if a is None or b is None else b-a,
                tau99_minus_tau9_repeated=delta((.99,'repeated'),(.9,'repeated')),
                tau99_minus_tau9_fresh=delta((.99,'fresh'),(.9,'fresh')),
                repeated_minus_native_tau99=delta((.99,'repeated'),(.99,'native')),
                fresh_minus_native_tau99=delta((.99,'fresh'),(.99,'native'))))
        complete=all(r['complete'] for r in rows)
        helper.atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),status='complete' if complete else 'running',
            rows=rows,methods=methods,paired_contrasts=contrasts,actual_checkpoint_pins=pins,
            registration_sha256=helper.digest(args.registration),goal_complete=False,limits=plan['limits']))
        if complete:return
        time.sleep(30)


if __name__=='__main__':main()
