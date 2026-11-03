"""Separate learned-V mean teachers from finite mean-backup pilot returns."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.audit_probability_value_pilots_v1 import equal

METHODS=('learned_mean','finite_mean','learned_min','finite_min')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path);args=parser.parse_args()
    plan=json.loads(args.registration.read_text());root=Path(plan['root']);output=Path(plan['analysis_output'])
    output.mkdir(parents=True,exist_ok=False);lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache={};pins={}
    def model(rid,step):
        key=f'{rid}:{step}';path=root/'runs'/rid/f'checkpoint_{step}.pkl';stat=path.stat()
        stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
        if key not in cache or pins[key]['stamp']!=stamp:
            _,saved,actual,sha=helper.load_parent(plan,rid,step)
            if key in pins:assert pins[key]['sha256']==sha
            pins[key]=dict(path=str(actual),sha256=sha,stamp=stamp);cache[key]=saved
        return cache[key]
    while True:
        helper.check_sources(plan['analysis_source_sha256']);rows=[]
        for rid in plan['all_run_ids']:
            run=root/'runs'/rid;per_step={};identity=plan['expected_identities'][rid]
            for step in helper.STEPS:
                files=plan['auxiliary_N9_evaluations'].get(rid,{})
                if files:
                    path=Path(files[str(step)]);assert helper.digest(path)==plan['auxiliary_N9_evaluation_sha256'][str(path)]
                    record=json.loads(path.read_text());helper.check_episodes(record,step)
                    assert record['run']==rid and record['selection_samples']==9 and record['selection_rule']=='max'
                else:record=helper.primary_record(run,step)
                if record is not None:
                    model(rid,step)
                    if files:assert record['checkpoint_sha256']==pins[f'{rid}:{step}']['sha256']
                per_step[str(step)]=None if record is None else record['success_rate']
            complete=False;stopped=None
            if(run/'stopped.json').exists():stopped=json.loads((run/'stopped.json').read_text())
            if(run/'completed.json').exists():
                done=json.loads((run/'completed.json').read_text())
                assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000 and not stopped
                complete=all(v is not None for v in per_step.values())
            probe_file=run/'diagnostics.jsonl';probes=[json.loads(x)for x in probe_file.read_text().splitlines()]if probe_file.exists()else[]
            if complete:assert probes[-1]['step']==1000000 and all(p['finite']for p in probes)
            rows.append(dict(run=rid,environment=identity['environment'],profile=identity['profile'],complete=complete,
                stopped=stopped,per_step=per_step,last3_mean=statistics.mean(per_step.values())if complete else None,
                last_training_step=probes[-1]['step']if probes else None))
        byid={r['run']:r for r in rows};paired=[]
        for pair in plan['pairs']:
            checks={}
            for step in helper.STEPS:
                if not all(f'{pair[k]}:{step}'in cache for k in METHODS):continue
                base=cache[f'{pair["learned_mean"]}:{step}'];bn=base['state']['agent']['network'];entry={}
                for name in('finite_mean','learned_min','finite_min'):
                    other=cache[f'{pair[name]}:{step}'];on=other['state']['agent']['network']
                    entry[name]=dict(BC_params_exact=equal(bn['params']['modules_actor_flow'],on['params']['modules_actor_flow']),
                        BC_Adam_mu_exact=equal(bn['opt_state']['0']['mu']['modules_actor_flow'],on['opt_state']['0']['mu']['modules_actor_flow']),
                        BC_Adam_nu_exact=equal(bn['opt_state']['0']['nu']['modules_actor_flow'],on['opt_state']['0']['nu']['modules_actor_flow']),
                        replay_RNG_exact=equal(base['replay_rng'],other['replay_rng']),agent_RNG_exact=equal(base['state']['agent']['rng'],other['state']['agent']['rng']))
                checks[str(step)]=entry
            values={k:byid[pair[k]]['last3_mean']for k in METHODS}
            delta=lambda a,b:None if values[a]is None or values[b]is None else values[a]-values[b]
            paired.append(dict(environment=pair['environment'],last3=values,
                finite_minus_learned_mean=delta('finite_mean','learned_mean'),learned_mean_minus_min=delta('learned_mean','learned_min'),
                finite_minus_learned_min=delta('finite_min','learned_min'),finite_mean_minus_min=delta('finite_mean','finite_min'),
                actual_BC_moment_RNG_checks=checks,
                all_actual_controls_exact=all(all(all(d.values())for d in step.values())for step in checks.values())if len(checks)==3 else None))
        methods=[]
        for name in METHODS:
            chosen=[byid[p[name]]for p in plan['pairs']];complete=all(r['complete']for r in chosen)
            methods.append(dict(method=name,completed_tasks=sum(r['complete']for r in chosen),
                three_task_mean=statistics.mean(r['last3_mean']for r in chosen)if complete else None))
        complete=all(r['complete']for r in rows)
        terminal=all(r['complete']or r['stopped']is not None for r in rows)
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),status='complete'if complete else'finished_with_stopped_runs'if terminal else'pilot_stopped_requires_review'if any(r['stopped']for r in rows)else'running',rows=rows,
            paired_contrasts=paired,methods=methods,actual_checkpoint_pins=pins,
            registration_sha256=helper.digest(args.registration),new_GPU_jobs=0,new_policy_evaluations=0,
            nine_task_mean=None,goal_complete=False,automatic_promotion=False,
            limits=['Three selected discovery tasks/seed0, not nine tasks or independent confirmation.',
                'Old learned-min models trained with inference configN32; their pinned N9 evaluations are used here. BC training does not query N, but actual BC equivalence is checked.',
                'Nonexact actual BC controls require a common-BC counterfactual before a value-only interpretation.',
                'Finite mean and learned mean retain2V fits, but the finite method does not use learnedV in its target. No equal overall training-compute claim.',
                'This ablation does not alter the already locked finite-mean candidate or fresh9/10/11 validation.',
                'Uniform9/fresh unused seeds would be required before promoting any newly selected alternate method.']))
        if terminal:return
        time.sleep(30)


if __name__=='__main__':main()
