"""Compare a completed policy flow with its SAME-model best9 BC reference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def episodes(row):
    assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
    assert row['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
    for field in('episode_successes','episode_rewards','episode_lengths'):
        assert len(row[field])==50 and np.isfinite(row[field]).all()
    assert set(row['episode_successes'])<={0.,1.}
    assert abs(statistics.mean(row['episode_successes'])-row['success_rate'])<1e-12


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    run=args.run.resolve();root=run.parent.parent;verifier=Verifier();config=json.loads((run/'config.json').read_text())
    identity=config['identity'];cfg=identity['config']
    assert identity['kind']=='rank_flow'and identity['profile']=='bounded_v2_rank_flow'
    assert identity['steps']==1000000 and identity['episodes']==50 and cfg['num_samples']==1 and cfg['selector_teacher_samples']==9
    assert cfg['discount']==.999 and cfg['expectile']==.9
    for path,digest in config['source_sha256'].items():assert verifier.digest(path)==digest,path
    done=json.loads((run/'completed.json').read_text());assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
    primary=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()];rows=[]
    folder=root/'analysis/rank_policy_diagnostics_v1'
    for step in(800000,900000,1000000):
        student=next(row for row in primary if row['step']==step);episodes(student)
        cp_sha=verifier.checkpoint(run/f'checkpoint_{step}.pkl',config,step)
        teacher_path=folder/f'{run.name}_{step}_teacher_cpu50.json';teacher=json.loads(teacher_path.read_text());episodes(teacher)
        assert teacher['policy_mode']=='teacher'and teacher['environment']==identity['environment']and teacher['training_seed']==identity['seed']
        assert teacher['checkpoint_sha256']==cp_sha and teacher['config_sha256']==verifier.digest(run/'config.json')
        assert teacher['evaluator_source_sha256']==verifier.digest('utils/flow_iql_research/evaluate_rank_flow_teacher_v1.py')
        diagnostic_path=folder/f'{run.name}_{step}_distribution.json';diagnostic=json.loads(diagnostic_path.read_text())
        assert diagnostic['checkpoint_sha256']==cp_sha and diagnostic['config_sha256']==verifier.digest(run/'config.json')
        assert diagnostic['diagnostic_source_sha256']==verifier.digest('utils/flow_iql_research/rank_flow_policy_diagnostics_v1.py')
        s=np.asarray(student['episode_successes']);t=np.asarray(teacher['episode_successes'])
        rows.append(dict(step=step,student_success=student['success_rate'],same_model_BC_best9_success=teacher['success_rate'],
            student_minus_reference=student['success_rate']-teacher['success_rate'],
            paired_resets=dict(student_only_success=int(np.sum((s==1)&(t==0))),reference_only_success=int(np.sum((s==0)&(t==1))),both_success=int(np.sum((s==1)&(t==1)))),
            distribution_diagnostics=diagnostic['summary'],checkpoint_sha256=cp_sha,
            teacher_evaluation_sha256=verifier.digest(teacher_path),distribution_sha256=verifier.digest(diagnostic_path),
            actual_Q_Adam_count=step,actual_V_Adam_count=2*step,full_state_finite=True))
    result=dict(time=time.time(),kind='completed_same_model_policy_student_reference_last3_audit',run=run.name,
        environment=identity['environment'],training_seed=identity['seed'],rows=rows,
        student_last3_mean=statistics.mean(row['student_success']for row in rows),
        same_model_reference_last3_mean=statistics.mean(row['same_model_BC_best9_success']for row in rows),
        last3_paired_difference=statistics.mean(row['student_minus_reference']for row in rows),
        unique_actual_model_checkpoints_audited=len(verifier.checkpoints),
        primary_evaluation_sha256=verifier.digest(run/'evaluations.jsonl'),config_sha256=verifier.digest(run/'config.json'),
        source_sha256=verifier.digest(__file__),
        interpretation='CompareactualdeployedpolicyflowsversusBCbest9ontheexactsameQ/BCmodels;noinferencecriticsourcemixingandnonewtrainingseed.',
        limits=['Oneenvironment/seed,notanine-taskwinortraining-seedreplication.',
            'Fixed-datasetFM/energydistance/modelQproxiesarenotguaranteesofpolicyreturn.',
            'HigherteacherQonfixedqueriesandloweractualsuccessdonotprovideoracleQerrorlabels;objectivesanddeploymentstatesdiffer.',
            'Pairedresetcountsaredescriptive,notsuccessfrom150independenttrainingruns.'],goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({key:result[key]for key in('environment','student_last3_mean','same_model_reference_last3_mean','last3_paired_difference','unique_actual_model_checkpoints_audited')}),flush=True)


if __name__=='__main__':main()
