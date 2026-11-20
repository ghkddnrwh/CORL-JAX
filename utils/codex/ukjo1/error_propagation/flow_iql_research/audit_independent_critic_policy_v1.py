"""Strict raw-episode, source and actual-model audit for ensemble evaluations."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier,provided_thresholds


def audit_job(plan,job,verifier):
    path=Path(job['output']);result=json.loads(path.read_text());step=job['step']
    assert result['kind']=='independent_critic_ensemble_policy_discovery'
    assert result['environment']==job['environment']and result['step']==result['learner_update_step']==step
    assert result['aggregation_rule']==job['rule']and result['selection_samples']==9
    assert result['num_independent_learners']==3 and result['num_Q_heads']==6
    assert result['total_Q_updates_across_members']==3*step and result['total_V_updates_across_members']==6*step
    assert result['n_episodes']==50 and result['first_episode_seed']==42 and result['eval_device']=='cpu'
    assert result['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
    for field in('episode_successes','episode_rewards','episode_lengths'):
        assert len(result[field])==50 and np.isfinite(result[field]).all()
    assert set(result['episode_successes'])<={0.,1.}
    assert abs(statistics.mean(result['episode_successes'])-result['success_rate'])<1e-12
    assert not result['primary_evaluations_modified']and not result['mixed_joint_learning_state_saved']
    assert result['selector_source_sha256']==plan['source_sha256']['utils/flow_iql_research/independent_critic_policy_v1.py']
    assert result['evaluator_source_sha256']==plan['source_sha256']['utils/flow_iql_research/evaluate_independent_critic_policy_v1.py']
    assert len(result['critic_members'])==len(job['critic_runs'])==len(job['member_pins'])==3
    complete=True;identities=[]
    for member,run_name,pin in zip(result['critic_members'],job['critic_runs'],job['member_pins']):
        run=Path(run_name);config=json.loads((run/'config.json').read_text());identity=config['identity'];cfg=identity['config']
        assert identity['environment']==job['environment']and identity['kind']=='ifql'and identity['profile']=='bounded_v2'
        assert identity['steps']==1000000 and identity['episodes']==50
        assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']in(9,32)
        assert member['training_seed']==identity['seed']and member['run']==run.name
        assert member['actual_Q_Adam_count']==step and member['actual_V_Adam_count']==2*step
        assert member['config_sha256']==pin['config_sha256']==verifier.digest(run/'config.json')
        for source,digest in config['source_sha256'].items():assert verifier.digest(source)==digest,source
        cp=run/f'checkpoint_{step}.pkl';assert Path(member['checkpoint']).resolve()==cp.resolve()
        assert member['checkpoint_sha256']==pin['checkpoint_sha256']==verifier.checkpoint(cp,config,step)
        if(run/'stopped.json').exists():raise AssertionError(f'Ineligible stopped learner {run}')
        if not(run/'completed.json').exists():complete=False
        else:
            done=json.loads((run/'completed.json').read_text());assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
        identities.append(identity)
    assert len({identity['seed']for identity in identities})==3
    configs=[{k:v for k,v in identity['config'].items()if k!='num_samples'}for identity in identities]
    assert configs[0]==configs[1]==configs[2]
    assert Path(job['actor_run'])==Path(job['critic_runs'][0])
    assert result['actor_training_seed']==identities[0]['seed']and result['actor_checkpoint_sha256']==job['member_pins'][0]['checkpoint_sha256']
    return dict(success=result['success_rate'],parent_learners_complete=complete,result_sha256=verifier.digest(path),
        learner_seeds=[identity['seed']for identity in identities])


def audit(plan,verifier):
    for source,digest in plan['source_sha256'].items():assert verifier.digest(source)==digest,source
    assert len(plan['jobs'])==54 and len(plan['environments'])==len(set(plan['environments']))==9
    valid={};errors=[]
    for job in plan['jobs']:
        if not Path(job['output']).exists():continue
        try:valid[job['id']]=audit_job(plan,job,verifier)
        except(AssertionError,KeyError,ValueError,OSError)as error:errors.append(dict(job=job['id'],error=repr(error)))
    methods=[]
    for rule in('mean','median_rank'):
        tasks=[];rule_errors=[error for error in errors if next(job for job in plan['jobs']if job['id']==error['job'])['rule']==rule]
        for environment in plan['environments']:
            jobs=[job for job in plan['jobs']if job['rule']==rule and job['environment']==environment]
            assert len(jobs)==3 and sorted(job['step']for job in jobs)==[800000,900000,1000000]
            available=[valid.get(job['id'])for job in jobs]
            complete=all(row is not None and row['parent_learners_complete']for row in available)
            scores={str(job['step']):valid[job['id']]['success']if job['id']in valid else None for job in jobs}
            tasks.append(dict(environment=environment,complete=complete,per_step=scores,
                last3_mean=statistics.mean(scores.values())if complete else None))
        count=sum(row['complete']for row in tasks)
        methods.append(dict(rule=rule,completed_tasks=count,tasks=tasks,
            nine_task_mean=statistics.mean(row['last3_mean']for row in tasks)if count==9 and not rule_errors else None))
    complete=len(valid)==54 and not errors and all(method['nine_task_mean']is not None for method in methods)
    selected=max(methods,key=lambda method:(method['nine_task_mean'],method['rule']=='mean'))if complete else None
    return dict(time=time.time(),status='complete'if complete else'incomplete',valid_evaluations=len(valid),errors=errors,
        methods=methods,selected_rule=selected['rule']if selected else None,
        selected_nine_task_mean=selected['nine_task_mean']if selected else None,
        discovery_qualifies=bool(complete and selected['nine_task_mean']>.47),
        result_sha256={job['id']:valid[job['id']]['result_sha256']for job in plan['jobs']if job['id']in valid},
        unique_model_checkpoints_verified=len(verifier.checkpoints),one_ensemble_group_only=True,
        fresh_confirmation_groups=plan['fresh_confirmation_groups'],goal_complete=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());verifier=Verifier();result=audit(plan,verifier)
    result.update(plan_sha256=verifier.digest(args.plan),audit_source_sha256=verifier.digest(__file__),
        scope='Independentcriticdiscoverygroup0;strict54rawCPU50andactualmembermodelchecks. Missingcellsarenotzero.')
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps({key:result[key]for key in('status','valid_evaluations','errors','unique_model_checkpoints_verified','selected_rule','selected_nine_task_mean','discovery_qualifies')}),flush=True)


if __name__=='__main__':main()
