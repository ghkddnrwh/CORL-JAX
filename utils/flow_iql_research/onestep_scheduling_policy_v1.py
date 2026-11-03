"""Pure scheduling/staged-confirmation rules; never alter recipe or scores."""
import statistics


def order_pending(plan,state,queue,started):
    """Fill a recipe's missing pilot environment, prioritize confirmation work.

    Keep every job definition and the positions of already-started jobs. The
    logical registration order resolves ties. This function performs no I/O.
    """
    registered={job['id']:job for job in state['jobs']}
    assert len({job['id']for job in queue})==len(queue)
    assert all(job['id']in registered and job==registered[job['id']]for job in queue)
    logical={job['id']:index for index,job in enumerate(state['jobs'])}
    coverage={}
    for job in state['jobs']:
        args=job['args']
        if job['id']in started and args['seed']==0 and args['env']in plan['pilot_environments']:
            coverage.setdefault(args['recipe'],set()).add(args['env'])
    attempt=state['attempts'][state['active_attempt']]if state.get('active_attempt')is not None else None
    def priority(job):
        args=job['args']
        confirm=attempt and state['phase']=='confirmation'and args['recipe']==attempt['recipe']and (
            (attempt['stage']=='discovery'and args['seed']==0)or
            (attempt['stage']=='fresh'and args['seed']in attempt['fresh_seeds']))
        missing=args['seed']==0 and args['env']in plan['pilot_environments']and len(coverage.get(args['recipe'],set()))>=2 and args['env']not in coverage[args['recipe']]
        return (0 if confirm else 1 if missing else 2,logical[job['id']])
    pending=iter(sorted((job for job in queue if job['id']not in started),key=priority))
    ordered=[job if job['id']in started else next(pending)for job in queue]
    assert {job['id']for job in ordered}=={job['id']for job in queue}
    assert all(left==right for left,right in zip(queue,ordered)if left['id']in started)
    return ordered


def discovery_confirmation(rows,attempt,plan,environments):
    chosen=[row for row in rows if row['recipe']==attempt['recipe']and row['seed']==0]
    if len(chosen)!=9 or not all(row['complete']for row in chosen):return None
    assert {row['environment']for row in chosen}==set(environments)and len(environments)==9
    primary=statistics.mean(row['primary_mean']for row in chosen)
    teacher=statistics.mean(row['teacher_mean']for row in chosen);limits=plan['thresholds']
    checks=dict(seed0_paired_mean=primary>=teacher-limits['mean_loss_tolerance']-1e-12,
        seed0_historical_mean=primary>=limits['historical_lean_seed0_nine_task_reference']-limits['mean_loss_tolerance']-1e-12)
    return dict(recipe=attempt['recipe'],seed=0,primary=primary,teacher=teacher,checks=checks,
        discovery_passed=all(checks.values()),scope='The same two discovery macro gates used by full confirmation; fresh seeds remain mandatory.')
