"""Use the user's nine-task MACRO goal; task losses remain diagnostics.

Registered before any CPU10/CPU50 outcome. Preserve V1 and its receipts. The
user explicitly permits some tasks to fall provided average performance holds;
a hard per-task cap would add an unnecessary requirement to that objective.
"""
from utils.flow_iql_research import watch_onestep_deploy_v1 as previous


original_pilot_candidates=previous.pilot_candidates
original_confirmation=previous.full_confirmation


def pilot_candidates(rows,recipes,plan):
    result=original_pilot_candidates(rows,recipes,plan)
    limit=plan['thresholds']['mean_loss_tolerance']
    for candidate in result:
        candidate['eligible']=candidate['pilot_primary_mean']>=candidate['pilot_teacher_mean']-limit-1e-12
        candidate['eligible']=candidate['eligible']and candidate['pilot_primary_mean']>=plan['thresholds']['pilot_historical_mean_reference']-limit-1e-12
        candidate['eligibility_scope']='Macro averages govern selection; per-task loss vectors are diagnostic, not a hard veto.'
    return result


def full_confirmation(rows,attempt,plan):
    result=original_confirmation(rows,attempt,plan)
    if result is not None:
        result['diagnostic_task_loss_cap_satisfied']=result['checks'].pop('fresh_max_task_loss')
        result['performance_passed']=all(result['checks'].values())
        result['eligibility_scope']='Nine-task means and fresh-seed macro confirmation; task-level regressions remain fully reported.'
    return result


if __name__=='__main__':
    previous.pilot_candidates=pilot_candidates
    previous.full_confirmation=full_confirmation
    previous.main()
