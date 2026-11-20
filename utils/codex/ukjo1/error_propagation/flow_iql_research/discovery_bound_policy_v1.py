"""Exact fixed-metric futility and one registered exploratory candidate choice.

Rejecting a discovery attempt never terminates its learners/evaluations. The
success gates and requirement for complete full9/fresh confirmation are fixed.
"""
import math


STEPS=('800000','900000','1000000')


def empirical_bound(rows,recipe,plan):
    chosen=[r for r in rows if r['recipe']==recipe and r['seed']==0]
    expected=set(plan['scheduling_amendment']['all_nine_environments'])
    if len(chosen)!=9 or {r['environment']for r in chosen}!=expected:return None
    known=[]
    for row in chosen:
        values=(row.get('per_step')or{}).get('primary',{})
        assert set(values)<=set(STEPS)
        for step,rate in values.items():
            count=round(float(rate)*50)
            assert 0<=count<=50 and abs(float(rate)-count/50)<1e-12
            known.append(dict(run=row['run'],environment=row['environment'],step=int(step),successes=count,episodes=50))
    denominator=1350;observed=sum(x['successes']for x in known);remaining=(27-len(known))*50
    threshold=plan['thresholds']['historical_lean_seed0_nine_task_reference']-plan['thresholds']['mean_loss_tolerance']
    minimum=math.ceil((threshold-1e-12)*denominator)
    return dict(recipe=recipe,known_points=known,observed_points=len(known),observed_successes=observed,
        missing_points=27-len(known),remaining_best_case_successes=remaining,denominator=denominator,
        upper_bound=(observed+remaining)/denominator,minimum_total_successes=minimum,
        unchanged_historical_minimum=threshold,impossible_even_if_all_missing_perfect=observed+remaining<minimum,
        scope='Deterministic bound on fixed seed0last3CPU50 metric only; no population estimate or instruction to stop learners.')


def preferred_candidate(state,plan,candidates):
    preference=plan.get('full9_candidate_preference')
    if not preference:return None
    recipe=preference['recipe']
    if recipe in state['tested_candidates']or any(a['recipe']==recipe for a in state['attempts']):return None
    number=preference['after_attempt']
    if len(state['attempts'])<=number:return None
    trigger=state['attempts'][number]
    if trigger['recipe']!=preference['after_recipe']or trigger['status']not in (
        'discovery_historical_gate_impossible','discovery_performance_failed','performance_failed','speed_failed'):return None
    jobs=[j for j in state['jobs']if j['args']['recipe']==recipe and j['args']['seed']==0]
    assert len(jobs)==9 and {j['args']['env']for j in jobs}==set(plan['scheduling_amendment']['all_nine_environments'])
    actual=next((r for r in candidates if r['recipe']==recipe),None)
    if actual is not None and actual['eligible']:
        return dict(actual,selection_kind='registered_preference_after_completed_pilot_screen',
            preference_evidence=preference['evidence'])
    return dict(recipe=recipe,eligible=False,original_pilot_screen_complete=actual is not None,
        selection_kind='registered_exploratory_full9_native_mixture',exploratory_promotion=True,
        preference_evidence=preference['evidence'],reason=preference['reason'],
        no_original_pilot_pass_claim=True,final_full9_fresh_and_speed_gates_unchanged=True)
