"""Protect the full nine-task objective from over-strict pilot screening.

Prefer recipes passing the three-task screen. If none remain, allow at most
two exploratory full-nine-task/fresh-seed confirmations from different loss
families. A failed pilot heuristic must not substitute for the actual nine-task
goal. Completion criteria and actual evaluation protocol remain unchanged.
"""
from utils.flow_iql_research import watch_onestep_deploy_v2 as macro


original_pick=macro.previous.pick_candidate


def family(recipe):
    stem=recipe.split('__')[0]
    return 'fqlraw'if stem.startswith('fqlraw_x')else stem.rsplit('_a',1)[0]


def pick_candidate(candidates,tested):
    qualified=original_pick(candidates,tested)
    if qualified is not None:return dict(qualified,selection_kind='passed_pilot_macro_screen')
    explored=[r for r in candidates if r['recipe']in tested and not r['eligible']]
    if len(explored)>=2:return None
    remaining=[r for r in candidates if r['recipe']not in tested]
    if not remaining:return None
    used={family(r['recipe'])for r in explored}
    different=[r for r in remaining if family(r['recipe'])not in used]
    pool=different or remaining
    # Reuse the same performance/simplicity ordering, preserving factual flags.
    picked=original_pick([dict(r,eligible=True)for r in pool],tested)
    original=next(r for r in pool if r['recipe']==picked['recipe'])
    return dict(original,selection_kind='exploratory_full9_despite_failed_pilot_screen',
        exploratory_promotion=True,reason='Nine-task performance cannot be established or refuted from three selected tasks alone.')


if __name__=='__main__':
    macro.previous.pilot_candidates=macro.pilot_candidates
    macro.previous.full_confirmation=macro.full_confirmation
    macro.previous.pick_candidate=pick_candidate
    macro.previous.main()
