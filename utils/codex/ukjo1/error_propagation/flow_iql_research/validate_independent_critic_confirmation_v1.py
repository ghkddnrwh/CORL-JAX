"""Exercise promotion safeguards without launching or modifying real GPU jobs."""
import json
from pathlib import Path
import tempfile

from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import promote, summarize


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); out = root/'analysis'; out.mkdir(); (root/'runs').mkdir()
        jobs = [dict(id=f'candidate_{i}',args=dict(seed=i)) for i in range(54)]
        old = [dict(id='old0'),dict(id='fair_control'),dict(id='old1')]
        (root/'queue.json').write_text(json.dumps(old)); original = (root/'queue.json').read_bytes()
        plan = dict(training_jobs=jobs,output_directory=str(out),preserve_front_control_ids=['fair_control'])
        evidence = dict(status='complete',errors=[],valid_evaluations=54,discovery_qualifies=True,
                        selected_rule='mean',selected_nine_task_mean=.48)
        rejected = []
        for changed in (dict(status='incomplete'),dict(valid_evaluations=53),dict(errors=['bad']),
                        dict(discovery_qualifies=False),dict(selected_rule='median_rank')):
            try: promote(root,plan,'mean',evidence|changed,'plan_pin')
            except AssertionError: rejected.append(changed)
            else: raise AssertionError(f'Unsafe promotion {changed}')
            assert (root/'queue.json').read_bytes() == original
            assert not (out/'promotion_receipt.json').exists()
        receipt = promote(root,plan,'mean',evidence,'plan_pin')
        queue = json.loads((root/'queue.json').read_text())
        assert queue == [old[1]]+jobs+[old[0],old[2]]
        after = (root/'queue.json').read_bytes(); receipt_bytes = receipt.read_bytes()
        promote(root,plan,'mean',evidence,'plan_pin')
        assert (root/'queue.json').read_bytes() == after and receipt.read_bytes() == receipt_bytes
        try: promote(root,plan,'mean',evidence,'changed_plan')
        except AssertionError: pass
        else: raise AssertionError('Changed promotion plan accepted')
        assert (root/'queue.json').read_bytes() == after
        environments = [f'env_{i}' for i in range(9)]
        summary_plan = dict(environments=environments,fresh_groups=[[3,4,5],[6,7,8]],threshold=.47)
        known = {}; valid = {}
        for group in range(2):
            for env in environments:
                for step in (800000,900000,1000000):
                    key = f'group{group+1}_{env}_{step}'
                    known[key] = dict(id=key,environment=env,step=step)
                    valid[key] = dict(success=.48 if group == 0 else .50,parent_learners_complete=True)
        complete = summarize(summary_plan,'mean',known,valid,evidence)
        assert complete['all_fresh_complete'] and abs(complete['fresh_two_group_nine_task_mean']-.49)<1e-12
        assert complete['fresh_exceeds_threshold'] and complete['all_groups_exceed_threshold'] and not complete['goal_complete']
        key = next(iter(valid)); valid[key]['parent_learners_complete'] = False
        pending = summarize(summary_plan,'mean',known,valid,evidence)
        assert not pending['all_fresh_complete'] and pending['fresh_two_group_nine_task_mean'] is None
        valid[key]['parent_learners_complete'] = True; del valid[key]
        missing = summarize(summary_plan,'mean',known,valid,evidence)
        assert not missing['all_fresh_complete'] and missing['fresh_two_group_nine_task_mean'] is None
        valid = {key:dict(success=.46,parent_learners_complete=True) for key in known}
        failed = summarize(summary_plan,'mean',known,valid,evidence)
        assert failed['all_fresh_complete'] and not failed['fresh_exceeds_threshold']
        assert not failed['all_groups_exceed_threshold'] and not failed['goal_complete']
    print(json.dumps(dict(passed=True,rejected_promotion_conditions=len(rejected),
        receipt_and_queue_idempotent=True,changed_plan_rejected=True,protected_controls_kept_first=True,
        missing_or_unfinished_not_zero=True,complete_failure_not_goal_success=True)))


if __name__ == '__main__': main()
