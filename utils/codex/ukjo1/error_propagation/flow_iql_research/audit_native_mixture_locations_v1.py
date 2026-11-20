"""Root review of all four-task same-model decoder comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    area=a.root/'analysis/native_mixture_locations_v1'
    output=area/'COMPLETE_FOUR_TASK_COMMON_DECODER_ROOT_AUDIT_v1.json'
    assert not output.exists()
    proposal=area/'prospective_common_decoder_study_v1.json';plan=json.loads(proposal.read_text())
    for path,digest in plan['source_sha256'].items():
        assert sha(path)==digest,path
    workers=json.loads((area/'worker_handles_v1.json').read_text())
    for worker in workers['workers']:
        state=json.loads((area/f'{worker["case"]}_worker_state_v1.json').read_text())
        assert state['status']=='complete' and state['completed']==state['expected']
    points=[];replay=None
    for job in plan['evaluations']:
        path=Path(job['output']);row=json.loads(path.read_text())
        assert row['run']==Path(job['run']).name and row['step']==job['step'] and row['decoder']==job['decoder']
        assert row['actor_artifact_sha256']==sha(job['actor'])==job['actor_sha256']
        assert row['config_sha256']==sha(Path(job['run'])/'config.json')==job['config_sha256']
        for source,digest in row['source_sha256'].items():
            assert sha(source)==digest
        assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
        assert row['optimizer_updates_performed']==0 and row['action_heads_evaluated']==1
        assert row['learned_model_trees_loaded']==['actor_params']
        assert not row['candidate_bank'] and not row['critic_loaded'] and not row['Flow_loaded']
        assert not row['counts_toward_native_primary']
        for key in ('episode_successes','episode_rewards','episode_lengths'):
            assert len(row[key])==50 and all(math.isfinite(float(v)) for v in row[key])
        assert all(v in (0,1) for v in row['episode_successes'])
        successes=int(sum(row['episode_successes']));assert abs(successes/50-row['success_rate'])<1e-12
        point=dict(case=job['case'],step=job['step'],decoder=job['decoder'],successes=successes,
                   rate=successes/50,path=str(path),sha256=sha(path),actor_sha256=job['actor_sha256'])
        if job['decoder']=='original_sample':
            ref=a.root/'analysis/evaluations_v1'/f"{row['run']}_{job['step']}_primary_n50.json"
            expected=json.loads(ref.read_text())
            assert all(row[key]==expected[key] for key in ('episode_successes','episode_rewards','episode_lengths'))
            replay=dict(**point,sealed_primary_path=str(ref),sealed_primary_sha256=sha(ref),
                        three_episode_arrays_exact=True)
        else:
            points.append(point)
    assert len(points)==24 and replay
    rows=[];references=[]
    for case in plan['cases']:
        run=next(j['run'] for j in plan['evaluations'] if j['case']==case)
        values={}
        for mode in ('primary','teacher'):
            count=0
            for step in plan['steps']:
                path=a.root/'analysis/evaluations_v1'/f'{Path(run).name}_{step}_{mode}_n50.json'
                row=json.loads(path.read_text());assert row['n_episodes']==50 and row['first_episode_seed']==42
                assert row['actor_artifact_sha256']==sha(Path(run)/f'actor_{step}.pkl')
                assert sum(row['episode_successes'])/50==row['success_rate']
                count+=sum(row['episode_successes']);references.append(dict(path=str(path),sha256=sha(path)))
            values['native_stochastic' if mode=='primary' else 'matched_best9_teacher']=count/150
        for decoder in ('sampled_component_location','highest_weight_location'):
            group=[v for v in points if v['case']==case and v['decoder']==decoder]
            assert len(group)==3
            values[decoder]=sum(v['successes'] for v in group)/150
        rows.append(dict(case=case,**values))
    result=dict(time=time.time(),status='pass',proposal_sha256=sha(proposal),source_sha256=sha(__file__),
        rows=rows,decoder_points=points,original_sample_reproduction=replay,sealed_native_references=references,
        actual_new_evaluated_episodes=1250,new_decoder_evaluated_episodes=1200,
        unchanged_trained_native_models=12,optimizer_updates_performed=0,
        subset_means={key:statistics.mean(row[key] for row in rows) for key in values},
        no_nine_task_or_fresh_or_isolated_latency_claim=True,
        scope='Same12nativecheckpoints, commonCPU50/reset42..91/last3. Two fixed global deployment decoders, no task/peak mixing or additional SGD. Four-task seed0 exploratory evidence only.')
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (area/'four_task_common_decoders_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='pass',rows=rows,subset_means=result['subset_means'],new_episodes=1250)))


if __name__=='__main__':
    main()
