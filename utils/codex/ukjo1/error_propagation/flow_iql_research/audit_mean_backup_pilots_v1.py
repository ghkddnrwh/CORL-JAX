"""Audit actual late models and CPU50 of the three mean-head backup pilots."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();root=args.root.resolve();verifier=Verifier();rows=[]
    for short in('ant','cubed','pz3'):
        run=root/'runs'/f'meb1_{short}_bounded_v2_mean_empirical_backup_g999_t90_s0'
        config=json.loads((run/'config.json').read_text());identity=config['identity'];cfg=identity['config']
        assert identity['kind']=='ifql'and identity['profile']=='bounded_v2_mean_empirical_backup'and identity['seed']==0
        assert identity['steps']==1000000 and identity['episodes']==50
        assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']==cfg['backup_samples']==9
        assert cfg['backup_Q_aggregation']=='mean'and cfg['backup_kind']=='finite_behavior_flow_mean_target_Q_expectile'
        for source,digest in config['source_sha256'].items():assert verifier.digest(source)==digest,source
        if not(run/'completed.json').exists():
            rows.append(dict(environment=identity['environment'],run=run.name,complete=False,last3_mean=None));continue
        completion=json.loads((run/'completed.json').read_text())
        assert completion['step']==completion['q_updates']==1000000 and completion['v_updates']==2000000
        records=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()];cells=[]
        for step in(800000,900000,1000000):
            found=[record for record in records if record['step']==step];assert len(found)==1;record=found[0]
            assert record['n_episodes']==50 and record['first_episode_seed']==42 and record['eval_device']=='cpu'
            assert record['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
            for field in('episode_successes','episode_rewards','episode_lengths'):
                assert len(record[field])==50 and np.isfinite(record[field]).all()
            assert set(record['episode_successes'])<={0.,1.}
            assert abs(statistics.mean(record['episode_successes'])-record['success_rate'])<1e-12
            cells.append(dict(step=step,success=record['success_rate'],
                checkpoint_sha256=verifier.checkpoint(run/f'checkpoint_{step}.pkl',config,step),
                actual_Q_Adam_count=step,actual_V_Adam_count=2*step,full_state_finite=True))
        rows.append(dict(environment=identity['environment'],run=run.name,complete=True,cells=cells,
            last3_mean=statistics.mean(cell['success']for cell in cells),
            config_sha256=verifier.digest(run/'config.json'),evaluation_sha256=verifier.digest(run/'evaluations.jsonl')))
    count=sum(row['complete']for row in rows)
    result=dict(time=time.time(),kind='mean_head_backup_three_pilot_last3_audit',rows=rows,completed_pilots=count,
        pilot_three_task_mean=statistics.mean(row['last3_mean']for row in rows)if count==3 else None,
        nine_task_mean=None,unique_actual_models_audited=len(verifier.checkpoints),source_sha256=verifier.digest(__file__),
        scope='Threepilots,notanine-taskcohort;knowncompletedcellsrequireactualmodels/sources/Adamclocks/rawCPU50. Nozerofill.',goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(completed=count,pilot_three_task_mean=result['pilot_three_task_mean'],
        values={row['environment']:row['last3_mean']for row in rows})),flush=True)


if __name__=='__main__':main()
