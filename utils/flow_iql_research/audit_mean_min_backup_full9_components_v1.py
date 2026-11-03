"""Compare all nine finite-backup min/mean controls on actual late models."""
import argparse
import json
from pathlib import Path
import pickle
import statistics
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.audit_probability_value_pilots_v1 import equal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();plan=json.loads(args.registration.read_text());root=Path(plan['root'])
    verifier=Verifier();rows=[];pins={}
    for job in plan['all9jobs']:
        short=job['id'].split('_')[1]
        names={'min':f'eb1_{short}_bounded_v2_empirical_backup_g999_t90_s0','mean':job['id']}
        configs={mode:json.loads((root/'runs'/rid/'config.json').read_text())for mode,rid in names.items()}
        common={}
        for mode,conf in configs.items():
            ident=conf['identity'];cfg=ident['config']
            assert ident['environment']==job['args']['env']and ident['seed']==0 and ident['kind']=='ifql'
            assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']==cfg['backup_samples']==9
            assert ident['steps']==1000000 and ident['episodes']==50
            common[mode]={k:v for k,v in cfg.items()if k not in('backup_kind','backup_Q_aggregation')}
            for path,sha in conf['source_sha256'].items():assert verifier.digest(path)==sha,path
            done=json.loads((root/'runs'/names[mode]/'completed.json').read_text())
            assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
        assert common['min']==common['mean']
        cells=[]
        for step in helper.STEPS:
            states={};scores={}
            for mode,rid in names.items():
                run=root/'runs'/rid;cp=run/f'checkpoint_{step}.pkl';sha=verifier.checkpoint(cp,configs[mode],step)
                with cp.open('rb')as stream:states[mode]=pickle.load(stream)
                assert verifier.digest(cp)==sha
                pins[f'{rid}:{step}']=dict(path=str(cp),sha256=sha,full_saved_state_finite=True,Q_Adam=step,V_Adam=2*step)
                scores[mode]=helper.primary_record(run,step)['success_rate']
            a,b=states['min'],states['mean'];an=a['state']['agent']['network'];bn=b['state']['agent']['network']
            checks=dict(BC_params_exact=equal(an['params']['modules_actor_flow'],bn['params']['modules_actor_flow']),
                BC_Adam_mu_exact=equal(an['opt_state']['0']['mu']['modules_actor_flow'],bn['opt_state']['0']['mu']['modules_actor_flow']),
                BC_Adam_nu_exact=equal(an['opt_state']['0']['nu']['modules_actor_flow'],bn['opt_state']['0']['nu']['modules_actor_flow']),
                replay_RNG_exact=equal(a['replay_rng'],b['replay_rng']),agent_RNG_exact=equal(a['state']['agent']['rng'],b['state']['agent']['rng']))
            cells.append(dict(step=step,scores=scores,checks=checks))
        rows.append(dict(environment=job['args']['env'],cells=cells,
            min_backup_last3=statistics.mean(c['scores']['min']for c in cells),
            mean_backup_last3=statistics.mean(c['scores']['mean']for c in cells),
            all_actual_BC_moments_and_RNG_exact=all(all(c['checks'].values())for c in cells)))
    assert len(rows)==9 and len(pins)==54
    result=dict(time=time.time(),status='complete',rows=rows,actual_checkpoint_pins=pins,actual_unique_late_models_verified=54,
        min_backup_nine_task_mean=statistics.mean(r['min_backup_last3']for r in rows),
        mean_backup_nine_task_mean=statistics.mean(r['mean_backup_last3']for r in rows),
        all_actual_BC_moments_and_RNG_exact=all(r['all_actual_BC_moments_and_RNG_exact']for r in rows),
        registration_sha256=verifier.digest(args.registration),source_sha256=verifier.digest(__file__),goal_complete=False,
        limits=['One exploratory seed0 control, not independent confirmation or a same-seed FQL comparison.',
            'Both methods use finite expectiles and projected targets. This contrast tests target twin-Q mean versus minimum, conditional on their actual trained BC weights.',
            'BC differences, if present, prohibit a value-only interpretation without a common-actual-BC counterfactual.',
            'Removing learned-V residual alone is not shown sufficient: the full9 finite-min method scored.4. A learned-V mean-head ablation is still missing.',
            'Finite-proposal expectile residual is not population-behavior residual. Critic/model/sampling errors remain.'])
    helper.atomic(args.output,result)
    print(json.dumps({k:result[k]for k in('status','actual_unique_late_models_verified','min_backup_nine_task_mean','mean_backup_nine_task_mean','all_actual_BC_moments_and_RNG_exact')}),flush=True)


if __name__=='__main__':main()
