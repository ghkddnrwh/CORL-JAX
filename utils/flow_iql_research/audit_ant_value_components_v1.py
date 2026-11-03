"""Four actual Gaussian value interventions at the same immutable 500K step."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.audit_gaussian_factorial_v1 import finite


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();verifier=Verifier();step=500000;rows=[];replay=[];agent_keys=[];common_config=None
    names=[('native','w3_ant_official_g999_t90_s0'),
        ('two_V_fits_fresh_unbounded','gauss_ablation_ant_v2_g999_t90_s0'),
        ('one_V_fit_old_bootstrap_projected','gauss_ablation_ant_bound_g999_t90_s0'),
        ('two_V_fits_fresh_projected','w1_ant_bounded_v2_g999_t90_s0')]
    fields=('actor_hidden_dims','value_hidden_dims','actor_loss','const_std',
            'layer_norm','actor_layer_norm','alpha','expectile','discount','tau','lr','batch_size')
    for label,name in names:
        run=root/'runs'/name;config=json.loads((run/'config.json').read_text());identity=config['identity'];cfg=identity['config']
        assert identity['environment']=='antmaze-large-navigate-singletask-v0'and identity['seed']==0 and identity['kind']=='iql'
        selected={field:cfg[field]for field in fields}
        # Public IQL hardcodes this constructor argument instead of storing it
        # in get_config(). Read its pinned source rather than inventing a field.
        assert 'state_dependent_std=False,'in Path('utils/flow_iql_research/vendor/iql.py').read_text()
        assert cfg['const_std']and cfg['actor_loss']=='awr'
        selected['actor_constructor_state_dependent_std']=False
        if common_config is None:common_config=selected
        else:assert selected==common_config
        for source,digest in config['source_sha256'].items():assert verifier.digest(source)==digest,source
        checkpoint=run/f'checkpoint_{step}.pkl'
        if not checkpoint.exists():checkpoint=root/'analysis/checkpoint_archive'/name/f'checkpoint_{step}.pkl'
        with checkpoint.open('rb')as stream:saved=pickle.load(stream)
        assert saved['step']==int(saved['state']['q_updates'])==step
        assert json.loads(json.dumps(saved['identity']))==identity
        assert saved['source_sha256']==config['source_sha256']and finite(saved['state'])
        profile=identity['profile'];vsteps=2 if profile in ('v2','bounded_v2')else 1
        assert int(saved['state']['v_updates'])==vsteps*step
        assert int(saved['state']['agent']['network']['opt_state']['0']['count'])==step
        own_count=int(saved['state']['v_opt']['0']['count'])
        assert own_count==(0 if profile=='official'else vsteps*step)
        replay.append(saved['replay_rng']);agent_keys.append(saved['state']['agent']['rng'])
        record=None;record_hash=None
        with(run/'diagnostics.jsonl').open()as stream:
            for line in stream:
                if not line.endswith('\n'):continue
                item=json.loads(line)
                if item['step']==step:
                    record=item;record_hash=hashlib.sha256(line.encode()).hexdigest();break
        assert record is not None and record['finite']
        bounds=json.loads((run/'dataset.json').read_text());assert bounds['return_upper']==0
        error_bound=float(np.hypot(record['q_above_upper_rms'],record['q_below_lower_rms']))
        rows.append(dict(label=label,run=name,step=step,profile=profile,V_Adam_updates=vsteps*step,
            Q_mean=record['q_mean'],Q_RMS=record['q_rms'],V_mean=record['v_mean'],
            known_return_Q_error_RMS_lower_bound=error_bound,return_interval=[bounds['return_lower'],bounds['return_upper']],
            expectile_score_RMS_proxy=record['score_rms'],value_loss_before_fit_window=record['value/value_loss'],
            value_loss_after_fit_window=record.get('value/post_fit_loss'),
            all_actual_state_arrays_finite=True,checkpoint_sha256=verifier.digest(checkpoint),
            diagnostic_record_sha256=record_hash,config_sha256=verifier.digest(run/'config.json')))
    for saved in replay[1:]:
        assert all(np.array_equal(np.asarray(a),np.asarray(b))for a,b in zip(replay[0],saved))
    assert all(np.array_equal(agent_keys[0],key)for key in agent_keys[1:])
    result=dict(time=time.time(),kind='matched_500K_Gaussian_value_component_stability_audit',rows=rows,
        common_scientific_config=common_config,replay_RNG_states_exact=True,agent_RNG_states_exact=True,
        certificate='FortrueQin[L,0],RMSerror>=RMSdistanceofpredictedQto[L,0]. Bothheadsandfixed1024dataprobes;smalllowerbounddoesnotproveaccuracy.',
        interpretation='At500Kbothunprojectedpackageshavepositiveout-of-rangeQ,includingtwoVfits+fresh. Bothprojectedpackagesareboundedinscale. EvidencefavorstargetprojectionasthestabilityfactorinthisoneAntstresscase;extraVfits+freshaloneisinsufficient.',
        limits=['Oneenvironment/trainingseedandonecommonstep;notgeneralizedcausalorperformanceconfirmation.',
                'ThepackagefactorcombinesVfitcountandfreshbootstrap;individualeffectsarenotisolated.',
                'Finitestatearraysdonotmeanstablevalues;theunprojectedQpredictionsviolateknownnonpositive-returnbounds.',
                'Theablationrunsarestilltraining;onlyactual800K/900K/1MCPU50canestablishtheirfinalperformance.',
                'Projectiondoesnotboundeveryneuralpredictionorcertifyinside-rangeQaccuracy.'],goal_complete=False)
    path=root/'analysis/Ant_value_component_stability_500K_v1.json';path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(rows=[{key:row[key]for key in ['label','Q_mean','Q_RMS','known_return_Q_error_RMS_lower_bound']}for row in rows],replay_and_agent_RNG_exact=True)),flush=True)


if __name__=='__main__':main()
