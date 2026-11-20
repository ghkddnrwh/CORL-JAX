"""One-shot root audit of matched-N9 native IFQL and retained V2 references."""
import argparse
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.audit_probability_value_pilots_v1 import equal
from utils.flow_iql_research.watch_native_ifql_n9_reference_v1 import validate_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();plan=json.loads(args.registration.read_text())
    helper.check_sources(plan['source_sha256']);verifier=Verifier();pins={};evaluations=[];checks=[]
    assert len(plan['jobs'])==27 and len(plan['environments'])==9
    for job in plan['jobs']:
        native=validate_result(plan,job,verifier)
        reference=Path(job['paired_V2_evaluation']);corrected=json.loads(reference.read_text())
        assert verifier.digest(reference)==plan['paired_V2_evaluation_sha256'][str(reference)]
        helper.check_episodes(corrected,job['step'])
        assert corrected['run']==job['paired_V2_run'] and corrected['selection_samples']==9
        assert corrected['selection_rule']=='max'
        states=[]
        for rid in (job['run_id'],job['paired_V2_run']):
            conf,saved,path,sha=helper.load_parent(plan,rid,job['step']);states.append(saved)
            assert sha==plan['checkpoint_sha256'][f'{rid}:{job["step"]}']
            state=saved['state'];pins[f'{rid}:{job["step"]}']=dict(path=str(path),sha256=sha,
                config_sha256=verifier.digest(Path(plan['root'])/'runs'/rid/'config.json'),
                actual_Q_Adam=int(state['agent']['network']['opt_state']['0']['count']),
                actual_V_total=int(state['v_updates']),actual_separate_V_Adam=int(state['v_opt']['0']['count']),
                all_saved_state_finite=True)
        assert corrected['checkpoint_sha256']==plan['checkpoint_sha256'][f'{job["paired_V2_run"]}:{job["step"]}']
        a,b=states;an=a['state']['agent']['network'];bn=b['state']['agent']['network']
        checks.append(dict(environment=job['environment'],step=job['step'],
            BC_params_exact=equal(an['params']['modules_actor_flow'],bn['params']['modules_actor_flow']),
            BC_Adam_mu_exact=equal(an['opt_state']['0']['mu']['modules_actor_flow'],bn['opt_state']['0']['mu']['modules_actor_flow']),
            BC_Adam_nu_exact=equal(an['opt_state']['0']['nu']['modules_actor_flow'],bn['opt_state']['0']['nu']['modules_actor_flow']),
            replay_rng_exact=equal(a['replay_rng'],b['replay_rng']),agent_rng_exact=equal(a['state']['agent']['rng'],b['state']['agent']['rng'])))
        evaluations.append(dict(environment=job['environment'],step=job['step'],native_N9=native['success_rate'],
            V2_N9=corrected['success_rate'],native_result_sha256=verifier.digest(job['output']),V2_result_sha256=verifier.digest(reference)))
    rows=[]
    for env in plan['environments']:
        chosen=[row for row in evaluations if row['environment']==env]
        assert sorted(row['step']for row in chosen)==[800000,900000,1000000]
        rows.append(dict(environment=env,per_step=chosen,native_N9_last3=statistics.mean(row['native_N9']for row in chosen),
            V2_N9_last3=statistics.mean(row['V2_N9']for row in chosen)))
    assert len(pins)==54
    result=dict(time=time.time(),status='complete',rows=rows,actual_unique_models_verified=54,
        actual_checkpoint_pins=pins,BC_and_RNG_checks=checks,
        native_N9_nine_task_mean=statistics.mean(row['native_N9_last3']for row in rows),
        V2_N9_nine_task_mean=statistics.mean(row['V2_N9_last3']for row in rows),
        all_BC_parameters_exact=all(row['BC_params_exact']for row in checks),
        all_replay_and_agent_RNG_states_exact=all(row['replay_rng_exact']and row['agent_rng_exact']for row in checks),
        source_sha256=verifier.digest(__file__),registration_sha256=verifier.digest(args.registration),goal_complete=False,
        interpretation='MatchedN9 exploratory seed0 reference. Actual BC differences prohibit attributing the entire own-policy contrast solely to the V-update bundle.',
        limits=['All nine tasks and three late CPU50 evaluations are included; no independent training replication.',
            'Native V1M and V2 separate V2M clocks at1M are audited separately.',
            'No episode bootstrap can substitute for training-seed uncertainty.',
            'Equal end RNG states do not prove equal historical in-memory data or numerical trajectories.',
            'Common-actual-BC counterfactuals are registered separately; this audit does not diagnose the cause of BC divergence.'])
    helper.atomic(args.output,result)
    print(json.dumps({k:result[k]for k in ('status','actual_unique_models_verified','native_N9_nine_task_mean','V2_N9_nine_task_mean','all_BC_parameters_exact')}),flush=True)


if __name__=='__main__':main()
