"""Hold actual BC weights fixed when comparing native and corrected critics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.watch_native_ifql_n9_reference_v1 import validate_result as native_result


def validate(plan,job,verifier):
    result=json.loads(Path(job['output']).read_text());helper.check_episodes(result,job['step'])
    assert result['value_run']==job['value_run'] and result['density_run']==job['density_run']
    assert result['density_kind']=='ifql' and result['inference_samples']==9
    assert result['retained_Q_updates']==result['retained_behavior_policy_updates']==job['step']
    vcount=plan['expected_final_clocks'][job['value_run']]['V_total']*job['step']//1000000
    assert result['retained_V_updates']==vcount
    assert result['value_checkpoint_sha256']==plan['checkpoint_sha256'][f"{job['value_run']}:{job['step']}"]
    assert result['density_checkpoint_sha256']==plan['checkpoint_sha256'][f"{job['density_run']}:{job['step']}"]
    assert result['source_sha256']==plan['source_sha256']['utils/flow_iql_research/evaluate_factorized_policy_v3.py']
    assert result['selection_source_sha256']==plan['source_sha256']['utils/flow_iql_research/selection_v2.py']
    assert result['evaluation_helper_sha256']==plan['source_sha256']['utils/flow_iql_research/train.py']
    if job['sanity']:
        assert result['all_network_params_equal_density_model'] and result['full_dynamic_state_equal_density_model']
        original=json.loads(Path(job['V2_N9_reference']).read_text())
        for field in ('episode_successes','episode_rewards','episode_lengths'):
            assert original[field]==result[field],f'Same-model N9 rollout mismatch: {job["id"]} {field}'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    args=parser.parse_args();plan=json.loads(args.registration.read_text());root=Path(plan['root'])
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verifier=Verifier();native_plan=json.loads(Path(plan['native_reference_registration']).read_text())
    running={};valid={};failed={};audited={}
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MUJOCO_GL='egl',PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    while True:
        helper.check_sources(plan['source_sha256'])
        assert verifier.digest(plan['native_reference_registration'])==plan['native_reference_registration_sha256']
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is not None:
                row['log'].close();del running[key]
                if code:failed[key]=dict(exit_code=code,log=str(row['path']))
        for job in plan['jobs']:
            if Path(job['output']).exists():
                result=validate(plan,job,verifier);valid[job['id']]=result['success_rate']
        sanity=[j for j in plan['jobs']if j['sanity']]
        sanity_complete=all(j['id']in valid for j in sanity)
        ready=[j for j in plan['jobs']if j['id']not in running and j['id']not in valid and j['id']not in failed
            and(j['sanity']or sanity_complete)]
        occupied={row['slot']for row in running.values()}
        for slot,cpus in enumerate(plan['CPUslots']):
            if slot in occupied or not ready or failed:continue
            job=ready.pop(0)
            for rid in (job['value_run'],job['density_run']):
                key=f'{rid}:{job["step"]}'
                if key not in audited:
                    _,_,checkpoint,sha=helper.load_parent(plan,rid,job['step'])
                    assert sha==plan['checkpoint_sha256'][key];audited[key]=dict(path=str(checkpoint),sha256=sha)
            logpath=output/(job['id']+'.log');log=logpath.open('a')
            command=['nice','-n','10','taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
                'utils.flow_iql_research.evaluate_factorized_policy_v3','--value-run',str(root/'runs'/job['value_run']),
                '--density-run',str(root/'runs'/job['density_run']),'--step',str(job['step']),
                '--samples','9','--output',job['output']]
            proc=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=proc,log=log,path=logpath,slot=slot,
                process_identity=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19])
        rows=[]
        for env in plan['environments']:
            jobs=[j for j in plan['jobs']if not j['sanity']and j['environment']==env];assert len(jobs)==3
            per_step={}
            for job in jobs:
                reference=Path(job['V2_N9_reference']);assert verifier.digest(reference)==plan['V2_N9_reference_sha256'][str(reference)]
                corrected=json.loads(reference.read_text());helper.check_episodes(corrected,job['step'])
                assert corrected['run']==job['density_run'] and corrected['selection_samples']==9 and corrected['selection_rule']=='max'
                assert corrected['checkpoint_sha256']==plan['checkpoint_sha256'][f"{job['density_run']}:{job['step']}"]
                matches=[j for j in native_plan['jobs']if j['environment']==env and j['step']==job['step']];assert len(matches)==1
                raw=native_result(native_plan,matches[0],verifier)['success_rate']if Path(matches[0]['output']).exists()else None
                common=valid.get(job['id']);v2=corrected['success_rate']
                per_step[str(job['step'])]=dict(native_own_BC=raw,native_Q_common_V2_BC=common,V2_Q_common_V2_BC=v2,
                    value_bundle_effect=None if common is None else v2-common,
                    BC_extraction_effect=None if raw is None or common is None else common-raw)
            complete=all(t['native_Q_common_V2_BC']is not None for t in per_step.values())
            common=statistics.mean(t['native_Q_common_V2_BC']for t in per_step.values())if complete else None
            corrected=statistics.mean(t['V2_Q_common_V2_BC']for t in per_step.values())
            rows.append(dict(environment=env,complete=complete,per_step=per_step,common_BC_native_Q_last3=common,
                common_BC_V2_Q_last3=corrected,value_bundle_last3_effect=None if common is None else corrected-common))
        complete=all(r['complete']for r in rows)
        native_complete=all(t['native_own_BC']is not None for r in rows for t in r['per_step'].values())
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='failure_requires_inspection'if failed else'complete'if complete else'sanity_running'if not sanity_complete else'common_BC_evaluations_running',
            valid_evaluations=len(valid),planned_evaluations=len(plan['jobs']),sanity_complete=sanity_complete,
            rows=rows,completed_tasks=sum(r['complete']for r in rows),
            native_own_BC_reference_complete=native_complete,
            common_BC_native_Q_nine_task_mean=statistics.mean(r['common_BC_native_Q_last3']for r in rows)if complete else None,
            common_BC_V2_Q_nine_task_mean=statistics.mean(r['common_BC_V2_Q_last3']for r in rows),
            common_BC_value_bundle_nine_task_effect=statistics.mean(r['value_bundle_last3_effect']for r in rows)if complete else None,
            failures=failed,running={k:dict(pid=v['process'].pid,slot=v['slot'],process_identity=v['process_identity'])for k,v in running.items()},
            actual_parent_pins=audited,registration_sha256=verifier.digest(args.registration),
            extra_GPU_jobs=0,goal_complete=False,
            limits=['Single training seed0 and one predefined commonBC source per task; this is a counterfactual value ablation, not an independent new method.',
                'The native and V2 own BC weights differ despite equal end RNG states; their causal difference is not diagnosed from the weights alone.',
                'Three same-model sanity evaluations must exactly reproduce existing N9 raw episode arrays before crossed evaluation proceeds.',
                'Value-bundle effects are conditional on the fixed V2 BC, not the separate effects of V-fit count/freshness/projection or a universal policy improvement.',
                'Only allnine tasks/alllateCPU50 checkpoints form macro means. No peak or environment-wise winner mixtures.']))
        if (complete and native_complete) or (failed and not running):return
        time.sleep(30)


if __name__=='__main__':main()
