"""Matched N9 native IFQL reference on retained native32 models, no training.

This removes the proposal-budget confound from the V2-vs-native value ablation.
Reference outputs cannot automatically promote a new candidate or complete goal.
"""
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
from utils.flow_iql_research.audit_probability_value_pilots_v1 import equal


def validate_result(plan,job,verifier):
    result=json.loads(Path(job['output']).read_text());helper.check_episodes(result,job['step'])
    assert result['run']==job['run_id'] and result['training_seed']==0
    assert result['environment']==job['environment'] and result['selection_rule']=='max' and result['selection_samples']==9
    assert result['proposal_source']in('auto','behavior') and result['primary_evaluations_modified']is False
    assert result['config_sha256']==verifier.digest(Path(plan['root'])/'runs'/job['run_id']/'config.json')
    assert result['checkpoint_sha256']==plan['checkpoint_sha256'][f"{job['run_id']}:{job['step']}"]
    for field,path in [('policy_source_sha256','utils/flow_iql_research/selection_v2.py'),
        ('evaluation_helper_source_sha256','utils/flow_iql_research/train.py'),
        ('evaluator_source_sha256','utils/flow_iql_research/evaluate_selection_v2.py')]:
        assert result[field]==plan['source_sha256'][path]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path)
    args=parser.parse_args();plan=json.loads(args.registration.read_text());output=Path(plan['output'])
    output.mkdir(parents=True,exist_ok=False);lock=(output/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verifier=Verifier()
    root=Path(plan['root']);running={};valid={};failed={};paired_checks={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MUJOCO_GL='egl',PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    while True:
        for path,digest in plan['source_sha256'].items():assert verifier.digest(path)==digest,path
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is not None:
                row['log'].close();del running[key]
                if code:failed[key]=dict(exit_code=code,log=str(row['path']))
        ready=[]
        for job in plan['jobs']:
            key=job['id'];run=root/'runs'/job['run_id'];checkpoint=Path(job['checkpoint'])
            assert verifier.digest(checkpoint)==plan['checkpoint_sha256'][f"{job['run_id']}:{job['step']}"]
            if Path(job['output']).exists():
                result=validate_result(plan,job,verifier);valid[key]=result['success_rate']
                if key not in paired_checks:
                    _,native,_,_=helper.load_parent(plan,job['run_id'],job['step'])
                    _,corrected,_,_=helper.load_parent(plan,job['paired_V2_run'],job['step'])
                    nn=native['state']['agent']['network'];cn=corrected['state']['agent']['network']
                    paired_checks[key]=dict(BC_params_exact=equal(nn['params']['modules_actor_flow'],cn['params']['modules_actor_flow']),
                        BC_Adam_mu_exact=equal(nn['opt_state']['0']['mu']['modules_actor_flow'],cn['opt_state']['0']['mu']['modules_actor_flow']),
                        BC_Adam_nu_exact=equal(nn['opt_state']['0']['nu']['modules_actor_flow'],cn['opt_state']['0']['nu']['modules_actor_flow']),
                        agent_rng_exact=equal(native['state']['agent']['rng'],corrected['state']['agent']['rng']),
                        replay_rng_exact=equal(native['replay_rng'],corrected['replay_rng']))
            elif key not in running and key not in failed:ready.append(job)
        used={r['slot']for r in running.values()}
        for slot,cpus in enumerate(plan['CPUslots']):
            if slot in used or not ready:continue
            job=ready.pop(0);logpath=output/(job['id']+'.log');log=logpath.open('a')
            command=['nice','-n','10','taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
                'utils.flow_iql_research.evaluate_selection_v2',str(root/'runs'/job['run_id']),
                '--checkpoint',job['checkpoint'],'--output',job['output'],'--rule','max','--samples','9',
                '--proposal-source','behavior','--episodes','50']
            proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=proc,log=log,path=logpath,slot=slot,
                process_identity=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19])
        rows=[]
        for environment in plan['environments']:
            jobs=[j for j in plan['jobs']if j['environment']==environment];assert len(jobs)==3
            values={str(j['step']):valid.get(j['id'])for j in jobs};v2={}
            for job in jobs:
                path=Path(job['paired_V2_evaluation']);assert verifier.digest(path)==plan['paired_V2_evaluation_sha256'][str(path)]
                value=json.loads(path.read_text());helper.check_episodes(value,job['step'])
                assert value['run']==job['paired_V2_run'] and value['selection_samples']==9 and value['selection_rule']=='max'
                assert value['checkpoint_sha256']==plan['checkpoint_sha256'][f"{job['paired_V2_run']}:{job['step']}"]
                v2[str(job['step'])]=value['success_rate']
            complete=all(v is not None for v in values.values())
            first=statistics.mean(values.values())if complete else None;second=statistics.mean(v2.values())
            rows.append(dict(environment=environment,complete=complete,native_N9_per_step=values,V2_N9_per_step=v2,
                native_N9_last3=first,V2_N9_last3=second,V2_minus_native=None if first is None else second-first))
        complete=all(r['complete']for r in rows)
        native=statistics.mean(r['native_N9_last3']for r in rows)if complete else None
        corrected=statistics.mean(r['V2_N9_last3']for r in rows)
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),status='complete'if complete else'failures_require_inspection'if failed else'running',
            rows=rows,completed_tasks=sum(r['complete']for r in rows),valid_evaluations=len(valid),
            native_N9_nine_task_mean=native,V2_N9_nine_task_mean=corrected,
            V2_minus_native_nine_task_mean=None if native is None else corrected-native,
            same_N9_BC_and_RNG_checks=paired_checks,
            all_observed_BC_and_RNG_checks_exact=all(all(v.values())for v in paired_checks.values())if paired_checks else None,
            failures=failed,running={k:dict(pid=v['process'].pid,slot=v['slot'],process_identity=v['process_identity'])for k,v in running.items()},
            registration_sha256=verifier.digest(args.registration),reference_only=True,
            additional_GPU_jobs=0,goal_complete=False,
            limits=['One exploratory training seed0 reference, not independent confirmation or a newly qualified candidate.',
                'Allnine tasks and allthree lateCPU50results are required for a native macro; no partial/peak/zero filling.',
                'Nativeparentswere trained with inferenceconfigN32;IFQL training does not query that setting, and source/actualBC comparisons check rather than assume equivalence.',
                'Same proposalbudget9 removes the N32-vsN9 comparison confound; the corrected bundle still changes Vfits,bootstraporder,andtargetprojection together.',
                'Any nonexact BC/replay/Adam checks must be reviewed before attributing returns solely to value fitting.']))
        if complete:return
        time.sleep(30)


if __name__=='__main__':main()
