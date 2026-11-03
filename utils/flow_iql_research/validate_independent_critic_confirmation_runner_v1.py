"""Exercise fresh three-member discovery-independent wiring on tiny CPU models.

This validation uses real Cube data, three fresh seeds, CPU50 evaluation and
the exact production readiness/episode/model auditor functions. It is outside
the benchmark run directory and is never a performance observation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job, summarize
from utils.flow_iql_research.audit_independent_critic_policy_v1 import audit_job
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier


def main():
    parser=argparse.ArgumentParser();parser.add_argument('production_plan',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    assert not(root/'validation.json').exists(),'Use a new validation output; inspect child handles after a timeout'
    production=json.loads(args.production_plan.read_text());envname='cube-single-play-singletask-v0'
    jobs=[copy.deepcopy(job)for job in production['training_jobs']if job['args']['env']==envname and job['args']['seed']in(3,4,5)]
    assert len(jobs)==3 and not any((root/'runs'/job['id']).exists()for job in jobs)
    expected={}
    for job in jobs:
        identity=copy.deepcopy(production['expected_identities'][job['id']])
        identity.update(training_device='cpu',chunk=2,probe_every=2,evaluation_steps=[2,800000,900000,1000000])
        identity['config'].update(actor_hidden_dims=[8]*4,value_hidden_dims=[8]*4)
        expected[job['id']]=identity
        job['args'].update({'train-device':'cpu','width':8,'chunk':2,'probe-every':2,'aux-eval-steps':'2','stop-at':2})
    tiny={**production,'training_jobs':jobs,'expected_identities':expected,'output_directory':str(root/'evaluations')}
    Path(tiny['output_directory']).mkdir()
    cpus=sorted(os.sched_getaffinity(0));affinity=cpus[-4:]
    prefix=['taskset','-c',','.join(map(str,affinity)),sys.executable,'-u','-m']
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    for job in jobs:
        command=prefix+[job['module'],'--output',str(root/'runs'/job['id'])]
        for key,value in job['args'].items():command+=['--'+key,str(value)]
        with(root/(job['id']+'.log')).open('w')as log:
            subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        config=json.loads((root/'runs'/job['id']/'config.json').read_text())
        assert config['identity']==expected[job['id']]
        assert config['source_sha256']==production['training_source_sha256']
    verifier=Verifier()
    try:ready_job(root,production,0,envname,2,'median_rank',verifier)
    except AssertionError:pass
    else:raise AssertionError('Tiny CPU identities accepted as production GPU identities')
    job=ready_job(root,tiny,0,envname,2,'median_rank',verifier)
    assert job is not None and job['id']=='group1_cube_2_median_rank'
    assert Path(job['actor_run']).name.startswith('ieconfirm_s3_')
    discovery=json.loads(Path(production['discovery_plan']).read_text())
    command=prefix+['utils.flow_iql_research.evaluate_independent_critic_policy_v1',
        '--actor-run',job['actor_run'],'--critic-runs',*job['critic_runs'],'--step','2',
        '--rule','median_rank','--output',job['output']]
    with(root/'ensemble.log').open('w')as log:
        subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    audited=audit_job(dict(source_sha256=discovery['source_sha256']),job,verifier)
    assert audited['learner_seeds']==[3,4,5]and not audited['parent_learners_complete']
    assert len(verifier.checkpoints)==3
    partial=summarize(production,'median_rank',{job['id']:job},{job['id']:audited},dict(selected_nine_task_mean=.5059259259259259))
    assert not partial['all_fresh_complete']and partial['fresh_two_group_nine_task_mean']is None
    assert all(group['completed_tasks']==0 and group['nine_task_mean']is None for group in partial['fresh_groups'])
    invalid={**tiny,'training_source_sha256':{}}
    try:ready_job(root,invalid,0,envname,2,'median_rank',verifier)
    except AssertionError:pass
    else:raise AssertionError('Wrong training sources accepted')
    result=dict(time=time.time(),actual_three_separate_CPU_learners=True,learner_seeds=[3,4,5],
        actual_member_updates=2,actual_Q_Adam_count=[2]*3,actual_V_Adam_count=[4]*3,
        real_dataset='cube-single-play-singletask-v0',production_training_sources_exact=True,
        expected_identity_factory_matches_after_explicit_tiny_overrides=True,
        production_GPU_identity_rejects_tiny_CPU_models=True,wrong_source_contract_rejected=True,
        actual_three_member_CPU50_evaluation_audited=True,all_actual_member_checkpoints_finite=True,
        first_BC_source_seed3=True,three_member_preparation_key_exact=True,
        incomplete_parents_and_nonprimary_step_do_not_produce_task_or_macro_means=True,
        production_queue_and_scores_unchanged=True,
        production_plan_sha256=hashlib.sha256(args.production_plan.read_bytes()).hexdigest(),
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='TinyCPUintegrationvalidationonly;no1MGPUconvergenceorpolicyperformanceclaim;excludedfromeverybenchmarkscore.')
    (root/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
