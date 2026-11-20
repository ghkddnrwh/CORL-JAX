"""Auxiliary 100K/300K curves; never used to choose ongoing interventions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from utils.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,read_source,sha


def prepare(path):
    plan=load_plan(path)
    jobs=[]
    for case in plan['cases']:
        parent=Path(case['parent_run'])
        for step in (100000,300000):
            checkpoint=parent/f'checkpoint_{step}.pkl'
            assert checkpoint.exists()
            for mode in ('lean','actor'):
                jobs.append(dict(id=f'{case["id"]}_{step}_{mode}',case=case['id'],mode=mode,
                    run=str(parent),checkpoint=str(checkpoint),step=step,
                    config_sha256=case['config_sha256'],checkpoint_sha256=sha(checkpoint)))
    extra=dict(protocol='auxiliary_early_learning_curve_CPU50_v1',created_unix=time.time(),
        parent_registration_sha256=sha(path),source_sha256=sha(__file__),jobs=jobs,
        reset_seeds=list(range(42,92)),episodes=50,CPU_cores_per_worker=4,optimizer_updates=0,
        limits='Exploratory diagnostic of when seed differences arose, registered after the first '
            '900K intervention results. No changes to ongoing methods, endpoints or primary metrics; '
            'no early checkpoint selection as a reported new method.')
    target=path.parent/'early_curve_registration_v1.json'
    assert not target.exists();atomic(target,extra)


def evaluate(path,job_id):
    from utils.flow_iql_research import sample_budget_core_v1 as core
    from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate as rollout
    from utils.propagation_research.seeded_reset import prepare_seeded_action_space
    import jax
    import ogbench
    plan=load_plan(path)
    extra=json.loads((path.parent/'early_curve_registration_v1.json').read_text())
    assert extra['source_sha256']==sha(__file__)
    job=next(j for j in extra['jobs'] if j['id']==job_id)
    case=next(c for c in plan['cases'] if c['id']==job['case'])
    conf,state=read_source(job)
    if job['mode']=='lean':
        policy=core.make_teacher_policy(state)
        payload=core.lean.policy_parameters(state.base)
    else:
        dataset=json.loads((Path(job['run'])/'dataset.json').read_text())
        metadata=core.actor_metadata(conf['identity']['config'],dataset['observation_dim'],dataset['action_dim'])
        policy=core.make_actor_policy(metadata)
        payload=state.actor.params
    assert all(d.platform=='cpu' for d in jax.devices())
    env=ogbench.make_env_and_datasets(case['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    result=rollout(namespace,env,policy,job['step'],50)
    env.close()
    result.update(protocol=extra['protocol'],case=case['id'],environment=case['environment'],
        training_seed=case['seed'],policy_mode=job['mode'],source=job,
        source_sha256=extra['source_sha256'],reset_metadata=reset,optimizer_updates=0,scope=extra['limits'])
    target=path.parent/'analysis/early_curve_evaluations'/f'{job_id}.json'
    assert not target.exists();atomic(target,result)
    print(json.dumps(dict(job=job_id,success=result['success_rate'])),flush=True)


def run(path):
    extra=json.loads((path.parent/'early_curve_registration_v1.json').read_text())
    assert extra['source_sha256']==sha(__file__)
    directory=path.parent/'analysis/early_curve_evaluations'
    directory.mkdir(exist_ok=True)
    queue=[j for j in extra['jobs'] if not (directory/f'{j["id"]}.json').exists()]
    active,done,failed={},[],[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',MUJOCO_GL='egl')
    while queue or active:
        for slot,item in list(active.items()):
            if item['process'].poll() is None:continue
            item['log'].close()
            (failed if item['process'].returncode else done).append(item['job'])
            del active[slot]
        for slot in range(8):
            if slot in active or not queue or failed:continue
            job=queue.pop(0)
            log=(directory/f'{job["id"]}.log').open('w')
            command=['taskset','-c',','.join(map(str,range(32+slot*4,36+slot*4))),sys.executable,'-u','-m',
                'utils.flow_iql_research.evaluate_critic_stabilization_early_v1','evaluate',
                '--plan',str(path),'--job',job['id']]
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            active[slot]=dict(process=process,log=log,job=job['id'])
        atomic(path.parent/'analysis/early_curve_status_v1.json',dict(
            completed=done,failures=failed,pending=len(queue),running={str(k):v['job'] for k,v in active.items()},
            timestamp_unix=time.time()))
        if failed and not active:raise RuntimeError(failed)
        time.sleep(3)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run','evaluate'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--job')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.plan.resolve())
    elif a.mode=='run':run(a.plan.resolve())
    else:evaluate(a.plan.resolve(),a.job)
