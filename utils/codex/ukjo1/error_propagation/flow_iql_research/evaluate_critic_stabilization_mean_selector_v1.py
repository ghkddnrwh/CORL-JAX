"""Exploratory mean-online-head selector at 1M; no learner/actor changes."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

from utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,read_source,sha


def prepare(path):
    plan=load_plan(path)
    root=path.parent/'analysis/mean_selector_control_v1'
    root.mkdir(exist_ok=False)
    atomic(root/'registration.json',dict(created_unix=time.time(),source_sha256=sha(__file__),
        parent_registration_sha256=sha(path),cases=plan['cases'],step=1000000,episodes=50,
        reset_seeds=list(range(42,92)),CPU_cores=4,optimizer_updates=0,
        rule='Current original 1M Flow9, argmax of mean of the two online critic heads, replacing min.',
        actor='Not refitted; no actor performance claim for the mean selector.',
        provenance='Same frozen Flow code as the temporal-average evaluator; only the score reducer is changed. '
            'Its current-only min control reproduced all six originals exactly (300 episodes plus 384 action probes).',
        limits='Exploratory deployment-only follow-up registered after observing the LR intervention failures. '
            'Not a training-stability result or replacement for registered 800K/900K/1M primary metrics.'))


def evaluate(path,label):
    from utils.codex.ukjo1.error_propagation.flow_iql_research import evaluate_critic_temporal_average_v1 as frozen
    from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as core
    from utils.codex.ukjo1.error_propagation.flow_iql_research.train_mean_empirical_backup_v1 import evaluate as rollout
    from utils.codex.ukjo1.error_propagation.propagation_research.seeded_reset import prepare_seeded_action_space
    import hashlib
    import jax
    import jax.numpy as jnp
    import ogbench
    plan=load_plan(path)
    root=path.parent/'analysis/mean_selector_control_v1'
    registration=json.loads((root/'registration.json').read_text())
    assert registration['source_sha256']==sha(__file__)
    case=next(c for c in plan['cases'] if c['id']==label)
    parent=Path(case['parent_run'])
    source=dict(run=str(parent),checkpoint=str(parent/'checkpoint_1000000.pkl'),step=1000000,
        config_sha256=case['config_sha256'],checkpoint_sha256=case['parent_checkpoints']['1000000'])
    _,state=read_source(source)
    text=inspect.getsource(frozen.run)
    begin=text.index('    def policy(');end=text.index('    policy=jax.jit',begin)
    text=textwrap.dedent(text[begin:end])
    assert text.count('jnp.argmax(.5*now+.5*past)')==1
    old="now=agent.network.select('critic')(observations,actions=actions).min(0)"
    assert text.count(old)==1
    text=text.replace('jnp.argmax(.5*now+.5*past)','jnp.argmax(now)').replace(old,old.replace('.min(0)','.mean(0)'))
    namespace=dict(template=state.base.agent,jax=jax,jnp=jnp)
    exec(compile(text,'<frozen-Flow-current-mean-head-selector>','exec'),namespace)
    policy=jax.jit(namespace['policy'],backend='cpu')
    payload=dict(current=core.lean.policy_parameters(state.base),anchor_Q=state.base.agent.network.params['modules_critic'])
    env=ogbench.make_env_and_datasets(case['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    model=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    result=rollout(model,env,policy,1000000,50)
    env.close()
    result.update(protocol='exploratory_mean_online_head_selector_CPU50_v1',case=label,
        environment=case['environment'],training_seed=case['seed'],method='online_mean_selector',
        optimizer_updates=0,source=source,source_sha256=sha(__file__),reset_metadata=reset,
        policy_source_sha256=hashlib.sha256(text.encode()).hexdigest(),registration_sha256=sha(root/'registration.json'),
        scope=registration['limits'])
    atomic(root/f'{label}.json',result)
    print(json.dumps(dict(case=label,success=result['success_rate'])),flush=True)


def run(path):
    plan=load_plan(path)
    root=path.parent/'analysis/mean_selector_control_v1'
    slots=[list(range(32+4*i,36+4*i)) for i in range(4)]+[list(range(56,60)),list(range(60,64))]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',MUJOCO_GL='egl')
    active=[]
    for cores,case in zip(slots,plan['cases']):
        if (root/f'{case["id"]}.json').exists():continue
        log=(root/f'{case["id"]}.log').open('w')
        command=['taskset','-c',','.join(map(str,cores)),sys.executable,'-u','-m',
            'utils.codex.ukjo1.error_propagation.flow_iql_research.evaluate_critic_stabilization_mean_selector_v1','evaluate','--plan',str(path),'--case',case['id']]
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
        active.append((case['id'],process,log))
    completed,failed=[],[]
    while active:
        for label,process,log in list(active):
            if process.poll() is None:continue
            log.close();(failed if process.returncode else completed).append(label)
            active.remove((label,process,log))
        atomic(root/'status.json',dict(completed=completed,failures=failed,running=[x[0] for x in active],timestamp_unix=time.time()))
        time.sleep(3)
    assert not failed,failed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','evaluate'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--case')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.plan.resolve())
    elif a.mode=='run':run(a.plan.resolve())
    else:evaluate(a.plan.resolve(),a.case)
