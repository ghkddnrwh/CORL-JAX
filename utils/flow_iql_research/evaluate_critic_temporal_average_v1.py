"""Deployment-only fixed 50:50 average of 800K/current Q scores (same Flow9)."""
import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

from utils.flow_iql_research import sample_budget_core_v1 as core
from utils.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic, read_source, sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def prepare(plan_path):
    parent = load_plan(plan_path)
    root = plan_path.parent
    path = root / 'temporal_average_registration_v1.json'
    assert not path.exists()
    plan = dict(protocol='fixed_800K_current_Q_score_average_v1', parent_registration_sha256=sha(plan_path),
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        cases=[c['id'] for c in parent['cases']], steps=[900000,1000000],
        current_Q_weight=.5, anchor_Q_weight=.5, anchor_step=800000,
        candidates=9, reset_seeds=list(range(42,92)), episodes=50, CPU_affinity_cores=4,
        score_rule='0.5*min_current_twin_Q + 0.5*min_800K_twin_Q; current Flow supplies the same nine candidates',
        actor='Unchanged original actor: not trained from the temporal-average teacher',
        optimizer_updates=0,
        limits='Deployment-only control; two Q inferences per action. No claim that training itself is stabilized.')
    atomic(path,plan)


def run(plan_path,case_id,step):
    plan=load_plan(plan_path)
    extra=json.loads((plan_path.parent/'temporal_average_registration_v1.json').read_text())
    assert extra['source_sha256'][str(Path(__file__).resolve())]==sha(__file__)
    case=next(c for c in plan['cases'] if c['id']==case_id)
    assert step in extra['steps'] and all(d.platform=='cpu' for d in jax.devices())
    parent=Path(case['parent_run'])
    sources=[dict(run=str(parent),checkpoint=str(parent/f'checkpoint_{s}.pkl'),step=s,
        config_sha256=case['config_sha256'],checkpoint_sha256=case['parent_checkpoints'][str(s)]) for s in (800000,step)]
    _,anchor=read_source(sources[0],False)
    _,current=read_source(sources[1])
    payload=dict(current=core.lean.policy_parameters(current.base),
                 anchor_Q=anchor['base']['agent']['network']['params']['modules_critic'])
    template=current.base.agent

    def policy(params,observation,key):
        agent=template.replace(network=template.network.replace(params=params['current']['base']))
        action_key,_=jax.random.split(key)
        actions=jax.random.normal(action_key,(9,agent.config['action_dim']))
        observations=jnp.repeat(jnp.expand_dims(observation,0),9,axis=0)
        for index in range(10):
            times=jnp.full((9,1),index/10)
            actions=actions+agent.network.select('actor_flow')(observations,actions,times,is_encoded=True)/10
        actions=jnp.clip(actions,-1,1)
        now=agent.network.select('critic')(observations,actions=actions).min(0)
        past_params=dict(params['current']['base'],modules_critic=params['anchor_Q'])
        past=agent.network.select('critic')(observations,actions=actions,params=past_params).min(0)
        return actions[jnp.argmax(.5*now+.5*past)]

    policy=jax.jit(policy,backend='cpu')
    env=ogbench.make_env_and_datasets(case['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    namespace=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    result=evaluate(namespace,env,policy,step,50)
    env.close()
    result.update(case=case_id,method='temporal_Q_average',environment=case['environment'],
        training_seed=case['seed'],sources=sources,source_sha256=extra['source_sha256'],
        registration_sha256=sha(plan_path.parent/'temporal_average_registration_v1.json'),
        optimizer_updates=0,reset_metadata=reset,scope=extra['limits'])
    out=plan_path.parent/'temporal_average_evaluations'/case_id/f'{step}_lean.json'
    assert not out.exists()
    atomic(out,result)
    print(json.dumps(dict(case=case_id,step=step,success=result['success_rate'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','evaluate'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case')
    p.add_argument('--step',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.plan.resolve())
    else:run(a.plan.resolve(),a.case,a.step)
