"""Current-Q-only rollout control for the temporal-average Flow implementation."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

from utils.flow_iql_research import evaluate_critic_temporal_average_v1 as average
from utils.flow_iql_research import sample_budget_core_v1 as core
from utils.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,read_source,sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def run(path,case_id):
    plan=load_plan(path)
    case=next(c for c in plan['cases'] if c['id']==case_id)
    parent=Path(case['parent_run'])
    source=dict(run=str(parent),checkpoint=str(parent/'checkpoint_1000000.pkl'),step=1000000,
        config_sha256=case['config_sha256'],checkpoint_sha256=case['parent_checkpoints']['1000000'])
    _,state=read_source(source)
    # Exercise the EXACT frozen averaging implementation's Flow code, replacing
    # only its final score combination. Never modify its source or training.
    source_text=inspect.getsource(average.run)
    start=source_text.index('    def policy(')
    end=source_text.index('    policy=jax.jit',start)
    policy_text=textwrap.dedent(source_text[start:end])
    assert policy_text.count('jnp.argmax(.5*now+.5*past)')==1
    policy_text=policy_text.replace('jnp.argmax(.5*now+.5*past)','jnp.argmax(now)')
    namespace=dict(template=state.base.agent,jax=jax,jnp=jnp)
    exec(compile(policy_text,'<frozen-average-current-only-control>','exec'),namespace)
    policy=jax.jit(namespace['policy'],backend='cpu')
    payload=dict(current=core.lean.policy_parameters(state.base),
                 anchor_Q=state.base.agent.network.params['modules_critic'])
    reference_policy=core.make_teacher_policy(state)
    params=payload['current']
    # Independent state/noise action probe as well as full reset-matched rollouts.
    obs=jax.random.normal(jax.random.PRNGKey(149),(64,state.base.agent.config['state_dim'])) if 'state_dim' in state.base.agent.config else None
    dataset=json.loads((parent/'dataset.json').read_text())
    obs=jax.random.normal(jax.random.PRNGKey(149),(64,dataset['observation_dim']))
    keys=jax.random.split(jax.random.PRNGKey(150),64)
    differences=[]
    for ob,key in zip(obs,keys):
        a=np.asarray(policy(payload,ob,key));b=np.asarray(reference_policy(params,ob,key))
        differences.append(float(np.max(np.abs(a-b))))
    env=ogbench.make_env_and_datasets(case['environment'],env_only=True)
    prepare_seeded_action_space(env,42)
    wrapped=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    result=evaluate(wrapped,env,policy,1000000,50)
    env.close()
    reference=json.loads((Path(case['baseline_evaluation_directory'])/'1000000_lean.json').read_text())
    exact={k:result[k]==reference[k] for k in ('episode_successes','episode_rewards','episode_lengths')}
    result.update(case=case_id,episode_arrays_exact=exact,action_probe_max_abs_difference=max(differences),
        average_source_sha256=sha(average.__file__),validation_source_sha256=sha(__file__),
        transformed_control_source_sha256=hashlib.sha256(policy_text.encode()).hexdigest(),
        reference_sha256=sha(Path(case['baseline_evaluation_directory'])/'1000000_lean.json'))
    output=path.parent/'analysis/temporal_average_controls'/f'{case_id}.json'
    atomic(output,result)
    assert all(exact.values()) and max(differences)==0., (exact,max(differences))
    print(json.dumps(dict(case=case_id,episode_arrays_exact=exact,action_probe_max_abs_difference=max(differences))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True)
    a=p.parse_args();run(a.plan.resolve(),a.case)
