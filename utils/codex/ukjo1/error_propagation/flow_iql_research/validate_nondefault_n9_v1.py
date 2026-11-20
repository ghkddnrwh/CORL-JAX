"""Check task IDs, frozen update equivalence, resume, and dataset provenance."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from utils.codex.ukjo1.error_propagation.flow_iql_research import nondefault_n9_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research import discount_0995_n9_core_v1 as old
from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import sha


def identical(a, b):
    x, y = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(x)==len(y) and all(np.array_equal(np.asarray(i), np.asarray(j)) for i,j in zip(x,y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'registration_v1.json').read_text())
    assert all(d.platform=='cpu' for d in jax.devices())
    for path, expected in plan['source_sha256'].items():
        assert sha(path)==expected, path
    assert len(core.TASKS)==len(set(core.ENVIRONMENTS))==40
    for task in core.TASKS:
        for recipe, gamma in core.DISCOUNTS.items():
            cfg = core.config_for(task['environment'], recipe)
            reference = core.base.config_for(task['family'], 'n9')
            reference['discount'] = gamma
            reference['onestep_specification'].update(name=recipe, discount=gamma)
            assert cfg==reference
            assert cfg['batch_size']==256 and cfg['backup_samples']==cfg['num_samples']==9
            assert cfg['onestep_specification']['objective']=='none'
            assert cfg['target_clipping'] and cfg['tau']==.005
    rng = np.random.RandomState(7364)
    batch = dict(observations=jnp.asarray(rng.randn(8,5),jnp.float32),
                 next_observations=jnp.asarray(rng.randn(8,5),jnp.float32),
                 actions=jnp.asarray(np.clip(rng.randn(8,2),-.9,.9),jnp.float32),
                 rewards=-jnp.ones(8), masks=jnp.asarray([0,1,1,0,1,1,1,1],jnp.float32))
    checks=[]
    for code in ('aml','puzzle4'):
        task=next(t for t in core.TASKS if t['code']==code)
        for recipe, gamma in core.DISCOUNTS.items():
            cfg=core.config_for(task['environment'],recipe,width=16)
            bounds=core.return_bounds(task['environment'],gamma,-1.,0.)
            state=core.create(0,batch['observations'][:1],batch['actions'][:1],cfg,recipe)
            update=jax.jit(core.build_update(recipe,bounds))
            reference=jax.jit(core.base.build_update('n9',bounds))
            actual=expected=state
            for _ in range(3):
                actual,_=update(actual,batch)
                expected,_=reference(expected,batch)
                assert identical(actual,expected)
            if gamma==.995:
                reference_cfg=old.config_for(task['family'],'g995_n9',width=16)
                assert cfg==reference_cfg
                old_state=old.create(0,batch['observations'][:1],batch['actions'][:1],reference_cfg,'g995_n9')
                old_update=jax.jit(old.build_update('g995_n9',bounds))
                for _ in range(3):old_state,_=old_update(old_state,batch)
                assert identical(actual,old_state)
            restored=flax.serialization.from_state_dict(state,flax.serialization.to_state_dict(actual))
            continued,_=update(actual,batch)
            resumed,_=update(restored,batch)
            assert identical(continued,resumed)
            probe=core.make_probe(recipe,bounds)(actual,batch)
            assert bool(probe['finite'])
            assert int(actual.base.q_updates)==int(actual.actor_updates)==3 and int(actual.base.v_updates)==0
            checks.append(dict(environment=task['environment'],discount=gamma,
                               original_N9_updates_bitwise_equal=True,resume_bitwise_equal=True,finite=True))
            print('NUMERICAL PASS',code,recipe,flush=True)
    tasks=[]
    for _,family,default in core.FAMILIES:
        env=ogbench.make_env_and_datasets(family,env_only=True)
        env.reset(seed=42)
        assert env.unwrapped._reward_task_id==default,(family,default,env.unwrapped._reward_task_id)
        env.close()
    for task in core.TASKS:
        env=ogbench.make_env_and_datasets(task['environment'],env_only=True)
        observation,_=env.reset(seed=42)
        assert env.unwrapped._reward_task_id==task['task_id']
        tasks.append(dict(**task,actual_task_id=env.unwrapped._reward_task_id,
                          observation_dim=int(observation.shape[-1]),action_dim=int(env.action_space.shape[-1])))
        env.close()
        print('TASK PASS',task['environment'],flush=True)
    datasets=[]
    directory=Path(plan['dataset_dir'])
    for _,family,_ in core.FAMILIES:
        name=core.dataset_name(family)
        ogbench.download_datasets([name],str(directory))
        for suffix in ('','-val'):
            path=directory/(name+suffix+'.npz')
            datasets.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    result=dict(status='PASS',time=time.time(),configurations_checked=80,real_update_checks=checks,
                tasks=tasks,datasets=datasets,packages=plan['packages'],
                scope='Task identity, original learning equations and resume checks; no performance claim.')
    (root/'validation/numerical_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',tasks=len(tasks),datasets=len(datasets),real_update_cases=len(checks))))


if __name__=='__main__':
    main()
