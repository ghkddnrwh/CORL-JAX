"""Verify EMA-only scope, exact update timing, legacy reuse and resume."""
import argparse
import json
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from utils.codex.ukjo1.error_propagation.flow_iql_research import default_ema_n9_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research import stability_study_core_v1 as historical
from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import sha


def identical(a,b):
    x,y=jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)
    return len(x)==len(y) and all(np.array_equal(np.asarray(i),np.asarray(j)) for i,j in zip(x,y))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    plan=json.loads((root/'registration_v1.json').read_text())
    assert all(d.platform=='cpu' for d in jax.devices())
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    for environment in core.ENVIRONMENTS:
        for recipe,rate in core.RATES.items():
            cfg=core.config_for(environment,recipe)
            expected=core.base.config_for(environment,'n9')
            expected['tau']=rate
            expected['onestep_specification'].update(name=recipe,target_EMA=rate)
            assert cfg==expected
            assert cfg['discount']==.999 and cfg['batch_size']==256
            assert cfg['backup_samples']==cfg['num_samples']==9
    rng=np.random.RandomState(761)
    batch={k:jnp.asarray(v,np.float32) for k,v in dict(
        observations=rng.normal(size=(8,5)),actions=rng.uniform(-.9,.9,(8,2)),
        next_observations=rng.normal(size=(8,5)),rewards=-rng.uniform(0,1,8),
        masks=np.array([1,1,1,1,1,1,0,0])).items()}
    cases=[]
    for environment in (core.ENVIRONMENTS[0],'puzzle-4x4-play-singletask-v0'):
        for recipe,rate in core.RATES.items():
            cfg=core.config_for(environment,recipe,16)
            bounds=core.return_bounds(environment,.999,-1.,0.)
            update=jax.jit(core.build_update(recipe,bounds))
            reference=jax.jit(core.base.build_update('n9',bounds))
            ema_formula=jax.jit(lambda online,target:jax.tree_util.tree_map(
                lambda p,t:rate*p+(1-rate)*t,online,target))
            for seed in (0,1):
                initial=core.create(seed,batch['observations'][:1],batch['actions'][:1],cfg,recipe)
                state=expected=initial
                for _ in range(3):
                    before=state
                    state,_=update(state,batch)
                    expected,_=reference(expected,batch)
                    assert identical(state,expected)
                    params=before.base.agent.network.params
                    target=ema_formula(params['modules_critic'],params['modules_target_critic'])
                    for a,b in zip(jax.tree_util.tree_leaves(target),jax.tree_util.tree_leaves(
                            state.base.agent.network.params['modules_target_critic'])):
                        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-7,atol=1e-7)
                restored=flax.serialization.from_state_dict(initial,flax.serialization.to_state_dict(state))
                continued,_=update(state,batch);resumed,_=update(restored,batch)
                assert identical(continued,resumed)
                legacy_equal=None
                if recipe=='ema0p001':
                    old_cfg=historical.config_for(environment,'slow_target',16)
                    legacy=historical.create(seed,batch['observations'][:1],batch['actions'][:1],old_cfg,'slow_target')
                    assert identical(initial,legacy)
                    old_update=jax.jit(historical.build_update('slow_target',bounds))
                    for _ in range(3):legacy,_=old_update(legacy,batch)
                    assert identical(state,legacy)
                    legacy_equal=True
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
                assert int(state.base.q_updates)==int(state.actor_updates)==3 and int(state.base.v_updates)==0
                cases.append(dict(environment=environment,recipe=recipe,seed=seed,rate=rate,
                    original_N9_update_exact=True,pre_update_online_Q_EMA_formula_checked=True,
                    resume_exact=True,legacy_slow_target_exact=legacy_equal,finite=True))
                print('PASS',environment,recipe,seed,flush=True)
    result=dict(status='PASS',time=time.time(),configurations_checked=20,real_update_cases=cases,
        mathematical_change='EMA coefficient only; old-online timing, N9 backup and actor loss preserved',
        reused_EMA001_equivalence=True)
    (root/'validation/numerical_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(cases))))


if __name__=='__main__':main()
