"""Real-gradient checks of baseline identity, intervention scope, and resume."""
import argparse
import json
from pathlib import Path
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from utils.flow_iql_research import stability_study_core_v1 as core
from utils.flow_iql_research import sample_budget_core_v1 as ref


def diff(a,b):
    aa=jax.tree_util.tree_leaves(a); bb=jax.tree_util.tree_leaves(b)
    assert len(aa)==len(bb)
    return max([0.]+[float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)))) for x,y in zip(aa,bb) if np.asarray(x).size])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); checks=[]
    for env in (core.ENVIRONMENTS[0],core.ENVIRONMENTS[-1]):
        rng=np.random.RandomState(761)
        batch={k:jnp.asarray(v,np.float32) for k,v in dict(
            observations=rng.normal(size=(8,5)),actions=rng.uniform(-.9,.9,(8,2)),
            next_observations=rng.normal(size=(8,5)),rewards=-rng.uniform(0,1,8),
            masks=np.array([1,1,1,1,1,1,0,0])).items()}
        bounds=(-1000.,0.)
        def initial(recipe):
            cfg=core.config_for(env,recipe,16)
            return core.create(2,jnp.zeros((1,5)),jnp.zeros((1,2)),cfg,recipe)
        control=initial('control')
        rcfg=ref.config_for(env,'n9',16)
        reference=ref.create(2,jnp.zeros((1,5)),jnp.zeros((1,2)),rcfg,'n9')
        control_fn=jax.jit(core.build_update('control',bounds))
        reference_fn=jax.jit(ref.build_update('n9',bounds))
        for _ in range(3):
            control,_=control_fn(control,batch); reference,_=reference_fn(reference,batch)
        exact=diff(control,reference); assert exact==0.,exact
        for recipe in core.RECIPES[1:]:
            state=initial(recipe)
            ordinary=initial('control')
            inputs=dict(batch)
            if recipe=='critic_batch1024':
                inputs['critic_extra']={k:jnp.stack([v,v*.99,v*1.01]) if k in ('observations','next_observations','actions') else jnp.stack([v,v,v]) for k,v in batch.items()}
                _,key=jax.random.split(state.base.agent.rng)
                large=core.target_inputs(state.base,inputs,recipe,key)
                small=core.target_inputs(ordinary.base,batch,'control',key)
                assert large[0]['observations'].shape[0]==32
                assert diff(large[2][:,:8],small[2])==0.
            fn=jax.jit(core.build_update(recipe,bounds))
            before=state
            state,logs=fn(state,inputs)
            ordinary,_=control_fn(ordinary,batch)
            assert diff(state.actor,ordinary.actor)==0., 'Old-state actor label changed at initialization'
            if recipe=='slow_target':
                expected=jax.tree_util.tree_map(lambda q,t:q*.001+t*.999,
                    before.base.agent.network.params['modules_critic'],before.base.agent.network.params['modules_target_critic'])
                assert diff(state.base.agent.network.params['modules_target_critic'],expected)<2e-7
            if recipe in core.DR3:
                assert np.isfinite(float(logs['dr3_loss'])) and abs(float(logs['dr3_loss']))>0
                terminal={**inputs,'masks':jnp.zeros(8)}
                _,zero=fn(before,terminal)
                assert float(zero['dr3_loss'])==0.
                # Gradients through BOTH online feature arguments are nonzero.
                net=before.base.agent.network
                _,phi=core.critic_features(net,batch['observations'],batch['actions'])
                assert phi.shape==(2,8,16)
            for _ in range(2):
                state,logs=fn(state,inputs); ordinary,_=control_fn(ordinary,batch)
            flow_difference=diff(state.base.agent.network.params['modules_actor_flow'],ordinary.base.agent.network.params['modules_actor_flow'])
            assert flow_difference==0.,flow_difference
            assert diff(state.base.agent.rng,ordinary.base.agent.rng)==0.
            assert diff(state.base.agent.network.params['modules_critic'],ordinary.base.agent.network.params['modules_critic'])>0.
            restored=flax.serialization.from_bytes(initial(recipe),flax.serialization.to_bytes(state))
            a,_=fn(state,inputs); b,_=fn(restored,inputs)
            assert diff(a,b)==0.
            candidates=core.fixed_candidates(state,batch['observations'])
            representation=core.representation_probe(state,batch,candidates)
            assert representation['scores'].shape==(9,8)
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(state))
            checks.append(dict(environment=env,recipe=recipe,reference_control_max_difference=exact,
                flow_max_difference=flow_difference,serialization_resume_exact=True,
                finite=True,critic_updates=int(state.base.q_updates),V_updates=int(state.base.v_updates)))
            print(json.dumps(checks[-1]),flush=True)
        # The frozen research environment uses JAX 0.3, before clear_caches.
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n')


if __name__=='__main__': main()
